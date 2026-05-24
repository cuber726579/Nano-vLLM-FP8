import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP
from nanovllm.quantization.base import QuantConfig


class Qwen3MoeTopKRouter(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        routing_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(router_logits.dtype), selected_experts


class Qwen3MoeExperts(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        assert self.intermediate_size % self.tp_size == 0
        self.local_intermediate_size = self.intermediate_size // self.tp_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                2 * self.local_intermediate_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.local_intermediate_size,
                self.hidden_size,
            )
        )
        self.gate_up_proj.weight_loader = self.gate_up_weight_loader
        self.down_proj.weight_loader = self.down_weight_loader

    def gate_up_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, int],
    ) -> None:
        expert_idx, shard_id = loaded_shard_id
        shard_size = self.local_intermediate_size
        shard_start = self.tp_rank * shard_size
        target_start = shard_id * shard_size
        loaded_weight = loaded_weight.narrow(0, shard_start, shard_size)
        param.data[expert_idx, :, target_start:target_start + shard_size].copy_(loaded_weight.t())

    def down_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_idx: int,
    ) -> None:
        shard_size = self.local_intermediate_size
        shard_start = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(1, shard_start, shard_size)
        param.data[expert_idx].copy_(loaded_weight.t())

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.is_cuda and hasattr(torch, "_grouped_mm"):
            output = self._forward_grouped(hidden_states, selected_experts, routing_weights)
        else:
            output = self._forward_fallback(hidden_states, selected_experts, routing_weights)
        if self.tp_size > 1:
            dist.all_reduce(output)
        return output

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        top_k = selected_experts.shape[-1]
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts.index_select(0, order)
        token_indices = torch.arange(num_tokens, device=hidden_states.device, dtype=torch.long)
        token_indices = token_indices.repeat_interleave(top_k).index_select(0, order)
        sorted_weights = flat_weights.index_select(0, order)

        expert_inputs = hidden_states.index_select(0, token_indices)
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        expert_offsets = torch.cumsum(expert_counts, dim=0).to(torch.int32)

        gate_up = torch._grouped_mm(expert_inputs, self.gate_up_proj, expert_offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_hidden = F.silu(gate) * up
        expert_outputs = torch._grouped_mm(expert_hidden, self.down_proj, expert_offsets)
        expert_outputs = expert_outputs * sorted_weights[:, None]

        final_hidden_states = torch.zeros(
            num_tokens,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        final_hidden_states.index_add_(0, token_indices, expert_outputs.to(hidden_states.dtype))
        return final_hidden_states

    def _forward_fallback(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hits = torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False).flatten()
        for expert_idx in expert_hits.tolist():
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states.index_select(0, token_idx)
            gate_up = current_state @ self.gate_up_proj[expert_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = current_hidden_states @ self.down_proj[expert_idx]
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing_weights, selected_experts = self.gate(hidden_states)
        return self.experts(hidden_states, selected_experts, routing_weights)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_parameters").get("rope_theta", 1000000),
            rope_type=getattr(config, "rope_parameters").get("rope_type", "default"),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx, quant_config=quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    _expert_weight_re = re.compile(
        r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.weight$"
    )

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("Qwen3 MoE quantized checkpoints are not supported yet.")
        self.model = Qwen3MoeModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def resolve_weight_name(self, weight_name: str) -> tuple[str, tuple[int, int] | int] | None:
        match = self._expert_weight_re.match(weight_name)
        if match is None:
            return None
        expert_prefix, expert_idx, proj_name = match.groups()
        expert_idx = int(expert_idx)
        if proj_name == "down_proj":
            return f"{expert_prefix}.down_proj", expert_idx
        shard_id = 0 if proj_name == "gate_proj" else 1
        return f"{expert_prefix}.gate_up_proj", (expert_idx, shard_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
