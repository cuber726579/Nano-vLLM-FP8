import json
import os
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope


@dataclass(slots=True)
class Eagle3LayerConfig:
    hidden_size: int
    intermediate_size: int
    hidden_act: str
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    attention_bias: bool
    head_dim: int | None
    rope_theta: float
    rope_type: str


@dataclass(slots=True)
class Eagle3Config:
    model: str
    draft_vocab_size: int
    target_vocab_size: int
    target_hidden_size: int
    norm_before_residual: bool
    layer_config: Eagle3LayerConfig


def load_eagle3_config(path: str) -> Eagle3Config:
    config_path = os.path.join(path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        raw_config = json.load(f)
    if raw_config.get("speculators_model_type") not in (None, "eagle3"):
        raise ValueError(f"Unsupported speculator config type: {raw_config.get('speculators_model_type')!r}")

    layer_raw = dict(raw_config["transformer_layer_config"])
    rope_scaling = layer_raw.get("rope_scaling") or {}
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    layer_config = Eagle3LayerConfig(
        hidden_size=int(layer_raw["hidden_size"]),
        intermediate_size=int(layer_raw["intermediate_size"]),
        hidden_act=layer_raw["hidden_act"],
        num_attention_heads=int(layer_raw["num_attention_heads"]),
        num_key_value_heads=int(layer_raw["num_key_value_heads"]),
        vocab_size=int(layer_raw["vocab_size"]),
        max_position_embeddings=int(layer_raw["max_position_embeddings"]),
        rms_norm_eps=float(layer_raw.get("rms_norm_eps", 1e-6)),
        attention_bias=bool(layer_raw.get("attention_bias", False)),
        head_dim=int(layer_raw["head_dim"]) if layer_raw.get("head_dim") is not None else None,
        rope_theta=float(layer_raw.get("rope_theta", 10000.0)),
        rope_type=rope_type,
    )
    target_hidden_size = raw_config.get("target_hidden_size") or layer_config.hidden_size
    return Eagle3Config(
        model=path,
        draft_vocab_size=int(raw_config["draft_vocab_size"]),
        target_vocab_size=layer_config.vocab_size,
        target_hidden_size=int(target_hidden_size),
        norm_before_residual=bool(raw_config.get("norm_before_residual", False)),
        layer_config=layer_config,
    )


class Eagle3Attention(nn.Module):

    def __init__(self, config: Eagle3LayerConfig) -> None:
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.total_num_heads * self.head_dim
        self.kv_size = self.total_num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_groups = self.total_num_heads // self.total_num_kv_heads
        self.qkv_proj = QKVParallelLinear(
            2 * config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            type=config.rope_type,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states.reshape(batch_size * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(batch_size * seq_len, self.total_num_heads, self.head_dim)
        k = k.view(batch_size * seq_len, self.total_num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.total_num_kv_heads, self.head_dim)

        positions = positions.reshape(-1)
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(batch_size, seq_len, self.total_num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.total_num_kv_heads, self.head_dim)
        if self.num_key_value_groups != 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).triu_(1)
        attn_weights = attn_weights.masked_fill(causal_mask, -torch.inf)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output.view(batch_size, seq_len, -1)


class Eagle3MLP(nn.Module):

    def __init__(self, config: Eagle3LayerConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)


class Eagle3DecoderLayer(nn.Module):

    def __init__(self, config: Eagle3LayerConfig, norm_before_residual: bool) -> None:
        super().__init__()
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config)
        self.mlp = Eagle3MLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.input_layernorm(embeds)
        if residual is None:
            if self.norm_before_residual:
                hidden_states = self.hidden_norm(hidden_states)
                residual = hidden_states
            else:
                residual = hidden_states
                hidden_states = self.hidden_norm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Eagle3Speculator(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    ignored_weight_prefixes = ("verifier",)

    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.config = config
        layer_config = config.layer_config
        self.hidden_size = layer_config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.target_vocab_size, layer_config.hidden_size)
        self.fc = ReplicatedLinear(3 * config.target_hidden_size, layer_config.hidden_size, bias=False)
        self.layers = nn.ModuleList([Eagle3DecoderLayer(layer_config, config.norm_before_residual)])
        self.norm = RMSNorm(layer_config.hidden_size, eps=layer_config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.draft_vocab_size, layer_config.hidden_size)
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.target_vocab_size, dtype=torch.bool))

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
            positions = positions.unsqueeze(1)
            hidden_states = hidden_states.unsqueeze(1)
        embeds = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, embeds, hidden_states, residual)
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states, all_tokens=True)

    def compute_logits(self, hidden_states: torch.Tensor, map_to_target_vocab: bool = True) -> torch.Tensor:
        logits = self.compute_draft_logits(hidden_states)
        if not map_to_target_vocab:
            return logits
        target_indices = torch.arange(self.draft_vocab_size, device=logits.device) + self.d2t
        mapped_logits = logits.new_full((*logits.shape[:-1], self.target_vocab_size), -torch.inf)
        mapped_logits[..., target_indices] = logits
        return mapped_logits

    def greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        draft_tokens = self.compute_draft_logits(hidden_states).argmax(dim=-1)
        return self.map_draft_to_target_tokens(draft_tokens)

    def map_draft_to_target_tokens(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        return draft_tokens + self.d2t[draft_tokens]

    def check_target_token_availability(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.t2d[target_tokens]

    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        num_speculative_tokens: int,
    ) -> torch.Tensor:
        token_prefix = input_ids.unsqueeze(1)
        position_prefix = positions.unsqueeze(1)
        hidden_prefix = hidden_states.unsqueeze(1)
        draft_tokens = []
        for _ in range(num_speculative_tokens):
            sample_hidden_states, next_hidden_states = self(token_prefix, position_prefix, hidden_prefix)
            next_token = self.greedy_sample(sample_hidden_states[:, -1])
            draft_tokens.append(next_token)
            token_prefix = torch.cat([token_prefix, next_token.unsqueeze(1)], dim=1)
            position_prefix = torch.cat([position_prefix, position_prefix[:, -1:] + 1], dim=1)
            hidden_prefix = torch.cat([hidden_prefix, next_hidden_states[:, -1:].contiguous()], dim=1)
        return torch.stack(draft_tokens, dim=1)
