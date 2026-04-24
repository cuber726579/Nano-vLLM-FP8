import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.quantization.base import LinearMethod
from nanovllm.utils.context import get_context


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output.mul_(torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps))
        output.mul_(1.0 + self.weight.float())
        return output.to(x.dtype)


class Qwen3_5RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float()
        output.mul_(torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps))
        output.mul_(self.weight.float())
        output.mul_(F.silu(gate.float()))
        return output.to(hidden_states.dtype)


class Qwen3_5Attention(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        q_proj_size = self.total_num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            q_proj_size,
            bias=config.attention_bias,
            linear_method=linear_method,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            linear_method=linear_method,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            linear_method=linear_method,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rope_parameters = config.rope_parameters
        rotary_dim = int(self.head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10000),
            type=rope_parameters.get("rope_type", "default"),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        gate = None
        if self.attn_output_gate:
            q = q.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = torch.chunk(q, 2, dim=-1)
            gate = gate.reshape(-1, self.num_heads * self.head_dim)
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = self.q_norm(q)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v).flatten(1, -1)
        if gate is not None:
            o.mul_(torch.sigmoid(gate))
        return self.o_proj(o)


class Qwen3_5MLP(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            linear_method=linear_method,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            linear_method=linear_method,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            linear_method=linear_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_v_heads = config.linear_num_value_heads
        self.total_num_k_heads = config.linear_num_key_heads
        assert self.total_num_v_heads % tp_size == 0
        assert self.total_num_k_heads % tp_size == 0
        self.num_v_heads = self.total_num_v_heads // tp_size
        self.num_k_heads = self.total_num_k_heads // tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size,
            config.linear_key_head_dim * config.linear_num_key_heads * 2
            + config.linear_value_head_dim * config.linear_num_value_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            config.linear_value_head_dim * config.linear_num_value_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            config.linear_num_value_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            config.linear_num_value_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            config.linear_value_head_dim * config.linear_num_value_heads,
            self.hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.conv_states: dict[int, torch.Tensor] = {}
        self.recurrent_states: dict[int, torch.Tensor] = {}

    def clear_sequence_states(self, seq_ids: list[int] | None = None) -> None:
        if seq_ids is None:
            self.conv_states.clear()
            self.recurrent_states.clear()
            return
        for seq_id in seq_ids:
            self.conv_states.pop(seq_id, None)
            self.recurrent_states.pop(seq_id, None)

    def _apply_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = mixed_qkv.shape[-1]
        if conv_state is None:
            conv_out = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
            new_conv_state = mixed_qkv[..., -self.conv_kernel_size :]
            if new_conv_state.shape[-1] < self.conv_kernel_size:
                new_conv_state = F.pad(new_conv_state, (self.conv_kernel_size - new_conv_state.shape[-1], 0))
            return conv_out, new_conv_state

        conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
        conv_out = F.conv1d(
            conv_input,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=0,
            groups=self.conv_dim,
        )
        conv_out = F.silu(conv_out[:, :, -seq_len:]).to(mixed_qkv.dtype)
        return conv_out, conv_input[..., -self.conv_kernel_size :]

    def _forward_one(
        self,
        seq_id: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        conv_state = self.conv_states.get(seq_id)
        recurrent_state = self.recurrent_states.get(seq_id)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv, new_conv_state = self._apply_conv(mixed_qkv, conv_state)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if recurrent_state is not None and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        self.conv_states[seq_id] = new_conv_state.detach()
        if last_recurrent_state is not None:
            self.recurrent_states[seq_id] = last_recurrent_state.detach()

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        seq_ids = context.seq_ids or []
        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            outputs = []
            for i, seq_id in enumerate(seq_ids):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                outputs.append(self._forward_one(seq_id, hidden_states[start:end].unsqueeze(0)).squeeze(0))
            return torch.cat(outputs, dim=0)

        outputs = []
        for i, seq_id in enumerate(seq_ids):
            outputs.append(self._forward_one(seq_id, hidden_states[i : i + 1].unsqueeze(0)).squeeze(0))
        return torch.cat(outputs, dim=0)


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, linear_method=linear_method)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, linear_method=linear_method)
        else:
            raise NotImplementedError(f"Unsupported Qwen3.5 layer_type: {self.layer_type!r}")
        self.mlp = Qwen3_5MLP(config, linear_method=linear_method)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5TextModel(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx, linear_method=linear_method) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5Model(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        self.language_model = Qwen3_5TextModel(config, linear_method=linear_method)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions)


class Qwen3_5ForCausalLM(nn.Module):
    ignored_weight_prefixes = ("model.visual.", "mtp.")

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        linear_method: LinearMethod | None = None,
    ) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config, linear_method=linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
