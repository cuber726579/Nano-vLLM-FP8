import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FP8_DTYPES


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def gather_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: list[int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = k_cache.shape[1]
    slot_indices = []
    for seq_idx, seq_len in enumerate(seq_lens):
        if seq_len == 0:
            continue
        positions = torch.arange(seq_len, device=block_tables.device, dtype=torch.long)
        block_ids = block_tables[seq_idx, positions // block_size].to(torch.long)
        slot_indices.append(block_ids * block_size + positions % block_size)
    if not slot_indices:
        shape = (0, k_cache.shape[-2], k_cache.shape[-1])
        return torch.empty(shape, device=k_cache.device, dtype=dtype), torch.empty(shape, device=v_cache.device, dtype=dtype)

    slot_indices = torch.cat(slot_indices)
    k = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1]).index_select(0, slot_indices).to(dtype=dtype)
    v = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1]).index_select(0, slot_indices).to(dtype=dtype)
    return k, v


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                if is_fp8_dtype(k_cache.dtype):
                    cu_seqlens_k = context.cu_seqlens_k
                    seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu().tolist()
                    k, v = gather_kvcache(k_cache, v_cache, context.block_tables, seq_lens, q.dtype)
                    o = flash_attn_varlen_func(q, k, v,
                                               max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                               max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
                                               softmax_scale=self.scale, causal=True)
                    return o
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if is_fp8_dtype(k_cache.dtype):
                seq_lens = context.context_lens.cpu().tolist()
                k, v = gather_kvcache(k_cache, v_cache, context.block_tables, seq_lens, q.dtype)
                batch_size = q.shape[0]
                cu_seqlens_q = torch.arange(batch_size + 1, device=q.device, dtype=torch.int32)
                cu_seqlens_k = torch.cat([
                    torch.zeros(1, device=q.device, dtype=torch.int32),
                    torch.cumsum(context.context_lens, dim=0, dtype=torch.int32),
                ])
                max_seqlen_k = max(seq_lens) if seq_lens else 0
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q,
                                           max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o
