from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from nanovllm.quantization.base import LinearMethod, QuantConfig, ceil_div


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, FP8_MAX: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)) / FP8_MAX, 1e-12)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def get_fp8_format_traits(fmt: str) -> tuple[torch.dtype, float]:
    fmt = fmt.lower()
    fmt_to_dtype_name = {
        "e4m3": "float8_e4m3fn",
        "e4m3fn": "float8_e4m3fn",
        "e5m2": "float8_e5m2",
    }
    dtype_name = fmt_to_dtype_name.get(fmt)
    if dtype_name is None:
        raise NotImplementedError(f"Unsupported FP8 format: {fmt!r}")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise NotImplementedError(f"Current PyTorch build does not support torch.{dtype_name}.")
    return dtype, float(torch.finfo(dtype).max)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
    fp8_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    if fp8_max is None:
        fp8_max = float(torch.finfo(dtype).max)
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, FP8_MAX=fp8_max)
    return y, s


@triton.jit
def _w8a8_block_fp8_matmul(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    block_n, block_k = block_size
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    block_size_m = min(max(triton.next_power_of_2(M), 16), 128)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=8,
    )
    return C


def dequantize_block_fp8(
    qweight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    block_n, block_k = block_size
    scale = weight_scale_inv.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
    scale = scale[: qweight.size(0), : qweight.size(1)]
    return (qweight.float() * scale).to(output_dtype)


def reference_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None,
    block_size: tuple[int, int],
) -> torch.Tensor:
    dequant_weight = dequantize_block_fp8(weight, weight_scale_inv, block_size, x.dtype)
    return F.linear(x, dequant_weight, bias)


class Fp8LinearMethod(LinearMethod):

    def __init__(self, quant_config: QuantConfig):
        if quant_config.activation_scheme != "dynamic":
            raise NotImplementedError(
                f"Unsupported FP8 activation scheme: {quant_config.activation_scheme!r}"
            )
        if quant_config.weight_block_size is None:
            raise ValueError("FP8 checkpoints require weight_block_size.")

        self.quant_config = quant_config
        self.weight_dtype, self.fp8_max = get_fp8_format_traits(quant_config.fmt)
        self.weight_block_size = quant_config.weight_block_size

    def scaled_output_size(self, size: int) -> int:
        return ceil_div(size, self.weight_block_size[0])

    def scaled_input_size(self, size: int) -> int:
        return ceil_div(size, self.weight_block_size[1])

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        block_n, block_k = self.weight_block_size
        if input_size % block_k != 0 or output_size % block_n != 0:
            raise ValueError(
                "FP8 block quantization requires dimensions divisible by the block size, "
                f"but got output={output_size}, input={input_size}, block_size={self.weight_block_size}."
            )

        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=self.weight_dtype),
            requires_grad=False,
        )
        layer.weight.weight_loader = layer.weight_loader

        layer.weight_scale_inv = nn.Parameter(
            torch.empty(
                self.scaled_output_size(output_size),
                self.scaled_input_size(input_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.weight_scale_inv.weight_loader = layer.weight_scale_loader

        if bias:
            layer.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
            layer.bias.weight_loader = layer.bias_loader
        else:
            layer.register_parameter("bias", None)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block_k = self.weight_block_size[1]
        x = x.contiguous()
        if bias is None:
            bias = layer.bias
        if not x.is_cuda or x.shape[-1] % block_k != 0:
            return reference_fp8_linear(
                x,
                layer.weight,
                layer.weight_scale_inv,
                bias,
                self.weight_block_size,
            )

        qinput, input_scale = act_quant(x, block_k, dtype=self.weight_dtype, fp8_max=self.fp8_max)
        output = w8a8_block_fp8_matmul_triton(
            qinput,
            layer.weight,
            input_scale,
            layer.weight_scale_inv,
            self.weight_block_size,
            output_dtype=x.dtype,
        )
        if bias is not None:
            output = output + bias
        return output
