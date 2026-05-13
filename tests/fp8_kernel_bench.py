import argparse
import time

import torch
from nanovllm.quantization.fp8 import w8a8_block_fp8_matmul_triton

"""
VLLM_LOGGING_LEVEL=ERROR python -m tests.fp8_kernel_bench --m 1 8 16 32 64 128 256 512 --iters 200 --warmup 50 --check
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Nano-vLLM and vLLM W8A8 block FP8 Triton kernels.")
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--check", action="store_true", help="Compare outputs for the first M.")
    return parser.parse_args()


def output_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def make_inputs(
    m: int,
    n: int,
    k: int,
    block_n: int,
    block_k: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert n % block_n == 0
    assert k % block_k == 0
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(fp8_dtype)
    a_scale = torch.rand(m, k // block_k, device="cuda", dtype=torch.float32) * 0.02 + 0.01
    b_scale = torch.rand(n // block_n, k // block_k, device="cuda", dtype=torch.float32) * 0.02 + 0.01
    return a, b, a_scale, b_scale


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm
    except ImportError as exc:
        raise RuntimeError("vLLM is required to run the comparison.") from exc

    fp8_dtype = torch.float8_e4m3fn
    out_dtype = output_dtype(args.dtype)
    block_size = (args.block_n, args.block_k)

    print(
        f"shape: N={args.n}, K={args.k}, block={block_size}, output_dtype={out_dtype}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )
    print(f"{'M':>6} {'nano ms':>12} {'vllm ms':>12} {'speedup(v/n)':>14}")

    for i, m in enumerate(args.m):
        a, b, a_scale, b_scale = make_inputs(m, args.n, args.k, args.block_n, args.block_k, fp8_dtype)

        def nano_fn():
            return w8a8_block_fp8_matmul_triton(a, b, a_scale, b_scale, block_size, output_dtype=out_dtype)

        def vllm_fn():
            return w8a8_triton_block_scaled_mm(
                a,
                b,
                a_scale,
                b_scale,
                list(block_size),
                output_dtype=out_dtype,
            )

        nano_ms = bench(nano_fn, args.warmup, args.iters)
        vllm_ms = bench(vllm_fn, args.warmup, args.iters)
        print(f"{m:6d} {nano_ms:12.4f} {vllm_ms:12.4f} {nano_ms / vllm_ms:14.3f}")

        if args.check and i == 0:
            nano_out = nano_fn()
            vllm_out = vllm_fn()
            torch.cuda.synchronize()
            diff = (nano_out.float() - vllm_out.float()).abs()
            print(f"check M={m}: max_abs={diff.max().item():.6f}, mean_abs={diff.mean().item():.6f}")


if __name__ == "__main__":
    main()
