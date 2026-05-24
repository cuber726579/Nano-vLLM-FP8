from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from random import randint, seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


DEFAULT_MODEL = ROOT / "Huggingface/models/Qwen/Qwen3-30B-A3B"

from nanovllm import SamplingParams


def resolve_model_path(model: str) -> str:
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path)

    for env_name in ("HF_HOME", "MODELSCOPE_CACHE"):
        cache = os.getenv(env_name)
        if cache is None:
            continue
        cache_path = Path(cache) / "models" / model
        if cache_path.is_dir():
            return str(cache_path)

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 MoE BF16 generation.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local Qwen3 MoE model path, or model id under HF_HOME/MODELSCOPE_CACHE.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size. Qwen3-30B-A3B supports 1, 2, or 4.")
    parser.add_argument("--num-seqs", type=int, default=8, help="Number of benchmark prompts.")
    parser.add_argument("--max-input-len", type=int, default=128, help="Maximum random prompt length.")
    parser.add_argument("--max-output-len", type=int, default=32, help="Maximum generated tokens per prompt.")
    parser.add_argument("--max-model-len", type=int, default=512, help="Runtime max model length.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512, help="Maximum scheduled tokens per step.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Maximum scheduled sequences.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup-tokens", type=int, default=1, help="Warmup decode tokens before timing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bar.")
    return parser.parse_args()


def make_inputs(args: argparse.Namespace) -> tuple[list[list[int]], list[SamplingParams]]:

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(1, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(1, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, sampling_params


def print_stats(
    model_path: str,
    args: argparse.Namespace,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    elapsed: float,
    engine_stats: dict,
) -> None:
    prompt_tokens = sum(len(prompt) for prompt in prompt_token_ids)
    requested_decode_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_requested_tokens = prompt_tokens + requested_decode_tokens

    print(f"Model: {model_path}")
    print(f"TP: {args.tp}")
    print(f"Prompts: {args.num_seqs}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Requested decode tokens: {requested_decode_tokens}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Requested-token throughput: {total_requested_tokens / elapsed:.2f} tok/s")

    if engine_stats:
        print(
            "Prefill: "
            f"{engine_stats['prefill_tokens']} tok, "
            f"{engine_stats['prefill_time']:.2f}s, "
            f"{engine_stats['prefill_throughput']:.2f} tok/s"
        )
        print(
            "Decode: "
            f"{engine_stats['decode_tokens']} tok, "
            f"{engine_stats['decode_time']:.2f}s, "
            f"{engine_stats['decode_throughput']:.2f} tok/s"
        )
        print(f"Engine total: {engine_stats['total_throughput']:.2f} tok/s")


def main() -> None:
    load_dotenv()
    args = parse_args()
    from nanovllm import LLM, SamplingParams

    seed(args.seed)
    model_path = resolve_model_path(args.model)
    max_num_seqs = args.max_num_seqs or args.num_seqs

    llm = LLM(
        model_path,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    prompt_token_ids, sampling_params = make_inputs(args)
    warmup_params = SamplingParams(max_tokens=args.warmup_tokens, ignore_eos=True)

    try:
        if args.warmup_tokens > 0:
            llm.generate([[0]], warmup_params, use_tqdm=False)

        start = time.perf_counter()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=not args.no_tqdm)
        elapsed = time.perf_counter() - start
        engine_stats = getattr(llm, "last_generate_stats", {})
    finally:
        llm.exit()

    print_stats(model_path, args, prompt_token_ids, sampling_params, elapsed, engine_stats)


if __name__ == "__main__":
    main()
