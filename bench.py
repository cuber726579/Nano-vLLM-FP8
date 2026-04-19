import os
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")

import time
from random import randint, seed
from pathlib import Path
from nanovllm import LLM, SamplingParams # Total: 133966tok, Time: 33.10s, Throughput: 4047.47tok/s
# from vllm import LLM, SamplingParams # Total: 133966tok, Time: 33.25s, Throughput: 4029.13tok/s


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    model_id = "Qwen/Qwen3-0.6B-FP8"
    path = str(Path(hf_home) / "models" / model_id)
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
