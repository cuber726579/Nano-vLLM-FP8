import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model_id = "Qwen/Qwen3-0.6B"
    path = Path(hf_home) / "models" / model_id
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(
        path,
        tensor_parallel_size=1, gpu_memory_utilization=0.9,
        kvcache_block_size=256
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=32)

    shared_prefix = "\n".join(
        f"Prefix-cache debug fact {i:03d}: The shared context stays exactly the same."
        for i in range(80)
    )
    prompts = [
        shared_prefix + "\n\nQuestion A: Summarize the shared facts in one sentence.",
        shared_prefix + "\n\nQuestion B: Count how many shared fact lines are provided.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    common_prefix_len = next(
        (i for i, (a, b) in enumerate(zip(*token_ids)) if a != b),
        min(len(token_ids[0]), len(token_ids[1])),
    )
    print(f"Common token prefix length: {common_prefix_len}")
    print(f"Full prompt token lengths: {[len(ids) for ids in token_ids]}")
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
