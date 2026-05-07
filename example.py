import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    model_id = "Qwen/Qwen3-0.6B-FP8"
    cache = os.getenv("MODELSCOPE_CACHE")
    path = str(Path(cache) / "models" / model_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, kv_cache_dtype="fp8")

    sampling_params = SamplingParams(
        temperature=0.6, max_tokens=256,
        top_k=30, top_p=0.9, min_p=0.05
    )

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
