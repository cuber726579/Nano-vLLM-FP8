import os
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("MODELSCOPE_CACHE")

from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    model_id = "RedHatAI/Qwen2-0.5B-Instruct-FP8"
    path = str(Path(hf_home) / "models" / model_id) # modelscope cache path
    # path = model_id # huggingface cache path
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        max_model_len=512,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.5,
        kv_cache_dtype="fp8"
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
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
