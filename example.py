import os
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")

from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model_id = "Qwen/Qwen3.5-9B"
    path = Path(hf_home) / "models" / model_id
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        max_num_seqs=1,
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
