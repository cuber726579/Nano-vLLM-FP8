import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    load_dotenv()
    cache = os.getenv("MODELSCOPE_CACHE")
    model_id = "Qwen/Qwen3.5-0.8B"
    path = Path(cache) / "models" / model_id

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(
        str(path),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=64,
    )

    raw_prompts = [
        "用一句话介绍你自己",
        "列出 5 个常见的采样参数",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert 0 < len(output["token_ids"]) <= sampling_params.max_tokens
        assert output["text"].strip()

    for prompt, output in zip(raw_prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
