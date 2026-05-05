import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    load_dotenv()
    hf_home = os.getenv("MODELSCOPE_CACHE")
    assert hf_home is not None, "MODELSCOPE_CACHE must be set in .env"

    model_id = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    path = Path(hf_home) / "models" / model_id
    assert path.is_dir(), f"model path does not exist: {path}"

    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        str(path),
        max_model_len=512,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.5,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        min_p=0.0,
        max_tokens=32,
    )
    raw_prompts = [
        "用一句话介绍你自己",
        "列出 5 个常见的采样参数",
        "用一句话解释 top_p 的作用",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
