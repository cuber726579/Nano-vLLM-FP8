import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams


MODEL_ID = "Qwen/Qwen3-4B-AWQ"

def resolve_model_path() -> Path:
    cache = os.getenv("MODELSCOPE_CACHE")
    return Path(cache) / "models" / MODEL_ID

def main() -> None:
    model_path = resolve_model_path()
    assert model_path.is_dir(), f"model path does not exist: {model_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        str(model_path),
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        quantization="awq",
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=8,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍你自己"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        outputs = llm.generate([prompt], sampling_params)
    finally:
        llm.exit()

    assert len(outputs) == 1
    assert 0 < len(outputs[0]["token_ids"]) <= sampling_params.max_tokens
    assert outputs[0]["text"].strip()
    print(f"Completion: {outputs[0]['text']!r}")


if __name__ == "__main__":
    main()
