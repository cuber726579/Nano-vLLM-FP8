import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.config import resolve_runtime_config
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.quantization.base import QuantConfig


MODEL_ID = "Qwen/Qwen3-4B-AWQ"


def resolve_model_path() -> Path:
    cache = os.getenv("MODELSCOPE_CACHE")
    return Path(cache) / "models" / MODEL_ID


def init_single_rank_process_group():
    if dist.is_initialized():
        return None
    tmpdir = tempfile.TemporaryDirectory()
    init_file = Path(tmpdir.name) / "dist_init"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    return tmpdir


def destroy_single_rank_process_group(tmpdir) -> None:
    if tmpdir is None:
        return
    dist.destroy_process_group()
    tmpdir.cleanup()


def test_qwen3_awq_config(model_path: Path) -> tuple[AutoConfig, QuantConfig]:
    hf_config = resolve_runtime_config(AutoConfig.from_pretrained(model_path))
    quant_config = QuantConfig.from_hf_config(hf_config, quantization="awq")

    assert hf_config.model_type == "qwen3"
    assert quant_config is not None
    assert quant_config.quant_method == "awq"
    assert quant_config.bits == 4
    assert quant_config.group_size == 128
    assert quant_config.zero_point is True
    assert quant_config.version == "gemm"
    return hf_config, quant_config


def test_qwen3_awq_qkv_loader(
    model_path: Path,
    hf_config: AutoConfig,
    quant_config: QuantConfig,
) -> None:
    tmpdir = init_single_rank_process_group()
    try:
        attn = Qwen3Attention(
            hf_config.hidden_size,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            max_position=hf_config.max_position_embeddings,
            head_dim=hf_config.head_dim,
            rms_norm_eps=hf_config.rms_norm_eps,
            qkv_bias=hf_config.attention_bias,
            rope_theta=hf_config.rope_parameters.get("rope_theta", 1000000),
            rope_type=hf_config.rope_parameters.get("rope_type", "default"),
            quant_config=quant_config,
            prefix="model.layers.0.self_attn",
        )

        assert tuple(attn.qkv_proj.qweight.shape) == (hf_config.hidden_size, 768)
        assert tuple(attn.qkv_proj.qzeros.shape) == (20, 768)
        assert tuple(attn.qkv_proj.scales.shape) == (20, 6144)

        with safe_open(model_path / "model.safetensors", "pt", "cpu") as weights:
            for module_name, shard_id in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
                for suffix in ("qweight", "qzeros", "scales"):
                    param = getattr(attn.qkv_proj, suffix)
                    tensor = weights.get_tensor(f"model.layers.0.self_attn.{module_name}.{suffix}")
                    param.weight_loader(param, tensor, shard_id)

            loaded_qweight = weights.get_tensor("model.layers.0.self_attn.q_proj.qweight")
            loaded_kweight = weights.get_tensor("model.layers.0.self_attn.k_proj.qweight")
            loaded_vweight = weights.get_tensor("model.layers.0.self_attn.v_proj.qweight")

        assert torch.equal(attn.qkv_proj.qweight[:, :512], loaded_qweight)
        assert torch.equal(attn.qkv_proj.qweight[:, 512:640], loaded_kweight)
        assert torch.equal(attn.qkv_proj.qweight[:, 640:], loaded_vweight)
    finally:
        destroy_single_rank_process_group(tmpdir)


def test_qwen3_awq_generate(model_path: Path) -> None:
    assert torch.cuda.is_available(), "Qwen AWQ generation test requires CUDA."

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


def main() -> None:
    model_path = resolve_model_path()
    assert model_path.is_dir(), f"model path does not exist: {model_path}"

    hf_config, quant_config = test_qwen3_awq_config(model_path)
    test_qwen3_awq_qkv_loader(model_path, hf_config, quant_config)

    test_qwen3_awq_generate(model_path)
    print("Qwen AWQ tests passed")


if __name__ == "__main__":
    main()
