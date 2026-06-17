import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.spec_decode import Eagle3Config, load_eagle3_config
from nanovllm.quantization import QuantConfig

KV_CACHE_DTYPE_ALIASES = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}


def validate_qwen3_moe_tensor_parallel(hf_config: AutoConfig, tensor_parallel_size: int) -> None:
    checks = {
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "moe_intermediate_size": hf_config.moe_intermediate_size,
        "vocab_size": hf_config.vocab_size,
    }
    invalid = [name for name, value in checks.items() if value % tensor_parallel_size != 0]
    if invalid:
        details = ", ".join(f"{name}={checks[name]}" for name in invalid)
        raise ValueError(
            "qwen3_moe tensor_parallel_size must divide attention heads, KV heads, "
            "MoE intermediate size, and vocab size. "
            f"Got tensor_parallel_size={tensor_parallel_size}, invalid: {details}."
        )


def normalize_rope_config(hf_config: AutoConfig) -> dict:
    # transformers >= v5.0.0 : rope_parameters
    if hasattr(hf_config, "rope_parameters") and hf_config.rope_parameters is not None:
        return hf_config.rope_parameters

    # transformers < v5.0.0 : rope_theta and rope_scaling
    rope_parameters = {}
    rope_theta = getattr(hf_config, "rope_theta", None) # Old Attribute
    if rope_theta is not None:
        rope_parameters["rope_theta"] = rope_theta

    rope_scaling = getattr(hf_config, "rope_scaling", None) # Old Attribute
    rope_parameters.setdefault("rope_type", "default")
    if rope_scaling is not None:
        rope_parameters.update(dict(rope_scaling))
        # Attribute Name Change in New Version (rope_scaling.type -> rope_parameters.rope_type)
        rope_parameters["rope_type"] = rope_parameters.pop("type", "default")

    return rope_parameters


def resolve_runtime_config(hf_config: AutoConfig) -> AutoConfig:
    if hf_config.model_type == "qwen3_5":
        text_config = hf_config.text_config
        text_config.rope_parameters = normalize_rope_config(text_config)
        return text_config

    hf_config.rope_parameters = normalize_rope_config(hf_config)
    return hf_config


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    quantization: str | None = None
    dist_init_method: str | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str | None = None
    enable_prefix_cache: bool = True
    quant_config: QuantConfig | None = None
    speculative_model: str | None = None
    speculative_method: str | None = None
    num_speculative_tokens: int = 3
    eagle3_target_layer_ids: tuple[int, int, int] | list[int] | None = None
    eagle3_config: Eagle3Config | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = resolve_runtime_config(AutoConfig.from_pretrained(self.model))
        self.quant_config = QuantConfig.from_hf_config(self.hf_config, self.quantization)
        if self.quant_config is not None:
            self.quantization = self.quant_config.quant_method
        if self.hf_config.model_type == "qwen3_moe":
            validate_qwen3_moe_tensor_parallel(self.hf_config, self.tensor_parallel_size)
            if self.quant_config is not None:
                raise NotImplementedError("qwen3_moe quantized checkpoints are not supported yet.")
            self.enforce_eager = True
        if self.hf_config.model_type == "qwen3_5_text":
            self.enable_prefix_cache = False
            self.enforce_eager = True
        if self.kv_cache_dtype is not None:
            if self.kv_cache_dtype not in KV_CACHE_DTYPE_ALIASES:
                choices = ", ".join(KV_CACHE_DTYPE_ALIASES)
                raise ValueError(f"Unsupported kv_cache_dtype: {self.kv_cache_dtype!r}. Choose from: {choices}")
            self.kv_cache_dtype = KV_CACHE_DTYPE_ALIASES[self.kv_cache_dtype]
            self.enforce_eager = True
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self._init_speculative_config()

    def _init_speculative_config(self):
        if self.speculative_model is None:
            return
        if not os.path.isdir(self.speculative_model):
            raise ValueError(f"speculative_model must be a local directory: {self.speculative_model!r}")

        self.speculative_method = (self.speculative_method or "eagle3").lower()
        if self.speculative_method != "eagle3":
            raise NotImplementedError(f"Unsupported speculative_method: {self.speculative_method!r}")
        if self.tensor_parallel_size != 1:
            raise NotImplementedError("Eagle3 speculative decoding v1 only supports tensor_parallel_size=1.")
        if self.hf_config.model_type != "qwen3":
            raise NotImplementedError("Eagle3 speculative decoding v1 only supports dense Qwen3 verifier models.")
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1.")

        layer_ids = self.eagle3_target_layer_ids
        if layer_ids is None:
            num_layers = self.hf_config.num_hidden_layers
            layer_ids = (2, num_layers // 2, num_layers - 3)
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if len(layer_ids) != 3:
            raise ValueError("eagle3_target_layer_ids must contain exactly 3 layer ids.")
        invalid = [layer_id for layer_id in layer_ids if layer_id < 0 or layer_id >= self.hf_config.num_hidden_layers]
        if invalid:
            raise ValueError(
                "eagle3_target_layer_ids out of range for verifier "
                f"num_hidden_layers={self.hf_config.num_hidden_layers}: {invalid}"
            )
        self.eagle3_target_layer_ids = layer_ids

        self.eagle3_config = load_eagle3_config(self.speculative_model)
        if self.eagle3_config.target_vocab_size != self.hf_config.vocab_size:
            raise ValueError(
                "Eagle3 speculator target vocab size does not match verifier vocab size: "
                f"{self.eagle3_config.target_vocab_size} != {self.hf_config.vocab_size}"
            )
        if self.eagle3_config.target_hidden_size != self.hf_config.hidden_size:
            raise ValueError(
                "Eagle3 speculator target hidden size does not match verifier hidden size: "
                f"{self.eagle3_config.target_hidden_size} != {self.hf_config.hidden_size}"
            )
        self.enforce_eager = True
