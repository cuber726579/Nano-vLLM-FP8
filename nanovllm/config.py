import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.quantization import QuantConfig

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


@dataclass
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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.rope_parameters = normalize_rope_config(self.hf_config)
        self.quant_config = QuantConfig.from_hf_config(self.hf_config, self.quantization)
        if self.quant_config is not None:
            self.quantization = self.quant_config.quant_method
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
