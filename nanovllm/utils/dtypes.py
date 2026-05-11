import torch

from nanovllm.config import Config


def get_model_dtype(hf_config):
    return getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)


def get_kv_cache_dtype(config: Config):
    if config.kv_cache_dtype is None:
        return get_model_dtype(config.hf_config)
    if config.kv_cache_dtype == "fp8_e4m3":
        return torch.float8_e4m3fn
    if config.kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported kv_cache_dtype: {config.kv_cache_dtype!r}")


def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
