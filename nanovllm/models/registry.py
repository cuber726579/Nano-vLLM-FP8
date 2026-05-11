from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.config import Config


model_dict = {
    "qwen2": "nanovllm.models.qwen2:Qwen2ForCausalLM",
    "qwen3": "nanovllm.models.qwen3:Qwen3ForCausalLM",
    "qwen3_5_text": "nanovllm.models.qwen3_5:Qwen3_5ForCausalLM",
    "llama": "nanovllm.models.llama:LlamaForCausalLM",
    "mistral": "nanovllm.models.mistral:MistralForCausalLM",
}


def _load_model_cls(model_path: str):
    module_name, class_name = model_path.split(":")
    module = import_module(module_name)
    return getattr(module, class_name)


def build_model_from_config(config: "Config"):
    hf_config = config.hf_config
    model_path = model_dict.get(hf_config.model_type)
    if model_path is None:
        raise NotImplementedError(f"Unsupported model_type: {hf_config.model_type!r}")
    model_cls = _load_model_cls(model_path)
    return model_cls(hf_config, quant_config=config.quant_config)
