from transformers import LlamaConfig

from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.quantization.base import QuantConfig


class LlamaForCausalLM(Qwen2ForCausalLM):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__(config, quant_config=quant_config)
