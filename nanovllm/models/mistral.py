from transformers import MistralConfig

from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.quantization.base import QuantConfig


class MistralForCausalLM(Qwen2ForCausalLM):

    def __init__(
        self,
        config: MistralConfig,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__(config, quant_config=quant_config)
