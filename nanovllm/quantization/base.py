from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class QuantConfig:
    quant_method: str
    activation_scheme: str
    fmt: str
    weight_block_size: tuple[int, int] | None = None

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        quantization: str | None = None,
    ) -> "QuantConfig | None":
        raw_config = getattr(hf_config, "quantization_config", None)
        if raw_config is None:
            if quantization is not None:
                raise ValueError(
                    f"Model {hf_config.model_type} has no quantization_config, but "
                    f"quantization={quantization!r} was requested."
                )
            return None

        if hasattr(raw_config, "to_dict"):
            raw_config = raw_config.to_dict()
        elif not isinstance(raw_config, dict):
            raw_config = dict(raw_config)

        quant_method = raw_config.get("quant_method")
        if quantization is not None and quantization != quant_method:
            raise ValueError(
                f"Requested quantization={quantization!r} does not match "
                f"checkpoint quant_method={quant_method!r}."
            )
        if quant_method != "fp8":
            raise NotImplementedError(f"Unsupported quantization method: {quant_method!r}")

        activation_scheme = raw_config.get("activation_scheme", "dynamic")
        fmt = raw_config.get("fmt", "e4m3")
        weight_block_size = raw_config.get("weight_block_size")
        if weight_block_size is not None:
            if len(weight_block_size) != 2:
                raise ValueError(f"Invalid weight_block_size: {weight_block_size!r}")
            weight_block_size = tuple(int(x) for x in weight_block_size)

        return cls(
            quant_method=quant_method,
            activation_scheme=activation_scheme,
            fmt=fmt,
            weight_block_size=weight_block_size,
        )


class LinearMethod(ABC):

    def scaled_output_size(self, size: int) -> int:
        return size

    def scaled_input_size(self, size: int) -> int:
        return size

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethod):

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        layer.weight = nn.Parameter(torch.empty(output_size, input_size))
        layer.weight.weight_loader = layer.weight_loader
        if bias:
            layer.bias = nn.Parameter(torch.empty(output_size))
            layer.bias.weight_loader = layer.bias_loader
        else:
            layer.register_parameter("bias", None)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = layer.bias
        return F.linear(x, layer.weight, bias)


def build_linear_method(quant_config: QuantConfig | None) -> LinearMethod:
    if quant_config is None:
        return UnquantizedLinearMethod()
    if quant_config.quant_method == "fp8":
        from nanovllm.quantization.fp8 import Fp8LinearMethod

        return Fp8LinearMethod(quant_config)
    raise NotImplementedError(f"Unsupported quantization method: {quant_config.quant_method!r}")
