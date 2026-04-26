from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
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
    excluded_modules: frozenset[str] = field(default_factory=frozenset)

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

        excluded_modules = cls._parse_excluded_modules(raw_config)

        return cls(
            quant_method=quant_method,
            activation_scheme=activation_scheme,
            fmt=fmt,
            weight_block_size=weight_block_size,
            excluded_modules=excluded_modules,
        )

    @staticmethod
    def _parse_excluded_modules(raw_config: dict[str, Any]) -> frozenset[str]:
        excluded_modules: set[str] = set()
        for field_name in ("modules_to_not_convert", "ignored_layers", "excluded_modules"):
            value = raw_config.get(field_name)
            if value is None:
                continue
            if isinstance(value, str):
                modules = [value]
            else:
                modules = value
            for module in modules:
                if module:
                    excluded_modules.add(str(module))
        return frozenset(excluded_modules)

    def is_module_excluded(self, module_name: str | None) -> bool:
        if not module_name:
            return False
        for excluded_module in self.excluded_modules:
            if module_name == excluded_module or module_name.startswith(excluded_module + "."):
                return True
            if any(char in excluded_module for char in "*?[]") and fnmatchcase(module_name, excluded_module):
                return True
        return False


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


class SelectiveLinearMethod(LinearMethod):

    def __init__(self, quant_config: QuantConfig):
        self.quant_config = quant_config
        self.unquantized_method = UnquantizedLinearMethod()
        if quant_config.quant_method == "fp8":
            from nanovllm.quantization.fp8 import Fp8LinearMethod

            self.quantized_method = Fp8LinearMethod(quant_config)
        else:
            raise NotImplementedError(f"Unsupported quantization method: {quant_config.quant_method!r}")

    def _select_method(self, layer: nn.Module) -> LinearMethod:
        module_names = tuple(getattr(layer, "quant_module_names", ()))
        excluded = [name for name in module_names if self.quant_config.is_module_excluded(name)]
        if not excluded:
            return self.quantized_method

        if len(module_names) > 1:
            alias_names = module_names[1:]
            excluded_aliases = [name for name in alias_names if self.quant_config.is_module_excluded(name)]
            if excluded_aliases and len(excluded_aliases) != len(alias_names):
                raise ValueError(
                    "Cannot partially exclude a packed linear layer from quantization: "
                    f"matched={excluded_aliases!r}, aliases={alias_names!r}."
                )
        return self.unquantized_method

    def scaled_output_size(self, size: int) -> int:
        return self.quantized_method.scaled_output_size(size)

    def scaled_input_size(self, size: int) -> int:
        return self.quantized_method.scaled_input_size(size)

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        selected_method = self._select_method(layer)
        layer.linear_method = selected_method
        selected_method.create_weights(layer, input_size, output_size, bias)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._select_method(layer).apply(layer, x, bias=bias)


def build_linear_method(quant_config: QuantConfig | None) -> LinearMethod:
    if quant_config is None:
        return UnquantizedLinearMethod()
    return SelectiveLinearMethod(quant_config)
