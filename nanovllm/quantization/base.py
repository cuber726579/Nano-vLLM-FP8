from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class QuantConfig:
    _EXCLUDED_MODULE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"modules_to_not_convert", "ignored_layers", "excluded_modules"}
    )
    _AWQ_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"quant_method", "bits", "w_bit", "group_size", "q_group_size", "zero_point", "version"}
    )
    _FP8_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"quant_method", "activation_scheme", "fmt", "weight_block_size"}
    )

    quant_method: str
    activation_scheme: str | None = None
    fmt: str | None = None
    weight_block_size: tuple[int, int] | None = None
    excluded_modules: frozenset[str] = field(default_factory=frozenset)
    bits: int | None = None
    group_size: int | None = None
    zero_point: bool | None = None
    version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

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
        raw_config = dict(raw_config)

        quant_method = raw_config.get("quant_method")
        if isinstance(quant_method, str):
            quant_method = quant_method.lower()
        requested_quantization = quantization.lower() if isinstance(quantization, str) else quantization
        if requested_quantization is not None and requested_quantization != quant_method:
            raise ValueError(
                f"Requested quantization={quantization!r} does not match "
                f"checkpoint quant_method={quant_method!r}."
            )

        excluded_modules = cls._parse_excluded_modules(raw_config)

        if quant_method == "awq":
            return cls._from_awq_config(raw_config, quant_method, excluded_modules)

        if quant_method == "fp8":
            return cls._from_fp8_config(raw_config, quant_method, excluded_modules)

        raise NotImplementedError(f"Unsupported quantization method: {quant_method!r}")

    @classmethod
    def _from_awq_config(
        cls,
        raw_config: dict[str, Any],
        quant_method: str,
        excluded_modules: frozenset[str],
    ) -> "QuantConfig":
        bits = cls._read_int_field(raw_config, "bits", "w_bit")
        group_size = cls._read_int_field(raw_config, "group_size", "q_group_size")
        zero_point = raw_config.get("zero_point")
        version = raw_config.get("version", "gemm")
        if isinstance(version, str):
            version = version.lower()
        elif version is None:
            version = "gemm"

        if bits != 4:
            raise NotImplementedError(f"Unsupported AWQ bits: {bits!r}. Only 4-bit AWQ is supported.")
        if group_size is None or group_size <= 0:
            raise ValueError(f"Invalid AWQ group_size: {group_size!r}")
        if zero_point is not True:
            raise NotImplementedError("Only zero-point AWQ checkpoints are supported.")
        if version not in ("gemm", "gemv"):
            raise NotImplementedError(f"Unsupported AWQ version: {version!r}. Choose 'gemm' or 'gemv'.")

        return cls(
            quant_method=quant_method,
            excluded_modules=excluded_modules,
            bits=bits,
            group_size=group_size,
            zero_point=zero_point,
            version=version,
            extra=cls._extra_fields(raw_config, cls._AWQ_FIELDS),
        )

    @classmethod
    def _from_fp8_config(
        cls,
        raw_config: dict[str, Any],
        quant_method: str,
        excluded_modules: frozenset[str],
    ) -> "QuantConfig":
        activation_scheme = raw_config.get("activation_scheme", "dynamic")
        if activation_scheme not in ("dynamic", "static"):
            raise NotImplementedError(f"Unsupported FP8 activation scheme: {activation_scheme!r}")

        weight_block_size = cls._parse_weight_block_size(raw_config.get("weight_block_size"))

        return cls(
            quant_method=quant_method,
            activation_scheme=activation_scheme,
            fmt=raw_config.get("fmt", "e4m3"),
            weight_block_size=weight_block_size,
            excluded_modules=excluded_modules,
            extra=cls._extra_fields(raw_config, cls._FP8_FIELDS),
        )

    @staticmethod
    def _parse_weight_block_size(weight_block_size: Any) -> tuple[int, int] | None:
        if weight_block_size is None:
            return None
        if len(weight_block_size) != 2:
            raise ValueError(f"Invalid weight_block_size: {weight_block_size!r}")
        return (int(weight_block_size[0]), int(weight_block_size[1]))

    @classmethod
    def _extra_fields(cls, raw_config: dict[str, Any], known_fields: frozenset[str]) -> dict[str, Any]:
        known_fields = known_fields | cls._EXCLUDED_MODULE_FIELDS
        return {key: value for key, value in raw_config.items() if key not in known_fields}

    @classmethod
    def _parse_excluded_modules(cls, raw_config: dict[str, Any]) -> frozenset[str]:
        excluded_modules: set[str] = set()
        for field_name in cls._EXCLUDED_MODULE_FIELDS:
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

    @staticmethod
    def _read_int_field(raw_config: dict[str, Any], *field_names: str) -> int | None:
        for field_name in field_names:
            value = raw_config.get(field_name)
            if value is not None:
                return int(value)
        return None

    def is_module_excluded(self, module_name: str | None) -> bool:
        if not module_name:
            return False
        for excluded_module in self.excluded_modules:
            if module_name == excluded_module or module_name.startswith(excluded_module + "."):
                return True
            if any(char in excluded_module for char in "*?[]") and fnmatchcase(module_name, excluded_module):
                return True
        return False

    def get_quant_method(self, layer: nn.Module, prefix: str | None) -> "LinearMethod | None":
        from nanovllm.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            raise TypeError(
                "QuantConfig.get_quant_method() only supports LinearBase layers, "
                f"got {type(layer).__name__}."
            )

        module_names = tuple(name for name in (prefix, *getattr(layer, "quant_module_aliases", ())) if name)
        excluded = [name for name in module_names if self.is_module_excluded(name)]
        if excluded:
            alias_names = module_names[1:]
            excluded_aliases = [name for name in alias_names if self.is_module_excluded(name)]
            if excluded_aliases and len(excluded_aliases) != len(alias_names):
                raise ValueError(
                    "Cannot partially exclude a packed linear layer from quantization: "
                    f"matched={excluded_aliases!r}, aliases={alias_names!r}."
                )
            return UnquantizedLinearMethod()

        if self.quant_method == "fp8":
            from nanovllm.quantization.fp8 import Fp8LinearMethod
            return Fp8LinearMethod(self)

        if self.quant_method == "awq":
            from nanovllm.quantization.awq import AwqLinearMethod
            return AwqLinearMethod(self)

        raise NotImplementedError(f"Unsupported quantization method: {self.quant_method!r}")


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

    if quant_config.quant_method == "awq":
        from nanovllm.quantization.awq import AwqLinearMethod
        return AwqLinearMethod(quant_config)

    raise NotImplementedError(f"Unsupported quantization method: {quant_config.quant_method!r}")
