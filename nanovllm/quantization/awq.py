from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.quantization.base import LinearMethod, QuantConfig, ceil_div


def awq_pack_factor(bits: int) -> int:
    if bits <= 0 or 32 % bits != 0:
        raise ValueError(f"Invalid AWQ bits: {bits!r}")
    return 32 // bits


def unpack_awq_packed_tensor(
    packed: torch.Tensor,
    bits: int,
    unpacked_size: int | None = None,
) -> torch.Tensor:
    pack_factor = awq_pack_factor(bits)
    mask = (1 << bits) - 1
    shifts = torch.arange(
        0,
        bits * pack_factor,
        bits,
        device=packed.device,
        dtype=torch.int32,
    )
    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(packed.to(torch.int32).unsqueeze(-1), shifts),
        mask,
    )
    unpacked = unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * pack_factor)
    if unpacked_size is not None:
        unpacked = unpacked[..., :unpacked_size]
    return unpacked


def dequantize_awq_gemm(
    qweight: torch.Tensor,
    qzeros: torch.Tensor | None,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    zero_point: bool,
    output_dtype: torch.dtype,
    input_size: int | None = None,
    output_size: int | None = None,
) -> torch.Tensor:
    pack_factor = awq_pack_factor(bits)
    input_size = qweight.size(0) if input_size is None else input_size
    output_size = qweight.size(1) * pack_factor if output_size is None else output_size
    num_groups = ceil_div(input_size, group_size)

    q = unpack_awq_packed_tensor(qweight[:input_size], bits, output_size).to(torch.float32)
    scale = scales[:num_groups, :output_size].to(device=q.device, dtype=torch.float32)
    if scale.shape != (num_groups, output_size):
        raise ValueError(
            "Invalid AWQ GEMM scales shape: "
            f"expected at least {(num_groups, output_size)}, got {tuple(scales.shape)}."
        )

    group_ids = torch.arange(input_size, device=q.device) // group_size
    scale = scale.index_select(0, group_ids)
    if zero_point:
        if qzeros is None:
            raise ValueError("Zero-point AWQ requires qzeros.")
        zeros = unpack_awq_packed_tensor(qzeros[:num_groups], bits, output_size).to(torch.float32) + 1
        if zeros.shape != (num_groups, output_size):
            raise ValueError(
                "Invalid AWQ GEMM qzeros shape: "
                f"expected at least {(num_groups, ceil_div(output_size, pack_factor))}, got {tuple(qzeros.shape)}."
            )
        q = q - zeros.index_select(0, group_ids)
    return (q * scale).t().contiguous().to(output_dtype)


def dequantize_awq_gemv(
    qweight: torch.Tensor,
    qzeros: torch.Tensor | None,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    zero_point: bool,
    output_dtype: torch.dtype,
    input_size: int | None = None,
    output_size: int | None = None,
) -> torch.Tensor:
    output_size = qweight.size(0) if output_size is None else output_size
    input_size = qweight.size(1) * awq_pack_factor(bits) if input_size is None else input_size
    num_groups = ceil_div(input_size, group_size)

    q = unpack_awq_packed_tensor(qweight[:output_size], bits, input_size).to(torch.float32)
    scale = scales[:output_size, :num_groups].to(device=q.device, dtype=torch.float32)
    if scale.shape != (output_size, num_groups):
        raise ValueError(
            "Invalid AWQ GEMV scales shape: "
            f"expected at least {(output_size, num_groups)}, got {tuple(scales.shape)}."
        )

    group_ids = torch.arange(input_size, device=q.device) // group_size
    scale = scale.index_select(1, group_ids)
    if zero_point:
        if qzeros is None:
            raise ValueError("Zero-point AWQ requires qzeros.")
        zeros = unpack_awq_packed_tensor(qzeros[:output_size], bits, num_groups).to(torch.float32) + 1
        if zeros.shape != (output_size, num_groups):
            raise ValueError(
                "Invalid AWQ GEMV qzeros shape: "
                f"expected at least {(output_size, ceil_div(num_groups, awq_pack_factor(bits)))}, "
                f"got {tuple(qzeros.shape)}."
            )
        q = q - zeros.index_select(1, group_ids)
    return (q * scale).contiguous().to(output_dtype)


def dequantize_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor | None,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
    zero_point: bool,
    output_dtype: torch.dtype | None = None,
    version: str = "gemm",
    input_size: int | None = None,
    output_size: int | None = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = scales.dtype
    if version == "gemm":
        return dequantize_awq_gemm(
            qweight,
            qzeros,
            scales,
            bits,
            group_size,
            zero_point,
            output_dtype,
            input_size=input_size,
            output_size=output_size,
        )
    if version == "gemv":
        return dequantize_awq_gemv(
            qweight,
            qzeros,
            scales,
            bits,
            group_size,
            zero_point,
            output_dtype,
            input_size=input_size,
            output_size=output_size,
        )
    raise NotImplementedError(f"Unsupported AWQ version: {version!r}")


def reference_awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor | None,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    bits: int,
    group_size: int,
    zero_point: bool,
    version: str,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    weight = dequantize_awq(
        qweight,
        qzeros,
        scales,
        bits,
        group_size,
        zero_point,
        output_dtype=x.dtype,
        version=version,
        input_size=input_size,
        output_size=output_size,
    )
    return F.linear(x, weight, bias)


def _shard_id_to_index(loaded_shard_id: int | str) -> int:
    if isinstance(loaded_shard_id, str):
        return {"q": 0, "k": 1, "v": 2}[loaded_shard_id]
    return int(loaded_shard_id)


def _copy_loaded_tensor(
    layer: nn.Module,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    tensor_name: str,
    loaded_shard_id: int | str | None,
) -> None:
    if param.data.shape != loaded_weight.shape:
        raise ValueError(
            "Cannot load AWQ parameter "
            f"{getattr(layer, 'quant_prefix', None) or '<unnamed>'}.{tensor_name}"
            f" shard={loaded_shard_id!r}: expected shape {tuple(param.data.shape)}, "
            f"got {tuple(loaded_weight.shape)}."
        )
    param.data.copy_(loaded_weight)


class AwqLinearMethod(LinearMethod):

    def __init__(self, quant_config: QuantConfig):
        if quant_config.bits != 4:
            raise NotImplementedError(f"Unsupported AWQ bits: {quant_config.bits!r}. Only 4-bit AWQ is supported.")
        if quant_config.group_size is None or quant_config.group_size <= 0:
            raise ValueError(f"Invalid AWQ group_size: {quant_config.group_size!r}")
        if quant_config.zero_point is not True:
            raise NotImplementedError("Only zero-point AWQ checkpoints are supported.")
        version = quant_config.version or "gemm"
        if version not in ("gemm", "gemv"):
            raise NotImplementedError(f"Unsupported AWQ version: {version!r}. Choose 'gemm' or 'gemv'.")

        self.quant_config = quant_config
        self.bits = quant_config.bits
        self.group_size = quant_config.group_size
        self.zero_point = quant_config.zero_point
        self.version = version
        self.pack_factor = awq_pack_factor(self.bits)

    def scaled_output_size(self, size: int) -> int:
        if self.version == "gemm":
            return ceil_div(size, self.pack_factor)
        return size

    def scaled_input_size(self, size: int) -> int:
        if self.version == "gemv":
            return ceil_div(size, self.pack_factor)
        return ceil_div(size, self.group_size)

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        self._validate_layer_shape(layer, input_size, output_size)
        layer.awq_input_size = input_size
        layer.awq_output_size = output_size
        layer.awq_bits = self.bits
        layer.awq_group_size = self.group_size
        layer.awq_version = self.version

        if self.version == "gemm":
            input_groups = ceil_div(input_size, self.group_size)
            packed_output_size = ceil_div(output_size, self.pack_factor)
            qweight_shape = (input_size, packed_output_size)
            qzeros_shape = (input_groups, packed_output_size)
            scales_shape = (input_groups, output_size)
        else:
            input_groups = ceil_div(input_size, self.group_size)
            packed_input_size = ceil_div(input_size, self.pack_factor)
            packed_group_size = ceil_div(input_groups, self.pack_factor)
            qweight_shape = (output_size, packed_input_size)
            qzeros_shape = (output_size, packed_group_size)
            scales_shape = (output_size, packed_group_size * self.pack_factor)

        layer.qweight = nn.Parameter(torch.empty(qweight_shape, dtype=torch.int32), requires_grad=False)
        layer.qweight.weight_loader = self._make_loader(layer, "qweight")
        layer.qzeros = nn.Parameter(torch.empty(qzeros_shape, dtype=torch.int32), requires_grad=False)
        layer.qzeros.weight_loader = self._make_loader(layer, "qzeros")
        layer.scales = nn.Parameter(torch.empty(scales_shape), requires_grad=False)
        layer.scales.weight_loader = self._make_loader(layer, "scales")
        layer.register_parameter("weight", None)

        if bias:
            layer.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
            layer.bias.weight_loader = layer.bias_loader
        else:
            layer.register_parameter("bias", None)

    def _validate_layer_shape(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        if self.version == "gemm" and output_size % self.pack_factor != 0:
            raise ValueError(
                "AWQ GEMM requires output_size to be divisible by the int4 pack factor, "
                f"but got output_size={output_size}, pack_factor={self.pack_factor}, "
                f"layer={getattr(layer, 'quant_prefix', None)!r}."
            )
        if self.version == "gemv" and input_size % self.pack_factor != 0:
            raise ValueError(
                "AWQ GEMV requires input_size to be divisible by the int4 pack factor, "
                f"but got input_size={input_size}, pack_factor={self.pack_factor}, "
                f"layer={getattr(layer, 'quant_prefix', None)!r}."
            )
        if getattr(layer, "tp_dim", None) == 1 and getattr(layer, "tp_size", 1) > 1:
            if input_size % self.group_size != 0:
                raise ValueError(
                    "AWQ row-parallel tensor parallelism requires input shards to align with group_size, "
                    f"but got input_size={input_size}, group_size={self.group_size}, "
                    f"layer={getattr(layer, 'quant_prefix', None)!r}."
                )

    def _make_loader(self, layer: nn.Module, tensor_name: str) -> Callable:
        def loader(
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int | str | None = None,
        ) -> None:
            self._load_param(layer, param, loaded_weight, tensor_name, loaded_shard_id)

        return loader

    def _load_param(
        self,
        layer: nn.Module,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        tensor_name: str,
        loaded_shard_id: int | str | None,
    ) -> None:
        if loaded_shard_id is not None:
            self._load_packed_output_shard(layer, param, loaded_weight, tensor_name, loaded_shard_id)
            return

        tp_dim = getattr(layer, "tp_dim", None)
        if tp_dim is None or getattr(layer, "tp_size", 1) == 1:
            _copy_loaded_tensor(layer, param, loaded_weight, tensor_name, loaded_shard_id)
            return

        if tp_dim == 0:
            shard_dim = self._output_dim(tensor_name)
            shard_size = param.data.size(shard_dim)
            start_idx = layer.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
            _copy_loaded_tensor(layer, param, loaded_weight, tensor_name, loaded_shard_id)
            return

        if tp_dim == 1:
            loaded_weight = self._slice_row_parallel_input(layer, param, loaded_weight, tensor_name)
            _copy_loaded_tensor(layer, param, loaded_weight, tensor_name, loaded_shard_id)
            return

        raise ValueError(f"Unsupported AWQ tensor parallel dimension: {tp_dim!r}")

    def _load_packed_output_shard(
        self,
        layer: nn.Module,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        tensor_name: str,
        loaded_shard_id: int | str,
    ) -> None:
        output_sizes = getattr(layer, "scale_output_sizes", None)
        if output_sizes is None:
            raise ValueError(
                "AWQ packed shard loading requires layer.scale_output_sizes, "
                f"layer={getattr(layer, 'quant_prefix', None)!r}."
            )
        shard_index = _shard_id_to_index(loaded_shard_id)
        if shard_index >= len(output_sizes):
            raise ValueError(f"Invalid AWQ packed shard id {loaded_shard_id!r} for sizes {output_sizes!r}.")

        output_dim = self._output_dim(tensor_name)
        transformed_sizes = [self._transform_output_size(tensor_name, int(size)) for size in output_sizes]
        shard_offset = sum(transformed_sizes[:shard_index])
        shard_size = transformed_sizes[shard_index]
        param_data = param.data.narrow(output_dim, shard_offset, shard_size)

        loaded_start = getattr(layer, "tp_rank", 0) * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, loaded_start, shard_size)
        if param_data.shape != loaded_weight.shape:
            raise ValueError(
                "Cannot load packed AWQ parameter "
                f"{getattr(layer, 'quant_prefix', None) or '<unnamed>'}.{tensor_name}"
                f" shard={loaded_shard_id!r}: expected shape {tuple(param_data.shape)}, "
                f"got {tuple(loaded_weight.shape)}."
            )
        param_data.copy_(loaded_weight)

    def _slice_row_parallel_input(
        self,
        layer: nn.Module,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        tensor_name: str,
    ) -> torch.Tensor:
        if self.version == "gemm":
            shard_dim = 0
            shard_size = param.data.size(shard_dim)
            start_idx = layer.tp_rank * shard_size
            return loaded_weight.narrow(shard_dim, start_idx, shard_size)

        if tensor_name == "qweight":
            shard_dim = 1
            shard_size = param.data.size(shard_dim)
            start_idx = layer.tp_rank * shard_size
            return loaded_weight.narrow(shard_dim, start_idx, shard_size)

        local_groups = ceil_div(layer.awq_input_size, self.group_size)
        if local_groups % self.pack_factor != 0:
            raise NotImplementedError(
                "AWQ GEMV row-parallel qzeros/scales loading requires group shards to be pack-aligned, "
                f"but got local_groups={local_groups}, pack_factor={self.pack_factor}, "
                f"layer={getattr(layer, 'quant_prefix', None)!r}."
            )
        start_group = layer.tp_rank * local_groups
        if tensor_name == "qzeros":
            shard_dim = 1
            shard_size = param.data.size(shard_dim)
            start_idx = start_group // self.pack_factor
            return loaded_weight.narrow(shard_dim, start_idx, shard_size)
        if tensor_name == "scales":
            shard_dim = 1
            shard_size = param.data.size(shard_dim)
            return loaded_weight.narrow(shard_dim, start_group, shard_size)
        raise ValueError(f"Unsupported AWQ tensor name: {tensor_name!r}")

    def _output_dim(self, tensor_name: str) -> int:
        if self.version == "gemm":
            return 1
        if tensor_name in ("qweight", "qzeros", "scales"):
            return 0
        raise ValueError(f"Unsupported AWQ tensor name: {tensor_name!r}")

    def _transform_output_size(self, tensor_name: str, output_size: int) -> int:
        if self.version == "gemm" and tensor_name in ("qweight", "qzeros"):
            return ceil_div(output_size, self.pack_factor)
        return output_size

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            bias = layer.bias
        return reference_awq_linear(
            x.contiguous(),
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
            self.bits,
            self.group_size,
            self.zero_point,
            self.version,
            layer.awq_input_size,
            layer.awq_output_size,
        )
