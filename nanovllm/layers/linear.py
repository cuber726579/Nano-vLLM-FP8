import torch
from torch import nn
import torch.distributed as dist

from nanovllm.quantization.base import LinearMethod, UnquantizedLinearMethod


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        linear_method: LinearMethod | None = None,
        module_name: str | None = None,
        module_aliases: tuple[str, ...] = (),
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_module_names = tuple(name for name in (module_name, *module_aliases) if name)
        self.linear_method = linear_method or UnquantizedLinearMethod()
        self.linear_method.create_weights(self, input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        linear_method: LinearMethod | None = None,
        module_name: str | None = None,
        module_aliases: tuple[str, ...] = (),
    ):
        super().__init__(
            input_size,
            output_size,
            bias,
            linear_method=linear_method,
            module_name=module_name,
            module_aliases=module_aliases,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_method.apply(self, x)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        linear_method: LinearMethod | None = None,
        module_name: str | None = None,
        module_aliases: tuple[str, ...] = (),
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            tp_dim=0,
            linear_method=linear_method,
            module_name=module_name,
            module_aliases=module_aliases,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_method.apply(self, x)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        linear_method: LinearMethod | None = None,
        module_name: str | None = None,
        module_aliases: tuple[str, ...] = (),
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            linear_method=linear_method,
            module_name=module_name,
            module_aliases=module_aliases,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        self.weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        output_sizes = [self.linear_method.scaled_output_size(size) for size in self.output_sizes]
        shard_offset = sum(output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        linear_method: LinearMethod | None = None,
        module_name: str | None = None,
        module_aliases: tuple[str, ...] = (),
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(
            hidden_size,
            output_size,
            bias,
            linear_method=linear_method,
            module_name=module_name,
            module_aliases=module_aliases,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        self.weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.linear_method.scaled_output_size(self.num_heads * self.head_size)
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.linear_method.scaled_output_size(self.num_kv_heads * self.head_size)
            shard_offset = self.linear_method.scaled_output_size(self.num_heads * self.head_size)
        else:
            shard_size = self.linear_method.scaled_output_size(self.num_kv_heads * self.head_size)
            shard_offset = self.linear_method.scaled_output_size(
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        linear_method: LinearMethod | None = None,
        module_name: str | None = None,
        module_aliases: tuple[str, ...] = (),
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            tp_dim=1,
            linear_method=linear_method,
            module_name=module_name,
            module_aliases=module_aliases,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        y = self.linear_method.apply(self, x, bias=bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
