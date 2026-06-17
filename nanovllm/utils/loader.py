import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _load_tensor(model: nn.Module, tensor_name: str, loaded_weight: torch.Tensor, shard_id=None):
    try:
        param = model.get_parameter(tensor_name)
    except AttributeError:
        buffer = model.get_buffer(tensor_name)
        buffer.data.copy_(loaded_weight)
        return
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    if shard_id is None:
        weight_loader(param, loaded_weight)
    else:
        weight_loader(param, loaded_weight, shard_id)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    ignored_weight_prefixes = tuple(getattr(model, "ignored_weight_prefixes", ()))
    resolve_weight_name = getattr(model, "resolve_weight_name", None)
    loaded_weight_names = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if resolve_weight_name is not None:
                    resolved = resolve_weight_name(weight_name)
                    if resolved is not None:
                        param_name, shard_id = resolved
                        _load_tensor(model, param_name, f.get_tensor(weight_name), shard_id)
                        loaded_weight_names.add(weight_name)
                        continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            _load_tensor(model, param_name, f.get_tensor(weight_name), shard_id)
                            loaded_weight_names.add(weight_name)
                        except AttributeError:
                            if weight_name.startswith(ignored_weight_prefixes):
                                break
                            raise
                        break
                else:
                    try:
                        _load_tensor(model, weight_name, f.get_tensor(weight_name))
                        loaded_weight_names.add(weight_name)
                    except AttributeError:
                        if weight_name.startswith(ignored_weight_prefixes):
                            continue
                        raise
    return loaded_weight_names
