import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    ignored_weight_prefixes = tuple(getattr(model, "ignored_weight_prefixes", ()))
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            if weight_name.startswith(ignored_weight_prefixes):
                                break
                            raise
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError:
                        if weight_name.startswith(ignored_weight_prefixes):
                            continue
                        raise
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
