import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

from transformers import AutoConfig
from nanovllm.config import resolve_runtime_config

from dotenv import load_dotenv
from safetensors import safe_open



DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FP8"
LAYER_RE = re.compile(r"(?:^|\.)(?:h|layers|blocks)\.(\d+)\.")


def resolve_model_path(model: str) -> str:
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path)

    cache_home = os.getenv("MODELSCOPE_CACHE")
    if cache_home:
        modelscope_path = Path(cache_home) / "models" / model
        if modelscope_path.is_dir():
            return str(modelscope_path)

    return model


def compact_config_dict(config) -> dict:
    keys = [
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rope_parameters",
        "rope_scaling",
        "torch_dtype",
        "quantization_config",
    ]
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


def load_config(model_path: str, full: bool):
    raw_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    runtime_config = resolve_runtime_config(raw_config)
    config_dict = runtime_config.to_dict() if full else compact_config_dict(runtime_config)
    return raw_config, runtime_config, config_dict


def safetensor_files(model_path: str) -> list[str]:
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under: {model_path}")
    return files


def tensor_meta(file: str, weight_name: str) -> tuple[list[int], str]:
    with safe_open(file, framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(weight_name)
        return tensor_slice.get_shape(), tensor_slice.get_dtype()


def iter_weights(model_path: str, layer_pattern: str | None):
    layer_regex = re.compile(layer_pattern) if layer_pattern else None
    for file in safetensor_files(model_path):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in sorted(f.keys(), key=natural_weight_key):
                if layer_regex and not layer_regex.search(name):
                    continue
                tensor_slice = f.get_slice(name)
                yield {
                    "file": os.path.basename(file),
                    "layer": layer_label(name),
                    "name": name,
                    "shape": tensor_slice.get_shape(),
                    "dtype": tensor_slice.get_dtype(),
                }


def layer_label(name: str) -> str:
    match = LAYER_RE.search(name)
    if match:
        return match.group(1)
    return "-"


def natural_weight_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def print_config(model_path: str, full: bool):
    raw_config, runtime_config, config_dict = load_config(model_path, full)
    print("=== Model Config ===")
    print(f"path: {model_path}")
    print(f"raw_model_type: {raw_config.model_type}")
    print(f"runtime_model_type: {runtime_config.model_type}")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False, default=str))
    print()


def print_weights(model_path: str, layer_pattern: str | None, max_weights: int | None):
    if safe_open is None:
        raise ImportError("safetensors is required to inspect weight name/shape/dtype.")

    print("=== Weight Tensors ===")
    print(f"{'layer':>5}  {'dtype':<12}  {'shape':<16}  name")
    print(f"{'-' * 5}  {'-' * 12}  {'-' * 16}  {'-' * 40}")

    count = 0
    current_file = None
    for info in iter_weights(model_path, layer_pattern):
        if current_file != info["file"]:
            current_file = info["file"]
            print(f"\n# {current_file}")
        shape = "x".join(str(dim) for dim in info["shape"])
        print(f"{info['layer']:>5}  {info['dtype']:<12}  {shape:<16}  {info['name']}")
        count += 1
        if max_weights is not None and count >= max_weights:
            print(f"\n... stopped after --max-weights={max_weights}")
            break

    print(f"\nTotal shown weights: {count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print model config and safetensors weight names, shapes, and dtypes."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL_ID,
        help="Local model path or model id under $MODELSCOPE_CACHE/models.",
    )
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Print the full runtime HF config instead of the compact summary.",
    )
    parser.add_argument(
        "--layer-pattern",
        help="Only show weights whose names match this regular expression, e.g. 'layers\\\\.(0|1)\\\\.'.",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        help="Limit the number of weight rows printed.",
    )
    return parser.parse_args()


def main():
    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    model_path = resolve_model_path(args.model)
    print_config(model_path, args.full_config)
    print_weights(model_path, args.layer_pattern, args.max_weights)


if __name__ == "__main__":
    main()
