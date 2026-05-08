from types import SimpleNamespace

import torch

from nanovllm.quantization.awq import dequantize_awq
from nanovllm.quantization.base import QuantConfig


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    pack_factor = 8
    assert values.shape[-1] % pack_factor == 0
    values = values.to(torch.int32).reshape(*values.shape[:-1], -1, pack_factor)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).reshape(
        *((1,) * (values.ndim - 1)),
        pack_factor,
    )
    return torch.sum(values << shifts, dim=-1).to(torch.int32)


def test_quant_config_awq_gemv_extra() -> None:
    hf_config = SimpleNamespace(
        model_type="llama",
        quantization_config={
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemv",
            "backend": "autoawq",
            "do_fuse": False,
            "modules_to_fuse": {"attention": ["q_proj", "k_proj", "v_proj"]},
            "exllama_config": None,
        },
    )
    quant_config = QuantConfig.from_hf_config(hf_config, quantization="awq")
    assert quant_config is not None
    assert quant_config.quant_method == "awq"
    assert quant_config.version == "gemv"
    assert quant_config.bits == 4
    assert quant_config.group_size == 128
    assert quant_config.extra["backend"] == "autoawq"
    assert quant_config.extra["modules_to_fuse"] == {"attention": ["q_proj", "k_proj", "v_proj"]}


def test_dequantize_awq_gemm() -> None:
    q = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7, 3, 2],
            [3, 4, 5, 6, 7, 6, 4, 3],
            [4, 5, 6, 7, 6, 5, 3, 2],
            [5, 6, 7, 6, 5, 4, 2, 1],
        ],
        dtype=torch.int32,
    )
    zeros = torch.tensor(
        [
            [2, 2, 2, 3, 3, 3, 1, 1],
            [3, 3, 2, 2, 1, 1, 2, 2],
        ],
        dtype=torch.int32,
    )
    scales = torch.tensor(
        [
            [0.25, 0.5, 0.75, 1.0, 0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25, 1.0, 0.75, 0.5, 0.25],
        ],
        dtype=torch.float32,
    )
    qweight = pack_int4(q)
    qzeros = pack_int4(zeros - 1)
    group_ids = torch.arange(q.size(0)) // 2
    expected = ((q.float() - zeros.index_select(0, group_ids).float()) * scales.index_select(0, group_ids)).t()
    actual = dequantize_awq(qweight, qzeros, scales, 4, 2, True, output_dtype=torch.float32, version="gemm")
    assert torch.allclose(actual, expected)


def test_dequantize_awq_gemv() -> None:
    q = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7, 3, 2],
            [3, 4, 5, 6, 7, 6, 4, 3],
            [4, 5, 6, 7, 6, 5, 3, 2],
        ],
        dtype=torch.int32,
    )
    zeros = torch.tensor(
        [
            [2, 2, 3, 3],
            [3, 2, 2, 1],
            [2, 3, 1, 2],
        ],
        dtype=torch.int32,
    )
    scales = torch.tensor(
        [
            [0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25],
            [0.5, 1.0, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    qweight = pack_int4(q)
    padded_zeros = torch.nn.functional.pad(zeros - 1, (0, 4))
    padded_scales = torch.nn.functional.pad(scales, (0, 4))
    qzeros = pack_int4(padded_zeros)
    group_ids = torch.arange(q.size(1)) // 2
    expected = (q.float() - zeros.index_select(1, group_ids).float()) * scales.index_select(1, group_ids)
    actual = dequantize_awq(
        qweight,
        qzeros,
        padded_scales,
        4,
        2,
        True,
        output_dtype=torch.float32,
        version="gemv",
    )
    assert torch.allclose(actual, expected)


if __name__ == "__main__":
    test_quant_config_awq_gemv_extra()
    test_dequantize_awq_gemm()
    test_dequantize_awq_gemv()
    print("AWQ tests passed")
