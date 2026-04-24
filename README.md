# Nano-vLLM-FP8

This project is an improved implementation based on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm/tree/f438ce463f24700fb1d4671934abd2714d9e865f), adding support for `Qwen3-0.6B-FP8` inference and text-only inference for local `Qwen3.5-9B` checkpoints.


## Key Features

* **FP8 Inference Support** - End-to-end inference support for `Qwen3-0.6B-FP8`
* **Qwen3.5 Text Support** - Supports the `qwen3_5` text backbone, including hybrid `linear_attention/full_attention` layers
* **Chunked Prefill Support** - Supports chunked prompt scheduling so long prefills can make progress under batched token budget limits
* **RoPE Compatibility** - Adds compatibility for Qwen3 RoPE configs across different transformers versions. See [ROPE.md](./ROPE.md) or upstream [PR #214](https://github.com/GeeeekExplorer/nano-vllm/pull/214) for the compatibility details.


## Installation

```bash
pip install git+https://github.com/cuber726579/Nano-vLLM-FP8.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B-FP8 \
  --local-dir ~/huggingface/Qwen3-0.6B-FP8/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for a minimal local `Qwen3.5-9B` inference example. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

Current limitation: the `Qwen3.5-9B` path is text-only. Vision/video inputs and the multimodal branch of `Qwen3_5ForConditionalGeneration` are not loaded by this runtime.

`bench.py` can be used to benchmark the FP8 inference path.

## RoPE Compatibility

Qwen3 RoPE config compatibility notes are documented in [ROPE.md](./ROPE.md).



## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4080 (32GB)
- Model: Qwen3-0.6B-FP8
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| Nano-vLLM-FP8  | 133,966     | 33.10    | 4047.47               |
| vLLM           | 133,966     | 33.25    | 4029.13               |

**Chunked Prefill Update:**

Nano-vLLM-FP8 now supports chunked prefill, allowing oversized prefills to be split across scheduling rounds instead of waiting for the whole prompt budget at once.

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| Nano-vLLM-FP8 + Chunked Prefill | 133,966 | 31.01 | 4320.04 |
