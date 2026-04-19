# Nano-vLLM-FP8

This project is an improved implementation based on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm/tree/f438ce463f24700fb1d4671934abd2714d9e865f), adding support for `Qwen3-0.6B-FP8` inference.


## Key Features

* FP8 **Inference Support** - End-to-end inference support for `Qwen3-0.6B-FP8`

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

See `example.py` for a minimal `Qwen3-0.6B-FP8` inference example. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

`bench.py` can be used to benchmark the FP8 inference path.

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
