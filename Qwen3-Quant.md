# Qwen3 量化配置总结

## 范围

- 统计时间: 2026-04-26
- 统计对象: Hugging Face 上 `Qwen/` 官方发布、`config.json` 中带有可比较 `quantization_config` 的 Qwen3 文本模型
- 不包含:
  - 非量化 BF16/Base/Instruct 原版
  - `GGUF` 等不走 Hugging Face `quantization_config` 标准字段的打包格式

## 配置文件链接索引

下面列出本文统计到的每个模型对应的 `config.json` 网络链接，便于直接核对 `quantization_config`。

### FP8

- `Qwen3-0.6B-FP8`: https://huggingface.co/Qwen/Qwen3-0.6B-FP8/blob/main/config.json
- `Qwen3-1.7B-FP8`: https://huggingface.co/Qwen/Qwen3-1.7B-FP8/blob/main/config.json
- `Qwen3-4B-FP8`: https://huggingface.co/Qwen/Qwen3-4B-FP8/blob/main/config.json
- `Qwen3-8B-FP8`: https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json
- `Qwen3-14B-FP8`: https://huggingface.co/Qwen/Qwen3-14B-FP8/blob/main/config.json
- `Qwen3-32B-FP8`: https://huggingface.co/Qwen/Qwen3-32B-FP8/blob/main/config.json
- `Qwen3-30B-A3B-FP8`: https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/blob/main/config.json
- `Qwen3-235B-A22B-FP8`: https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8/blob/main/config.json

### 2507 FP8

- `Qwen3-4B-Thinking-2507-FP8`: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507-FP8/blob/main/config.json
- `Qwen3-4B-Instruct-2507-FP8`: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8/blob/main/config.json
- `Qwen3-30B-A3B-Thinking-2507-FP8`: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8/blob/main/config.json
- `Qwen3-30B-A3B-Instruct-2507-FP8`: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/blob/main/config.json
- `Qwen3-235B-A22B-Thinking-2507-FP8`: https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8/blob/main/config.json
- `Qwen3-235B-A22B-Instruct-2507-FP8`: https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8/blob/main/config.json

### AWQ

- `Qwen3-4B-AWQ`: https://huggingface.co/Qwen/Qwen3-4B-AWQ/blob/main/config.json
- `Qwen3-8B-AWQ`: https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/main/config.json
- `Qwen3-14B-AWQ`: https://huggingface.co/Qwen/Qwen3-14B-AWQ/blob/main/config.json
- `Qwen3-32B-AWQ`: https://huggingface.co/Qwen/Qwen3-32B-AWQ/blob/main/config.json

### GPTQ

- `Qwen3-0.6B-GPTQ-Int8`: https://huggingface.co/Qwen/Qwen3-0.6B-GPTQ-Int8/blob/main/config.json
- `Qwen3-1.7B-GPTQ-Int8`: https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8/blob/main/config.json
- `Qwen3-30B-A3B-GPTQ-Int4`: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/blob/main/config.json
- `Qwen3-235B-A22B-GPTQ-Int4`: https://huggingface.co/Qwen/Qwen3-235B-A22B-GPTQ-Int4/blob/main/config.json

## 先看结论

- Qwen3 官方文本量化模型目前主要分成 `FP8`、`AWQ`、`GPTQ` 三类。
- `FP8` 族最统一，核心字段基本都是:
  - `quant_method="fp8"`
  - `activation_scheme="dynamic"`
  - `fmt="e4m3"`
  - `weight_block_size=[128, 128]`
- `AWQ` 族核心字段基本都是:
  - `quant_method="awq"`
  - `bits=4`
  - `group_size=128`
  - `version="gemm"`
  - `zero_point=true`
- `GPTQ` 族分成两支:
  - 小模型 dense: `GPTQ-Int8`
  - 大模型 MoE: `GPTQ-Int4`
- 真正影响兼容实现的最大差异，不是 `fmt` 或 `group_size`，而是 `modules_to_not_convert` 是否存在，以及它排除了哪些层。

## 兼容实现最重要的模式

### 1. 标准 dense FP8

- 典型模型: `Qwen3-0.6B-FP8` 到 `Qwen3-32B-FP8`
- `quantization_config` 只包含 4 个核心字段
- 没有 `modules_to_not_convert`

### 2. 2507 dense FP8

- 典型模型: `Qwen3-4B-Thinking-2507-FP8`、`Qwen3-4B-Instruct-2507-FP8`
- 在标准 FP8 基础上新增 `modules_to_not_convert`
- 排除模式基本是:
  - `lm_head`
  - 每层 `input_layernorm`
  - 每层 `post_attention_layernorm`

### 3. MoE FP8

- 典型模型: `Qwen3-30B-A3B-FP8`、`Qwen3-235B-A22B-FP8`
- 同样是标准 FP8 核心字段
- 但会显式排除:
  - `lm_head`
  - 每层 `input_layernorm`
  - 每层 `post_attention_layernorm`
  - 每层 `mlp.gate`

### 4. AWQ

- 官方 Qwen3 AWQ 目前是标准 `W4/G128`
- `modules_to_not_convert` 通常是 `null`
- 8B 版本比 4B/14B/32B 多了一些运行时提示字段，例如 `backend="autoawq"`、`do_fuse=false`

### 5. GPTQ

- `Qwen3-0.6B-GPTQ-Int8` / `Qwen3-1.7B-GPTQ-Int8`:
  - `bits=8`
  - 带 `meta`、`pack_dtype="int32"`、`lm_head=false`
- `Qwen3-30B-A3B-GPTQ-Int4` / `Qwen3-235B-A22B-GPTQ-Int4`:
  - `bits=4`
  - 带 `checkpoint_format="gptq"`、`damp_percent=0.01`、`desc_act=false`、`sym=true`、`true_sequential=true`

## 标准 FP8 模型

| 模型 | 架构 | `quantization_config` 核心字段 | 排除列表 | 备注 | 来源 |
| --- | --- | --- | --- | --- | --- |
| `Qwen3-0.6B-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | 无 | `max_position_embeddings=40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-0.6B-FP8/blob/main/config.json) |
| `Qwen3-1.7B-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | 无 | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-1.7B-FP8/blob/main/config.json) |
| `Qwen3-4B-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | 无 | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-4B-FP8/blob/main/config.json) |
| `Qwen3-8B-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | 无 | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json) |
| `Qwen3-14B-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | 无 | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-14B-FP8/blob/main/config.json) |
| `Qwen3-32B-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | 无 | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-32B-FP8/blob/main/config.json) |
| `Qwen3-30B-A3B-FP8` | MoE | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` + 全层 `mlp.gate` | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/blob/main/config.json) |
| `Qwen3-235B-A22B-FP8` | MoE | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` + 全层 `mlp.gate` | `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8/blob/main/config.json) |

## 2507 FP8 模型

| 模型 | 架构 | `quantization_config` 核心字段 | 排除列表 | 备注 | 来源 |
| --- | --- | --- | --- | --- | --- |
| `Qwen3-4B-Thinking-2507-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` | `max_position_embeddings=262144`, `rope_theta=5000000` | [HF](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507-FP8/blob/main/config.json) |
| `Qwen3-4B-Instruct-2507-FP8` | dense | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` | `262144`, `rope_theta=5000000` | [HF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8/blob/main/config.json) |
| `Qwen3-30B-A3B-Thinking-2507-FP8` | MoE | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` + 全层 `mlp.gate` | `262144`, `rope_theta=10000000` | [HF](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8/blob/main/config.json) |
| `Qwen3-30B-A3B-Instruct-2507-FP8` | MoE | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` + 全层 `mlp.gate` | `262144`, `rope_theta=10000000` | [HF](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/blob/main/config.json) |
| `Qwen3-235B-A22B-Thinking-2507-FP8` | MoE | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` + 全层 `mlp.gate` | `262144`, `rope_theta=5000000` | [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8/blob/main/config.json) |
| `Qwen3-235B-A22B-Instruct-2507-FP8` | MoE | `fp8`, `dynamic`, `e4m3`, `[128,128]` | `lm_head` + 全层 `input_layernorm` + 全层 `post_attention_layernorm` + 全层 `mlp.gate` | `262144`, `rope_theta=5000000` | [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8/blob/main/config.json) |

## AWQ 模型

| 模型 | 架构 | `quantization_config` 核心字段 | 排除列表 | 备注 | 来源 |
| --- | --- | --- | --- | --- | --- |
| `Qwen3-4B-AWQ` | dense | `awq`, `bits=4`, `group_size=128`, `version="gemm"`, `zero_point=true` | `modules_to_not_convert=null` | `torch_dtype=float16`, `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-4B-AWQ/blob/main/config.json) |
| `Qwen3-8B-AWQ` | dense | 同上 | `modules_to_not_convert=null` | 额外带 `backend="autoawq"`、`do_fuse=false`、`modules_to_fuse=null`、`exllama_config=null` | [HF](https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/main/config.json) |
| `Qwen3-14B-AWQ` | dense | `awq`, `bits=4`, `group_size=128`, `version="gemm"`, `zero_point=true` | `modules_to_not_convert=null` | `torch_dtype=float16`, `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-14B-AWQ/blob/main/config.json) |
| `Qwen3-32B-AWQ` | dense | `awq`, `bits=4`, `group_size=128`, `version="gemm"`, `zero_point=true` | `modules_to_not_convert=null` | `torch_dtype=float16`, `40960`, `rope_theta=1000000` | [HF](https://huggingface.co/Qwen/Qwen3-32B-AWQ/blob/main/config.json) |

## GPTQ 模型

| 模型 | 架构 | `quantization_config` 核心字段 | 排除列表 | 备注 | 来源 |
| --- | --- | --- | --- | --- | --- |
| `Qwen3-0.6B-GPTQ-Int8` | dense | `gptq`, `bits=8`, `group_size=128`, `sym=true`, `desc_act=false` | 无显式 `modules_to_not_convert` | 带 `lm_head=false`、`meta`、`pack_dtype="int32"` | [HF](https://huggingface.co/Qwen/Qwen3-0.6B-GPTQ-Int8/blob/main/config.json) |
| `Qwen3-1.7B-GPTQ-Int8` | dense | `gptq`, `bits=8`, `group_size=128`, `sym=true`, `desc_act=false` | 无显式 `modules_to_not_convert` | 带 `lm_head=false`、`meta`、`pack_dtype="int32"` | [HF](https://huggingface.co/Qwen/Qwen3-1.7B-GPTQ-Int8/blob/main/config.json) |
| `Qwen3-30B-A3B-GPTQ-Int4` | MoE | `gptq`, `bits=4`, `group_size=128`, `checkpoint_format="gptq"`, `damp_percent=0.01`, `desc_act=false`, `sym=true`, `true_sequential=true` | 无显式 `modules_to_not_convert` | 还带 `static_groups=false` | [HF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/blob/main/config.json) |
| `Qwen3-235B-A22B-GPTQ-Int4` | MoE | `gptq`, `bits=4`, `group_size=128`, `checkpoint_format="gptq"`, `damp_percent=0.01`, `desc_act=false`, `sym=true`, `true_sequential=true` | 无显式 `modules_to_not_convert` | 还带 `static_groups=false` | [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B-GPTQ-Int4/blob/main/config.json) |

## 对当前项目的直接启示

### 最少要支持的字段

- `quant_method`
- `activation_scheme`
- `fmt`
- `weight_block_size`
- `modules_to_not_convert`
- `bits`
- `group_size`
- `zero_point`
- `version`
- `checkpoint_format`
- `desc_act`
- `sym`
- `true_sequential`

### 建议的兼容优先级

1. 先把 `modules_to_not_convert` 支持补上。
2. 把它内部统一归一为一个排除集合，兼容未来可能出现的 `ignored_layers` / `excluded_modules`。
3. FP8 路径里不要假设“所有 Linear 都量化”。
4. AWQ/GPTQ 如果暂时不做完整执行支持，也至少要在配置解析层识别字段并给出清晰报错。

### 可以先做的简化

- 对 Qwen3 官方 FP8，目前可以先默认:
  - `activation_scheme == "dynamic"`
  - `fmt == "e4m3"`
  - `weight_block_size == (128, 128)`
- 但不能默认:
  - `modules_to_not_convert` 不存在
  - `所有 linear 层都走 FP8`

## 一个适合代码里的内部归一化视图

```python
NormalizedQuantConfig(
    family="fp8" | "awq" | "gptq",
    bits=8 | 4 | None,
    fmt="e4m3" | None,
    activation_scheme="dynamic" | None,
    weight_block_size=(128, 128) | None,
    group_size=128 | None,
    zero_point=True | None,
    excluded_modules=set[str],
    extra=dict[str, Any],
)
```

这样做的好处是:

- `modules_to_not_convert`、`ignored_layers`、`excluded_modules` 都能收敛到 `excluded_modules`
- 执行层只关心“这个层要不要量化”，不关心原始字段名来自哪个生态
- 后续扩展到 `e5m2`、`static activation` 或别的后端时更容易
