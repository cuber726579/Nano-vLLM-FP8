# AWQ 模型分析: Qwen3-4B-AWQ

本文以本地 checkpoint `./Huggingface/models/Qwen/Qwen3-4B-AWQ` 为例，说明 AWQ 模型的配置、权重布局，以及它接入当前 nano-vLLM 量化框架时需要关注的点。

## 结论

- 这个模型是 Qwen3 dense 架构，`model_type="qwen3"`，36 层，hidden size 2560，attention heads 32，KV heads 8。
- 量化方式是标准 Qwen3 AWQ W4A16:
  - `quant_method="awq"`
  - `bits=4`
  - `group_size=128`
  - `zero_point=true`
  - `version="gemm"`
- AWQ 只量化大矩阵 Linear 权重。Embedding、RMSNorm、q/k norm、tied `lm_head` 等仍保留 BF16 权重。
- safetensors 中每个被量化 Linear 不再是一个 `.weight`，而是三组张量:
  - `qweight`: int4 packed 到 int32
  - `qzeros`: zero point packed 到 int32
  - `scales`: FP16 scale
- 当前项目的 `QuantConfig` 只接受 `quant_method="fp8"`。所以这个模型现在会在配置解析阶段报 `Unsupported quantization method: 'awq'`，还不能直接由 `LLM(...)` 加载。

## 模型配置

本地 `config.json` 里的核心字段如下:

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 2560,
  "intermediate_size": 9728,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "max_position_embeddings": 40960,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "torch_dtype": "float16",
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "modules_to_not_convert": null,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": true
  }
}
```

这里的 `torch_dtype="float16"` 表示推理计算通常按 FP16 走；AWQ 权重本身是 4bit packed 权重，运行时要在 GEMM kernel 中按 group scale / zero point 反量化后参与矩阵乘。

## AWQ 是什么

AWQ 是 Activation-aware Weight Quantization。它只量化权重，不量化激活，所以常见描述是 W4A16: weight 4bit，activation FP16/BF16。

它的核心想法是: 权重量化误差对模型输出的影响并不均匀，哪些通道更重要可以从激活分布里看出来。AWQ 会在离线量化阶段用校准数据估计激活敏感度，对重要通道做保护或缩放，让 4bit 权重在推理时尽量接近原始模型效果。

对推理框架来说，AWQ checkpoint 主要带来两个变化:

- 权重文件里没有普通的浮点 Linear weight，而是 packed int4 权重加 scale / zero point。
- forward 不能直接 `F.linear(x, weight)`，需要 AWQ 专用 LinearMethod 或 kernel。

## 权重文件结构

本地目录只有一个主权重文件:

```text
model.safetensors
```

safetensors header 显示:

```text
tensor_count: 902
metadata: {"format": "pt", "lm_head.weight": "model.embed_tokens.weight"}
```

按后缀统计:

| 后缀 | 数量 | dtype | 含义 |
| --- | ---: | --- | --- |
| `qweight` | 252 | I32 | int4 权重打包后的 int32 存储 |
| `qzeros` | 252 | I32 | zero point 打包后的 int32 存储 |
| `scales` | 252 | F16 | 每个 group 的缩放因子 |
| `weight` | 146 | BF16 | 未量化模块权重 |

252 个量化模块来自:

```text
36 layers * 7 linear modules per layer = 252
```

每层 7 个被 AWQ 量化的 Linear 是:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

第 0 层的实际张量形状如下:

| 模块 | `qweight` | `qzeros` | `scales` |
| --- | --- | --- | --- |
| `self_attn.q_proj` | I32 `[2560, 512]` | I32 `[20, 512]` | F16 `[20, 4096]` |
| `self_attn.k_proj` | I32 `[2560, 128]` | I32 `[20, 128]` | F16 `[20, 1024]` |
| `self_attn.v_proj` | I32 `[2560, 128]` | I32 `[20, 128]` | F16 `[20, 1024]` |
| `self_attn.o_proj` | I32 `[4096, 320]` | I32 `[32, 320]` | F16 `[32, 2560]` |
| `mlp.gate_proj` | I32 `[2560, 1216]` | I32 `[20, 1216]` | F16 `[20, 9728]` |
| `mlp.up_proj` | I32 `[2560, 1216]` | I32 `[20, 1216]` | F16 `[20, 9728]` |
| `mlp.down_proj` | I32 `[9728, 320]` | I32 `[76, 320]` | F16 `[76, 2560]` |

这些 shape 可以反推出 AWQ 的打包规则:

- `shape` 按照 `[in_features, out_features]` 的格式
- `group_size=128`，所以 `q_proj` 输入维 2560 会分成 `2560 / 128 = 20` 个 group，`scales` 第一维就是 20。
- 每个 int32 存 8 个 4bit 值，所以 `q_proj` 输出维 4096 会变成 `4096 / 8 = 512`，对应 `qweight` 第二维 512。
- `qzeros` 同样按 int32 打包 zero point，形状和 packed 输出维相关。

未量化权重例子:

```text
model.embed_tokens.weight BF16 [151936, 2560]
model.layers.0.input_layernorm.weight BF16 [2560]
model.layers.0.post_attention_layernorm.weight BF16 [2560]
```

由于 `tie_word_embeddings=true`，safetensors metadata 里标记:

```text
lm_head.weight -> model.embed_tokens.weight
```

因此 `lm_head` 不单独存一份权重。

## 和当前 nano-vLLM 代码的关系

当前项目已经有一套比较适合扩展 AWQ 的量化抽象:

- `nanovllm/config.py` 会从 HF config 读取 `quantization_config`。
- `nanovllm/quantization/base.py` 有 `QuantConfig` 和 `LinearMethod`。
- `LinearBase` 会根据 `quant_config.get_quant_method()` 给 Linear 层创建不同的权重参数和 forward 方法。
- `Qwen3ForCausalLM.packed_modules_mapping` 会把 HF checkpoint 的分离权重映射到项目里的 packed linear:
  - `q_proj/k_proj/v_proj` -> `qkv_proj`
  - `gate_proj/up_proj` -> `gate_up_proj`

但是现在只注册了 FP8:

```python
if quant_method != "fp8":
    raise NotImplementedError(f"Unsupported quantization method: {quant_method!r}")
```

所以加载 `Qwen3-4B-AWQ` 的第一处阻塞点不是权重 shape，而是配置解析。即使放开配置，现有 Linear 参数也只认识普通 `weight` 或 FP8 scale，不会创建 `qweight/qzeros/scales` 这些 AWQ 参数。

## 接入 AWQ 需要做什么

最小可运行路径建议分两步走。

第一步先做正确性 fallback:

- 扩展 `QuantConfig`，接受 AWQ 字段:
  - `bits`
  - `group_size`
  - `zero_point`
  - `version`
- 新增 `nanovllm/quantization/awq.py`。
- 实现 `AwqLinearMethod.create_weights()`，注册:
  - `qweight`
  - `qzeros`
  - `scales`
  - 可选 `bias`
- 实现 PyTorch reference 反量化:
  - unpack int32 -> int4
  - unpack zero point
  - 按 group 应用 `(q - zero) * scale`
  - 得到临时 FP16/BF16 weight 后走 `F.linear`
- 让 loader 能把 checkpoint 的 `*.qweight/*.qzeros/*.scales` 正确切片到 packed QKV 和 gate/up 层。

第二步再做性能路径:

- 实现 AWQ GEMM kernel，直接消费 `qweight/qzeros/scales`。
- 输入激活保持 FP16/BF16，kernel 内按需反量化 int4 权重并累加。
- 对不满足约束的 shape 回退到 reference 路径。

这样可以先保证 checkpoint 能完整加载、短 prompt 能出非空文本，再逐步把性能补上。

## 分片风险点

AWQ 最容易出错的是 Tensor Parallel 分片。普通浮点 weight 可以直接沿行或列切，但 AWQ 的权重已经 packed:

- 4bit pack factor 是 8，输出维分片要和 8 对齐。
- `group_size=128` 影响输入维分组，输入维分片要保持 group 边界一致。
- `qweight`、`qzeros`、`scales` 的切片维度不完全等价，不能简单复用普通 `weight_loader`。
- packed QKV 和 packed gate/up 会把多个 HF 模块拼到一个项目参数里，scale 和 zero point 也要按同样逻辑拼接。

建议先支持 TP=1，把真实 `Qwen3-4B-AWQ` 跑通；再为 TP>1 增加带 shape 校验的 shard loader。

## 验证建议

可以按下面顺序验证:

1. 读取 `config.json`，确认 AWQ 字段解析结果正确。
2. 读取 `model.safetensors` header，确认所有量化 Linear 都有 `qweight/qzeros/scales`。
3. 用一个小矩阵写 AWQ unpack / dequant 单测，验证 int4 和 zero point 解包方向。
4. 加载真实 checkpoint，检查没有 unknown parameter 或 missing parameter。
5. 用 1 个短 prompt 进行端到端生成，确认输出非空。
6. 再测试 batch prompts、较长 prompt、TP>1 和性能 kernel。

## 一句话总结

`Qwen3-4B-AWQ` 是一个典型的 W4A16、group size 128、zero point、GEMM 版 AWQ checkpoint。它的权重布局非常规整，适合作为本项目 AWQ 支持的首个目标模型；当前主要缺口是 `QuantConfig` 不接受 `awq`，以及 Linear 层还没有 `qweight/qzeros/scales` 的参数注册、加载和计算路径。
