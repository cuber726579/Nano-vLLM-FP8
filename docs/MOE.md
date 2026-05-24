# MoE 总结: Qwen3-30B-A3B

本文以本地 checkpoint `Huggingface/models/Qwen/Qwen3-30B-A3B/config.json` 为例，整理 Qwen3 MoE 的核心机制、配置参数，以及接入推理框架时需要关注的点。

## 1. MoE 是什么

MoE, Mixture of Experts, 混合专家模型，是一种稀疏激活的 Transformer 结构。Dense 模型在每个 token 上都会计算所有 FFN/MLP 参数；MoE 模型把 FFN/MLP 拆成多个 expert，然后由 router/gate 为每个 token 选择少量 expert 参与计算。

简化流程如下:

```text
hidden_states
  -> attention
  -> router/gate 对每个 token 打分
  -> 选择 Top-K experts
  -> 被选中的 experts 分别计算
  -> 按 router 权重加权合并
  -> 输出到下一层
```

所以 MoE 的核心收益是: 模型可以拥有很大的总参数容量，但每个 token 只激活其中一小部分参数。

## 2. Qwen3-30B-A3B 的整体结构

这个 checkpoint 的关键结构参数如下:

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "hidden_size": 2048,
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "intermediate_size": 6144,
  "moe_intermediate_size": 768,
  "num_experts": 128,
  "num_experts_per_tok": 8
}
```

可以概括为:

- 共有 `48` 个 decoder layer。
- 每个 MoE 层有 `128` 个 expert。
- 每个 token 每层只选择 `8` 个 expert。
- 每个 expert 的 FFN 中间维度是 `768`。
- 模型总参数量约为 30B 级别，但每个 token 的激活参数量约为 A3B 级别。

`Qwen3-30B-A3B` 名字里的 `A3B` 指 activated parameters per token，大意是单 token 实际激活的参数量约 3B，而不是每次都计算全部 30B 参数。

## 3. MoE 相关配置参数

`config.json` 中直接控制 MoE 行为的字段如下:

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `architectures` | `Qwen3MoeForCausalLM` | 使用 Qwen3 MoE 版因果语言模型结构 |
| `model_type` | `qwen3_moe` | 模型类型是 Qwen3 MoE |
| `num_experts` | `128` | 每个 MoE 层里的 expert 数量 |
| `num_experts_per_tok` | `8` | 每个 token 路由到的 expert 数量，即 Top-8 routing |
| `moe_intermediate_size` | `768` | 每个 expert 内部 FFN 的中间层维度 |
| `decoder_sparse_step` | `1` | MoE 层出现频率，`1` 表示每层都可以是 MoE 层 |
| `mlp_only_layers` | `[]` | 指定哪些层使用普通 MLP 而不是 MoE；空列表表示没有手动排除 |
| `norm_topk_prob` | `true` | 对选中的 Top-K expert 权重重新归一化 |
| `output_router_logits` | `false` | forward 默认不额外输出 router logits |
| `router_aux_loss_coef` | `0.001` | 训练时 router 辅助 loss 系数，用于 expert 负载均衡 |

另外几个字段不是 MoE 专属，但会影响 MoE 层形状:

| 参数 | 当前值 | 对 MoE 的影响 |
| --- | ---: | --- |
| `hidden_size` | `2048` | router 输入维度、expert 输入/输出维度 |
| `hidden_act` | `silu` | expert 内部激活函数 |
| `intermediate_size` | `6144` | 普通 dense MLP 的中间维度；如果某些层被设为 `mlp_only_layers` 会用到 |
| `num_hidden_layers` | `48` | 决定总共有多少个 decoder layer |

## 4. 每层是否使用 MoE

Qwen3 MoE decoder layer 的选择逻辑可以概括为:

```python
if (
    layer_idx not in config.mlp_only_layers
    and config.num_experts > 0
    and (layer_idx + 1) % config.decoder_sparse_step == 0
):
    mlp = Qwen3MoeSparseMoeBlock(config)
else:
    mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
```

对于当前配置:

```text
decoder_sparse_step = 1
mlp_only_layers = []
num_experts = 128
```

因此 `48` 个 decoder layer 都会使用 `Qwen3MoeSparseMoeBlock`，不会落到普通 dense MLP 分支。

## 5. Router/Gate 的行为

Router 负责为每个 token 选择 expert。Qwen3 MoE 中 router 的关键形状是:

```text
router weight: [num_experts, hidden_size] = [128, 2048]
router logits: [num_tokens, num_experts] = [num_tokens, 128]
selected experts: [num_tokens, num_experts_per_tok] = [num_tokens, 8]
routing weights: [num_tokens, 8]
```

计算过程:

```text
1. 对 hidden_states 做线性映射，得到每个 token 对 128 个 expert 的 logits
2. 对 logits 做 softmax，得到 expert 概率
3. 取 Top-8 expert
4. 如果 norm_topk_prob=true，则把这 8 个权重重新归一化
5. 将 token 分发给对应 expert，并按 routing weights 加权累加输出
```

当前配置的 `output_router_logits=false` 表示普通推理时不返回 router logits；训练或诊断 expert 分布时才可能需要打开。

## 6. Expert 内部形状

每个 expert 是一个 gated MLP，结构类似:

```text
gate = Linear(hidden_size -> moe_intermediate_size)
up   = Linear(hidden_size -> moe_intermediate_size)
act  = silu(gate) * up
down = Linear(moe_intermediate_size -> hidden_size)
```

在 Qwen3 MoE 实现中，`gate` 和 `up` 通常被打包为一个 fused `gate_up_proj`:

```text
gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
            = [128, 1536, 2048]

down_proj:    [num_experts, hidden_size, moe_intermediate_size]
            = [128, 2048, 768]
```

单个 expert 的 FFN 参数量约为:

```text
gate/up: 2 * 768 * 2048
down:    2048 * 768
total:   4,718,592
```

单个 MoE 层包含 `128` 个 expert，因此 expert 权重本身约为:

```text
4,718,592 * 128 = 603,979,776
```

但单个 token 每层只激活 `8` 个 expert，因此实际参与该 token 计算的 expert 参数约为:

```text
4,718,592 * 8 = 37,748,736
```

这就是 MoE 的稀疏激活来源。

## 7. Dense 与 MoE 的区别

Dense FFN:

```text
每个 token -> 同一组 MLP 参数 -> 全量计算
```

MoE FFN:

```text
每个 token -> router 选择 Top-K experts -> 只计算被选中的 experts -> 加权合并
```

对推理系统来说，MoE 和 dense 的差别主要体现在:

- 权重加载仍然需要加载所有 experts，显存/内存更接近总参数量。
- 单 token 计算只走 Top-K experts，计算量更接近激活参数量。
- batch 内不同 token 可能命中不同 experts，需要高效 dispatch、grouped GEMM 或 expert parallel。
- router/gate 通常不适合随意量化，否则 expert 选择会变得不稳定。

## 8. 接入当前项目时的关注点

当前仓库主要已有 dense Qwen3/Qwen3.5 路径。若要支持 `qwen3_moe`，需要额外处理:

- `model_type="qwen3_moe"` 的 registry 分发。
- `Qwen3MoeSparseMoeBlock`、router、experts 的模块实现。
- expert 权重加载，特别是 `mlp.experts.gate_up_proj` 和 `mlp.experts.down_proj`。
- Top-K routing 和 token 到 expert 的 dispatch。
- TP/EP 下 expert 权重的切分与聚合。
- FP8/AWQ/GPTQ 等量化路径中对 expert 权重、router gate 的排除或特殊处理。

对 Qwen3 MoE FP8 checkpoint，量化配置里常见的 `modules_to_not_convert` 会排除 `mlp.gate`，这和上面提到的 router 稳定性有关。

## 9. 小结

`Qwen3-30B-A3B` 是一个典型的稀疏 MoE 模型:

- `48` 层 decoder。
- 每层 `128` 个 expert。
- 每个 token 每层只激活 `8` 个 expert。
- 每个 expert 的中间维度是 `768`。
- `decoder_sparse_step=1` 且 `mlp_only_layers=[]`，因此所有 decoder layer 都走 MoE 分支。
- MoE 带来的是更大的总参数容量和较低的单 token 计算量，但推理实现会比 dense 模型复杂，尤其是 expert dispatch、权重加载、并行切分和量化策略。
