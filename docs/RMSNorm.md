# RMSNorm -- 以Qwen3为例

本文以 `nanovllm/models/qwen3.py` 为准，整理 Qwen3 dense 路径中 `RMSNorm` 的出现位置，并按 forward 时是否携带 `residual` 分类。

## 总览

Qwen3 中共有三类 RMSNorm:

| 模块 | 成员 | 位置 | 输入形态 | 是否带 residual |
| --- | --- | --- | --- | --- |
| `Qwen3Attention` | `q_norm` | attention 内部 query norm | `[num_tokens, num_heads, head_dim]` | 否 |
| `Qwen3Attention` | `k_norm` | attention 内部 key norm | `[num_tokens, num_kv_heads, head_dim]` | 否 |
| `Qwen3DecoderLayer` | `input_layernorm` | 每层 attention 前 | `[num_tokens, hidden_size]` | 首层首次调用否，之后通常是 |
| `Qwen3DecoderLayer` | `post_attention_layernorm` | 每层 MLP 前 | `[num_tokens, hidden_size]` | 是 |
| `Qwen3Model` | `norm` | 所有 decoder layers 后的 final norm | `[num_tokens, hidden_size]` | 是 |

## 无 residual 的 RMSNorm

### 1. Attention 内部 q/k norm

`Qwen3Attention` 在 `qkv_bias == False` 时创建 query/key 专用 RMSNorm:

```python
self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

forward 中先将 `q`、`k` reshape 成三维张量，再分别归一化:

```python
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
q = self.q_norm(q)
k = self.k_norm(k)
```

这两处没有传入 `residual`，因此会走 `RMSNorm.forward(..., residual=None)`，也就是 `rms_forward(x)`。它们的最后一维是 `head_dim`，不是 `hidden_size`。

### 2. 每层首次 input_layernorm

`Qwen3Model.forward` 初始化:

```python
residual = None
```

因此第一个 decoder layer 第一次进入 `input_layernorm` 时会走无 residual 分支:

```python
hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
```

这一路输入是二维 hidden states:

```text
[num_tokens, hidden_size]
```

它同样会调用 `rms_forward(x)`，但输入 rank 和 q/k norm 不同。

## 有 residual 的 RMSNorm

### 1. 后续层的 input_layernorm

从第一个 decoder layer 返回后，`residual` 不再是 `None`。后续 decoder layer 的 `input_layernorm` 会走带 residual 分支:

```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

这会调用 `add_rms_forward(x, residual)`，输入通常是二维:

```text
[num_tokens, hidden_size]
```

### 2. 每层 post_attention_layernorm

attention 输出后，`post_attention_layernorm` 总是携带 residual:

```python
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
```

这一路也走 `add_rms_forward(x, residual)`，用于 attention output 与上一轮 residual 的 fused add + RMSNorm。

### 3. Final norm

所有 decoder layers 结束后，`Qwen3Model` 的最终 norm 也携带 residual:

```python
hidden_states, _ = self.norm(hidden_states, residual)
```

因此 final norm 走 `add_rms_forward(x, residual)`。

## 对 torch.compile recompile 的影响

`rms_forward(x)` 同时服务两种不同形态:

- q/k norm: 三维 `[num_tokens, num_heads_or_kv_heads, head_dim]`
- 首层 input norm: 二维 `[num_tokens, hidden_size]`

所以如果单独 `torch.compile` `rms_forward`，同一个 Python 函数会同时遇到 2D 和 3D 输入，容易触发 rank guard failure，例如:

```text
tensor 'x' rank mismatch. expected 2, actual 3
tensor 'x' rank mismatch. expected 3, actual 2
```

相比之下，`add_rms_forward(x, residual)` 基本只处理 hidden states 的 residual path，输入主要稳定在二维 `[num_tokens, hidden_size]`。它仍可能因为 prefill/decode 的 `num_tokens` 变化而 recompile，但不会像 `rms_forward` 一样频繁遇到 2D/3D rank 切换。
