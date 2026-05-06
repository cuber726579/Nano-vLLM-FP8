# Troubleshooting

## `Sampler.forward` 在 `torch.compile` 下触发 Inductor 报错

### 报错现象

运行 `python example.py` 时，生成阶段卡在 `Generating: 0%` 附近，并在
`nanovllm/layers/sampler.py` 的 `Sampler.forward` 中报错。典型栈如下：

```text
torch._inductor.exc.InductorError: TypeError: list indices must be integers or slices, not NoneType
```

报错前通常还能看到 Inductor 对大词表 scan/reduction 的 codegen 信息：

```text
Error in codegen for ComputedBuffer(...)
SplitScan(... scan_ranges=[151936] ...)
```

这里的 `151936` 是 Qwen 系模型的词表大小，说明错误发生在 sampler 对 vocab
维度做过滤和累计概率计算时。

### 触发条件

这个问题通常出现在 `Sampler.forward` 同时满足以下条件时：

- 使用 `@torch.compile`
- 对 logits 做 `torch.sort(..., dim=-1)`
- 为 top-p sampling 计算 `probs.cumsum(dim=-1)`
- vocab size 很大，例如 Qwen 的 `151936`
- 同时存在动态 batch、`top_p` / `top_k` / `min_p` 等运行时采样参数

相关代码路径：

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
probs = torch.softmax(sorted_logits, dim=-1)
top_p_mask = probs.cumsum(dim=-1) > top_ps.unsqueeze(1)
```

### 原因

这不是采样逻辑本身的确定性错误，而是 `torch.compile` 触发了 PyTorch
Inductor/Triton 的编译器边界问题。

不加 `torch.compile` 时，这些算子会走 PyTorch eager CUDA kernel：

```text
torch.sort -> torch.softmax -> torch.cumsum -> masked_fill
```

这些 kernel 是独立执行的，通常可以正常运行。

加上 `@torch.compile` 后，TorchDynamo 会捕获 `Sampler.forward`，Inductor 会尝试
优化并生成 Triton kernel。`top_p` 所需的 `cumsum(dim=-1)` 属于 scan 操作，在大词表
维度上会被 Inductor 表示为 `SplitScan`。当前组合下，Inductor 的 Triton codegen
可能拿不到正确的 tensor dimension，最终触发：

```text
TypeError: list indices must be integers or slices, not NoneType
```

旧的稳定 sampler 没有这个问题，是因为它只做 temperature、softmax 和 exponential
sampling，没有 `sort + cumsum + top_p mask` 这条复杂路径。

### 解决方案

不要编译带 top-p/top-k/min-p 过滤逻辑的 `Sampler.forward`。保留模型主体的优化路径，
但让 sampler eager 执行。

修改前：

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(...):
        ...
```

修改后：

```python
class Sampler(nn.Module):
    def forward(...):
        ...
```

这样可以绕开 Inductor 对 `cumsum` scan 的 codegen bug。采样阶段只处理最终 logits，
相比模型主体计算量较小，去掉 `torch.compile` 的性能影响通常比生成阶段直接崩溃更可控。

如果确实想在 sampler 里保留少量编译优化，只编译纯 tensor、shape 相对稳定、无 Python
分支的小 helper，例如 temperature scaling 或最终的 sampling helper；不要编译包含
`sort + cumsum + top_p mask` 的完整 `Sampler.forward`，并给这些 helper 保留 eager
fallback。

### 验证方式

修复后重新运行：

```bash
python example.py
```

如果不再出现下面的 Inductor 报错，即说明该问题已绕开：

```text
torch._inductor.exc.InductorError
SplitScan(... scan_ranges=[151936] ...)
```

也可以先用小张量验证 sampler eager 路径是否能正常返回 token：

```bash
python -c 'import torch; from nanovllm.layers.sampler import Sampler; s=Sampler(); logits=torch.randn(2, 16); temperatures=torch.ones(2); top_ps=torch.tensor([1.0,0.9]); top_ks=torch.tensor([-1,5], dtype=torch.int32); min_ps=torch.tensor([0.0,0.05]); print(s(logits, temperatures, top_ps, top_ks, min_ps))'
```
