# Prefix Cache 与 Chunked Prefill

本文总结 `prefix cache` 在 `chunked prefill` 场景下的处理方式，以及相关改动为什么需要同时涉及 `BlockManager`、`Scheduler`、`ModelRunner` 和 `Sequence`。

## 背景

`prefix cache` 的目标是复用已经计算过的完整 prompt block，避免多个请求共享相同前缀时重复填充 KV cache。

`chunked prefill` 的目标是在 prompt 很长、单轮 token budget 不够时，把 prefill 拆成多轮调度。每一轮只计算一段 prompt token，直到整个 prompt 的 KV cache 都准备好，再进入 decode。

这两个机制叠加时，需要回答几个问题：

- 一个新序列开始 prefill 前，能复用多少个已有 prefix block？
- 当前 KV block 余量是否足够分配未缓存的部分？
- 本轮 chunked prefill 只计算了 prompt 的一部分时，哪些 block 可以写入 prefix cache？
- prefill 还没完成时，是否应该 append 模型输出 token 并进入 decode？

## 核心思路

新的处理方式把 prefix cache 拆成三个阶段：

1. 调度前先检查 cache 命中数量。
2. 分配时先复用命中的 cached block，再分配新的 KV block。
3. 本轮 prefill 完成后，把新完成的完整 block 写回 prefix cache。

这样可以保证 chunked prefill 每一轮只计算尚未缓存、且本轮 budget 允许处理的 token。

## BlockManager

`BlockManager.can_allocate(seq)` 不再只是返回布尔值，而是返回 cache 命中的完整 block 数：

```text
-1  表示 KV block 不足，当前不能调度
>=0 表示可以复用的 prefix cache block 数量
```

开启 prefix cache 时，它会按 block 顺序计算 rolling hash，并在 `hash_to_block_id` 中查找对应 block。只有 hash 命中且 `token_ids` 完全一致，才认为该 block 可以复用。

`BlockManager.allocate(seq, num_cached_blocks)` 根据命中的 block 数进行分配：

- 前 `num_cached_blocks` 个 block 从 prefix cache 复用。
- 如果 cached block 当前已经在使用中，则增加 `ref_count`。
- 如果 cached block 在 free list 中，则移入 used set。
- 未命中的后续 block 重新从 free list 分配。

分配完成后：

```python
seq.num_cached_tokens = num_cached_blocks * self.block_size
```

这表示这些 token 的 KV cache 已经可用，后续 prefill 不需要重新计算。

另一个关键变化是新增 `BlockManager.hash_blocks(seq)`。它在一轮 prefill 完成后，根据本轮调度覆盖的完整 block 范围更新 hash：

```text
start = seq.num_cached_tokens // block_size
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // block_size
```

只有完整完成的 block 才写入 prefix cache，避免把半个 block 的 token 错误缓存。

## Scheduler

调度等待队列中的新序列时，`Scheduler` 会先调用：

```python
num_cached_blocks = self.block_manager.can_allocate(seq)
```

如果返回 `-1`，说明 KV block 不够，当前轮次不能调度该序列。

如果可以调度，则本轮需要 prefill 的 token 数不再是完整 prompt 长度，而是：

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

这使得 prompt 前缀已经命中的部分不会重新进入模型计算。

对 chunked prefill，scheduler 只调度当前 token budget 能容纳的部分：

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

prefill 完成的判断也从“本轮是否处理完 `num_tokens`”改成“已缓存 token 加本轮 scheduled token 是否到达 prompt token 总数”：

```python
seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
```

在 `postprocess()` 中，scheduler 会先把本轮完成的完整 block 写入 prefix cache，再推进 `num_cached_tokens`：

```python
self.block_manager.hash_blocks(seq)
seq.num_cached_tokens += seq.num_scheduled_tokens
```

如果当前仍处于 prefill 且 prompt 还没处理完，则不会 append 模型输出 token：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

这避免了 chunked prefill 的中间轮次提前进入 decode。

## ModelRunner

`ModelRunner` 构造模型输入时，从 `seq.num_cached_tokens` 开始取本轮需要计算的 token：

```python
start = seq.num_cached_tokens
end = start + seq.num_scheduled_tokens
```

`seqlen_k` 也对应当前已经参与 attention 的结束位置：

```python
seqlen_k = end
```

这让模型看到的 key/value 长度与当前 prefill 进度一致，而不是直接使用完整序列长度。

## Sequence

`Sequence` 增加 `is_prefill` 状态，用来明确当前序列是否仍处于 prefill 阶段。

初始值为：

```python
self.is_prefill = True
```

当 prefill 完成并 append 第一个生成 token 后，才切换为 decode 状态：

```python
seq.is_prefill = False
```

这个状态也会参与序列化。处于 prefill 时，需要保留完整 token 列表；进入 decode 后，只需要传递 `last_token`：

```python
last_state = self.token_ids if self.is_prefill else self.last_token
```

这避免了仅根据 `num_cached_tokens` 或 `num_completion_tokens` 推断状态时，在 chunked prefill 中间轮次出现歧义。

## 整体效果

这组改动使 prefix cache 在 chunked prefill 下具备更明确的生命周期：

- 调度前发现可复用 prefix block。
- 分配时复用已有 block，只为未命中部分申请新 block。
- 每轮 prefill 只计算尚未缓存的 prompt token。
- 只有完整完成的 block 会写入 prefix cache。
- prefill 未完成时不会 append 输出 token，也不会进入 decode。
- 被 preempt 的序列重新回到 waiting 队列时恢复 prefill 状态。

最终效果是：共享前缀可以被正确复用，长 prompt 可以分块 prefill，并且两者同时启用时不会重复计算已缓存前缀，也不会把未完成的 prefill 当成 decode 处理。
