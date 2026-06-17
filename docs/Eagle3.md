# Eagle3 适配方案

## Summary

- 新增 `docs/EAGLE3.md`，作为当前项目接入 EAGLE-3 speculative decoding 的技术方案文档。
- v1 目标: 支持 Qwen3 dense verifier + `RedHatAI/Qwen3-8B-speculator.eagle3` 这类 Eagle3 speculator，先保证 greedy 输出与普通 decode 完全一致。
- v1 不引入 `speculators` 运行时依赖，按公开 config/权重格式在项目内实现最小 Eagle3 runtime；采样分布校正、TP>1、持久 draft KV cache 放到后续阶段。

## 官方 vLLM 用法

官方 vLLM 现在支持两种常见入口，和本项目的实现思路可以对照着看。

1. 直接 serve speculator 模型，vLLM 会根据模型里的 `speculators_config` 自动把 verifier 和 speculator 组起来:

```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

2. 显式指定 target model + speculative config:

```bash
vllm serve Qwen/Qwen3-8B \
  -tp 1 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "method": "eagle3"
  }'
```

官方文档也说明了同样的配置键可通过 Python `LLM(..., speculative_config={...})` 传入。这里的 `method` 取值就是 `eagle3`，`num_speculative_tokens` 控制一次草拟多少个 token。

## Key Changes

- 公共配置接口:
  - `LLM(..., speculative_model=..., speculative_method="eagle3", num_speculative_tokens=3, eagle3_target_layer_ids=None)`。
  - `SamplingParams.temperature=0.0` 表示 greedy；Eagle3 v1 仅启用 greedy，非 greedy 请求回退普通 decode。
  - 输出结构保持不变: `generate()` 仍返回 `{"text", "token_ids"}`。

- 模型与权重:
  - 新增 Eagle3 speculator 配置解析，读取 `draft_vocab_size`、`transformer_layer_config`、`norm_before_residual`、`target_hidden_size`、`d2t/t2d`。
  - 新增本地 `Eagle3Speculator`，实现 `fc + Eagle3DecoderLayer + norm + lm_head`，支持 draft vocab 到 target vocab 的 token 映射。
  - 扩展 loader 支持加载 buffer，例如 `d2t`、`t2d`；忽略 verifier-only 权重。

- 主模型与执行流:
  - Qwen3 forward 增加可选 auxiliary hidden capture，拼接 3 个 verifier 层 hidden states。
  - 默认 hidden layer ids: 若 speculator config 未提供，则用 `[2, num_hidden_layers // 2, num_hidden_layers - 3]`。
  - Decode 阶段流程: 先跑当前 `last_token` 得到 `logits0 + eagle_hidden`，Eagle3 draft K 个 token，再用主模型一次 varlen forward 验证 K 个 draft token。
  - Acceptance 使用 greedy longest-prefix: 接受连续匹配 draft token；首个不匹配时追加 target token；全部匹配时追加 bonus token。
  - Scheduler/Sequence/BlockManager 增加多 token append 与预留槽位能力；被拒绝 draft 的 KV 保留为 stale 数据，后续按真实 sequence length 覆盖，不参与 attention。

## Test Plan

- 单元测试:
  - Eagle3 config fixture 覆盖 RedHatAI config、默认 layer ids、draft/target vocab size、`d2t/t2d` 映射。
  - Eagle3 synthetic forward 覆盖 shape、greedy proposal、target vocab 映射中的 `-inf` 填充。
  - Qwen3 hidden capture 验证 3 层 hidden 拼接 shape，并确认未启用 Eagle3 时 logits 不变。
  - Acceptance 覆盖 0 accept、partial accept、all accept + bonus、EOS、`max_tokens` 截断。

- 集成测试:
  - 本地 `Qwen/Qwen3-8B` + `RedHatAI/Qwen3-8B-speculator.eagle3` greedy token ids 与普通 decode 完全一致。
  - `num_speculative_tokens=1/3/5` 均可运行，并记录 average accepted length。
  - 非 greedy sampling params 自动回退普通 decode。
  - 不支持场景给清晰错误: `tensor_parallel_size > 1`、非 Qwen3 verifier、缺失 `d2t/t2d`、speculator/verifier hidden size 不匹配。

## Assumptions

- v1 只做 Qwen3 dense verifier；Qwen2/Llama/Mistral 可后续复用同一接口扩展。
- v1 强制 `enforce_eager=True`，暂不接 CUDA graph。
- v1 使用 correctness-first 的 per-proposal draft attention，不实现跨 step 持久 draft KV cache。
- 参考来源: [Speculators Eagle3 docs](https://vllm-project.github.io/speculators/user_guide/algorithms/eagle3/)、[Speculators serve tutorial](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/serve_vllm/)、[vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)、[RedHatAI Qwen3-8B Eagle3 config](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3/blob/main/config.json)、[HF custom Eagle3 code](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3/blob/main/eagle3.py)、[vLLM Eagle proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py)。
