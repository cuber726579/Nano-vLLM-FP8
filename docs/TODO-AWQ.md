# TODO-AWQ

Nano-vLLM-FP8 AWQ 量化推理功能待办清单，按落地顺序排序。

---

## 目标范围

- **首批目标模型**: Qwen3 官方 dense AWQ checkpoint，例如 `Qwen/Qwen3-4B-AWQ`、`Qwen/Qwen3-8B-AWQ`、`Qwen/Qwen3-14B-AWQ`、`Qwen/Qwen3-32B-AWQ`
- **首批目标配置**:
  - `quant_method="awq"`
  - `bits=4`
  - `group_size=128`
  - `version="gemm"`
  - `zero_point=true`
- **首批不做**:
  - GPTQ / GGUF / BitsAndBytes
  - AWQ `version="gemv"`、ExLlama、Marlin 等其它执行后端
  - AWQ MoE 专用优化

## 已完成

- `QuantConfig.from_hf_config()` 已支持 `quant_method="awq"`，并解析 `bits` / `w_bit`、`group_size` / `q_group_size`、`zero_point`、`version` 和 `extra` 字段。
- AWQ 配置已复用 `modules_to_not_convert`、`ignored_layers`、`excluded_modules` 到统一的 `excluded_modules` 排除逻辑。
- `AwqLinearMethod` 已接入 `QuantConfig.get_quant_method()` 和 `build_linear_method()`，被排除层继续走 `UnquantizedLinearMethod`。
- `nanovllm/quantization/awq.py` 已提供 AWQ int4 pack/unpack、GEMM/GEMV dequant fallback 和 `dequantize + F.linear` 参考执行路径。
- `AwqLinearMethod.create_weights()` 已注册 `qweight`、`qzeros`、`scales` 和可选 `bias` 参数，并标记量化权重为 `requires_grad=False`。
- AWQ loader 已为 `qweight` / `qzeros` / `scales` 提供基础加载逻辑，并覆盖 replicated / column / row / merged / qkv 的基础切分规则。
- 已支持 `version="gemv"` 的参数形状、loader 切片和 PyTorch dequant fallback。
- 已兼容常见 AutoAWQ/HF 字段别名与提示字段，包括 `w_bit`、`q_group_size`、`backend`、`do_fuse`、`modules_to_fuse`、`exllama_config`。
- 已兼容 `modules_to_fuse` 配置字段，但不执行 fused module 替换。
- 已新增 Llama / Mistral dense decoder 入口，可复用 Llama-like 的 Qwen2 block 实现和 AWQ 线性层。
- 已添加 GEMM / GEMV synthetic dequant 测试，以及 Qwen3 packed QKV loader 覆盖。

---

## 高优先级

### 1. 扩展 `QuantConfig` 支持 AWQ 字段

- **状态**: 已完成，见“已完成”。

### 2. 新增 AWQ 线性方法入口

- **状态**: 已完成，见“已完成”。

### 3. 注册 AWQ checkpoint 权重参数

- **状态**: 已实现基础路径。
- **现状**: `AwqLinearMethod.create_weights()` 已注册 AWQ packed 权重参数，并为常见 linear / packed linear 提供 loader。
- **AWQ 常见参数**:
  - `qweight`
  - `qzeros`
  - `scales`
  - 可选 `bias`
- **改动点**:
  - `AwqLinearMethod.create_weights()` 注册 packed int32 权重和 scale 参数。
  - 明确每个参数的形状、dtype 和 `requires_grad=False`。
  - 为 `ColumnParallelLinear`、`MergedColumnParallelLinear`、`QKVParallelLinear` 补齐 AWQ scale / zero / qweight 的 shard loader。
  - 对 packed QKV 和 gate/up 投影确认 HF checkpoint 命名能正确映射到本项目 `packed_modules_mapping`。
  - **已覆盖**: Qwen3 packed QKV 的 `qweight` / `qzeros` / `scales` loader 已有真实 checkpoint 测试。

### 4. 先实现可验证的 dequant fallback

- **状态**: 已完成，见“已完成”。

### 5. 增加 CUDA/Triton AWQ GEMM 路径

- **现状**: dequant fallback 正确但性能和显存峰值都不理想。
- **改动点**:
  - 实现 W4A16 GEMM Triton kernel，输入保持 FP16/BF16，权重按 int4 packed 读取。
  - kernel 内按 `group_size` 加载 `scales` / `qzeros` 并反量化后累加。
  - 输出 dtype 跟随输入 dtype。
  - 对不满足 kernel 条件的形状回退到 reference 路径。
- **验收标准**:
  - 单层 linear 输出与 reference 路径误差稳定。
  - decode 阶段相比 dequant fallback 有明显速度提升。

### 6. 处理 Tensor Parallel 分片

- **状态**: 部分实现。
- **风险点**:
  - AWQ packed 权重通常沿输出维打包，TP 分片必须与 pack factor 对齐。
  - `qzeros` 和 `scales` 的分片维度不一定与 `qweight` 完全相同。
- **改动点**:
  - 明确 `qweight`、`qzeros`、`scales` 在 replicated / column / row / merged / qkv linear 中的 shard 规则。
  - 对无法整除的分片给出带层名和形状的清晰错误。
  - 覆盖 TP=1 的基础路径，再扩展 TP>1。
  - **已覆盖**: AWQ loader 已实现 replicated / column / row / merged / qkv 的基础切分规则，并对 pack factor、group shard 对齐和 shape mismatch 给出错误。
  - **待补充**: TP>1 的真实模型端到端验证和更多 packed merged projection 场景。

### 7. 真实模型端到端验证

- **状态**: 部分实现。
- **建议模型**:
  - `Qwen/Qwen3-4B-AWQ` 作为首个真实 checkpoint
  - `Qwen/Qwen3-8B-AWQ` 用于覆盖额外配置字段
- **验证项**:
  - `LLM(..., quantization="awq")` 能通过配置校验。
  - safetensors 权重全部加载，无未知参数、缺失参数或 shape mismatch。
  - prompt 生成可以稳定输出非空文本。
  - 与 FP16/BF16 或 Transformers/AutoAWQ 参考输出做短 prompt 粗略对齐。
  - **已覆盖**: 已添加 `Qwen/Qwen3-4B-AWQ` 的配置解析、QKV 权重加载和 CUDA 生成冒烟测试。
  - **待补充**: `Qwen/Qwen3-8B-AWQ`、batch prompts、以及与 Transformers/AutoAWQ 输出粗略对齐。

---

## 中优先级

### 8. 更好的错误信息

- **状态**: 部分实现。
- 在 `AwqLinearMethod.create_weights()` 和 loader 中加入层名、参数名、期望 shape、实际 shape。
- 对不支持的 AWQ 变体给出明确报错，例如:
  - `bits != 4`
  - `zero_point is False`
  - `version != "gemm"`
  - `group_size` 不是 kernel 支持值
  - **已覆盖**: AWQ 参数加载、packed shard、TP 维度、shape mismatch、pack factor 和不支持变体已有明确报错。
  - **待补充**: Triton kernel 接入后补充 kernel 条件和 fallback 原因。

### 9. 性能基准

- 在 `bench.py` 中增加 AWQ 场景。
- 对比:
  - BF16 原始权重
  - AWQ dequant fallback
  - AWQ Triton GEMM
  - 当前 FP8 路径
- 分别记录 prefill、decode、显存峰值和 tokens/s。

### 10. 缓存反量化权重的调试模式

- **用途**: 方便定位 kernel 数值误差。
- **方案**:
  - 提供仅调试使用的 eager dequant cache。
  - 第一次 forward 时反量化并缓存浮点权重，后续直接 `F.linear`。
  - 默认关闭，避免显存占用违背 AWQ 初衷。

## 低优先级

### 11. 更多 AWQ 生态兼容

- 研究 AutoAWQ、llm-awq、vLLM、Transformers 不同 checkpoint 的字段和权重布局差异。
- 评估是否需要真正执行 `modules_to_fuse` 对应的 fused module 替换。

### 12. 非 Qwen 模型架构

- 当前项目主要模型实现集中在 Qwen2 / Qwen3 / Qwen3.5。
- AWQ 线性层可复用，DeepSeek 等专用 MLA/MoE 架构仍需要对应 `models/*.py` 和 `model_runner` 分发。

---

## 测试计划

### 单元测试

- `QuantConfig.from_hf_config()`:
  - 接受标准 Qwen3 AWQ 配置。
  - 拒绝不支持的 bits / version / zero_point。
  - 正确合并 `modules_to_not_convert` / `ignored_layers` / `excluded_modules`。
- `dequantize_awq()`:
  - 覆盖固定小矩阵、不同 group 边界、带 zero point 的反量化。
  - 与纯 PyTorch unpack 参考实现对齐。
  - **状态**: 已覆盖 GEMM / GEMV 小矩阵反量化。
- `AwqLinearMethod.apply()`:
  - reference 路径与浮点线性层输出对齐。
  - packed QKV / gate_up 分片 scale loader 正确。
  - **状态**: 已覆盖 Qwen3 packed QKV loader；`apply()` 与浮点线性层对齐、gate/up 分片仍待补充。

### 集成测试

- 使用本地或缓存的 `Qwen/Qwen3-4B-AWQ`:
  - 模型初始化
  - 权重加载
  - 单 prompt prefill + decode
  - batch prompts
  - **状态**: 已覆盖配置解析、QKV loader 和单 prompt 生成冒烟；但当前生成结果可能退化为重复符号，例如 `Completion: '!!!!!!!!'`，说明数值路径仍需和参考实现对齐；batch prompts 待补充。
- 使用 `Qwen/Qwen3-8B-AWQ`:
  - 覆盖 `backend="autoawq"`、`do_fuse=false` 等额外字段被安全忽略。
  - **状态**: 配置额外字段已通过 synthetic config 覆盖；真实 8B checkpoint 待补充。

## 已知风险

- AWQ packed 权重布局必须以真实 safetensors 为准，不能只按配置字段推断。
- TP 分片和 packed int4 的 pack factor 对齐最容易引入静默数值错误。
- 当前 `Qwen/Qwen3-4B-AWQ` 端到端生成可以跑完，但曾出现输出全是重复感叹号的退化结果；这通常意味着某处权重反量化、zero point、scale 维度、QKV 拼接或 dtype 路径仍有数值偏差，不能只用“非空输出”作为通过标准。
- Triton kernel 的数值路径要同时覆盖 FP16 和 BF16 输入。
- 首批 Qwen3 AWQ 是 dense 模型，后续 MoE AWQ 可能需要额外处理 expert 权重和 router 排除策略。
