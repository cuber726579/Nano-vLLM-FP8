# TODO-FP8

Nano-vLLM-FP8 FP8 量化推理功能待办清单，按优先级排序。

---

## 已完成

### 1. `modules_to_not_convert` 支持

- **状态**: 已从 `quantization_config` 解析并归一到 `QuantConfig.excluded_modules`，Qwen3/Qwen3.5 线性层会携带 HF 模块名用于选择 FP8 或非量化方法。
- **已覆盖影响**: HF FP8 checkpoint 经常排除 lm_head 等敏感层不量化。这些层：
  - 维度可能不被 FP8 block size 整除，导致 `create_weights()` 报 `ValueError`
  - 即使维度对齐，safetensors 中存的是 float 权重而非 FP8+scale 对，加载会得到错误数据

### 2. `ignored_layers` / `excluded_modules` 支持

- **状态**: 已与 `modules_to_not_convert` 统一处理，维护一个排除集合，在创建 Linear 层时查询。

### 3. `e5m2` 格式支持

- **状态**: 已支持 `e4m3`/`e4m3fn` 和 `e5m2`，`Fp8LinearMethod` 会根据 `fmt` 选择对应 torch dtype，并把对应 FP8 最大值传给激活量化 kernel。

---

## 中优先级

### 4. `activation_scale_ub` 支持

- **现状**: `act_quant_kernel` 已按 FP8 格式使用默认最大值，但还不支持从配置中覆盖 activation scale 上限。
- **影响**: 无法配置激活 scale 上限，对有 outlier 的模型可能产生较大量化误差
- **改动点**:
  - `QuantConfig` — 新增 `activation_scale_ub: float | None` 字段
  - `Fp8LinearMethod` — 将值传给 `act_quant()`
  - `act_quant_kernel` — 用 kernel 参数替代硬编码常量

### 5. 静态激活量化 (`activation_scheme="static"`)

- **现状**: `Fp8LinearMethod.__init__` 中直接 `raise NotImplementedError`
- **影响**: 无法使用预校准的静态 activation scale，每次前向都要重新量化激活
- **改动点**:
  - 权重加载时从 checkpoint 读取预存 activation scale
  - `Fp8LinearMethod.apply()` — static 路径跳过 `act_quant()`，直接用预存 scale
  - 需要定义 activation scale 的存储格式（per-tensor / per-channel / per-block）

---

## 低优先级

### 6. FP8 KV Cache

- **现状**: KV Cache 使用模型原生 dtype（BF16/FP16），未做 FP8 量化
- **影响**: KV Cache 是推理时主要显存消费者之一，FP8 可节省约 50% KV Cache 显存
- **改动点**:
  - `kvcache.py` — 新增 FP8 存储格式
  - Attention 层 — k/v 写入 cache 前做 `act_quant()`，读取后做 scale 还原
  - 需注意精度损失对长序列质量的影响

### 7. FP8 Attention 计算

- **现状**: Attention 计算（FlashAttention）使用 BF16/FP16，只有 QKV/O 投影用了 FP8
- **影响**: Attention 本身是 compute-bound 操作的瓶颈，FP8 可加速
- **改动点**:
  - 需 Triton 实现的 FP8 FlashAttention kernel 或依赖外部库
  - 复杂度较高，收益主要在大 batch / 长序列场景

### 8. 非 Qwen 模型架构

- **现状**: 只有 `qwen3.py` 和 `qwen3_5.py` 两个模型文件，model_runner 只分发这两种 `model_type`
- **影响**: 无法运行 Llama、Mistral、DeepSeek 等主流模型的 FP8 checkpoint
- **改动点**:
  - 新增 `nanovllm/models/llama.py`、`nanovllm/models/deepseek.py` 等
  - model_runner 新增模型分发逻辑
  - 可参考 qwen3.py 的实现模式，复用现有 layers

---

## 已知问题

### 权重加载失败时错误信息不指明层名

- `Fp8LinearMethod.create_weights()` 中 `input_size % block_k != 0 or output_size % block_n != 0` 报 `ValueError` 但不包含层名
- 大型模型（几十上百层）难以定位具体是哪个层出错
- 修复: 在异常信息中加入 layer 标识，或由调用方（模型 `__init__`）捕获后补充上下文

### Embedding / LM Head 不支持 FP8 权重

- `VocabParallelEmbedding` 和 `ParallelLMHead` 不接受 `linear_method`，始终用浮点格式
- 如果 FP8 checkpoint 量化为 vocab embedding，safetensors 中存的是 FP8+scale，加载会失败或数据错乱

### Qwen3.5 linear attention 中 FP8 收益有限

- `torch_chunk_gated_delta_rule` 和 `torch_recurrent_gated_delta_rule` 内部强制转 `float32`
- FP8 只在输入投影（`in_proj_qkv` 等）节省显存带宽，矩阵计算部分已经扩展为 float32

### `get_model_dtype` 跨 transformers 版本脆弱

- 回退链 `dtype` → `torch_dtype` 在新版 transformers 中行为可能不一致
- 建议: 直接读取 `torch_dtype` 并做一次统一映射
