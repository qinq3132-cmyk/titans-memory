# Neural Memory 量化加噪方案

## 一、Neural Memory 完整工作过程

### 整体定位

`NeuralMemory` 是 MAC Transformer 中的长期记忆模块。它本质上是一个**权重随输入序列动态演化的 MLP**（`MemoryMLP`），每处理一段序列后其权重就被"更新"一次，从而将历史信息编码进权重。

每次 `NeuralMemory.forward(seq)` 调用内部依次完成两个完全独立的阶段：

---

### 阶段 1 — Store（写入记忆 / 权重更新）

```
输入序列 seq
    │
    ├─ store_norm(seq)
    ├─ to_keys(seq)    → keys
    ├─ to_values(seq)  → values
    │
    ├─ 自适应超参推导：
    │      adaptive_lr     = to_adaptive_step(seq)
    │      decay_factor    = to_decay_factor(chunked_seq)
    │      adaptive_momtum = to_momentum(chunked_seq)
    │
    │  对每个 chunk（c 个 token 为一组）:
    │      loss_t = MSE( functional_call(W_t, keys_t), values_t )
    │      grad_t = ∂loss_t/∂W_t × adaptive_lr   ← vmap per-sample 梯度
    │      surprise_t = −grad_t
    │
    │  surprise → 动量累积（AssocScan）→ 谱归一化 → 权重衰减（AssocScan）
    │                                         ↓
    │                               updates: W_0, W_1, ..., W_T
    │
    └─ 返回 updates（运行时张量，存入 NeuralMemState）
```

**关键点**：
- `updates` 是随时间演化的权重快照序列，**不是** `nn.Parameter`，是每次 forward 重新计算的运行时张量。
- 梯度计算通过 `torch.func.vmap(grad(...))` 完成，与 Retrieve 路径天然解耦。

---

### 阶段 2 — Retrieve（读取记忆 / 推理）

```
输入 retrieve_seq  +  updates（W_0...W_T，来自 Store 阶段）
    │
    ├─ retrieve_norm(seq)
    ├─ to_queries(seq) → queries
    ├─ q_norm(queries)
    │
    │  reshape:
    │      weights: [b, n_chunks, *w_shape] → [(b·n_chunks), *w_shape]
    │      queries: [b h (n·c) d]           → [(b·h·n), c, d]
    │
    │  核心读取：
    │      values = functional_call(memory_model, W_t, queries_t)
    │             = M_{W_t}(q_t)    ← 用当前 chunk 的权重快照做前向
    │
    ├─ multihead_rmsnorm → retrieve_gate → merge_heads → combine_heads
    │
    └─ 返回 retrieved [b, n, dim]  → 拼入注意力上下文
```

---

## 二、量化加噪方案

### 动机

模拟神经记忆存储介质（类比 memristor / SRAM）的低精度读取特性：
- **写入（Store）** 用全精度浮点，权重更新正常进行
- **读取（Retrieve）** 对动态权重施加量化误差 + 高斯噪声，模拟实际硬件的读取失真

### 介入位置

`retrieve_memories()` 内，`functional_call` **之前**，对 `weights` 字典中每个张量做量化+加噪：

```
Store 阶段（完全不变）:
    seq → keys/values → per_sample_grad_fn → assoc_scan
    → updates W_t（float32，干净）  ✓ 梯度正常

Retrieve 阶段（加噪路径）:
    W_t (float32)
        │
        ▼  ← 介入点（retrieve_memories 入口）
    W_q  = data_quantization(W_t, bit=N)      # 均匀量化
    W_q  = add_noise(W_q, n_scale=σ)          # 叠加高斯噪声（可选）
        │
        ▼
    functional_call(memory_model, W_q, queries)
        │
        ▼
    retrieved value（带噪输出）→ 后续注意力层
```

### 梯度解耦说明

Store 和 Retrieve 在同一个 `forward()` 中顺序执行，但**梯度完全解耦**：
- Store 梯度：通过 `torch.func.grad` 对 `keys/values` 求，不经过 Retrieve 路径
- Retrieve 梯度：经过量化算子，若需要可用 STE（`(w_q - w).detach() + w`）保留梯度通路
- 实验场景（仅推理加噪）：`model.eval()` 时激活量化，`model.train()` 时透传，训练完全不受影响

---

## 三、实现细节

### 新增文件：`ref-project/titans_pytorch/quant_noise_mem.py`

封装对动态权重字典的量化+加噪操作，直接复用 `quantization_noise/quant_util.py`。

### 修改文件：`ref-project/titans_pytorch/neural_memory.py`

1. `NeuralMemory.__init__` 新增参数：
   ```python
   quant_noise_cfg: dict | None = None
   # 默认值：bit=4, noise_scale=0.05, noise_method='add', noise_range='max'
   ```

2. `retrieve_memories()` 中 `functional_call` 前插入：
   ```python
   if self.quant_noise_enabled and not self.training:
       weights_for_retrieve = quantize_weight_dict(dict(weights), **self.quant_noise_params)
   else:
       weights_for_retrieve = dict(weights)
   values = functional_call(self.memory_model, weights_for_retrieve, queries)
   ```

### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bit` | `4` | 均匀量化位数 |
| `noise_scale` | `0.05` | 高斯噪声强度（相对于权重最大值） |
| `noise_method` | `'add'` | 噪声叠加方式（加法） |
| `noise_range` | `'max'` | 噪声幅度基准（权重最大值） |

### 使用方式（train_mac_v3.py）

```python
neural_memory_kwargs=dict(
    ...
    quant_noise_cfg=dict(
        enabled=True,
        bit=4,
        noise_scale=0.05,
        noise_method='add',
        noise_range='max',
    ),
)
```

评估时直接 `model.eval()`，量化噪声自动激活，无需修改评估脚本。

---

## 四、涉及文件

| 文件 | 操作 |
|------|------|
| `ref-project/titans_pytorch/quant_noise_mem.py` | **新建**，量化+加噪函数 |
| `ref-project/titans_pytorch/neural_memory.py` | **修改** `__init__` + `retrieve_memories` |
| `train_mac_v3.py` | **修改** `neural_memory_kwargs` 传入 `quant_noise_cfg` |
| `train_mac_multigpu_v2.py` | 同上（可选，v2 主要用于对比基线） |
| `titans_pytorch/neural_memory.py`（current 版本） | **不改** |
