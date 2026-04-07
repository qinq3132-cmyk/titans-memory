# Titans MAC Transformer — 训练开发日志

## 硬件环境

- **节点**: 8× NVIDIA H100 80GB HBM3 (Driver 580.126.09, CUDA 13.0)
- **训练使用**: GPU 4, 5, 6, 7 (4卡)
- **conda 环境**: `titans` (Python 3.10)

---

## 快速命令参考

```bash
# 激活环境
conda activate titans

# 单卡训练 (GPU 7, ~60h)
python train_mac.py

# 多卡训练 (GPU 4,5,6,7, ~15h)
torchrun --nproc_per_node=4 train_mac_multigpu.py

# 监控 TensorBoard (另开终端)
tensorboard --logdir=runs/ --port=6006

# 查看 GPU 状态
nvidia-smi
```

---

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `train_mac.py` | 单卡训练脚本 (原始版本, GPU 7) |
| `train_mac_multigpu.py` | 4卡 DDP 训练脚本 (GPU 4,5,6,7) |
| `checkpoints/` | 训练 checkpoint 自动保存目录 |
| `runs/` | TensorBoard 日志目录 |

---

## 训练配置

| 参数 | 单卡 | 多卡 | 说明 |
|------|------|------|------|
| GPU | 1× H100 | 4× H100 | |
| per-GPU batch size | 4 | 4 | |
| gradient accumulation | 4 | 1 | 多卡下降为1保持等效batch一致 |
| **effective batch** | **16** | **16** | 完全一致 |
| learning rate | 2e-4 | 2e-4 | |
| total steps | 100k | 100k | |
| checkpoint 间隔 | — | 1000 步 | |
| 预估时间 | ~60h | ~15h | |

---

## 开发变更日志

### 2025-03-24 — 初始化多卡训练

**问题**: 单卡 H100 训练预估 60h，太慢。GPU 4,5,6,7 空闲可用。

**变更**:

1. **新建 `train_mac_multigpu.py`**
   - 基于 `train_mac.py` 改造为 PyTorch DDP (DistributedDataParallel)
   - `CUDA_VISIBLE_DEVICES="4,5,6,7"` 写入脚本内部，启动命令简化
   - `DistributedSampler` 替代原始 DataLoader，支持分布式数据分片
   - 日志/生成/验证仅在 rank 0 执行

2. **保持训练效果一致**
   - `GRADIENT_ACCUMULATE_EVERY` 从 4 → 1
   - 等效 global batch = 4 GPU × 4 bs × 1 accum = 16 (与单卡一致)
   - 学习率、优化器、模型结构不变

3. **新增 Checkpoint 保存 & 断点续训**
   - 每 1000 步自动保存到 `checkpoints/ckpt_step_XXXXXX.pt`
   - 训练结束保存 `_final.pt`
   - `RESUME_FROM_CHECKPOINT=True` 自动从最新 checkpoint 恢复
   - 保存内容: model_state_dict, optimizer_state_dict, step

4. **新增 TensorBoard 可视化**
   - `Loss/train` — 每步训练 loss
   - `Loss/val` — 每 100 步验证 loss
   - `BPC/train` & `BPC/val` — bits per character (enwik8 标准指标)
   - `Generated` — 模型生成文本样本
   - 日志保存到 `runs/` 目录

---

### 2026-04-07 — 长期记忆推理阶段量化加噪

**背景**: 以 v2 训练的模型为基础，模拟长期记忆（Neural Memory）存储介质在读取时存在低精度量化误差与高斯读取噪声的场景（类比 memristor / SRAM 硬件特性）。要求：写入（Store / 权重更新）过程保持全精度，仅推理（Retrieve）阶段施加量化噪声。

#### 核心机制

Neural Memory 每次 forward 分两个独立阶段：

- **Store**：对当前 chunk 的 keys/values 用 `torch.func.grad` 计算 per-sample 梯度，更新动态权重 $W_t$（全精度，与 Retrieve 路径天然解耦）
- **Retrieve**：用 $W_t$ 作为参数，通过 `functional_call(memory_model, W_t, queries)` 读取记忆值

量化加噪介入点：`retrieve_memories()` 内，`functional_call` **之前**，对动态权重字典施加：

```
W_t (float32)
    → data_quantization(W_t, bit=4)     # 4-bit 均匀量化
    → add_noise(W_q, n_scale=0.05)      # 高斯噪声（相对权重最大值 5%）
    → functional_call(memory_model, W_q, queries)
```

训练时（`model.train()`）完全透传原始权重，评估时（`model.eval()`）自动激活，零侵入训练流程。

#### 新增 / 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `ref-project/titans_pytorch/quant_noise_mem.py` | **新建** | 封装 `quantize_weight_dict()` 函数，复用 `quantization_noise/quant_util.py` |
| `ref-project/titans_pytorch/neural_memory.py` | **修改** | `__init__` 新增 `quant_noise_cfg` 参数；`retrieve_memories` 加入量化路径 |
| `train_mac_v3.py` | **修改** | `neural_memory_kwargs` 传入 `quant_noise_cfg`（enabled=True） |
| `train_mac_multigpu_v2.py` | **修改** | 同上 |
| `docs/neural_memory_quant_noise.md` | **新建** | 完整方案说明 + 数据流分析 |

#### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用（False 则退化为原始行为） |
| `bit` | `4` | 均匀量化位数 |
| `noise_scale` | `0.05` | 高斯噪声强度（相对权重最大值） |
| `noise_method` | `'add'` | 噪声叠加方式 |
| `noise_range` | `'max'` | 噪声幅度基准 |

调整参数只需修改训练脚本中的 `quant_noise_cfg` 字典，无需改动模型代码。

---

*后续变更请追加到「开发变更日志」末尾，保持时间倒序。*
