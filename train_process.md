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

*后续变更请追加到「开发变更日志」末尾，保持时间倒序。*
