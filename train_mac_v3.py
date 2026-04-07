"""
Single-GPU training script for MAC Transformer on enwik8 (v3).

完全对齐 ref-project/train_comparison.py 的 titans 配置：
  - 单 GPU（GPU ID=1）
  - batch_size=8, gradient_accumulate=4 → 有效批次大小=32
  - use_flex_attn=False
  - 原始 HyperConnections（ref-project 的 titans_pytorch）
  - 无 wandb，改用 TensorBoard
  - 支持断点续训（RESUME_FROM_CHECKPOINT=True）

Launch command:
    python train_mac_v3.py

Monitor training:
    tensorboard --logdir=runs_v3/ --port=6008
"""

import os
import sys

# ── Select GPU (set before any CUDA call) ────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── CUDA / compiler environment ──────────────────────────────────────────────
os.environ["CUDA_HOME"] = "/opt/conda"
_nvidia_base = "/opt/conda/envs/titans/lib/python3.10/site-packages/nvidia"
_extra_includes = ":".join([
    f"{_nvidia_base}/cusparse/include",
    f"{_nvidia_base}/cublas/include",
    f"{_nvidia_base}/cufft/include",
    f"{_nvidia_base}/curand/include",
    f"{_nvidia_base}/cusolver/include",
    f"{_nvidia_base}/nvjitlink/include",
    f"{_nvidia_base}/cuda_runtime/include",
])
os.environ["CPATH"] = _extra_includes + (
    ":" + os.environ["CPATH"] if os.environ.get("CPATH") else ""
)
_extra_libs = ":".join([
    "/opt/conda/lib",
    f"{_nvidia_base}/cuda_runtime/lib",
    f"{_nvidia_base}/cusparse/lib",
    f"{_nvidia_base}/cublas/lib",
    f"{_nvidia_base}/nvjitlink/lib",
])
os.environ["LD_LIBRARY_PATH"] = _extra_libs + (
    ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
)

# ── 使用 ref-project 的 titans_pytorch（原始 HyperConnections）───────────────
_ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-project")
sys.path.insert(0, _ref_dir)

import random
import gzip
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import tqdm
from adam_atan2_pytorch import AdoptAtan2
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ═══════════════════════════════════════════════════════════════════════════
# 超参数（完全对齐 ref-project/train_comparison.py titans 配置）
# ═══════════════════════════════════════════════════════════════════════════

NUM_BATCHES           = int(1e5)
BATCH_SIZE            = 8          # 与 ref 一致（单 GPU bs=8）
GRADIENT_ACCUMULATE_EVERY = 4      # 与 ref 一致（accum=4）
                                   # 有效批次大小 = 8 × 4 = 32
LEARNING_RATE         = 2e-4
VALIDATE_EVERY        = 100
GENERATE_EVERY        = 500
PRIME_LENGTH          = 100
GENERATE_LENGTH       = 512
SHOULD_GENERATE       = True
SEQ_LEN               = 512        # 与 ref 一致

# ── Checkpoint ───────────────────────────────────────────────────────────────
CHECKPOINT_DIR        = Path('./checkpoints_v3')
RESUME_FROM_CHECKPOINT = True

# ── 模型结构（完全对齐 ref titans 配置）──────────────────────────────────────
NUM_TOKENS            = 256
DIM                   = 384
DEPTH                 = 8
SEGMENT_LEN           = 32
NUM_PERSIST_MEM       = 4
NUM_LONGTERM_MEM      = 4
NEURAL_MEM_LAYERS     = (2, 4, 6)
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE  = 128
NEURAL_MEMORY_DEPTH   = 2
USE_FLEX_ATTN         = False      # 与 ref 一致
SLIDING_WINDOWS       = True       # 与 ref 一致

RUN_NAME = f"mac-v3-refconfig-1gpu-bs{BATCH_SIZE}x{GRADIENT_ACCUMULATE_EVERY}"

# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def cycle(loader):
    while True:
        for data in loader:
            yield data

def load_best_checkpoint(checkpoint_dir):
    """返回 best.pt 路径（含 step 信息），不存在则返回 None。"""
    p = checkpoint_dir / 'best.pt'
    return p if p.exists() else None

def save_best_checkpoint(model, optim, step, checkpoint_dir, best_val_loss):
    """原子写：先写 .tmp 再 rename，确保 best.pt 始终完整。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_path  = checkpoint_dir / 'best.pt.tmp'
    best_path = checkpoint_dir / 'best.pt'
    torch.save({
        'step': step,
        'best_step': step,                      # 方便人眼查看
        'model': model.state_dict(),            # ref-project 兼容 key
        'model_state_dict': model.state_dict(), # evaluate_seqlen_ref.py 兼容 key
        'optimizer_state_dict': optim.state_dict(),
        'best_val_loss': best_val_loss,
    }, tmp_path)
    tmp_path.rename(best_path)                  # 原子替换
    print(f'  ★ best.pt updated → step={step}, val_loss={best_val_loss:.4f}')

# ═══════════════════════════════════════════════════════════════════════════
# 主训练流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── TensorBoard ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs_v3/{RUN_NAME}_{timestamp}')

    # ── 打印配置 ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("train_mac_v3.py — 完全对齐 ref-project titans 配置")
    print("=" * 65)
    print(f"  GPU              : {os.environ['CUDA_VISIBLE_DEVICES']} (CUDA_VISIBLE_DEVICES)")
    print(f"  batch_size       : {BATCH_SIZE}")
    print(f"  gradient_accum   : {GRADIENT_ACCUMULATE_EVERY}")
    print(f"  有效批次大小     : {BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY}")
    print(f"  seq_len          : {SEQ_LEN}")
    print(f"  learning_rate    : {LEARNING_RATE}")
    print(f"  num_batches      : {NUM_BATCHES}")
    print(f"  use_flex_attn    : {USE_FLEX_ATTN}")
    print(f"  sliding_window   : {SLIDING_WINDOWS}")
    print(f"  HyperConnections : 原始 (ref-project)")
    print(f"  checkpoint_dir   : {CHECKPOINT_DIR}")
    print("=" * 65)

    # ── 数据 ─────────────────────────────────────────────────────────────
    with gzip.open('./data/enwik8.gz') as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
        data_train, data_val = np.split(data, [int(90e6)])
        data_train, data_val = map(torch.from_numpy, (data_train, data_val))

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)

    train_iter = cycle(train_loader)
    val_iter   = cycle(val_loader)

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = MemoryAsContextTransformer(
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        segment_len=SEGMENT_LEN,
        num_persist_mem_tokens=NUM_PERSIST_MEM,
        num_longterm_mem_tokens=NUM_LONGTERM_MEM,
        neural_memory_layers=NEURAL_MEM_LAYERS,
        neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
        neural_mem_gate_attn_output=False,
        neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=USE_FLEX_ATTN,
        sliding_window_attn=SLIDING_WINDOWS,
        neural_memory_model=MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH),
        neural_memory_kwargs=dict(
            dim_head=64,
            heads=4,
            attn_pool_chunks=True,
            qk_rmsnorm=True,
            momentum=True,
            momentum_order=1,
            default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False,
            per_parameter_lr_modulation=True,
            spectral_norm_surprises=True,
        ),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {total_params:,} params ({total_params/1e6:.2f}M)')

    # ── 优化器 ───────────────────────────────────────────────────────────
    optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

    # ── 断点续训 ─────────────────────────────────────────────────────────
    start_step    = 0
    best_val_loss = float('inf')
    best_step     = -1

    if RESUME_FROM_CHECKPOINT:
        ckpt_path = load_best_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is not None:
            print(f'  ↻ Resuming from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt.get('model', None))
            if sd is None:
                raise RuntimeError(f"Unknown checkpoint format: {list(ckpt.keys())}")
            model.load_state_dict(sd, assign=True)
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            start_step    = ckpt['step'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf')) or float('inf')
            best_step     = ckpt.get('best_step', ckpt['step'])
            del ckpt
            model.to(device)   # assign=True 后需重新移到 GPU
            print(f'    resume step={start_step}, best_step={best_step}, best_val_loss={best_val_loss:.4f}')
        else:
            print('  ✦ No checkpoint found, starting from scratch.')

    # ── 训练循环 ─────────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(start_step, NUM_BATCHES), initial=start_step,
                     total=NUM_BATCHES, mininterval=10., desc='training')

    for i in pbar:
        model.train()

        # ── 梯度累积 + 计算 train accuracy ──
        train_acc_sum, train_acc_count = 0.0, 0
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(train_iter).to(device)
            loss  = model(batch, return_loss=True)
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()
            with torch.no_grad():
                logits = model(batch, return_loss=False)
                labels = batch[:, 1:]
                preds  = logits[:, :-1].argmax(dim=-1)
                train_acc_sum   += (preds == labels).float().sum().item()
                train_acc_count += labels.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        train_loss = loss.item()
        train_acc  = train_acc_sum / max(train_acc_count, 1)

        print(f'training loss: {train_loss:.4f}  acc: {train_acc:.4f}')
        writer.add_scalar('Loss/train',     train_loss,      i)
        writer.add_scalar('Accuracy/train', train_acc,       i)
        writer.add_scalar('GradNorm',       grad_norm.item(), i)
        writer.add_scalar('BPC/train',      train_loss / np.log(2), i)

        # ── 验证 ─────────────────────────────────────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_batch  = next(val_iter).to(device)
                val_loss   = model(val_batch, return_loss=True)
                val_logits = model(val_batch, return_loss=False)
                val_labels = val_batch[:, 1:]
                val_preds  = val_logits[:, :-1].argmax(dim=-1)
                val_acc    = (val_preds == val_labels).float().mean().item()

            val_loss_val = val_loss.item()
            print(f'validation loss: {val_loss_val:.4f}  acc: {val_acc:.4f}')
            writer.add_scalar('Loss/val',     val_loss_val,           i)
            writer.add_scalar('Accuracy/val', val_acc,                i)
            writer.add_scalar('BPC/val',      val_loss_val / np.log(2), i)

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                save_best_checkpoint(model, optim, step=i,
                                     checkpoint_dir=CHECKPOINT_DIR,
                                     best_val_loss=best_val_loss)

        # ── 生成样本 ─────────────────────────────────────────────────
        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:PRIME_LENGTH].to(device)
            prime = decode_tokens(inp)
            print(f'{prime} \n\n {"*" * 100}')
            sample = model.sample(inp[None, ...], GENERATE_LENGTH)
            output_str = decode_tokens(sample[0])
            print(output_str)
            writer.add_text('Generated', f'```\n{output_str}\n```', i)

    writer.close()
    # 读出最终 best 的 step 信息（供日志展示）
    final_ckpt = load_best_checkpoint(CHECKPOINT_DIR)
    if final_ckpt:
        info = torch.load(final_ckpt, map_location='cpu')
        best_step_final = info.get('best_step', info.get('step', '?'))
        print(f'\n✓ Training complete.')
        print(f'  Best checkpoint : {CHECKPOINT_DIR}/best.pt  (step {best_step_final})')
        print(f'  Best val_loss   : {best_val_loss:.4f}')
    print(f'  TensorBoard: tensorboard --logdir=runs_v3/ --port=6008')


if __name__ == '__main__':
    main()
