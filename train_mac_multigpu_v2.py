"""
Multi-GPU DDP training script for MAC Transformer on enwik8 (v2).

与之前 train_mac_multigpu.py 的关键区别（对齐 ref-project 的设置）：
  1. 使用 ref-project 的 titans_pytorch（原始 HyperConnections，非 MC 版本）
  2. 有效批次大小 = 32（4 GPU × 4 bs × 2 accum），与 ref 的 8×4=32 一致
  3. use_flex_attn = False（与 ref 一致，避免长序列评估时的潜在问题）
  4. 去掉 store_with_lookahead_value 参数（ref 版本的 neural_memory.py 不支持）
  5. checkpoint 保存格式兼容 ref-project（'model' key）

Launch command:
    torchrun --nproc_per_node=4 train_mac_multigpu_v2.py

Monitor training:
    tensorboard --logdir=runs_v2/ --port=6006
"""

import os
import sys

# ── Select GPUs (set before any CUDA call) ───────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ── CUDA / compiler environment (must be set before any torch import) ────────
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

# ── 使用 ref-project 的 titans_pytorch（原始 HyperConnections）──────────────
# 必须在 import titans_pytorch 之前把 ref-project 路径插入 sys.path 最前面
_ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-project")
sys.path.insert(0, _ref_dir)

import random
import gzip
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import tqdm

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
)

# ── Distributed setup ────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

# ── Constants ────────────────────────────────────────────────────────────────

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4                  # per-GPU batch size
GRADIENT_ACCUMULATE_EVERY = 2   # effective global batch = 4 GPUs × 4 bs × 2 accum = 32
                                # 与 ref-project 的 8 bs × 4 accum = 32 一致
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# ── Checkpoint related ────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path('./checkpoints_v2')
CHECKPOINT_EVERY = 2500        # save checkpoint every N steps
RESUME_FROM_CHECKPOINT = True   # auto-resume from latest checkpoint if exists

# ── Neural memory related ───────────────────────────────────────────────────
# 完全对齐 ref-project/train_comparison.py 中的 titans 配置

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
NEURAL_MEM_SPEC_NORM_SURPRISES = True
# 注意：ref 版本不支持 store_with_lookahead_value，这里去掉

# ── Experiment related ───────────────────────────────────────────────────────

PROJECT_NAME = 'titans-mac-transformer-v2'
RUN_NAME = f'mac-v2-4gpu-refHC-bs32-{NUM_LONGTERM_MEM}ltm-layers{NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False  # turn on to pipe experiment to cloud

# ── Perf related ─────────────────────────────────────────────────────────────

USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = False          # ★ 关键改动：与 ref 一致，使用手动分段注意力
USE_FAST_INFERENCE = False

# ── Helpers ──────────────────────────────────────────────────────────────────

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optim, step, epoch, path, best_val_loss=None):
    """Save model, optimizer, and training state (rank 0 only).
    
    保存两种格式的 state_dict：
      - 'model_state_dict': 兼容当前项目的加载方式
      - 'model': 兼容 ref-project 的加载方式
    """
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict()  # unwrap DDP
    torch.save({
        'step': step,
        'epoch': epoch,
        'model': state_dict,                 # ref-project 格式
        'model_state_dict': state_dict,      # 当前项目格式
        'optimizer_state_dict': optim.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)
    print(f'  ✓ Checkpoint saved: {path}  (step {step})')

def load_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory, return path or None."""
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted(checkpoint_dir.glob('ckpt_step_*.pt'))
    return ckpts[-1] if ckpts else None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        print("=" * 70)
        print("TITANS MAC Transformer v2 — 对齐 ref-project 设置")
        print("=" * 70)
        print(f"  HyperConnections : 原始版本 (get_init_and_expand_reduce_stream_functions)")
        print(f"  use_flex_attn    : {USE_FLEX_ATTN}")
        print(f"  有效批次大小     : {4} GPUs × {BATCH_SIZE} bs × {GRADIENT_ACCUMULATE_EVERY} accum = {4 * BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY}")
        print(f"  SEQ_LEN          : {SEQ_LEN}")
        print(f"  LR               : {LEARNING_RATE}")
        print(f"  Checkpoint dir   : {CHECKPOINT_DIR}")
        print("=" * 70)

    # ── Logging: wandb + TensorBoard (only on rank 0) ────────────────────
    writer = None
    if is_main_process():
        import wandb
        wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
        wandb.run.name = RUN_NAME
        wandb.run.save()

        log_dir = f'runs_v2/{RUN_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir=log_dir)
        print(f'  ✦ TensorBoard logs → {log_dir}')
        print(f'    Run: tensorboard --logdir=runs_v2/ --port=6006')

    # ── Memory model ─────────────────────────────────────────────────────
    neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)

    # ── Model ────────────────────────────────────────────────────────────
    # 使用 ref-project 的 titans_pytorch，确保 HyperConnections 是原始版本
    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=384,
        depth=8,
        segment_len=WINDOW_SIZE,
        num_persist_mem_tokens=NUM_PERSIST_MEM,
        num_longterm_mem_tokens=NUM_LONGTERM_MEM,
        neural_memory_layers=NEURAL_MEM_LAYERS,
        neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
        neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
        neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        use_flex_attn=USE_FLEX_ATTN,
        sliding_window_attn=SLIDING_WINDOWS,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=64,
            heads=4,
            attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm=NEURAL_MEM_QK_NORM,
            momentum=NEURAL_MEM_MOMENTUM,
            momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
            use_accelerated_scan=USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR,
            spectral_norm_surprises=NEURAL_MEM_SPEC_NORM_SURPRISES,
            # 注意：不传 store_with_lookahead_value（ref 版本不支持）
            # 推理阶段对动态权重量化+加噪（eval 模式下生效，训练不受影响）
            quant_noise_cfg=dict(
                enabled=True,
                bit=4,            # 均匀量化位数
                noise_scale=0.05, # 高斯噪声强度（相对权重最大值）
                noise_method='add',
                noise_range='max',
            ),
        )
    ).to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  总参数量: {total_params:,} ({total_params/1e6:.2f}M)')
        print(f'  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)')

    # ── Wrap with DDP ────────────────────────────────────────────────────
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # ── Data ─────────────────────────────────────────────────────────────
    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    def infinite_train_loader():
        """Yields batches indefinitely, bumping epoch for proper shuffling."""
        epoch = 0
        while True:
            train_sampler.set_epoch(epoch)
            for batch in train_loader:
                yield batch.to(device, non_blocking=True)
            epoch += 1

    def infinite_val_loader():
        epoch = 0
        while True:
            val_sampler.set_epoch(epoch)
            for batch in val_loader:
                yield batch.to(device, non_blocking=True)
            epoch += 1

    train_iter = infinite_train_loader()
    val_iter   = infinite_val_loader()

    # ── Optimizer ────────────────────────────────────────────────────────
    optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_FROM_CHECKPOINT:
        ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is not None:
            if is_main_process():
                print(f'  ↻ Resuming from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # 兼容两种 state_dict key
            sd = ckpt.get('model_state_dict', ckpt.get('model', None))
            if sd is not None:
                model.module.load_state_dict(sd, assign=True)
            else:
                raise RuntimeError(f"Checkpoint has no 'model_state_dict' or 'model' key: {list(ckpt.keys())}")
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            start_step = ckpt['step'] + 1
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf')) or float('inf')
            del ckpt
            # 需要重新把模型放到对应 GPU 上（assign=True 可能把参数留在 CPU）
            model.module.to(device)
            if is_main_process():
                print(f'    Resuming from step {start_step}, best_val_loss={best_val_loss:.4f}')
        else:
            if is_main_process():
                print('  ✦ No checkpoint found, starting from scratch.')

    # ── Fast-forward data iterators if resuming ──────────────────────────
    # 注意：TextSamplerDataset 每次 __getitem__ 都是独立随机采样（rand_start），
    # 与 index/step 无关，重启后数据顺序本来就无法完全复原。
    # 因此直接跳过 fast-forward，从随机位置继续即可，不影响训练质量。
    # （跳过也避免了 resume 时迭代几万次 next() 的额外等待时间）
    if start_step > 0 and is_main_process():
        print(f'  ↻ Data iterator starts fresh (random sampling, no fast-forward needed)')
    dist.barrier()

    # ── Training loop ────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(start_step, NUM_BATCHES), initial=start_step,
                     total=NUM_BATCHES, mininterval=10., desc='training',
                     disable=not is_main_process())

    for i in pbar:
        model.train()

        train_acc_sum = 0.0
        train_acc_count = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(train_iter)
            loss = model(batch, return_loss=True)
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()
            # ── 计算 training accuracy (token-level top-1) ──
            with torch.no_grad():
                logits = model(batch, return_loss=False)       # (B, N, vocab)
                labels = batch[:, 1:]                          # 目标: 第 2 个 token 开始
                preds  = logits[:, :-1].argmax(dim=-1)         # 预测: 去掉最后一个位置
                train_acc_sum   += (preds == labels).float().sum().item()
                train_acc_count += labels.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        train_loss = loss.item()
        train_acc  = train_acc_sum / max(train_acc_count, 1)

        if is_main_process():
            print(f'training loss: {train_loss:.4f}  acc: {train_acc:.4f}')
            import wandb
            wandb.log(dict(loss=train_loss, acc=train_acc, grad_norm=grad_norm.item()), step=i)
            # ── TensorBoard ──
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Accuracy/train', train_acc, i)
            writer.add_scalar('GradNorm', grad_norm.item(), i)

        # ── Validation ───────────────────────────────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(val_iter)
                val_loss = model(val_batch, return_loss=True)
                # ── 计算 validation accuracy ──
                val_logits = model(val_batch, return_loss=False)
                val_labels = val_batch[:, 1:]
                val_preds  = val_logits[:, :-1].argmax(dim=-1)
                val_acc    = (val_preds == val_labels).float().mean().item()
            val_loss_val = val_loss.item()
            if is_main_process():
                print(f'validation loss: {val_loss_val:.4f}  acc: {val_acc:.4f}')
                writer.add_scalar('Loss/val', val_loss_val, i)
                writer.add_scalar('Accuracy/val', val_acc, i)
                # ── BPC (bits per character) — standard enwik8 metric ──
                train_bpc = train_loss / np.log(2)
                val_bpc   = val_loss_val / np.log(2)
                writer.add_scalar('BPC/train', train_bpc, i)
                writer.add_scalar('BPC/val', val_bpc, i)

                # ── 保存最佳模型 ──
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    save_checkpoint(
                        model, optim, step=i, epoch=0,
                        path=CHECKPOINT_DIR / 'best.pt',
                        best_val_loss=best_val_loss,
                    )
                    print(f'  ★ New best val_loss: {best_val_loss:.4f} at step {i}')

        # ── Generation ───────────────────────────────────────────────
        if SHOULD_GENERATE and i % GENERATE_EVERY == 0 and is_main_process():
            model.eval()
            inp = random.choice(val_dataset)[:PRIME_LENGTH].to(device)
            prime = decode_tokens(inp)
            print(f'{prime} \n\n {"*" * 100}')

            sample = model.module.sample(
                inp[None, ...], GENERATE_LENGTH, use_cache=USE_FAST_INFERENCE
            )
            output_str = decode_tokens(sample[0])
            print(output_str)
            writer.add_text('Generated', f'```\n{output_str}\n```', i)

        # ── Save checkpoint ──────────────────────────────────────────
        if i > 0 and i % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                model, optim, step=i, epoch=0,
                path=CHECKPOINT_DIR / f'ckpt_step_{i:06d}.pt',
                best_val_loss=best_val_loss,
            )

        dist.barrier()

    # ── Final checkpoint ─────────────────────────────────────────────────
    save_checkpoint(
        model, optim, step=NUM_BATCHES, epoch=0,
        path=CHECKPOINT_DIR / f'ckpt_step_{NUM_BATCHES:06d}_final.pt',
        best_val_loss=best_val_loss,
    )

    if is_main_process() and writer is not None:
        writer.close()
        print('\n✓ Training complete. TensorBoard logs saved to runs_v2/')
        print(f'  Best validation loss: {best_val_loss:.4f}')

    # ── Cleanup ──────────────────────────────────────────────────────────
    cleanup_distributed()

if __name__ == "__main__":
    main()
