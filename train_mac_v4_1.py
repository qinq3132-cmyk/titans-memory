"""
Multi-GPU DDP training script for MAC Transformer on enwik8 (v4.1 — QAT + noise curriculum).

v4.1 = v4 + 噪声 curriculum。
  - 训练前向注入 4-bit 量化 + 高斯噪声（apply_in_training=True, STE 回传）→ 真正模拟部署读条件
  - noise_scale 从 0 线性升到 0.05（前 NOISE_WARMUP_STEPS 步），之后保持 0.05
    → 避免 v4（step0 全强度）的崩盘：模型先学会基本任务（噪声≈0），再逐步适配读噪声
  - bit=4 恒定；验证时用全强度 0.05 噪声（=部署读条件），val_loss 直接反映量化鲁棒性
  - 其余对齐 ref-project titans：4 GPU × 4 bs × 2 accum = 32，use_flex_attn=False

Launch:
    torchrun --nproc_per_node=4 train_mac_v4_1.py
    # 冒烟: NUM_BATCHES=50 torchrun --nproc_per_node=4 train_mac_v4_1.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,4,7"
os.environ["CUDA_HOME"] = "/opt/conda"
_nvidia_base = "/opt/conda/envs/titans/lib/python3.10/site-packages/nvidia"
_extra_includes = ":".join([
    f"{_nvidia_base}/cusparse/include", f"{_nvidia_base}/cublas/include",
    f"{_nvidia_base}/cufft/include", f"{_nvidia_base}/curand/include",
    f"{_nvidia_base}/cusolver/include", f"{_nvidia_base}/nvjitlink/include",
    f"{_nvidia_base}/cuda_runtime/include",
])
os.environ["CPATH"] = _extra_includes + (":" + os.environ["CPATH"] if os.environ.get("CPATH") else "")
_extra_libs = ":".join([
    "/opt/conda/lib", f"{_nvidia_base}/cuda_runtime/lib",
    f"{_nvidia_base}/cusparse/lib", f"{_nvidia_base}/cublas/lib", f"{_nvidia_base}/nvjitlink/lib",
])
os.environ["LD_LIBRARY_PATH"] = _extra_libs + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")

_ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-project")
sys.path.insert(0, _ref_dir)

import random, gzip
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
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ── Distributed setup ────────────────────────────────────────────────────────
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
def cleanup_distributed():
    dist.destroy_process_group()
def is_main_process():
    return dist.get_rank() == 0

# ── Constants ────────────────────────────────────────────────────────────────
NUM_BATCHES = int(os.environ.get("NUM_BATCHES", int(1e5)))
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 2   # 4×4×2 = 32
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

CHECKPOINT_DIR = Path('./checkpoints_v4_1')
CHECKPOINT_EVERY = 2500
RESUME_FROM_CHECKPOINT = True

# ── QAT + curriculum 配置 ★ v4.1 核心 ───────────────────────────────────────
QUANT_BIT = 4
QUANT_NOISE_MAX = 0.05            # 训练终态 / 验证用的噪声强度（=部署读条件）
NOISE_WARMUP_STEPS = 10000        # noise_scale 从 0 线性升到 QUANT_NOISE_MAX 的步数

def curr_noise_scale(step):
    """curriculum: 0 → QUANT_NOISE_MAX 线性，超过 warmup 后保持。"""
    return QUANT_NOISE_MAX * min(1.0, step / NOISE_WARMUP_STEPS)

# ── Experiment / perf ────────────────────────────────────────────────────────
PROJECT_NAME = 'titans-mac-transformer-v4_1'
RUN_NAME = f'mac-v4.1-4gpu-qat-curr{NOISE_WARMUP_STEPS}-quant{QUANT_BIT}b'
WANDB_ONLINE = False
USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = False
USE_FAST_INFERENCE = False
DATA_PATH = "/home/titans-pytorch-main/data/enwik8.gz"

# ── Neural memory 结构（对齐 ref-project）────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────
def decode_token(token): return str(chr(max(32, token)))
def decode_tokens(tokens): return ''.join(list(map(decode_token, tokens)))

def set_noise_scale(model, scale):
    """更新所有 NeuralMemory 模块的 noise_scale（训练 curriculum / 验证全强度）。"""
    for mod in model.module.modules():
        if hasattr(mod, '_quant_noise_cfg'):
            mod._quant_noise_cfg['noise_scale'] = scale

def save_checkpoint(model, optim, step, epoch, path, best_val_loss=None):
    if not is_main_process(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.module.state_dict()
    torch.save({'step': step, 'epoch': epoch, 'model': sd, 'model_state_dict': sd,
                'optimizer_state_dict': optim.state_dict(), 'best_val_loss': best_val_loss}, path)
    print(f'  ✓ Checkpoint saved: {path}  (step {step})', flush=True)

def load_latest_checkpoint(checkpoint_dir):
    if not checkpoint_dir.exists(): return None
    ckpts = sorted(checkpoint_dir.glob('ckpt_step_*.pt'))
    return ckpts[-1] if ckpts else None

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        print("=" * 74, flush=True)
        print("TITANS MAC Transformer v4.1 — QAT + noise curriculum", flush=True)
        print("=" * 74, flush=True)
        print(f"  GPU              : {os.environ['CUDA_VISIBLE_DEVICES']} (4 GPUs, DDP)", flush=True)
        print(f"  有效批次         : 4 × {BATCH_SIZE} × {GRADIENT_ACCUMULATE_EVERY} = {4*BATCH_SIZE*GRADIENT_ACCUMULATE_EVERY}", flush=True)
        print(f"  QAT              : quant={QUANT_BIT}b, apply_in_training=True", flush=True)
        print(f"  noise curriculum : 0 → {QUANT_NOISE_MAX} over {NOISE_WARMUP_STEPS} steps (之后恒定 {QUANT_NOISE_MAX})", flush=True)
        print(f"  验证噪声         : {QUANT_NOISE_MAX}（=部署读条件）", flush=True)
        print(f"  Checkpoint dir   : {CHECKPOINT_DIR}", flush=True)
        print("=" * 74, flush=True)

    writer = None
    if is_main_process():
        import wandb
        wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
        wandb.run.name = RUN_NAME; wandb.run.save()
        log_dir = f'runs_v4_1/{RUN_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir=log_dir)
        print(f'  ✦ TensorBoard → {log_dir}', flush=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = MemoryAsContextTransformer(
        num_tokens=256, dim=384, depth=8, segment_len=WINDOW_SIZE,
        num_persist_mem_tokens=NUM_PERSIST_MEM, num_longterm_mem_tokens=NUM_LONGTERM_MEM,
        neural_memory_layers=NEURAL_MEM_LAYERS, neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
        neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
        neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        use_flex_attn=USE_FLEX_ATTN, sliding_window_attn=SLIDING_WINDOWS,
        neural_memory_model=MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH),
        neural_memory_kwargs=dict(
            dim_head=64, heads=4, attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm=NEURAL_MEM_QK_NORM, momentum=NEURAL_MEM_MOMENTUM,
            momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
            use_accelerated_scan=USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR,
            spectral_norm_surprises=NEURAL_MEM_SPEC_NORM_SURPRISES,
            quant_noise_cfg=dict(enabled=True, bit=QUANT_BIT, noise_scale=QUANT_NOISE_MAX,
                                  noise_method='add', noise_range='max', apply_in_training=True),
        )
    ).to(device)

    if is_main_process():
        tp = sum(p.numel() for p in model.parameters())
        print(f'  总参数量: {tp:,} ({tp/1e6:.2f}M)', flush=True)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ── Data ─────────────────────────────────────────────────────────────
    with gzip.open(DATA_PATH) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        data_train, data_val = np.split(data, [int(90e6)])
        data_train, data_val = map(torch.from_numpy, (data_train, data_val))

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__(); self.data = data; self.seq_len = seq_len
        def __getitem__(self, index):
            rs = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            return self.data[rs: rs + self.seq_len + 1].long()
        def __len__(self): return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, sampler=val_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)

    def infinite(sampler, loader):
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            for b in loader: yield b.to(device, non_blocking=True)
            epoch += 1
    train_iter = infinite(train_sampler, train_loader)
    val_iter   = infinite(val_sampler, val_loader)

    optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0; best_val_loss = float('inf')
    if RESUME_FROM_CHECKPOINT:
        ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is not None:
            if is_main_process(): print(f'  ↻ Resuming from {ckpt_path}', flush=True)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt.get('model', None))
            if sd is not None: model.module.load_state_dict(sd, assign=True)
            else: raise RuntimeError("Checkpoint missing 'model_state_dict'/'model'")
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            start_step = ckpt['step'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf')) or float('inf')
            del ckpt; model.module.to(device)
            if is_main_process(): print(f'    from step {start_step}, best_val_loss={best_val_loss:.4f}', flush=True)
        else:
            if is_main_process(): print('  ✦ No checkpoint, starting from scratch.', flush=True)
    dist.barrier()

    # ── Training loop ─────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(start_step, NUM_BATCHES), initial=start_step, total=NUM_BATCHES,
                     mininterval=10., desc='training', disable=not is_main_process())

    for i in pbar:
        model.train()
        # ★ curriculum：训练前向用当前 step 对应的 noise_scale
        ns = curr_noise_scale(i)
        set_noise_scale(model, ns)

        train_acc_sum = 0.0; train_acc_count = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(train_iter)
            loss = model(batch, return_loss=True)
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()
            with torch.no_grad():
                logits = model(batch, return_loss=False)
                labels = batch[:, 1:]; preds = logits[:, :-1].argmax(dim=-1)
                train_acc_sum += (preds == labels).float().sum().item()
                train_acc_count += labels.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step(); optim.zero_grad()
        train_loss = loss.item(); train_acc = train_acc_sum / max(train_acc_count, 1)

        if is_main_process():
            print(f'training loss: {train_loss:.4f}  acc: {train_acc:.4f}  noise: {ns:.4f}', flush=True)
            import wandb
            wandb.log(dict(loss=train_loss, acc=train_acc, grad_norm=grad_norm.item(), noise=ns), step=i)
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Accuracy/train', train_acc, i)
            writer.add_scalar('GradNorm', grad_norm.item(), i)
            writer.add_scalar('noise_scale', ns, i)

        # ── Validation（用全强度噪声 = 部署读条件）──────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            set_noise_scale(model, QUANT_NOISE_MAX)   # 验证用部署全噪声
            with torch.no_grad():
                val_batch = next(val_iter)
                val_loss = model(val_batch, return_loss=True)
                val_logits = model(val_batch, return_loss=False)
                val_labels = val_batch[:, 1:]; val_preds = val_logits[:, :-1].argmax(dim=-1)
                val_acc = (val_preds == val_labels).float().mean().item()
            val_loss_val = val_loss.item()
            if is_main_process():
                print(f'validation loss: {val_loss_val:.4f}  acc: {val_acc:.4f}  (noise={QUANT_NOISE_MAX})', flush=True)
                writer.add_scalar('Loss/val', val_loss_val, i)
                writer.add_scalar('Accuracy/val', val_acc, i)
                writer.add_scalar('BPC/train', train_loss / np.log(2), i)
                writer.add_scalar('BPC/val',   val_loss_val / np.log(2), i)
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    save_checkpoint(model, optim, step=i, epoch=0,
                                    path=CHECKPOINT_DIR / 'best.pt', best_val_loss=best_val_loss)
                    print(f'  ★ New best val_loss: {best_val_loss:.4f} at step {i}', flush=True)

        # ── Generation ──────────────────────────────────────────────
        if SHOULD_GENERATE and i % GENERATE_EVERY == 0 and is_main_process():
            model.eval(); set_noise_scale(model, QUANT_NOISE_MAX)
            inp = random.choice(val_dataset)[:PRIME_LENGTH].to(device)
            print(f'{decode_tokens(inp)} \n\n {"*" * 100}')
            print(decode_tokens(model.module.sample(inp[None, ...], GENERATE_LENGTH,
                                                    use_cache=USE_FAST_INFERENCE)[0]))

        if i > 0 and i % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optim, step=i, epoch=0,
                            path=CHECKPOINT_DIR / f'ckpt_step_{i:06d}.pt', best_val_loss=best_val_loss)
        dist.barrier()

    save_checkpoint(model, optim, step=NUM_BATCHES, epoch=0,
                    path=CHECKPOINT_DIR / f'ckpt_step_{NUM_BATCHES:06d}_final.pt', best_val_loss=best_val_loss)
    if is_main_process() and writer is not None:
        writer.close()
        print('\n✓ v4.1 (QAT+curriculum) training complete.', flush=True)
        print(f'  Best val_loss: {best_val_loss:.4f}')
        print(f'  TensorBoard: tensorboard --logdir=runs_v4_1/ --port=6010')
    cleanup_distributed()

if __name__ == "__main__":
    main()
