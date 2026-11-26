"""
Multi-GPU DDP training script for MAC Transformer on enwik8 (v7 — 骨干 4-bit fake-quant QAT).

v7 = v6 + 部署阶梯的训练侧适应（实验 10-12 的后续）：
  - 目标：把骨干（FFN / Attention QKV,out / NM K,V,Q 投影）的全部权重也放到
    忆阻器 4-bit 上，用 QAT 把 PTQ 的 +8.8% 代价压下来（W_t 历史压缩比 ~5×）
  - 机制：骨干目标 nn.Linear 原地替换为 FakeQuantLinear——训练前向对连续
    fp32 权重做 per-channel 对称 4-bit 量化仿真（STE 直通），权重保持连续
  - curriculum：bits 8 → 4 线性 ramp over FQ_WARMUP_STEPS（v4 教训：step 0
    全强度注入会崩；8-bit 起步扰动 ~0.4% 极小）
  - W_t 条件不变：v6 全套（4-bit+5% 读噪+存储+5% 写噪，恒定——v6 已适应）
  - 起点：v6 best 续训（fp32 权重，step 重置 0）
  - 部署：训完后做一次真正的 per-channel 4-bit 量化（ch_mse PTQ），预期近无损

Launch:
    torchrun --nproc_per_node=4 train_mac_v7.py
    # 冒烟: NUM_BATCHES=2000 torchrun --nproc_per_node=4 train_mac_v7.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"
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
from titans_pytorch.fake_quant_linear import (
    FakeQuantLinear, swap_backbone_fakequant, set_fq_step, get_fq_step)

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
NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 20000))   # v7：从 v6 best 续训 20k 步（骨干 4-bit 适应）
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = int(os.environ.get("GRAD_ACCUM", 2))   # 可用环境变量覆盖（GPU 不足时用更大累积保持等效 batch=32）
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = False
SEQ_LEN = 512

CHECKPOINT_DIR = Path('./checkpoints_v7')
INIT_CKPT = Path('./checkpoints_v6/best.pt')   # v7 起点：v6 best（fp32 权重）
CHECKPOINT_EVERY = 2500
RESUME_FROM_CHECKPOINT = True

# ── V7 骨干 fake-quant 配置 ★ v7 核心 ────────────────────────────────────────
FQ_WARMUP_STEPS = int(os.environ.get("FQ_WARMUP_STEPS", 5000))  # bits 8→4 线性 ramp
FQ_START_BITS   = 8
FQ_END_BITS     = 4   # = 部署条件（per-channel 4-bit，忆阻器 4-bit MLC）

# ── QAT + curriculum 配置 ★ v6 核心：v5 + 写噪声 ────────────────────────────
QUANT_BIT = 4
QUANT_NOISE_MAX = 0.05            # 读噪声（v5 已适应，恒定 0.05，无 curriculum）
NOISE_WARMUP_STEPS = 10000        # （仅用于 resume step 判断读噪声是否满；v6 起点已过）
WRITE_NOISE_MAX = 0.05            # ★ 写噪声（编程噪声）终态
WRITE_NOISE_WARMUP = 5000         # 写噪声 curriculum：0→WRITE_NOISE_MAX over 5k 步（v6 新扰动，需渐进）

def curr_write_noise(step, start):
    """写噪声 curriculum：从 v6 训练起点 start 起，0→WRITE_NOISE_MAX 线性升，超过 warmup 后恒定。"""
    prog = max(0, step - start)
    return WRITE_NOISE_MAX * min(1.0, prog / WRITE_NOISE_WARMUP)

def curr_noise_scale(step):
    """curriculum: 0 → QUANT_NOISE_MAX 线性，超过 warmup 后保持。"""
    return QUANT_NOISE_MAX * min(1.0, step / NOISE_WARMUP_STEPS)

# ── Experiment / perf ────────────────────────────────────────────────────────
PROJECT_NAME = 'titans-mac-transformer-v7'
RUN_NAME = f'mac-v7-4gpu-backbonefq{FQ_END_BITS}b-warmup{FQ_WARMUP_STEPS}'
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

def set_write_noise_scale(model, scale):
    """更新所有 NeuralMemory 模块的 write_noise_scale（写噪声 curriculum）。"""
    for mod in model.module.modules():
        if hasattr(mod, '_quant_noise_cfg'):
            mod._quant_noise_cfg['write_noise_scale'] = scale

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
        print("TITANS MAC Transformer v7 — v6 + 骨干 4-bit fake-quant QAT", flush=True)
        print("=" * 74, flush=True)
        print(f"  GPU              : {os.environ['CUDA_VISIBLE_DEVICES']} (4 GPUs, DDP)", flush=True)
        print(f"  有效批次         : 4 × {BATCH_SIZE} × {GRADIENT_ACCUMULATE_EVERY} = {4*BATCH_SIZE*GRADIENT_ACCUMULATE_EVERY}", flush=True)
        print(f"  QAT              : quant={QUANT_BIT}b, apply_in_training=True, quantized_storage=True", flush=True)
        print(f"  W_t 噪声         : 读 {QUANT_NOISE_MAX} + 写 {WRITE_NOISE_MAX} 恒定（v6 已适应，不再 ramp）", flush=True)
        print(f"  ★ 骨干 fake-quant: per-channel {FQ_END_BITS}-bit（FFN/Attn/NM 投影，权重 fp32 连续）", flush=True)
        print(f"  ★ curriculum     : bits {FQ_START_BITS} → {FQ_END_BITS} over {FQ_WARMUP_STEPS} 步（v4 教训：渐进注入）", flush=True)
        print(f"  起点             : v6 best 续训（fp32 权重），step 重置 0，训 {NUM_BATCHES} 步", flush=True)
        print(f"  Checkpoint dir   : {CHECKPOINT_DIR}", flush=True)
        print("=" * 74, flush=True)

    writer = None
    if is_main_process():
        import wandb
        wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
        wandb.run.name = RUN_NAME; wandb.run.save()
        log_dir = f'runs_v7/{RUN_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
                                  noise_method='add', noise_range='max', apply_in_training=True,
                                  quantized_storage=True,            # v5：cell 存 4-bit 离散级
                                  write_noise_scale=WRITE_NOISE_MAX), # ★ v6：写噪声(编程噪声)，运行时由 curriculum 覆盖
        )
    ).to(device)

    if is_main_process():
        tp = sum(p.numel() for p in model.parameters())
        print(f'  总参数量: {tp:,} ({tp/1e6:.2f}M)', flush=True)

    # ★ v7：骨干目标 nn.Linear → FakeQuantLinear（DDP 包装前替换；参数键名不变，
    #   v6 checkpoint 可直接加载）
    n_fq = swap_backbone_fakequant(model, warmup_steps=FQ_WARMUP_STEPS,
                                   start_bits=FQ_START_BITS, end_bits=FQ_END_BITS)
    if is_main_process():
        print(f'  ★ 骨干 fake-quant: {n_fq} 个 Linear 已替换 '
              f'(bits {FQ_START_BITS}→{FQ_END_BITS} over {FQ_WARMUP_STEPS} 步)', flush=True)

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
        if ckpt_path is None and (CHECKPOINT_DIR / 'best.pt').exists():
            ckpt_path = CHECKPOINT_DIR / 'best.pt'   # ★ v7 断点续训：优先自己的 best.pt
        if ckpt_path is None and INIT_CKPT.exists():
            ckpt_path = INIT_CKPT   # ★ v7 起点 = v6 best（fp32 权重）
        if ckpt_path is not None:
            if is_main_process(): print(f'  ↻ Resuming from {ckpt_path}', flush=True)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt.get('model', None))
            if sd is not None: model.module.load_state_dict(sd, assign=True)
            else: raise RuntimeError("Checkpoint missing 'model_state_dict'/'model'")
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            # v7 从 v6 best 续训：加载 v6 权重+优化器，但重置 step=0、best_val=inf，
            # 让 v7 以 NUM_BATCHES 全新步数训练（骨干 fake-quant curriculum 基于 step 0 起）
            start_step = 0
            best_val_loss = float('inf')
            del ckpt; model.module.to(device)
            if is_main_process(): print(f'    loaded v6 weights; v7 step reset to 0, best_val=inf', flush=True)
        else:
            if is_main_process(): print('  ✦ No checkpoint, starting from scratch.', flush=True)
    dist.barrier()

    # ── Training loop ─────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(start_step, NUM_BATCHES), initial=start_step, total=NUM_BATCHES,
                     mininterval=10., desc='training', disable=not is_main_process())

    for i in pbar:
        model.train()
        # ★ v7：W_t 条件恒定（v6 已适应读+写噪声，不再 ramp）；骨干 fake-quant
        #   bits 由 curriculum 控制（set_fq_step 每步更新）
        set_noise_scale(model, QUANT_NOISE_MAX)
        set_write_noise_scale(model, WRITE_NOISE_MAX)
        set_fq_step(i)                                   # 骨干 bits curriculum: 8→4
        fq_bits = FakeQuantLinear.current_bits()

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
            print(f'training loss: {train_loss:.4f}  acc: {train_acc:.4f}  rnoise: {QUANT_NOISE_MAX}  wnoise: {WRITE_NOISE_MAX}  fq_bits: {fq_bits}', flush=True)
            import wandb
            wandb.log(dict(loss=train_loss, acc=train_acc, grad_norm=grad_norm.item(),
                           noise=QUANT_NOISE_MAX, write_noise=WRITE_NOISE_MAX, fq_bits=fq_bits), step=i)
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Accuracy/train', train_acc, i)
            writer.add_scalar('GradNorm', grad_norm.item(), i)
            writer.add_scalar('noise_scale', QUANT_NOISE_MAX, i)
            writer.add_scalar('write_noise', WRITE_NOISE_MAX, i)
            writer.add_scalar('fq_bits', fq_bits, i)

        # ── Validation（用全强度噪声 = 部署读条件）──────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            set_noise_scale(model, QUANT_NOISE_MAX)   # 验证用部署全读噪声
            set_write_noise_scale(model, WRITE_NOISE_MAX)  # 验证用部署全写噪声
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
            model.eval(); set_noise_scale(model, QUANT_NOISE_MAX); set_write_noise_scale(model, WRITE_NOISE_MAX)
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
        print('\n✓ v7 (v6+骨干 fake-quant QAT) training complete.', flush=True)
        print(f'  Best val_loss: {best_val_loss:.4f}')
        print(f'  TensorBoard: tensorboard --logdir=runs_v7/ --port=6010')
    cleanup_distributed()

if __name__ == "__main__":
    main()
