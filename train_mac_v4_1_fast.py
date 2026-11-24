"""
Multi-GPU DDP training script for MAC Transformer on enwik8 (v4.1-fast).

v4.1-fast = v4.1 的训练逻辑不变，纯性能优化版：
  1. 每个 micro-batch 只做【一次】前向：直接取 logits 自己算 CE loss + acc，
     去掉 v4.1 里为算 accuracy 多做的 no_grad 前向（约省 1/3 计算量）。
     输入用 batch[:, :-1]（511 tokens），与 return_loss=True 内部切片完全一致 → 数值等价。
  2. 去掉每步末尾多余的 dist.barrier()（DDP 的 backward allreduce 已同步梯度）。
  3. 合并 .item() 同步点（acc 用 tensor 累加，每步只同步一次），减少 GPU 空等。
  4. 验证也只做一次前向（loss + acc 复用同一份 logits）。
  5. 首个梯度累积 micro-batch 使用 DDP no_sync，避免重复 all-reduce。
  6. 默认开启 TF32、static DDP graph，并关闭周期性生成；均可用环境变量覆盖。
  7. 可选 accelerated scan / torch.compile（需要隔离 GPU 做 A/B 测试）。

  其余与 v4.1 完全一致：QAT 4-bit + noise curriculum（0 → 0.05 over 10000 steps）。
  默认 batch=8、accum=1，保持有效 batch=32；可用 BATCH_SIZE/GRADIENT_ACCUMULATE_EVERY 覆盖。
  验证用全强度噪声 0.05，checkpoint 目录沿用 checkpoints_v4_1 → 可无缝续跑。

Launch:
    torchrun --nproc_per_node=4 train_mac_v4_1_fast.py
    # 冒烟: NUM_BATCHES=50 torchrun --nproc_per_node=4 train_mac_v4_1_fast.py
"""

import os
import sys
from contextlib import nullcontext

# Keep the historical default, but allow the launcher to select a disjoint set
# of GPUs when another training job is running.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,3,4,7")
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


def _fast_quantize_weight_dict(weights_dict, bit=4, noise_scale=0.05,
                               noise_method='add', noise_range='max'):
    """Tensor-only QAT for the dynamic memory weights.

    The project quantizer performs several ``.item()`` calls for every weight
    tensor. Those calls synchronize the host with the GPU dozens of times per
    training step. This preserves the symmetric 4-bit/max-range behavior while
    keeping all reductions and branches on-device.
    """
    quant_range = float(2 ** bit - 2)
    result = {}
    for name, weight in weights_dict.items():
        max_abs = weight.abs().amax()
        scale = (2. * max_abs / quant_range).clamp_min(torch.finfo(weight.dtype).tiny)
        quantized = (weight.clamp(-max_abs, max_abs) / scale).round() * scale

        if noise_scale > 0.0:
            if noise_range == 'max':
                factor = quantized.max()
            elif noise_range == 'std':
                factor = quantized.std()
            elif noise_range == 'max_min':
                factor = quantized.max() - quantized.min()
            elif noise_range == 'maxabs_2':
                factor = 2. * quantized.abs().amax()
            else:
                raise ValueError(f'unknown noise_range: {noise_range}')

            noise = torch.randn_like(quantized).clamp_(-3., 3.)
            if noise_method == 'add':
                quantized = quantized + noise * factor * noise_scale
            elif noise_method == 'mul':
                quantized = quantized * (noise * noise_scale + 1.)
            else:
                raise ValueError(f'unknown noise_method: {noise_method}')

        # STE: forward uses quantized/noisy weights, gradient is identity.
        result[name] = (quantized - weight).detach() + weight
    return result


def install_fast_quantizer():
    """Patch only this process; the original v4.1 process is untouched."""
    import titans_pytorch.neural_memory as neural_memory
    neural_memory.quantize_weight_dict = _fast_quantize_weight_dict

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
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRADIENT_ACCUMULATE_EVERY = int(os.environ.get("GRADIENT_ACCUMULATE_EVERY", "1"))

LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = os.environ.get("SHOULD_GENERATE", "0") == "1"
SEQ_LEN = 512

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints_v4_1"))
CHECKPOINT_EVERY = 2500
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME", "1") == "1"

# ── QAT + curriculum 配置（与 v4.1 一致）────────────────────────────────────
QUANT_BIT = 4
QUANT_NOISE_MAX = 0.05
NOISE_WARMUP_STEPS = 10000

def curr_noise_scale(step):
    """curriculum: 0 → QUANT_NOISE_MAX 线性，超过 warmup 后保持。"""
    return QUANT_NOISE_MAX * min(1.0, step / NOISE_WARMUP_STEPS)

# ── Experiment / perf ────────────────────────────────────────────────────────
PROJECT_NAME = 'titans-mac-transformer-v4_1'
RUN_NAME = f'mac-v4.1-4gpu-qat-curr{NOISE_WARMUP_STEPS}-quant{QUANT_BIT}b'
WANDB_ONLINE = False
USE_ACCELERATED_SCAN = os.environ.get("USE_ACCELERATED_SCAN", "0") == "1"
USE_FLEX_ATTN = False
USE_FAST_INFERENCE = os.environ.get("USE_FAST_INFERENCE", "1") == "1"
DATA_PATH = "/home/titans-pytorch-main/data/enwik8.gz"

# ★ 性能开关（不影响数值语义）
USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"   # 实验性：串行 scan 结构，需实测
USE_TF32 = os.environ.get("USE_TF32", "1") == "1"
USE_STATIC_GRAPH = os.environ.get("USE_STATIC_GRAPH", "1") == "1"
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))

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

def set_noise_scale(model, scale, noise_modules=None):
    """更新 NeuralMemory 的 noise_scale，不重复遍历整个模型。"""
    if noise_modules is None:
        root = model.module if isinstance(model, DDP) else model
        if hasattr(root, '_orig_mod'):
            root = root._orig_mod
        noise_modules = [mod for mod in root.modules() if hasattr(mod, '_quant_noise_cfg')]
    for mod in noise_modules:
        mod._quant_noise_cfg['noise_scale'] = scale

def save_checkpoint(model, optim, step, epoch, path, best_val_loss=None):
    if not is_main_process(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    m = model.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    sd = m.state_dict()
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
    install_fast_quantizer()
    if USE_TF32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process():
        print("=" * 74, flush=True)
        print("TITANS MAC Transformer v4.1-fast — QAT + noise curriculum (perf build)", flush=True)
        print("=" * 74, flush=True)
        print(f"  GPU              : {os.environ['CUDA_VISIBLE_DEVICES']} (4 GPUs, DDP)", flush=True)
        print(f"  有效批次         : 4 × {BATCH_SIZE} × {GRADIENT_ACCUMULATE_EVERY} = {4*BATCH_SIZE*GRADIENT_ACCUMULATE_EVERY}", flush=True)
        print(f"  TF32             : {USE_TF32}", flush=True)
        print(f"  accelerated scan : {USE_ACCELERATED_SCAN}", flush=True)
        print(f"  static DDP graph : {USE_STATIC_GRAPH}", flush=True)
        print(f"  generate         : {SHOULD_GENERATE}", flush=True)
        print(f"  QAT              : quant={QUANT_BIT}b, apply_in_training=True", flush=True)
        print(f"  noise curriculum : 0 → {QUANT_NOISE_MAX} over {NOISE_WARMUP_STEPS} steps (之后恒定 {QUANT_NOISE_MAX})", flush=True)
        print(f"  验证噪声         : {QUANT_NOISE_MAX}（=部署读条件）", flush=True)
        print(f"  torch.compile    : {USE_COMPILE}", flush=True)
        print(f"  Checkpoint dir   : {CHECKPOINT_DIR}", flush=True)
        print("=" * 74, flush=True)

    writer = None
    wandb = None
    if is_main_process():
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
        wandb.run.name = RUN_NAME; wandb.run.save()
        log_dir = f'runs_v4_1/{RUN_NAME}_fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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

    optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

    # ── Resume（先载入权重再包 DDP/compile）──────────────────────────────
    start_step = 0; best_val_loss = float('inf')
    if RESUME_FROM_CHECKPOINT:
        ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is not None:
            if is_main_process(): print(f'  ↻ Resuming from {ckpt_path}', flush=True)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt.get('model', None))
            if sd is not None: model.load_state_dict(sd, assign=True)
            else: raise RuntimeError("Checkpoint missing 'model_state_dict'/'model'")
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            start_step = ckpt['step'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf')) or float('inf')
            del ckpt; model.to(device)
            if is_main_process(): print(f'    from step {start_step}, best_val_loss={best_val_loss:.4f}', flush=True)
        else:
            if is_main_process(): print('  ✦ No checkpoint, starting from scratch.', flush=True)

    if USE_COMPILE:
        model = torch.compile(model)

    ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank)
    if USE_STATIC_GRAPH:
        # The 17 unused parameters are fixed for this architecture; static_graph
        # handles them without traversing the graph on every backward pass.
        ddp_kwargs['static_graph'] = True
    else:
        ddp_kwargs['find_unused_parameters'] = True
    model = DDP(model, **ddp_kwargs)
    noise_modules = [mod for mod in model.module.modules() if hasattr(mod, '_quant_noise_cfg')]
    dist.barrier()

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
                              num_workers=2, pin_memory=True, persistent_workers=True,
                              prefetch_factor=2, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, sampler=val_sampler,
                              num_workers=2, pin_memory=True, persistent_workers=True,
                              prefetch_factor=2, drop_last=True)

    def infinite(sampler, loader):
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            for b in loader: yield b.to(device, non_blocking=True)
            epoch += 1
    train_iter = infinite(train_sampler, train_loader)
    val_iter   = infinite(val_sampler, val_loader)

    # ── Training loop ─────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(start_step, NUM_BATCHES), initial=start_step, total=NUM_BATCHES,
                     mininterval=10., desc='training', disable=not is_main_process())

    for i in pbar:
        model.train()
        # ★ curriculum：训练前向用当前 step 对应的 noise_scale
        ns = curr_noise_scale(i)
        set_noise_scale(model, ns, noise_modules)

        should_log = is_main_process() and (i % LOG_EVERY == 0)
        need_metrics = is_main_process() and (should_log or i % VALIDATE_EVERY == 0)
        acc_correct = torch.zeros((), device=device, dtype=torch.long)
        acc_total = 0
        for micro in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(train_iter)
            inp, labels = batch[:, :-1], batch[:, 1:]
            # DDP only all-reduces the final micro-batch; this keeps the same
            # accumulated gradient while avoiding one full communication round.
            sync_context = nullcontext() if micro == GRADIENT_ACCUMULATE_EVERY - 1 else model.no_sync()
            with sync_context:
                # ★ 单次前向：logits 同时用于 loss 和 acc
                logits = model(inp, return_loss=False)
                loss = F.cross_entropy(logits.transpose(1, 2), labels)
                (loss / GRADIENT_ACCUMULATE_EVERY).backward()
            with torch.no_grad():
                acc_correct += (logits.argmax(dim=-1) == labels).sum()
                acc_total += labels.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step(); optim.zero_grad(set_to_none=True)
        train_loss = loss.detach().item() if need_metrics else None
        train_acc = acc_correct.item() / max(acc_total, 1) if need_metrics else None

        if should_log:
            print(f'training loss: {train_loss:.4f}  acc: {train_acc:.4f}  noise: {ns:.4f}', flush=True)
            gn = grad_norm.item()
            wandb.log(dict(loss=train_loss, acc=train_acc, grad_norm=gn, noise=ns), step=i)
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Accuracy/train', train_acc, i)
            writer.add_scalar('GradNorm', gn, i)
            writer.add_scalar('noise_scale', ns, i)

        # ── Validation（用全强度噪声 = 部署读条件）──────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            set_noise_scale(model, QUANT_NOISE_MAX, noise_modules)
            with torch.no_grad():
                val_batch = next(val_iter)
                val_inp, val_labels = val_batch[:, :-1], val_batch[:, 1:]
                val_logits = model(val_inp, return_loss=False)
                val_loss_val = F.cross_entropy(val_logits.transpose(1, 2), val_labels).item()
                val_acc = (val_logits.argmax(dim=-1) == val_labels).float().mean().item()
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
            model.eval(); set_noise_scale(model, QUANT_NOISE_MAX, noise_modules)
            inp = random.choice(val_dataset)[:PRIME_LENGTH].to(device)
            print(f'{decode_tokens(inp)} \n\n {"*" * 100}')
            m = model.module._orig_mod if hasattr(model.module, '_orig_mod') else model.module
            print(decode_tokens(m.sample(inp[None, ...], GENERATE_LENGTH,
                                         use_cache=USE_FAST_INFERENCE)[0]))

        if i > 0 and i % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optim, step=i, epoch=0,
                            path=CHECKPOINT_DIR / f'ckpt_step_{i:06d}.pt', best_val_loss=best_val_loss)
        # ★ 不再每步 dist.barrier()：DDP backward 已同步梯度，省掉全局同步开销

    save_checkpoint(model, optim, step=NUM_BATCHES, epoch=0,
                    path=CHECKPOINT_DIR / f'ckpt_step_{NUM_BATCHES:06d}_final.pt', best_val_loss=best_val_loss)
    if is_main_process() and writer is not None:
        writer.close()
        print('\n✓ v4.1-fast (QAT+curriculum) training complete.', flush=True)
        print(f'  Best val_loss: {best_val_loss:.4f}')
        print(f'  TensorBoard: tensorboard --logdir=runs_v4_1/ --port=6010')
    cleanup_distributed()

if __name__ == "__main__":
    main()
