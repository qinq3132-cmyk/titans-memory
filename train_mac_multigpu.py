# /// script
# dependencies = [
#     "accelerate",
#     "adam-atan2-pytorch>=0.1.18",
#     "setuptools",
#     "titans-pytorch",
#     "tqdm",
#     "wandb"
# ]
# ///

"""
Multi-GPU DDP training script for MAC Transformer on enwik8.
Uses GPUs 4,5,6,7 (4× H100-80GB).

Launch command:
    torchrun --nproc_per_node=4 train_mac_multigpu.py
"""

import os

# ── Select GPUs (set before any CUDA call) ───────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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

import random
import gzip
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import tqdm

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
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
GRADIENT_ACCUMULATE_EVERY = 1   # effective global batch = 4 GPUs × 4 bs × 1 accum = 16 (same as single-GPU)
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# ── Neural memory related ───────────────────────────────────────────────────

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
NEURAL_MEM_SPEC_NORM_SURPRISES = True
NEURAL_MEM_STORE_WITH_LOOKAHEAD_VALUE = False

# ── Experiment related ───────────────────────────────────────────────────────

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac-4gpu - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False  # turn on to pipe experiment to cloud

# ── Perf related ─────────────────────────────────────────────────────────────

USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = True
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

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # ── wandb (only on rank 0) ───────────────────────────────────────────
    if is_main_process():
        import wandb
        wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
        wandb.run.name = RUN_NAME
        wandb.run.save()

    # ── Memory model ─────────────────────────────────────────────────────
    if USE_MEM_ATTENTION_MODEL:
        neural_memory_model = MemoryAttention(dim=64)
    else:
        neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)

    # ── Model ────────────────────────────────────────────────────────────
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
            store_with_lookahead_value=NEURAL_MEM_STORE_WITH_LOOKAHEAD_VALUE
        )
    ).to(device)

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

    # NOTE: we do NOT use cycle() with DistributedSampler directly because
    # the sampler must call set_epoch() each epoch for proper shuffling.
    # Instead we rebuild the iterator when exhausted.

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

    # ── Training loop ────────────────────────────────────────────────────
    pbar = tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training',
                     disable=not is_main_process())

    for i in pbar:
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_iter), return_loss=True)
            # scale loss by grad-accum steps so that the averaged gradient
            # magnitude stays the same regardless of accumulation count
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if is_main_process():
            print(f'training loss: {loss.item():.4f}')
            import wandb
            wandb.log(dict(loss=loss.item()))

        # ── Validation ───────────────────────────────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = model(next(val_iter), return_loss=True)
            if is_main_process():
                print(f'validation loss: {val_loss.item():.4f}')

        # ── Generation ───────────────────────────────────────────────
        if SHOULD_GENERATE and i % GENERATE_EVERY == 0 and is_main_process():
            model.eval()
            inp = random.choice(val_dataset)[:PRIME_LENGTH].to(device)
            prime = decode_tokens(inp)
            print(f'{prime} \n\n {"*" * 100}')

            # sample from the unwrapped model (no DDP wrapper)
            sample = model.module.sample(
                inp[None, ...], GENERATE_LENGTH, use_cache=USE_FAST_INFERENCE
            )
            output_str = decode_tokens(sample[0])
            print(output_str)

        # sync all ranks before next step
        dist.barrier()

    # ── Cleanup ──────────────────────────────────────────────────────────
    cleanup_distributed()

if __name__ == "__main__":
    main()
