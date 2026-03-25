"""
Scan all checkpoints and find the one with lowest val loss / BPC.
Uses 50 random windows for a stable estimate, no text generation.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

import gzip
import sys
import numpy as np
import torch
from pathlib import Path

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_DIR       = Path('./checkpoints')
NUM_EVAL_WINDOWS = 50    # more windows → more stable estimate
SEQ_LEN        = 513
SEED           = 42
# ────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)

# ── Load validation data once ────────────────────────────────────────────────
print("Loading enwik8 validation data ...")
with gzip.open('./data/enwik8.gz') as f:
    data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
_, data_val = np.split(data, [int(90e6)])
data_val = torch.from_numpy(data_val).long()
print(f"Val data size: {data_val.size(0):,} bytes\n")

# ── Pre-sample fixed windows so every ckpt sees the same data ────────────────
max_start = data_val.size(0) - SEQ_LEN
starts = torch.randint(0, max_start, (NUM_EVAL_WINDOWS,), generator=torch.Generator().manual_seed(SEED))
seqs = torch.stack([data_val[s:s + SEQ_LEN] for s in starts])  # (N, SEQ_LEN)

# ── Build model (same arch as training) ─────────────────────────────────────
print("Building model ...")
model = MemoryAsContextTransformer(
    num_tokens=256, dim=384, depth=8, segment_len=32,
    num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
    neural_memory_layers=(2, 4, 6),
    neural_memory_segment_len=4, neural_memory_batch_size=128,
    neural_mem_gate_attn_output=False, neural_mem_weight_residual=True,
    neural_memory_qkv_receives_diff_views=True,
    use_flex_attn=False,
    sliding_window_attn=True,
    neural_memory_model=MemoryMLP(dim=64, depth=2),
    neural_memory_kwargs=dict(
        dim_head=64, heads=4, attn_pool_chunks=True,
        qk_rmsnorm=True, momentum=True, momentum_order=1,
        default_step_transform_max_lr=1e-1, use_accelerated_scan=False,
        per_parameter_lr_modulation=True, spectral_norm_surprises=True,
        store_with_lookahead_value=False
    )
).cuda()

# ── Scan all checkpoints ─────────────────────────────────────────────────────
ckpts = sorted(CKPT_DIR.glob('ckpt_step_*.pt'))
if not ckpts:
    print(f"No checkpoints found in {CKPT_DIR}")
    sys.exit(1)

print(f"Found {len(ckpts)} checkpoints. Evaluating ...\n")
print(f"{'Checkpoint':<45} {'Step':>8} {'Val Loss':>10} {'BPC':>8}")
print("-" * 75)

results = []

for ckpt_path in ckpts:
    # load weights
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    step = ckpt.get('step', -1)
    del ckpt
    torch.cuda.empty_cache()

    model.eval()
    losses = []
    with torch.no_grad():
        for seq in seqs:
            inp = seq.unsqueeze(0).cuda()
            loss = model(inp, return_loss=True)
            losses.append(loss.item())

    avg_loss = float(np.mean(losses))
    bpc      = avg_loss / np.log(2)
    results.append((ckpt_path, step, avg_loss, bpc))

    # print immediately so progress is visible
    marker = ""
    print(f"{ckpt_path.name:<45} {step:>8,} {avg_loss:>10.4f} {bpc:>8.4f} {marker}", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x[2])  # sort by val loss ascending

best_path, best_step, best_loss, best_bpc = results[0]

print("\n" + "=" * 75)
print("RESULTS (sorted by val loss, best → worst)")
print("=" * 75)
print(f"{'Rank':<6} {'Checkpoint':<45} {'Step':>8} {'Val Loss':>10} {'BPC':>8}")
print("-" * 75)
for rank, (p, s, l, b) in enumerate(results, 1):
    tag = " ◀ BEST" if rank == 1 else (" ◀ TOP5" if rank <= 5 else "")
    print(f"{rank:<6} {p.name:<45} {s:>8,} {l:>10.4f} {b:>8.4f}{tag}")

print("\n" + "=" * 75)
print(f"  BEST CHECKPOINT : {best_path.name}")
print(f"  Step            : {best_step:,}")
print(f"  Val Loss        : {best_loss:.4f}")
print(f"  BPC             : {best_bpc:.4f}")
print("=" * 75)
