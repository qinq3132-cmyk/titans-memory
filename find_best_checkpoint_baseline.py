"""
Scan all baseline Transformer checkpoints and find the one with lowest val loss / BPC.
Uses 50 random windows for a stable estimate, no text generation.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from rotary_embedding_torch import RotaryEmbedding

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_DIR         = Path('./checkpoints_baseline')
NUM_EVAL_WINDOWS = 50    # more windows → more stable estimate
SEQ_LEN          = 513   # 512 + 1 for target
SEED             = 42
# ────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)

# ── Baseline Transformer 模型定义（与 train_baseline_transformer.py 完全一致）──

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim_head  = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.dim_head)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.num_heads

        x_normed = self.norm(x)
        qkv = self.to_qkv(x_normed).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, self.dim_head).transpose(1, 2), qkv)

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff   = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        num_tokens = 256,
        dim        = 384,
        depth      = 8,
        num_heads  = 8,
        ff_mult    = 4,
        dropout    = 0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers    = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult, dropout)
            for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        # 权重绑定
        self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, return_loss=False):
        if return_loss:
            x, target = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.to_logits(x)

        if return_loss:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            return loss

        return logits


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
print("Building Baseline Transformer ...")
model = BaselineTransformer(
    num_tokens=256,
    dim=384,
    depth=8,
    num_heads=8,
    ff_mult=4,
    dropout=0.0,
).cuda()

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)\n")

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

    print(f"{ckpt_path.name:<45} {step:>8,} {avg_loss:>10.4f} {bpc:>8.4f}", flush=True)

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
