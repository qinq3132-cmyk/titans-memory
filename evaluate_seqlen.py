"""
随序列长度变化的性能对比评估
比较 TITANS (MAC) vs Baseline Transformer

评估方式：
  - 对每个序列长度，把整段序列喂给模型
  - 只统计后半段 token 的 loss/acc（确保模型已经"看到"足够的历史）
  - 对比随序列加长，两者性能的变化趋势

运行：
    python evaluate_seqlen.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/opt/conda"
_nvidia_base = "/opt/conda/envs/titans/lib/python3.10/site-packages/nvidia"
_extra_libs = ":".join([
    "/opt/conda/lib",
    f"{_nvidia_base}/cuda_runtime/lib",
    f"{_nvidia_base}/cusparse/lib",
    f"{_nvidia_base}/cublas/lib",
    f"{_nvidia_base}/nvjitlink/lib",
])
os.environ["LD_LIBRARY_PATH"] = _extra_libs + (
    ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

import gzip
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotary_embedding_torch import RotaryEmbedding
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH     = "./data/enwik8.gz"
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
NUM_EVAL_SEQS    = 200          # 每个长度评估多少条序列
EVAL_LAST_N      = 64           # 只评估最后N个token（与之前实验保持一致）
OUTPUT_DIR       = Path("./eval_seqlen_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# TITANS_CKPT   = "./checkpoints/ckpt_step_097500.pt"
TITANS_CKPT   = "/home/titans_v0.5/titans-memory/ref-project/model_latest.pt"
BASELINE_CKPT = "./checkpoints_baseline/best.pt"

# ─────────────────────────────────────────────
# Baseline Transformer 定义（与训练脚本一致）
# ─────────────────────────────────────────────
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
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head  = dim // num_heads
        self.norm      = nn.LayerNorm(dim)
        self.to_qkv    = nn.Linear(dim, dim * 3, bias=False)
        self.to_out    = nn.Linear(dim, dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.dim_head)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.num_heads
        x_normed = self.norm(x)
        qkv = self.to_qkv(x_normed).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, self.dim_head).transpose(1, 2), qkv)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ff   = FeedForward(dim, ff_mult, dropout)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, num_tokens=256, dim=384, depth=8, num_heads=8, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers    = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult) for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_logits.weight = self.token_emb.weight

    def forward(self, x):
        x = self.token_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.to_logits(x)


# ─────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────
class SeqDataset(Dataset):
    """从验证集顺序切出固定长度的序列"""
    def __init__(self, data: torch.Tensor, seq_len: int, num_seqs: int):
        self.data    = data
        self.seq_len = seq_len
        # 从验证集均匀采样起始点，避免只看前面一段
        total_avail = len(data) - seq_len - 1
        step = max(1, total_avail // num_seqs)
        self.starts = [i * step for i in range(num_seqs) if i * step + seq_len + 1 <= len(data)]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return self.data[s : s + self.seq_len + 1].long()


def load_val_data():
    with gzip.open(DATASET_PATH) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, val = np.split(data, [int(90e6)])
    return torch.from_numpy(val)


# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────
def load_titans(ckpt_path):
    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=384,
        depth=8,
        segment_len=32,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(2, 4, 6),
        neural_memory_segment_len=4,
        neural_memory_batch_size=128,
        neural_mem_gate_attn_output=False,
        neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=False,   # 与训练时一致（train_mac_multigpu.py: USE_FLEX_ATTN=True）
        sliding_window_attn=True,
        neural_memory_model=MemoryMLP(dim=64, depth=2),
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
            store_with_lookahead_value=False,
        ),
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  TITANS loaded  ({params:.1f}M params)")
    return model


def load_baseline(ckpt_path):
    model = BaselineTransformer(num_tokens=256, dim=384, depth=8, num_heads=8).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Baseline loaded ({params:.1f}M params)")
    return model


# ─────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_data, seq_len, num_seqs, eval_last_n, model_name):
    """
    对给定序列长度，评估模型在最后 eval_last_n 个 token 上的 loss / acc。
    前面的 token 作为历史上下文，不计入统计。
    """
    dataset    = SeqDataset(val_data, seq_len, num_seqs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    eval_start = seq_len - eval_last_n             # 从这个位置开始统计（固定最后N个token）

    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0

    for batch in tqdm(dataloader, desc=f"  {model_name} seq={seq_len}", leave=False):
        batch = batch.to(DEVICE)          # (1, seq_len+1)
        inp   = batch[:, :-1]             # (1, seq_len)
        tgt   = batch[:, 1:]              # (1, seq_len)

        try:
            logits = model(inp)           # (1, seq_len, vocab)
        except Exception as e:
            print(f"  [SKIP] {model_name} seq={seq_len}: {e}")
            break

        # 只取后半段
        logits_eval = logits[:, eval_start:, :]   # (1, eval_len, vocab)
        tgt_eval    = tgt[:, eval_start:]          # (1, eval_len)

        loss = F.cross_entropy(
            logits_eval.reshape(-1, logits_eval.size(-1)),
            tgt_eval.reshape(-1),
        )
        total_loss    += loss.item()
        total_correct += (logits_eval.argmax(-1) == tgt_eval).sum().item()
        total_tokens  += tgt_eval.numel()

    if total_tokens == 0:
        return None, None, None

    avg_loss = total_loss / len(dataset)
    ppl      = math.exp(avg_loss)
    acc      = total_correct / total_tokens * 100
    bpc      = avg_loss / math.log(2)
    return avg_loss, ppl, acc, bpc


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("随序列长度变化的性能对比：TITANS vs Baseline")
    print("=" * 60)
    print(f"Device       : {DEVICE}")
    print(f"序列长度     : {SEQUENCE_LENGTHS}")
    print(f"每长度序列数 : {NUM_EVAL_SEQS}")
    print(f"只评估最后N : {EVAL_LAST_N} tokens")
    print("=" * 60)

    print("\n加载数据...")
    val_data = load_val_data()
    print(f"验证集大小: {len(val_data):,} bytes")

    print("\n加载模型...")
    titans   = load_titans(TITANS_CKPT)
    baseline = load_baseline(BASELINE_CKPT)

    results = {
        "titans":   {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []},
        "baseline": {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []},
    }

    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n──── seq_len = {seq_len} ────")

        for model_name, model in [("titans", titans), ("baseline", baseline)]:
            result = evaluate(model, val_data, seq_len, NUM_EVAL_SEQS, EVAL_LAST_N, model_name)
            if result[0] is None:
                print(f"  [{model_name}] 跳过（forward 失败）")
                continue
            avg_loss, ppl, acc, bpc = result
            results[model_name]["seq_lens"].append(seq_len)
            results[model_name]["loss"].append(avg_loss)
            results[model_name]["ppl"].append(ppl)
            results[model_name]["acc"].append(acc)
            results[model_name]["bpc"].append(bpc)
            print(f"  [{model_name:8}]  loss={avg_loss:.4f}  PPL={ppl:.3f}  BPC={bpc:.4f}  Acc={acc:.2f}%")

    torch.cuda.empty_cache()

    # ── 保存数值结果 ───────────────────────────────────
    summary_path = OUTPUT_DIR / "results.txt"
    with open(summary_path, "w") as f:
        f.write("seq_len,model,loss,ppl,bpc,acc\n")
        for model_name in ["titans", "baseline"]:
            for i, sl in enumerate(results[model_name]["seq_lens"]):
                f.write(f"{sl},{model_name},"
                        f"{results[model_name]['loss'][i]:.6f},"
                        f"{results[model_name]['ppl'][i]:.4f},"
                        f"{results[model_name]['bpc'][i]:.6f},"
                        f"{results[model_name]['acc'][i]:.4f}\n")
    print(f"\n数值结果已保存: {summary_path}")

    # ── 绘图 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TITANS vs Baseline: Performance vs Sequence Length\n"
                 f"(Evaluated on last {EVAL_LAST_N} tokens of each sequence)",
                 fontsize=13, fontweight="bold")

    styles = {
        "titans":   dict(color="#2E86AB", marker="o", linewidth=2.5, markersize=8, label="TITANS (MAC)"),
        "baseline": dict(color="#F18F01", marker="s", linewidth=2.5, markersize=8, label="Baseline Transformer"),
    }

    # Plot 1: BPC
    ax = axes[0]
    for name in ["titans", "baseline"]:
        r = results[name]
        if r["seq_lens"]:
            ax.plot(r["seq_lens"], r["bpc"], **styles[name])
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("BPC (↓)", fontsize=11)
    ax.set_title("Bits Per Character", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Plot 2: PPL
    ax = axes[1]
    for name in ["titans", "baseline"]:
        r = results[name]
        if r["seq_lens"]:
            ax.plot(r["seq_lens"], r["ppl"], **styles[name])
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Perplexity (↓)", fontsize=11)
    ax.set_title("Perplexity", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Plot 3: Accuracy
    ax = axes[2]
    for name in ["titans", "baseline"]:
        r = results[name]
        if r["seq_lens"]:
            ax.plot(r["seq_lens"], r["acc"], **styles[name])
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Accuracy % (↑)", fontsize=11)
    ax.set_title("Next-Character Accuracy", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "seqlen_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {fig_path}")

    # ── 终端打印汇总 ──────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'seq_len':>8} │ {'TITANS BPC':>10} {'Base BPC':>10} │ {'TITANS Acc':>10} {'Base Acc':>10}")
    print("-" * 70)
    all_lens = sorted(set(results["titans"]["seq_lens"]) | set(results["baseline"]["seq_lens"]))
    for sl in all_lens:
        t_bpc = results["titans"]["bpc"][results["titans"]["seq_lens"].index(sl)] \
                if sl in results["titans"]["seq_lens"] else float("nan")
        b_bpc = results["baseline"]["bpc"][results["baseline"]["seq_lens"].index(sl)] \
                if sl in results["baseline"]["seq_lens"] else float("nan")
        t_acc = results["titans"]["acc"][results["titans"]["seq_lens"].index(sl)] \
                if sl in results["titans"]["seq_lens"] else float("nan")
        b_acc = results["baseline"]["acc"][results["baseline"]["seq_lens"].index(sl)] \
                if sl in results["baseline"]["seq_lens"] else float("nan")
        better = "◀ TITANS" if t_bpc < b_bpc else ("◀ Base" if b_bpc < t_bpc else "")
        print(f"{sl:>8} │ {t_bpc:>10.4f} {b_bpc:>10.4f} │ {t_acc:>9.2f}% {b_acc:>9.2f}%  {better}")
    print("=" * 70)
    print("\n完成！")


if __name__ == "__main__":
    main()
