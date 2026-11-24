"""
eval_baseline_replot_v3.py
--------------------------
用 v3 评估口径（NUM_EVAL_SEQS=200，full_seq + second_half，同样的 val/SeqDataset）
重新评估 Baseline 标准 Transformer（无 Neural Memory），并把它的 fp32 曲线叠加到
v3 的两张量化噪声图上（quant_noise_v3_full_curves / second_half_curves）。

Baseline 无 Neural Memory，quant_noise 机制不适用，故只贡献一条 fp32 曲线作为参考。

运行：
    python eval_baseline_replot_v3.py
"""

import os
import sys

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
import csv
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

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH     = "/home/titans-pytorch-main/data/enwik8.gz"
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
NUM_EVAL_SEQS    = 200
OUTPUT_DIR       = Path("./eval_quant_noise_v3")

BASELINE_CKPT    = "./checkpoints_baseline/ckpt_step_100000_final.pt"

# v3 三种配置的绘图样式（与 evaluate_quant_noise_v3_4gpu.py 一致）
VARIANTS = [
    {"name": "fp32_clean",  "label": "FP32 Clean (no quant)",
     "color": "#2E86AB", "marker": "o"},
    {"name": "quant_only",  "label": "4-bit Quant Only (no noise)",
     "color": "#E84855", "marker": "^"},
    {"name": "quant_noise", "label": "4-bit Quant + Gaussian Noise (σ=5%)",
     "color": "#F18F01", "marker": "s"},
]

BASELINE_STYLE = dict(
    color="#222222", marker="D", linestyle="--",
    linewidth=2.0, markersize=7, label="Baseline Transformer (no Neural Memory)",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Transformer 定义（与 evaluate_seqlen.py / 训练脚本一致）
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.num_heads  = num_heads
        self.dim_head   = dim // num_heads
        self.norm       = nn.LayerNorm(dim)
        self.to_qkv     = nn.Linear(dim, dim * 3, bias=False)
        self.to_out     = nn.Linear(dim, dim, bias=False)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集（与 v3 评估完全一致）
# ═══════════════════════════════════════════════════════════════════════════════

class SeqDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int, num_seqs: int):
        self.data    = data
        self.seq_len = seq_len
        total_avail  = len(data) - seq_len - 1
        step = max(1, total_avail // num_seqs)
        self.starts = [i * step for i in range(num_seqs)
                       if i * step + seq_len + 1 <= len(data)]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return self.data[s : s + self.seq_len + 1].long()


def load_val_data() -> torch.Tensor:
    with gzip.open(DATASET_PATH) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, val = np.split(data, [int(90e6)])
    return torch.from_numpy(val)


# ═══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_baseline(ckpt_path: str):
    model = BaselineTransformer(num_tokens=256, dim=384, depth=8, num_heads=8)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Baseline loaded ({params:.2f}M params) from {ckpt_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 评估函数（与 v3 evaluate_model 一致：full_seq + second_half 一次 forward 同时统计）
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_baseline(model, val_data: torch.Tensor, seq_len: int, num_seqs: int):
    dataset    = SeqDataset(val_data, seq_len, num_seqs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    half_start = seq_len // 2

    full_loss, full_correct, full_tokens, full_batches = 0.0, 0, 0, 0
    half_loss, half_correct, half_tokens = 0.0, 0, 0

    for batch in tqdm(dataloader, desc=f"    baseline seq={seq_len}", leave=False):
        batch = batch.to(DEVICE)
        inp   = batch[:, :-1]
        tgt   = batch[:, 1:]

        try:
            logits = model(inp)
        except Exception as e:
            print(f"    [SKIP] baseline seq={seq_len}: {e}", flush=True)
            _none4 = (None, None, None, None)
            return _none4, _none4

        loss_full = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        full_loss    += loss_full.item()
        full_correct += (logits.argmax(-1) == tgt).sum().item()
        full_tokens  += tgt.numel()
        full_batches += 1

        logits_h = logits[:, half_start:, :]
        tgt_h     = tgt[:, half_start:]
        loss_half = F.cross_entropy(
            logits_h.reshape(-1, logits_h.size(-1)), tgt_h.reshape(-1))
        half_loss    += loss_half.item()
        half_correct += (logits_h.argmax(-1) == tgt_h).sum().item()
        half_tokens  += tgt_h.numel()

    if full_batches == 0:
        _none4 = (None, None, None, None)
        return _none4, _none4

    def to_metrics(total_loss, correct, tokens, batches):
        avg_loss = total_loss / batches
        return (avg_loss, math.exp(min(avg_loss, 20)),
                correct / tokens * 100.0, avg_loss / math.log(2))

    return (to_metrics(full_loss, full_correct, full_tokens, full_batches),
            to_metrics(half_loss, half_correct, half_tokens, full_batches))


# ═══════════════════════════════════════════════════════════════════════════════
# 读取 v3 CSV 结果
# ═══════════════════════════════════════════════════════════════════════════════

def read_v3_csv(path: Path):
    results = {v["name"]: {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []}
               for v in VARIANTS}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["variant"]
            if name not in results:
                continue
            results[name]["seq_lens"].append(int(row["seq_len"]))
            results[name]["loss"].append(float(row["loss"]))
            results[name]["ppl"].append(float(row["ppl"]))
            results[name]["bpc"].append(float(row["bpc"]))
            results[name]["acc"].append(float(row["acc_pct"]))
    for n in results:
        order = sorted(range(len(results[n]["seq_lens"])),
                       key=lambda i: results[n]["seq_lens"][i])
        for k in results[n]:
            results[n][k] = [results[n][k][i] for i in order]
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图（v3 三配置 + Baseline 叠加）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_with_baseline(v3_results, baseline, output_path: Path,
                        mode_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"V3 TITANS: Neural Memory Quantization Noise Impact  [{mode_label}]\n"
        "(Inference-only quant+noise on dynamic weights, eval mode)\n"
        f"V3 Checkpoint: checkpoints_v3/best.pt  |  Baseline: {os.path.basename(BASELINE_CKPT)}  "
        f"|  NUM_EVAL_SEQS={NUM_EVAL_SEQS}",
        fontsize=11, fontweight="bold"
    )

    plot_specs = [
        ("bpc", "BPC (↓)",        "Bits Per Character"),
        ("ppl", "Perplexity (↓)", "Perplexity"),
        ("acc", "Accuracy % (↑)", "Next-Character Accuracy"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, plot_specs):
        for v in VARIANTS:
            r = v3_results[v["name"]]
            if not r["seq_lens"]:
                continue
            ax.plot(r["seq_lens"], r[metric],
                    color=v["color"], marker=v["marker"],
                    linewidth=2.5, markersize=8, label=v["label"])
        if baseline["seq_lens"]:
            ax.plot(baseline["seq_lens"], baseline[metric], **BASELINE_STYLE)
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        xticks = [sl for sl in SEQUENCE_LENGTHS if sl <= 4096]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("Baseline 重评估 + v3 曲线叠加（full_seq / second_half）")
    print("=" * 75)
    print(f"  Baseline ckpt : {BASELINE_CKPT}")
    print(f"  Device        : {DEVICE}")
    print(f"  序列长度      : {SEQUENCE_LENGTHS}")
    print(f"  每长度序列数  : {NUM_EVAL_SEQS}")
    print(f"  v3 结果目录   : {OUTPUT_DIR}/")
    print("=" * 75)

    if not Path(BASELINE_CKPT).exists():
        print(f"\n[ERROR] Baseline checkpoint 不存在: {BASELINE_CKPT}")
        return

    print("\n加载验证数据...")
    val_data = load_val_data()
    print(f"  验证集大小: {len(val_data):,} bytes")

    print("\n加载 Baseline 模型...")
    model = load_baseline(BASELINE_CKPT)

    baseline_full = {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []}
    baseline_half = {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []}

    print("\n评估 Baseline（full_seq + second_half 同时统计）...")
    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n──── seq_len = {seq_len} ────")
        mf, mh = evaluate_baseline(model, val_data, seq_len, NUM_EVAL_SEQS)
        if mf[0] is None:
            print(f"  [baseline] seq={seq_len} 跳过（forward 失败）")
            continue
        for res, m in ((baseline_full, mf), (baseline_half, mh)):
            res["seq_lens"].append(seq_len)
            res["loss"].append(m[0])
            res["ppl"].append(m[1])
            res["acc"].append(m[2])
            res["bpc"].append(m[3])
        print(f"  [baseline] seq={seq_len:>5d}"
              f"  FULL: BPC={mf[3]:.4f} Acc={mf[2]:.2f}%"
              f"  |  HALF: BPC={mh[3]:.4f} Acc={mh[2]:.2f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 保存 baseline 结果 CSV
    for res, name in ((baseline_full, "baseline_full_results.csv"),
                      (baseline_half, "baseline_second_half_results.csv")):
        p = OUTPUT_DIR / name
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seq_len", "variant", "loss", "ppl", "bpc", "acc_pct"])
            for i, sl in enumerate(res["seq_lens"]):
                w.writerow([sl, "baseline",
                            f"{res['loss'][i]:.6f}", f"{res['ppl'][i]:.4f}",
                            f"{res['bpc'][i]:.6f}", f"{res['acc'][i]:.4f}"])
        print(f"  CSV 已保存: {p}")

    # 读取 v3 结果
    print("\n读取 v3 评估结果...")
    v3_full = read_v3_csv(OUTPUT_DIR / "quant_noise_v3_full_results.csv")
    v3_half = read_v3_csv(OUTPUT_DIR / "quant_noise_v3_second_half_results.csv")

    # 重绘两张图（叠加 baseline）
    print("\n重绘图表（叠加 Baseline 曲线）...")
    plot_with_baseline(v3_full, baseline_full,
                       OUTPUT_DIR / "quant_noise_v3_full_curves.png",
                       mode_label="Full Sequence (all tokens)")
    plot_with_baseline(v3_half, baseline_half,
                       OUTPUT_DIR / "quant_noise_v3_second_half_curves.png",
                       mode_label="Second Half (1st half = warm-up context)")

    # 终端汇总
    print("\n" + "=" * 60)
    print("Baseline BPC 对比（full_seq）")
    print(f"{'seq_len':>8} │ {'Baseline':>10} │ {'v3 fp32':>10} │ {'v3 q+noise':>10}")
    print("-" * 60)
    for i, sl in enumerate(baseline_full["seq_lens"]):
        b_bpc = baseline_full["bpc"][i]
        f_bpc = v3_full["fp32_clean"]["bpc"][i] if sl in v3_full["fp32_clean"]["seq_lens"] else float("nan")
        q_bpc = v3_full["quant_noise"]["bpc"][i] if sl in v3_full["quant_noise"]["seq_lens"] else float("nan")
        print(f"{sl:>8} │ {b_bpc:>10.4f} │ {f_bpc:>10.4f} │ {q_bpc:>10.4f}")
    print("=" * 60)

    print(f"\n✓ 完成。图表已更新（叠加 Baseline）于: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
