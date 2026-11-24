"""
plot_v4_1_vs_fp32_with_baseline.py
----------------------------------
3 条线对比，BPC + Accuracy 两张图：
  1. FP32 Titans 数字基线（v3 干净训练 + FP32 推理）
  2. 忆阻器部署（v4.1 QAT + 4-bit/5% 读噪声推理）
  3. 标准 Transformer FP32（无 Neural Memory，干净训练 + FP32 推理）—— ≥1024 崩溃

BPC 图用 log y 轴（兼顾 Titans 的小差距与 baseline 的崩溃）；Acc 图线性 y。
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_cjk)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
plt.rcParams["axes.unicode_minus"] = False

SEQ = [128, 256, 512, 1024, 2048, 4096]

def read_variant(path, variant):
    bpc, acc = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                sl = int(row["seq_len"])
                bpc[sl] = float(row["bpc"])
                acc[sl] = float(row["acc_pct"])
    return bpc, acc

# 三组数据
titans_fp32_bpc, titans_fp32_acc = read_variant(
    "eval_quant_noise_v3/quant_noise_v3_full_results.csv", "fp32_clean")      # Titans FP32 数字基线
v41_mem_bpc, v41_mem_acc = read_variant(
    "eval_quant_noise_v4_1/quant_noise_v4_1_full_results.csv", "quant_noise")  # v4.1 忆阻器部署
baseline_bpc, baseline_acc = read_variant(
    "eval_quant_noise_v3/baseline_full_results.csv", "baseline")               # 标准 Transformer FP32

SERIES = [
    ("FP32 Titans 数字基线（v3 干净训练+FP32）", titans_fp32_bpc, titans_fp32_acc, "#2E86AB", "o", 2.8, 10),
    ("忆阻器部署（v4.1 QAT+读噪声）",            v41_mem_bpc,    v41_mem_acc,    "#E84855", "^", 2.8, 10),
    ("标准 Transformer FP32（无记忆）",          baseline_bpc,   baseline_acc,   "#888888", "D", 2.2, 8),
]

# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: BPC（log y）
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

xs_t = sorted(titans_fp32_bpc)
for label, bpc, _, color, mk, lw, ms in SERIES:
    xs = sorted(bpc)
    ys = [bpc[x] for x in xs]
    ax1.plot(xs, ys, color=color, marker=mk, linewidth=lw, markersize=ms, label=label)
# 标注 v4.1 vs Titans 的掉幅
for x in xs_t:
    if x in v41_mem_bpc and x in titans_fp32_bpc:
        d = (v41_mem_bpc[x] - titans_fp32_bpc[x]) / titans_fp32_bpc[x] * 100
        ax1.annotate(f"+{d:.1f}%", xy=(x, v41_mem_bpc[x]), xytext=(8, 12),
                     textcoords="offset points", fontsize=9, color="#B00000", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E84855", alpha=0.85))
ax1.set_xlabel("Sequence Length", fontsize=12)
ax1.set_ylabel("BPC (↓, log scale)", fontsize=12)
ax1.set_title("BPC：忆阻器部署 vs FP32 基线 vs 标准Transformer\n（标准Transformer 在 ≥1024 崩溃；Titans 全程稳定）",
              fontsize=12.5, fontweight="bold")
ax1.legend(fontsize=9.5, loc="upper left")
ax1.grid(True, alpha=0.3, which="both")
ax1.set_xscale("log", base=2); ax1.set_yscale("log")
ax1.set_xticks(SEQ); ax1.set_xticklabels([str(x) for x in SEQ])

# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: Accuracy（线性 y）
# ═══════════════════════════════════════════════════════════════════════════
for label, _, acc, color, mk, lw, ms in SERIES:
    xs = sorted(acc)
    ys = [acc[x] for x in xs]
    ax2.plot(xs, ys, color=color, marker=mk, linewidth=lw, markersize=ms, label=label)
# 标注 v4.1 vs Titans 的 acc 掉幅
for x in xs_t:
    if x in v41_mem_acc and x in titans_fp32_acc:
        d = titans_fp32_acc[x] - v41_mem_acc[x]
        ax2.annotate(f"−{d:.1f}%", xy=(x, v41_mem_acc[x]), xytext=(8, -16),
                     textcoords="offset points", fontsize=9, color="#B00000", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E84855", alpha=0.85))
ax2.set_xlabel("Sequence Length", fontsize=12)
ax2.set_ylabel("Next-Character Accuracy % (↑)", fontsize=12)
ax2.set_title("Accuracy：忆阻器部署 vs FP32 基线 vs 标准Transformer\n（标准Transformer ≥1024 暴跌；Titans 维持 69-73%）",
              fontsize=12.5, fontweight="bold")
ax2.legend(fontsize=9.5, loc="lower left")
ax2.grid(True, alpha=0.3)
ax2.set_xscale("log", base=2)
ax2.set_xticks(SEQ); ax2.set_xticklabels([str(x) for x in SEQ])
ax2.set_ylim(0, 80)

plt.tight_layout()
out = Path("docs/v4_1_vs_fp32_with_baseline.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved: {out}")

print("\n=== BPC 掉幅 (v4.1 部署 vs Titans FP32 基线) ===")
print(f"{'seq':>6} │ {'TitansFP32':>10} {'v4.1部署':>9} {'标准Trans':>9} │ 掉幅")
for x in SEQ:
    t = titans_fp32_bpc.get(x); v = v41_mem_bpc.get(x); b = baseline_bpc.get(x)
    if t and v:
        print(f"{x:>6} │ {t:>10.4f} {v:>9.4f} {b:>9.4f} │ +{(v-t)/t*100:.2f}%")

print("\n=== Acc 掉幅 (v4.1 部署 vs Titans FP32 基线) ===")
print(f"{'seq':>6} │ {'TitansFP32':>10} {'v4.1部署':>9} {'标准Trans':>9} │ 掉幅(pp)")
for x in SEQ:
    t = titans_fp32_acc.get(x); v = v41_mem_acc.get(x); b = baseline_acc.get(x)
    if t and v:
        print(f"{x:>6} │ {t:>9.2f}% {v:>8.2f}% {b:>8.2f}% │ -{t-v:.2f}pp")
