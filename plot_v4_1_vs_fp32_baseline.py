"""
plot_v4_1_vs_fp32_baseline.py
------------------------------
对比「忆阻器部署条件 (v4.1 QAT + 读噪声)」vs「FP32 数字基线 (干净训练模型在 FP32 推理)」。
  - FP32 基线 = v3 的 fp32_clean（干净训练 + 干净推理 = 理想数字实现）
  - 部署条件 = v4.1 的 quant_noise（QAT 训练 + 4-bit+5% 读噪声推理 = 忆阻器 CIM 部署）
  画 2 个面板：左=BPC 两条线 + 差距阴影；右=各 seq_len 的掉幅% 柱状图。
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_cjk)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
plt.rcParams["axes.unicode_minus"] = False

SEQ = [128, 256, 512, 1024, 2048, 4096]

def read_bpc(path, variant):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                d[int(row["seq_len"])] = float(row["bpc"])
    return d

# FP32 基线：v3 干净训练模型，fp32 推理（理想数字实现）
fp32 = read_bpc("eval_quant_noise_v3/quant_noise_v3_full_results.csv", "fp32_clean")
# 部署条件：v4.1 QAT 模型，4-bit+5% 读噪声推理（忆阻器 CIM）
v41 = read_bpc("eval_quant_noise_v4_1/quant_noise_v4_1_full_results.csv", "quant_noise")

xs = sorted(set(fp32) & set(v41))
fp_y = [fp32[x] for x in xs]
v41_y = [v41[x] for x in xs]
drop = [(v41_y[i] - fp_y[i]) / fp_y[i] * 100 for i in range(len(xs))]

# ── 画图 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6),
                                gridspec_kw={"width_ratios": [1.4, 1]})

# 左：BPC 两条线 + 差距阴影
ax1.plot(xs, fp_y,  color="#2E86AB", marker="o", linewidth=2.8, markersize=10,
         label="FP32 数字基线（v3 干净训练 + FP32 推理）", zorder=3)
ax1.plot(xs, v41_y, color="#E84855", marker="^", linewidth=2.8, markersize=10,
         label="忆阻器部署（v4.1 QAT + 4-bit/5% 读噪声）", zorder=3)
ax1.fill_between(xs, fp_y, v41_y, color="#E84855", alpha=0.12, label="部署损失（掉幅）")
# 标注每个点的掉幅
for x, fy, vy in zip(xs, fp_y, v41_y):
    ax1.annotate(f"+{(vy-fy)/fy*100:.1f}%", xy=(x, (fy+vy)/2), ha="center", va="center",
                 fontsize=9.5, color="#B00000", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E84855", alpha=0.85))
ax1.set_xlabel("Sequence Length", fontsize=12)
ax1.set_ylabel("BPC (↓)", fontsize=12)
ax1.set_title("忆阻器 CIM 部署 vs FP32 数字基线\n(v4.1 QAT + 读噪声推理  vs  v3 干净训练 + FP32 推理)",
              fontsize=12.5, fontweight="bold")
ax1.legend(fontsize=10, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_xscale("log", base=2)
ax1.set_xticks(SEQ); ax1.set_xticklabels([str(x) for x in SEQ])
ax1.set_ylim(min(min(fp_y), min(v41_y)) - 0.02, max(max(fp_y), max(v41_y)) + 0.03)

# 右：掉幅% 柱状图
colors = ["#E84855" if d > 0 else "#2E86AB" for d in drop]
bars = ax2.bar([str(x) for x in xs], drop, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
for b, d in zip(bars, drop):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, f"{d:.1f}%",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_xlabel("Sequence Length", fontsize=12)
ax2.set_ylabel("BPC 掉幅 %  (相对 FP32 基线)", fontsize=12)
ax2.set_title("各序列长度下的部署损失", fontsize=12.5, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim(0, max(drop) * 1.25)

plt.tight_layout()
out = Path("docs/v4_1_vs_fp32_baseline.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved: {out}")
print("\n=== 掉幅明细（v4.1 读噪声部署 vs v3 FP32 基线）===")
print(f"{'seq_len':>8} │ {'FP32基线':>9} │ {'v4.1部署':>9} │ {'掉幅':>8}")
for x, fy, vy, d in zip(xs, fp_y, v41_y, drop):
    print(f"{x:>8} │ {fy:>9.4f} │ {vy:>9.4f} │ {d:>7.2f}%")
avg = sum(drop)/len(drop)
print(f"{'平均':>8} │ {'':>9} │ {'':>9} │ {avg:>7.2f}%")
