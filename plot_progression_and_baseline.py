"""
plot_progression_and_baseline.py
--------------------------------
图1：一步步加噪量化的演进 — v2/v3/v4.1/v5/v6 的 quant_noise BPC（部署读条件）。
图2：有无 Neural Memory（Titans）结构对比 — 标准Transformer(崩溃) vs Titans FP32(理想) vs Titans v6(忆阻器部署)。
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
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                d[int(row["seq_len"])] = float(row["bpc"])
    return d

def read_baseline(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[int(row["seq_len"])] = float(row["bpc"])
    return d

# ═══════════════════════════════════════════════════════════════════════════
# 图1：一步步加噪量化演进
# ═══════════════════════════════════════════════════════════════════════════
PROG = [
    ("v2  干净训练",            read_variant("eval_quant_noise_v2/quant_noise_v2_full_results.csv", "quant_noise"),     "#7f8c8d", "o"),
    ("v3  干净训练(同v2)",      read_variant("eval_quant_noise_v3/quant_noise_v3_full_results.csv", "quant_noise"),     "#2E86AB", "s"),
    ("v4.1  +QAT+curriculum",   read_variant("eval_quant_noise_v4_1/quant_noise_v4_1_full_results.csv", "quant_noise"), "#F18F01", "^"),
    ("v5  +量化存储(4bit cell)", read_variant("eval_quant_noise_v5/quant_noise_v5_full_results.csv", "quant_noise"),   "#9b59b6", "D"),
    ("v6  +写噪声",             read_variant("eval_quant_noise_v6/quant_noise_v6_full_results.csv", "quant_noise"),     "#E84855", "o"),
]

fig, (ax1, ax2p) = plt.subplots(1, 2, figsize=(20, 6),
                                gridspec_kw={"width_ratios": [1.3, 1]})

# 左：BPC vs seq_len
for label, d, color, mk in PROG:
    xs = sorted(d); ys = [d[x] for x in xs]
    lw = 3.0 if "v4.1" in label or "v6" in label else 2.0
    ax1.plot(xs, ys, color=color, marker=mk, linewidth=lw, markersize=9, label=label)
ax1.set_xlabel("Sequence Length", fontsize=12); ax1.set_ylabel("quant_noise BPC (↓)", fontsize=12)
ax1.set_title("一步步加噪量化的效果演进（quant_noise BPC = 部署读条件）", fontsize=12.5, fontweight="bold")
ax1.legend(fontsize=10, loc="upper right"); ax1.grid(True, alpha=0.3)
ax1.set_xscale("log", base=2); ax1.set_xticks(SEQ); ax1.set_xticklabels([str(x) for x in SEQ])
# 标注关键跃迁 v3→v4.1
ax1.annotate("QAT+curriculum\n15%→<3% 退化\n(关键跃迁)", xy=(2048, 1.485), xytext=(120, 1.62),
             fontsize=9.5, color="#B00000", fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="->", color="#B00000", lw=1.5))
ax1.annotate("v5/v6: 量化存储+写噪声\n几乎免费(≈v4.1)", xy=(4096, 1.416), xytext=(4096, 1.55),
             fontsize=9, color="#7f8c8d", ha="center",
             arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2))

# 右：@2048 的 BPC 柱状（直观看演进）
labels = [p[0].split()[0] for p in PROG]   # v2,v3,v4.1,v5,v6
vals = [p[1].get(2048) for p in PROG]
colors = [p[2] for p in PROG]
bars = ax2p.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
for b, v in zip(bars, vals):
    ax2p.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2p.set_ylabel("quant_noise BPC @2048 (↓)", fontsize=12)
ax2p.set_title("@2048 BPC 演进（v3→v4.1 大降，v5/v6 持平）", fontsize=12.5, fontweight="bold")
ax2p.grid(True, alpha=0.3, axis="y"); ax2p.set_ylim(1.35, 1.62)

plt.tight_layout()
out1 = Path("docs/progression_quant_noise.png")
plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close()
print(f"saved: {out1}")

# ═══════════════════════════════════════════════════════════════════════════
# 图2：有无 Titans 结构对比
# ═══════════════════════════════════════════════════════════════════════════
baseline = read_baseline("eval_quant_noise_v3/baseline_full_results.csv")           # 标准 Transformer FP32
titans_fp32 = read_variant("eval_quant_noise_v3/quant_noise_v3_full_results.csv", "fp32_clean")  # Titans FP32 理想
v6_qn = read_variant("eval_quant_noise_v6/quant_noise_v6_full_results.csv", "quant_noise")      # Titans v6 忆阻器部署

fig, ax = plt.subplots(figsize=(11, 7))
xs = sorted(baseline)
ax.plot(xs, [baseline[x] for x in xs], color="#888888", marker="D", linewidth=2.5, markersize=10,
        linestyle="--", label="标准 Transformer FP32（无 Neural Memory）")
ax.plot(sorted(titans_fp32), [titans_fp32[x] for x in sorted(titans_fp32)], color="#2E86AB", marker="o",
        linewidth=2.8, markersize=10, label="Titans FP32（数字基线，理想）")
ax.plot(sorted(v6_qn), [v6_qn[x] for x in sorted(v6_qn)], color="#E84855", marker="^",
        linewidth=3.0, markersize=11, label="Titans v6 忆阻器部署（4-bit + 读写噪声，全非理想性）")

# 标注崩溃区
ax.axvspan(1024, 4096, alpha=0.08, color="#888888")
ax.text(2048, 4.5, "标准 Transformer\n≥1024 崩溃区", ha="center", va="center",
        fontsize=11, color="#555555", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888", alpha=0.9))
# 标注 Titans 稳定
ax.annotate("Titans 全程稳定\n(v6 部署 ≈ FP32 理想，仅 +4.5%)", xy=(4096, 1.416), xytext=(1500, 1.9),
            fontsize=10.5, color="#B00000", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#B00000", lw=1.5))

ax.set_xlabel("Sequence Length", fontsize=13); ax.set_ylabel("BPC (↓, log scale)", fontsize=13)
ax.set_title("有无 Neural Memory（Titans）结构对比\n标准 Transformer ≥1024 崩溃；Titans 即使忆阻器全非理想性也稳定",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper left"); ax.grid(True, alpha=0.3, which="both")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(SEQ); ax.set_xticklabels([str(x) for x in SEQ])
ax.set_ylim(1.2, 7)

plt.tight_layout()
out2 = Path("docs/titans_vs_standard_transformer.png")
plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
print(f"saved: {out2}")

# 打印关键数字
print("\n=== 图1 演进 @2048 quant_noise BPC ===")
for label, d, _, _ in PROG:
    print(f"  {label:28s}: {d.get(2048):.4f}")
print("\n=== 图2 @2048 ===")
print(f"  标准 Transformer FP32: {baseline.get(2048):.4f}  (崩溃)")
print(f"  Titans FP32 理想     : {titans_fp32.get(2048):.4f}")
print(f"  Titans v6 部署       : {v6_qn.get(2048):.4f}  (+{(v6_qn[2048]-titans_fp32[2048])/titans_fp32[2048]*100:.1f}%)")
