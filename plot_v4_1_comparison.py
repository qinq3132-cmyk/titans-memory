"""
plot_v4_1_comparison.py
------------------------
读取 v2 / v3 / v4.1 的正式评估 CSV（full_seq），画对比图：
  3 个子图（fp32_clean / quant_only / quant_noise），每个子图 3 条线（v2/v3/v4.1）= BPC vs seq_len。
直观看 v4.1（QAT+curriculum）是否在量化噪声下优于 v2/v3。

输出：docs/v4_1_vs_v2_v3_comparison.png
"""

import os, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_cjk)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
plt.rcParams["axes.unicode_minus"] = False

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

VERSIONS = [
    ("v2",     "eval_quant_noise_v2/quant_noise_v2_full_results.csv",     "#9b59b6", "o"),
    ("v3",     "eval_quant_noise_v3/quant_noise_v3_full_results.csv",     "#2E86AB", "s"),
    ("v4.1",   "eval_quant_noise_v4_1/quant_noise_v4_1_full_results.csv",  "#E84855", "^"),
]
CONFIGS = ["fp32_clean", "quant_only", "quant_noise"]
CONFIG_LABELS = {
    "fp32_clean":  "FP32 Clean（无量化，基线）",
    "quant_only":  "4-bit Quant Only（无噪声）",
    "quant_noise": "4-bit Quant + 5% Noise（★部署读条件）",
}

def read_csv(path):
    """{variant: {seq_len: bpc}}"""
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d.setdefault(row["variant"], {})[int(row["seq_len"])] = float(row["bpc"])
    return d

# ── 读取 ──
data = {}  # version -> {config -> {seq_len: bpc}}
for ver, path, _, _ in VERSIONS:
    raw = read_csv(path)
    data[ver] = raw

# ── 画图：3 子图（每个 config 一图，3 条 version 线）──
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("v4.1 (QAT+curriculum) vs v2/v3 — 量化噪声鲁棒性对比（full_seq BPC, 200 seqs）\n"
             "v4.1: 训练时注入 4-bit+5% 读噪声（curriculum 0→0.05）；v2/v3: 干净训练，仅 eval 量化",
             fontsize=13, fontweight="bold")

for ax, cfg in zip(axes, CONFIGS):
    for ver, _, color, marker in VERSIONS:
        m = data[ver].get(cfg, {})
        xs = sorted(m.keys())
        ys = [m[x] for x in xs]
        lw = 3.0 if ver == "v4.1" else 2.0
        ax.plot(xs, ys, color=color, marker=marker, linewidth=lw, markersize=9,
                label=f"{ver}", zorder=3 if ver=="v4.1" else 2)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("BPC (↓)", fontsize=11)
    ax.set_title(CONFIG_LABELS[cfg], fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQUENCE_LENGTHS)
    ax.set_xticklabels([str(x) for x in SEQUENCE_LENGTHS])

plt.tight_layout()
out1 = Path("docs/v4_1_vs_v2_v3_comparison.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved: {out1}")

# ── 第二张图：退化幅度（quant_noise - fp32_clean）% 对比 ──
fig, ax = plt.subplots(figsize=(9, 6))
for ver, _, color, marker in VERSIONS:
    fp = data[ver].get("fp32_clean", {})
    qn = data[ver].get("quant_noise", {})
    xs = sorted(set(fp) & set(qn))
    degr = [(qn[x] - fp[x]) / fp[x] * 100 for x in xs]
    ax.plot(xs, degr, color=color, marker=marker, linewidth=2.5, markersize=9, label=ver)
ax.set_xlabel("Sequence Length", fontsize=12)
ax.set_ylabel("量化噪声退化 (%)  (↓ 越好)", fontsize=12)
ax.set_title("量化噪声导致的 BPC 退化幅度对比\n(quant_noise 相对 fp32_clean 的 BPC 增幅)", fontsize=13, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale("log", base=2)
ax.set_xticks(SEQUENCE_LENGTHS)
ax.set_xticklabels([str(x) for x in SEQUENCE_LENGTHS])
plt.tight_layout()
out2 = Path("docs/v4_1_degradation_comparison.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved: {out2}")

# ── 打印关键数字 ──
print("\n=== quant_noise BPC（部署读条件）对比 ===")
print(f"{'seq_len':>8} │ {'v2':>8} {'v3':>8} {'v4.1':>8} │ v4.1-v3")
for sl in SEQUENCE_LENGTHS:
    v2 = data['v2'].get('quant_noise',{}).get(sl)
    v3 = data['v3'].get('quant_noise',{}).get(sl)
    v41 = data['v4.1'].get('quant_noise',{}).get(sl)
    if v2 and v3 and v41:
        print(f"{sl:>8} │ {v2:>8.4f} {v3:>8.4f} {v41:>8.4f} │ {v41-v3:+.4f}")

print("\n=== 退化% (quant_noise vs fp32_clean) ===")
print(f"{'seq_len':>8} │ {'v2':>8} {'v3':>8} {'v4.1':>8}")
for sl in SEQUENCE_LENGTHS:
    fp2,qn2 = data['v2'].get('fp32_clean',{}).get(sl), data['v2'].get('quant_noise',{}).get(sl)
    fp3,qn3 = data['v3'].get('fp32_clean',{}).get(sl), data['v3'].get('quant_noise',{}).get(sl)
    fp41,qn41 = data['v4.1'].get('fp32_clean',{}).get(sl), data['v4.1'].get('quant_noise',{}).get(sl)
    if all([fp2,qn2,fp3,qn3,fp41,qn41]):
        print(f"{sl:>8} │ {(qn2-fp2)/fp2*100:>7.2f}% {(qn3-fp3)/fp3*100:>7.2f}% {(qn41-fp41)/fp41*100:>7.2f}%")
