"""
plot_v6_comparison.py
---------------------
v3 / v4.1 / v5 / v6 四版本对比，3 配置 + 退化%。
v6 = v5 + 写噪声(编程噪声)；看写噪声是否加代价。
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
VERSIONS = [
    ("v3",   "eval_quant_noise_v3/quant_noise_v3_full_results.csv",     "#2E86AB", "s"),
    ("v4.1", "eval_quant_noise_v4_1/quant_noise_v4_1_full_results.csv",  "#F18F01", "^"),
    ("v5",   "eval_quant_noise_v5/quant_noise_v5_full_results.csv",     "#9b59b6", "D"),
    ("v6",   "eval_quant_noise_v6/quant_noise_v6_full_results.csv",     "#E84855", "o"),
]
CONFIGS = ["fp32_clean", "quant_only", "quant_noise"]
CONFIG_LABELS = {
    "fp32_clean":  "FP32 Clean（无量化，基线）",
    "quant_only":  "4-bit Quant Only（无噪声）",
    "quant_noise": "4-bit Quant + 5% Noise（★部署读条件）",
}

def read_csv(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d.setdefault(row["variant"], {})[int(row["seq_len"])] = float(row["bpc"])
    return d

data = {ver: read_csv(path) for ver, path, _, _ in VERSIONS}

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("v3/v4.1/v5/v6 对比 — 量化噪声鲁棒性（full_seq BPC, 200 seqs）\n"
             "v6=v5+写噪声；v5=v4.1+量化存储(4-bit cell)；v4.1=QAT+curriculum；v3=干净训练",
             fontsize=12.5, fontweight="bold")
for ax, cfg in zip(axes, CONFIGS):
    for ver, _, color, marker in VERSIONS:
        m = data[ver].get(cfg, {})
        xs = sorted(m.keys()); ys = [m[x] for x in xs]
        lw = 3.0 if ver == "v6" else 2.0
        ax.plot(xs, ys, color=color, marker=marker, linewidth=lw, markersize=9,
                label=ver, zorder=3 if ver == "v6" else 2)
    ax.set_xlabel("Sequence Length", fontsize=11); ax.set_ylabel("BPC (↓)", fontsize=11)
    ax.set_title(CONFIG_LABELS[cfg], fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xscale("log", base=2)
    ax.set_xticks(SEQ); ax.set_xticklabels([str(x) for x in SEQ])
plt.tight_layout()
out = Path("docs/v6_vs_v3_v4_1_v5_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"saved: {out}")

# 退化图
fig, ax = plt.subplots(figsize=(9, 6))
for ver, _, color, marker in VERSIONS:
    fp = data[ver].get("fp32_clean", {}); qn = data[ver].get("quant_noise", {})
    xs = sorted(set(fp) & set(qn)); degr = [(qn[x]-fp[x])/fp[x]*100 for x in xs]
    ax.plot(xs, degr, color=color, marker=marker, linewidth=2.5, markersize=9, label=ver)
ax.set_xlabel("Sequence Length", fontsize=12); ax.set_ylabel("量化噪声退化 % (↓越好)", fontsize=12)
ax.set_title("退化对比：写噪声(v6) vs 量化存储(v5) vs QAT(v4.1) vs 干净(v3)", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_xscale("log", base=2)
ax.set_xticks(SEQ); ax.set_xticklabels([str(x) for x in SEQ])
plt.tight_layout()
out2 = Path("docs/v6_degradation_comparison.png")
plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
print(f"saved: {out2}")
