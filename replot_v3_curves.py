"""
replot_v3_curves.py
-------------------
从已保存的 CSV 重新绘制 v3 量化噪声图表，生成两种视图各两张：
  1. v3-only（3 配置：fp32_clean / quant_only / quant_noise）—— 可读三配置差异
  2. with_baseline（叠加 Baseline 标准 Transformer fp32 曲线）—— 展示无记忆网络崩溃对比

读取：
  eval_quant_noise_v3/quant_noise_v3_full_results.csv
  eval_quant_noise_v3/quant_noise_v3_second_half_results.csv
  eval_quant_noise_v3/baseline_full_results.csv
  eval_quant_noise_v3/baseline_second_half_results.csv

输出：
  quant_noise_v3_full_curves.png            (v3-only, 覆盖)
  quant_noise_v3_full_curves_with_baseline.png
  quant_noise_v3_second_half_curves.png    (v3-only, 覆盖)
  quant_noise_v3_second_half_curves_with_baseline.png

运行：
    python replot_v3_curves.py
"""

import os
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR       = Path("./eval_quant_noise_v3")
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

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


def read_csv(path: Path, variant_filter=None):
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["variant"]
            if variant_filter and name not in variant_filter:
                continue
            results.setdefault(name, {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []})
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


def plot(results, baseline, output_path: Path, mode_label: str, include_baseline: bool):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    title = f"V3 TITANS: Neural Memory Quantization Noise Impact  [{mode_label}]"
    if include_baseline:
        title += "\n+ Baseline Transformer (no Neural Memory) for comparison"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    plot_specs = [
        ("bpc", "BPC (↓)",        "Bits Per Character"),
        ("ppl", "Perplexity (↓)", "Perplexity"),
        ("acc", "Accuracy % (↑)", "Next-Character Accuracy"),
    ]

    for ax, (metric, ylabel, title_str) in zip(axes, plot_specs):
        for v in VARIANTS:
            r = results.get(v["name"], {})
            if not r.get("seq_lens"):
                continue
            ax.plot(r["seq_lens"], r[metric],
                    color=v["color"], marker=v["marker"],
                    linewidth=2.5, markersize=8, label=v["label"])
        if include_baseline and baseline and baseline.get("seq_lens"):
            ax.plot(baseline["seq_lens"], baseline[metric], **BASELINE_STYLE)
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        xticks = [sl for sl in SEQUENCE_LENGTHS if sl <= 4096]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {output_path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("重绘 v3 量化噪声图表（v3-only + with_baseline 两套）")
    print("=" * 60)

    v3_names = {v["name"] for v in VARIANTS}
    v3_full = read_csv(OUTPUT_DIR / "quant_noise_v3_full_results.csv", v3_names)
    v3_half = read_csv(OUTPUT_DIR / "quant_noise_v3_second_half_results.csv", v3_names)
    bl_full = read_csv(OUTPUT_DIR / "baseline_full_results.csv").get("baseline", {})
    bl_half = read_csv(OUTPUT_DIR / "baseline_second_half_results.csv").get("baseline", {})

    print("\n[v3-only]")
    plot(v3_full, None, OUTPUT_DIR / "quant_noise_v3_full_curves.png",
         "Full Sequence (all tokens)", include_baseline=False)
    plot(v3_half, None, OUTPUT_DIR / "quant_noise_v3_second_half_curves.png",
         "Second Half (1st half = warm-up context)", include_baseline=False)

    print("\n[with_baseline]")
    plot(v3_full, bl_full, OUTPUT_DIR / "quant_noise_v3_full_curves_with_baseline.png",
         "Full Sequence (all tokens)", include_baseline=True)
    plot(v3_half, bl_half, OUTPUT_DIR / "quant_noise_v3_second_half_curves_with_baseline.png",
         "Second Half (1st half = warm-up context)", include_baseline=True)

    print(f"\n✓ 完成。共 4 张图，位于: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
