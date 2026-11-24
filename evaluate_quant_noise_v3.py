"""
evaluate_quant_noise_v3.py
-------------------------
对 v3 checkpoint（train_mac_v3.py 训练，量化感知配置）进行多序列长度推理评估，
对比长期记忆（Neural Memory）推理阶段三种量化噪声配置下的性能：

  1. fp32_clean   — 不量化不加噪（全精度对照组）
  2. quant_only   — 4-bit 均匀量化，无噪声
  3. quant_noise  — 4-bit 均匀量化 + 5% 高斯噪声

每个 seq_len 下同时输出两种评估方式的结果：
  - full_seq   : 完整序列所有 token（与论文基准对齐）
  - second_half: 只统计后半段 token（前半段作预热，反映长程记忆能力）

输出：
  eval_quant_noise_v3/
    ├── quant_noise_v3_full_results.csv
    ├── quant_noise_v3_second_half_results.csv
    ├── quant_noise_v3_full_summary.txt
    ├── quant_noise_v3_second_half_summary.txt
    ├── quant_noise_v3_full_curves.png
    └── quant_noise_v3_second_half_curves.png

运行：
    python evaluate_quant_noise_v3.py
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH     = "/home/titans-pytorch-main/data/enwik8.gz"
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
NUM_EVAL_SEQS    = 200
OUTPUT_DIR       = Path("./eval_quant_noise_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

# v3 checkpoint 路径
V3_CKPT = "./checkpoints_v3/best.pt"

# ── 三种量化噪声配置 ────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name":        "fp32_clean",
        "label":       "FP32 Clean (no quant)",
        "color":       "#2E86AB",
        "marker":      "o",
        "quant_noise_cfg": dict(enabled=False),
    },
    {
        "name":        "quant_only",
        "label":       "4-bit Quant Only (no noise)",
        "color":       "#E84855",
        "marker":      "^",
        "quant_noise_cfg": dict(
            enabled=True,
            bit=4,
            noise_scale=0.0,
            noise_method="add",
            noise_range="max",
        ),
    },
    {
        "name":        "quant_noise",
        "label":       "4-bit Quant + Gaussian Noise (σ=5%)",
        "color":       "#F18F01",
        "marker":      "s",
        "quant_noise_cfg": dict(
            enabled=True,
            bit=4,
            noise_scale=0.05,
            noise_method="add",
            noise_range="max",
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# 数据集
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
# Checkpoint 加载工具
# ═══════════════════════════════════════════════════════════════════════════════

def smart_load_state_dict(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            for v in ckpt.values():
                if isinstance(v, dict) and any("weight" in k for k in v.keys()):
                    state_dict = v
                    break
            else:
                state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        for name, param in model.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].clone().contiguous()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [WARN] Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_v3_model(ckpt_path: str, quant_noise_cfg: dict, variant_name: str):
    ref_dir = str(Path(__file__).parent / "ref-project")

    saved_modules = {k: v for k, v in sys.modules.items()
                     if k.startswith("titans_pytorch")}
    for k in saved_modules:
        del sys.modules[k]
    sys.path.insert(0, ref_dir)

    try:
        from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

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
            use_flex_attn=False,
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
                quant_noise_cfg=quant_noise_cfg,
            ),
        )

        model = smart_load_state_dict(model, ckpt_path)
        model.to(DEVICE)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [{variant_name:15s}] loaded  ({params:.2f}M params)")
        return model

    finally:
        sys.path.remove(ref_dir)
        for k in list(sys.modules.keys()):
            if k.startswith("titans_pytorch"):
                del sys.modules[k]
        sys.modules.update(saved_modules)


# ═══════════════════════════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, val_data: torch.Tensor, seq_len: int,
                   num_seqs: int, variant_name: str):
    dataset    = SeqDataset(val_data, seq_len, num_seqs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    half_start = seq_len // 2

    full_loss, full_correct, full_tokens, full_batches = 0.0, 0, 0, 0
    half_loss, half_correct, half_tokens = 0.0, 0, 0

    for batch in tqdm(dataloader, desc=f"    seq={seq_len}", leave=False):
        batch = batch.to(DEVICE)
        inp   = batch[:, :-1]
        tgt   = batch[:, 1:]

        try:
            logits = model(inp)
        except Exception as e:
            print(f"    [SKIP] {variant_name} seq={seq_len}: {e}")
            _none4 = (None, None, None, None)
            return _none4, _none4

        loss_full = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )
        full_loss    += loss_full.item()
        full_correct += (logits.argmax(-1) == tgt).sum().item()
        full_tokens  += tgt.numel()
        full_batches += 1

        logits_h = logits[:, half_start:, :]
        tgt_h    = tgt[:, half_start:]
        loss_half = F.cross_entropy(
            logits_h.reshape(-1, logits_h.size(-1)),
            tgt_h.reshape(-1),
        )
        half_loss    += loss_half.item()
        half_correct += (logits_h.argmax(-1) == tgt_h).sum().item()
        half_tokens  += tgt_h.numel()

    if full_batches == 0:
        _none4 = (None, None, None, None)
        return _none4, _none4

    def to_metrics(total_loss, correct, tokens, batches):
        avg_loss = total_loss / batches
        return (
            avg_loss,
            math.exp(min(avg_loss, 20)),
            correct / tokens * 100.0,
            avg_loss / math.log(2),
        )

    metrics_full = to_metrics(full_loss, full_correct, full_tokens, full_batches)
    metrics_half = to_metrics(half_loss, half_correct, half_tokens, full_batches)
    return metrics_full, metrics_half


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, variants: list, output_path: Path, mode: str, mode_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"V3 TITANS: Neural Memory Quantization Noise Impact  [{mode_label}]\n"
        "(Inference-only quant+noise on dynamic weights, eval mode)\n"
        f"Checkpoint: {V3_CKPT}",
        fontsize=11, fontweight="bold"
    )

    plot_specs = [
        ("bpc",  "BPC (↓)",        "Bits Per Character"),
        ("ppl",  "Perplexity (↓)", "Perplexity"),
        ("acc",  "Accuracy % (↑)", "Next-Character Accuracy"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, plot_specs):
        for v in variants:
            name = v["name"]
            r = results[name]
            if not r["seq_lens"]:
                continue
            ax.plot(
                r["seq_lens"], r[metric],
                color=v["color"], marker=v["marker"],
                linewidth=2.5, markersize=8,
                label=v["label"],
            )
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
    print("V3 TITANS — Neural Memory 量化加噪推理对比评估")
    print("=" * 75)
    print(f"  Checkpoint    : {V3_CKPT}")
    print(f"  Device        : {DEVICE}")
    print(f"  序列长度      : {SEQUENCE_LENGTHS}")
    print(f"  每长度序列数  : {NUM_EVAL_SEQS}")
    print(f"  评估范围      : full_seq(全序列) + second_half(后半段) 同时输出")
    print(f"  输出目录      : {OUTPUT_DIR}/")
    print()
    print("  三种配置:")
    for v in VARIANTS:
        print(f"    [{v['name']:15s}] {v['label']}")
    print("=" * 75)

    if not Path(V3_CKPT).exists():
        print(f"\n[ERROR] V3 checkpoint 不存在: {V3_CKPT}")
        return

    print("\n加载验证数据...")
    val_data = load_val_data()
    print(f"  验证集大小: {len(val_data):,} bytes")

    print("\n加载模型（同一 checkpoint，三种量化噪声配置）...")
    models = {}
    for v in VARIANTS:
        models[v["name"]] = load_v3_model(V3_CKPT, v["quant_noise_cfg"], v["name"])

    def empty_results():
        return {
            v["name"]: {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []}
            for v in VARIANTS
        }
    results_full = empty_results()
    results_half = empty_results()

    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n──── seq_len = {seq_len} ────")
        for v in VARIANTS:
            name  = v["name"]
            model = models[name]
            mf, mh = evaluate_model(model, val_data, seq_len, NUM_EVAL_SEQS, name)

            if mf[0] is None:
                print(f"  [{name:15s}] 跳过（forward 失败）")
                continue

            for res, m in ((results_full, mf), (results_half, mh)):
                res[name]["seq_lens"].append(seq_len)
                res[name]["loss"].append(m[0])
                res[name]["ppl"].append(m[1])
                res[name]["acc"].append(m[2])
                res[name]["bpc"].append(m[3])

            print(f"  [{name:15s}]"
                  f"  FULL: BPC={mf[3]:.4f} Acc={mf[2]:.2f}%"
                  f"  |  HALF: BPC={mh[3]:.4f} Acc={mh[2]:.2f}%")

    for m in models.values():
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def save_csv(results, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq_len", "variant", "loss", "ppl", "bpc", "acc_pct"])
            for v in VARIANTS:
                name = v["name"]
                for i, sl in enumerate(results[name]["seq_lens"]):
                    writer.writerow([
                        sl, name,
                        f"{results[name]['loss'][i]:.6f}",
                        f"{results[name]['ppl'][i]:.4f}",
                        f"{results[name]['bpc'][i]:.6f}",
                        f"{results[name]['acc'][i]:.4f}",
                    ])
        print(f"  CSV 已保存: {path}")

    def save_summary(results, path, title):
        variant_names = [v["name"] for v in VARIANTS]
        col_w = 16
        all_lens = sorted(set().union(*(results[n]["seq_lens"] for n in variant_names)))
        with open(path, "w") as f:
            for metric, label in (("bpc", "BPC"), ("acc", "Accuracy %")):
                header = f"{'seq_len':>8} │ " + " │ ".join(f"{label+'_'+n:>{col_w}}" for n in variant_names)
                sep = "=" * len(header)
                f.write(sep + "\n")
                f.write(f"{title} — {label}\n")
                f.write(sep + "\n")
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                fmt = ".4f"
                for sl in all_lens:
                    row = []
                    for n in variant_names:
                        if sl in results[n]["seq_lens"]:
                            idx = results[n]["seq_lens"].index(sl)
                            row.append(f"{results[n][metric][idx]:>{col_w}{fmt}}")
                        else:
                            row.append(f"{'N/A':>{col_w}}")
                    f.write(f"{sl:>8} │ " + " │ ".join(row) + "\n")
                f.write(sep + "\n\n")
        print(f"  汇总文本已保存: {path}")

    def print_table(results, title):
        variant_names = [v["name"] for v in VARIANTS]
        col_w = 15
        sep = "─" * (8 + 3 + (col_w + 3) * len(variant_names))
        header = " │ ".join(f"{n:>{col_w}}" for n in variant_names)
        print(f"\n{'='*len(sep)}")
        print(f"{title}")
        for metric, label in (("bpc", "BPC (↓越好)"), ("acc", "Accuracy % (↑越好)")):
            print(f"\n{label}")
            print(f"{'seq_len':>8} │ {header}")
            print(sep)
            for sl in sorted(set().union(*(results[n]["seq_lens"] for n in variant_names))):
                vals = []
                for n in variant_names:
                    if sl in results[n]["seq_lens"]:
                        idx = results[n]["seq_lens"].index(sl)
                        vals.append(f"{results[n][metric][idx]:>{col_w}.4f}")
                    else:
                        vals.append(f"{'N/A':>{col_w}}")
                print(f"{sl:>8} │ {' │ '.join(vals)}")
        print("=" * len(sep))

    print("\n保存结果...")

    save_csv(results_full,  OUTPUT_DIR / "quant_noise_v3_full_results.csv")
    save_summary(results_full, OUTPUT_DIR / "quant_noise_v3_full_summary.txt",
                 "V3 TITANS 量化加噪对比（full_seq：全序列所有 token）")
    plot_results(results_full, VARIANTS,
                 OUTPUT_DIR / "quant_noise_v3_full_curves.png",
                 mode="full_seq",
                 mode_label="Full Sequence (all tokens)")

    save_csv(results_half,  OUTPUT_DIR / "quant_noise_v3_second_half_results.csv")
    save_summary(results_half, OUTPUT_DIR / "quant_noise_v3_second_half_summary.txt",
                 "V3 TITANS 量化加噪对比（second_half：后半段 token，前半段作预热）")
    plot_results(results_half, VARIANTS,
                 OUTPUT_DIR / "quant_noise_v3_second_half_curves.png",
                 mode="second_half",
                 mode_label="Second Half (1st half = warm-up context)")

    print_table(results_full, "全序列（full_seq）")
    print_table(results_half, "后半段（second_half）")

    print(f"\n✓ 完成。所有输出在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
