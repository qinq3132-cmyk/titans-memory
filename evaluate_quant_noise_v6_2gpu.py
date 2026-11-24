"""
evaluate_quant_noise_v6_4gpu.py
-------------------------------
对 v3 checkpoint（train_mac_v3.py 训练，量化感知配置）进行多序列长度推理评估，
对比长期记忆（Neural Memory）推理阶段三种量化噪声配置下的性能：

  1. fp32_clean   — 不量化不加噪（全精度对照组）
  2. quant_only   — 4-bit 均匀量化，无噪声
  3. quant_noise  — 4-bit 均匀量化 + 5% 高斯噪声

4-GPU 并行版：将 3 variants × 6 seq_len = 18 个任务动态分配到 4 张 GPU，
每个 worker 按需加载对应 variant 模型（带缓存），batch_size=1 与 v2 评估一致。

每个 seq_len 下同时输出两种评估方式的结果：
  - full_seq   : 完整序列所有 token（与论文基准对齐）
  - second_half: 只统计后半段 token（前半段作预热，反映长程记忆能力）

输出：
  eval_quant_noise_v6/
    ├── quant_noise_v6_full_results.csv
    ├── quant_noise_v6_second_half_results.csv
    ├── quant_noise_v6_full_summary.txt
    ├── quant_noise_v6_second_half_summary.txt
    ├── quant_noise_v6_full_curves.png
    └── quant_noise_v6_second_half_curves.png

运行：
    python evaluate_quant_noise_v6_4gpu.py
"""

import os
import sys

# ── 4 卡 GPU 选择（0,3,4,5 为空闲卡，避开 1/2/6 上其他任务）──────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"
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
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH     = "/home/titans-pytorch-main/data/enwik8.gz"
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
NUM_EVAL_SEQS    = 200
NUM_GPUS         = 4
OUTPUT_DIR       = Path("./eval_quant_noise_v6")

# v3 checkpoint 路径
V3_CKPT = "./checkpoints_v6/best.pt"

# ── 三种量化噪声配置 ────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name":        "fp32_clean",
        "label":       "FP32 Clean (no quant)",
        "color":       "#2E86AB",
        "marker":      "o",
        "quant_noise_cfg": dict(enabled=False, quantized_storage=True, write_noise_scale=0.05),
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
            quantized_storage=True, write_noise_scale=0.05,
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
            quantized_storage=True, write_noise_scale=0.05,
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
# 模型加载（指定 GPU）
# ═══════════════════════════════════════════════════════════════════════════════

def load_v3_model(ckpt_path: str, quant_noise_cfg: dict, variant_name: str,
                  device: str):
    ref_dir = str(Path("/home/titans_v0.5/titans-memory/ref-project"))

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
        model.to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [GPU {device}] [{variant_name:15s}] loaded  ({params:.2f}M params)",
              flush=True)
        return model

    finally:
        sys.path.remove(ref_dir)
        for k in list(sys.modules.keys()):
            if k.startswith("titans_pytorch"):
                del sys.modules[k]
        sys.modules.update(saved_modules)


# ═══════════════════════════════════════════════════════════════════════════════
# 评估函数（batch_size=1，与 v2 评估方法一致）
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, val_data: torch.Tensor, seq_len: int,
                   num_seqs: int, variant_name: str, device: str):
    val_data_dev = val_data
    dataset    = SeqDataset(val_data_dev, seq_len, num_seqs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    half_start = seq_len // 2

    full_loss, full_correct, full_tokens, full_batches = 0.0, 0, 0, 0
    half_loss, half_correct, half_tokens = 0.0, 0, 0

    for batch in tqdm(dataloader, desc=f"    seq={seq_len}", leave=False):
        batch = batch.to(device)
        inp   = batch[:, :-1]
        tgt   = batch[:, 1:]

        try:
            logits = model(inp)
        except Exception as e:
            print(f"    [SKIP] {variant_name} seq={seq_len}: {e}", flush=True)
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
# GPU worker：动态拉取任务，按 variant 缓存模型
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_worker(rank: int, num_gpus: int,
               task_queue, result_queue, ckpt_path: str):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(f"  [worker {rank}] started on {device}", flush=True)

    val_data = load_val_data()

    model_cache = {}   # variant_idx -> model

    while True:
        task = task_queue.get()
        if task is None:
            break
        variant_idx, seq_len = task
        v = VARIANTS[variant_idx]
        name = v["name"]

        if variant_idx not in model_cache:
            model_cache[variant_idx] = load_v3_model(
                ckpt_path, v["quant_noise_cfg"], name, device=device)

        model = model_cache[variant_idx]
        mf, mh = evaluate_model(model, val_data, seq_len,
                                NUM_EVAL_SEQS, name, device=device)

        result_queue.put((variant_idx, seq_len, mf, mh))
        if mf[0] is not None:
            print(f"  [worker {rank}] done  {name:15s} seq={seq_len:>5d}"
                  f"  FULL BPC={mf[3]:.4f}  HALF BPC={mh[3]:.4f}", flush=True)
        else:
            print(f"  [worker {rank}] FAIL  {name:15s} seq={seq_len:>5d}", flush=True)

    for m in model_cache.values():
        del m
    torch.cuda.empty_cache()
    print(f"  [worker {rank}] exiting", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, variants: list, output_path: Path,
                 mode: str, mode_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"V6 TITANS: Neural Memory Quantization Noise Impact  [{mode_label}]\n"
        "(Inference-only quant+noise on dynamic weights, eval mode)\n"
        f"Checkpoint: {V3_CKPT}  |  4-GPU parallel  |  NUM_EVAL_SEQS={NUM_EVAL_SEQS}",
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
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 75)
    print("V6 TITANS — Neural Memory 量化加噪推理对比评估  (4-GPU 并行版)")
    print("=" * 75)
    print(f"  Checkpoint    : {V3_CKPT}")
    print(f"  GPU 数量      : {NUM_GPUS} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
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

    n_gpus = torch.cuda.device_count()
    print(f"\n检测到 {n_gpus} 张可用 GPU")
    num_gpus = min(NUM_GPUS, n_gpus)
    if n_gpus < NUM_GPUS:
        print(f"[WARN] 期望 {NUM_GPUS} 卡，实际 {n_gpus} 卡，将使用 {num_gpus} 卡")

    # ── 构建任务列表：18 = 3 variants × 6 seq_len ──
    # 按 seq_len 降序 + variant 轮转，使最大任务先被拉取、负载自均衡
    tasks = []
    for sl in sorted(SEQUENCE_LENGTHS, reverse=True):
        for vi in range(len(VARIANTS)):
            tasks.append((vi, sl))
    total_tasks = len(tasks)

    ctx = mp.get_context("spawn")
    task_queue   = ctx.Queue()
    result_queue = ctx.Queue()

    for t in tasks:
        task_queue.put(t)
    for _ in range(num_gpus):
        task_queue.put(None)

    print(f"\n启动 {num_gpus} 个 GPU worker，共 {total_tasks} 个任务...")
    t0 = time.time()

    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=gpu_worker,
            args=(rank, num_gpus, task_queue, result_queue, V3_CKPT),
        )
        p.start()
        procs.append(p)

    # ── 收集结果 ──
    raw_results = {}
    for _ in range(total_tasks):
        variant_idx, seq_len, mf, mh = result_queue.get()
        raw_results[(variant_idx, seq_len)] = (mf, mh)

    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f"\n所有 worker 完成，耗时 {elapsed:.1f}s ({elapsed/60:.2f} min)")

    # ── 组装结果 ──
    def empty_results():
        return {
            v["name"]: {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []}
            for v in VARIANTS
        }
    results_full = empty_results()
    results_half = empty_results()

    for vi, v in enumerate(VARIANTS):
        for sl in SEQUENCE_LENGTHS:
            mf, mh = raw_results.get((vi, sl), ((None,)*4, (None,)*4))
            if mf[0] is None:
                print(f"  [{v['name']:15s}] seq={sl} 跳过（forward 失败）")
                continue
            for res, m in ((results_full, mf), (results_half, mh)):
                res[v["name"]]["seq_lens"].append(sl)
                res[v["name"]]["loss"].append(m[0])
                res[v["name"]]["ppl"].append(m[1])
                res[v["name"]]["acc"].append(m[2])
                res[v["name"]]["bpc"].append(m[3])

            print(f"  [{v['name']:15s}] seq={sl:>5d}"
                  f"  FULL: BPC={mf[3]:.4f} Acc={mf[2]:.2f}%"
                  f"  |  HALF: BPC={mh[3]:.4f} Acc={mh[2]:.2f}%")

    # ── 保存 ──
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
                f.write(f"{title}\n")
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

    save_csv(results_full,  OUTPUT_DIR / "quant_noise_v6_full_results.csv")
    save_summary(results_full, OUTPUT_DIR / "quant_noise_v6_full_summary.txt",
                 "V6 TITANS 量化加噪对比（full_seq：全序列所有 token）")
    plot_results(results_full, VARIANTS,
                 OUTPUT_DIR / "quant_noise_v6_full_curves.png",
                 mode="full_seq",
                 mode_label="Full Sequence (all tokens)")

    save_csv(results_half,  OUTPUT_DIR / "quant_noise_v6_second_half_results.csv")
    save_summary(results_half, OUTPUT_DIR / "quant_noise_v6_second_half_summary.txt",
                 "V6 TITANS 量化加噪对比（second_half：后半段 token，前半段作预热）")
    plot_results(results_half, VARIANTS,
                 OUTPUT_DIR / "quant_noise_v6_second_half_curves.png",
                 mode="second_half",
                 mode_label="Second Half (1st half = warm-up context)")

    print_table(results_full, "全序列（full_seq）")
    print_table(results_half, "后半段（second_half）")

    print(f"\n✓ 完成。所有输出在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
