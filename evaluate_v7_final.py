"""
evaluate_backbone_quant_v6.py
-----------------------------
实验 10：冻结权重分级量化（部署阶梯 PTQ）——逐步把忆阻器部署范围从 NM 记忆
扩展到整个网络，量化每一步的性能代价。

背景：W_t（NM 记忆 MLP）是唯一需要在线写的部分，已完整验证（v3~v6）。
其余模块（FFN / Attention QKV/out / NM K/V,Q 投影）部署时冻结，可走
"推理 CIM"路线——本实验用 load 时 PTQ（对称均匀量化，与 W_t 同一量化器）
模拟这些权重驻留忆阻器阵列的效应，按模块分组逐步累加：

  8-bit 阶梯: s0_fp32_all → s1_wt_only → s2_+nm@8 → s4_+attn@8 → s6_+ffn@8
  4-bit 阶梯: s1_wt_only → s3_+nm@4 → s5_+attn@4 → s7_all4bit(完整部署)

注：本实验为 eval 侧 PTQ（不重训）。分组：
  nm   = to_keys/to_values/to_queries + combine_heads/to_attn_logits（NM 通路）
  attn = to_qkv / to_out
  ffn  = layers.N.6.*（FFN 两矩阵）
norm / embedding / to_logits / mem tokens 保持 fp32（标准 PTQ 惯例）。

输出：eval_backbone_quant_v6/（full_seq + second_half，口径与既有评估一致）

运行：
    conda activate titans
    EVAL_SEQS=8 python evaluate_backbone_quant_v6.py      # 冒烟
    python evaluate_backbone_quant_v6.py                  # 完整（4 GPU ~60 min）
"""

import os
import sys
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,3,4")
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
NUM_EVAL_SEQS    = int(os.environ.get("EVAL_SEQS", 200))
NUM_GPUS         = 4
OUTPUT_DIR       = Path("./eval_v7_final")

V6_CKPT = "./checkpoints_v7/ckpt_step_002000_final.pt"

# ── W_t 部署条件（v6）/ 全干净 ──────────────────────────────────────────────
WQ = dict(enabled=True, bit=4, noise_scale=0.05, noise_method="add", noise_range="max",
          quantized_storage=True, write_noise_scale=0.05)
CLEAN_W = dict(enabled=False, bit=4, noise_scale=0.0, noise_method="add", noise_range="max",
               quantized_storage=False, write_noise_scale=0.0)

# ── 分组匹配规则 ────────────────────────────────────────────────────────────
NM_SUFFIX   = ('to_keys.weight', 'to_values.weight', 'to_queries.weight',
               'combine_heads.weight', 'to_attn_logits.weight')
ATTN_SUFFIX = ('to_qkv.weight', 'to_out.weight')
FFN_PAT     = re.compile(r'layers\.\d+\.6\.\d+\.weight$')

# ── 8 个 variant：两条阶梯（8-bit / 4-bit）逐级累加 ─────────────────────────
VARIANTS = [
    {"name": "v7_fp32_all", "label": "v7 ckpt, all fp32 (clean baseline)",
     "color": "#888888", "marker": "o",
     "quant_noise_cfg": {**CLEAN_W}, "backbone": {}},
    {"name": "v7_wt_only", "label": "v7 ckpt: W_t only (deploy ref)",
     "color": "#2E86AB", "marker": "o",
     "quant_noise_cfg": {**WQ}, "backbone": {}},
    {"name": "v7_4bit_chmax", "label": "v7: backbone 4bit per-channel max (matched to QAT)",
     "color": "#E84855", "marker": "^",
     "quant_noise_cfg": {**WQ}, "backbone": {"nm": 4, "attn": 4, "ffn": 4, "method": "ch_max"}},
    {"name": "v7_4bit_chmse", "label": "v7: backbone 4bit ch_mse (deployment optimizer)",
     "color": "#F18F01", "marker": "s",
     "quant_noise_cfg": {**WQ}, "backbone": {"nm": 4, "attn": 4, "ffn": 4, "method": "ch_mse"}},
]


# ═══════════════════════════════════════════════════════════════════════════════
# 骨干权重的 load-time PTQ
# ═══════════════════════════════════════════════════════════════════════════════

def _quant_chmax(w, bit):
    """per-channel（每输出行）对称 max 量化 —— 与 V7 训练时的 fake-quant 量化器一致"""
    th = w.abs().amax(dim=-1, keepdim=True)
    scale = (2.0 * th) / (2**bit - 2) + 1e-12
    return torch.clamp(torch.round(w / scale), -(2**(bit-1)-1), (2**(bit-1)-1)) * scale


def _quant_ch_mse(w, bit):
    """per-channel + MSE 最优裁剪（实验 11 最优 data-free 方法）"""
    best_q, best_err = None, None
    for alpha in [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
        th = w.abs().amax(dim=-1, keepdim=True) * alpha
        w_c = torch.clamp(w, -th, th)
        scale = (2.0 * th) / (2**bit - 2) + 1e-12
        q = torch.clamp(torch.round(w_c / scale), -(2**(bit-1)-1), (2**(bit-1)-1)) * scale
        err = (q - w).pow(2).sum().item()
        if best_err is None or err < best_err:
            best_err, best_q = err, q
    return best_q


def apply_backbone_quant(model, spec: dict) -> int:
    method = spec.get('method', 'ch_max')
    fn = {'ch_max': _quant_chmax, 'ch_mse': _quant_ch_mse}[method]
    n_q = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            bit = None
            if ('nm' in spec) and name.endswith(NM_SUFFIX):
                bit = spec['nm']
            elif ('attn' in spec) and name.endswith(ATTN_SUFFIX):
                bit = spec['attn']
            elif ('ffn' in spec) and FFN_PAT.search(name):
                bit = spec['ffn']
            if bit and p.dim() == 2:
                p.data = fn(p.data, bit)
                n_q += 1
    return n_q


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集 / checkpoint / 模型加载
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


def load_v6_model(ckpt_path: str, quant_noise_cfg: dict, backbone_spec: dict,
                  variant_name: str, device: str):
    ref_dir = str(Path(__file__).resolve().parent / "ref-project")

    saved_modules = {k: v for k, v in sys.modules.items()
                     if k.startswith("titans_pytorch") or k.startswith("quantization_noise")}
    for k in saved_modules:
        del sys.modules[k]
    # 注意：只插入 ref-project（其内含修改版 titans_pytorch）。
    # 项目根目录也有一份旧版 titans_pytorch（无 quant_noise_cfg），插入根目录会遮蔽它。
    # quantization_noise 包由 quant_noise_mem 导入时自行注入项目根路径，无需在此处理。
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
        n_q = apply_backbone_quant(model, backbone_spec)
        model.to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [GPU {device}] [{variant_name:14s}] loaded ({params:.2f}M params, "
              f"{n_q} tensors backbone-quantized)", flush=True)
        return model

    finally:
        sys.path.remove(ref_dir)
        for k in list(sys.modules.keys()):
            if k.startswith("titans_pytorch") or k.startswith("quantization_noise"):
                del sys.modules[k]
        sys.modules.update(saved_modules)


# ═══════════════════════════════════════════════════════════════════════════════
# 评估函数（batch_size=1，口径一致）
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, val_data: torch.Tensor, seq_len: int,
                   num_seqs: int, variant_name: str, device: str):
    dataset    = SeqDataset(val_data, seq_len, num_seqs)
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
# GPU worker
# ═══════════════════════════════════════════════════════════════════════════════

def gpu_worker(rank: int, num_gpus: int,
               task_queue, result_queue, ckpt_path: str):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(f"  [worker {rank}] started on {device}", flush=True)

    val_data = load_val_data()

    model_cache = {}

    while True:
        task = task_queue.get()
        if task is None:
            break
        variant_idx, seq_len = task
        v = VARIANTS[variant_idx]
        name = v["name"]

        if variant_idx not in model_cache:
            model_cache[variant_idx] = load_v6_model(
                ckpt_path, v["quant_noise_cfg"], v["backbone"], name, device=device)

        model = model_cache[variant_idx]
        mf, mh = evaluate_model(model, val_data, seq_len,
                                NUM_EVAL_SEQS, name, device=device)

        result_queue.put((variant_idx, seq_len, mf, mh))
        if mf[0] is not None:
            print(f"  [worker {rank}] done  {name:14s} seq={seq_len:>5d}"
                  f"  FULL BPC={mf[3]:.4f}  HALF BPC={mh[3]:.4f}", flush=True)
        else:
            print(f"  [worker {rank}] FAIL  {name:14s} seq={seq_len:>5d}", flush=True)

    for m in model_cache.values():
        del m
    torch.cuda.empty_cache()
    print(f"  [worker {rank}] exiting", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图（按 8bit / 4bit 两条阶梯分面）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, variants: list, output_path: Path, mode_label: str):
    ladders = [
        ("8-bit ladder", [v for v in variants if v["name"] in
                          ("s0_fp32_all", "s1_wt_only", "s2_nm_q8", "s4_attn_q8", "s6_ffn_q8")]),
        ("4-bit ladder", [v for v in variants if v["name"] in
                          ("s1_wt_only", "s3_nm_q4", "s5_attn_q4", "s7_all4bit")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"V6 TITANS: Deployment Ladder PTQ (frozen weights)  [{mode_label}]\n"
        "(W_t: 4-bit+noise deploy condition; backbone groups quantized stepwise, load-time PTQ)\n"
        f"Checkpoint: {V6_CKPT}  |  NUM_EVAL_SEQS={NUM_EVAL_SEQS}",
        fontsize=11, fontweight="bold"
    )

    for ax, (metric, ylabel, title) in zip(axes, [
            ("bpc", "BPC (↓)", "Bits Per Character"),
            ("ppl", "Perplexity (↓)", "Perplexity"),
            ("acc", "Accuracy % (↑)", "Next-Character Accuracy")]):
        for ladder_name, vs in ladders:
            for v in vs:
                r = results[v["name"]]
                if not r["seq_lens"]:
                    continue
                ax.plot(r["seq_lens"], r[metric],
                        color=v["color"], marker=v["marker"],
                        linewidth=2.2, markersize=7,
                        linestyle="-" if "8bit" in ladder_name else "--",
                        label=f"[{ladder_name.split()[0]}] {v['label']}")
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEQUENCE_LENGTHS)
        ax.set_xticklabels([str(x) for x in SEQUENCE_LENGTHS])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 75)
    print("V6 TITANS — V7 最终评估：骨干 QAT 2k 步后的 4-bit 部署效果（对照 v6 PTQ 基线）")
    print("=" * 75)
    print(f"  Checkpoint    : {V6_CKPT}")
    print(f"  GPU 数量      : {NUM_GPUS} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"  序列长度      : {SEQUENCE_LENGTHS}")
    print(f"  每长度序列数  : {NUM_EVAL_SEQS}")
    print(f"  输出目录      : {OUTPUT_DIR}/")
    print()
    for v in VARIANTS:
        print(f"    [{v['name']:14s}] W_t=v6部署  backbone={v['backbone'] or '{}'}")
    print("=" * 75)

    if not Path(V6_CKPT).exists():
        print(f"\n[ERROR] checkpoint 不存在: {V6_CKPT}")
        return

    n_gpus = torch.cuda.device_count()
    print(f"\n检测到 {n_gpus} 张可用 GPU")
    num_gpus = min(NUM_GPUS, n_gpus)

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
            args=(rank, num_gpus, task_queue, result_queue, V6_CKPT),
        )
        p.start()
        procs.append(p)

    raw_results = {}
    for _ in range(total_tasks):
        variant_idx, seq_len, mf, mh = result_queue.get()
        raw_results[(variant_idx, seq_len)] = (mf, mh)

    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f"\n所有 worker 完成，耗时 {elapsed:.1f}s ({elapsed/60:.2f} min)")

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
                print(f"  [{v['name']:14s}] seq={sl} 跳过（forward 失败）")
                continue
            for res, m in ((results_full, mf), (results_half, mh)):
                res[v["name"]]["seq_lens"].append(sl)
                res[v["name"]]["loss"].append(m[0])
                res[v["name"]]["ppl"].append(m[1])
                res[v["name"]]["acc"].append(m[2])
                res[v["name"]]["bpc"].append(m[3])

            print(f"  [{v['name']:14s}] seq={sl:>5d}"
                  f"  FULL: BPC={mf[3]:.4f} Acc={mf[2]:.2f}%"
                  f"  |  HALF: BPC={mh[3]:.4f} Acc={mh[2]:.2f}%")

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
        col_w = 14
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
                for sl in all_lens:
                    row = []
                    for n in variant_names:
                        if sl in results[n]["seq_lens"]:
                            idx = results[n]["seq_lens"].index(sl)
                            row.append(f"{results[n][metric][idx]:>{col_w}.4f}")
                        else:
                            row.append(f"{'N/A':>{col_w}}")
                    f.write(f"{sl:>8} │ " + " │ ".join(row) + "\n")
                f.write(sep + "\n\n")
        print(f"  汇总文本已保存: {path}")

    def print_table(results, title):
        variant_names = [v["name"] for v in VARIANTS]
        col_w = 13
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

    save_csv(results_full,  OUTPUT_DIR / "backbone_quant_full_results.csv")
    save_summary(results_full, OUTPUT_DIR / "backbone_quant_full_summary.txt",
                 "V6 部署阶梯 PTQ（full_seq：全序列所有 token）")
    plot_results(results_full, VARIANTS,
                 OUTPUT_DIR / "backbone_quant_full_curves.png",
                 mode_label="Full Sequence (all tokens)")

    save_csv(results_half,  OUTPUT_DIR / "backbone_quant_second_half_results.csv")
    save_summary(results_half, OUTPUT_DIR / "backbone_quant_second_half_summary.txt",
                 "V6 部署阶梯 PTQ（second_half：后半段 token，前半段作预热）")
    plot_results(results_half, VARIANTS,
                 OUTPUT_DIR / "backbone_quant_second_half_curves.png",
                 mode_label="Second Half (1st half = warm-up context)")

    print_table(results_full, "全序列（full_seq）")
    print_table(results_half, "后半段（second_half）")

    print(f"\n✓ 完成。所有输出在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
