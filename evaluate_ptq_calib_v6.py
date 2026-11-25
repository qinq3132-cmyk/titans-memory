"""
evaluate_ptq_calib_v6.py
------------------------
实验 11：进阶 PTQ 方法对比——在全 4-bit 部署配置（实验 10 的 s7_all4bit，+8.8%）
上，比较 4 种 data-free PTQ 方法，看校准/进阶量化能把部署代价压掉多少。

四种方法（全部只作用于权重张量本身，不需要激活校准数据）：
  max     per-tensor 对称最大值            —— 实验 10 的基线方法
  mse     per-tensor + MSE 最优裁剪         —— 搜索裁剪比例 α∈[0.80,1.00]，
            每个张量独立选择最小化 ||Q(w)-w||² 的 α（对抗离群值）
  ch_max  per-channel（每个输出行独立 scale）—— 线性层标准做法；物理上对应
            忆阻器阵列每行共享一个参考电导/位线电压
  ch_mse  per-channel + MSE 最优裁剪        —— 预期最优

范围：全 4-bit 部署（nm/attn/ffn 全 4-bit，W_t 维持 v6 部署条件）。
norm / embedding / to_logits / mem-tokens 保持 fp32（与实验 10 一致）。

输出：eval_ptq_calib_v6/
运行：
    conda activate titans
    EVAL_SEQS=8 python evaluate_ptq_calib_v6.py     # 冒烟
    python evaluate_ptq_calib_v6.py                 # 完整（4 GPU ~30-40 min）
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
OUTPUT_DIR       = Path("./eval_ptq_calib_v6")

V6_CKPT = "./checkpoints_v6/best.pt"

WQ = dict(enabled=True, bit=4, noise_scale=0.05, noise_method="add", noise_range="max",
          quantized_storage=True, write_noise_scale=0.05)
CLEAN_W = dict(enabled=False, bit=4, noise_scale=0.0, noise_method="add", noise_range="max",
               quantized_storage=False, write_noise_scale=0.0)

NM_SUFFIX   = ('to_keys.weight', 'to_values.weight', 'to_queries.weight',
               'combine_heads.weight', 'to_attn_logits.weight')
ATTN_SUFFIX = ('to_qkv.weight', 'to_out.weight')
FFN_PAT     = re.compile(r'layers\.\d+\.6\.\d+\.weight$')

MSE_ALPHAS = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]

# ── 6 个 variant：全 4-bit 部署 × 4 种 PTQ 方法 + 2 个参照 ───────────────────
VARIANTS = [
    {"name": "s0_fp32_all", "label": "s0: all fp32 (clean)",
     "color": "#888888", "marker": "o",
     "quant_noise_cfg": {**CLEAN_W}, "backbone": {}, "ptq_method": None},
    {"name": "s1_wt_only", "label": "s1: W_t only (v6 deploy)",
     "color": "#2E86AB", "marker": "o",
     "quant_noise_cfg": {**WQ}, "backbone": {}, "ptq_method": None},
    {"name": "s7_max", "label": "s7 PTQ=max (exp10 baseline)",
     "color": "#E84855", "marker": "^",
     "quant_noise_cfg": {**WQ},
     "backbone": {"nm": 4, "attn": 4, "ffn": 4}, "ptq_method": "max"},
    {"name": "s7_mse", "label": "s7 PTQ=mse-clip",
     "color": "#F18F01", "marker": "v",
     "quant_noise_cfg": {**WQ},
     "backbone": {"nm": 4, "attn": 4, "ffn": 4}, "ptq_method": "mse"},
    {"name": "s7_ch_max", "label": "s7 PTQ=per-channel max",
     "color": "#20A39E", "marker": "s",
     "quant_noise_cfg": {**WQ},
     "backbone": {"nm": 4, "attn": 4, "ffn": 4}, "ptq_method": "ch_max"},
    {"name": "s7_ch_mse", "label": "s7 PTQ=per-channel + mse-clip",
     "color": "#7A5195", "marker": "D",
     "quant_noise_cfg": {**WQ},
     "backbone": {"nm": 4, "attn": 4, "ffn": 4}, "ptq_method": "ch_mse"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# PTQ 方法实现（全部 data-free，eval-only，无需 STE）
# ═══════════════════════════════════════════════════════════════════════════════

def _sym_quant(t, scale, bit):
    """对称均匀量化（eval-only，无 STE）。scale: 标量或可广播张量。"""
    q = torch.clamp(
        torch.round(t / scale),
        -(2**(bit - 1) - 1), (2**(bit - 1) - 1),
    ) * scale
    return q


def _quant_with_clip(w, bit, method, alpha):
    """按 method 对权重 w 量化，裁剪比例 alpha（1.0 = 不裁剪）。"""
    per_channel = method.startswith('ch_')
    if per_channel:
        th = w.abs().amax(dim=-1, keepdim=True) * alpha   # 每输出行独立阈值
        w_c = torch.clamp(w, -th, th)
    else:
        th = w.abs().amax() * alpha
        w_c = torch.clamp(w, -th, th)
    scale = (2.0 * th) / (2**bit - 2) + 1e-12
    return _sym_quant(w_c, scale, bit)


def _mse_clip_quant(w, bit, method):
    """搜索裁剪比例，选择最小化权重重建 MSE 的量化结果（data-free 校准）。"""
    best_q, best_err = None, None
    for alpha in MSE_ALPHAS:
        q = _quant_with_clip(w, bit, method, alpha)
        err = (q - w).pow(2).sum().item()
        if best_err is None or err < best_err:
            best_err, best_q = err, q
    return best_q


def quantize_weight_ptq(w, bit, method):
    if bit is None or bit <= 0 or method is None:
        return w
    if method == 'max':
        return _quant_with_clip(w, bit, 'max', 1.0)
    if method == 'mse':
        return _mse_clip_quant(w, bit, 'mse')
    if method == 'ch_max':
        return _quant_with_clip(w, bit, 'ch_max', 1.0)
    if method == 'ch_mse':
        return _mse_clip_quant(w, bit, 'ch_mse')
    raise ValueError(f"unknown PTQ method: {method}")


def apply_backbone_quant(model, spec: dict, method: str) -> int:
    """按 spec={'nm':bit,'attn':bit,'ffn':bit} + method 对冻结权重做 PTQ。"""
    if not spec:
        return 0
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
                p.data = quantize_weight_ptq(p.data, bit, method)
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
                  ptq_method, variant_name: str, device: str):
    ref_dir = str(Path(__file__).resolve().parent / "ref-project")

    saved_modules = {k: v for k, v in sys.modules.items()
                     if k.startswith("titans_pytorch") or k.startswith("quantization_noise")}
    for k in saved_modules:
        del sys.modules[k]
    # 注意：只插 ref-project——项目根的旧版 titans_pytorch（无 quant_noise_cfg）会遮蔽
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
        n_q = apply_backbone_quant(model, backbone_spec, ptq_method)
        model.to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [GPU {device}] [{variant_name:12s}] loaded ({params:.2f}M params, "
              f"{n_q} tensors PTQ[{ptq_method}])", flush=True)
        return model

    finally:
        sys.path.remove(ref_dir)
        for k in list(sys.modules.keys()):
            if k.startswith("titans_pytorch") or k.startswith("quantization_noise"):
                del sys.modules[k]
        sys.modules.update(saved_modules)


# ═══════════════════════════════════════════════════════════════════════════════
# 评估（口径一致）
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
                ckpt_path, v["quant_noise_cfg"], v["backbone"], v["ptq_method"],
                name, device=device)

        model = model_cache[variant_idx]
        mf, mh = evaluate_model(model, val_data, seq_len,
                                NUM_EVAL_SEQS, name, device=device)

        result_queue.put((variant_idx, seq_len, mf, mh))
        if mf[0] is not None:
            print(f"  [worker {rank}] done  {name:12s} seq={seq_len:>5d}"
                  f"  FULL BPC={mf[3]:.4f}  HALF BPC={mh[3]:.4f}", flush=True)
        else:
            print(f"  [worker {rank}] FAIL  {name:12s} seq={seq_len:>5d}", flush=True)

    for m in model_cache.values():
        del m
    torch.cuda.empty_cache()
    print(f"  [worker {rank}] exiting", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图 / 保存
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, variants: list, output_path: Path, mode_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"V6 TITANS: Advanced PTQ Methods on Full-4bit Deployment  [{mode_label}]\n"
        "(all-4bit backbone (nm/attn/ffn) + v6 W_t condition; data-free weight-only PTQ)\n"
        f"Checkpoint: {V6_CKPT}  |  NUM_EVAL_SEQS={NUM_EVAL_SEQS}",
        fontsize=11, fontweight="bold"
    )

    for ax, (metric, ylabel, title) in zip(axes, [
            ("bpc", "BPC (↓)", "Bits Per Character"),
            ("ppl", "Perplexity (↓)", "Perplexity"),
            ("acc", "Accuracy % (↑)", "Next-Character Accuracy")]):
        for v in variants:
            r = results[v["name"]]
            if not r["seq_lens"]:
                continue
            ax.plot(r["seq_lens"], r[metric],
                    color=v["color"], marker=v["marker"],
                    linewidth=2.2, markersize=7, label=v["label"])
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEQUENCE_LENGTHS)
        ax.set_xticklabels([str(x) for x in SEQUENCE_LENGTHS])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 75)
    print("V6 TITANS — 实验 11：进阶 PTQ 对比（全 4-bit 部署 × 4 种 data-free 方法）")
    print("=" * 75)
    print(f"  Checkpoint    : {V6_CKPT}")
    print(f"  GPU 数量      : {NUM_GPUS} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"  序列长度      : {SEQUENCE_LENGTHS}")
    print(f"  每长度序列数  : {NUM_EVAL_SEQS}")
    print(f"  输出目录      : {OUTPUT_DIR}/")
    print()
    for v in VARIANTS:
        print(f"    [{v['name']:12s}] method={v['ptq_method']}  backbone={v['backbone'] or '{}'}")
    print("=" * 75)

    if not Path(V6_CKPT).exists():
        print(f"\n[ERROR] checkpoint 不存在: {V6_CKPT}")
        return

    n_gpus = torch.cuda.device_count()
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
        p = ctx.Process(target=gpu_worker,
                        args=(rank, num_gpus, task_queue, result_queue, V6_CKPT))
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
                print(f"  [{v['name']:12s}] seq={sl} 跳过（forward 失败）")
                continue
            for res, m in ((results_full, mf), (results_half, mh)):
                res[v["name"]]["seq_lens"].append(sl)
                res[v["name"]]["loss"].append(m[0])
                res[v["name"]]["ppl"].append(m[1])
                res[v["name"]]["acc"].append(m[2])
                res[v["name"]]["bpc"].append(m[3])

            print(f"  [{v['name']:12s}] seq={sl:>5d}"
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
        col_w = 12
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
        col_w = 11
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

    save_csv(results_full,  OUTPUT_DIR / "ptq_full_results.csv")
    save_summary(results_full, OUTPUT_DIR / "ptq_full_summary.txt",
                 "V6 进阶 PTQ 对比（full_seq：全序列所有 token）")
    plot_results(results_full, VARIANTS,
                 OUTPUT_DIR / "ptq_full_curves.png",
                 mode_label="Full Sequence (all tokens)")

    save_csv(results_half,  OUTPUT_DIR / "ptq_second_half_results.csv")
    save_summary(results_half, OUTPUT_DIR / "ptq_second_half_summary.txt",
                 "V6 进阶 PTQ 对比（second_half：后半段 token，前半段作预热）")
    plot_results(results_half, VARIANTS,
                 OUTPUT_DIR / "ptq_second_half_curves.png",
                 mode_label="Second Half (1st half = warm-up context)")

    print_table(results_full, "全序列（full_seq）")
    print_table(results_half, "后半段（second_half）")

    print(f"\n✓ 完成。所有输出在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
