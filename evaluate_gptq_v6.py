"""
evaluate_gptq_v6.py
-------------------
实验 12：GPTQ（data-aware PTQ）——在全 4-bit 部署配置上做逐层误差补偿，
对比 data-free 的 ch_mse（实验 11）。

GPTQ 核心（Frantar et al. 2022）：
  - 目标是最小化层**输出**的重建误差（用输入激活的二阶矩 H = E[x x^T] 加权），
    而非权重本身的 MSE —— 权重误差沿数据不激动的方向是免费的
  - 逐列量化，每列的量化误差按 H^-1 的结构补偿到未量化的列
  - 可选 act_order：按 diag(H) 降序排列列，误差补偿更有效
  - 输出只是不一样的权重数值 → 忆阻器硬件零改动，换一套编程值即可

关键实施细节：**校准激活必须在部署条件下采集**——W_t 的 4-bit+5% 读噪声、
存储吸附、写噪声全部开启（与推理时一致），否则 Hessian 统计与实际输入
分布失配，补偿会失效。

流程：
  Phase A（主进程，单卡）：部署条件下跑 256 条校准序列，hook 各目标
           nn.Linear 的输入 → 累积 H → 逐层 GPTQ（per-channel 4-bit）→ 存盘
  Phase B（4 卡并行）：标准 200-seq 协议评估 4 个 variant：
           s0_fp32_all / s1_wt_only / s7_ch_mse（实验11最优，同口径复跑） /
           s7_gptq（本次主角）

输出：eval_gptq_v6/
运行：
    conda activate titans
    EVAL_SEQS=8 python evaluate_gptq_v6.py     # 冒烟
    python evaluate_gptq_v6.py                 # 完整
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
OUTPUT_DIR       = Path("./eval_gptq_v6")

V6_CKPT = "./checkpoints_v6/best.pt"

# ── GPTQ 设置 ───────────────────────────────────────────────────────────────
CALIB_SEQS     = int(os.environ.get("CALIB_SEQS", 256))   # 校准序列数（512 tokens/条）
CALIB_SEQ_LEN  = 512
GPTQ_BIT       = 4
GPTQ_DAMP      = 0.01        # Hessian 阻尼系数（对角均值比例）
GPTQ_ACT_ORDER = True        # 按 diag(H) 降序的列排序
MSE_ALPHAS     = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]

WQ = dict(enabled=True, bit=4, noise_scale=0.05, noise_method="add", noise_range="max",
          quantized_storage=True, write_noise_scale=0.05)
CLEAN_W = dict(enabled=False, bit=4, noise_scale=0.0, noise_method="add", noise_range="max",
               quantized_storage=False, write_noise_scale=0.0)

NM_SUFFIX   = ('to_keys.weight', 'to_values.weight', 'to_queries.weight',
               'combine_heads.weight', 'to_attn_logits.weight')
ATTN_SUFFIX = ('to_qkv.weight', 'to_out.weight')
FFN_PAT     = re.compile(r'layers\.\d+\.6\.\d+\.weight$')

VARIANTS = [
    {"name": "s0_fp32_all", "label": "s0: all fp32 (clean)",
     "color": "#888888", "marker": "o",
     "quant_noise_cfg": {**CLEAN_W}, "backbone": {}},
    {"name": "s1_wt_only", "label": "s1: W_t only (v6 deploy)",
     "color": "#2E86AB", "marker": "o",
     "quant_noise_cfg": {**WQ}, "backbone": {}},
    {"name": "s7_ch_mse", "label": "s7: ch_mse PTQ (exp11 best)",
     "color": "#F18F01", "marker": "^",
     "quant_noise_cfg": {**WQ}, "backbone": {"nm": 4, "attn": 4, "ffn": 4}},
    {"name": "s7_gptq", "label": "s7: GPTQ (data-aware)",
     "color": "#7A5195", "marker": "D",
     "quant_noise_cfg": {**WQ}, "backbone": {"nm": 4, "attn": 4, "ffn": 4}},
]


# ═══════════════════════════════════════════════════════════════════════════════
# 权重分组与 data-free 量化（ch_mse，用于非 GPTQ 覆盖的部分和对照组）
# ═══════════════════════════════════════════════════════════════════════════════

def group_of(name):
    if name.endswith(NM_SUFFIX):
        return 'nm'
    if name.endswith(ATTN_SUFFIX):
        return 'attn'
    if FFN_PAT.search(name):
        return 'ffn'
    return None


def _sym_quant(t, scale, bit):
    return torch.clamp(
        torch.round(t / scale),
        -(2**(bit - 1) - 1), (2**(bit - 1) - 1),
    ) * scale


def _quant_ch_mse(w, bit):
    """per-channel + MSE 最优裁剪（实验 11 最优方法，data-free）。"""
    best_q, best_err = None, None
    for alpha in MSE_ALPHAS:
        th = w.abs().amax(dim=-1, keepdim=True) * alpha
        w_c = torch.clamp(w, -th, th)
        scale = (2.0 * th) / (2**bit - 2) + 1e-12
        q = _sym_quant(w_c, scale, bit)
        err = (q - w).pow(2).sum().item()
        if best_err is None or err < best_err:
            best_err, best_q = err, q
    return best_q


# ═══════════════════════════════════════════════════════════════════════════════
# GPTQ 核心
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gptq_quantize_weight(W, H, bit=GPTQ_BIT, damp_ratio=GPTQ_DAMP,
                         act_order=GPTQ_ACT_ORDER):
    """
    W: (out, in) 原始权重；H: (in, in) 输入二阶矩统计 E[x x^T]。
    返回 (Q, info)：Q 为 GPTQ 量化后的权重（per-channel 4-bit + 误差补偿）。
    """
    W = W.clone().float()
    out_f, in_f = W.shape
    H = H.clone().float()

    # 死列保护
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # act_order：按 diag(H) 降序排布列（先排 H 再做 cholesky 链）
    perm = torch.arange(in_f, device=W.device)
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]

    # 阻尼 + cholesky 链（H_inv 的上三角分解，迭代补偿用它）
    damp = damp_ratio * torch.mean(torch.diag(H))
    H += torch.eye(in_f, device=W.device) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)

    # per-channel scale（来自原始 W，逐输出行）
    scale = W.abs().amax(dim=1, keepdim=True) * 2.0 / (2**bit - 2) + 1e-12
    q_max = 2**(bit - 1) - 1

    Q = torch.zeros_like(W)
    for col in range(in_f):
        w = W[:, col:col + 1]                       # (out, 1) 列向量，避免广播错误
        d = U[col, col]
        q = torch.clamp(torch.round(w / scale), -q_max, q_max) * scale   # (out, 1)
        Q[:, col:col + 1] = q
        err = (w - q) / d
        if col + 1 < in_f:
            W[:, col + 1:] -= err @ U[col, col + 1:].unsqueeze(0)
        W[:, col:col + 1] = q

    if act_order:
        invperm = torch.argsort(perm)
        Q = Q[:, invperm]

    info = dict(clip_ratio=1.0, dead_cols=int(dead.sum().item()))
    return Q, info


def collect_hessians_and_quantize(model, val_data: torch.Tensor,
                                  calib_seqs: int, calib_len: int, device: str):
    """
    在部署条件下（W_t 噪声开启）跑校准序列，hook 目标 nn.Linear 的输入，
    累积 H = sum x^T x，然后逐层 GPTQ。
    返回 {param_name: quantized_weight(cpu float32)} 与覆盖层数。
    """
    # 1. 找到目标模块（参数名匹配分组且模块是 nn.Linear）
    targets = {}   # param_name -> module
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            param_name = f"{mod_name}.weight"
            if group_of(param_name) is not None:
                targets[param_name] = mod
    print(f"  GPTQ 覆盖 {len(targets)} 个 Linear 层 "
          f"({sum(m.weight.numel() for m in targets.values())/1e6:.2f}M 参数)", flush=True)

    # 2. 注册 pre-forward hooks 累积 H
    Hs = {pn: torch.zeros(mod.weight.shape[1], mod.weight.shape[1],
                          device=device, dtype=torch.float32)
          for pn, mod in targets.items()}
    handles = []
    def make_hook(pn):
        def pre_hook(module, args):
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            Hs[pn] += x.T @ x
        return pre_hook
    for pn, mod in targets.items():
        handles.append(mod.register_forward_pre_hook(make_hook(pn)))

    # 3. 部署条件下跑校准序列
    calib_data = val_data[: calib_seqs * (calib_len + 1)]
    model.eval()
    t0 = time.time()
    for i in tqdm(range(calib_seqs), desc="    calib", leave=False):
        seq = calib_data[i * (calib_len + 1) : (i + 1) * (calib_len + 1)].long().unsqueeze(0).to(device)
        _ = model(seq)   # forward 全程 no-grad？hook 需要值，model.eval 下前向即可
    for h in handles:
        h.remove()
    print(f"  校准完成：{calib_seqs} 条 × {calib_len} tokens，耗时 {time.time()-t0:.1f}s",
          flush=True)

    # 4. 逐层 GPTQ
    Qs = {}
    t0 = time.time()
    for pn, mod in tqdm(targets.items(), desc="    gptq", leave=False):
        Q, info = gptq_quantize_weight(mod.weight.detach(), Hs[pn])
        Qs[pn] = Q.cpu()
    print(f"  GPTQ 完成，耗时 {time.time()-t0:.1f}s", flush=True)

    del Hs
    torch.cuda.empty_cache()
    return Qs, len(targets)


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
                  variant_name: str, device: str, gptq_weights=None):
    ref_dir = str(Path(__file__).resolve().parent / "ref-project")

    saved_modules = {k: v for k, v in sys.modules.items()
                     if k.startswith("titans_pytorch") or k.startswith("quantization_noise")}
    for k in saved_modules:
        del sys.modules[k]
    # 只插 ref-project（项目根的旧版 titans_pytorch 会遮蔽修改版）
    sys.path.insert(0, ref_dir)

    try:
        from titans_pytorch import MemoryAsContextTransformer, MemoryMLP
        from quantization_noise.quant_util import data_quantization

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

        # 骨干量化
        if variant_name == "s7_gptq":
            assert gptq_weights is not None, "gptq_weights 未提供"
            n_g = 0
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in gptq_weights:
                        p.data = gptq_weights[name].to(p.data.dtype)
                        n_g += 1
            print(f"  [{variant_name}] GPTQ 权重覆盖 {n_g} 层", flush=True)
        elif backbone_spec:
            n_q = 0
            with torch.no_grad():
                for name, p in model.named_parameters():
                    bit = None
                    if ('nm' in backbone_spec) and name.endswith(NM_SUFFIX):
                        bit = backbone_spec['nm']
                    elif ('attn' in backbone_spec) and name.endswith(ATTN_SUFFIX):
                        bit = backbone_spec['attn']
                    elif ('ffn' in backbone_spec) and FFN_PAT.search(name):
                        bit = backbone_spec['ffn']
                    if bit and p.dim() == 2:
                        p.data = _quant_ch_mse(p.data, bit)
                        n_q += 1
            print(f"  [{variant_name}] ch_mse 量化 {n_q} 张量", flush=True)

        model.to(device)
        model.eval()
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

def gpu_worker(rank: int, num_gpus: int, task_queue, result_queue,
               ckpt_path: str, gptq_path: str | None):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(f"  [worker {rank}] started on {device}", flush=True)

    gptq_weights = None
    if gptq_path and Path(gptq_path).exists():
        blob = torch.load(gptq_path, map_location="cpu")
        gptq_weights = blob["weights"]

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
                ckpt_path, v["quant_noise_cfg"], v["backbone"], name,
                device=device, gptq_weights=gptq_weights)

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
# 绘图 / 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, variants: list, output_path: Path, mode_label: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"V6 TITANS: GPTQ (data-aware PTQ) vs data-free ch_mse  [{mode_label}]\n"
        "(all-4bit backbone + v6 W_t condition; calibration under deployment noise)\n"
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
    print("V6 TITANS — 实验 12：GPTQ（data-aware PTQ）vs ch_mse")
    print("=" * 75)
    print(f"  Checkpoint    : {V6_CKPT}")
    print(f"  校准          : {CALIB_SEQS} 条 × {CALIB_SEQ_LEN} tokens（部署条件下采集）")
    print(f"  GPU 数量      : {NUM_GPUS} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"  每长度序列数  : {NUM_EVAL_SEQS}")
    print(f"  输出目录      : {OUTPUT_DIR}/")
    print("=" * 75)

    if not Path(V6_CKPT).exists():
        print(f"\n[ERROR] checkpoint 不存在: {V6_CKPT}")
        return

    device = "cuda:0"
    val = load_val_data()

    # ── Phase A：校准 + GPTQ（主进程一次完成）────────────────────────────────
    gptq_path = OUTPUT_DIR / "gptq_weights.pt"
    if (not Path(gptq_path).exists()) or os.environ.get("FORCE_CALIB"):
        print("\n[Phase A] 构建 v6 部署条件模型，采集校准激活并执行 GPTQ...")
        model = load_v6_model(V6_CKPT, {**WQ}, {}, "calib", device)
        Qs, n_cov = collect_hessians_and_quantize(
            model, val, CALIB_SEQS, CALIB_SEQ_LEN, device)
        torch.save({"weights": Qs, "bit": GPTQ_BIT}, gptq_path)
        print(f"  已保存 {n_cov} 层 GPTQ 权重 → {gptq_path}")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"\n[Phase A] 已有缓存: {gptq_path}（FORCE_CALIB=1 可强制重算）")

    # ── Phase B：标准协议评估 ────────────────────────────────────────────────
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

    print(f"\n[Phase B] 启动 {num_gpus} 个 GPU worker，共 {total_tasks} 个任务...")
    t0 = time.time()

    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(target=gpu_worker,
                        args=(rank, num_gpus, task_queue, result_queue,
                              V6_CKPT, str(gptq_path)))
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
        col_w = 12
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

    save_csv(results_full,  OUTPUT_DIR / "gptq_full_results.csv")
    save_summary(results_full, OUTPUT_DIR / "gptq_full_summary.txt",
                 "V6 GPTQ vs ch_mse（full_seq：全序列所有 token）")
    plot_results(results_full, VARIANTS,
                 OUTPUT_DIR / "gptq_full_curves.png",
                 mode_label="Full Sequence (all tokens)")

    save_csv(results_half,  OUTPUT_DIR / "gptq_second_half_results.csv")
    save_summary(results_half, OUTPUT_DIR / "gptq_second_half_summary.txt",
                 "V6 GPTQ vs ch_mse（second_half：后半段 token，前半段作预热）")
    plot_results(results_half, VARIANTS,
                 OUTPUT_DIR / "gptq_second_half_curves.png",
                 mode_label="Second Half (1st half = warm-up context)")

    print_table(results_full, "全序列（full_seq）")
    print_table(results_half, "后半段（second_half）")

    print(f"\n✓ 完成。所有输出在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
