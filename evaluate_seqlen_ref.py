"""
随序列长度变化的性能对比评估（加载参考项目checkpoint版本）
比较：
  - TITANS (ref-project checkpoint, 使用原始 HyperConnections)
  - TITANS (当前 checkpoint, 使用 MC HyperConnections)
  - Baseline Transformer

评估方式：
  - 对每个序列长度，把整段序列喂给模型
  - 只统计后半段 token 的 loss/acc（确保模型已经"看到"足够的历史）
  - 对比随序列加长，三者性能的变化趋势

运行：
    python evaluate_seqlen_ref.py
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
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotary_embedding_torch import RotaryEmbedding

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH     = "./data/enwik8.gz"
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
NUM_EVAL_SEQS    = 200          # 每个长度评估多少条序列
EVAL_LAST_N      = 64           # 只评估最后N个token
OUTPUT_DIR       = Path("./eval_seqlen_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── checkpoint 路径 ──────────────────────────
# 参考项目 TITANS（原始 HyperConnections 训练）
REF_TITANS_CKPT   = "./ref-project/model_latest.pt"
# 当前 TITANS（MC HyperConnections 训练）
CUR_TITANS_CKPT   = "./checkpoints_v2/ckpt_step_100000_final.pt"
# Baseline Transformer
BASELINE_CKPT     = "./checkpoints_baseline/best.pt"

# 控制评估哪些模型（设为 False 可跳过）
EVAL_REF_TITANS   = True
EVAL_CUR_TITANS   = True
EVAL_BASELINE     = True


# ─────────────────────────────────────────────
# Baseline Transformer 定义（与训练脚本一致）
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head  = dim // num_heads
        self.norm      = nn.LayerNorm(dim)
        self.to_qkv    = nn.Linear(dim, dim * 3, bias=False)
        self.to_out    = nn.Linear(dim, dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.dim_head)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.num_heads
        x_normed = self.norm(x)
        qkv = self.to_qkv(x_normed).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, self.dim_head).transpose(1, 2), qkv)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ff   = FeedForward(dim, ff_mult, dropout)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, num_tokens=256, dim=384, depth=8, num_heads=8, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers    = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult) for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_logits.weight = self.token_emb.weight

    def forward(self, x):
        x = self.token_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.to_logits(x)


# ─────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────
class SeqDataset(Dataset):
    """从验证集顺序切出固定长度的序列"""
    def __init__(self, data: torch.Tensor, seq_len: int, num_seqs: int):
        self.data    = data
        self.seq_len = seq_len
        total_avail = len(data) - seq_len - 1
        step = max(1, total_avail // num_seqs)
        self.starts = [i * step for i in range(num_seqs) if i * step + seq_len + 1 <= len(data)]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return self.data[s : s + self.seq_len + 1].long()


def load_val_data():
    with gzip.open(DATASET_PATH) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, val = np.split(data, [int(90e6)])
    return torch.from_numpy(val)


# ─────────────────────────────────────────────
# 智能 checkpoint 加载
# ─────────────────────────────────────────────
def smart_load_state_dict(model, ckpt_path):
    """
    自动识别 checkpoint 格式并加载：
      1. 纯 state_dict（key 直接是参数名）
      2. {'model': state_dict, ...}
      3. {'model_state_dict': state_dict, ...}
    同时处理 expanded tensor (memory_model_parameters) 的问题。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ── 提取 state_dict ──────────────────────
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "longterm_mems" in ckpt or "token_emb.weight" in ckpt:
            # 纯 state_dict，没有包装
            state_dict = ckpt
        else:
            # 尝试找到第一个是 dict 的 value
            for k, v in ckpt.items():
                if isinstance(v, dict) and any("weight" in kk for kk in v.keys()):
                    state_dict = v
                    break
            else:
                raise RuntimeError(
                    f"无法识别 checkpoint 格式，keys: {list(ckpt.keys())[:10]}"
                )
    else:
        state_dict = ckpt

    # ── 加载（使用 assign=True 来处理 expanded tensor 问题）──
    # assign=True 会直接替换参数张量，而不是 copy_ 到已有的张量中
    # 这样就不会触发 expanded tensor 的 "more than one element refers to
    # a single memory location" 错误
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        # PyTorch < 2.1 不支持 assign 参数，手动处理
        # 先 clone 所有 tensor
        cloned = {k: v.clone().contiguous() for k, v in state_dict.items()}
        # 对 expanded tensor 的参数做特殊处理
        model_sd = model.state_dict()
        for name, param in model.named_parameters():
            if name in cloned:
                param.data = cloned[name]
        # 再尝试加载剩余的 buffer 等
        missing, unexpected = model.load_state_dict(cloned, strict=False)
        if missing:
            print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    return model


# ─────────────────────────────────────────────
# 模型加载函数
# ─────────────────────────────────────────────

def load_ref_titans(ckpt_path):
    """
    加载参考项目的 TITANS checkpoint。
    使用 ref-project 的模型代码（原始 HyperConnections）。
    """
    # 使用 ref-project 目录下的 titans_pytorch
    ref_dir = str(Path(__file__).parent / "ref-project")
    # 临时修改 sys.path 以使用 ref-project 的代码
    original_modules = {}
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("titans_pytorch"):
            original_modules[mod_name] = sys.modules.pop(mod_name)

    sys.path.insert(0, ref_dir)

    try:
        from titans_pytorch import MemoryAsContextTransformer as RefMAC
        from titans_pytorch import MemoryMLP as RefMemoryMLP

        model = RefMAC(
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
            neural_memory_model=RefMemoryMLP(dim=64, depth=2),
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
            ),
        )

        model = smart_load_state_dict(model, ckpt_path)
        model = model.to(DEVICE)
        model.eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Ref TITANS loaded  ({params:.1f}M params) [original HyperConnections]")
        return model

    finally:
        # 恢复原来的 titans_pytorch 模块
        sys.path.remove(ref_dir)
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("titans_pytorch"):
                sys.modules.pop(mod_name, None)
        sys.modules.update(original_modules)


def load_cur_titans(ckpt_path):
    """
    加载 v2 训练的 TITANS checkpoint（train_mac_multigpu_v2.py）。
    v2 使用 ref-project 的原始 HyperConnections，4GPU×bs4×accum2=32 有效批次。
    """
    # v2 与 ref 使用相同的模型代码，复用 load_ref_titans 的加载逻辑
    ref_dir = str(Path(__file__).parent / "ref-project")
    original_modules = {}
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("titans_pytorch"):
            original_modules[mod_name] = sys.modules.pop(mod_name)

    sys.path.insert(0, ref_dir)

    try:
        from titans_pytorch import MemoryAsContextTransformer as RefMAC
        from titans_pytorch import MemoryMLP as RefMemoryMLP

        model = RefMAC(
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
            neural_memory_model=RefMemoryMLP(dim=64, depth=2),
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
            ),
        )

        model = smart_load_state_dict(model, ckpt_path)
        model = model.to(DEVICE)
        model.eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  V2 TITANS loaded  ({params:.1f}M params) [original HC, 4GPU bs32]")
        return model

    finally:
        sys.path.remove(ref_dir)
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("titans_pytorch"):
                sys.modules.pop(mod_name, None)
        sys.modules.update(original_modules)


def load_baseline(ckpt_path):
    """加载 Baseline Transformer"""
    model = BaselineTransformer(num_tokens=256, dim=384, depth=8, num_heads=8)
    model = smart_load_state_dict(model, ckpt_path)
    model = model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Baseline loaded ({params:.1f}M params)")
    return model


# ─────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_data, seq_len, num_seqs, eval_last_n, model_name):
    """
    对给定序列长度，评估模型在完整序列所有 token 上的 loss / acc。
    """
    dataset    = SeqDataset(val_data, seq_len, num_seqs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0

    for batch in tqdm(dataloader, desc=f"  {model_name} seq={seq_len}", leave=False):
        batch = batch.to(DEVICE)
        inp   = batch[:, :-1]
        tgt   = batch[:, 1:]

        try:
            logits = model(inp)
        except Exception as e:
            print(f"  [SKIP] {model_name} seq={seq_len}: {e}")
            break

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        )
        total_loss    += loss.item()
        total_correct += (logits.argmax(-1) == tgt).sum().item()
        total_tokens  += tgt.numel()

    if total_tokens == 0:
        return None, None, None, None

    avg_loss = total_loss / len(dataset)
    ppl      = math.exp(avg_loss)
    acc      = total_correct / total_tokens * 100
    bpc      = avg_loss / math.log(2)
    return avg_loss, ppl, acc, bpc


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("随序列长度变化的性能对比：Ref TITANS vs Cur TITANS vs Baseline")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"序列长度     : {SEQUENCE_LENGTHS}")
    print(f"每长度序列数 : {NUM_EVAL_SEQS}")
    print(f"评估范围     : 完整序列所有 token")
    print("=" * 70)

    print("\n加载数据...")
    val_data = load_val_data()
    print(f"验证集大小: {len(val_data):,} bytes")

    # ── 加载模型 ──────────────────────────────
    models = {}  # name -> model

    print("\n加载模型...")

    if EVAL_REF_TITANS and Path(REF_TITANS_CKPT).exists():
        models["ref_titans"] = load_ref_titans(REF_TITANS_CKPT)
    elif EVAL_REF_TITANS:
        print(f"  [SKIP] Ref TITANS checkpoint not found: {REF_TITANS_CKPT}")

    if EVAL_CUR_TITANS and Path(CUR_TITANS_CKPT).exists():
        models["cur_titans"] = load_cur_titans(CUR_TITANS_CKPT)
    elif EVAL_CUR_TITANS:
        print(f"  [SKIP] Cur TITANS checkpoint not found: {CUR_TITANS_CKPT}")

    if EVAL_BASELINE and Path(BASELINE_CKPT).exists():
        models["baseline"] = load_baseline(BASELINE_CKPT)
    elif EVAL_BASELINE:
        print(f"  [SKIP] Baseline checkpoint not found: {BASELINE_CKPT}")

    if not models:
        print("\n没有可用的模型，退出。")
        return

    print(f"\n已加载 {len(models)} 个模型: {list(models.keys())}")

    # ── 评估 ──────────────────────────────────
    results = {name: {"seq_lens": [], "loss": [], "ppl": [], "acc": [], "bpc": []}
               for name in models}

    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n──── seq_len = {seq_len} ────")

        for model_name, model in models.items():
            result = evaluate(model, val_data, seq_len, NUM_EVAL_SEQS, EVAL_LAST_N, model_name)
            if result[0] is None:
                print(f"  [{model_name}] 跳过（forward 失败）")
                continue
            avg_loss, ppl, acc, bpc = result
            results[model_name]["seq_lens"].append(seq_len)
            results[model_name]["loss"].append(avg_loss)
            results[model_name]["ppl"].append(ppl)
            results[model_name]["acc"].append(acc)
            results[model_name]["bpc"].append(bpc)
            print(f"  [{model_name:12}]  loss={avg_loss:.4f}  PPL={ppl:.3f}  BPC={bpc:.4f}  Acc={acc:.2f}%")

    torch.cuda.empty_cache()

    # ── 保存数值结果 ──────────────────────────
    summary_path = OUTPUT_DIR / "results_ref.txt"
    with open(summary_path, "w") as f:
        f.write("seq_len,model,loss,ppl,bpc,acc\n")
        for model_name in models:
            for i, sl in enumerate(results[model_name]["seq_lens"]):
                f.write(f"{sl},{model_name},"
                        f"{results[model_name]['loss'][i]:.6f},"
                        f"{results[model_name]['ppl'][i]:.4f},"
                        f"{results[model_name]['bpc'][i]:.6f},"
                        f"{results[model_name]['acc'][i]:.4f}\n")
    print(f"\n数值结果已保存: {summary_path}")

    # ── 绘图 ──────────────────────────────────
    styles = {
        "ref_titans":  dict(color="#2E86AB", marker="o", linewidth=2.5, markersize=8,
                            label="TITANS (Ref, original HC)"),
        "cur_titans":  dict(color="#E84855", marker="^", linewidth=2.5, markersize=8,
                            label="TITANS (V2, original HC, 4GPU bs32)"),
        "baseline":    dict(color="#F18F01", marker="s", linewidth=2.5, markersize=8,
                            label="Baseline Transformer"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Ref TITANS vs Cur TITANS vs Baseline: Performance vs Sequence Length\n"
                 "(Evaluated on full sequence — all tokens)",
                 fontsize=13, fontweight="bold")

    # Plot 1: BPC
    ax = axes[0]
    for name in models:
        r = results[name]
        if r["seq_lens"]:
            ax.plot(r["seq_lens"], r["bpc"], **styles[name])
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("BPC (↓)", fontsize=11)
    ax.set_title("Bits Per Character", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Plot 2: PPL
    ax = axes[1]
    for name in models:
        r = results[name]
        if r["seq_lens"]:
            ax.plot(r["seq_lens"], r["ppl"], **styles[name])
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Perplexity (↓)", fontsize=11)
    ax.set_title("Perplexity", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Plot 3: Accuracy
    ax = axes[2]
    for name in models:
        r = results[name]
        if r["seq_lens"]:
            ax.plot(r["seq_lens"], r["acc"], **styles[name])
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Accuracy % (↑)", fontsize=11)
    ax.set_title("Next-Character Accuracy", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "seqlen_comparison_ref.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {fig_path}")

    # ── 终端打印汇总 ──────────────────────────
    all_names = list(models.keys())
    header_bpc = "  ".join(f"{n:>12}" for n in all_names)
    header_acc = "  ".join(f"{n:>12}" for n in all_names)

    print("\n" + "=" * 80)
    print(f"{'seq_len':>8} │ BPC: {header_bpc} │ Acc: {header_acc}")
    print("-" * 80)
    all_lens = sorted(set().union(*(results[n]["seq_lens"] for n in all_names)))
    for sl in all_lens:
        bpc_vals = []
        acc_vals = []
        for n in all_names:
            if sl in results[n]["seq_lens"]:
                idx = results[n]["seq_lens"].index(sl)
                bpc_vals.append(f"{results[n]['bpc'][idx]:>12.4f}")
                acc_vals.append(f"{results[n]['acc'][idx]:>11.2f}%")
            else:
                bpc_vals.append(f"{'N/A':>12}")
                acc_vals.append(f"{'N/A':>12}")
        print(f"{sl:>8} │ BPC: {'  '.join(bpc_vals)} │ Acc: {'  '.join(acc_vals)}")
    print("=" * 80)
    print("\n完成！")


if __name__ == "__main__":
    main()
