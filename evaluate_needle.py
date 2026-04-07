"""
远程记忆能力评测（Needle-in-a-Haystack，字符级简化版）

适合小模型 + enwik8 字符级数据集的长程依赖测试。

核心思路：
  1. 从 enwik8 验证集取一段真实文本作为"干草堆"（haystack）
  2. 在某个位置插入一个 3-byte 的"标记-值"对：[255, V, 255]
     其中 V 是随机 byte (0~254)
  3. 在序列末尾再放一个 [255]，看模型能否预测出 V
  4. 改变序列总长度和"针"的插入距离，测试模型的远程记忆能力

为什么这个测试有效：
  - enwik8 中 byte 255 极其罕见（几乎不出现），所以 [255, V, 255] 是一个
    模型在正常训练中不可能见过的独特模式
  - 如果模型在末尾看到 [255] 后能预测出 V，说明它"记住"了远处的信息
  - 滑窗模型（窗口=32）在距离 > 32 时完全无法做到
  - 有 Neural Memory 的模型理论上可以

评估指标：
  - Top-1 Accuracy: 模型预测的最高概率 token == V 的比例
  - Target Rank: V 在模型输出概率分布中的排名（越低越好）
  - Target Prob: 模型给 V 的概率（越高越好）

运行：
    python evaluate_needle.py
"""

import os
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

import sys
import gzip
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotary_embedding_torch import RotaryEmbedding

# ── 加载 ref-project 的 titans_pytorch ──
_ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-project")
sys.path.insert(0, _ref_dir)
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "./data/enwik8.gz"
OUTPUT_DIR   = Path("./eval_needle_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 测试参数
NEEDLE_DISTANCES = [16, 32, 64, 128, 256, 512, 1024, 2048]
# "针"距离序列末尾多远（以 token 为单位）
# 16, 32: 在滑窗范围内（window=36）
# 64+:    超出滑窗范围，只有 NM 能"记住"

NUM_TRIALS   = 100    # 每个距离测试多少次（取平均）
MARKER_BYTE  = 255    # 用作标记的特殊 byte（enwik8 中极少出现）

# Checkpoints
REF_TITANS_CKPT = "./ref-project/model_latest.pt"
CUR_TITANS_CKPT = "./checkpoints/ckpt_step_097500.pt"
BASELINE_CKPT   = "./checkpoints_baseline/best.pt"

# ═══════════════════════════════════════════════════════════
# Baseline Transformer 定义
# ═══════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, inner_dim),
            nn.GELU(), nn.Linear(inner_dim, dim),
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
    def __init__(self, dim, num_heads, ff_mult=4):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ff   = FeedForward(dim, ff_mult)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class BaselineTransformer(nn.Module):
    def __init__(self, num_tokens=256, dim=384, depth=8, num_heads=8, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers    = nn.ModuleList([TransformerBlock(dim, num_heads, ff_mult) for _ in range(depth)])
        self.norm      = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_logits.weight = self.token_emb.weight
    def forward(self, x):
        x = self.token_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.to_logits(x)

# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════
def smart_load_state_dict(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            # 可能整个 dict 就是 state_dict
            if any(k.startswith("token_emb") or k.startswith("layers") or k.startswith("norm") for k in ckpt.keys()):
                state_dict = ckpt
            else:
                raise RuntimeError(f"Unknown checkpoint format: {list(ckpt.keys())[:10]}")
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, assign=True)
    return model


def load_ref_titans():
    model = MemoryAsContextTransformer(
        num_tokens=256, dim=384, depth=8, segment_len=32,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=(2, 4, 6),
        neural_memory_segment_len=4, neural_memory_batch_size=128,
        neural_mem_gate_attn_output=False, neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=False, sliding_window_attn=True,
        neural_memory_model=MemoryMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(
            dim_head=64, heads=4, attn_pool_chunks=True, qk_rmsnorm=True,
            momentum=True, momentum_order=1, default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False, per_parameter_lr_modulation=True,
            spectral_norm_surprises=True,
        ),
    )
    smart_load_state_dict(model, REF_TITANS_CKPT)
    model = model.to(DEVICE).eval()
    print(f"  ✓ ref_titans loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    return model


def load_cur_titans():
    # 当前版本使用本地的 titans_pytorch（MC HyperConnections）
    # 需要临时切换 sys.path
    saved_path = sys.path.copy()
    sys.path = [p for p in sys.path if 'ref-project' not in p]

    # 重新导入当前版本
    import importlib
    import titans_pytorch as cur_tp
    importlib.reload(cur_tp)
    from titans_pytorch import MemoryAsContextTransformer as CurMAC, MemoryMLP as CurMLP

    model = CurMAC(
        num_tokens=256, dim=384, depth=8, segment_len=32,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=(2, 4, 6),
        neural_memory_segment_len=4, neural_memory_batch_size=128,
        neural_mem_gate_attn_output=False, neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=False, sliding_window_attn=True,
        neural_memory_model=CurMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(
            dim_head=64, heads=4, attn_pool_chunks=True, qk_rmsnorm=True,
            momentum=True, momentum_order=1, default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False, per_parameter_lr_modulation=True,
            spectral_norm_surprises=True, store_with_lookahead_value=False,
        ),
    )
    smart_load_state_dict(model, CUR_TITANS_CKPT)
    model = model.to(DEVICE).eval()
    sys.path = saved_path
    print(f"  ✓ cur_titans loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    return model


def load_baseline():
    model = BaselineTransformer(num_tokens=256, dim=384, depth=8, num_heads=8)
    smart_load_state_dict(model, BASELINE_CKPT)
    model = model.to(DEVICE).eval()
    print(f"  ✓ baseline loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    return model


# ═══════════════════════════════════════════════════════════
# 大海捞针测试
# ═══════════════════════════════════════════════════════════
def make_needle_sequence(haystack_data, needle_distance, needle_value):
    """
    构造一条测试序列：

    [真实enwik8文本...] [255, V, 255] [真实enwik8文本...] [255]
    |<--- 前缀 --->|   |<- 针 ->|    |<--- 填充 ---->|   |<-查询

    needle_distance: "针"到序列末尾的距离（token数）
    needle_value:    要"记住"的值 V (0~254)

    返回：
      seq: (total_len,) 的 LongTensor，最后一个 token 是 255（查询标记）
      target: V（正确答案，模型应在最后位置预测出 V）
    """
    # 需要的总长度 = needle_distance + 一些前缀
    prefix_len = 64  # 针前面放一些正常文本作为"暖身"
    # 针本身占 3 个位置: [255, V, 255]
    # 查询占 1 个位置: [255]
    # 填充 = needle_distance - 3 - 1  (针到查询之间的真实文本)
    fill_len = needle_distance - 4  # 减去针(3) + 查询(1)

    if fill_len < 0:
        fill_len = 0

    total_needed = prefix_len + 3 + fill_len + 1

    # 从 haystack 随机取一段真实文本
    max_start = len(haystack_data) - total_needed - 10
    if max_start < 0:
        max_start = 0
    start = random.randint(0, max_start)
    raw = haystack_data[start: start + total_needed].clone()

    # 替换掉原始文本中的所有 255（避免干扰）
    raw[raw == MARKER_BYTE] = 254

    # 构造序列
    seq = torch.zeros(total_needed, dtype=torch.long)

    # 前缀：真实文本
    seq[:prefix_len] = raw[:prefix_len]

    # 插入针：[255, V, 255]
    needle_pos = prefix_len
    seq[needle_pos]     = MARKER_BYTE    # 255
    seq[needle_pos + 1] = needle_value   # V
    seq[needle_pos + 2] = MARKER_BYTE    # 255

    # 填充：真实文本（继续用后面的 enwik8 数据）
    fill_start = needle_pos + 3
    seq[fill_start: fill_start + fill_len] = raw[prefix_len: prefix_len + fill_len]

    # 查询：序列最后一个 token 是 255
    seq[-1] = MARKER_BYTE  # 255

    return seq, needle_value


@torch.no_grad()
def evaluate_needle(model, haystack_data, needle_distance, num_trials, model_name):
    """
    在给定距离下测试模型的远程记忆能力。

    返回：
      top1_acc:    模型 argmax 预测 == V 的比例
      avg_rank:    V 在概率分布中的平均排名 (0=最高)
      avg_prob:    V 的平均概率
    """
    correct = 0
    total_rank = 0
    total_prob = 0.0

    for trial in range(num_trials):
        # 随机选择要记忆的值
        needle_value = random.randint(0, 254)

        seq, target = make_needle_sequence(haystack_data, needle_distance, needle_value)
        inp = seq.unsqueeze(0).to(DEVICE)  # (1, total_len)

        try:
            logits = model(inp)  # (1, total_len, 256)
        except Exception as e:
            print(f"    [SKIP] {model_name} dist={needle_distance}: {e}")
            continue

        # 取最后一个位置的 logits（模型看到 255 后应该预测 V）
        last_logits = logits[0, -1, :]  # (256,)
        probs = F.softmax(last_logits, dim=-1)

        # Top-1 预测
        pred = last_logits.argmax().item()
        if pred == target:
            correct += 1

        # V 的排名
        sorted_indices = probs.argsort(descending=True)
        rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item()
        total_rank += rank

        # V 的概率
        total_prob += probs[target].item()

    n = num_trials
    return correct / n * 100, total_rank / n, total_prob / n


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("远程记忆能力评测 (Needle-in-a-Haystack)")
    print("=" * 70)

    # 加载数据
    print("\n加载 enwik8 验证集...")
    with gzip.open(DATASET_PATH) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, val_data = np.split(data, [int(90e6)])
    haystack = torch.from_numpy(val_data)
    print(f"  验证集大小: {len(haystack):,} bytes")
    # 检查 byte 255 在验证集中的出现次数
    n255 = (haystack == 255).sum().item()
    print(f"  Byte 255 出现次数: {n255} ({n255/len(haystack)*100:.4f}%)")

    # 加载模型
    print("\n加载模型...")
    models = {}

    if Path(REF_TITANS_CKPT).exists():
        models["ref_titans"] = load_ref_titans()
    else:
        print(f"  ✗ ref_titans checkpoint not found: {REF_TITANS_CKPT}")

    if Path(CUR_TITANS_CKPT).exists():
        models["cur_titans"] = load_cur_titans()
    else:
        print(f"  ✗ cur_titans checkpoint not found: {CUR_TITANS_CKPT}")

    if Path(BASELINE_CKPT).exists():
        models["baseline"] = load_baseline()
    else:
        print(f"  ✗ baseline checkpoint not found: {BASELINE_CKPT}")

    if not models:
        print("没有可用的模型！")
        return

    # 运行测试
    print(f"\n测试配置:")
    print(f"  针距离: {NEEDLE_DISTANCES}")
    print(f"  每距离试验数: {NUM_TRIALS}")
    print(f"  标记 byte: {MARKER_BYTE}")

    # results[model_name][distance] = (acc, rank, prob)
    results = {name: {} for name in models}

    for dist in NEEDLE_DISTANCES:
        print(f"\n──── needle_distance = {dist} tokens ────")
        for model_name, model in models.items():
            acc, avg_rank, avg_prob = evaluate_needle(
                model, haystack, dist, NUM_TRIALS, model_name
            )
            results[model_name][dist] = (acc, avg_rank, avg_prob)
            in_window = "✓ 窗口内" if dist <= 36 else "✗ 窗口外"
            print(f"  {model_name:15s}: acc={acc:5.1f}%  rank={avg_rank:6.1f}  prob={avg_prob:.4f}  ({in_window})")

    # ── 打印汇总表格 ──
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"\n{'距离':>6s}", end="")
    for name in models:
        print(f"  │ {name:>15s} acc", end="")
    print()
    print("─" * (8 + 22 * len(models)))

    for dist in NEEDLE_DISTANCES:
        marker = " ←窗口" if dist <= 36 else ""
        print(f"{dist:>6d}", end="")
        for name in models:
            acc = results[name][dist][0]
            print(f"  │ {acc:>15.1f}%", end="")
        print(f"  {marker}")

    # ── 保存 CSV ──
    csv_path = OUTPUT_DIR / "needle_results.csv"
    with open(csv_path, "w") as f:
        f.write("distance,model,accuracy,avg_rank,avg_prob\n")
        for name in models:
            for dist in NEEDLE_DISTANCES:
                acc, rank, prob = results[name][dist]
                f.write(f"{dist},{name},{acc:.2f},{rank:.2f},{prob:.6f}\n")
    print(f"\n结果已保存: {csv_path}")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        "ref_titans": "#2196F3",
        "cur_titans": "#4CAF50",
        "baseline":   "#FF5722",
    }
    labels = {
        "ref_titans": "TITANS (ref, original HC)",
        "cur_titans": "TITANS (cur, MC HC)",
        "baseline":   "Baseline (full attn)",
    }

    for name in models:
        dists = NEEDLE_DISTANCES
        accs   = [results[name][d][0] for d in dists]
        ranks  = [results[name][d][1] for d in dists]
        probs  = [results[name][d][2] for d in dists]

        c = colors.get(name, "#999999")
        l = labels.get(name, name)

        axes[0].plot(dists, accs,  'o-', color=c, label=l, linewidth=2, markersize=6)
        axes[1].plot(dists, ranks, 'o-', color=c, label=l, linewidth=2, markersize=6)
        axes[2].plot(dists, probs, 'o-', color=c, label=l, linewidth=2, markersize=6)

    # 标记滑窗边界
    for ax in axes:
        ax.axvline(x=36, color='gray', linestyle='--', alpha=0.5, label='sliding window=36')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Needle Distance (tokens)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Top-1 Accuracy (%)')
    axes[0].set_title('远程记忆准确率')
    axes[0].set_ylim(-5, 105)

    axes[1].set_ylabel('Average Rank (lower=better)')
    axes[1].set_title('目标值在预测中的排名')
    axes[1].invert_yaxis()

    axes[2].set_ylabel('Target Probability')
    axes[2].set_title('模型给正确答案的概率')

    plt.suptitle('Needle-in-a-Haystack: 远程记忆能力评测\n'
                 '在距离 D 处插入 [255, V, 255]，在末尾放 [255]，测试模型能否预测 V',
                 fontsize=12, fontproperties=None)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "needle_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {fig_path}")
    plt.close()

    # ── 分析 ──
    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)
    print("""
测试说明：
  - 在真实 enwik8 文本中插入标记 [255, V, 255]（V 是随机 byte）
  - 在序列末尾放 [255]，看模型能否预测出 V
  - 距离 ≤ 36 tokens: 在滑窗 attention 的窗口范围内
  - 距离 > 36 tokens:  超出滑窗范围

预期行为：
  - baseline (full attn):  距离 ≤ 512 时应该能找到（训练长度内），超过则崩
  - 滑窗模型无 NM:         距离 ≤ 36 时能找到，超过则完全找不到
  - TITANS (滑窗 + NM):    距离 > 36 时如果仍能找到 → 证明 NM 在工作
    """)


if __name__ == "__main__":
    main()
