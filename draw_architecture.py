"""
draw_architecture.py
--------------------
绘制当前代码中新架构（MemoryAsContextTransformer + NeuralMemory + 量化噪声注入）
的详细架构图。

输出：docs/architecture_overview.png
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# ── 注册中文字体（Noto Sans CJK）──
_cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_cjk_font)
_cjk_prop = font_manager.FontProperties(fname=_cjk_font)
plt.rcParams["font.family"] = _cjk_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/home/titans_v0.5/titans-memory/docs/architecture_overview.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 配色 ──
C_STD   = "#2E86AB"   # 标准 Transformer 组件
C_MEM   = "#F18F01"   # Neural Memory 组件
C_TOK   = "#5BB85B"   # 记忆 token
C_QUANT = "#E84855"   # 量化噪声注入点（研究新增）
C_BG    = "#FAFAFA"
C_BOX   = "#FFFFFF"
C_TXT   = "#222222"

fig, ax = plt.subplots(figsize=(15, 20))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor(C_BG)


def box(x, y, w, h, text, fc=C_BOX, ec=C_TXT, lw=1.5, fs=11, bold=False, tc=C_TXT, rounded=0.02):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                      boxstyle=f"round,pad=0.1,rounding_size={rounded}",
                      facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3,
            wrap=True)


def arrow(x1, y1, x2, y2, color=C_TXT, lw=1.8, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=18,
                        color=color, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


def panel(x, y, w, h, title, fc="#FFFFFF", ec="#888888"):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.2,rounding_size=0.5",
                       facecolor=fc, edgecolor=ec, linewidth=2.0,
                       linestyle="--", zorder=1)
    ax.add_patch(b)
    ax.text(x - w/2 + 1.5, y + h/2 - 1.5, title, ha="left", va="top",
            fontsize=11.5, fontweight="bold", color=ec, zorder=3)


# ═══════════════════════════════════════════════════════════════════════════════
# 标题
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(50, 98.5,
        "Titans 新架构：Memory-As-Context Transformer + Neural Memory（含量化噪声注入）",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_TXT)
ax.text(50, 96.2,
        "配置：num_tokens=256, dim=384, depth=8, segment_len=32, "
        "neural_memory_layers=(2,4,6), MemoryMLP(dim=64,depth=2)",
        ha="center", va="center", fontsize=9.5, color="#555555")

# ═══════════════════════════════════════════════════════════════════════════════
# Panel A：整体前向流（y 88 → 56）
# ═══════════════════════════════════════════════════════════════════════════════
panel(50, 72, 90, 38, "A. 整体前向流  MemoryAsContextTransformer", fc="#FFFFFF", ec=C_STD)

box(50, 88.5, 40, 3.2, "Input tokens  (B, L)   vocab = 256", fc=C_STD, ec=C_STD, tc="white", bold=True, fs=10.5)
arrow(50, 86.8, 50, 85.4)
box(50, 84.0, 40, 3.2, "Token Embedding   (256 → 384)", fc=C_STD, ec=C_STD, tc="white", fs=10.5)
arrow(50, 82.4, 50, 81.0)
box(50, 79.6, 52, 3.0, "切分为 segment（segment_len = 32），逐段处理", fc=C_BOX, ec=C_STD, fs=10)
arrow(50, 78.0, 50, 76.6)

# 每个 segment 的输入上下文（三部分拼接）
ax.text(50, 75.6, "每个 segment 的上下文 = 以下三部分拼接：",
        ha="center", va="center", fontsize=9.5, color="#555555")
box(20, 73.3, 26, 3.6, "Persistent Mem\n(4 tokens, 学习固定)", fc=C_TOK, ec=C_TOK, tc="white", fs=9.5, bold=True)
box(50, 73.3, 26, 3.6, "Long-term Mem\n(4 tokens ← Neural Memory)", fc=C_MEM, ec=C_MEM, tc="white", fs=9.5, bold=True)
box(80, 73.3, 26, 3.6, "Segment tokens\n(32 tokens)", fc=C_STD, ec=C_STD, tc="white", fs=9.5, bold=True)

arrow(50, 71.2, 50, 69.8)

# 8 层 Transformer
box(50, 68.0, 70, 3.0, "Transformer Layers × 8  (dim=384, sliding window self-attn, HyperConnection)",
    fc=C_BOX, ec=C_STD, fs=9.8, bold=True)

# 8 个小层框
layer_y = 65.2
xs = [20, 28.6, 37.2, 45.7, 54.3, 62.9, 71.4, 80.0]
labels = ["L0", "L1", "L2★", "L3", "L4★", "L5", "L6★", "L7"]
for x, lab in zip(xs, labels):
    is_mem = "★" in lab
    box(x, layer_y, 7.0, 2.6, lab,
        fc=C_MEM if is_mem else C_STD,
        ec=C_MEM if is_mem else C_STD,
        tc="white", fs=9, bold=True)
ax.text(50, 62.8, "★ = 含 Neural Memory（layers 2, 4, 6）   其余层 = Sliding-Window Self-Attn + FFN",
        ha="center", va="center", fontsize=8.8, color="#555555")

arrow(50, 61.6, 50, 60.2)
box(50, 58.8, 44, 3.0, "Norm → ToLogits (tied weights)  →  next-char 分布", fc=C_STD, ec=C_STD, tc="white", fs=10)
# 连接到 B 区的箭头（从 L2/4/6 的 Neural Memory 下钻到 Panel B）
arrow(37.2, 63.9, 30, 56.5, color=C_MEM, lw=2.2, rad=0.15)
arrow(54.3, 63.9, 50, 56.5, color=C_MEM, lw=2.2, rad=0.0)
arrow(71.4, 63.9, 70, 56.5, color=C_MEM, lw=2.2, rad=-0.15)
ax.text(50, 59.0, "Neural Memory 细节见下方 ↓", ha="center", va="center",
        fontsize=8.8, color=C_MEM, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B：Neural Memory 模块（y 54 → 6）
# ═══════════════════════════════════════════════════════════════════════════════
panel(50, 30, 92, 50,
      "B. NeuralMemory 模块（出现在 layers 2, 4, 6）  MemoryMLP(dim=64, depth=2) 作联想记忆",
      fc="#FFFFFF", ec=C_MEM)

# 输入
box(50, 52.5, 40, 2.8, "segment hidden  (fp32)", fc=C_BOX, ec=C_TXT, fs=9.5, bold=True)

# ── STORE 路径（左） ──
ax.text(22, 50.2, "STORE（写入，fp32，带梯度）", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_MEM)
arrow(35, 52.0, 22, 50.0, color=C_MEM, lw=1.8, rad=0.2)

box(22, 47.5, 34, 2.8, "计算 surprise\n(spectral_norm_surprises)", fc=C_BOX, ec=C_MEM, fs=9)
arrow(22, 46.1, 22, 44.9, color=C_MEM)
box(22, 43.6, 34, 2.8, "自适应学习率\n(per_parameter_lr_modulation + momentum order=1)",
    fc=C_BOX, ec=C_MEM, fs=9)
arrow(22, 42.2, 22, 40.8, color=C_MEM)
box(22, 39.5, 34, 2.8, "更新 MemoryMLP 权重\n(functional_call + grad)", fc=C_MEM, ec=C_MEM, tc="white", fs=9, bold=True)
arrow(22, 38.0, 22, 35.5, color=C_MEM, lw=2.2)

# 权重（fp32，动态）—— 中间产物
box(50, 34.5, 30, 3.2, "weights (fp32, 动态)\n= 存储的压缩 K/V 记忆", fc=C_TOK, ec=C_TOK, tc="white", fs=9.3, bold=True)
arrow(22, 35.5, 38, 34.5, color=C_MEM, lw=2.2)  # store → weights

# ── RETRIEVE 路径（右） ──
ax.text(78, 50.2, "RETRIEVE（读出）", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_MEM)
arrow(65, 52.0, 78, 50.0, color=C_MEM, lw=1.8, rad=-0.2)

box(78, 47.5, 34, 2.8, "to_queries → q_norm → queries (fp32)", fc=C_BOX, ec=C_MEM, fs=9)
arrow(78, 46.1, 78, 44.9, color=C_MEM)

# ★ 量化噪声注入点（核心） ★
box(78, 43.6, 38, 3.4,
    "★ quantize_weight_dict(weights)\n4-bit 均匀量化 + 5% 加性高斯噪声   (仅 eval 模式；train 透传)",
    fc=C_QUANT, ec=C_QUANT, tc="white", fs=9, bold=True)
arrow(78, 42.0, 78, 40.6, color=C_QUANT, lw=2.2)
box(78, 39.3, 34, 2.8, "functional_call(MemoryMLP,\nquantized_weights, queries)",
    fc=C_MEM, ec=C_MEM, tc="white", fs=9, bold=True)
arrow(78, 37.8, 78, 35.5, color=C_MEM, lw=2.2)

# weights → quantize 的箭头（从 weights 流入 retrieve 的 quantize 步）
arrow(62, 34.5, 78, 40.5, color=C_QUANT, lw=2.0, ls="--", rad=-0.25)
ax.text(72, 37.8, "读出时量化\n(memristor 模拟)", ha="center", va="center",
        fontsize=8, color=C_QUANT, fontweight="bold")

# retrieve 输出 → values
arrow(78, 35.5, 78, 33.2, color=C_MEM, lw=2.2)
box(78, 31.8, 34, 2.8, "values → multihead_rmsnorm\n→ gate → 注入为 Long-term Mem", fc=C_BOX, ec=C_MEM, fs=9)

# 回到整体（Long-term Mem tokens 流回下一个 segment 的上下文）
arrow(78, 30.4, 92, 30.4, color=C_MEM, lw=2.2)
arrow(92, 30.4, 92, 73.3, color=C_MEM, lw=2.0, ls="--", rad=0.0)
arrow(92, 73.3, 63, 73.3, color=C_MEM, lw=2.0, ls="--", rad=0.0)
ax.text(93, 50, "Long-term\nMem tokens\n(4) 回流\n到下一段", ha="center", va="center",
        fontsize=8, color=C_MEM, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# 图例
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_STD,   "标准 Transformer 组件\n(embed / sliding-window attn / FFN / logits)"),
    (C_MEM,   "Neural Memory 组件\n(MemoryMLP / store / retrieve)"),
    (C_TOK,   "记忆 token 与动态权重\n(persistent mem / long-term mem / weights)"),
    (C_QUANT, "★ 量化噪声注入点（研究新增）\n4-bit + 5% 高斯，仅 eval，retrieve 前"),
]
lx = 8
for color, label in legend_items:
    ax.add_patch(mpatches.Rectangle((lx, 6.0), 2.2, 2.2, facecolor=color, edgecolor="black"))
    ax.text(lx + 2.8, 7.1, label, ha="left", va="center", fontsize=8.5, color=C_TXT)
    lx += 23

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"saved: {OUT}")
