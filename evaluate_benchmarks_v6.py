"""
evaluate_benchmarks_v6.py
-------------------------
在 v6 checkpoint（量化存储 + 写噪声，部署条件最完整版本）之上，运行三个
**零训练改动** 的 benchmark（纯评估，复用现有 checkpoint，不改动任何权重）：

  Benchmark 1a — 段落重复探针（passage repeat probe，主指标）
      取一段 64 字符的自然文本（val 区域，prose 启发式筛选），在上下文
      depth（25%/50%/75%）处插入一次，序列末尾**完整重复**一次。
      对比两次出现时同一段文本的 BPC / Acc：
          delta_bpc = first_occurrence_BPC − second_occurrence_BPC
      即"见过 vs 没见过"的差值 = 纯记忆检索贡献。特别关注后半段
      （chars 32-63，超出 36-token 注意力窗口）——那里的改善只能来自
      Neural Memory 的跨段检索。有效上下文长度曲线 = delta 随长度/距离
      的衰减。此任务不需要模型学过任何指令格式，是纯 LM 场景下最干净
      的 test-time 记忆检索测试。

  Benchmark 1b — 随机 needle NIAH（次要指标）
      needle 为 16 个随机字母数字字符（prior 为零），末尾重复前 8 字符，
      统计后 8 字符 completion 的预测。用于检验"随机串 induction"是否
      被学到（小模型大概率没有——本身就是有价值的对照发现）。

  Benchmark 2 — 流式超长上下文 BPC（含 per-position 位置曲线）
      把 enwik8 val 数据流式（单次 forward，内部按 32-token 段滚动）喂到
      32K/64K/128K，按 512-char 窗口记录 BPC / Acc 曲线 → 考察记忆巩固
      的长期稳定性，并输出首窗 vs 末窗 BPC 差值（漂移指标，衔接忆阻器
      retention/drift 担忧）。

三种设备条件 variant 与 v6 量化噪声评估完全一致（公平对比）：
  1. fp32_clean   — 量化存储保持开启，但读路径不量化不加噪
  2. quant_only   — 4-bit 读量化，无读噪声
  3. quant_noise  — 4-bit 读量化 + 5% 读噪声（部署读条件）

输出：eval_benchmarks_v6/
  ├── repeat_results.csv / repeat_summary.txt / repeat_curves.png   (1a 主指标)
  ├── niah_results.csv / niah_summary.txt / niah_curves.png         (1b 次要)
  ├── streaming_results.csv / streaming_overall.csv
  ├── streaming_summary.txt / streaming_curves.png

运行：
    conda activate titans
    python evaluate_benchmarks_v6.py                 # 完整评估（单卡 ~3-5h）
    python evaluate_benchmarks_v6.py --quick         # 冒烟测试（~5 min）
    python evaluate_benchmarks_v6.py --variants fp32_clean,quant_noise
    python evaluate_benchmarks_v6.py --skip-needle --skip-stream   # 只跑段落探针
"""

import os
import sys

# ── GPU 选择（可用环境变量覆盖，默认单卡 3）────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
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

import argparse
import csv
import gzip
import math
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = "/home/titans-pytorch-main/data/enwik8.gz"
CKPT_PATH    = "./checkpoints_v6/best.pt"
OUTPUT_DIR   = Path("./eval_benchmarks_v6")

SEED = 20260818

# ── Benchmark 1a: 段落重复探针 ─────────────────────────────────────────────
REPEAT_LENGTHS    = [512, 1024, 2048, 4096, 8192, 16384]
REPEAT_DEPTHS     = [0.25, 0.50, 0.75]
REPEAT_NUM_SEQS   = 5
REPEAT_PASSAGE    = 64     # passage 长度 = 2 个注意力段（32+32）
REPEAT_HALF       = 32     # 前半段（段内注意力可见）/ 后半段（仅记忆可达）

# ── Benchmark 1b: 随机 needle NIAH（预期为 0 的对照指标，算力从简）────────
NIAH_LENGTHS   = [512, 2048, 8192]
NIAH_DEPTHS    = [0.25, 0.50, 0.75]
NIAH_NUM_SEQS  = 3
NEEDLE_LEN     = 16
PROMPT_LEN     = 8
NEEDLE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ── Benchmark 2: 流式超长 BPC ──────────────────────────────────────────────
STREAM_LENGTHS  = [32768, 65536, 131072]
STREAM_NUM_SEQS = 2
STREAM_WINDOW   = 512

# ── 三种设备条件（与 evaluate_quant_noise_v6_2gpu.py 完全一致）─────────────
VARIANTS = [
    {
        "name":  "fp32_clean",
        "label": "FP32 Clean (no read quant)",
        "color": "#2E86AB",
        "marker": "o",
        "quant_noise_cfg": dict(enabled=False, quantized_storage=True, write_noise_scale=0.05),
    },
    {
        "name":  "quant_only",
        "label": "4-bit Quant Only (no noise)",
        "color": "#E84855",
        "marker": "^",
        "quant_noise_cfg": dict(
            enabled=True, bit=4, noise_scale=0.0,
            noise_method="add", noise_range="max",
            quantized_storage=True, write_noise_scale=0.05,
        ),
    },
    {
        "name":  "quant_noise",
        "label": "4-bit Quant + Gaussian Noise (sigma=5%)",
        "color": "#F18F01",
        "marker": "s",
        "quant_noise_cfg": dict(
            enabled=True, bit=4, noise_scale=0.05,
            noise_method="add", noise_range="max",
            quantized_storage=True, write_noise_scale=0.05,
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 数据与模型
# ═══════════════════════════════════════════════════════════════════════════════

def load_val_data() -> np.ndarray:
    with gzip.open(DATASET_PATH) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, val = np.split(data, [int(90e6)])
    return val


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


def load_v6_model(ckpt_path: str, quant_noise_cfg: dict, variant_name: str, device: str):
    """构造与 evaluate_quant_noise_v6_2gpu.py 完全一致的模型并加载 checkpoint。"""
    ref_dir = str(Path(__file__).resolve().parent / "ref-project")

    saved_modules = {k: v for k, v in sys.modules.items() if k.startswith("titans_pytorch")}
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
        print(f"  [{variant_name:15s}] model loaded ({params:.2f}M params)", flush=True)
        return model

    finally:
        sys.path.remove(ref_dir)
        for k in list(sys.modules.keys()):
            if k.startswith("titans_pytorch"):
                del sys.modules[k]
        sys.modules.update(saved_modules)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 1a — 段落重复探针
# ═══════════════════════════════════════════════════════════════════════════════

def _is_prose(b: bytes) -> bool:
    """启发式：字母+空格占比高、无 XML 标记字符，尽量避免选到标记密集区。"""
    letters = sum(1 for x in b if 65 <= x <= 90 or 97 <= x <= 122)
    spaces = sum(1 for x in b if x == 32)
    return (letters + spaces) / len(b) >= 0.6 and b"<" not in b and b"&" not in b


def build_repeat_sequence(val: np.ndarray, total_len: int, depth_frac: float,
                          rng: np.random.Generator, max_tries: int = 200):
    """
    seq = base[:p] + b' ' + passage + base[p+nwl : L-nwl] + b' ' + passage
    passage（64 字符自然文本）出现两次：
      第一次在深度 p 处（记忆写入源），第二次在序列末尾（检索测试）。
    返回 (seq uint8 array, tgt_index_masks dict, distance)
      masks 指向 tgt 的下标区间（tgt[i] 预测 seq[i+1]）：
        first_half1 / first_half2 : 第一次出现的段落前半/后半（各 32 char）
        second_half1 / second_half2: 第二次出现的段落前半/后半
    """
    pl = REPEAT_PASSAGE
    nwl = pl + 1  # 前导空格 + passage

    p = int(depth_frac * (total_len - 2 * nwl))
    p = max(1, min(p, total_len - 2 * nwl))

    for _ in range(max_tries):
        s = int(rng.integers(0, max(1, len(val) - total_len - 1)))
        base = val[s : s + total_len]

        # 在 val 的其他区域选一段 prose（以空格开头）
        ps = int(rng.integers(0, len(val) - pl - 2))
        if val[ps] != 32:
            continue
        cand = val[ps + 1 : ps + 1 + pl]
        if not _is_prose(cand.tobytes()):
            continue
        passage = cand.tobytes()

        # passage 不得自然出现在 haystack 中（保证检索源唯一）
        if base.tobytes().find(passage) != -1:
            continue

        seq = base[:p].tobytes() + b" " + passage \
            + base[p + nwl : total_len - nwl].tobytes() + b" " + passage
        assert len(seq) == total_len

        # tgt 下标 mask：段落字符位于 seq[q : q+pl)，其预测位于 tgt [q-1, q+pl-2]
        q1, q2 = p + 1, total_len - nwl + 1
        masks = dict(
            first_half1=(q1 - 1, q1 - 1 + REPEAT_HALF),
            first_half2=(q1 - 1 + REPEAT_HALF, q1 - 1 + pl),
            second_half1=(q2 - 1, q2 - 1 + REPEAT_HALF),
            second_half2=(q2 - 1 + REPEAT_HALF, q2 - 1 + pl),
        )
        distance = q2 - (q1 + pl)  # 第一次段落结束 → 第二次段落开始的字符距离
        return np.frombuffer(seq, dtype=np.uint8).copy(), masks, distance

    raise RuntimeError("无法生成 prose passage，重试超限")


@torch.no_grad()
def run_repeat_probe(model, specs, device: str, desc: str):
    """
    specs: [(seq_len, depth, seq_np, masks, distance), ...]
    返回按 (seq_len, depth) 聚合的 rows。
    核心指标 delta_bpc_half2 = first_half2_BPC − second_half2_BPC
      （段落后半段超出注意力窗口，其改善只能来自 Neural Memory 检索）
    """
    groups = {}
    for (seq_len, depth, seq_np, masks, distance) in specs:
        groups.setdefault((seq_len, depth), []).append((seq_np, masks, distance))

    keys = ["first_half1", "first_half2", "second_half1", "second_half2"]
    rows = []
    for (seq_len, depth), items in tqdm(groups.items(), desc=desc, leave=False):
        agg = {k: {"loss": [], "correct": 0, "tokens": 0} for k in keys}
        for seq_np, masks, distance in tqdm(items, desc=f"  seq={seq_len} d={depth}", leave=False):
            L = len(seq_np)
            inp = torch.from_numpy(seq_np[:-1].astype(np.int64)).unsqueeze(0).to(device)
            tgt = torch.from_numpy(seq_np[1:].astype(np.int64)).unsqueeze(0).to(device)

            try:
                logits = model(inp)
            except Exception as e:
                print(f"    [SKIP] forward 失败 (seq={seq_len}): {e}", flush=True)
                continue

            per_tok = F.cross_entropy(logits[0], tgt[0], reduction="none")  # (L-1,)
            correct = (logits.argmax(-1)[0] == tgt[0])

            for k in keys:
                lo, hi = masks[k]
                agg[k]["loss"].append(per_tok[lo:hi].mean().item())
                agg[k]["correct"] += correct[lo:hi].sum().item()
                agg[k]["tokens"] += hi - lo

            del logits, inp, tgt, per_tok, correct

        if not agg["first_half2"]["loss"]:
            continue

        def _bpc(k):
            return float(np.mean(agg[k]["loss"])) / math.log(2)

        def _acc(k):
            return agg[k]["correct"] / max(1, agg[k]["tokens"]) * 100.0

        rows.append(dict(
            variant="", seq_len=seq_len, depth_frac=depth,
            distance_chars=int(np.mean([d for _, _, d in items])),
            n_seqs=len(agg["first_half2"]["loss"]),
            bpc_first_half1=_bpc("first_half1"), bpc_first_half2=_bpc("first_half2"),
            bpc_second_half1=_bpc("second_half1"), bpc_second_half2=_bpc("second_half2"),
            acc_first_half2=_acc("first_half2"), acc_second_half2=_acc("second_half2"),
            delta_bpc_half2=_bpc("first_half2") - _bpc("second_half2"),
            delta_bpc_half1=_bpc("first_half1") - _bpc("second_half1"),
            delta_acc_half2=_acc("second_half2") - _acc("first_half2"),
        ))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 1b — 随机 needle NIAH
# ═══════════════════════════════════════════════════════════════════════════════

def make_needle(rng: np.random.Generator) -> bytes:
    return "".join(rng.choice(list(NEEDLE_CHARSET), size=NEEDLE_LEN)).encode("ascii")


def build_niah_sequence(val: np.ndarray, total_len: int, depth_frac: float,
                        rng: np.random.Generator, max_tries: int = 100):
    """
    seq = base[:p] + b' ' + needle + base[p+nwl : L-nwl] + b' ' + needle
    needle（16 随机字母数字字符）出现两次：中间为记忆写入源，末尾前 8 字符
    为 recall prompt，最后 8 字符为 completion（评分只看这 8 个位置）。
    """
    nwl = NEEDLE_LEN + 1

    for _ in range(max_tries):
        needle = make_needle(rng)
        prefix = needle[:PROMPT_LEN]

        total_avail = len(val) - total_len - 1
        s = int(rng.integers(0, max(1, total_avail)))
        base = val[s : s + total_len]

        if base.tobytes().find(needle) != -1 or base.tobytes().find(prefix) != -1:
            continue

        p = int(depth_frac * (total_len - 2 * nwl))
        p = max(1, min(p, total_len - 2 * nwl))

        seq = base[:p].tobytes() + b" " + needle \
            + base[p + nwl : total_len - nwl].tobytes() + b" " + needle
        assert len(seq) == total_len

        distance = (total_len - nwl) - p - nwl
        return np.frombuffer(seq, dtype=np.uint8).copy(), distance

    raise RuntimeError("无法生成唯一 needle，重试超限")


@torch.no_grad()
def run_niah(model, specs, device: str, desc: str):
    n_comp = NEEDLE_LEN - PROMPT_LEN
    groups = {}
    for (seq_len, depth, seq_np, distance) in specs:
        groups.setdefault((seq_len, depth), []).append((seq_np, distance))

    rows = []
    for (seq_len, depth), items in tqdm(groups.items(), desc=desc, leave=False):
        acc_chars, exact_flags, losses = [], [], []
        for seq_np, distance in tqdm(items, desc=f"  seq={seq_len} d={depth}", leave=False):
            L = len(seq_np)
            inp = torch.from_numpy(seq_np[:-1].astype(np.int64)).unsqueeze(0).to(device)
            tgt = torch.from_numpy(seq_np[1:].astype(np.int64)).unsqueeze(0).to(device)

            try:
                logits = model(inp)
            except Exception as e:
                print(f"    [SKIP] forward 失败 (seq={seq_len}): {e}", flush=True)
                continue

            lgt = logits[0, L - 1 - n_comp : L - 1, :]      # 预测末尾 8 个 completion 字符
            tt  = tgt[0, L - 1 - n_comp : L - 1]
            losses.append(F.cross_entropy(lgt, tt).item())
            acc_chars.append((lgt.argmax(-1) == tt).float().mean().item() * 100.0)
            exact_flags.append(bool((lgt.argmax(-1) == tt).all().item()))

            del logits, inp, tgt

        if not losses:
            continue
        rows.append(dict(
            variant="", seq_len=seq_len, depth_frac=depth,
            distance_chars=int(np.mean([d for _, d in items])),
            n_seqs=len(losses),
            acc_char_pct=float(np.mean(acc_chars)),
            acc_exact_pct=float(np.mean(exact_flags)) * 100.0,
            bpc_completion=float(np.mean(losses)) / math.log(2),
        ))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 2 — 流式超长 BPC
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_streaming(model, specs, device: str, window: int, desc: str):
    window_rows, overall_rows = [], []
    for (total_len, seq_idx, stream_np) in tqdm(specs, desc=desc, leave=False):
        L = len(stream_np)
        inp = torch.from_numpy(stream_np[:-1].astype(np.int64)).unsqueeze(0).to(device)
        tgt = torch.from_numpy(stream_np[1:].astype(np.int64)).unsqueeze(0).to(device)

        try:
            logits = model(inp)
        except Exception as e:
            print(f"  [SKIP] forward 失败 (len={total_len}): {e}", flush=True)
            continue

        per_tok_loss = F.cross_entropy(
            logits[0], tgt[0], reduction="none"
        ).detach().cpu().numpy()
        correct = (logits.argmax(-1)[0] == tgt[0]).detach().cpu().numpy()

        n_win = (L - 1) // window
        first_bpc, last_bpc = None, None
        for w in range(n_win):
            lo, hi = w * window, (w + 1) * window
            w_bpc = per_tok_loss[lo:hi].mean() / math.log(2)
            w_acc = correct[lo:hi].mean() * 100.0
            if w == 0:
                first_bpc = w_bpc
            last_bpc = w_bpc
            window_rows.append(dict(
                variant="", total_len=total_len, seq_idx=seq_idx,
                window_start=lo + 1, window_end=hi,
                bpc=float(w_bpc), acc_pct=float(w_acc),
            ))

        overall_rows.append(dict(
            variant="", total_len=total_len, seq_idx=seq_idx,
            bpc=float(per_tok_loss.mean() / math.log(2)),
            acc_pct=float(correct.mean() * 100.0),
            bpc_first_window=float(first_bpc),
            bpc_last_window=float(last_bpc),
            bpc_drift=float(last_bpc - first_bpc),
        ))
        print(f"    len={total_len} seq#{seq_idx}: BPC={overall_rows[-1]['bpc']:.4f} "
              f"Acc={overall_rows[-1]['acc_pct']:.2f}% "
              f"first-win={first_bpc:.3f} last-win={last_bpc:.3f} "
              f"drift={overall_rows[-1]['bpc_drift']:+.4f}", flush=True)

        del logits, inp, tgt, per_tok_loss, correct
        torch.cuda.empty_cache()

    return window_rows, overall_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 输出：CSV / 汇总 / 绘图
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(rows, path, header):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[h] for h in header])
    print(f"  CSV 已保存: {path}")


def write_repeat_summary(rows, path):
    with open(path, "w") as f:
        f.write("V6 TITANS — 段落重复探针（passage repeat probe）结果汇总\n")
        f.write("64 字符 prose 上下文中出现两次（depth 处 + 末尾），对比两次出现的 BPC/Acc\n")
        f.write("half1 = 段落前 32 字符（段内注意力可见）| half2 = 后 32 字符（仅 Neural Memory 可达）\n")
        f.write("delta_bpc_half2 = first − second > 0 表示记忆检索有贡献（越大越好）\n")
        f.write("delta_acc_half2 = second − first（检索带来的准确率提升，pp）\n\n")
        header = f"{'variant':>14} {'seq_len':>8} {'depth':>6} {'distance':>9} " \
                 f"{'BPC_1st_h2':>10} {'BPC_2nd_h2':>10} {'ΔBPC_h2':>9} " \
                 f"{'Acc_1st_h2':>10} {'Acc_2nd_h2':>10} {'ΔAcc_h2':>8} {'ΔBPC_h1':>8}"
        f.write(header + "\n" + "-" * len(header) + "\n")
        for r in rows:
            f.write(f"{r['variant']:>14} {r['seq_len']:>8} {r['depth_frac']:>6.2f} "
                    f"{r['distance_chars']:>9} {r['bpc_first_half2']:>10.4f} "
                    f"{r['bpc_second_half2']:>10.4f} {r['delta_bpc_half2']:>+9.4f} "
                    f"{r['acc_first_half2']:>10.2f} {r['acc_second_half2']:>10.2f} "
                    f"{r['delta_acc_half2']:>+8.2f} {r['delta_bpc_half1']:>+8.4f}\n")
    print(f"  汇总已保存: {path}")


def write_niah_summary(rows, path):
    with open(path, "w") as f:
        f.write("V6 TITANS — 随机 needle NIAH 结果汇总\n")
        f.write("needle=16 随机字母数字字符, recall prompt=前8字符, completion=后8字符\n")
        f.write("acc_char: completion 逐字符 top-1 | acc_exact: 8 字符全对比例\n\n")
        header = f"{'variant':>14} {'seq_len':>8} {'depth':>6} {'distance':>9} " \
                 f"{'acc_char%':>10} {'acc_exact%':>11} {'bpc_comp':>9}"
        f.write(header + "\n" + "-" * len(header) + "\n")
        for r in rows:
            f.write(f"{r['variant']:>14} {r['seq_len']:>8} {r['depth_frac']:>6.2f} "
                    f"{r['distance_chars']:>9} {r['acc_char_pct']:>10.2f} "
                    f"{r['acc_exact_pct']:>11.2f} {r['bpc_completion']:>9.4f}\n")
    print(f"  汇总已保存: {path}")


def write_stream_summary(overall_rows, path):
    with open(path, "w") as f:
        f.write("V6 TITANS — 流式超长上下文结果汇总\n")
        f.write("bpc_drift = 末窗 BPC - 首窗 BPC（记忆巩固稳定性 / 漂移指标）\n\n")
        header = f"{'variant':>14} {'total_len':>10} {'n_seqs':>7} {'BPC':>8} {'Acc%':>8} " \
                 f"{'BPC@first':>10} {'BPC@last':>9} {'drift':>8}"
        f.write(header + "\n" + "-" * len(header) + "\n")
        groups = {}
        for r in overall_rows:
            groups.setdefault((r["variant"], r["total_len"]), []).append(r)
        for (v, L), items in sorted(groups.items()):
            agg = {k: float(np.mean([r[k] for r in items]))
                   for k in ("bpc", "acc_pct", "bpc_first_window", "bpc_last_window", "bpc_drift")}
            f.write(f"{v:>14} {L:>10} {len(items):>7} {agg['bpc']:>8.4f} {agg['acc_pct']:>8.2f} "
                    f"{agg['bpc_first_window']:>10.4f} {agg['bpc_last_window']:>9.4f} "
                    f"{agg['bpc_drift']:>+8.4f}\n")
    print(f"  汇总已保存: {path}")


def plot_repeat(rows, variants, out_path):
    if not rows:
        return
    depths = sorted({r["depth_frac"] for r in rows})
    lens = sorted({r["seq_len"] for r in rows})
    fig, axes = plt.subplots(1, len(depths) + 1, figsize=(6 * (len(depths) + 1), 5.5),
                             squeeze=False)
    fig.suptitle(
        "V6 TITANS — Passage Repeat Probe (memory retrieval benefit)\n"
        "delta BPC = first_occurrence − second_occurrence on passage 2nd half "
        "(chars 32-63, beyond 36-token attention window → Neural Memory only)\n"
        f"Checkpoint: {CKPT_PATH}", fontsize=11, fontweight="bold")

    # (a-c) delta BPC vs seq_len，按 depth 分面板
    for col, d in enumerate(depths):
        ax = axes[0][col]
        for v in variants:
            pts = sorted([(r["seq_len"], r["delta_bpc_half2"])
                          for r in rows if r["variant"] == v["name"] and r["depth_frac"] == d])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=v["color"], marker=v["marker"], lw=2, ms=7, label=v["label"])
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens)
        ax.set_xticklabels([str(x) for x in lens], fontsize=8)
        ax.set_title(f"depth = {int(d*100)}% of context", fontsize=10, fontweight="bold")
        ax.set_xlabel("context length (chars)")
        ax.set_ylabel("ΔBPC (retrieval benefit, ↑)" if col == 0 else "")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    # (d) BPC first vs second（depth=0.5 为例）
    ax = axes[0][len(depths)]
    dmid = depths[len(depths) // 2]
    for v in variants:
        pts1 = sorted([(r["seq_len"], r["bpc_first_half2"])
                       for r in rows if r["variant"] == v["name"] and r["depth_frac"] == dmid])
        pts2 = sorted([(r["seq_len"], r["bpc_second_half2"])
                       for r in rows if r["variant"] == v["name"] and r["depth_frac"] == dmid])
        if pts1:
            ax.plot([p[0] for p in pts1], [p[1] for p in pts1],
                    color=v["color"], ls="--", marker="x", ms=6, lw=1.5,
                    label=f"{v['name']} first seen")
        if pts2:
            ax.plot([p[0] for p in pts2], [p[1] for p in pts2],
                    color=v["color"], marker=v["marker"], lw=2, ms=7,
                    label=f"{v['name']} repeated")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lens)
    ax.set_xticklabels([str(x) for x in lens], fontsize=8)
    ax.set_title(f"BPC on passage 2nd half (depth={int(dmid*100)}%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("context length (chars)")
    ax.set_ylabel("BPC (↓)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")


def plot_niah(rows, variants, out_path):
    if not rows:
        return
    depths = sorted({r["depth_frac"] for r in rows})
    lens = sorted({r["seq_len"] for r in rows})
    fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 5.5),
                             squeeze=False, sharey=True)
    fig.suptitle(
        "V6 TITANS — Random-needle NIAH (completion char accuracy)\n"
        "(needle 16 random alnum chars; recall by completing last 8 chars; "
        "retrieval via Neural Memory only)\n"
        f"Checkpoint: {CKPT_PATH}", fontsize=11, fontweight="bold")

    for col, d in enumerate(depths):
        ax = axes[0][col]
        for v in variants:
            pts = sorted([(r["seq_len"], r["acc_char_pct"])
                          for r in rows if r["variant"] == v["name"] and r["depth_frac"] == d])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=v["color"], marker=v["marker"], lw=2, ms=7, label=v["label"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens)
        ax.set_xticklabels([str(x) for x in lens], fontsize=8)
        ax.set_ylim(-2, 102)
        ax.set_title(f"needle depth = {int(d*100)}%", fontsize=10, fontweight="bold")
        ax.set_xlabel("context length (chars)")
        ax.set_ylabel("completion char Acc % (↑)" if col == 0 else "")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")


def plot_streaming(window_rows, variants, out_path):
    if not window_rows:
        return
    lens = sorted({r["total_len"] for r in window_rows})
    fig, axes = plt.subplots(1, len(lens), figsize=(6.5 * len(lens), 5.5), squeeze=False)
    fig.suptitle(
        "V6 TITANS — Streaming Long-Context BPC vs Position\n"
        "(window=512 chars; flat curve = stable memory consolidation)",
        fontsize=11, fontweight="bold")

    for col, L in enumerate(lens):
        ax = axes[0][col]
        for v in variants:
            agg = {}
            for r in window_rows:
                if r["variant"] == v["name"] and r["total_len"] == L:
                    agg.setdefault(r["window_start"], []).append(r["bpc"])
            xs = sorted(agg)
            ax.plot(xs, [np.mean(agg[x]) for x in xs],
                    color=v["color"], lw=1.8, label=v["label"])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("position in stream (chars)")
        ax.set_ylabel("BPC (↓)" if col == 0 else "")
        ax.set_title(f"stream length = {L}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, which="both")
        if col == 0:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="V6 zero-training benchmarks: passage-repeat probe + needle NIAH + streaming BPC")
    ap.add_argument("--quick", action="store_true", help="冒烟模式（少序列、短长度）")
    ap.add_argument("--variants", type=str, default="fp32_clean,quant_only,quant_noise",
                    help="逗号分隔的 variant 名")
    ap.add_argument("--skip-repeat", action="store_true", help="跳过段落重复探针")
    ap.add_argument("--skip-needle", action="store_true", help="跳过随机 needle NIAH")
    ap.add_argument("--skip-stream", action="store_true", help="跳过流式超长 BPC")
    ap.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    ap.add_argument("--merge", type=str, default=None, metavar="PARENT_DIR",
                    help="合并模式：读取 PARENT_DIR 下各 <variant>/ 子目录的结果 CSV，"
                         "输出合并的 CSV/汇总/图表到 PARENT_DIR（多 GPU 并行跑完后执行一次）")
    return ap.parse_args()


def _conv(v: str):
    """CSV 字符串 → int/float（失败则原样返回）。"""
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def run_merge(parent_dir: Path):
    """合并模式：汇总各 variant 子目录的结果，输出合并 CSV/汇总/图表。"""
    parent_dir = Path(parent_dir)
    variants_present = []
    rows = {"repeat": [], "niah": [], "stream": [], "stream_overall": []}

    for d in sorted(parent_dir.iterdir()):
        if not d.is_dir() or d.name not in {v["name"] for v in VARIANTS}:
            continue
        variants_present.append(next(v for v in VARIANTS if v["name"] == d.name))
        pairs = [
            ("repeat_results.csv", "repeat"),
            ("niah_results.csv", "niah"),
            ("streaming_results.csv", "stream"),
            ("streaming_overall.csv", "stream_overall"),
        ]
        for fname, key in pairs:
            f = d / fname
            if not f.exists():
                continue
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    r["variant"] = d.name
                    rows[key].append({k: _conv(v) for k, v in r.items()})
            print(f"  读取 {f}")

    if not variants_present:
        print(f"[ERROR] {parent_dir} 下没有 variant 子目录")
        return

    print(f"\n合并 {len(variants_present)} 个 variant: "
          f"{[v['name'] for v in variants_present]}")

    print("\n保存合并结果...")
    if rows["repeat"]:
        save_csv(rows["repeat"], parent_dir / "repeat_results.csv",
                 ["variant", "seq_len", "depth_frac", "distance_chars", "n_seqs",
                  "bpc_first_half1", "bpc_first_half2", "bpc_second_half1",
                  "bpc_second_half2", "acc_first_half2", "acc_second_half2",
                  "delta_bpc_half2", "delta_bpc_half1", "delta_acc_half2"])
        write_repeat_summary(rows["repeat"], parent_dir / "repeat_summary.txt")
        plot_repeat(rows["repeat"], variants_present, parent_dir / "repeat_curves.png")

    if rows["niah"]:
        save_csv(rows["niah"], parent_dir / "niah_results.csv",
                 ["variant", "seq_len", "depth_frac", "distance_chars",
                  "n_seqs", "acc_char_pct", "acc_exact_pct", "bpc_completion"])
        write_niah_summary(rows["niah"], parent_dir / "niah_summary.txt")
        plot_niah(rows["niah"], variants_present, parent_dir / "niah_curves.png")

    if rows["stream"]:
        save_csv(rows["stream"], parent_dir / "streaming_results.csv",
                 ["variant", "total_len", "seq_idx", "window_start", "window_end",
                  "bpc", "acc_pct"])
        save_csv(rows["stream_overall"], parent_dir / "streaming_overall.csv",
                 ["variant", "total_len", "seq_idx", "bpc", "acc_pct",
                  "bpc_first_window", "bpc_last_window", "bpc_drift"])
        write_stream_summary(rows["stream_overall"], parent_dir / "streaming_summary.txt")
        plot_streaming(rows["stream"], variants_present, parent_dir / "streaming_curves.png")

    # 控制台汇总
    if rows["repeat"]:
        print(f"\n{'='*75}\n段落重复探针汇总（合并）\n{'='*75}")
        for r in rows["repeat"]:
            print(f"  {r['variant']:>14} len={r['seq_len']:>6} depth={r['depth_frac']:.2f} "
                  f"ΔBPC={r['delta_bpc_half2']:+.4f} ΔAcc={r['delta_acc_half2']:+.2f}pp")
    if rows["stream_overall"]:
        print(f"\n{'='*75}\nStreaming 汇总（合并）\n{'='*75}")
        for r in rows["stream_overall"]:
            print(f"  {r['variant']:>14} len={r['total_len']:>7} BPC={r['bpc']:.4f} "
                  f"drift={r['bpc_drift']:+.4f}")

    print(f"\n✓ 合并完成。输出在: {parent_dir}/")


def main():
    args = parse_args()

    if args.merge:
        run_merge(Path(args.merge))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_lengths = [512, 2048] if args.quick else REPEAT_LENGTHS
    repeat_n       = 2 if args.quick else REPEAT_NUM_SEQS
    niah_lengths   = [512, 2048] if args.quick else NIAH_LENGTHS
    niah_n         = 2 if args.quick else NIAH_NUM_SEQS
    stream_lengths = [8192, 32768] if args.quick else STREAM_LENGTHS
    stream_n       = 1 if args.quick else STREAM_NUM_SEQS

    variants = [v for v in VARIANTS if v["name"] in set(args.variants.split(","))]
    assert variants, "没有有效的 variant"

    print("=" * 75)
    print("V6 TITANS — 零训练改动 Benchmark 评估（段落探针 + needle NIAH + 流式 BPC）")
    print("=" * 75)
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  Variants   : {[v['name'] for v in variants]}")
    print(f"  Repeat     : lengths={repeat_lengths} depths={REPEAT_DEPTHS} n={repeat_n}"
          f"{' [已跳过]' if args.skip_repeat else ''}")
    print(f"  Needle     : lengths={niah_lengths} depths={NIAH_DEPTHS} n={niah_n}"
          f"{' [已跳过]' if args.skip_needle else ''}")
    print(f"  Streaming  : lengths={stream_lengths} n={stream_n} window={STREAM_WINDOW}"
          f"{' [已跳过]' if args.skip_stream else ''}")
    print(f"  输出目录   : {out_dir}/")
    print("=" * 75)

    if not Path(CKPT_PATH).exists():
        print(f"\n[ERROR] checkpoint 不存在: {CKPT_PATH}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    val = load_val_data()
    print(f"\nval 数据: {len(val):,} bytes | device: {device}")
    rng = np.random.default_rng(SEED)

    # ── 预生成全部评测序列（固定种子，跨 variant 完全一致，保证公平对比）────
    repeat_specs, niah_specs, stream_specs = [], [], []

    if not args.skip_repeat:
        print("\n生成段落重复探针序列...")
        for L in repeat_lengths:
            for d in REPEAT_DEPTHS:
                for i in range(repeat_n):
                    seq, masks, dist = build_repeat_sequence(val, L, d, rng)
                    repeat_specs.append((L, d, seq, masks, dist))
        print(f"  共 {len(repeat_specs)} 条序列")

    if not args.skip_needle:
        print("生成 needle NIAH 序列...")
        for L in niah_lengths:
            for d in NIAH_DEPTHS:
                for i in range(niah_n):
                    seq, dist = build_niah_sequence(val, L, d, rng)
                    niah_specs.append((L, d, seq, dist))
        print(f"  共 {len(niah_specs)} 条序列")

    if not args.skip_stream:
        print("生成流式序列...")
        offset = 0
        for L in stream_lengths:
            for i in range(stream_n):
                stream_specs.append((L, i, val[offset : offset + L].copy()))
                offset += L + 4096
        assert offset < len(val), "val 数据不足以覆盖所有流式序列"
        print(f"  共 {len(stream_specs)} 条流式序列（总 {sum(s[0] for s in stream_specs):,} bytes）")

    # ── 逐 variant 评估（顺序加载，节省显存）───────────────────────────────
    all_repeat_rows, all_niah_rows, all_window_rows, all_overall_rows = [], [], [], []

    for v in variants:
        print(f"\n{'─'*70}\nVariant: {v['name']} — {v['label']}\n{'─'*70}")
        model = load_v6_model(CKPT_PATH, v["quant_noise_cfg"], v["name"], device)

        if not args.skip_repeat:
            t0 = time.time()
            rows = run_repeat_probe(model, repeat_specs, device,
                                    desc=f"Repeat [{v['name']}]")
            for r in rows:
                r["variant"] = v["name"]
            all_repeat_rows.extend(rows)
            print(f"  Repeat probe 完成，耗时 {time.time()-t0:.1f}s，{len(rows)} 个聚合结果")

        if not args.skip_needle:
            t0 = time.time()
            rows = run_niah(model, niah_specs, device, desc=f"NIAH [{v['name']}]")
            for r in rows:
                r["variant"] = v["name"]
            all_niah_rows.extend(rows)
            print(f"  Needle NIAH 完成，耗时 {time.time()-t0:.1f}s，{len(rows)} 个聚合结果")

        if not args.skip_stream:
            t0 = time.time()
            w_rows, o_rows = run_streaming(model, stream_specs, device,
                                           STREAM_WINDOW, desc=f"Stream [{v['name']}]")
            for r in w_rows:
                r["variant"] = v["name"]
            for r in o_rows:
                r["variant"] = v["name"]
            all_window_rows.extend(w_rows)
            all_overall_rows.extend(o_rows)
            print(f"  Streaming 完成，耗时 {time.time()-t0:.1f}s")

        del model
        torch.cuda.empty_cache()

    # ── 保存 ────────────────────────────────────────────────────────────────
    print("\n保存结果...")
    if all_repeat_rows:
        save_csv(all_repeat_rows, out_dir / "repeat_results.csv",
                 ["variant", "seq_len", "depth_frac", "distance_chars", "n_seqs",
                  "bpc_first_half1", "bpc_first_half2", "bpc_second_half1",
                  "bpc_second_half2", "acc_first_half2", "acc_second_half2",
                  "delta_bpc_half2", "delta_bpc_half1", "delta_acc_half2"])
        write_repeat_summary(all_repeat_rows, out_dir / "repeat_summary.txt")
        plot_repeat(all_repeat_rows, variants, out_dir / "repeat_curves.png")

    if all_niah_rows:
        save_csv(all_niah_rows, out_dir / "niah_results.csv",
                 ["variant", "seq_len", "depth_frac", "distance_chars",
                  "n_seqs", "acc_char_pct", "acc_exact_pct", "bpc_completion"])
        write_niah_summary(all_niah_rows, out_dir / "niah_summary.txt")
        plot_niah(all_niah_rows, variants, out_dir / "niah_curves.png")

    if all_window_rows:
        save_csv(all_window_rows, out_dir / "streaming_results.csv",
                 ["variant", "total_len", "seq_idx", "window_start", "window_end",
                  "bpc", "acc_pct"])
        save_csv(all_overall_rows, out_dir / "streaming_overall.csv",
                 ["variant", "total_len", "seq_idx", "bpc", "acc_pct",
                  "bpc_first_window", "bpc_last_window", "bpc_drift"])
        write_stream_summary(all_overall_rows, out_dir / "streaming_summary.txt")
        plot_streaming(all_window_rows, variants, out_dir / "streaming_curves.png")

    # ── 控制台汇总 ──────────────────────────────────────────────────────────
    if all_repeat_rows:
        print(f"\n{'='*75}\n段落重复探针汇总（ΔBPC_half2 > 0 = 记忆检索有贡献）\n{'='*75}")
        for r in all_repeat_rows:
            print(f"  {r['variant']:>14} len={r['seq_len']:>6} depth={r['depth_frac']:.2f} "
                  f"dist={r['distance_chars']:>6} "
                  f"BPC 1st/2nd(h2)={r['bpc_first_half2']:.3f}/{r['bpc_second_half2']:.3f} "
                  f"ΔBPC={r['delta_bpc_half2']:+.4f} ΔAcc={r['delta_acc_half2']:+.2f}pp")
    if all_niah_rows:
        print(f"\n{'='*75}\nNeedle NIAH 汇总\n{'='*75}")
        for r in all_niah_rows:
            print(f"  {r['variant']:>14} len={r['seq_len']:>6} depth={r['depth_frac']:.2f} "
                  f"dist={r['distance_chars']:>6} acc={r['acc_char_pct']:6.2f}% "
                  f"exact={r['acc_exact_pct']:6.2f}% bpc={r['bpc_completion']:.3f}")
    if all_overall_rows:
        print(f"\n{'='*75}\nStreaming 汇总（总体 BPC / drift）\n{'='*75}")
        for r in all_overall_rows:
            print(f"  {r['variant']:>14} len={r['total_len']:>7} BPC={r['bpc']:.4f} "
                  f"Acc={r['acc_pct']:.2f}% drift={r['bpc_drift']:+.4f}")

    print(f"\n✓ 完成。所有输出在: {out_dir}/")


if __name__ == "__main__":
    main()
