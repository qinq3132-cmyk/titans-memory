"""
fake_quant_linear.py
--------------------
V7 骨干 QAT 的 fake-quant 机制（实验 10-12 部署阶梯的训练侧适应）。

原理：把骨干中的目标 nn.Linear 原地替换为 FakeQuantLinear——前向时对
**连续 fp32 权重**做 per-channel 对称量化仿真（STE 直通梯度），权重本身
保持连续、由 Adam 更新。部署时再做一次真正的量化，因模型已适应而近无损
（W_t 路径的历史压缩比 ~5×：PTQ 15% → QAT <3%）。

部署映射：训练仿真的量化器 = 部署时忆阻器 4-bit MLC 的 per-channel 量化
（每输出行独立 scale —— 对应阵列每行独立位线参考电压，实验 11 已验证
per-channel 是最优 PTQ 路线）。

curriculum（v4 教训：step 0 全强度注入会崩）：
    bits(step) = end_bits + (start_bits - end_bits) * max(0, 1 - step/warmup)
    从 start_bits（默认 8，扰动 ~0.4% 极小）线性降到 end_bits（默认 4，
    = 部署条件），warmup 默认 5000 步。

分组（与实验 10-12 完全一致）：
    nm   = to_keys/to_values/to_queries + combine_heads/to_attn_logits
    attn = to_qkv / to_out
    ffn  = layers.N.6.*（FFN 两矩阵）
norm / embedding / to_logits / mem tokens 不换（保持 fp32）。

⚠️ 注意：仓库根目录存在旧版 titans_pytorch（无 quant_noise_cfg），导入
本模块的脚本只能把 ref-project 加入 sys.path。
"""

import re

import torch
from torch import nn
import torch.nn.functional as F

# ── 全局步数状态（训练循环每步更新；所有 FakeQuantLinear 实例共享）─────────
_FQ_STATE = {'step': 0}


def set_fq_step(step):
    _FQ_STATE['step'] = int(step)


def get_fq_step():
    return _FQ_STATE['step']


# ── 分组匹配（与实验 10-12 一致）───────────────────────────────────────────
NM_SUFFIX   = ('to_keys.weight', 'to_values.weight', 'to_queries.weight',
               'combine_heads.weight', 'to_attn_logits.weight')
ATTN_SUFFIX = ('to_qkv.weight', 'to_out.weight')
FFN_PAT     = re.compile(r'layers\.\d+\.6\.\d+\.weight$')


def backbone_group_of(param_name):
    """返回参数所属的骨干分组（'nm'/'attn'/'ffn'）或 None。"""
    if param_name.endswith(NM_SUFFIX):
        return 'nm'
    if param_name.endswith(ATTN_SUFFIX):
        return 'attn'
    if FFN_PAT.search(param_name):
        return 'ffn'
    return None


class FakeQuantLinear(nn.Linear):
    """per-channel 对称 fake-quant 线性层（QAT 训练用）。"""

    # curriculum 参数（类级别，swap 时统一设置）
    fq_warmup_steps = 5000
    fq_start_bits   = 8
    fq_end_bits     = 4

    @classmethod
    def current_bits(cls):
        step = _FQ_STATE['step']
        ramp = max(0.0, 1.0 - step / max(1, cls.fq_warmup_steps))
        bits = cls.fq_end_bits + (cls.fq_start_bits - cls.fq_end_bits) * ramp
        return max(cls.fq_end_bits, int(round(bits)))

    def forward(self, x):
        w = self.weight
        bits = type(self).current_bits()
        if bits >= 16:
            return F.linear(x, w, self.bias)
        # per-channel 对称量化：每输出行独立 scale（= 阵列每行独立参考电压）
        scale = w.abs().amax(dim=1, keepdim=True) * 2.0 / (2**bits - 2) + 1e-12
        w_q = torch.clamp(
            torch.round(w / scale),
            -(2**(bits - 1) - 1), (2**(bits - 1) - 1),
        ) * scale
        # STE：前向用量化值，反向梯度直通到连续权重
        w_q = w + (w_q - w).detach()
        return F.linear(x, w_q, self.bias)


@torch.no_grad()
def swap_backbone_fakequant(model, warmup_steps=5000, start_bits=8, end_bits=4,
                            step_state=None):
    """
    把骨干中的目标 nn.Linear 原地替换为 FakeQuantLinear（权重原样保留）。
    必须在 DDP 包装之前调用。返回替换的张量组数量（每层 1 个）。
    """
    FakeQuantLinear.fq_warmup_steps = warmup_steps
    FakeQuantLinear.fq_start_bits   = start_bits
    FakeQuantLinear.fq_end_bits     = end_bits
    if step_state is not None:
        global _FQ_STATE
        _FQ_STATE = step_state

    n = 0
    def _walk(mod, prefix=''):
        nonlocal n
        for cname, cmod in mod.named_children():
            mpath = f"{prefix}.{cname}" if prefix else cname
            wname = f"{mpath}.weight"
            if backbone_group_of(wname) is not None and isinstance(cmod, nn.Linear):
                fq = FakeQuantLinear(cmod.in_features, cmod.out_features,
                                     bias=(cmod.bias is not None))
                fq.weight.data = cmod.weight.data.clone()
                if cmod.bias is not None:
                    fq.bias.data = cmod.bias.data.clone()
                setattr(mod, cname, fq)
                n += 1
            else:
                _walk(cmod, mpath)
    _walk(model)
    return n
