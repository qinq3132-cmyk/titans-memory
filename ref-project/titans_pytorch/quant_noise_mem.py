"""
quant_noise_mem.py
------------------
对 NeuralMemory retrieve 阶段的动态权重字典施加量化 + 高斯噪声。

直接复用项目根目录下 quantization_noise/quant_util.py 中的：
  - data_quantization()  : 均匀量化（对称，STE 直通梯度）
  - add_noise()          : 叠加有界高斯噪声

使用示例（在 neural_memory.py 的 retrieve_memories 中）:
    from titans_pytorch.quant_noise_mem import quantize_weight_dict

    weights_for_retrieve = quantize_weight_dict(
        dict(weights),
        bit=4,
        noise_scale=0.05,
        noise_method='add',
        noise_range='max',
    )
    values = functional_call(self.memory_model, weights_for_retrieve, queries)
"""

import os
import sys

# 将项目根目录（titans-memory/）加入 sys.path，以便能找到 quantization_noise 包
_this_dir = os.path.dirname(os.path.abspath(__file__))          # ref-project/titans_pytorch/
_project_root = os.path.abspath(os.path.join(_this_dir, '../../'))  # titans-memory/
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from quantization_noise.quant_util import data_quantization, add_noise

import torch


def quantize_weight_dict(
    weights_dict: dict,
    bit: int = 4,
    noise_scale: float = 0.05,
    noise_method: str = 'add',
    noise_range: str = 'max',) -> dict:
    """
    对动态权重字典中的每个张量做均匀量化 + 高斯加噪。

    Args:
        weights_dict  : {param_name: Tensor}，来自 NeuralMemory 的运行时权重快照。
        bit           : 量化位数（默认 4-bit）。
        noise_scale   : 高斯噪声强度，相对于权重绝对最大值的比例（默认 0.05）。
                        设为 0 则只做量化，不加噪声。
        noise_method  : 噪声叠加方式，'add'（加法）或 'mul'（乘法），默认 'add'。
        noise_range   : 噪声幅度基准，'max' / 'std' / 'max_min' / 'maxabs_2'，默认 'max'。

    Returns:
        同结构的新字典，值为量化+加噪后的张量。
        注意：使用 (w_q - w).detach() + w 的 STE 形式，保留梯度通路。
    """
    result = {}
    for name, w in weights_dict.items():
        # 空张量保护（单 token decode 时某些权重张量可能 numel==0，max() 会报错）
        if hasattr(w, 'numel') and w.numel() == 0:
            result[name] = w
            continue
        # 1. 均匀量化（对称，bit 位）
        w_q = data_quantization(w, symmetric=True, bit=bit)

        # 2. 叠加高斯噪声（可选）
        if noise_scale > 0.0:
            w_q = add_noise(w_q, method=noise_method,
                            n_scale=noise_scale, n_range=noise_range)

        # 3. STE：前向用量化噪声权重，反向梯度直通到原始权重
        #    （虽然 store/retrieve 梯度已天然解耦，保留此形式更通用）
        w_q = (w_q - w).detach() + w

        result[name] = w_q
    return result


def quantize_activation(t, bit: int):
    """
    NM 层激活值（DAC 模拟输入 / ADC 模拟输出）的对称均匀量化，STE 直通梯度。

    与 data_quantization 的 symmetric 路径数学一致（data_range=2·max, 
    quant_range=2^bit-2, zero_point=0），但全部使用张量运算、不含 .item()，
    因此可在 torch.func.vmap 内安全使用（store 路径的 per-sample grad 会被
    vmap，data_quantization 里的 .item() 在 vmap 下会报错）。

    物理含义：
      - dac_bits  : 数字投影后的 k/q 向量经 DAC 变为模拟电压输入 VMM
      - adc_bits  : VMM 读出电流和经 ADC 变回数字值 M(k)

    前向 = 量化值，反向梯度直通（与权重侧 STE 形式一致）。
    """
    if bit is None or bit <= 0:
        return t
    th = t.abs().amax()
    # 对称均匀量化：scale = 2·max / (2^bit - 2)，零点居中
    scale = (2.0 * th) / (2**bit - 2) + 1e-12
    q = torch.clamp(
        torch.round(t / scale),
        -(2**(bit - 1) - 1), (2**(bit - 1) - 1),
    ) * scale
    # STE：前向用量化值，反向梯度直通到原始激活
    return t + (q - t).detach()
