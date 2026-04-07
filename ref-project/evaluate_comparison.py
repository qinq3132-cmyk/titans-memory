"""
公平对比评估：比较三种模型在不同序列长度下的性能

评估三种独立训练的模型：
1. TITANS (完整版): Long-term Memory + Neural Memory
2. Without NM: 只有Long-term Memory，无Neural Memory  
3. Vanilla Transformer: 标准Transformer，无任何记忆模块

使用方法:
    python evaluate_comparison.py
"""

import os
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ============================================
# 配置
# ============================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = "./data/enwik8.gz"
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048]
TARGET_EVAL_TOKENS = 50000
EVAL_LAST_N_TOKENS = 64  # 只评估最后N个token，确保公平比较
OUTPUT_DIR = Path("./comparison_fair_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型checkpoint路径
# TITANS使用已训练好的checkpoint，其他两个需要训练
CHECKPOINTS = {
    'titans': './checkpoints_bs8_acc4_seq512_old/model_latest.pt',  # 已训练好的
    'without_nm': './checkpoints_without_nm/model_best.pt',
    'vanilla': './checkpoints_vanilla/model_final.pt',
}

# 基础模型配置
BASE_CONFIG = {
    'num_tokens': 256,
    'dim': 384,
    'depth': 8,
    'segment_len': 32,
}

print("=" * 80)
print("公平对比评估：TITANS vs Without NM vs Vanilla Transformer")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"序列长度: {SEQUENCE_LENGTHS}")
print(f"每个长度评估tokens: {TARGET_EVAL_TOKENS}")
print(f"只评估最后N个token: {EVAL_LAST_N_TOKENS}")
print("=" * 80)


# ============================================
# 创建模型
# ============================================
def create_model(model_type):
    """根据模型类型创建模型"""
    
    if model_type == 'titans':
        neural_memory_model = MemoryMLP(dim=64, depth=2)
        model = MemoryAsContextTransformer(
            **BASE_CONFIG,
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
            neural_memory_model=neural_memory_model,
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
                spectral_norm_surprises=True
            )
        )
        
    elif model_type == 'without_nm':
        model = MemoryAsContextTransformer(
            **BASE_CONFIG,
            num_persist_mem_tokens=4,
            num_longterm_mem_tokens=4,
            neural_memory_layers=tuple(),
            use_flex_attn=False,
            sliding_window_attn=True,
        )
        
    elif model_type == 'vanilla':
        model = MemoryAsContextTransformer(
            **BASE_CONFIG,
            num_persist_mem_tokens=0,
            num_longterm_mem_tokens=0,
            neural_memory_layers=tuple(),
            use_flex_attn=False,
            sliding_window_attn=True,
        )
    
    return model.to(DEVICE)


def load_model(model_type, checkpoint_path):
    """加载模型和checkpoint"""
    if not Path(checkpoint_path).exists():
        print(f"  [WARNING] Checkpoint not found: {checkpoint_path}")
        return None
    
    model = create_model(model_type)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return model


# ============================================
# 数据准备
# ============================================
class TextSamplerDataset(Dataset):
    """固定顺序遍历数据集"""
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        # 计算可以生成多少个完整的序列
        self.num_samples = (len(data) - 1) // seq_len

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len + 1
        return self.data[start:end].long()

    def __len__(self):
        return self.num_samples


def prepare_dataloader(sequence_length, batch_size):
    """准备数据加载器"""
    with gzip.open(DATASET_PATH) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        _, data_val = np.split(data, [int(90e6)])
        data_val = torch.from_numpy(data_val)
    
    dataset = TextSamplerDataset(data_val, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


# ============================================
# 评估函数
# ============================================
def compute_metrics(model, dataloader, target_tokens, eval_last_n=None, desc="Evaluating"):
    """计算评估指标"""
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"  {desc}"):
            if total_tokens >= target_tokens:
                break
            
            data = data.to(DEVICE)
            inp, target = data[:, :-1], data[:, 1:]
            
            logits = model(inp)
            
            # 只评估最后N个token
            if eval_last_n is not None and eval_last_n < target.size(1):
                logits = logits[:, -eval_last_n:, :]
                target = target[:, -eval_last_n:]
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target).sum().item()
            total_correct += correct
            total_tokens += target.numel()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    ppl = np.exp(avg_loss)
    accuracy = (total_correct / total_tokens) * 100
    
    return avg_loss, ppl, accuracy


# ============================================
# 主评估循环
# ============================================
def main():
    # 检查哪些checkpoint存在
    available_models = {}
    for model_type, ckpt_path in CHECKPOINTS.items():
        if Path(ckpt_path).exists():
            available_models[model_type] = ckpt_path
            print(f"✓ {model_type}: {ckpt_path}")
        else:
            print(f"✗ {model_type}: {ckpt_path} (not found)")
    
    if not available_models:
        print("\n没有找到任何checkpoint！请先训练模型。")
        print("训练命令:")
        print("  python train_comparison.py --model titans --gpu 0")
        print("  python train_comparison.py --model without_nm --gpu 0")
        print("  python train_comparison.py --model vanilla --gpu 0")
        return
    
    print(f"\n找到 {len(available_models)} 个模型checkpoint")
    print()
    
    # 存储结果
    results = {model_type: {'seq_lens': [], 'losses': [], 'ppls': [], 'accuracies': []} 
               for model_type in available_models}
    
    # 对每个序列长度进行评估
    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n{'='*80}")
        print(f"序列长度: {seq_len}")
        print(f"{'='*80}")
        
        # 调整批次大小
        if seq_len <= 256:
            batch_size = 4
        elif seq_len <= 512:
            batch_size = 2
        else:
            batch_size = 1
        
        print(f"批次大小: {batch_size}")
        dataloader = prepare_dataloader(seq_len, batch_size)
        
        for model_type, ckpt_path in available_models.items():
            print(f"\n评估模型: {model_type}")
            model = load_model(model_type, ckpt_path)
            
            if model is None:
                continue
            
            loss, ppl, acc = compute_metrics(
                model, dataloader, TARGET_EVAL_TOKENS,
                eval_last_n=EVAL_LAST_N_TOKENS,
                desc=f"{model_type} (seq={seq_len})"
            )
            
            print(f"  Loss: {loss:.4f}, PPL: {ppl:.2f}, Accuracy: {acc:.2f}%")
            
            results[model_type]['seq_lens'].append(seq_len)
            results[model_type]['losses'].append(loss)
            results[model_type]['ppls'].append(ppl)
            results[model_type]['accuracies'].append(acc)
            
            del model
            torch.cuda.empty_cache()
    
    # ============================================
    # 生成可视化
    # ============================================
    print("\n生成可视化...")
    
    colors = {
        'titans': '#2E86AB',      # 蓝色
        'without_nm': '#A23B72',  # 紫色
        'vanilla': '#F18F01',     # 橙色
    }
    labels = {
        'titans': 'TITANS (Full)',
        'without_nm': 'Without NM',
        'vanilla': 'Vanilla Transformer',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fair Comparison: TITANS vs Without NM vs Vanilla Transformer', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: PPL
    ax1 = axes[0]
    for model_type in available_models:
        if results[model_type]['seq_lens']:
            ax1.plot(results[model_type]['seq_lens'], results[model_type]['ppls'],
                    marker='o', linewidth=2, markersize=8,
                    label=labels[model_type], color=colors[model_type])
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax1.set_title('Perplexity Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Accuracy
    ax2 = axes[1]
    for model_type in available_models:
        if results[model_type]['seq_lens']:
            ax2.plot(results[model_type]['seq_lens'], results[model_type]['accuracies'],
                    marker='o', linewidth=2, markersize=8,
                    label=labels[model_type], color=colors[model_type])
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fair_comparison.png', dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {OUTPUT_DIR / 'fair_comparison.png'}")
    
    # ============================================
    # 保存结果摘要
    # ============================================
    summary_path = OUTPUT_DIR / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("公平对比实验结果摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("模型说明:\n")
        f.write("- TITANS (Full): Long-term Memory + Neural Memory\n")
        f.write("- Without NM: 只有Long-term Memory\n")
        f.write("- Vanilla Transformer: 无任何记忆模块\n\n")
        
        f.write("详细结果:\n")
        f.write("-" * 80 + "\n")
        
        for seq_len in SEQUENCE_LENGTHS:
            f.write(f"\n序列长度: {seq_len}\n")
            for model_type in available_models:
                if seq_len in results[model_type]['seq_lens']:
                    idx = results[model_type]['seq_lens'].index(seq_len)
                    f.write(f"  {labels[model_type]:25} - PPL: {results[model_type]['ppls'][idx]:6.2f}, "
                           f"Accuracy: {results[model_type]['accuracies'][idx]:5.2f}%\n")
    
    print(f"摘要已保存: {summary_path}")
    print("\n评估完成！")


if __name__ == "__main__":
    main()
