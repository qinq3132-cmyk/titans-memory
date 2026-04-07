"""
公平对比实验：训练三种不同的模型架构
1. TITANS (完整版): Long-term Memory + Neural Memory
2. Without NM: 只有Long-term Memory，无Neural Memory  
3. Vanilla Transformer: 标准Transformer，无任何记忆模块

使用方法:
    python train_comparison.py --model titans --gpu 0
    python train_comparison.py --model without_nm --gpu 0
    python train_comparison.py --model vanilla --gpu 0
"""

import os
import argparse
from datetime import datetime

# 必须在 import torch 之前解析参数并设置 GPU
parser = argparse.ArgumentParser(description='Train models for fair comparison')
parser.add_argument('--model', type=str, required=True, 
                    choices=['titans', 'without_nm', 'vanilla'],
                    help='Model type: titans (full), without_nm (no Neural Memory), vanilla (standard transformer)')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
parser.add_argument('--num_batches', type=int, default=100000, help='Number of training batches')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging (enabled by default)')
args = parser.parse_args()

# 设置 GPU - 必须在 import torch 之前!
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import random
import tqdm
import gzip
import numpy as np
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
)

import wandb

# 设置 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device} (GPU {args.gpu})')

# ============================================
# 训练配置
# ============================================
NUM_BATCHES = args.num_batches
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = args.lr
VALIDATE_EVERY = 100
SEQ_LEN = args.seq_len
USE_WANDB = not args.no_wandb  # 默认启用wandb

# 基础模型配置（三种模型共享）
BASE_CONFIG = {
    'num_tokens': 256,
    'dim': 384,
    'depth': 8,
    'segment_len': 32,
}

# 根据模型类型设置不同的配置
MODEL_TYPE = args.model

# 生成带有日期和参数的文件名前缀
DATE_STR = datetime.now().strftime('%Y%m%d')
EFFECTIVE_BS = BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY
FILE_PREFIX = f"{MODEL_TYPE}_{DATE_STR}_bs{BATCH_SIZE}x{GRADIENT_ACCUMULATE_EVERY}_seq{SEQ_LEN}_lr{LEARNING_RATE}"

# 注意: TITANS模型已在 checkpoints_bs8_acc4_seq512/ 中训练好
# 这里只需要训练 without_nm 和 vanilla 两种模型
if MODEL_TYPE == 'titans':
    print("警告: TITANS模型已经训练好，checkpoint在 checkpoints_bs8_acc4_seq512/")
    print("如果要重新训练，请继续；否则请使用 --model without_nm 或 --model vanilla")

# 创建带有日期和参数的checkpoint目录
CHECKPOINT_DIR = Path(f"./checkpoints/{FILE_PREFIX}")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 初始化wandb
if USE_WANDB:
    wandb.init(
        project='titans-comparison',
        name=f'{DATE_STR}_{MODEL_TYPE}_bs{BATCH_SIZE}x{GRADIENT_ACCUMULATE_EVERY}_seq{SEQ_LEN}',
        config={
            'model_type': MODEL_TYPE,
            'batch_size': BATCH_SIZE,
            'gradient_accumulate_every': GRADIENT_ACCUMULATE_EVERY,
            'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY,
            'learning_rate': LEARNING_RATE,
            'seq_len': SEQ_LEN,
            'num_batches': NUM_BATCHES,
            **BASE_CONFIG,
        }
    )
    print(f"WandB initialized: {wandb.run.name}")
else:
    print("WandB disabled")

print("=" * 80)
print(f"模型类型: {MODEL_TYPE}")
print(f"训练日期: {DATE_STR}")
print(f"Batch Size: {BATCH_SIZE} x {GRADIENT_ACCUMULATE_EVERY} = {EFFECTIVE_BS}")
print(f"Checkpoint目录: {CHECKPOINT_DIR}")
print(f"WandB: {'enabled' if USE_WANDB else 'disabled'}")
print("=" * 80)

# ============================================
# 辅助函数
# ============================================
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    if 32 <= token <= 126:
        return chr(token)
    elif token == 10:
        return '\n'
    elif token == 9:
        return ' '
    else:
        return ' '

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# ============================================
# 创建模型
# ============================================
def create_model(model_type):
    """
    根据模型类型创建不同配置的模型
    
    Args:
        model_type: 'titans', 'without_nm', 或 'vanilla'
    
    Returns:
        model: 创建的模型
        config_desc: 配置描述字符串
    """
    
    if model_type == 'titans':
        # 完整的TITANS模型：Long-term Memory + Neural Memory
        neural_memory_model = MemoryMLP(dim=64, depth=2)
        
        model = MemoryAsContextTransformer(
            **BASE_CONFIG,
            num_persist_mem_tokens=4,          # 永久记忆
            num_longterm_mem_tokens=4,         # 长期记忆
            neural_memory_layers=(2, 4, 6),    # Neural Memory层
            neural_memory_segment_len=4,
            neural_memory_batch_size=128,
            neural_mem_gate_attn_output=False,
            neural_mem_weight_residual=True,
            neural_memory_qkv_receives_diff_views=True,
            use_flex_attn=False,               # 禁用（与混合精度有dtype不匹配问题）
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
                use_accelerated_scan=False,  # 禁用（需要 CUDA_HOME 环境变量）
                per_parameter_lr_modulation=True,
                spectral_norm_surprises=True
            )
        )
        config_desc = "TITANS (Full): Long-term Memory + Neural Memory"
        
    elif model_type == 'without_nm':
        # 没有Neural Memory，但保留Long-term Memory
        model = MemoryAsContextTransformer(
            **BASE_CONFIG,
            num_persist_mem_tokens=4,          # 永久记忆
            num_longterm_mem_tokens=4,         # 长期记忆
            neural_memory_layers=tuple(),      # 空tuple = 不使用Neural Memory
            use_flex_attn=False,               # 禁用（与混合精度有dtype不匹配问题）
            sliding_window_attn=True,
        )
        config_desc = "Without NM: Long-term Memory only (no Neural Memory)"
        
    elif model_type == 'vanilla':
        # 标准Transformer，无任何记忆模块
        model = MemoryAsContextTransformer(
            **BASE_CONFIG,
            num_persist_mem_tokens=0,          # 无永久记忆
            num_longterm_mem_tokens=0,         # 无长期记忆
            neural_memory_layers=tuple(),      # 无Neural Memory
            use_flex_attn=False,               # 禁用（与混合精度有dtype不匹配问题）
            sliding_window_attn=True,
        )
        config_desc = "Vanilla Transformer: No memory modules"
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device), config_desc

# 创建模型
model, config_desc = create_model(MODEL_TYPE)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n配置: {config_desc}")
print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print()

# ============================================
# 准备数据
# ============================================
with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================
# 优化器
# ============================================
optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

# ============================================
# 训练循环
# ============================================
print("开始训练...")
print(f"批次大小: {BATCH_SIZE}, 梯度累积: {GRADIENT_ACCUMULATE_EVERY}")
print(f"有效批次大小: {BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY}")
print(f"序列长度: {SEQ_LEN}")
print(f"总训练步数: {NUM_BATCHES}")
print()

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

best_val_loss = float('inf')

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    
    # 梯度累积
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        data = next(train_loader)
        loss = model(data, return_loss=True)
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()
    
    print(f'training loss: {loss.item():.4f}')
    
    # 梯度裁剪和优化步骤
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    
    # 记录训练指标到wandb
    if USE_WANDB:
        wandb.log({
            'train/loss': loss.item(),
            'train/grad_norm': grad_norm.item(),
            'train/learning_rate': optim.param_groups[0]['lr'],
        }, step=i)
    
    # 验证
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_data = next(val_loader)
            val_loss = model(val_data, return_loss=True)
            print(f'validation loss: {val_loss.item():.4f}')
            
            # 记录验证指标到wandb
            if USE_WANDB:
                wandb.log({
                    'val/loss': val_loss.item(),
                }, step=i)
            
            # 保存最佳模型
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save({
                    'model': model.state_dict(),
                    'step': i,
                    'val_loss': val_loss.item(),
                    'model_type': MODEL_TYPE,
                    'config': BASE_CONFIG,
                    'batch_size': BATCH_SIZE,
                    'seq_len': SEQ_LEN,
                    'learning_rate': LEARNING_RATE,
                }, CHECKPOINT_DIR / f'{FILE_PREFIX}_best.pt')
                
                # 记录最佳模型到wandb
                if USE_WANDB:
                    wandb.run.summary['best_val_loss'] = best_val_loss
                    wandb.run.summary['best_step'] = i
    
    # 定期保存checkpoint
    if i > 0 and i % args.save_every == 0:
        torch.save({
            'model': model.state_dict(),
            'step': i,
            'model_type': MODEL_TYPE,
            'config': BASE_CONFIG,
            'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN,
            'learning_rate': LEARNING_RATE,
        }, CHECKPOINT_DIR / f'{FILE_PREFIX}_step_{i}.pt')
        print(f'Checkpoint saved: {CHECKPOINT_DIR}/{FILE_PREFIX}_step_{i}.pt')
        
        # 记录checkpoint保存到wandb
        if USE_WANDB:
            wandb.log({'checkpoint/step': i}, step=i)

# 保存最终模型
torch.save({
    'model': model.state_dict(),
    'step': NUM_BATCHES,
    'model_type': MODEL_TYPE,
    'config': BASE_CONFIG,
    'batch_size': BATCH_SIZE,
    'seq_len': SEQ_LEN,
    'learning_rate': LEARNING_RATE,
}, CHECKPOINT_DIR / f'{FILE_PREFIX}_final.pt')

print(f"\n训练完成！")
print(f"最终checkpoint: {CHECKPOINT_DIR}/{FILE_PREFIX}_final.pt")
print(f"最佳checkpoint: {CHECKPOINT_DIR}/{FILE_PREFIX}_best.pt")
print(f"最佳验证loss: {best_val_loss:.4f}")

# 记录最终统计到wandb
if USE_WANDB:
    wandb.run.summary['final_step'] = NUM_BATCHES
    wandb.run.summary['best_val_loss'] = best_val_loss
    wandb.run.summary['total_params'] = sum(p.numel() for p in model.parameters())
    wandb.finish()
    print("WandB logging finished")
