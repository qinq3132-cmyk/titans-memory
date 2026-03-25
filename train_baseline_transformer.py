"""
标准 Transformer baseline 对比训练脚本（多卡 DDP）
用于与 Titans MAC Transformer 进行公平对比：
  - 相同数据集：enwik8
  - 相同训练步数：100K steps
  - 相同 effective batch size：16（4 GPU × batch_size=4 × accum=1）
  - 相似模型规模：dim=384, depth=8（无 NeuralMemory, 无 persistent/longterm memory）
  - 相同优化器：AdoptAtan2, lr=2e-4

启动命令：
    torchrun --nproc_per_node=4 train_baseline_transformer.py

监控：
    tensorboard --logdir=runs_baseline/ --port=6007
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_HOME"] = "/opt/conda"
_nvidia_base = "/opt/conda/envs/titans/lib/python3.10/site-packages/nvidia"
_extra_includes = ":".join([
    f"{_nvidia_base}/cusparse/include",
    f"{_nvidia_base}/cublas/include",
    f"{_nvidia_base}/cufft/include",
    f"{_nvidia_base}/curand/include",
    f"{_nvidia_base}/cusolver/include",
    f"{_nvidia_base}/nvjitlink/include",
    f"{_nvidia_base}/cuda_runtime/include",
])
os.environ["CPATH"] = _extra_includes + (
    ":" + os.environ["CPATH"] if os.environ.get("CPATH") else ""
)
_extra_libs = ":".join([
    "/opt/conda/lib",
    f"{_nvidia_base}/cuda_runtime/lib",
    f"{_nvidia_base}/cusparse/lib",
    f"{_nvidia_base}/cublas/lib",
    f"{_nvidia_base}/nvjitlink/lib",
])
os.environ["LD_LIBRARY_PATH"] = _extra_libs + (
    ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
)

import math
import gzip
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import tqdm
from adam_atan2_pytorch import AdoptAtan2
from rotary_embedding_torch import RotaryEmbedding

# ── 分布式工具 ────────────────────────────────────────────────────────────────

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

# ── 超参数（与 MAC Transformer 保持一致）────────────────────────────────────

NUM_BATCHES             = int(1e5)
BATCH_SIZE              = 4         # 每卡 batch size
GRADIENT_ACCUMULATE_EVERY = 1       # 4 GPU × 4 × 1 = 16，与 Titans 相同
LEARNING_RATE           = 2e-4
VALIDATE_EVERY          = 100
GENERATE_EVERY          = 500
PRIME_LENGTH            = 100
GENERATE_LENGTH         = 512
SEQ_LEN                 = 512

# 模型结构（与 Titans 对齐）
NUM_TOKENS  = 256       # 字节级词表
DIM         = 384       # 隐层维度
DEPTH       = 8         # Transformer 层数
NUM_HEADS   = 8         # 注意力头数（dim_head = 384/8 = 48）
FF_MULT     = 4         # FFN 扩展倍数
DROPOUT     = 0.0       # 与 Titans 保持一致，不加 dropout

# Checkpoint
CHECKPOINT_DIR    = Path('./checkpoints_baseline')
CHECKPOINT_EVERY  = 2500
RESUME_FROM_CHECKPOINT = True

RUN_NAME = 'baseline-transformer-4gpu'

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# ── Checkpoint 工具 ───────────────────────────────────────────────────────────

def save_checkpoint(model, optim, step, path, val_loss=None, best_val_loss=None):
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }, path)
    print(f'  ✓ Checkpoint 已保存: {path}  (step {step})')

def load_latest_checkpoint(checkpoint_dir):
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted(checkpoint_dir.glob('ckpt_step_*.pt'))
    return ckpts[-1] if ckpts else None

# ── 标准 Transformer 模型定义 ─────────────────────────────────────────────────

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
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim_head  = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # RoPE — 与 Titans 保持一致
        self.rotary_emb = RotaryEmbedding(self.dim_head)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.num_heads

        x_normed = self.norm(x)
        qkv = self.to_qkv(x_normed).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, self.dim_head).transpose(1, 2), qkv)

        # 应用 RoPE 旋转位置编码
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # causal mask via PyTorch scaled_dot_product_attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff   = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class BaselineTransformer(nn.Module):
    """
    标准 decoder-only Transformer（GPT 风格）
    - 字节级词表（256）
    - RoPE 旋转位置编码（与 Titans 保持一致，支持任意长度）
    - Pre-Norm 结构（LayerNorm 在 attention/FF 内部）
    - 无 persistent memory，无 neural memory
    """
    def __init__(
        self,
        num_tokens = 256,
        dim        = 384,
        depth      = 8,
        num_heads  = 8,
        ff_mult    = 4,
        dropout    = 0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        # RoPE 在每个 attention 层内部处理位置，无需全局 positional encoding
        self.layers    = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult, dropout)
            for _ in range(depth)
        ])
        self.norm    = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

        # 权重绑定（token embedding 与输出 projection 共享）
        self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, return_loss=False):
        """
        x: (batch, seq_len+1) 若 return_loss=True，最后一位是 target
           (batch, seq_len)   若 return_loss=False
        """
        if return_loss:
            x, target = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)       # (b, seq, dim)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.to_logits(x)  # (b, seq, vocab)

        if return_loss:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            return loss

        return logits

    @torch.no_grad()
    def sample(self, prime, seq_len, temperature=1.0):
        """自回归采样，用于生成文本"""
        self.eval()
        generated = prime.clone()  # (1, prime_len)

        for _ in tqdm.tqdm(range(seq_len - generated.size(1)), desc='generating'):
            logits = self(generated)[:, -1, :]  # (1, vocab)
            if temperature != 1.0:
                logits = logits / temperature
            probs  = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated

# ── 参数量统计 ────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # ── TensorBoard（仅 rank 0）──────────────────────────────────────────
    writer = None
    if is_main_process():
        log_dir = f'runs_baseline/{RUN_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer  = SummaryWriter(log_dir=log_dir)
        print(f'\n{"="*60}')
        print(f'  Baseline Transformer 训练')
        print(f'  TensorBoard: tensorboard --logdir=runs_baseline/ --port=6007')
        print(f'{"="*60}\n')

    # ── 构建模型 ──────────────────────────────────────────────────────────
    model = BaselineTransformer(
        num_tokens  = NUM_TOKENS,
        dim         = DIM,
        depth       = DEPTH,
        num_heads   = NUM_HEADS,
        ff_mult     = FF_MULT,
        dropout     = DROPOUT,
    ).to(device)

    if is_main_process():
        n_params = count_parameters(model)
        print(f'  模型参数量: {n_params:,}  ({n_params/1e6:.2f}M)')

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # ── 数据 ──────────────────────────────────────────────────────────────
    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        data_train, data_val = np.split(data, [int(90e6)])
        data_train, data_val = map(torch.from_numpy, (data_train, data_val))

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data    = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            return self.data[rand_start: rand_start + self.seq_len + 1].long()

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val,   SEQ_LEN)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler, num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              sampler=val_sampler,   num_workers=2,
                              pin_memory=True, drop_last=True)

    def infinite_loader(loader, sampler):
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            for batch in loader:
                yield batch.to(device, non_blocking=True)
            epoch += 1

    train_iter = infinite_loader(train_loader, train_sampler)
    val_iter   = infinite_loader(val_loader,   val_sampler)

    # ── 优化器 ────────────────────────────────────────────────────────────
    optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

    # ── 恢复 Checkpoint ───────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float('inf')
    if RESUME_FROM_CHECKPOINT:
        ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is not None:
            if is_main_process():
                print(f'  ↻ 从 {ckpt_path} 恢复训练')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.module.load_state_dict(ckpt['model_state_dict'])
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            start_step = ckpt['step'] + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            del ckpt
            if is_main_process():
                print(f'    从 step {start_step} 继续, 历史最优 val_loss={best_val_loss:.4f}')
        else:
            if is_main_process():
                print('  ✦ 未找到 checkpoint，从头开始训练')

    # 快进数据迭代器
    if start_step > 0 and is_main_process():
        print(f'  ⏩ 快进数据 {start_step} 步...')
    for _ in range(start_step):
        next(train_iter)
    dist.barrier()

    # ── 训练循环 ──────────────────────────────────────────────────────────
    pbar = tqdm.tqdm(
        range(start_step, NUM_BATCHES), initial=start_step,
        total=NUM_BATCHES, mininterval=10., desc='baseline training',
        disable=not is_main_process()
    )

    for i in pbar:
        model.train()

        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_iter), return_loss=True)
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        train_loss = loss.item()

        if is_main_process():
            pbar.set_postfix(loss=f'{train_loss:.4f}')
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('BPC/train',  train_loss / np.log(2), i)

        # ── 验证 ──────────────────────────────────────────────────────
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = model(next(val_iter), return_loss=True).item()
            if is_main_process():
                val_bpc = val_loss / np.log(2)
                print(f'  [step {i:>6}] train_loss={train_loss:.4f}  '
                      f'val_loss={val_loss:.4f}  val_BPC={val_bpc:.4f}')
                writer.add_scalar('Loss/val', val_loss, i)
                writer.add_scalar('BPC/val',  val_bpc,  i)

                # ── 保存 Best Checkpoint ─────────────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optim, step=i,
                        path=CHECKPOINT_DIR / 'best.pt',
                        val_loss=val_loss, best_val_loss=best_val_loss
                    )
                    print(f'  ★ 新最优! val_loss={val_loss:.4f}  BPC={val_bpc:.4f}')

        # ── 生成样本 ───────────────────────────────────────────────────
        if i % GENERATE_EVERY == 0 and is_main_process():
            model.eval()
            inp = random.choice(val_dataset)[:PRIME_LENGTH].unsqueeze(0).to(device)
            prime_str = decode_tokens(inp[0])
            print(f'\n  [生成] prime: {prime_str!r}')
            sample = model.module.sample(inp, seq_len=GENERATE_LENGTH)
            output_str = decode_tokens(sample[0])
            print(f'  {output_str}\n')
            writer.add_text('Generated', f'**Prime:**\n```\n{prime_str}\n```\n\n**Output:**\n```\n{output_str}\n```', i)

        # ── 保存 Checkpoint ────────────────────────────────────────────
        if i > 0 and i % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optim, step=i,
                            path=CHECKPOINT_DIR / f'ckpt_step_{i:06d}.pt',
                            best_val_loss=best_val_loss)

        dist.barrier()

    # ── 最终 Checkpoint ───────────────────────────────────────────────────
    save_checkpoint(model, optim, step=NUM_BATCHES,
                    path=CHECKPOINT_DIR / f'ckpt_step_{NUM_BATCHES:06d}_final.pt',
                    best_val_loss=best_val_loss)

    if is_main_process():
        writer.close()
        print('\n  训练完成！')

    cleanup_distributed()


if __name__ == '__main__':
    main()
