"""Quick script to check generation quality of latest checkpoint."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
os.environ["CPATH"] = _extra_includes + (":" + os.environ["CPATH"] if os.environ.get("CPATH") else "")
_extra_libs = ":".join([
    "/opt/conda/lib",
    f"{_nvidia_base}/cuda_runtime/lib",
    f"{_nvidia_base}/cusparse/lib",
    f"{_nvidia_base}/cublas/lib",
    f"{_nvidia_base}/nvjitlink/lib",
])
os.environ["LD_LIBRARY_PATH"] = _extra_libs + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")

import gzip
import numpy as np
import torch
from pathlib import Path

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# ── Build model (same arch as training) ──
model = MemoryAsContextTransformer(
    num_tokens=256, dim=384, depth=8, segment_len=32,
    num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
    neural_memory_layers=(2, 4, 6),
    neural_memory_segment_len=4, neural_memory_batch_size=128,
    neural_mem_gate_attn_output=False, neural_mem_weight_residual=True,
    neural_memory_qkv_receives_diff_views=True,
    use_flex_attn=False,  # disable for inference simplicity
    sliding_window_attn=True,
    neural_memory_model=MemoryMLP(dim=64, depth=2),
    neural_memory_kwargs=dict(
        dim_head=64, heads=4, attn_pool_chunks=True,
        qk_rmsnorm=True, momentum=True, momentum_order=1,
        default_step_transform_max_lr=1e-1, use_accelerated_scan=False,
        per_parameter_lr_modulation=True, spectral_norm_surprises=True,
        store_with_lookahead_value=False
    )
).cuda()

# ── Load latest checkpoint ──
ckpt_dir = Path('./checkpoints')
ckpts = sorted(ckpt_dir.glob('ckpt_step_*.pt'))
latest = ckpts[-1]
print(f"Loading checkpoint: {latest}")

ckpt = torch.load(latest, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
step = ckpt['step']
print(f"Step: {step}")
del ckpt

model.eval()

# ── Load validation data ──
with gzip.open('./data/enwik8.gz') as f:
    data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, data_val = np.split(data, [int(90e6)])
    data_val = torch.from_numpy(data_val)

# ── Test 1: Compute validation loss / BPC ──
print("\n" + "="*60)
print("VALIDATION LOSS / BPC")
print("="*60)
losses = []
for _ in range(20):
    start = torch.randint(0, data_val.size(0) - 513, (1,))
    seq = data_val[start:start+513].long().unsqueeze(0).cuda()
    with torch.no_grad():
        loss = model(seq, return_loss=True)
    losses.append(loss.item())
avg_loss = sum(losses) / len(losses)
bpc = avg_loss / np.log(2)
print(f"Avg val loss: {avg_loss:.4f}")
print(f"BPC: {bpc:.4f}")
print(f"(enwik8 SOTA ~0.98 BPC, decent ~1.1-1.3 BPC, random ~8.0 BPC)")

# ── Test 2: Generate from wiki-style prompts ──
print("\n" + "="*60)
print("GENERATION SAMPLES")
print("="*60)

prompts = [
    b"The United States of America is a country",
    b"In 1945, the Second World War ended when",
    b"<title>Albert Einstein</title>\n<text>Albert Einstein was",
]

for raw_prompt in prompts:
    prompt_tensor = torch.tensor(list(raw_prompt), dtype=torch.long).cuda()
    prompt_str = raw_prompt.decode('utf-8')
    
    print(f"\n--- PROMPT ({len(raw_prompt)} bytes) ---")
    print(prompt_str)
    print("--- GENERATED ---")
    
    with torch.no_grad():
        sample = model.sample(prompt_tensor[None, ...], seq_len=len(raw_prompt)+256, use_cache=False)
    
    output = decode_tokens(sample[0])
    print(output)
    print()

# ── Test 3: Actual val data continuation ──
print("\n" + "="*60)
print("VAL DATA CONTINUATION (prime=100 bytes)")
print("="*60)

for i in range(3):
    start = torch.randint(0, data_val.size(0) - 600, (1,))
    seq = data_val[start:start+100].long().cuda()
    
    prime_str = decode_tokens(seq)
    print(f"\n--- PRIME #{i+1} ---")
    print(prime_str)
    print("--- CONTINUATION ---")
    
    with torch.no_grad():
        sample = model.sample(seq[None, ...], seq_len=400, use_cache=False)
    
    output = decode_tokens(sample[0])
    print(output)
