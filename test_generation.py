"""Generation quality showcase — loads ckpt_step_097500.pt."""
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

# ── Load checkpoint ──
CKPT_PATH = Path('./checkpoints/ckpt_step_097500.pt')
print(f"Loading checkpoint: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location='cpu')
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

GENERATE_LEN = 400   # characters to generate per prompt

# ── Prompt categories ────────────────────────────────────────────────────────
prompt_groups = {
    "Wikipedia article openings": [
        b"<title>Isaac Newton</title>\n<text>Isaac Newton was",
        b"<title>World War II</title>\n<text>World War II was a global conflict that",
        b"<title>Python (programming language)</title>\n<text>Python is a high-level",
        b"<title>Black hole</title>\n<text>A black hole is a region of spacetime where",
        b"<title>Renaissance</title>\n<text>The Renaissance was a period in European history",
        b"<title>DNA</title>\n<text>Deoxyribonucleic acid (DNA) is a molecule that",
    ],
    "Encyclopedic facts": [
        b"The capital of France is Paris, which",
        b"The speed of light in a vacuum is approximately",
        b"The human brain contains approximately",
        b"Mount Everest, the tallest mountain on Earth,",
        b"The Roman Empire fell in 476 AD when",
        b"Charles Darwin proposed the theory of evolution",
    ],
    "Historical events": [
        b"In 1969, Neil Armstrong became the first human to",
        b"The French Revolution began in 1789 when",
        b"In 1945, the Second World War ended when",
        b"The Berlin Wall fell in 1989, leading to",
        b"The discovery of penicillin by Alexander Fleming in 1928",
    ],
    "Science & technology": [
        b"The theory of general relativity, developed by Einstein,",
        b"Quantum mechanics describes the behavior of particles at",
        b"The invention of the internet transformed",
        b"CRISPR is a gene-editing technology that allows",
        b"Neural networks are computational models inspired by",
    ],
    "Wiki markup continuation": [
        b"==History==\nThe origins of the",
        b"==References==\n* [[",
        b"{{Infobox scientist\n| name = ",
        b"[[Category:Nobel Prize winners]]\n\n==Biography==\n",
    ],
}

def run_prompts(groups, generate_len):
    for group_name, prompts in groups.items():
        print("\n" + "=" * 70)
        print(f"  {group_name.upper()}")
        print("=" * 70)
        for raw_prompt in prompts:
            prompt_tensor = torch.tensor(list(raw_prompt), dtype=torch.long).cuda()
            prompt_str = raw_prompt.decode('utf-8', errors='replace')
            print(f"\n{'─'*60}")
            print(f"PROMPT ({len(raw_prompt)} bytes): {repr(prompt_str)[:80]}")
            print(f"{'─'*60}")
            with torch.no_grad():
                sample = model.sample(
                    prompt_tensor[None, ...],
                    seq_len=len(raw_prompt) + generate_len,
                    use_cache=False
                )
            # only print the generated portion (after the prompt)
            generated = decode_tokens(sample[0][len(raw_prompt):])
            print(f"[PROMPT] {prompt_str}")
            print(f"[GENERATED] {generated}")

run_prompts(prompt_groups, GENERATE_LEN)

# ── Test 3: Few-shot prompting（上下文引导）────────────────────────────────────
# 原理：模型训练于字符续写，不懂"问答"。
# 通过在 prompt 中预置 2-3 个示例，让模型从上下文中"模仿"格式。
# 示例需要贴近 enwik8（Wikipedia 风格），效果最好。
print("\n\n" + "=" * 70)
print("  FEW-SHOT PROMPTED GENERATION（上下文引导，对比无引导效果）")
print("=" * 70)

# Few-shot 模板：示例尽量贴近 enwik8 的 Wikipedia 写作风格
FEW_SHOT_PREFIX = """\
<title>Albert Einstein</title>
<text>Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest scientists of all time. He developed the theory of relativity.</text>

<title>Marie Curie</title>
<text>Marie Curie (7 November 1867 - 4 July 1934) was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize.</text>

<title>{title}</title>
<text>"""

few_shot_queries = [
    ("Isaac Newton",        "Isaac Newton"),
    ("World War II",        "World War II"),
    ("DNA",                 "DNA"),
    ("The Internet",        "The Internet"),
    ("Black hole",          "Black hole"),
    ("Charles Darwin",      "Charles Darwin"),
]

FEW_SHOT_GENERATE_LEN = 300

for title, display in few_shot_queries:
    prompt_str = FEW_SHOT_PREFIX.replace("{title}", title)
    raw = prompt_str.encode("utf-8")
    prompt_tensor = torch.tensor(list(raw), dtype=torch.long).cuda()

    print(f"\n{'─'*60}")
    print(f"[FEW-SHOT TARGET] {display}")
    print(f"[PROMPT LENGTH]   {len(raw)} bytes")
    print(f"{'─'*60}")

    with torch.no_grad():
        sample = model.sample(
            prompt_tensor[None, ...],
            seq_len=len(raw) + FEW_SHOT_GENERATE_LEN,
            use_cache=False
        )
    generated = decode_tokens(sample[0][len(raw):])
    print(f"[GENERATED] {generated}")

# ── Actual val data continuation ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  REAL VAL DATA CONTINUATION  (prime = 150 bytes → generate 400)")
print("=" * 70)

torch.manual_seed(99)   # fixed seed for reproducible showcase
NUM_VAL_CONTINUATIONS = 8

for i in range(NUM_VAL_CONTINUATIONS):
    start = torch.randint(0, data_val.size(0) - 700, (1,))
    seq = data_val[start : start + 150].long().cuda()

    prime_str = decode_tokens(seq)
    print(f"\n{'─'*60}")
    print(f"PRIME #{i+1}:")
    print(prime_str)
    print("CONTINUATION:")
    with torch.no_grad():
        sample = model.sample(seq[None, ...], seq_len=550, use_cache=False)
    output = decode_tokens(sample[0][150:])
    print(output)
