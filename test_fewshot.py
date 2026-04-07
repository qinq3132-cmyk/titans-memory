"""Few-shot 对比测试：参考 test_generation 写法，比较无引导 vs few-shot 引导。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

import torch
from pathlib import Path
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

CKPT_PATH = Path("./checkpoints/ckpt_step_097500.pt")
GENERATE_LEN = 220
TEMPERATURE = 0.75
MIN_P = 0.08
NUM_CANDIDATES = 4


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(map(decode_token, tokens))


def score_text(topic: str, text: str) -> float:
    lower = text.lower()
    topic_lower = topic.lower()
    score = 0.0

    if topic_lower in lower:
        score += 3.0

    score += min(2.0, lower.count(topic_lower) * 0.5)

    bad_patterns = [
        "</revision>",
        "<contributor>",
        "<timestamp>",
        "<id>",
        "[[category:",
    ]
    for pattern in bad_patterns:
        if pattern in lower:
            score -= 1.0

    if len(text.strip()) < 40:
        score -= 1.0

    return score


def generate_one(model, prompt_bytes, extra_len=220, temperature=0.75, min_p=0.08):
    prompt_tensor = torch.tensor(list(prompt_bytes), dtype=torch.long).cuda()
    with torch.no_grad():
        continuation = model.sample(
            prompt_tensor[None, ...],
            seq_len=len(prompt_bytes) + extra_len,
            temperature=temperature,
            filter_kwargs=dict(min_p=min_p),
            use_cache=False,
        )

    generated_tokens = continuation[0]
    text = decode_tokens(generated_tokens)
    if text.strip() == "":
        raw = generated_tokens.tolist()
        print(f"  [DEBUG] empty-like text, token_len={len(raw)}, first30={raw[:30]}")
    return text


def generate_best(model, topic, prompt_bytes, extra_len=220, num_candidates=4):
    candidates = []
    temps = [0.65, 0.75, 0.85, 0.95][:num_candidates]
    for t in temps:
        text = generate_one(model, prompt_bytes, extra_len=extra_len, temperature=t, min_p=MIN_P)
        candidates.append((score_text(topic, text), t, text))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_temp, best_text = candidates[0]
    return best_text, best_score, best_temp


print("Loading model...", flush=True)
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
        store_with_lookahead_value=False,
    ),
).cuda()

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
del ckpt
model.eval()
print(f"Model loaded from: {CKPT_PATH}\n", flush=True)


cases = [
    {
        "topic": "Isaac Newton",
        "bare": b"<title>Isaac Newton</title>\n<text>Isaac Newton was",
        "fewshot": """\
<title>Albert Einstein</title>
<text>Albert Einstein was a theoretical physicist known for the theory of relativity.

<title>Marie Curie</title>
<text>Marie Curie was a physicist and chemist known for pioneering research on radioactivity.

<title>Isaac Newton</title>
<text>Isaac Newton was""".encode("utf-8"),
    },
    {
        "topic": "World War II",
        "bare": b"In 1945, the Second World War ended when",
        "fewshot": """\
Question: Who developed the theory of relativity?
Answer: The theory of relativity was developed by Albert Einstein.

Question: What is DNA?
Answer: DNA is the molecule that carries genetic information in living organisms.

Question: When did World War II end?
Answer:""".encode("utf-8"),
    },
    {
        "topic": "DNA",
        "bare": b"<title>DNA</title>\n<text>Deoxyribonucleic acid (DNA) is",
        "fewshot": """\
<title>Cell (biology)</title>
<text>A cell is the basic structural and functional unit of living organisms.

<title>Protein</title>
<text>Proteins are large biomolecules composed of amino acid residues.

<title>DNA</title>
<text>Deoxyribonucleic acid (DNA) is""".encode("utf-8"),
    },
]


for item in cases:
    topic = item["topic"]
    bare_prompt = item["bare"]
    fs_prompt = item["fewshot"]

    print("=" * 68)
    print(f"TOPIC: {topic}")
    print("=" * 68)

    print(f"\n[无引导] prompt ({len(bare_prompt)} bytes):")
    print(bare_prompt.decode("utf-8", errors="replace"))
    print("[GENERATED]")
    bare_text, bare_score, bare_temp = generate_best(
        model, topic, bare_prompt, extra_len=GENERATE_LEN, num_candidates=NUM_CANDIDATES
    )
    print(f"[best score={bare_score:.2f}, temp={bare_temp:.2f}]")
    print(bare_text)

    print(f"\n[Few-shot 引导] prompt ({len(fs_prompt)} bytes):")
    print(fs_prompt.decode("utf-8", errors="replace")[:220] + "...")
    print("[GENERATED]")
    fs_text, fs_score, fs_temp = generate_best(
        model, topic, fs_prompt, extra_len=GENERATE_LEN, num_candidates=NUM_CANDIDATES
    )
    print(f"[best score={fs_score:.2f}, temp={fs_temp:.2f}]")
    print(fs_text)
    print()
