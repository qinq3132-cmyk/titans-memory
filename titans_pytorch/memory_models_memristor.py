"""
memory_models_memristor.py
==========================
Drop-in replacements for memory_models.py that simulate memristor
non-idealities on every static matrix-vector multiplication (MVM).

Memristor simulation pipeline (applied to each weight matrix):
    1. Quantize  : map float weights to finite conductance levels (uniform)
    2. Read noise: add per-forward Gaussian noise (σ = noise_ratio × W_range)
    3. STE       : straight-through estimator keeps gradients flowing

Usage
-----
    from titans_pytorch.memory_models_memristor import (
        MemristorConfig,
        MemoryMLP_Memristor,
        MemoryAttention_Memristor,
        GatedResidualMemoryMLP_Memristor,
        FactorizedMemoryMLP_Memristor,
        MemorySwiGluMLP_Memristor,
    )

    cfg = MemristorConfig(bits=4, noise_ratio=0.05)
    mem = MemoryMLP_Memristor(dim=64, depth=2, memristor_cfg=cfg)
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange


# ── helpers (same as original) ────────────────────────────────────────────────

def l2norm(t: Tensor) -> Tensor:
    return F.normalize(t, dim=-1)


# ── norms (identical to original, kept here to keep the file self-contained) ──

class LayerNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        gamma = self.gamma
        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')
        return self.ln(x) * (gamma + 1.)


class ResidualNorm(Module):
    def __init__(self, dim: int, model: Module):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.model(x)) + x


# ══════════════════════════════════════════════════════════════════════════════
# Memristor simulation core
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemristorConfig:
    """
    Parameters that control the memristor non-ideality simulation.

    Attributes
    ----------
    bits : int
        Number of quantization bits (conductance levels = 2**bits).
        Typical values: 2 (ternary-like), 4, 6, 8.
        Set to 0 to disable quantization.
    noise_ratio : float
        Magnitude of read noise relative to the weight range.
        noise_std = noise_ratio * (W_max - W_min)
        Set to 0.0 to disable noise.
    symmetric : bool
        If True,  quantize to [-2^(b-1)+1, 2^(b-1)-1] levels.
        If False, quantize to [-2^(b-1),   2^(b-1)-1] levels.
    noise_enabled_in_training : bool
        Whether to inject read noise during training forward passes.
        Quantization is always applied in both train and eval.
    noise_enabled_in_eval : bool
        Whether to inject read noise during eval / inference.
    """
    bits: int = 4
    noise_ratio: float = 0.05
    symmetric: bool = True
    noise_enabled_in_training: bool = True
    noise_enabled_in_eval: bool = True


class MemristorMVM(Module):
    """
    Replaces a single bare-parameter matrix multiplication  ``x @ W``
    with a memristor-aware version:

        W_q  = Quantize(W)            # finite conductance levels  (STE)
        W_qn = W_q + N(0, σ²)        # per-forward read noise     (detach)
        out  = x @ W_qn

    The weight Parameter ``W`` is registered on this module so that the
    optimizer can update it directly. Pass the original Parameter (or a
    new one) as ``weight`` at construction time.
    """

    def __init__(self, weight: Parameter, cfg: MemristorConfig):
        super().__init__()
        self.weight = weight          # keep the original Parameter reference
        self.cfg = cfg

    # ── quantization (STE) ───────────────────────────────────────────────────

    def _quantize(self, w: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.bits <= 0:
            return w

        if cfg.symmetric:
            thd_pos =  2 ** (cfg.bits - 1) - 1
            thd_neg = -2 ** (cfg.bits - 1) + 1
        else:
            thd_pos =  2 ** (cfg.bits - 1) - 1
            thd_neg = -2 ** (cfg.bits - 1)

        w_max = w.detach().abs().max().clamp(min=1e-8)
        scale = w_max / thd_pos

        # STE: round in forward, identity gradient
        w_int = (w / scale).clamp(thd_neg, thd_pos)
        w_int = w_int + (w_int.round() - w_int).detach()
        return w_int * scale

    # ── read noise ───────────────────────────────────────────────────────────

    def _add_read_noise(self, w: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.noise_ratio == 0.0:
            return w

        inject = (self.training and cfg.noise_enabled_in_training) or \
                 (not self.training and cfg.noise_enabled_in_eval)
        if not inject:
            return w

        w_range = w.detach().max() - w.detach().min()
        sigma = cfg.noise_ratio * w_range
        noise = torch.randn_like(w) * sigma
        # detach noise so it doesn't carry gradients
        return w + noise.detach()

    # ── forward ──────────────────────────────────────────────────────────────

    def simulate(self, w: Tensor) -> Tensor:
        """Return the effective weight after quant + noise."""
        w = self._quantize(w)
        w = self._add_read_noise(w)
        return w

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.simulate(self.weight)


# ══════════════════════════════════════════════════════════════════════════════
# Memristor-aware memory models
# All classes are drop-in replacements for their originals in memory_models.py.
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CFG = MemristorConfig()


class MemoryMLP_Memristor(Module):
    """
    Drop-in replacement for MemoryMLP with memristor MVM simulation.

    Architecture: same as original MemoryMLP
        x → (x @ W0) → GELU → (x @ W1) → … → out
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        expansion_factor: float = 2.,
        memristor_cfg: MemristorConfig = _DEFAULT_CFG,
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        # create raw Parameters (same init as original)
        raw_weights = ParameterList([
            Parameter(torch.randn(dim_in, dim_out))
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        for w in raw_weights:
            nn.init.xavier_uniform_(w)

        # wrap each Parameter in a MemristorMVM module
        self.mvm_layers = ModuleList([
            MemristorMVM(w, memristor_cfg) for w in raw_weights
        ])

    def forward(self, x: Tensor) -> Tensor:
        for ind, mvm in enumerate(self.mvm_layers):
            if ind != 0:
                x = F.gelu(x)
            x = mvm(x)
        return x


class GatedResidualMemoryMLP_Memristor(Module):
    """
    Drop-in replacement for GatedResidualMemoryMLP with memristor MVM simulation.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        expansion_factor: float = 4.,
        memristor_cfg: MemristorConfig = _DEFAULT_CFG,
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.block_mvm = ModuleList()
        for _ in range(depth):
            w1 = Parameter(torch.randn(dim, dim_hidden))
            w2 = Parameter(torch.randn(dim_hidden, dim))
            wg = Parameter(torch.randn(dim * 2, dim))
            for w in (w1, w2, wg):
                nn.init.xavier_uniform_(w)
            self.block_mvm.append(ModuleList([
                MemristorMVM(w1, memristor_cfg),
                MemristorMVM(w2, memristor_cfg),
                MemristorMVM(wg, memristor_cfg),
            ]))

        fp = Parameter(torch.randn(dim, dim))
        nn.init.xavier_uniform_(fp)
        self.final_proj_mvm = MemristorMVM(fp, memristor_cfg)

    def forward(self, x: Tensor) -> Tensor:
        for mvm1, mvm2, mvm_gate in self.block_mvm:
            res = x
            hidden = F.gelu(mvm1(x))
            branch_out = mvm2(hidden)
            gates = mvm_gate(cat((branch_out, res), dim=-1))
            x = res.lerp(branch_out, gates.sigmoid())
        return self.final_proj_mvm(x)


class FactorizedMemoryMLP_Memristor(Module):
    """
    Drop-in replacement for FactorizedMemoryMLP with memristor MVM simulation.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        k: int = 32,
        memristor_cfg: MemristorConfig = _DEFAULT_CFG,
    ):
        super().__init__()
        self.mvm_pairs = ModuleList()
        for _ in range(depth):
            w1 = Parameter(torch.randn(dim, k))
            w2 = Parameter(torch.randn(k, dim))
            nn.init.xavier_uniform_(w1)
            nn.init.xavier_uniform_(w2)
            self.mvm_pairs.append(ModuleList([
                MemristorMVM(w1, memristor_cfg),
                MemristorMVM(w2, memristor_cfg),
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for ind, (mvm1, mvm2) in enumerate(self.mvm_pairs):
            if ind != 0:
                x = F.gelu(x)
            x = mvm2(mvm1(x))
        return x


class MemorySwiGluMLP_Memristor(Module):
    """
    Drop-in replacement for MemorySwiGluMLP with memristor MVM simulation.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        expansion_factor: float = 4.,
        memristor_cfg: MemristorConfig = _DEFAULT_CFG,
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.mvm_blocks = ModuleList()
        for _ in range(depth):
            w1 = Parameter(torch.randn(dim, dim_inner * 2))
            w2 = Parameter(torch.randn(dim_inner, dim))
            nn.init.xavier_uniform_(w1)
            nn.init.xavier_uniform_(w2)
            self.mvm_blocks.append(ModuleList([
                MemristorMVM(w1, memristor_cfg),
                MemristorMVM(w2, memristor_cfg),
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for mvm1, mvm2 in self.mvm_blocks:
            residual = x
            x, gates = mvm1(x).chunk(2, dim=-1)
            x = x * F.gelu(gates)
            x = mvm2(x) + residual
        return self.norm(x)


class MemoryAttention_Memristor(Module):
    """
    Drop-in replacement for MemoryAttention with memristor MVM simulation.
    Simulates: Wq, Wk, Wv, FF_w1, FF_w2 — all 5 weight matrices.
    """

    def __init__(
        self,
        dim: int,
        scale: float = 8.,
        expansion_factor: float = 2.,
        memristor_cfg: MemristorConfig = _DEFAULT_CFG,
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        wq  = Parameter(torch.randn(dim, dim))
        wk  = Parameter(torch.randn(dim, dim))
        wv  = Parameter(torch.randn(dim, dim))
        fw1 = Parameter(torch.randn(dim, dim_ff_hidden))
        fw2 = Parameter(torch.randn(dim_ff_hidden, dim))

        for w in (wq, wk, wv, fw1, fw2):
            nn.init.xavier_uniform_(w)

        self.mvm_q   = MemristorMVM(wq,  memristor_cfg)
        self.mvm_k   = MemristorMVM(wk,  memristor_cfg)
        self.mvm_v   = MemristorMVM(wv,  memristor_cfg)
        self.mvm_ff1 = MemristorMVM(fw1, memristor_cfg)
        self.mvm_ff2 = MemristorMVM(fw2, memristor_cfg)

    def forward(self, x: Tensor) -> Tensor:
        q = l2norm(self.mvm_q(x))
        k = l2norm(self.mvm_k(x))
        v = self.mvm_v(x)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )

        ff_out = self.mvm_ff2(F.gelu(self.mvm_ff1(x)))

        return attn_out + ff_out
