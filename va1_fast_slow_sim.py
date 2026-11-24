"""Va1: single-device fast/slow differential memory simulation.

This is intentionally independent from the v4/v5 training entry points.  It
models one signed memory weight with a differential conductance pair:

    w_eff = (G_plus_slow + G_plus_fast) - (G_minus_slow + G_minus_fast)

An update is converted to a ternary pulse.  Fast conductance relaxes after
each chunk; repeated pulses crossing the consolidation threshold are moved to
the slow conductance.  A batch contains independent context streams and each
stream/head owns an independent device state.

Examples
--------
    python va1_fast_slow_sim.py --smoke
    python va1_fast_slow_sim.py --steps 32 --batch-size 4 --heads 4 --dim 16

The simulator is deliberately small and deterministic enough for ablation
work.  It is a Va1 physics/algorithm prototype, not a replacement checkpoint
format for train_mac_v5.py.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class Va1Config:
    """Parameters for the fast/slow device dynamics."""

    dim: int = 16
    heads: int = 4
    chunk_len: int = 8
    learning_rate: float = 0.35
    pulse_threshold: float = 0.05
    pulse_gain: float = 0.20
    fast_decay: float = 0.90
    consolidation_threshold: float = 0.80
    consolidation_step: float = 0.10
    conductance_max: float = 2.0
    read_noise: float = 0.0

    def validate(self) -> None:
        if self.dim <= 0 or self.heads <= 0 or self.chunk_len <= 0:
            raise ValueError("dim, heads, and chunk_len must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.pulse_threshold <= 0 or self.pulse_gain <= 0:
            raise ValueError("pulse_threshold and pulse_gain must be positive")
        if not 0.0 <= self.fast_decay < 1.0:
            raise ValueError("fast_decay must be in [0, 1)")
        if self.consolidation_threshold <= 0 or self.consolidation_step <= 0:
            raise ValueError("consolidation parameters must be positive")
        if self.conductance_max <= 0 or self.read_noise < 0:
            raise ValueError("conductance_max must be positive and read_noise non-negative")


class DifferentialFastSlowDevice:
    """A vectorized bank of differential fast/slow devices.

    The first two dimensions are ``batch`` and ``head``.  They are not shared:
    a pulse in one context cannot modify another context's state.
    """

    def __init__(self, config: Va1Config, device: torch.device, dtype: torch.dtype = torch.float32):
        config.validate()
        self.config = config
        self.device = device
        self.dtype = dtype
        self._shape: Optional[tuple[int, int, int, int]] = None
        self.reset(1)

    def reset(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.config.heads, self.config.dim, self.config.dim)
        self._shape = shape
        zeros = lambda: torch.zeros(shape, device=self.device, dtype=self.dtype)
        # Each signed weight is represented by two non-negative conductances.
        self.slow_plus = zeros()
        self.slow_minus = zeros()
        self.fast_plus = zeros()
        self.fast_minus = zeros()
        self.total_pulses = 0
        self.total_consolidations = 0

    @property
    def batch_size(self) -> int:
        if self._shape is None:
            raise RuntimeError("device has not been initialized")
        return self._shape[0]

    def _check_delta(self, delta: torch.Tensor) -> None:
        if self._shape is None or tuple(delta.shape) != self._shape:
            raise ValueError(f"delta must have shape {self._shape}, got {tuple(delta.shape)}")
        if delta.device != self.device:
            raise ValueError(f"delta is on {delta.device}, expected {self.device}")

    def effective_weight(self, noisy: bool = True) -> torch.Tensor:
        """Read the effective signed weight as ``G+ - G-``."""
        weight = (self.slow_plus + self.fast_plus) - (self.slow_minus + self.fast_minus)
        if noisy and self.config.read_noise > 0.0:
            scale = weight.detach().abs().amax().clamp_min(torch.finfo(weight.dtype).tiny)
            weight = weight + torch.randn_like(weight) * (self.config.read_noise * scale)
        return weight

    def _consolidate(self) -> int:
        threshold = self.config.consolidation_threshold
        # Move complete threshold crossings to the persistent state and leave
        # the sub-threshold remainder in the fast state.
        plus_count = torch.floor(self.fast_plus / threshold)
        minus_count = torch.floor(self.fast_minus / threshold)
        self.slow_plus.add_(plus_count * self.config.consolidation_step)
        self.slow_minus.add_(minus_count * self.config.consolidation_step)
        self.fast_plus.sub_(plus_count * threshold)
        self.fast_minus.sub_(minus_count * threshold)
        self.slow_plus.clamp_(0.0, self.config.conductance_max)
        self.slow_minus.clamp_(0.0, self.config.conductance_max)
        return int((plus_count + minus_count).sum().item())

    @torch.no_grad()
    def apply_delta(self, delta: torch.Tensor) -> Dict[str, float]:
        """Apply one chunk update and return pulse/consolidation statistics."""
        self._check_delta(delta)
        cfg = self.config
        pulse = torch.where(
            delta > cfg.pulse_threshold,
            torch.ones_like(delta),
            torch.where(delta < -cfg.pulse_threshold, -torch.ones_like(delta), torch.zeros_like(delta)),
        )

        self.fast_plus.mul_(cfg.fast_decay)
        self.fast_minus.mul_(cfg.fast_decay)
        self.fast_plus.add_(pulse.clamp_min(0.0) * cfg.pulse_gain)
        self.fast_minus.add_((-pulse).clamp_min(0.0) * cfg.pulse_gain)

        pulse_count = int(pulse.ne(0).sum().item())
        consolidation_count = self._consolidate()
        self.total_pulses += pulse_count
        self.total_consolidations += consolidation_count
        return {
            "pulse_count": float(pulse_count),
            "consolidation_count": float(consolidation_count),
            "fast_l1": float((self.fast_plus + self.fast_minus).mean().item()),
            "slow_l1": float((self.slow_plus + self.slow_minus).mean().item()),
        }

    @torch.no_grad()
    def idle(self, steps: int = 1) -> None:
        """Advance time without a pulse (natural fast-state relaxation)."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        factor = self.config.fast_decay ** steps
        self.fast_plus.mul_(factor)
        self.fast_minus.mul_(factor)

    def stats(self) -> Dict[str, float]:
        weight = self.effective_weight(noisy=False)
        return {
            "effective_abs_mean": float(weight.abs().mean().item()),
            "effective_abs_max": float(weight.abs().amax().item()),
            "fast_l1": float((self.fast_plus + self.fast_minus).mean().item()),
            "slow_l1": float((self.slow_plus + self.slow_minus).mean().item()),
            "total_pulses": float(self.total_pulses),
            "total_consolidations": float(self.total_consolidations),
        }


class Va1AssociativeMemory:
    """Chunked key/value memory driven by the Va1 device bank."""

    def __init__(self, config: Va1Config, device: torch.device):
        self.config = config
        self.device = device
        self.device_bank = DifferentialFastSlowDevice(config, device)

    def reset(self, batch_size: int) -> None:
        self.device_bank.reset(batch_size)

    @torch.no_grad()
    def process(self, keys: torch.Tensor, values: torch.Tensor) -> Dict[str, object]:
        """Process a context in temporal chunk order.

        Args:
            keys, values: tensors with shape ``[batch, heads, tokens, dim]``.
                Each context/head has its own dynamic memory state.
        """
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("keys and values must have identical [batch, heads, tokens, dim] shapes")
        batch, heads, tokens, dim = keys.shape
        expected = (self.config.heads, self.config.dim)
        if (heads, dim) != expected:
            raise ValueError(f"expected heads/dim {expected}, got {(heads, dim)}")
        if self.device_bank.batch_size != batch:
            self.reset(batch)
        if tokens < self.config.chunk_len:
            raise ValueError("tokens must be at least chunk_len")

        predictions = []
        losses = []
        chunk_stats = []
        for start in range(0, tokens - self.config.chunk_len + 1, self.config.chunk_len):
            end = start + self.config.chunk_len
            key_chunk = keys[:, :, start:end]
            value_chunk = values[:, :, start:end]
            weight = self.device_bank.effective_weight(noisy=self.config.read_noise > 0.0)
            prediction = torch.einsum("bhcd,bhde->bhce", key_chunk, weight)
            error = value_chunk - prediction
            # Delta is a local least-squares descent direction.  The device
            # itself only receives the ternary representation of this delta.
            delta = self.config.learning_rate * torch.einsum("bhcd,bhce->bhde", key_chunk, error)
            delta = delta / float(self.config.chunk_len)
            stats = self.device_bank.apply_delta(delta)
            predictions.append(prediction)
            losses.append(error.square().mean())
            chunk_stats.append(stats)

        return {
            "predictions": torch.cat(predictions, dim=2),
            "losses": torch.stack(losses),
            "chunk_stats": chunk_stats,
            "device_stats": self.device_bank.stats(),
        }


def make_synthetic_stream(config: Va1Config, batch_size: int, steps: int, device: torch.device,
                          seed: int, repeated: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed key/value task with optional repeated key directions."""
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = steps * config.chunk_len
    target = torch.randn(config.heads, config.dim, config.dim, generator=generator, device=device) / math.sqrt(config.dim)
    if repeated:
        base = torch.randn(batch_size, config.heads, 1, config.dim, generator=generator, device=device)
        keys = base.expand(batch_size, config.heads, tokens, config.dim).clone()
    else:
        keys = torch.randn(batch_size, config.heads, tokens, config.dim, generator=generator, device=device)
    values = torch.einsum("bhcd,hde->bhce", keys, target)
    return keys, values


def run_smoke(device: torch.device) -> Dict[str, float]:
    """Run invariant checks that do not require a GPU or dataset."""
    cfg = Va1Config(dim=4, heads=2, chunk_len=2, pulse_threshold=0.5,
                    pulse_gain=0.30, fast_decay=0.80, consolidation_threshold=0.90,
                    consolidation_step=0.20)
    bank = DifferentialFastSlowDevice(cfg, device)
    delta = torch.zeros((1, cfg.heads, cfg.dim, cfg.dim), device=device)

    # Positive pulse must increase G+ and effective weight.
    delta[..., 0, 0] = 1.0
    before = bank.effective_weight(noisy=False)[0, 0, 0, 0].item()
    bank.apply_delta(delta)
    after = bank.effective_weight(noisy=False)[0, 0, 0, 0].item()
    assert after > before and bank.fast_plus[0, 0, 0, 0] > 0

    # No pulse: fast state relaxes while slow state remains unchanged.
    slow_before = bank.slow_plus.clone()
    fast_before = bank.fast_plus.clone()
    bank.idle(8)
    assert torch.allclose(bank.slow_plus, slow_before)
    assert bank.fast_plus[0, 0, 0, 0] < fast_before[0, 0, 0, 0]

    # Repeated pulses consolidate into slow conductance.
    bank.reset(1)
    for _ in range(8):
        bank.apply_delta(delta)
    assert bank.slow_plus[0, 0, 0, 0] > 0
    assert bank.fast_plus[0, 0, 0, 0] < cfg.consolidation_threshold
    consolidation_probe = bank.stats()

    # Batch streams are isolated: a pulse in context 0 must not appear in
    # context 1, even though both are processed by the same vectorized bank.
    bank.reset(2)
    batch_delta = delta.expand(2, -1, -1, -1).clone()
    batch_delta[1].zero_()
    bank.apply_delta(batch_delta)
    assert bank.fast_plus[0].sum() > 0
    assert bank.fast_plus[1].sum() == 0

    # Negative pulse acts on G- and makes effective weight negative.
    bank.reset(1)
    bank.apply_delta(-delta)
    assert bank.fast_minus[0, 0, 0, 0] > 0
    assert bank.effective_weight(noisy=False)[0, 0, 0, 0] < 0

    memory = Va1AssociativeMemory(cfg, device)
    keys, values = make_synthetic_stream(cfg, batch_size=2, steps=6, device=device, seed=7, repeated=True)
    result = memory.process(keys, values)
    assert result["predictions"].isfinite().all()
    assert result["losses"].isfinite().all()
    assert memory.device_bank.batch_size == 2

    summary = {
        "smoke_loss_first": float(result["losses"][0].item()),
        "smoke_loss_last": float(result["losses"][-1].item()),
        "smoke_pulses": result["device_stats"]["total_pulses"],
        "smoke_stream_consolidations": result["device_stats"]["total_consolidations"],
        "smoke_probe_consolidations": consolidation_probe["total_consolidations"],
        "smoke_probe_slow_l1": consolidation_probe["slow_l1"],
        "smoke_final_fast_l1": result["device_stats"]["fast_l1"],
        "smoke_final_slow_l1": result["device_stats"]["slow_l1"],
    }
    return summary


def run_demo(args: argparse.Namespace) -> Dict[str, object]:
    config = Va1Config(dim=args.dim, heads=args.heads, chunk_len=args.chunk_len,
                       learning_rate=args.learning_rate, pulse_threshold=args.pulse_threshold,
                       pulse_gain=args.pulse_gain, fast_decay=args.fast_decay,
                       consolidation_threshold=args.consolidation_threshold,
                       consolidation_step=args.consolidation_step, read_noise=args.read_noise)
    device = torch.device(args.device)
    memory = Va1AssociativeMemory(config, device)
    keys, values = make_synthetic_stream(config, args.batch_size, args.steps, device,
                                         args.seed, repeated=args.repeated)
    result = memory.process(keys, values)
    summary = {
        "config": asdict(config),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "mean_loss": float(result["losses"].mean().item()),
        "first_loss": float(result["losses"][0].item()),
        "last_loss": float(result["losses"][-1].item()),
        "device_stats": result["device_stats"],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="run invariant checks and exit")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32, help="number of temporal chunks")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--pulse-threshold", type=float, default=0.05)
    parser.add_argument("--pulse-gain", type=float, default=0.20)
    parser.add_argument("--fast-decay", type=float, default=0.90)
    parser.add_argument("--consolidation-threshold", type=float, default=0.80)
    parser.add_argument("--consolidation-step", type=float, default=0.10)
    parser.add_argument("--read-noise", type=float, default=0.0)
    parser.add_argument("--repeated", action="store_true", help="repeat one key direction to stress consolidation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", type=str, default=None, help="optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    summary = run_smoke(device) if args.smoke else run_demo(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
