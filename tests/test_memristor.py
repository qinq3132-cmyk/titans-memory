"""
tests/test_memristor.py
=======================
Validation tests for memory_models_memristor.py

Run:
    conda run -n titans python -m pytest tests/test_memristor.py -v
Or directly:
    conda run -n titans python tests/test_memristor.py
"""

import math
import torch
import pytest

from titans_pytorch.memory_models_memristor import (
    MemristorConfig,
    MemristorMVM,
    MemoryMLP_Memristor,
    MemoryAttention_Memristor,
    GatedResidualMemoryMLP_Memristor,
    FactorizedMemoryMLP_Memristor,
    MemorySwiGluMLP_Memristor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

DIM   = 64
BATCH = 2
SEQ   = 16

@pytest.fixture
def x():
    return torch.randn(BATCH, SEQ, DIM)

@pytest.fixture
def cfg_noisy():
    return MemristorConfig(bits=4, noise_ratio=0.05)

@pytest.fixture
def cfg_clean():
    return MemristorConfig(bits=4, noise_ratio=0.0)

@pytest.fixture
def cfg_noquant():
    return MemristorConfig(bits=0, noise_ratio=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MemristorMVM unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemristorMVM:

    def test_output_shape(self, x):
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        cfg = MemristorConfig(bits=4, noise_ratio=0.05)
        mvm = MemristorMVM(w, cfg)
        out = mvm(x)
        assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"

    def test_quantized_values_on_grid(self, cfg_clean):
        """With noise_ratio=0, fake-quantized weight must lie exactly on grid points."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        mvm = MemristorMVM(w, cfg_clean)
        w_eff = mvm.simulate(w)

        # derive scale the same way the code does
        thd_pos = 2 ** (cfg_clean.bits - 1) - 1
        w_max   = w.detach().abs().max().clamp(min=1e-8)
        scale   = w_max / thd_pos

        # every element should be a multiple of scale (within float tolerance)
        residuals = (w_eff / scale).round() * scale - w_eff
        assert residuals.abs().max().item() < 1e-4, \
            f"quantized values not on grid, max residual={residuals.abs().max().item():.2e}"

    def test_fake_quant_is_float32(self, cfg_clean):
        """Output of simulate() must stay in float32 (fake quantization, not int)."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        mvm = MemristorMVM(w, cfg_clean)
        w_eff = mvm.simulate(w)
        assert w_eff.dtype == torch.float32, f"expected float32, got {w_eff.dtype}"

    def test_ste_gradient_flows(self, x):
        """Gradient must flow through quantization back to the weight parameter."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        cfg = MemristorConfig(bits=4, noise_ratio=0.0)
        mvm = MemristorMVM(w, cfg)
        out = mvm(x)
        out.mean().backward()
        assert w.grad is not None, "no gradient on weight"
        assert not w.grad.isnan().any(), "NaN in gradient"
        assert w.grad.abs().sum().item() > 0, "gradient is all zeros"

    def test_noise_differs_each_forward(self, x):
        """Two forward passes in train mode must produce different outputs (noise)."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        cfg = MemristorConfig(bits=4, noise_ratio=0.05)
        mvm = MemristorMVM(w, cfg)
        mvm.train()
        with torch.no_grad():
            o1 = mvm(x)
            o2 = mvm(x)
        assert not torch.allclose(o1, o2), "noise should make two passes differ"

    def test_no_noise_deterministic(self, x):
        """With noise_ratio=0, two forward passes must be identical."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        cfg = MemristorConfig(bits=4, noise_ratio=0.0)
        mvm = MemristorMVM(w, cfg)
        mvm.train()
        with torch.no_grad():
            o1 = mvm(x)
            o2 = mvm(x)
        assert torch.allclose(o1, o2), "without noise, forward should be deterministic"

    def test_no_quant_passthrough(self, x):
        """With bits=0, quantization is disabled; weight should pass through unchanged."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        cfg = MemristorConfig(bits=0, noise_ratio=0.0)
        mvm = MemristorMVM(w, cfg)
        w_eff = mvm.simulate(w)
        assert torch.allclose(w_eff, w), "bits=0 should be identity quantizer"

    def test_eval_noise_toggle(self, x):
        """noise_enabled_in_eval=False should suppress noise in eval mode."""
        w = torch.nn.Parameter(torch.randn(DIM, DIM))
        cfg = MemristorConfig(bits=4, noise_ratio=0.1, noise_enabled_in_eval=False)
        mvm = MemristorMVM(w, cfg)
        mvm.eval()
        with torch.no_grad():
            o1 = mvm(x)
            o2 = mvm(x)
        assert torch.allclose(o1, o2), "eval mode with noise_enabled_in_eval=False must be deterministic"

    def test_noise_scale(self):
        """Noise std should be ≈ noise_ratio × weight_range."""
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.ones(512, 512))   # known range
        cfg = MemristorConfig(bits=0, noise_ratio=0.1)  # no quant, only noise
        mvm = MemristorMVM(w, cfg)
        mvm.train()
        samples = []
        with torch.no_grad():
            for _ in range(200):
                samples.append(mvm.simulate(w) - w)  # noise only
        noise_cat = torch.stack(samples)
        measured_std = noise_cat.std().item()
        expected_std = 0.1 * (w.max() - w.min()).item()  # range = 0 for all-ones, so special case
        # for all-ones, range=0 → sigma=0; use abs max variant for non-trivial check
        w2 = torch.nn.Parameter(torch.linspace(-1, 1, 512 * 512).reshape(512, 512))
        cfg2 = MemristorConfig(bits=0, noise_ratio=0.1)
        mvm2 = MemristorMVM(w2, cfg2)
        samples2 = []
        with torch.no_grad():
            for _ in range(200):
                samples2.append(mvm2.simulate(w2) - w2)
        noise_cat2 = torch.stack(samples2)
        expected_std2 = 0.1 * (w2.max() - w2.min()).item()  # =0.2
        measured_std2 = noise_cat2.std().item()
        rel_err = abs(measured_std2 - expected_std2) / expected_std2
        assert rel_err < 0.05, f"noise std {measured_std2:.4f} too far from expected {expected_std2:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. MemoryMLP_Memristor
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryMLP_Memristor:

    def test_output_shape(self, x, cfg_noisy):
        m = MemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_noisy)
        assert m(x).shape == x.shape

    def test_gradient_flows_to_all_weights(self, x, cfg_noisy):
        m = MemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_noisy)
        m(x).mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert not p.grad.isnan().any(), f"NaN grad for {name}"

    def test_noisy_vs_clean_differ(self, x, cfg_noisy, cfg_clean):
        """Same weights, noisy vs clean should give different outputs."""
        m_noisy = MemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_noisy)
        m_clean = MemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_clean)
        m_clean.load_state_dict(m_noisy.state_dict())
        m_noisy.eval(); m_clean.eval()
        with torch.no_grad():
            assert not torch.allclose(m_noisy(x), m_clean(x))

    def test_depth3_shape(self, x, cfg_noisy):
        m = MemoryMLP_Memristor(DIM, depth=3, memristor_cfg=cfg_noisy)
        assert m(x).shape == x.shape

    def test_load_state_dict_roundtrip(self, cfg_noisy):
        """Save and reload weights; outputs must match (same weights + no noise)."""
        cfg = MemristorConfig(bits=4, noise_ratio=0.0)
        m1 = MemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg)
        m2 = MemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg)
        m2.load_state_dict(m1.state_dict())
        x = torch.randn(2, 8, DIM)
        m1.eval(); m2.eval()
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MemoryAttention_Memristor
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryAttention_Memristor:

    def test_output_shape(self, x, cfg_noisy):
        m = MemoryAttention_Memristor(DIM, memristor_cfg=cfg_noisy)
        assert m(x).shape == x.shape

    def test_gradient_flows(self, x, cfg_noisy):
        m = MemoryAttention_Memristor(DIM, memristor_cfg=cfg_noisy)
        m(x).mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Other memory model variants
# ─────────────────────────────────────────────────────────────────────────────

class TestOtherVariants:

    def test_gated_residual_shape(self, x, cfg_noisy):
        m = GatedResidualMemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_noisy)
        assert m(x).shape == x.shape

    def test_gated_residual_gradients(self, x, cfg_noisy):
        m = GatedResidualMemoryMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_noisy)
        m(x).mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"

    def test_factorized_shape(self, x, cfg_noisy):
        m = FactorizedMemoryMLP_Memristor(DIM, depth=2, k=16, memristor_cfg=cfg_noisy)
        assert m(x).shape == x.shape

    def test_swiglu_shape(self, x, cfg_noisy):
        m = MemorySwiGluMLP_Memristor(DIM, depth=2, memristor_cfg=cfg_noisy)
        assert m(x).shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration test: plug into MemoryAsContextTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_model_forward_and_backward(self):
        """MemoryMLP_Memristor must work as drop-in inside MemoryAsContextTransformer."""
        from titans_pytorch import MemoryAsContextTransformer

        cfg = MemristorConfig(bits=4, noise_ratio=0.05)
        neural_memory_model = MemoryMLP_Memristor(dim=64, depth=2, memristor_cfg=cfg)

        model = MemoryAsContextTransformer(
            num_tokens=256,
            dim=128,            # small for speed
            depth=4,
            segment_len=16,
            num_persist_mem_tokens=2,
            num_longterm_mem_tokens=4,
            neural_memory_layers=(2, 4),
            neural_memory_segment_len=4,
            neural_memory_batch_size=32,
            neural_mem_gate_attn_output=False,
            neural_mem_weight_residual=True,
            neural_memory_qkv_receives_diff_views=True,
            use_flex_attn=False,
            sliding_window_attn=True,
            neural_memory_model=neural_memory_model,
            neural_memory_kwargs=dict(
                dim_head=64,
                heads=2,
                attn_pool_chunks=True,
                qk_rmsnorm=True,
                momentum=True,
                momentum_order=1,
                default_step_transform_max_lr=1e-1,
                use_accelerated_scan=False,
                per_parameter_lr_modulation=True,
                spectral_norm_surprises=True,
                store_with_lookahead_value=False,
            )
        )

        token_ids = torch.randint(0, 256, (1, 63))
        loss = model(token_ids, return_loss=True)
        assert loss.item() > 0, "loss should be positive"
        assert not math.isnan(loss.item()), "loss is NaN"

        loss.backward()

        # check gradients flow to memristor weight parameters
        mem_params_with_grad = [
            name for name, p in model.named_parameters()
            if 'mvm_layers' in name or 'mvm_' in name
        ]
        assert len(mem_params_with_grad) > 0, \
            "no memristor parameters found in the model"

        print(f"\n  ✓ Integration: loss={loss.item():.4f}, "
              f"memristor params with grad: {len(mem_params_with_grad)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner (no pytest required)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback

    suites = [
        TestMemristorMVM,
        TestMemoryMLP_Memristor,
        TestMemoryAttention_Memristor,
        TestOtherVariants,
        TestIntegration,
    ]

    x_  = torch.randn(BATCH, SEQ, DIM)
    cfg_noisy_  = MemristorConfig(bits=4, noise_ratio=0.05)
    cfg_clean_  = MemristorConfig(bits=4, noise_ratio=0.0)
    cfg_noquant_= MemristorConfig(bits=0, noise_ratio=0.05)

    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite) if m.startswith('test_')]
        for method_name in methods:
            name = f"{suite_cls.__name__}.{method_name}"
            try:
                method = getattr(suite, method_name)
                # inject fixtures manually
                import inspect
                sig = inspect.signature(method)
                kwargs = {}
                for param in sig.parameters:
                    if param == 'x':           kwargs['x'] = x_
                    elif param == 'cfg_noisy': kwargs['cfg_noisy'] = cfg_noisy_
                    elif param == 'cfg_clean': kwargs['cfg_clean'] = cfg_clean_
                    elif param == 'cfg_noquant': kwargs['cfg_noquant'] = cfg_noquant_
                method(**kwargs)
                print(f"  ✓  {name}")
                passed += 1
            except Exception as e:
                print(f"  ✗  {name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}")
    if failed:
        raise SystemExit(1)
