import importlib

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.modules.bi_contract import bi_fused_gdn_gating, bi_rms_norm_gated, gdn_contract
from xorl.ops.linear_attention.modules.fused_norm_gate import FusedRMSNormGated


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HV, DV = 32, 128


class _FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))

        return launch


def _rel_err(got: torch.Tensor, exp: torch.Tensor) -> float:
    return float((got.float() - exp.float()).norm() / exp.float().norm().clamp_min(1e-12))


def _gating_inputs(T, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(1, T, HV, generator=gen, device="cuda", dtype=torch.bfloat16).float()
    b = torch.randn(1, T, HV, generator=gen, device="cuda", dtype=torch.bfloat16)
    A_log = torch.empty(HV, device="cuda", dtype=torch.float32).uniform_(0, 2, generator=gen).log()
    dt_bias = torch.rand(HV, device="cuda", dtype=torch.float32, generator=gen)
    return A_log, a, b, dt_bias


def _assert_gating_forward_and_backward_match_reference_composition():
    A_log, a, b, dt_bias = _gating_inputs(512, seed=1)
    actual_A = A_log.clone().requires_grad_(True)
    actual_a = a.clone().requires_grad_(True)
    actual_b = b.clone().requires_grad_(True)
    actual_dt = dt_bias.clone().requires_grad_(True)
    g, beta = bi_fused_gdn_gating(actual_A, actual_a, actual_b, actual_dt)

    reference_A = A_log.clone().requires_grad_(True)
    reference_a = a.clone().requires_grad_(True)
    reference_b = b.clone().requires_grad_(True)
    reference_dt = dt_bias.clone().requires_grad_(True)
    g_ref = -reference_A.exp().view(1, 1, -1) * F.softplus(reference_a + reference_dt.view(1, 1, -1))
    beta_ref = reference_b.float().sigmoid()
    assert g.dtype == torch.float32 and g.shape == g_ref.shape
    # the serving kernel's tl.log(1+tl.exp) vs torch softplus is a 1-ulp fp32 term
    assert torch.allclose(g, g_ref, rtol=1e-5, atol=1e-5)
    # fp32-beta convention: full precision, no bf16 round; tl.sigmoid vs
    # torch.sigmoid is a 1-ulp fp32 term
    assert beta.dtype == torch.float32
    assert torch.allclose(beta, beta_ref, rtol=1e-6, atol=1e-6)
    assert not torch.equal(beta, beta_ref.to(b.dtype).float()), "beta must not be bf16-rounded"
    (g.square() + beta.square()).sum().backward()
    (g_ref.square() + beta_ref.square()).sum().backward()

    for got, exp in (
        (actual_a.grad, reference_a.grad),
        (actual_b.grad, reference_b.grad),
        (actual_A.grad, reference_A.grad),
        (actual_dt.grad, reference_dt.grad),
    ):
        assert _rel_err(got, exp) < 1e-2


def _norm_inputs(T, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(1, T, HV, DV, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
    z = torch.randn(1, T, HV, DV, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(DV, generator=gen, device="cuda", dtype=torch.bfloat16).abs() + 0.5
    return x, z, w


@requires_cuda
@pytest.mark.gpu
def test_gdn_gating_norm_and_exact_model_dispatch_policy():
    _assert_gating_forward_and_backward_match_reference_composition()

    x, z, w = _norm_inputs(256, seed=2)
    eps = 1e-6
    actual_x = x.clone().requires_grad_(True)
    actual_z = z.clone().requires_grad_(True)
    actual_w = w.clone().requires_grad_(True)
    y = bi_rms_norm_gated(actual_x, actual_w, actual_z, eps)

    reference_x = x.clone().requires_grad_(True)
    reference_z = z.clone().requires_grad_(True)
    reference_w = w.clone().requires_grad_(True)
    xf = reference_x.float()
    n = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    y_ref = ((n * reference_w.float()) * (reference_z.float() * torch.sigmoid(reference_z.float()))).to(x.dtype)
    assert y.shape == x.shape and y.dtype == x.dtype
    # rare bf16-ULP tail vs the torch composition is the contracted serving term
    assert torch.allclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2)
    # per-row numerics must not depend on the number of rows in the launch
    y_sub = bi_rms_norm_gated(x[:, :3], w, z[:, :3], eps)
    assert torch.equal(y[:, :3], y_sub)
    y.float().square().sum().backward()
    y_ref.float().square().sum().backward()
    for got, exp in (
        (actual_x.grad, reference_x.grad),
        (actual_z.grad, reference_z.grad),
        (actual_w.grad, reference_w.grad),
    ):
        assert _rel_err(got, exp) < 1e-2

    _assert_fused_rms_norm_gated_module_routes_under_exact_model_program()


def _assert_fused_rms_norm_gated_module_routes_under_exact_model_program():
    x, z, w = _norm_inputs(128, seed=3)
    module = FusedRMSNormGated(DV, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(w)

    with gdn_contract(False):
        y_off = module(x, z)

    with gdn_contract(True):
        y_on = module(x, z)
    assert torch.equal(y_on, bi_rms_norm_gated(x, module.weight, z, module.eps))
    assert torch.allclose(y_on.float(), y_off.float(), rtol=1e-2, atol=1e-2)

    # unverified configs void the bitwise claim and must fail loud
    with gdn_contract(True), pytest.raises(NotImplementedError):
        module(x, z, residual=torch.zeros_like(x))

    _assert_gated_deltanet_gating_routes_under_exact_model_program()


def _assert_gated_deltanet_gating_routes_under_exact_model_program():
    from xorl.models.layers.gated_deltanet import GatedDeltaNet  # noqa: PLC0415

    ordinary = (
        GatedDeltaNet(
            hidden_size=256,
            expand_v=1.0,
            head_dim=64,
            num_heads=2,
            num_v_heads=4,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            norm_eps=1e-6,
            exact_contract=False,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .train()
    )
    exact = (
        GatedDeltaNet(
            hidden_size=256,
            expand_v=1.0,
            head_dim=64,
            num_heads=2,
            num_v_heads=4,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            norm_eps=1e-6,
            exact_contract=True,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .train()
    )
    exact.load_state_dict(ordinary.state_dict())
    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(1, 256, 256, generator=gen, device="cuda", dtype=torch.bfloat16)

    y_off, _, _ = ordinary(hidden)
    y_on, _, _ = exact(hidden)
    assert y_on.shape == y_off.shape
    assert torch.allclose(y_on.float(), y_off.float(), rtol=5e-2, atol=5e-2)


def test_solve_tril_num_warps_pinned():
    # the tl.sum forward substitution reassociates with num_warps; determinism
    # (and the serving contract) requires the pinned 2-warp variant
    import sys  # noqa: PLC0415

    import xorl.ops.linear_attention.ops.utils.solve_tril  # noqa: F401, PLC0415

    mod = sys.modules["xorl.ops.linear_attention.ops.utils.solve_tril"]
    for kernel in (
        mod.solve_tril_16x16_kernel,
        mod.merge_16x16_to_32x32_inverse_kernel,
        mod.merge_16x16_to_64x64_inverse_kernel,
    ):
        for cfg in kernel.fn.configs:
            assert cfg.num_warps == 2

    _assert_kkt_reduction_geometry_matches_the_serving_contract()


def _assert_kkt_reduction_geometry_matches_the_serving_contract():
    module = importlib.import_module("xorl.ops.linear_attention.ops.common.chunk_scaled_dot_kkt")
    contract_kernel = _FakeKernel()
    autotuned_kernel = _FakeKernel()
    original_contract_kernel = module._chunk_scaled_dot_kkt_fwd_kernel
    original_autotuned_kernel = module.chunk_scaled_dot_kkt_fwd_kernel
    module._chunk_scaled_dot_kkt_fwd_kernel = contract_kernel
    module.chunk_scaled_dot_kkt_fwd_kernel = autotuned_kernel
    try:
        k = torch.empty(1, 64, 32, 128)
        g = torch.empty(1, 64, 32)
        beta = torch.empty(1, 64, 32)

        with gdn_contract(True):
            module.chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta)
        assert not autotuned_kernel.calls
        _, kwargs = contract_kernel.calls.pop()
        assert kwargs["BK"] == 64
        assert kwargs["num_warps"] == 8
        assert kwargs["num_stages"] == 3
        assert kwargs["IS_VARLEN"] is False
        assert kwargs["USE_G"] is True
        assert kwargs["SAFE_EXP"] is True

        with gdn_contract(False):
            module.chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta)
        assert not contract_kernel.calls
        _, kwargs = autotuned_kernel.calls.pop()
        assert "BK" not in kwargs
        assert "num_warps" not in kwargs
        assert "num_stages" not in kwargs
    finally:
        module._chunk_scaled_dot_kkt_fwd_kernel = original_contract_kernel
        module.chunk_scaled_dot_kkt_fwd_kernel = original_autotuned_kernel
