import pytest
import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.modules.bi_contract import (
    bi_fused_gdn_gating,
    bi_rms_norm_gated,
    is_gdn_contract_enabled,
)
from xorl.ops.linear_attention.modules.fused_norm_gate import FusedRMSNormGated


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HV, DV = 32, 128


def _rel_err(got: torch.Tensor, exp: torch.Tensor) -> float:
    return float((got.float() - exp.float()).norm() / exp.float().norm().clamp_min(1e-12))


def _gating_inputs(T, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(1, T, HV, generator=gen, device="cuda", dtype=torch.bfloat16).float()
    b = torch.randn(1, T, HV, generator=gen, device="cuda", dtype=torch.bfloat16)
    A_log = torch.empty(HV, device="cuda", dtype=torch.float32).uniform_(0, 2, generator=gen).log()
    dt_bias = torch.rand(HV, device="cuda", dtype=torch.float32, generator=gen)
    return A_log, a, b, dt_bias


@requires_cuda
@pytest.mark.gpu
def test_gating_forward_matches_reference_composition():
    A_log, a, b, dt_bias = _gating_inputs(2048)
    g, beta = bi_fused_gdn_gating(A_log, a, b, dt_bias)
    g_ref = -A_log.exp().view(1, 1, -1) * F.softplus(a + dt_bias.view(1, 1, -1))
    beta_ref = b.float().sigmoid()
    assert g.dtype == torch.float32 and g.shape == g_ref.shape
    # the serving kernel's tl.log(1+tl.exp) vs torch softplus is a 1-ulp fp32 term
    assert torch.allclose(g, g_ref, rtol=1e-5, atol=1e-5)
    # fp32-beta convention: full precision, no bf16 round; tl.sigmoid vs
    # torch.sigmoid is a 1-ulp fp32 term
    assert beta.dtype == torch.float32
    assert torch.allclose(beta, beta_ref, rtol=1e-6, atol=1e-6)
    assert not torch.equal(beta, beta_ref.to(b.dtype).float()), "beta must not be bf16-rounded"


@requires_cuda
@pytest.mark.gpu
def test_gating_backward_matches_autograd_reference():
    A_log, a, b, dt_bias = _gating_inputs(512, seed=1)
    ag = a.clone().requires_grad_(True)
    bg = b.clone().requires_grad_(True)
    Ag = A_log.clone().requires_grad_(True)
    dg = dt_bias.clone().requires_grad_(True)
    g, beta = bi_fused_gdn_gating(Ag, ag, bg, dg)
    (g.square() + beta.square()).sum().backward()

    ar = a.clone().requires_grad_(True)
    br = b.clone().requires_grad_(True)
    Ar = A_log.clone().requires_grad_(True)
    dr = dt_bias.clone().requires_grad_(True)
    g_ref = -Ar.exp().view(1, 1, -1) * F.softplus(ar + dr.view(1, 1, -1))
    beta_ref = br.float().sigmoid()
    (g_ref.square() + beta_ref.square()).sum().backward()

    for got, exp in ((ag.grad, ar.grad), (bg.grad, br.grad), (Ag.grad, Ar.grad), (dg.grad, dr.grad)):
        assert _rel_err(got, exp) < 1e-2


def _norm_inputs(T, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(1, T, HV, DV, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
    z = torch.randn(1, T, HV, DV, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(DV, generator=gen, device="cuda", dtype=torch.bfloat16).abs() + 0.5
    return x, z, w


@requires_cuda
@pytest.mark.gpu
def test_gated_norm_forward_matches_reference_and_is_row_invariant():
    x, z, w = _norm_inputs(1024)
    eps = 1e-6
    y = bi_rms_norm_gated(x, w, z, eps)
    xf = x.float()
    n = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    y_ref = ((n * w.float()) * (z.float() * torch.sigmoid(z.float()))).to(x.dtype)
    assert y.shape == x.shape and y.dtype == x.dtype
    # rare bf16-ULP tail vs the torch composition is the contracted serving term
    assert torch.allclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2)
    # per-row numerics must not depend on the number of rows in the launch
    y_sub = bi_rms_norm_gated(x[:, :3], w, z[:, :3], eps)
    assert torch.equal(y[:, :3], y_sub)


@requires_cuda
@pytest.mark.gpu
def test_gated_norm_backward_matches_autograd_reference():
    x, z, w = _norm_inputs(256, seed=2)
    eps = 1e-6
    xg = x.clone().requires_grad_(True)
    zg = z.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    bi_rms_norm_gated(xg, wg, zg, eps).float().square().sum().backward()

    xr = x.clone().requires_grad_(True)
    zr = z.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    xf = xr.float()
    n = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    ((n * wr.float()) * (zr.float() * torch.sigmoid(zr.float()))).square().sum().backward()

    for got, exp in ((xg.grad, xr.grad), (zg.grad, zr.grad), (wg.grad, wr.grad)):
        assert _rel_err(got, exp) < 1e-2


@requires_cuda
@pytest.mark.gpu
def test_fused_rms_norm_gated_module_routes_under_flag(monkeypatch):
    x, z, w = _norm_inputs(128, seed=3)
    module = FusedRMSNormGated(DV, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(w)

    monkeypatch.delenv("XORL_BI_GDN", raising=False)
    assert not is_gdn_contract_enabled()
    y_off = module(x, z)

    monkeypatch.setenv("XORL_BI_GDN", "1")
    assert is_gdn_contract_enabled()
    y_on = module(x, z)
    assert torch.equal(y_on, bi_rms_norm_gated(x, module.weight, z, module.eps))
    assert torch.allclose(y_on.float(), y_off.float(), rtol=1e-2, atol=1e-2)

    # unverified configs void the bitwise claim and must fail loud
    with pytest.raises(NotImplementedError):
        module(x, z, residual=torch.zeros_like(x))


@requires_cuda
@pytest.mark.gpu
def test_gated_deltanet_gating_routes_under_flag(monkeypatch):
    from xorl.ops.linear_attention.layers.gated_deltanet import GatedDeltaNet  # noqa: PLC0415

    module = (
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
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .train()
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(1, 256, 256, generator=gen, device="cuda", dtype=torch.bfloat16)

    monkeypatch.delenv("XORL_BI_GDN", raising=False)
    y_off, _, _ = module(hidden)
    monkeypatch.setenv("XORL_BI_GDN", "1")
    y_on, _, _ = module(hidden)
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
