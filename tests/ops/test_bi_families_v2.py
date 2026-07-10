"""Families-v2 op gates (N6): redefined frozen reduction trees, default-off.

The v2 kernels are PROVISIONAL until the N6 re-certification completes; these
gates pin the properties that hold regardless of the final bits: explicit-tree
determinism, batch-composition/cross-M invariance, v1-equality where v2 shares
v1's tree (the head GEMM K-chain), and byte-identical vendoring across engines.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from xorl.models.layers import normalization as nz
from xorl.ops import batch_invariant_ops as v1
from xorl.ops import bi_families_v2 as v2
from xorl.ops.loss.bi_fused_lm_head import bi_fused_per_token_ce


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

H, DQK, V = 1024, 128, 20480


def _mk(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype).cuda()


def _ref_rms(x, w, eps=1e-6, residual=None, zero_centered=False):
    xf = x.double()
    if residual is not None:
        xf = (x.float() + residual.float()).to(torch.bfloat16).double()
    var = xf.pow(2).mean(-1, keepdim=True)
    wf = w.double() + 1.0 if zero_centered else w.double()
    return (xf * torch.rsqrt(var + eps) * wf).to(torch.bfloat16)


def _max_ulp(a, b):
    return (a.view(torch.int16).to(torch.int32) - b.view(torch.int16).to(torch.int32)).abs().max().item()


@requires_cuda
def test_norm_v2_matches_fp64_reference_within_1ulp():
    x, r, w = _mk((32, H), 1), _mk((32, H), 2), _mk((H,), 3)
    out, res_out = v2.rms_norm_v2(x, w, residual=r)
    assert _max_ulp(out, _ref_rms(x, w, residual=r)) <= 1
    assert torch.equal(res_out, (x.float() + r.float()).to(torch.bfloat16))
    assert _max_ulp(v2.rms_norm_v2(x, w), _ref_rms(x, w)) <= 1
    assert _max_ulp(v2.rms_norm_v2(x, w, zero_centered=True), _ref_rms(x, w, zero_centered=True)) <= 1


@requires_cuda
def test_norm_v2_batch_composition_invariant():
    x, r, w = _mk((64, H), 4), _mk((64, H), 5), _mk((H,), 6)
    full_o, full_ro = v2.rms_norm_v2(x, w, residual=r)
    for m in (1, 2, 17):
        o, ro = v2.rms_norm_v2(x[:m].contiguous(), w, residual=r[:m].contiguous())
        assert torch.equal(o, full_o[:m]) and torch.equal(ro, full_ro[:m])


@requires_cuda
def test_qk_norm_v2_strided_equals_contiguous_and_inplace():
    T, n_q, n_kv = 64, 8, 2
    packed = _mk((T, (n_q + 2 * n_kv) * DQK), 7)
    qw = _mk((DQK,), 8)
    qview = packed[:, : n_q * DQK]
    out = v2.qk_norm_v2(qview, qw, head_dim=DQK)
    assert torch.equal(out, v2.qk_norm_v2(qview.contiguous(), qw, head_dim=DQK))
    ref = _ref_rms(qview.contiguous().reshape(-1, DQK), qw).reshape(T, n_q * DQK)
    assert _max_ulp(out, ref) <= 1
    scratch = packed.clone()
    sview = scratch[:, : n_q * DQK]
    v2.qk_norm_v2(sview, qw, head_dim=DQK, out=sview)
    assert torch.equal(sview, out)
    assert torch.equal(scratch[:, n_q * DQK :], packed[:, n_q * DQK :])


@requires_cuda
def test_head_v2_logits_bitwise_equal_v1_gemm_and_one_tree():
    hidden, weight = _mk((16, H), 9), _mk((V, H), 10)
    toks = torch.randint(0, V, (16,), device="cuda", generator=None)
    logits, lse_d = v2.head_v2_full_logits_with_lse(hidden, weight)
    ref = torch.empty((16, V), dtype=torch.float32, device="cuda")
    v1._bi_lm_head_chunk_gemm_fp32(hidden, weight.t(), ref)  # the v1 head GEMM
    assert torch.equal(logits, ref)
    lp, lse_s, sel = v2.head_v2_selected_logprob(hidden, weight, toks)
    assert torch.equal(lse_d, lse_s)  # ONE tree for decode and scoring
    assert torch.equal(sel, logits.gather(1, toks[:, None]).squeeze(1))
    lp_d = torch.clamp_max(logits.gather(1, toks[:, None]).squeeze(1) - lse_d, 0.0)
    assert torch.equal(lp_d, lp)


@requires_cuda
def test_head_v2_selected_bitwise_equal_v1_and_cross_m_invariant():
    hidden, weight = _mk((32, H), 11), _mk((V, H), 12)
    toks = torch.arange(32, device="cuda") * 601 % V
    lp, lse, sel = v2.head_v2_selected_logprob(hidden, weight, toks)
    _, _, sel1 = v1.bi_lm_head_selected_logprob(hidden, weight, toks)
    assert torch.equal(sel, sel1)
    ref_lse = torch.logsumexp(
        torch.empty((32, V), dtype=torch.float32, device="cuda").copy_(hidden.float() @ weight.float().t()),
        dim=-1,
    )
    assert (lse - ref_lse).abs().max().item() < 1e-3
    for m in (1, 4):
        lp_m, lse_m, _ = v2.head_v2_selected_logprob(hidden[:m].contiguous(), weight, toks[:m].contiguous())
        assert torch.equal(lp_m, lp[:m]) and torch.equal(lse_m, lse[:m])


@requires_cuda
def test_head_v2_temperature_matches_v1_selected():
    hidden, weight = _mk((8, H), 13), _mk((V, H), 14)
    toks = torch.arange(8, device="cuda") * 917 % V
    temp = torch.full((8,), 0.7, dtype=torch.float32, device="cuda")
    _, _, sel = v2.head_v2_selected_logprob(hidden, weight, toks, temp)
    _, _, sel1 = v1.bi_lm_head_selected_logprob(hidden, weight, toks, temp)
    assert torch.equal(sel, sel1)


def test_families_v2_default_on_with_kill_switch(monkeypatch):
    # The N6 re-anchor: default ON for contract lanes; either env-pair var = 0 kills.
    monkeypatch.delenv("XORL_FAMILIES_V2", raising=False)
    monkeypatch.delenv("SGLANG_FAMILIES_V2", raising=False)
    assert v2.families_v2_enabled()
    monkeypatch.setenv("XORL_FAMILIES_V2", "0")
    assert not v2.families_v2_enabled()
    monkeypatch.setenv("XORL_FAMILIES_V2", "1")
    monkeypatch.setenv("SGLANG_FAMILIES_V2", "0")
    assert not v2.families_v2_enabled()


def test_vendored_module_is_self_contained():
    src = Path(v2.__file__).read_text()
    for banned in ("import xorl", "from xorl", "import sglang", "from sglang"):
        assert banned not in src, f"engine import {banned!r} in the vendored module"


def test_vendored_copies_are_byte_identical_if_sibling_present():
    sibling = os.environ.get("BI_FAMILIES_V2_SIBLING", "")
    if not sibling or not Path(sibling).is_file():
        pytest.skip("sibling engine checkout not available")
    ours = hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest()
    theirs = hashlib.sha256(Path(sibling).read_bytes()).hexdigest()
    assert ours == theirs, "vendored bi_families_v2.py copies drifted"


@requires_cuda
def test_trainer_norm_dispatchers_route_to_v2_under_flag(monkeypatch):
    monkeypatch.setenv("XORL_FAMILIES_V2", "1")
    x, r, w = _mk((16, H), 21), _mk((16, H), 22), _mk((H,), 23)
    assert torch.equal(nz.fast_sglang_rms_norm(x, w, 1e-6), v2.rms_norm_v2(x, w))
    assert torch.equal(nz.fast_batch_invariant_rms_norm(x, w, 1e-6), v2.rms_norm_v2(x, w))
    o, ro = nz.fast_sglang_residual_rms_norm(x, r, w, 1e-6)
    o2, ro2 = v2.rms_norm_v2(x, w, residual=r)
    assert torch.equal(o, o2) and torch.equal(ro, ro2)
    # family unification: both no-residual dispatchers now share ONE tree
    assert torch.equal(nz.fast_sglang_rms_norm(x, w, 1e-6), nz.fast_batch_invariant_rms_norm(x, w, 1e-6))


@requires_cuda
def test_trainer_norm_v2_backward_matches_autograd_reference(monkeypatch):
    monkeypatch.setenv("XORL_FAMILIES_V2", "1")
    x = _mk((8, H), 24).requires_grad_(True)
    r = _mk((8, H), 25).requires_grad_(True)
    w = _mk((H,), 26).requires_grad_(True)
    out, res_out = nz.fast_sglang_residual_rms_norm(x, r, w, 1e-6)
    loss = out.float().square().sum() + res_out.float().sum()
    loss.backward()

    x2 = x.detach().clone().requires_grad_(True)
    r2 = r.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    res_ref = x2 + r2
    ref = nz.sglang_residual_rms_norm(res_ref, w2, 1e-6)
    (ref.float().square().sum() + res_ref.float().sum()).backward()
    for got, want in ((x.grad, x2.grad), (r.grad, r2.grad), (w.grad, w2.grad)):
        assert torch.allclose(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_bi_fused_ce_v2_forward_matches_head_v2_and_backward_runs(monkeypatch):
    monkeypatch.setenv("XORL_FAMILIES_V2", "1")
    hidden = _mk((32, H), 27).requires_grad_(True)
    weight = _mk((V, H), 28).requires_grad_(True)
    labels = (torch.arange(32, device="cuda") * 601 % V).long()
    ce = bi_fused_per_token_ce(hidden, weight, labels)
    lp, _, _ = v2.head_v2_selected_logprob(hidden.detach(), weight.detach(), labels)
    assert torch.equal(ce, -lp)
    ce.sum().backward()

    h2 = hidden.detach().clone().requires_grad_(True)
    w2 = weight.detach().clone().requires_grad_(True)
    ref = torch.nn.functional.cross_entropy(h2.float() @ w2.float().t(), labels, reduction="none")
    ref.sum().backward()
    assert torch.allclose(hidden.grad.float(), h2.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(weight.grad.float(), w2.grad.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_v2_norms_safe_under_nonreentrant_checkpoint(monkeypatch):
    # The §6 trap: aten-interpose norms change saved-tensor metadata between
    # forward and recompute and make use_reentrant=False checkpoint RAISE.
    # v2 norms are module-level autograd Functions — the same kernel runs in
    # both passes, so checkpointing must be green with the flag on.
    monkeypatch.setenv("XORL_FAMILIES_V2", "1")
    x = _mk((16, H), 31).requires_grad_(True)
    r = _mk((16, H), 32).requires_grad_(True)
    w = _mk((H,), 33).requires_grad_(True)

    def block(x, r):
        out, res_out = nz.fast_sglang_residual_rms_norm(x, r, w, 1e-6)
        return nz.fast_batch_invariant_rms_norm(out, w, 1e-6) + res_out

    y = checkpoint(block, x, r, use_reentrant=False)
    y.float().square().sum().backward()
    for g in (x.grad, r.grad, w.grad):
        assert g is not None and torch.isfinite(g.float()).all()
