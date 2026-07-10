"""Contract tests for the explicit RMSNorm kernel-family API.

Two batch-invariant RMSNorm kernel families coexist and disagree at 1 ulp on
rare bf16 boundary values, so a silent family flip against serving seeds K3
divergence (the 2026-07-04 norm-seed incident). These tests pin:

- the family funnel (``bi_rms_norm`` / ``bi_fused_add_rms_norm``) routes each
  family to its exact legacy kernel, bitwise;
- family-declared module calls are bitwise identical to the legacy
  ``force_sglang_residual`` call shapes they replace;
- family violations (residual stream on a no-residual site, fused-add through
  family-1) raise instead of silently flipping;
- undeclared calls in a parity configuration warn loudly (and raise under
  ``XORL_RMSNORM_REQUIRE_FAMILY=1``);
- the two families genuinely differ on the k3(1-ulp)-seed shape, so the
  bitwise gates in this file and the cross-engine file have teeth.
"""

import warnings

import pytest
import torch

import xorl.models.layers.normalization as normalization
from xorl.models.layers.attention.multi_head_attention import MultiHeadAttention
from xorl.models.layers.normalization import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNorm,
    fast_zero_centered_batch_invariant_rms_norm,
    native_zero_centered_rms_norm,
)
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from xorl.ops.batch_invariant_ops import (
    bi_fused_add_rms_norm,
    bi_rms_norm,
    fused_add_rms_norm_batch_invariant,
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
    set_trunk_linear_contract,
    sglang_rms_norm_batch_invariant,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

EPS = 1e-6
# The k3(1-ulp)-seed shape: qk-norm rows x head_dim, where the two families
# disagree on a handful of boundary values for any fixed seed.
SEED_SHAPE = (4096, 128)
HIDDEN_SHAPE = (1024, 2048)


def _make(shape, seed, device="cuda", dtype=torch.bfloat16):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)


# --------------------------------------------------------------------------- #
# Structural guards (CPU).
# --------------------------------------------------------------------------- #
def test_unknown_family_rejected_at_construction_and_call():
    with pytest.raises(ValueError, match="Unknown RMSNorm family"):
        RMSNorm(4, eps=EPS, family="serving_qk")
    norm = RMSNorm(4, eps=EPS)
    with pytest.raises(ValueError, match="Unknown RMSNorm family"):
        norm(torch.ones(1, 4), family="family-3")


def test_no_residual_family_rejects_residual_stream():
    norm = RMSNorm(4, eps=EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
    with pytest.raises(ValueError, match="residual stream"):
        norm(torch.ones(1, 4), residual=torch.ones(1, 4), prenorm=True)


def test_no_residual_family_rejects_residual_tree_force():
    norm = RMSNorm(4, eps=EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
    with pytest.raises(ValueError, match="family flip"):
        norm(torch.ones(1, 4), force_sglang_residual=True)
    with pytest.raises(ValueError, match="family flip"):
        norm(torch.ones(1, 4), force_sglang_residual_kernel=True)


def test_fused_add_through_no_residual_family_raises():
    x = torch.ones(2, 4)
    with pytest.raises(ValueError, match="no fused-add kernel"):
        bi_fused_add_rms_norm(x, x, torch.ones(4), EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)


def test_funnel_rejects_unknown_family():
    x = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Unknown RMSNorm family"):
        bi_rms_norm(x, torch.ones(4), EPS, family="serving")


def test_zero_centered_rejects_residual_tree_family():
    x = torch.ones(2, 4)
    with pytest.raises(ValueError, match="only exists in the 'serving_no_residual' family"):
        bi_rms_norm(x, torch.ones(4), EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE, zero_centered=True)


# --------------------------------------------------------------------------- #
# Modeling declarations: the K3 parity models must declare their families.
# --------------------------------------------------------------------------- #
def test_dense_qwen3_declares_site_families():
    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        _attn_implementation="eager",
    )
    layer0 = Qwen3DecoderLayer(cfg, layer_idx=0)
    layer1 = Qwen3DecoderLayer(cfg, layer_idx=1)
    assert layer0.input_layernorm.family == RMS_NORM_FAMILY_NO_RESIDUAL
    assert layer1.input_layernorm.family == RMS_NORM_FAMILY_RESIDUAL_TREE
    assert layer0.post_attention_layernorm.family == RMS_NORM_FAMILY_RESIDUAL_TREE
    assert layer0.self_attn.q_norm.family == RMS_NORM_FAMILY_NO_RESIDUAL
    assert layer0.self_attn.k_norm.family == RMS_NORM_FAMILY_NO_RESIDUAL


def test_shared_attention_qk_norms_declare_no_residual_family():
    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        _attn_implementation="eager",
    )
    attn = MultiHeadAttention(cfg, layer_idx=0)
    assert attn.q_norm.family == RMS_NORM_FAMILY_NO_RESIDUAL
    assert attn.k_norm.family == RMS_NORM_FAMILY_NO_RESIDUAL


# --------------------------------------------------------------------------- #
# Loud tripwire for undeclared parity-lane calls.
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
def test_undeclared_family_warns_in_parity_lane():
    normalization._WARNED_UNDECLARED_FAMILY.clear()
    norm = RMSNorm(64, eps=EPS, mode="sglang_fused").to("cuda")
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True), torch.no_grad():
        with pytest.warns(UserWarning, match="without a declared kernel family"):
            norm(x)
    # Warned once per (mode, call-shape); a second call is silent.
    normalization._WARNED_UNDECLARED_FAMILY.clear()


@requires_cuda
@pytest.mark.gpu
def test_undeclared_family_raises_when_required(monkeypatch):
    monkeypatch.setenv("XORL_RMSNORM_REQUIRE_FAMILY", "1")
    norm = RMSNorm(64, eps=EPS, mode="sglang_fused").to("cuda")
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True), torch.no_grad():
        with pytest.raises(RuntimeError, match="without a declared kernel family"):
            norm(x)
        # Declared calls pass.
        norm(x, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        norm(x, family=RMS_NORM_FAMILY_RESIDUAL_TREE)


@requires_cuda
@pytest.mark.gpu
def test_declared_and_legacy_calls_do_not_warn():
    norm = RMSNorm(64, eps=EPS, mode="sglang_fused", family=RMS_NORM_FAMILY_NO_RESIDUAL).to("cuda")
    tree = RMSNorm(64, eps=EPS, mode="sglang_fused").to("cuda")
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True), torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error")
        norm(x)
        tree(x, force_sglang_residual=True)  # legacy-explicit stays silent


# --------------------------------------------------------------------------- #
# Bitwise: the funnel routes each family to its exact legacy kernel, the
# family-declared module calls match the legacy call shapes, and the two
# families genuinely differ on the seed shape.
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [SEED_SHAPE, HIDDEN_SHAPE])
def test_funnel_matches_legacy_kernels_bitwise(shape):
    x = _make(shape, 0)
    w = _make((shape[-1],), 300)
    with torch.no_grad():
        assert torch.equal(
            bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL),
            rms_norm_batch_invariant(x, w, EPS),
        )
        assert torch.equal(
            bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE),
            sglang_rms_norm_batch_invariant(x, w, EPS),
        )
        r = _make(shape, 1)
        out, rout = bi_fused_add_rms_norm(x, r, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
        ref_out, ref_rout = fused_add_rms_norm_batch_invariant(x, r, w, EPS)
        assert torch.equal(out, ref_out)
        assert torch.equal(rout, ref_rout)


@requires_cuda
@pytest.mark.gpu
def test_zero_centered_twin_is_family1_with_fold():
    """The Qwen3.5 zero-centered twin (#468) registered in the family API: the
    funnel's zero_centered form, the differentiable wrapper, the raw family-1
    kernel on the folded operands, and the interpose lane all agree bitwise."""
    x = _make(SEED_SHAPE, 5)
    w = _make((SEED_SHAPE[-1],), 302)
    with torch.no_grad():
        funnel = bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True)
        wrapper = fast_zero_centered_batch_invariant_rms_norm(x, w, EPS)
        raw = rms_norm_batch_invariant(x.float(), 1.0 + w.float(), EPS).type_as(x)
        with set_batch_invariant_mode(True):
            interpose = native_zero_centered_rms_norm(x, w, EPS)
    assert torch.equal(funnel, wrapper)
    assert torch.equal(funnel, raw)
    assert torch.equal(funnel, interpose), "zero-centered twin diverged from the interpose lane"


@requires_cuda
@pytest.mark.gpu
def test_trunk_contract_lane_dispatches_family1_and_warns_undeclared():
    """#467 composition: under the scoped trunk-contract lane (global interpose
    off), no-residual sglang_fused dispatch is the family-1 kernel with real
    gradients -- for declared serving_no_residual sites without a warning, and
    for undeclared sites with the loud tripwire."""
    x = _make(SEED_SHAPE, 0)
    w = _make((SEED_SHAPE[-1],), 300)
    declared = RMSNorm(SEED_SHAPE[-1], eps=EPS, mode="sglang_fused", family=RMS_NORM_FAMILY_NO_RESIDUAL).to("cuda")
    undeclared = RMSNorm(SEED_SHAPE[-1], eps=EPS, mode="sglang_fused").to("cuda")
    for m in (declared, undeclared):
        with torch.no_grad():
            m.weight.copy_(w)
    normalization._WARNED_UNDECLARED_FAMILY.clear()
    set_trunk_linear_contract(True)
    try:
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                out_declared = declared(x)
            with pytest.warns(UserWarning, match="without a declared kernel family"):
                out_undeclared = undeclared(x)
    finally:
        set_trunk_linear_contract(False)
        normalization._WARNED_UNDECLARED_FAMILY.clear()
    ref = rms_norm_batch_invariant(x, w, EPS)
    assert torch.equal(out_declared, ref), "trunk-lane declared qk-norm left family-1"
    assert torch.equal(out_undeclared, ref)


@requires_cuda
@pytest.mark.gpu
def test_families_differ_on_seed_shape():
    """The tripwire vitality check: if the two families ever collapse to the
    same bits on the seed shape, every bitwise family gate is vacuous and this
    contract should be re-evaluated."""
    x = _make(SEED_SHAPE, 0)
    w = _make((SEED_SHAPE[-1],), 300)
    with torch.no_grad():
        f1 = bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        f2 = bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
    n_diff = (f1 != f2).sum().item()
    assert n_diff > 0, "families agree bitwise on the seed shape; the family gates are vacuous"
    # 1-ulp seeds are rare: a large diff count means a real kernel change.
    assert n_diff < x.numel() * 1e-3


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["native", "sglang", "sglang_fused"])
@pytest.mark.parametrize("bi", [False, True])
def test_family_declared_module_calls_match_legacy_bitwise(mode, bi):
    """Family declarations replace the legacy force_sglang_residual call-site
    exprs without changing a single bit, in every (mode, batch-invariant) lane."""
    x = _make(HIDDEN_SHAPE, 100)
    r = _make(HIDDEN_SHAPE, 200)
    w = _make((HIDDEN_SHAPE[-1],), 300)

    def make_norm(**kwargs):
        norm = RMSNorm(HIDDEN_SHAPE[-1], eps=EPS, mode=mode, **kwargs).to("cuda")
        with torch.no_grad():
            norm.weight.copy_(w)
        return norm

    with set_batch_invariant_mode(bi), torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # qk-norm / layer-0 input: declared no-residual == legacy bare call.
        assert torch.equal(make_norm(family=RMS_NORM_FAMILY_NO_RESIDUAL)(x), make_norm()(x))
        # input layernorm layer>0 / final norm: declared residual tree ==
        # legacy force_sglang_residual=(mode in sglang modes).
        legacy_force = mode in ("sglang", "sglang_fused")
        assert torch.equal(
            make_norm(family=RMS_NORM_FAMILY_RESIDUAL_TREE)(x),
            make_norm()(x, force_sglang_residual=legacy_force),
        )
        # post-attention: declared residual tree == legacy residual call.
        out_new, rout_new = make_norm(family=RMS_NORM_FAMILY_RESIDUAL_TREE)(x, residual=r, prenorm=True)
        out_old, rout_old = make_norm()(x, residual=r, prenorm=True)
        assert torch.equal(out_new, out_old)
        assert torch.equal(rout_new, rout_old)
