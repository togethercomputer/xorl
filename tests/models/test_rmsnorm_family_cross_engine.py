"""Cross-engine RMSNorm family gates: xorl dispatch vs SGLang dispatch, bitwise.

For each serving site-class, the kernel xorl dispatches must be BITWISE equal
to the kernel SGLang dispatches, on adversarial shapes including the
[4096, 128] k3(1-ulp)-seed shape. These gates are the permanent tripwire
against silent family flips in either engine, in both directions.

Requires SGLang's ``batch_invariant_ops`` importable (pure Triton, no
sgl_kernel); point ``SGLANG_REPO`` at an SGLang checkout's ``python``
directory if it is not installed. Skips otherwise.
"""

import os
import sys

import pytest
import torch


_sglang_repo = os.environ.get("SGLANG_REPO")
if _sglang_repo and _sglang_repo not in sys.path:
    sys.path.insert(0, _sglang_repo)

sgl_bio = pytest.importorskip(
    "sglang.srt.batch_invariant_ops",
    reason="SGLang batch_invariant_ops not importable (set SGLANG_REPO=/path/to/sglang/python)",
)

from xorl.models.layers.normalization import (  # noqa: E402
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNorm,
    fast_batch_invariant_rms_norm,
    fast_zero_centered_batch_invariant_rms_norm,
)
from xorl.ops.batch_invariant_ops import (  # noqa: E402
    bi_fused_add_rms_norm,
    bi_rms_norm,
    set_batch_invariant_mode,
    set_trunk_linear_contract,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
pytestmark = [requires_cuda, pytest.mark.gpu]

EPS = 1e-6
# Adversarial shapes: the qk-norm k3(1-ulp)-seed shape and larger hidden-size
# shapes (both engines' families disagree on a handful of elements at each).
SHAPES = [(4096, 128), (32768, 128), (1024, 2048), (8192, 4096)]


def _make(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)


def _xorl_module(hidden, family, weight):
    norm = RMSNorm(hidden, eps=EPS, mode="sglang_fused", family=family).to("cuda")
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


@pytest.mark.parametrize("shape", SHAPES)
def test_qk_norm_site_class_bitwise(shape):
    """qk-norm / layer-0 input layernorm: xorl's parity-lane dispatch (native ->
    aten::rms_norm interpose) must bit-match SGLang's residual-is-None dispatch
    (family-1 ``rms_norm_batch_invariant``)."""
    x = _make(shape, 0)
    w = _make((shape[-1],), 300)
    norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_NO_RESIDUAL, w)
    with set_batch_invariant_mode(True), torch.no_grad():
        xorl_out = norm(x)
    serving_out = sgl_bio.rms_norm_batch_invariant(x, w, EPS)
    assert torch.equal(xorl_out, serving_out), "qk-norm site-class diverged from serving family-1"


@pytest.mark.parametrize("shape", SHAPES)
def test_presummed_residual_tree_site_class_bitwise(shape):
    """Input layernorm at layer>0 / final norm: xorl normalizes the pre-summed
    single tensor through the residual tree; SGLang fuses the add. On the same
    summed value both must produce identical bits (gate via a zero residual and
    via SGLang's single-tensor residual-tree kernel)."""
    x = _make(shape, 0)
    w = _make((shape[-1],), 300)
    norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_RESIDUAL_TREE, w)
    with set_batch_invariant_mode(True), torch.no_grad():
        xorl_out = norm(x)
    serving_single = sgl_bio.rms_norm_residual_tree_batch_invariant(x, w, EPS)
    serving_fused, serving_residual = sgl_bio.fused_add_rms_norm_batch_invariant(x, torch.zeros_like(x), w, EPS)
    assert torch.equal(serving_residual, x)
    assert torch.equal(xorl_out, serving_single), "pre-summed site-class diverged from serving residual tree"
    assert torch.equal(xorl_out, serving_fused), "pre-summed site-class diverged from serving fused-add tree"


@pytest.mark.parametrize("shape", SHAPES)
def test_post_attention_residual_site_class_bitwise(shape):
    """Post-attention layernorm: xorl's fused residual dispatch must bit-match
    SGLang's fused residual dispatch, on both the normed output and the carried
    residual stream."""
    x = _make(shape, 0)
    r = _make(shape, 1)
    w = _make((shape[-1],), 300)
    norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_RESIDUAL_TREE, w)
    with set_batch_invariant_mode(True), torch.no_grad():
        xorl_out, xorl_residual = norm(x, residual=r, prenorm=True)
    serving_out, serving_residual = sgl_bio.fused_add_rms_norm_batch_invariant(x, r, w, EPS)
    assert torch.equal(xorl_residual, serving_residual), "residual carry diverged from serving"
    assert torch.equal(xorl_out, serving_out), "post-attention site-class diverged from serving"


@pytest.mark.parametrize("family", [RMS_NORM_FAMILY_NO_RESIDUAL, RMS_NORM_FAMILY_RESIDUAL_TREE])
@pytest.mark.parametrize("shape", SHAPES)
def test_family_funnels_agree_cross_engine(family, shape):
    """The vendored family funnels themselves must agree bitwise per family."""
    x = _make(shape, 7)
    w = _make((shape[-1],), 301)
    with torch.no_grad():
        assert torch.equal(
            bi_rms_norm(x, w, EPS, family=family),
            sgl_bio.bi_rms_norm(x, w, EPS, family=family),
        )


def test_fused_add_funnels_agree_cross_engine():
    x = _make((4096, 128), 7)
    r = _make((4096, 128), 8)
    w = _make((128,), 301)
    with torch.no_grad():
        x_out, x_rout = bi_fused_add_rms_norm(x, r, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
        s_out, s_rout = sgl_bio.bi_fused_add_rms_norm(x, r, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
    assert torch.equal(x_out, s_out)
    assert torch.equal(x_rout, s_rout)


def test_families_differ_cross_engine_gate_has_teeth():
    """If the two serving families ever agree bitwise on the seed shape, the
    site-class gates above cannot catch a family flip; fail loudly instead."""
    x = _make((4096, 128), 0)
    w = _make((128,), 300)
    f1 = sgl_bio.rms_norm_batch_invariant(x, w, EPS)
    f2 = sgl_bio.rms_norm_residual_tree_batch_invariant(x, w, EPS)
    assert not torch.equal(f1, f2), "serving families agree on the seed shape; gates are vacuous"


@pytest.mark.parametrize("shape", SHAPES)
def test_trunk_contract_lane_qk_norm_site_class_bitwise(shape):
    """The scoped trunk-contract lane (XORL_BI_TRUNK_LINEAR, no global interpose)
    must dispatch the same family-1 kernel as serving's residual-is-None path --
    the inverse case (a qk-norm family flip) is what this
    gate would catch."""
    x = _make(shape, 0)
    w = _make((shape[-1],), 300)
    norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_NO_RESIDUAL, w)
    set_trunk_linear_contract(True)
    try:
        with torch.no_grad():
            xorl_out = norm(x)
    finally:
        set_trunk_linear_contract(False)
    serving_out = sgl_bio.rms_norm_batch_invariant(x, w, EPS)
    assert torch.equal(xorl_out, serving_out), "trunk-lane qk-norm diverged from serving family-1"
    # The differentiable family-1 wrapper is the same dispatch.
    with torch.no_grad():
        assert torch.equal(fast_batch_invariant_rms_norm(x, w, EPS), serving_out)


@pytest.mark.parametrize("shape", SHAPES)
def test_zero_centered_family1_twin_bitwise(shape):
    """The Qwen3.5 zero-centered (Gemma-style) twin is family-1 with a fp32
    ``1 + weight`` fold; both engines' funnels and xorl's differentiable wrapper
    must agree bitwise."""
    x = _make(shape, 5)
    w = _make((shape[-1],), 302)
    with torch.no_grad():
        xorl_fn = fast_zero_centered_batch_invariant_rms_norm(x, w, EPS)
        xorl_funnel = bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True)
        serving_funnel = sgl_bio.bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True)
    assert torch.equal(xorl_fn, xorl_funnel), "zero-centered wrapper diverged from the xorl funnel"
    assert torch.equal(xorl_fn, serving_funnel), "zero-centered twin diverged cross-engine"
