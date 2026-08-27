"""Cross-engine RMSNorm family gates: xorl dispatch vs SGLang dispatch, bitwise.

For each serving site-class, the kernel xorl dispatches must be BITWISE equal
to the kernel SGLang dispatches, on adversarial shapes including the
[4096, 128] k3(1-ulp)-seed shape. These gates are the permanent tripwire
against silent family flips in either engine, in both directions.

Requires SGLang's ``batch_invariant_ops`` importable (pure Triton, no
sgl_kernel). Install SGLang in the test environment or provide its Python
directory through the interpreter's normal ``PYTHONPATH`` setup. Skips
otherwise.
"""

import pytest
import torch


sgl_bio = pytest.importorskip(
    "sglang.srt.batch_invariant_ops",
    reason="SGLang batch_invariant_ops not importable",
)
sgl_bio_v2 = pytest.importorskip(
    "sglang.srt.batch_invariant_ops.bi_families_v2",
    reason="SGLang bi_families_v2 not importable",
)

from xorl.models.layers.normalization import (  # noqa: E402
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNorm,
    fast_zero_centered_batch_invariant_rms_norm,
)
from xorl.ops.batch_invariant_ops import (  # noqa: E402
    bi_rms_norm,
    set_batch_invariant_mode,
)
from xorl.ops.bi_families_v2 import rms_norm_v2 as xorl_rms_norm_v2  # noqa: E402


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


def test_rmsnorm_site_class_cross_engine_bitwise_policy():
    """qk-norm / layer-0 input layernorm: xorl's declared no-residual site
    (``mode="sglang_fused"``) executes the v2 reduction tree and must bit-match
    serving's v2 (``bi_families_v2.rms_norm_v2``) — the pinned SGLang revision
    selects v2 structurally for exact-lane sites. The ``aten::rms_norm``
    interpose remains the family-1 surface and must bit-match serving's
    family-1 ``rms_norm_batch_invariant``."""
    for shape in SHAPES:
        x = _make(shape, 0)
        w = _make((shape[-1],), 300)
        norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_NO_RESIDUAL, w)
        with set_batch_invariant_mode(True), torch.no_grad():
            xorl_out = norm(x)
        serving_v2 = sgl_bio_v2.rms_norm_v2(x, w, EPS)
        assert torch.equal(xorl_out, serving_v2), f"qk-norm {shape} diverged from serving v2"

        with set_batch_invariant_mode(True), torch.no_grad():
            interpose_out = torch.nn.functional.rms_norm(x, (shape[-1],), w, EPS)
        serving_out = sgl_bio.rms_norm_batch_invariant(x, w, EPS)
        assert torch.equal(interpose_out, serving_out), f"qk-norm interpose {shape} diverged from serving family-1"
        serving_funnel = sgl_bio.bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        assert torch.equal(interpose_out, serving_funnel), f"qk-norm funnel {shape} diverged"

        if shape == SHAPES[0]:
            family2 = sgl_bio.rms_norm_residual_tree_batch_invariant(x, w, EPS)
            assert not torch.equal(serving_out, family2), "serving families agree on the seed shape; gate is vacuous"
            assert not torch.equal(serving_v2, serving_out), "v1/v2 trees agree on the seed shape; gate is vacuous"

    _assert_presummed_residual_tree_site_class_bitwise()
    _assert_post_attention_residual_site_class_bitwise()
    _assert_zero_centered_family1_twin_bitwise()
    _assert_zero_centered_families_v2_candidate_bitwise()


def _assert_presummed_residual_tree_site_class_bitwise():
    """Input layernorm at layer>0 / final norm: xorl normalizes the pre-summed
    single tensor through the v2 reduction (the tree serving executes for
    exact-lane sites); SGLang's family-2 v1 surfaces remain internally
    consistent with each other."""
    for shape in SHAPES:
        x = _make(shape, 0)
        w = _make((shape[-1],), 300)
        norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_RESIDUAL_TREE, w)
        with set_batch_invariant_mode(True), torch.no_grad():
            xorl_out = norm(x)
        serving_v2 = sgl_bio_v2.rms_norm_v2(x, w, EPS)
        assert torch.equal(xorl_out, serving_v2), f"pre-summed {shape} diverged from serving v2"

        serving_single = sgl_bio.rms_norm_residual_tree_batch_invariant(x, w, EPS)
        serving_fused, serving_residual = sgl_bio.fused_add_rms_norm_batch_invariant(x, torch.zeros_like(x), w, EPS)
        serving_funnel = sgl_bio.bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
        assert torch.equal(serving_residual, x)
        assert torch.equal(serving_single, serving_fused), f"v1 single vs fused-add tree {shape} diverged"
        assert torch.equal(serving_single, serving_funnel), f"v1 funnel {shape} diverged"


def _assert_post_attention_residual_site_class_bitwise():
    """Post-attention layernorm: xorl's fused residual dispatch must bit-match
    serving's v2 fused-residual tree, on both the normed output and the carried
    residual stream; SGLang's v1 fused surfaces remain internally consistent."""
    for shape in SHAPES:
        x = _make(shape, 0)
        r = _make(shape, 1)
        w = _make((shape[-1],), 300)
        norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_RESIDUAL_TREE, w)
        with set_batch_invariant_mode(True), torch.no_grad():
            xorl_out, xorl_residual = norm(x, residual=r, prenorm=True)
        v2_out, v2_residual = sgl_bio_v2.rms_norm_v2(x, w, EPS, residual=r)
        assert torch.equal(xorl_residual, v2_residual), f"residual carry {shape} diverged from serving v2"
        assert torch.equal(xorl_out, v2_out), f"post-attention {shape} diverged from serving v2"

        serving_out, serving_residual = sgl_bio.fused_add_rms_norm_batch_invariant(x, r, w, EPS)
        funnel_out, funnel_residual = sgl_bio.bi_fused_add_rms_norm(
            x,
            r,
            w,
            EPS,
            family=RMS_NORM_FAMILY_RESIDUAL_TREE,
        )
        assert torch.equal(xorl_residual, serving_residual), f"residual carry {shape} diverged from v1 serving"
        assert torch.equal(serving_out, funnel_out), f"v1 fused vs funnel {shape} diverged"
        assert torch.equal(serving_residual, funnel_residual), f"v1 residual funnel carry {shape} diverged"


def _assert_zero_centered_family1_twin_bitwise():
    """The Qwen3.5 zero-centered (Gemma-style) twin is family-1 with a fp32
    ``1 + weight`` fold; both engines' funnels and xorl's differentiable wrapper
    must agree bitwise."""
    for shape in SHAPES:
        x = _make(shape, 5)
        w = _make((shape[-1],), 302)
        with torch.no_grad():
            xorl_fn = fast_zero_centered_batch_invariant_rms_norm(x, w, EPS)
            xorl_funnel = bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True)
            serving_funnel = sgl_bio.bi_rms_norm(
                x,
                w,
                EPS,
                family=RMS_NORM_FAMILY_NO_RESIDUAL,
                zero_centered=True,
            )
        assert torch.equal(xorl_fn, xorl_funnel), f"zero-centered wrapper {shape} diverged from xorl"
        assert torch.equal(xorl_fn, serving_funnel), f"zero-centered twin {shape} diverged cross-engine"


def _assert_zero_centered_families_v2_candidate_bitwise():
    """The opt-in Qwen families-v2 candidate is one shared arithmetic tree.

    Cover both the q/k-style no-residual form and the decoder-layer fused
    residual form.  The small shape selects the fused realization; the hidden
    size exercises the production-width epilogue.
    """
    for shape in ((64, 128), (64, 3840)):
        for with_residual in (False, True):
            x = _make(shape, 11)
            w = _make((shape[-1],), 312)
            residual = _make(shape, 12) if with_residual else None
            with torch.no_grad():
                xorl_result = xorl_rms_norm_v2(x, w, EPS, residual=residual, zero_centered=True)
                serving_result = sgl_bio.rms_norm_v2(x, w, EPS, residual=residual, zero_centered=True)

            if with_residual:
                xorl_out, xorl_residual = xorl_result
                serving_out, serving_residual = serving_result
                assert torch.equal(xorl_residual, serving_residual), f"residual {shape} diverged"
                assert torch.equal(xorl_out, serving_out), f"output {shape} diverged"
            else:
                assert torch.equal(xorl_result, serving_result), f"no-residual {shape} diverged"
