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

from xorl.models.layers.normalization import (  # noqa: E402
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNorm,
    fast_zero_centered_batch_invariant_rms_norm,
)
from xorl.ops.sglang.batch_invariant_ops import (  # noqa: E402
    bi_rms_norm,
    set_batch_invariant_mode,
)
from xorl.ops.sglang.bi_families_v2 import rms_norm_v2 as xorl_rms_norm_v2  # noqa: E402


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
    """qk-norm / layer-0 input layernorm: xorl's parity-lane dispatch (native ->
    aten::rms_norm interpose) must bit-match SGLang's residual-is-None dispatch
    (family-1 ``rms_norm_batch_invariant``)."""
    for shape in SHAPES:
        x = _make(shape, 0)
        w = _make((shape[-1],), 300)
        norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_NO_RESIDUAL, w)
        with set_batch_invariant_mode(True), torch.no_grad():
            xorl_out = norm(x)
        serving_out = sgl_bio.rms_norm_batch_invariant(x, w, EPS)
        assert torch.equal(xorl_out, serving_out), f"qk-norm {shape} diverged from serving family-1"

        serving_funnel = sgl_bio.bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        assert torch.equal(xorl_out, serving_funnel), f"qk-norm funnel {shape} diverged"

        if shape == SHAPES[0]:
            family2 = sgl_bio.rms_norm_residual_tree_batch_invariant(x, w, EPS)
            assert not torch.equal(serving_out, family2), "serving families agree on the seed shape; gate is vacuous"

    _assert_presummed_residual_tree_site_fp32_single_rounditwise()
    _assert_post_attention_residual_site_fp32_single_rounditwise()
    _assert_zero_centered_family1_twin_bitwise()
    _assert_zero_centered_families_v2_candidate_bitwise()


def _assert_presummed_residual_tree_site_fp32_single_rounditwise():
    """Input layernorm at layer>0 / final norm: xorl normalizes the pre-summed
    single tensor through the residual tree; SGLang fuses the add. On the same
    summed value both must produce identical bits (gate via a zero residual and
    via SGLang's single-tensor residual-tree kernel)."""
    for shape in SHAPES:
        x = _make(shape, 0)
        w = _make((shape[-1],), 300)
        norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_RESIDUAL_TREE, w)
        with set_batch_invariant_mode(True), torch.no_grad():
            xorl_out = norm(x)
        serving_single = sgl_bio.rms_norm_residual_tree_batch_invariant(x, w, EPS)
        serving_fused, serving_residual = sgl_bio.fused_add_rms_norm_batch_invariant(x, torch.zeros_like(x), w, EPS)
        serving_funnel = sgl_bio.bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
        assert torch.equal(serving_residual, x)
        assert torch.equal(xorl_out, serving_single), f"pre-summed {shape} diverged from residual tree"
        assert torch.equal(xorl_out, serving_fused), f"pre-summed {shape} diverged from fused-add tree"
        assert torch.equal(xorl_out, serving_funnel), f"pre-summed funnel {shape} diverged"


def _assert_post_attention_residual_site_fp32_single_rounditwise():
    """Post-attention layernorm: xorl's fused residual dispatch must bit-match
    SGLang's fused residual dispatch, on both the normed output and the carried
    residual stream."""
    for shape in SHAPES:
        x = _make(shape, 0)
        r = _make(shape, 1)
        w = _make((shape[-1],), 300)
        norm = _xorl_module(shape[-1], RMS_NORM_FAMILY_RESIDUAL_TREE, w)
        with set_batch_invariant_mode(True), torch.no_grad():
            xorl_out, xorl_residual = norm(x, residual=r, prenorm=True)
        serving_out, serving_residual = sgl_bio.fused_add_rms_norm_batch_invariant(x, r, w, EPS)
        funnel_out, funnel_residual = sgl_bio.bi_fused_add_rms_norm(
            x,
            r,
            w,
            EPS,
            family=RMS_NORM_FAMILY_RESIDUAL_TREE,
        )
        assert torch.equal(xorl_residual, serving_residual), f"residual carry {shape} diverged from serving"
        assert torch.equal(xorl_out, serving_out), f"post-attention {shape} diverged from serving"
        assert torch.equal(xorl_residual, funnel_residual), f"residual funnel carry {shape} diverged"
        assert torch.equal(xorl_out, funnel_out), f"post-attention funnel {shape} diverged"


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
