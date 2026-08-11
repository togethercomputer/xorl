"""Contract gates for the families-v2 hidden-dimension RMSNorm trees.

The v2 trees are the frozen production contract for the batch-invariance lane.
These gates pin the properties the design rests on:

- the tree is a function of the hidden size alone, so a row normalizes to the
  same bits whatever batch it arrives in;
- the two *realizations* of that tree (one fused launch, or split into
  partials / inverse-rms / normalize) are bitwise identical, which is what
  makes the fused-vs-split dispatch a pure performance choice. Each
  realization is forced explicitly here rather than reached through the
  dispatch rule, so the gate keeps its teeth when that rule changes;
- the dispatch rule itself is exercised directly, including the shipped hidden
  sizes where it must select the fused realization;

Correctness against an fp64 reference is a wrongness check, not a bit gate: v2
defines its own bits, and the reference cannot arbitrate between two trees that
both round correctly.
"""

import pytest
import torch

import xorl.models.layers.normalization as normalization
from xorl.ops.bi_families_v2 import (
    V2_NORM_SPLIT_MIN_TILES,
    V2_NORM_TILE,
    families_v2_enabled,
    rms_norm_v2,
)


EPS = 1e-6
H = 3840

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
pytestmark = pytest.mark.gpu


def _payload(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16).cuda()


def _max_ulp(a, b):
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    return (ai - bi).abs().max().item()


def _reference(x, weight, eps=EPS, residual=None, zero_centered=False):
    xf = x.double()
    if residual is not None:
        xf = (x.float() + residual.float()).to(torch.bfloat16).double()
    variance = xf.pow(2).mean(-1, keepdim=True)
    wf = weight.double() + 1.0 if zero_centered else weight.double()
    return (xf * torch.rsqrt(variance + eps) * wf).to(torch.bfloat16)


def _force(monkeypatch, realization):
    """Pin the realization so the bit gates never depend on the dispatch rule."""
    import xorl.ops.bi_families_v2 as module

    monkeypatch.setattr(module, "_v2_norm_use_split", lambda *_a, **_k: realization == "split")


# --- the tree: correctness and batch invariance ------------------------------


def _assert_norm_v2_within_one_ulp_of_fp64_reference():
    x, residual, weight = _payload((32, H), 1), _payload((32, H), 2), _payload((H,), 3)
    out, residual_out = rms_norm_v2(x, weight, EPS, residual=residual)
    assert _max_ulp(out, _reference(x, weight, residual=residual)) <= 1
    assert torch.equal(residual_out, (x.float() + residual.float()).to(torch.bfloat16))
    assert _max_ulp(rms_norm_v2(x, weight, EPS), _reference(x, weight)) <= 1
    zero_centered = rms_norm_v2(x, weight, EPS, zero_centered=True)
    assert _max_ulp(zero_centered, _reference(x, weight, zero_centered=True)) <= 1
    zero_centered_residual, zero_centered_residual_out = rms_norm_v2(
        x,
        weight,
        EPS,
        residual=residual,
        zero_centered=True,
    )
    assert torch.equal(zero_centered_residual_out, residual_out)
    assert (
        _max_ulp(
            zero_centered_residual,
            _reference(x, weight, residual=residual, zero_centered=True),
        )
        <= 1
    )

    _assert_norm_v2_is_batch_composition_invariant()
    _assert_norm_v2_is_run_to_run_bitwise()


def _assert_norm_v2_is_batch_composition_invariant():
    x, residual, weight = _payload((128, H), 4), _payload((128, H), 5), _payload((H,), 6)
    full_out, full_residual = rms_norm_v2(x, weight, EPS, residual=residual)
    for rows in (1, 2, 17, 64):
        out, residual_out = rms_norm_v2(x[:rows].contiguous(), weight, EPS, residual=residual[:rows].contiguous())
        assert torch.equal(out, full_out[:rows])
        assert torch.equal(residual_out, full_residual[:rows])


def _assert_norm_v2_is_run_to_run_bitwise():
    x, residual, weight = _payload((64, H), 7), _payload((64, H), 8), _payload((H,), 9)
    first = rms_norm_v2(x, weight, EPS, residual=residual)
    for _ in range(3):
        again = rms_norm_v2(x, weight, EPS, residual=residual)
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


# --- the two realizations of that tree are bitwise identical -----------------
#
# Forced explicitly: the dispatch rule is a performance heuristic and may select
# either realization for any shape, so a gate that let the rule choose would
# silently degrade into comparing one realization against itself.


@requires_cuda
def test_norm_v2_numerical_realization_and_dispatch_policy(monkeypatch):
    _assert_norm_v2_within_one_ulp_of_fp64_reference()

    import xorl.ops.bi_families_v2 as module

    split_calls = []
    original_split = module._rms_norm_v2_split

    def counting_split(*args, **kwargs):
        split_calls.append(1)
        return original_split(*args, **kwargs)

    monkeypatch.setattr(module, "_rms_norm_v2_split", counting_split)

    # Tail, aligned, and deep split-tile shapes at the smallest and largest
    # useful row counts cover the kernel geometry; intermediate row literals
    # do not select different code while each realization is forced.
    for hidden_size in (H, 4096, 12288):
        for rows in (1, 512):
            x = _payload((rows, hidden_size), 600 + rows)
            residual = _payload((rows, hidden_size), 700 + rows)
            weight = _payload((hidden_size,), 800 + hidden_size)

            _force(monkeypatch, "fused")
            fused_out, fused_residual = rms_norm_v2(x, weight, EPS, residual=residual)
            fused_plain = rms_norm_v2(x, weight, EPS)
            fused_zero_centered = rms_norm_v2(x, weight, EPS, zero_centered=True)
            fused_zero_centered_residual = rms_norm_v2(
                x,
                weight,
                EPS,
                residual=residual,
                zero_centered=True,
            )

            _force(monkeypatch, "split")
            split_out, split_residual = rms_norm_v2(x, weight, EPS, residual=residual)
            split_plain = rms_norm_v2(x, weight, EPS)
            split_zero_centered = rms_norm_v2(x, weight, EPS, zero_centered=True)
            split_zero_centered_residual = rms_norm_v2(
                x,
                weight,
                EPS,
                residual=residual,
                zero_centered=True,
            )

            context = f"hidden_size={hidden_size}, rows={rows}"
            assert torch.equal(fused_out, split_out), context
            assert torch.equal(fused_residual, split_residual), context
            assert torch.equal(fused_plain, split_plain), context
            assert torch.equal(fused_zero_centered, split_zero_centered), context
            assert torch.equal(fused_zero_centered_residual[0], split_zero_centered_residual[0]), context
            assert torch.equal(fused_zero_centered_residual[1], split_zero_centered_residual[1]), context
    assert split_calls, "forcing the split realization did not reach _rms_norm_v2_split"

    monkeypatch.undo()
    with monkeypatch.context() as dispatch_patch:
        _assert_norm_v2_dispatch_policy(dispatch_patch)
    with monkeypatch.context() as reachability_patch:
        _assert_norm_v2_reaches_trainer_dispatch(reachability_patch)


# --- the dispatch rule -------------------------------------------------------


def _assert_norm_v2_dispatch_policy(monkeypatch):
    import xorl.ops.bi_families_v2 as module

    split_calls = []
    original_split = module._rms_norm_v2_split

    def counting_split(*args, **kwargs):
        split_calls.append(args[0].shape)
        return original_split(*args, **kwargs)

    monkeypatch.setattr(module, "_rms_norm_v2_split", counting_split)

    # Common shipped hidden sizes stay below the measured split boundary.
    for hidden_size in (2048, 3840, 4096):
        n_tiles = -(-hidden_size // V2_NORM_TILE)
        assert n_tiles < V2_NORM_SPLIT_MIN_TILES
        # Once the tile count is below the threshold, row count cannot change
        # the decision; retain only both row-count extremes.
        for rows in (1, 2048):
            assert module._v2_norm_use_split(rows, n_tiles) is False
            x = _payload((rows, hidden_size), rows * 7 + hidden_size)
            weight = _payload((hidden_size,), rows * 11 + hidden_size)
            module.rms_norm_v2(x, weight, EPS, residual=torch.zeros_like(x))
    assert split_calls == [], f"split realization ran at shipped hidden sizes: {split_calls}"

    # Split only when the split-kernel tile chain is deep and rows are few.
    shallow = V2_NORM_SPLIT_MIN_TILES - 1
    assert module._v2_norm_use_split(1, shallow) is False

    deep = V2_NORM_SPLIT_MIN_TILES
    assert module._v2_norm_use_split(deep, deep) is True
    assert module._v2_norm_use_split(deep + 1, deep) is False

    # The rejected rule used the fused kernel's 4096-wide chunk count,
    # understating split parallelism by exactly 8x.
    hidden_size = 5120
    split_tiles = -(-hidden_size // V2_NORM_TILE)
    fused_chunks = -(-hidden_size // 4096)
    assert split_tiles == 10
    assert fused_chunks == 2
    assert module._v2_norm_use_split(1, split_tiles) is True
    assert module._v2_norm_use_split(1, fused_chunks) is False

    # Prove the production dispatcher, not just its decision helper, reaches
    # the split realization at a deep tile shape.
    deep_hidden = 16384
    deep_x = _payload((8, deep_hidden), 8 * 7 + deep_hidden)
    deep_weight = _payload((deep_hidden,), 8 * 11 + deep_hidden)
    module.rms_norm_v2(deep_x, deep_weight, EPS, residual=torch.zeros_like(deep_x))
    assert split_calls == [(8, deep_hidden)]


def _assert_norm_v2_reaches_trainer_dispatch(monkeypatch):
    assert families_v2_enabled() is True

    x, residual, weight = _payload((64, H), 15), _payload((64, H), 16), _payload((H,), 17)
    expected, expected_residual = rms_norm_v2(x, weight, EPS, residual=residual)
    expected_plain = rms_norm_v2(x, weight, EPS)

    assert torch.equal(normalization.fast_sglang_rms_norm(x, weight, EPS), expected_plain)
    assert torch.equal(normalization.fast_batch_invariant_rms_norm(x, weight, EPS), expected_plain)
    fused_output, fused_residual = normalization.fast_sglang_residual_rms_norm(x, residual, weight, EPS)
    assert torch.equal(fused_output, expected)
    assert torch.equal(fused_residual, expected_residual)
