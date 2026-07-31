"""Contract gates for the families-v2 norm trees (hidden-dim RMSNorm + qk-norm).

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
- the strided qk-norm entry normalizes packed qkv views in place without
  touching the k/v bytes beside it;

Correctness against an fp64 reference is a wrongness check, not a bit gate: v2
defines its own bits, and the reference cannot arbitrate between two trees that
both round correctly.
"""

import pytest
import torch

from xorl.ops.bi_families_v2 import (
    V2_NORM_SPLIT_MIN_TILES,
    V2_NORM_TILE,
    qk_norm_v2,
    rms_norm_v2,
)


EPS = 1e-6
H = 3840
HEAD_DIM = 128

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


@requires_cuda
def test_norm_v2_within_one_ulp_of_fp64_reference():
    x, residual, weight = _payload((32, H), 1), _payload((32, H), 2), _payload((H,), 3)
    out, residual_out = rms_norm_v2(x, weight, EPS, residual=residual)
    assert _max_ulp(out, _reference(x, weight, residual=residual)) <= 1
    assert torch.equal(residual_out, (x.float() + residual.float()).to(torch.bfloat16))
    assert _max_ulp(rms_norm_v2(x, weight, EPS), _reference(x, weight)) <= 1
    zero_centered = rms_norm_v2(x, weight, EPS, zero_centered=True)
    assert _max_ulp(zero_centered, _reference(x, weight, zero_centered=True)) <= 1


@requires_cuda
def test_norm_v2_is_batch_composition_invariant():
    x, residual, weight = _payload((128, H), 4), _payload((128, H), 5), _payload((H,), 6)
    full_out, full_residual = rms_norm_v2(x, weight, EPS, residual=residual)
    for rows in (1, 2, 17, 64):
        out, residual_out = rms_norm_v2(x[:rows].contiguous(), weight, EPS, residual=residual[:rows].contiguous())
        assert torch.equal(out, full_out[:rows])
        assert torch.equal(residual_out, full_residual[:rows])


@requires_cuda
def test_norm_v2_is_run_to_run_bitwise():
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
@pytest.mark.parametrize("hidden_size", [H, 4096, 12288])
@pytest.mark.parametrize("rows", [1, 17, 64, 512])
def test_fused_and_split_realizations_are_bitwise_identical(monkeypatch, hidden_size, rows):
    x = _payload((rows, hidden_size), 600 + rows)
    residual = _payload((rows, hidden_size), 700 + rows)
    weight = _payload((hidden_size,), 800 + hidden_size)

    _force(monkeypatch, "fused")
    fused_out, fused_residual = rms_norm_v2(x, weight, EPS, residual=residual)
    fused_plain = rms_norm_v2(x, weight, EPS)
    fused_zero_centered = rms_norm_v2(x, weight, EPS, zero_centered=True)

    _force(monkeypatch, "split")
    split_out, split_residual = rms_norm_v2(x, weight, EPS, residual=residual)
    split_plain = rms_norm_v2(x, weight, EPS)
    split_zero_centered = rms_norm_v2(x, weight, EPS, zero_centered=True)

    assert torch.equal(fused_out, split_out)
    assert torch.equal(fused_residual, split_residual)
    assert torch.equal(fused_plain, split_plain)
    assert torch.equal(fused_zero_centered, split_zero_centered)


@requires_cuda
def test_split_realization_is_actually_exercised(monkeypatch):
    """Guard the guard: prove forcing 'split' reaches the split kernels.

    Without this, a refactor that dropped the split realization entirely would
    leave the equality tests above passing vacuously.
    """
    import xorl.ops.bi_families_v2 as module

    calls = []
    original = module._rms_norm_v2_split

    def counting_split(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_rms_norm_v2_split", counting_split)
    _force(monkeypatch, "split")
    rms_norm_v2(_payload((512, H), 11), _payload((H,), 12), EPS)
    assert calls, "forcing the split realization did not reach _rms_norm_v2_split"


# --- the dispatch rule -------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("hidden_size", [2048, 3840, 4096])
@pytest.mark.parametrize("rows", [1, 64, 256, 512, 2048])
def test_shipped_hidden_sizes_always_take_the_fused_realization(hidden_size, rows):
    """Common shipped hidden sizes stay below the measured split boundary."""
    import xorl.ops.bi_families_v2 as module

    n_tiles = -(-hidden_size // V2_NORM_TILE)
    assert n_tiles < V2_NORM_SPLIT_MIN_TILES
    assert module._v2_norm_use_split(rows, n_tiles) is False


@requires_cuda
def test_dispatch_rule_needs_a_deep_tile_chain_and_few_rows():
    """Split only when the split-kernel tile chain is deep and rows are few."""
    import xorl.ops.bi_families_v2 as module

    shallow = V2_NORM_SPLIT_MIN_TILES - 1
    assert module._v2_norm_use_split(1, shallow) is False

    deep = V2_NORM_SPLIT_MIN_TILES
    assert module._v2_norm_use_split(deep, deep) is True
    assert module._v2_norm_use_split(deep + 1, deep) is False


@requires_cuda
def test_dispatch_uses_the_split_kernels_tile_basis():
    """The rejected rule was passed the fused kernel's 4096-wide chunk count,
    understating split parallelism by exactly 8x."""
    import xorl.ops.bi_families_v2 as module

    hidden_size = 5120
    split_tiles = -(-hidden_size // V2_NORM_TILE)
    fused_chunks = -(-hidden_size // 4096)
    assert split_tiles == 10
    assert fused_chunks == 2
    assert module._v2_norm_use_split(1, split_tiles) is True
    assert module._v2_norm_use_split(1, fused_chunks) is False


# --- qk-norm -----------------------------------------------------------------


@requires_cuda
def test_qk_norm_v2_strided_matches_contiguous_and_leaves_kv_untouched():
    tokens, n_q, n_kv = 256, 8, 2
    packed = _payload((tokens, (n_q + 2 * n_kv) * HEAD_DIM), 13)
    weight = _payload((HEAD_DIM,), 14)
    q_view = packed[:, : n_q * HEAD_DIM]

    out = qk_norm_v2(q_view, weight, EPS, head_dim=HEAD_DIM)
    assert torch.equal(out, qk_norm_v2(q_view.contiguous(), weight, EPS, head_dim=HEAD_DIM))
    reference = _reference(q_view.contiguous().reshape(-1, HEAD_DIM), weight)
    assert _max_ulp(out, reference.reshape(tokens, n_q * HEAD_DIM)) <= 1

    scratch = packed.clone()
    scratch_view = scratch[:, : n_q * HEAD_DIM]
    qk_norm_v2(scratch_view, weight, EPS, head_dim=HEAD_DIM, out=scratch_view)
    assert torch.equal(scratch_view, out), "in-place qk-norm diverged from out-of-place"
    assert torch.equal(scratch[:, n_q * HEAD_DIM :], packed[:, n_q * HEAD_DIM :]), "k/v bytes moved"
