"""Tests for traceable_chunked_cross_entropy (the no-inner-compile chunked CE used by the
whole-step torch.compile path).

Validates that the static-unroll + per-chunk-checkpoint chunked CE matches a vanilla
full-logits F.cross_entropy in BOTH value and gradient, across chunk counts (including a
non-divisible N) and with ignored positions — the gradient check is what proves the
activation-checkpoint recompute is wired correctly.
"""

import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss.compiled_cross_entropy import traceable_chunked_cross_entropy

IGNORE_INDEX = -100


def _make_inputs(n=10, hidden=16, vocab=32, n_ignore=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(n, hidden, generator=g, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(vocab, hidden, generator=g, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, vocab, (n,), generator=g)
    if n_ignore:
        labels[torch.randperm(n, generator=g)[:n_ignore]] = IGNORE_INDEX
    return hidden_states, weight, labels


def _reference(hidden_states, weight, labels, reduction):
    logits = (hidden_states @ weight.t()).float()
    return torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=IGNORE_INDEX, reduction=reduction
    )


@pytest.mark.parametrize("num_chunks", [1, 3, 8, 10, 16])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_forward_matches_full_logits(num_chunks, reduction):
    # n=10 is intentionally NOT divisible by 3/8/16 → exercises the ragged last chunk.
    hidden_states, weight, labels = _make_inputs(n=10)
    got = traceable_chunked_cross_entropy(
        hidden_states, weight, labels, ignore_index=IGNORE_INDEX,
        num_chunks=num_chunks, reduction=reduction,
    )
    expected = _reference(hidden_states, weight, labels, reduction)
    assert_close(got, expected)


@pytest.mark.parametrize("num_chunks", [1, 3, 8])
def test_backward_matches_full_logits(num_chunks):
    """Grads w.r.t. hidden_states AND weight must match the unchunked autograd reference —
    i.e. the per-chunk checkpoint recompute reconstructs the same backward."""
    h1, w1, labels = _make_inputs(n=10)
    h2, w2 = h1.detach().clone().requires_grad_(True), w1.detach().clone().requires_grad_(True)

    traceable_chunked_cross_entropy(
        h1, w1, labels, ignore_index=IGNORE_INDEX, num_chunks=num_chunks, reduction="sum"
    ).backward()
    _reference(h2, w2, labels, "sum").backward()

    assert_close(h1.grad, h2.grad)
    assert_close(w1.grad, w2.grad)


def test_all_ignored_is_finite():
    """All labels ignored → zero loss, finite grads (no NaN from a 0/0 mean)."""
    hidden_states, weight, labels = _make_inputs(n=8, n_ignore=0)
    labels[:] = IGNORE_INDEX
    loss = traceable_chunked_cross_entropy(
        hidden_states, weight, labels, ignore_index=IGNORE_INDEX, num_chunks=4, reduction="mean"
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(hidden_states.grad).all()


def test_inference_path_no_checkpoint():
    """Under no_grad (inference) the value still matches; the checkpoint branch is skipped."""
    hidden_states, weight, labels = _make_inputs(n=10)
    with torch.no_grad():
        got = traceable_chunked_cross_entropy(
            hidden_states, weight, labels, ignore_index=IGNORE_INDEX, num_chunks=3, reduction="sum"
        )
        expected = _reference(hidden_states, weight, labels, "sum")
    assert_close(got, expected)
