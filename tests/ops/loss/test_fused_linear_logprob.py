"""Single-GPU tests for the chunked fused selected-token log-probability path.

Validates that :func:`fused_selected_logprob_ce` (chunked cuBLAS matmul + fused
quack CuTeDSL cross-entropy):
  * matches ``F.cross_entropy(reduction="none")`` in forward (the selected-token
    log-probability ``-log p(y_t)``);
  * matches autograd gradients for ``grad_h``, ``grad_W`` and ``grad_b``;
  * respects ``ctx.needs_input_grad`` — a frozen output layer yields ``grad_W is
    None`` while ``grad_h`` stays correct (the LoRA RL backward fix); and
  * bounds peak activation memory to the chunk size (never the full ``[N, V_local]``).
"""

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.loss.fused_linear_logprob import fused_selected_logprob_ce
from xorl.ops.loss.importance_sampling_loss import importance_sampling_loss_function
from xorl.ops.loss.per_token_ce import compute_per_token_ce


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="fused selected-logprob path requires CUDA"),
]


def _ref_ce(h, w, b, labels, ignore_index, temperature):
    logits = (h @ w.t()).float()
    if b is not None:
        logits = logits + b.float()[None, :]
    logits = logits / temperature
    return F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)


def _make_inputs(N, H, V, dtype, has_bias, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(N, H, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(V, H, device="cuda", dtype=dtype, generator=g) / (H**0.5)
    b = torch.randn(V, device="cuda", dtype=dtype, generator=g) if has_bias else None
    labels = torch.randint(0, V, (N,), device="cuda", generator=g)
    labels[::7] = -100  # exercise ignore_index
    return h, w, b, labels


def test_fused_selected_logprob_matches_eager_forward_and_backward():
    # Dtype, bias, and temperature are independent branches; pairwise cases
    # cover both values without a Cartesian product.
    for dtype, has_bias, temperature in (
        (torch.bfloat16, False, 1.0),
        (torch.float32, True, 0.7),
    ):
        h, w, b, labels = _make_inputs(128, 256, 1024, dtype, has_bias)
        valid = (labels != -100).sum().clamp(min=1).float()

        hf = h.clone().requires_grad_(True)
        wf = w.clone().requires_grad_(True)
        bf = b.clone().requires_grad_(True) if has_bias else None
        ce = fused_selected_logprob_ce(hf, wf, labels, bias=bf, ignore_index=-100, temperature=temperature)
        (ce.sum() / valid).backward()

        hr = h.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        br = b.clone().requires_grad_(True) if has_bias else None
        ref = _ref_ce(hr, wr, br, labels, -100, temperature)
        (ref.sum() / valid).backward()

        context = f"dtype={dtype}, bias={has_bias}, temperature={temperature}"
        tol = 5e-2 if dtype == torch.bfloat16 else 3e-3
        assert torch.isfinite((-ce)[labels != -100]).all()
        assert (ce - ref).abs().max().item() < tol, f"forward mismatch: {context}"
        assert (hf.grad - hr.grad).abs().max().item() < tol, f"grad_h mismatch: {context}"
        assert (wf.grad - wr.grad).abs().max().item() < tol, f"grad_W mismatch: {context}"
        if has_bias:
            assert (bf.grad - br.grad).abs().max().item() < tol, f"grad_b mismatch: {context}"

    _assert_input_gradient_policy_for_frozen_output_layer()
    _assert_irregular_tail_shape_matches_eager()
    _assert_production_vocab_paths_are_finite_and_match_eager()
    _assert_loss_dispatchers_match_eager()


def _assert_input_gradient_policy_for_frozen_output_layer():
    h, w, b, labels = _make_inputs(128, 256, 1024, torch.bfloat16, has_bias=True)
    hf = h.clone().requires_grad_(True)
    wf = w.clone().requires_grad_(False)  # frozen output head
    bf = b.clone().requires_grad_(False)
    ce = fused_selected_logprob_ce(hf, wf, labels, bias=bf, ignore_index=-100)
    ce.sum().backward()

    assert hf.grad is not None
    assert wf.grad is None, "frozen weight must receive no gradient (no wasted work)"
    assert bf.grad is None, "frozen bias must receive no gradient"

    hr = h.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    br = b.clone().requires_grad_(True)
    ref = _ref_ce(hr, wr, br, labels, -100, 1.0)
    ref.sum().backward()
    assert (hf.grad - hr.grad).abs().max().item() < 5e-2, "grad_h must match even with frozen head"

    h, w, _, labels = _make_inputs(64, 128, 512, torch.bfloat16, has_bias=False)
    out = fused_selected_logprob_ce(h, w, labels, ignore_index=-100)
    assert not out.requires_grad


def _assert_irregular_tail_shape_matches_eager():
    h, w, b, labels = _make_inputs(37, 130, 777, torch.bfloat16, has_bias=True)
    hf = h.clone().requires_grad_(True)
    wf = w.clone().requires_grad_(True)
    ce = fused_selected_logprob_ce(hf, wf, labels, bias=b, ignore_index=-100)
    ce.sum().backward()
    ref = _ref_ce(h, w, b, labels, -100, 1.0)
    assert (ce - ref).abs().max().item() < 5e-2


def _assert_loss_dispatchers_match_eager():
    from xorl.ops.loss.causallm_loss import causallm_loss_function  # noqa: PLC0415

    h, w, _, labels = _make_inputs(96, 192, 800, torch.bfloat16, has_bias=False)
    fused = compute_per_token_ce(h, w, labels, ignore_index=-100, ce_mode="fused_quack")
    eager = compute_per_token_ce(h, w, labels, ignore_index=-100, ce_mode="eager")
    assert (fused - eager).abs().max().item() < 5e-2

    h3, lab3 = h.view(1, 96, 192), labels.view(1, 96)
    fused = causallm_loss_function(h3, w, lab3, ignore_index=-100, ce_mode="fused_quack")
    eager = causallm_loss_function(h3, w, lab3, ignore_index=-100, ce_mode="eager")
    assert torch.isfinite(fused.loss).all()
    assert (fused.loss - eager.loss).abs().item() < 5e-2

    quack = causallm_loss_function(h3, w, lab3, ignore_index=-100, ce_mode="quack_linear", return_per_token=True)
    eager = causallm_loss_function(h3, w, lab3, ignore_index=-100, ce_mode="eager", return_per_token=True)

    assert torch.isfinite(quack.loss).all()
    assert (quack.loss - eager.loss).abs().item() < 5e-2
    assert (quack.per_token_loss - eager.per_token_loss).abs().max().item() < 5e-2
    assert (quack.per_token_logprobs - eager.per_token_logprobs).abs().max().item() < 5e-2

    _assert_importance_sampling_loss_with_fused_mode()


def _assert_production_vocab_paths_are_finite_and_match_eager():
    """Regression: the quack CE fwd kernel launched cluster blocks whose column
    tile was entirely out of range at these vocab sizes ((cluster_n-1)*tile_n >=
    V), reducing max over zero elements (-inf) and NaN-ing the LSE for every
    row. 151936/201088 are the Qwen3 / GPT-OSS vocabs; tests previously only
    covered V <= 65536-class shapes where every cluster block owns columns."""
    # The Qwen integration case and the largest direct GPT-OSS case cover the
    # two production boundaries that the small-vocabulary dispatcher case misses.
    from xorl.ops.loss.causallm_loss import causallm_loss_function  # noqa: PLC0415

    N, H, V = 512, 1024, 151936
    h, w, _, labels = _make_inputs(N, H, V, torch.bfloat16, has_bias=False)
    h3, lab3 = h.view(1, N, H), labels.view(1, N)
    quack = causallm_loss_function(h3, w, lab3, ignore_index=-100, ce_mode="quack_linear", return_per_token=True)
    eager = causallm_loss_function(h3, w, lab3, ignore_index=-100, ce_mode="eager", return_per_token=True)
    assert torch.isfinite(quack.loss).all()
    assert (quack.loss - eager.loss).abs().item() < 5e-2
    assert (quack.per_token_loss - eager.per_token_loss).abs().max().item() < 5e-2

    del h3, lab3, quack, eager, h, w, labels

    V = 201088
    h, w, b, labels = _make_inputs(N, H, V, torch.bfloat16, has_bias=False)
    hf = h.clone().requires_grad_(True)
    wf = w.clone().requires_grad_(True)
    ce = fused_selected_logprob_ce(hf, wf, labels, bias=b, ignore_index=-100, chunk_size=256)
    assert torch.isfinite(ce[labels != -100]).all()
    ref = _ref_ce(h, w, b, labels, -100, 1.0)
    assert (ce - ref).abs().max().item() < 5e-2
    ce.sum().backward()
    assert torch.isfinite(hf.grad).all() and torch.isfinite(wf.grad).all()


def test_causallm_fused_quack_does_not_materialize_full_logits():
    """Regression for the OOM bug: ``causallm_loss_function`` had no fused_quack
    branch, so fused_quack fell through to the eager ``hidden @ weight.t()`` path
    and materialized the full ``[N, V]`` logits (60.6 GB at 65k×248k → OOM). With
    the fix it goes through the chunked path, so peak activation stays far below
    the full logits tile. This is the assertion that actually distinguishes the
    fixed code from the broken fall-through (a loss-match test passes either way,
    since the old fall-through == eager)."""
    from xorl.ops.loss.causallm_loss import causallm_loss_function  # noqa: PLC0415

    N, H, V = 16384, 2048, 50000
    h = torch.randn(1, N, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(V, H, device="cuda", dtype=torch.bfloat16) / (H**0.5)).requires_grad_()
    labels = torch.randint(0, V, (1, N), device="cuda")
    full_tile_mb = N * V * 4 / 1024 / 1024  # full [N, V] fp32 logits (the eager path)

    causallm_loss_function(h, w, labels, ignore_index=-100, ce_mode="fused_quack").loss.backward()
    h.grad = w.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    causallm_loss_function(h, w, labels, ignore_index=-100, ce_mode="fused_quack").loss.backward()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 1024 / 1024

    assert peak_mb < full_tile_mb / 2, (
        f"fused_quack via causallm_loss_function used {peak_mb:.0f} MB; the full "
        f"[N, V] logits tile is {full_tile_mb:.0f} MB — chunking is not engaged "
        f"(fell through to the eager full-logits path)."
    )


def _assert_importance_sampling_loss_with_fused_mode():
    """End-to-end RL loss with ce_mode="fused_quack": gradient flows through the
    surrogate (frozen output head — the LoRA RL case)."""
    N, H, V = 64, 128, 1000
    h, w, _, labels = _make_inputs(N, H, V, torch.bfloat16, has_bias=False)
    labels[:] = labels.clamp_min(0)  # no ignore here
    h = h.requires_grad_(True)
    w = w.requires_grad_(False)  # frozen output layer
    old_logprobs = torch.randn(N, device="cuda")
    advantages = torch.randn(N, device="cuda")

    out = importance_sampling_loss_function(
        h, w, labels, old_logprobs, advantages, ignore_index=-100, ce_mode="fused_quack"
    )
    out.loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert w.grad is None, "frozen output head must receive no gradient"
