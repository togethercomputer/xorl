"""Streaming forward-KL parity + correctness (CPU, hermetic).

`_StreamingForwardKL` is the forward-KL counterpart of `_StreamingReverseKL`:
it computes per-token KL(p_T || p_S) over vocab chunks without ever materializing
the full-vocab logits. These tests pin it to a naive dense autograd reference
(ground truth) for both forward value and gradients, plus the existing compiled
forward-KL reference, chunking invariance, ignore-index masking, lowmem parity,
and a tiny fp64 gradcheck.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.ops.loss.conftest import assert_close
from xorl.objectives.opd_streaming_kl import (
    streaming_forward_kl_function,
    streaming_forward_kl_lowmem_function,
)
from xorl.ops.loss import opd_loss_function


pytestmark = pytest.mark.cpu

IGNORE_INDEX = -100


def _dense_forward_kl(student_hidden, student_weight, teacher_hidden, teacher_weight, labels):
    """Ground-truth dense forward KL(p_T || p_S), per token, zeroed at ignore_index.

    Teacher is detached (no grad), so gradients flow only into the student tensors.
    """
    s_logits = (student_hidden @ student_weight.t()).float()
    t_logits = (teacher_hidden @ teacher_weight.t()).float().detach()
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    t_probs = t_log_probs.exp()
    token_kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    valid = (labels != IGNORE_INDEX).to(token_kl.dtype)
    return token_kl * valid


@pytest.fixture
def inputs():
    torch.manual_seed(11)
    tokens, student_h, teacher_h, vocab = 8, 16, 12, 40
    student_hidden = torch.randn(tokens, student_h, dtype=torch.float32) / student_h**0.5
    student_weight = torch.randn(vocab, student_h, dtype=torch.float32) / student_h**0.5
    teacher_hidden = torch.randn(tokens, teacher_h, dtype=torch.float32) / teacher_h**0.5
    teacher_weight = torch.randn(vocab, teacher_h, dtype=torch.float32) / teacher_h**0.5
    labels = torch.randint(0, vocab, (tokens,))
    # A couple of ignore-index tokens to exercise the masking path.
    labels[0] = IGNORE_INDEX
    labels[5] = IGNORE_INDEX
    return student_hidden, student_weight, teacher_hidden, teacher_weight, labels


def test_backward_matches_dense_reference(inputs):
    student_hidden, student_weight, teacher_hidden, teacher_weight, labels = inputs

    # Streaming path.
    sh_a = student_hidden.clone().requires_grad_(True)
    sw_a = student_weight.clone().requires_grad_(True)
    th_a = teacher_hidden.clone().requires_grad_(True)
    tw_a = teacher_weight.clone().requires_grad_(True)
    kl_a = streaming_forward_kl_function(sh_a, sw_a, th_a, tw_a, labels, vocab_chunk_size=7)
    kl_a.sum().backward()

    # Dense ground-truth.
    sh_b = student_hidden.clone().requires_grad_(True)
    sw_b = student_weight.clone().requires_grad_(True)
    th_b = teacher_hidden.clone().requires_grad_(True)
    tw_b = teacher_weight.clone().requires_grad_(True)
    kl_b = _dense_forward_kl(sh_b, sw_b, th_b, tw_b, labels)
    kl_b.sum().backward()

    assert_close(kl_a, kl_b)
    assert_close(sh_a.grad, sh_b.grad)
    assert_close(sw_a.grad, sw_b.grad)
    # Teacher is detached on both paths -> no teacher grad.
    assert th_a.grad is None
    assert tw_a.grad is None
    assert th_b.grad is None
    assert tw_b.grad is None

    _assert_chunking_invariance(inputs)
    _assert_ignore_index_zero_loss_and_grad(inputs)
    _assert_lowmem_matches_streaming(inputs)
    _assert_opd_loss_function_forward_kl_streaming_matches_compiled()


def _assert_chunking_invariance(inputs):
    """Online accumulation must be identical across chunkings (multi-chunk vs one)."""
    student_hidden, student_weight, teacher_hidden, teacher_weight, labels = inputs

    sh = student_hidden.clone().requires_grad_(True)
    sw = student_weight.clone().requires_grad_(True)
    kl = streaming_forward_kl_function(sh, sw, teacher_hidden, teacher_weight, labels, vocab_chunk_size=7)
    kl.sum().backward()

    # Single-chunk reference (chunk size equals the fixture's vocabulary size).
    sh_ref = student_hidden.clone().requires_grad_(True)
    sw_ref = student_weight.clone().requires_grad_(True)
    kl_ref = streaming_forward_kl_function(sh_ref, sw_ref, teacher_hidden, teacher_weight, labels, vocab_chunk_size=40)
    kl_ref.sum().backward()

    assert_close(kl, kl_ref)
    assert_close(sh.grad, sh_ref.grad)
    assert_close(sw.grad, sw_ref.grad)


def _assert_ignore_index_zero_loss_and_grad(inputs):
    student_hidden, student_weight, teacher_hidden, teacher_weight, labels = inputs
    sh = student_hidden.clone().requires_grad_(True)
    sw = student_weight.clone().requires_grad_(True)
    kl = streaming_forward_kl_function(sh, sw, teacher_hidden, teacher_weight, labels, vocab_chunk_size=7)
    kl.sum().backward()

    ignored = labels == IGNORE_INDEX
    assert ignored.any()
    # Ignored tokens contribute exactly zero loss.
    assert torch.equal(kl[ignored], torch.zeros_like(kl[ignored]))
    # ... and zero gradient into the per-token student hidden rows.
    assert torch.equal(sh.grad[ignored], torch.zeros_like(sh.grad[ignored]))
    # Valid tokens are not zero.
    assert kl[~ignored].abs().sum() > 0


def _assert_lowmem_matches_streaming(inputs):
    """The lowmem forward-KL path must be loss- and gradient-identical to plain streaming."""
    student_hidden, student_weight, teacher_hidden, teacher_weight, labels = inputs

    def run(fn):
        sh = student_hidden.clone().requires_grad_(True)
        sw = student_weight.clone().requires_grad_(True)
        kl = fn(sh, sw, teacher_hidden, teacher_weight, labels, vocab_chunk_size=7)
        kl.sum().backward()
        return kl, sh.grad, sw.grad

    kl_a, gh_a, gw_a = run(streaming_forward_kl_function)
    kl_b, gh_b, gw_b = run(streaming_forward_kl_lowmem_function)

    assert_close(kl_b, kl_a)
    assert_close(gh_b, gh_a)
    assert_close(gw_b, gw_a)


def _assert_opd_loss_function_forward_kl_streaming_matches_compiled():
    """End-to-end dispatch: forward_kl_full on the streaming backend matches the
    compiled (torch_compile) backend through opd_loss_function, with finite grads.
    """
    torch.manual_seed(13)
    batch, seq, vocab, student_h, teacher_h = 2, 5, 17, 6, 8
    hidden_states = torch.randn(batch, seq, student_h) / student_h**0.5
    weight = torch.randn(vocab, student_h) / student_h**0.5
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, 0] = IGNORE_INDEX
    teacher_hidden_states = torch.randn(batch, seq, teacher_h) / teacher_h**0.5
    teacher_weight = torch.randn(vocab, teacher_h) / teacher_h**0.5
    teacher_weights = torch.linspace(0.5, 1.5, steps=batch * seq).view(batch, seq)

    def run(kl_backend):
        hs = hidden_states.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        out = opd_loss_function(
            hidden_states=hs,
            weight=w,
            labels=labels,
            teacher_hidden_states=teacher_hidden_states,
            teacher_lm_head_weight=teacher_weight,
            teacher_weights=teacher_weights,
            kl_backend=kl_backend,
            vocab_chunk_size=5,
            loss_mode="forward_kl_full",
        )
        out.loss.backward()
        return out.loss, hs.grad, w.grad

    loss_c, gh_c, gw_c = run("torch_compile")
    for backend in ("streaming", "tilelang"):
        loss_s, gh_s, gw_s = run(backend)

        assert_close(loss_s, loss_c)
        assert_close(gh_s, gh_c)
        assert_close(gw_s, gw_c)
        assert gh_s.isfinite().all() and gw_s.isfinite().all()

    _assert_opd_loss_function_forward_kl_streaming_rejects_logprob_clamp()


def _assert_opd_loss_function_forward_kl_streaming_rejects_logprob_clamp():
    """The streaming forward-KL backend never materializes student log-probs, so
    log_prob_min_clamp must fail loud rather than be silently ignored.
    """
    torch.manual_seed(13)
    batch, seq, vocab, student_h, teacher_h = 2, 4, 9, 6, 8
    hidden_states = torch.randn(batch, seq, student_h)
    weight = torch.randn(vocab, student_h)
    labels = torch.randint(0, vocab, (batch, seq))
    teacher_hidden_states = torch.randn(batch, seq, teacher_h)
    teacher_weight = torch.randn(vocab, teacher_h)

    with pytest.raises(ValueError, match="log_prob_min_clamp"):
        opd_loss_function(
            hidden_states=hidden_states,
            weight=weight,
            labels=labels,
            teacher_hidden_states=teacher_hidden_states,
            teacher_lm_head_weight=teacher_weight,
            kl_backend="streaming",
            vocab_chunk_size=5,
            loss_mode="forward_kl_full",
            log_prob_min_clamp=-30.0,
        )


def test_gradcheck_fp64():
    """fp64 gradcheck on a tiny instance -- the strongest correctness guard."""
    torch.manual_seed(3)
    tokens, student_h, teacher_h, vocab = 4, 5, 4, 9
    student_hidden = torch.randn(tokens, student_h, dtype=torch.float64).requires_grad_(True)
    student_weight = torch.randn(vocab, student_h, dtype=torch.float64).requires_grad_(True)
    teacher_hidden = torch.randn(tokens, teacher_h, dtype=torch.float64)
    teacher_weight = torch.randn(vocab, teacher_h, dtype=torch.float64)
    labels = torch.randint(0, vocab, (tokens,))
    labels[1] = IGNORE_INDEX

    def fn(sh, sw):
        return streaming_forward_kl_function(sh, sw, teacher_hidden, teacher_weight, labels, vocab_chunk_size=3)

    assert torch.autograd.gradcheck(fn, (student_hidden, student_weight), atol=1e-6, rtol=1e-4)
