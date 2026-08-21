"""Tests for the full-vocab and KL-estimator OPD loss modes.

xorl ships teacher hidden states + the cached teacher LM head, so the full
teacher distribution is materialized at loss time. That dominates VERL's
truncated `forward_kl_topk` for any loss/diagnostic that benefits from the
full distribution — these tests cover the full-vocab equivalents plus the
single-sample KL estimator family ported from VERL's kl_penalty.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from xorl.objectives.opd_loss import (
    LOSS_MODE_FORWARD_KL_FULL,
    _kl_penalty_estimator,
    opd_loss_function,
)
from xorl.ops.loss.compiled_cross_entropy import (
    compiled_sampled_token_logprobs_function,
)


pytestmark = [pytest.mark.cpu]


def _make_synthetic_inputs(
    batch: int = 2,
    seq: int = 4,
    hidden: int = 8,
    vocab: int = 16,
    seed: int = 0,
):
    """Build student/teacher hidden states + LM heads + labels for CPU tests."""
    torch.manual_seed(seed)
    student_hidden = torch.randn(batch, seq, hidden, dtype=torch.float32, requires_grad=True)
    teacher_hidden = torch.randn(batch, seq, hidden, dtype=torch.float32)
    student_weight = torch.randn(vocab, hidden, dtype=torch.float32, requires_grad=True)
    teacher_weight = torch.randn(vocab, hidden, dtype=torch.float32)
    labels = torch.randint(0, vocab, (batch, seq), dtype=torch.long)
    # Mask one token per sequence
    labels[:, 0] = -100
    return student_hidden, student_weight, teacher_hidden, teacher_weight, labels


# ---------------------------------------------------------------------------
# Loss-mode dispatch
# ---------------------------------------------------------------------------


def test_full_vocab_modes_diagnostics_weighting_clamp_and_dispatch_policy():
    """Full-vocab modes expose one stable policy across diagnostics and weighting branches."""
    sh, sw, th, tw, labels = _make_synthetic_inputs()
    reverse = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
    )
    # Diagnostic fields default to 0.0 because emit_full_vocab_diagnostics=False.
    assert reverse.metrics["opd_teacher_entropy"] == 0.0
    assert reverse.metrics["opd_top1_agreement"] == 0.0
    assert reverse.metrics["opd_pg_clipfrac"] == 0.0
    # PG-mode-only metrics default to 0.0 when use_policy_gradient=False.
    assert reverse.metrics["opd_pg_clipfrac_lower"] == 0.0
    assert reverse.metrics["opd_ppo_kl"] == 0.0
    assert reverse.metrics["valid_tokens"] > 0
    assert torch.isfinite(reverse.loss).item()

    reverse_diagnostics = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        emit_full_vocab_diagnostics=True,
    )
    assert reverse_diagnostics.metrics["opd_teacher_entropy"] >= 0.0
    assert reverse_diagnostics.metrics["opd_student_entropy"] >= 0.0
    assert 0.0 <= reverse_diagnostics.metrics["opd_top1_agreement"] <= 1.0

    # Reference: eager computation.
    s_logits = sh @ sw.t()
    t_logits = th @ tw.t()
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    ref_token_kl = (t_log_probs.exp() * (t_log_probs - s_log_probs)).sum(dim=-1)
    valid = (labels != -100).to(ref_token_kl.dtype)
    ref_token_kl = ref_token_kl * valid
    ref_mean = ref_token_kl[labels != -100].mean().item()

    forward = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        loss_mode=LOSS_MODE_FORWARD_KL_FULL,
    )
    assert forward.metrics["opd_kl"] == pytest.approx(ref_mean, rel=1e-5, abs=1e-5)

    forward_diagnostics = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        loss_mode=LOSS_MODE_FORWARD_KL_FULL,
        emit_full_vocab_diagnostics=True,
    )
    assert forward_diagnostics.metrics["opd_teacher_entropy"] >= 0.0
    assert 0.0 <= forward_diagnostics.metrics["opd_top1_agreement"] <= 1.0

    with pytest.raises(ValueError, match="forward_kl_topk"):
        opd_loss_function(
            hidden_states=sh,
            weight=sw,
            labels=labels,
            teacher_hidden_states=th,
            teacher_lm_head_weight=tw,
            loss_mode="forward_kl_topk",
        )

    clamped = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        loss_max_clamp=0.1,
    )
    assert clamped.metrics["opd_loss_max"] <= 0.1 + 1e-6
    assert clamped.metrics["opd_loss_min"] >= -0.1 - 1e-6

    base = opd_loss_function(
        hidden_states=sh.clone().detach().requires_grad_(True),
        weight=sw.clone().detach().requires_grad_(True),
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
    )
    scaled = opd_loss_function(
        hidden_states=sh.clone().detach().requires_grad_(True),
        weight=sw.clone().detach().requires_grad_(True),
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        use_task_rewards=True,
        distillation_loss_coef=2.5,
    )
    assert scaled.loss.item() == pytest.approx(base.loss.item() * 2.5, rel=1e-5)

    with_disabled_coef = opd_loss_function(
        hidden_states=sh.clone().detach().requires_grad_(True),
        weight=sw.clone().detach().requires_grad_(True),
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        use_task_rewards=False,
        distillation_loss_coef=999.0,
    )
    assert with_disabled_coef.loss.item() == pytest.approx(base.loss.item(), rel=1e-5)

    common_args = dict(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
    )
    keys_reverse = set(opd_loss_function(**common_args).metrics)
    keys_reverse_diag = set(opd_loss_function(**common_args, emit_full_vocab_diagnostics=True).metrics)
    keys_forward = set(opd_loss_function(**common_args, loss_mode=LOSS_MODE_FORWARD_KL_FULL).metrics)
    keys_forward_diag = set(
        opd_loss_function(
            **common_args,
            loss_mode=LOSS_MODE_FORWARD_KL_FULL,
            emit_full_vocab_diagnostics=True,
        ).metrics
    )
    keys_k3 = set(opd_loss_function(**common_args, loss_mode="k3").metrics)
    assert keys_reverse == keys_reverse_diag == keys_forward == keys_forward_diag == keys_k3
    _assert_kl_estimator_values_straight_through_gradient_and_dispatch()
    _assert_policy_gradient_mode_requires_inputs_and_emits_finite_metrics()
    _assert_compiled_sampled_token_logprobs_safe_with_ignored_labels()


# ---------------------------------------------------------------------------
# KL estimators (k1/k2/k3/abs/mse/low_var_kl) — byte-for-byte against VERL formula
# ---------------------------------------------------------------------------


def _assert_kl_estimator_values_straight_through_gradient_and_dispatch():
    """Estimator modes match VERL, including k3+ gradients and OPD dispatch."""
    torch.manual_seed(7)
    logp = torch.randn(32) * 0.5
    ref = torch.randn(32) * 0.5

    for mode in ("k1", "kl", "abs", "mse", "k2", "k3", "low_var_kl"):
        out = _kl_penalty_estimator(logp, ref, mode)

        if mode in ("kl", "k1"):
            expected = logp - ref
        elif mode == "abs":
            expected = (logp - ref).abs()
        elif mode in ("mse", "k2"):
            expected = 0.5 * (logp - ref).square()
        elif mode in ("k3", "low_var_kl"):
            kl = (ref - logp).clamp(min=-20, max=20)
            expected = (kl.exp() - kl - 1).clamp(min=-10, max=10)
        else:
            raise AssertionError(mode)
        torch.testing.assert_close(out, expected)

    torch.manual_seed(11)
    grad_logp = torch.randn(8, requires_grad=True)
    grad_ref = torch.randn(8)

    out = _kl_penalty_estimator(grad_logp, grad_ref, "k3+")
    # Forward value matches plain k3.
    out_k3 = _kl_penalty_estimator(grad_logp.detach(), grad_ref, "k3")
    torch.testing.assert_close(out.detach(), out_k3)
    # Backward gradient matches k2.
    out.sum().backward()
    expected_grad = grad_logp.detach() - grad_ref  # d/dlogp of 0.5*(logp-ref)^2
    torch.testing.assert_close(grad_logp.grad, expected_grad)

    sh, sw, th, tw, labels = _make_synthetic_inputs()
    result = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        loss_mode="k3",
    )
    # k3 estimator is always >= -10 by construction.
    assert result.metrics["opd_loss_min"] >= -10.0
    assert result.metrics["opd_abs_loss"] >= 0.0
    # Full-vocab diagnostics are zero on the estimator path.
    assert result.metrics["opd_teacher_entropy"] == 0.0


# ---------------------------------------------------------------------------
# Policy-gradient mode
# ---------------------------------------------------------------------------


def _assert_policy_gradient_mode_requires_inputs_and_emits_finite_metrics():
    sh, sw, th, tw, labels = _make_synthetic_inputs()
    with pytest.raises(ValueError, match="old_logprobs"):
        opd_loss_function(
            hidden_states=sh,
            weight=sw,
            labels=labels,
            teacher_hidden_states=th,
            teacher_lm_head_weight=tw,
            loss_mode="k1",
            use_policy_gradient=True,
        )

    # Fake old logprobs ~ student's current logprob to keep ratio near 1.
    old_lp = torch.zeros_like(labels, dtype=torch.float32) - 2.5
    result = opd_loss_function(
        hidden_states=sh,
        weight=sw,
        labels=labels,
        teacher_hidden_states=th,
        teacher_lm_head_weight=tw,
        loss_mode="k1",
        use_policy_gradient=True,
        old_logprobs=old_lp,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert torch.isfinite(result.loss).item()
    # opd_pg_clipfrac is in [0, 1].
    assert 0.0 <= result.metrics["opd_pg_clipfrac"] <= 1.0
    # VERL-parity: opd_pg_clipfrac_lower in [0, 1], opd_ppo_kl finite.
    # (`actor/pg_clipfrac_lower`, `actor/ppo_kl` in core_algos.py:1365-1367.)
    assert 0.0 <= result.metrics["opd_pg_clipfrac_lower"] <= 1.0
    assert "opd_ppo_kl" in result.metrics
    assert math.isfinite(result.metrics["opd_ppo_kl"])
    # Backward-compatibility: opd_kl still emitted (the distillation KL value).
    assert "opd_kl" in result.metrics


def _assert_compiled_sampled_token_logprobs_safe_with_ignored_labels():
    sh, sw, th, tw, labels = _make_synthetic_inputs()
    sh_flat = sh.reshape(-1, sh.size(-1))
    th_flat = th.reshape(-1, th.size(-1))
    lab_flat = labels.reshape(-1)
    # Pass ignore_index labels through — the function clamps gather index to >= 0
    # internally and zeros via valid mask, so this must not throw.
    student_lp, teacher_lp = compiled_sampled_token_logprobs_function(
        student_hidden_states=sh_flat,
        student_weight=sw,
        teacher_hidden_states=th_flat,
        teacher_weight=tw,
        labels=lab_flat,
        ignore_index=-100,
    )
    assert student_lp.shape == teacher_lp.shape == lab_flat.shape
    assert (student_lp[lab_flat == -100] == 0).all()
    assert (teacher_lp[lab_flat == -100] == 0).all()
