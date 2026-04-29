"""Tests for the softmax auxiliary (Z-)loss term in causallm_loss_function."""

import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss.causallm_loss import causallm_loss_function


def _reference_z_loss(hidden_states, weight, labels, ignore_index=-100):
    """Reference matching OLMo's Z-loss exactly.

    OLMo (olmo/train.py::cross_entropy_loss with reduction="sum",
    then divided by batch_size_in_tokens in train_micro_batch) computes:
        z_squared = logits.logsumexp(-1).pow(2)
        z_squared = (z_squared * (labels != ignore_index)).sum()
        z_loss   = z_squared / num_valid_tokens
    """
    h = hidden_states.view(-1, hidden_states.size(-1))
    lab = labels.view(-1)
    logits = (h.float() @ weight.float().t()).float()
    z_squared = logits.logsumexp(-1).pow(2)
    z_squared = (z_squared * (lab != ignore_index)).sum()
    return z_squared / (lab != ignore_index).sum().clamp(min=1)


def _reference_ce_loss(hidden_states, weight, labels, ignore_index=-100):
    h = hidden_states.view(-1, hidden_states.size(-1))
    lab = labels.view(-1)
    logits = (h @ weight.t()).float()
    valid_count = (lab != ignore_index).sum().clamp(min=1)
    per_token = torch.nn.functional.cross_entropy(logits, lab, reduction="none", ignore_index=ignore_index)
    return per_token.sum() / valid_count


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    B, S, V, H = 2, 6, 32, 16
    hidden_states = torch.randn(B, S, H) / (H**0.5)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    # Mark a couple positions as ignore so masking is exercised.
    labels[0, 0] = -100
    labels[1, -1] = -100
    return hidden_states, weight, labels


def test_eager_z_loss_matches_reference(inputs):
    hidden_states, weight, labels = inputs
    ce_ref = _reference_ce_loss(hidden_states, weight, labels)
    z_ref = _reference_z_loss(hidden_states, weight, labels)
    coef = 1e-3

    out = causallm_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        ce_mode="eager",
        z_loss_coef=coef,
    )

    assert out.metrics is not None
    assert_close(out.metrics["ce_loss"], ce_ref)
    assert_close(out.metrics["z_loss"], z_ref)
    assert_close(out.loss, ce_ref + coef * z_ref)


def test_eager_no_z_loss_when_coef_zero(inputs):
    hidden_states, weight, labels = inputs
    ce_ref = _reference_ce_loss(hidden_states, weight, labels)

    out = causallm_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        ce_mode="eager",
        z_loss_coef=0.0,
    )

    assert out.metrics is None
    assert_close(out.loss, ce_ref)


def test_eager_z_loss_grad_flows(inputs):
    hidden_states, weight, labels = inputs
    hidden_states = hidden_states.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)

    out = causallm_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        ce_mode="eager",
        z_loss_coef=1.0,
    )
    out.loss.backward()

    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_eager_z_loss_zero_when_logits_centered():
    """If logits are all zeros, logsumexp = log(V) (constant) so Z-loss = log(V)^2."""
    torch.manual_seed(1)
    B, S, V, H = 1, 3, 8, 4
    hidden_states = torch.zeros(B, S, H)
    weight = torch.zeros(V, H)
    labels = torch.zeros(B, S, dtype=torch.long)

    out = causallm_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        ce_mode="eager",
        z_loss_coef=1.0,
    )
    expected_z = torch.tensor(float(torch.log(torch.tensor(V)).item() ** 2))
    assert_close(out.metrics["z_loss"], expected_z)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled CE+LSE^2 path requires CUDA")
def test_compiled_z_loss_matches_eager(inputs):
    """The fused/compiled CE+LSE^2 kernel must agree with the eager reference."""
    hidden_states, weight, labels = inputs
    hidden_states = hidden_states.cuda()
    weight = weight.cuda()
    labels = labels.cuda()

    coef = 1e-3
    out_eager = causallm_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        ce_mode="eager",
        z_loss_coef=coef,
    )
    out_compiled = causallm_loss_function(
        hidden_states=hidden_states,
        weight=weight,
        labels=labels,
        ce_mode="compiled",
        num_chunks=2,
        z_loss_coef=coef,
    )

    assert_close(out_compiled.metrics["ce_loss"], out_eager.metrics["ce_loss"])
    assert_close(out_compiled.metrics["z_loss"], out_eager.metrics["z_loss"])
    assert_close(out_compiled.loss, out_eager.loss)


def test_tp_path_rejects_z_loss():
    """TP path must error out clearly when Z-loss is requested."""
    torch.manual_seed(2)
    B, S, V, H = 1, 2, 8, 4
    hidden_states = torch.randn(B, S, H)
    weight = torch.randn(V, H)
    labels = torch.zeros(B, S, dtype=torch.long)

    # Pass a non-None tp_group sentinel; we want to fail before any collective.
    class _Sentinel:
        pass

    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        causallm_loss_function(
            hidden_states=hidden_states,
            weight=weight,
            labels=labels,
            tp_group=_Sentinel(),
            z_loss_coef=1e-3,
        )
