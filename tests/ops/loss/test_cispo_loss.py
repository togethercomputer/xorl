import pytest
import torch
import torch.nn.functional as F

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss import TokenPartial, cispo_loss_function, get_loss_function


pytestmark = pytest.mark.cpu

IGNORE_INDEX = -100


@pytest.fixture
def inputs():
    torch.manual_seed(11)
    batch, seq, vocab, hidden = 3, 5, 12, 16
    hidden_states = torch.randn(batch, seq, hidden) / (hidden**0.5)
    weight = torch.randn(vocab, hidden)
    labels = torch.randint(0, vocab, (batch, seq))
    mask = torch.tensor(
        [[1, 1, 0, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    labels[~mask] = IGNORE_INDEX
    return {
        "hidden_states": hidden_states,
        "weight": weight,
        "labels": labels,
        "old_logprobs": torch.randn(batch, seq) * 0.3 - 1.5,
        "advantages": torch.randn(batch, seq),
        "mask": mask,
    }


def call_loss(data, **kwargs):
    return cispo_loss_function(
        hidden_states=data["hidden_states"],
        weight=data["weight"],
        labels=data["labels"],
        old_logprobs=data["old_logprobs"],
        advantages=data["advantages"],
        ignore_index=IGNORE_INDEX,
        ce_mode="eager",
        **kwargs,
    )


def test_registry_exposes_cispo():
    assert get_loss_function("cispo") is cispo_loss_function


def test_matches_detached_clipped_ratio_objective(inputs):
    unit = TokenPartial(scale=torch.tensor(1.0))
    output = call_loss(inputs, loss_reducer=unit, metric_reducer=unit)
    ratio = torch.exp(output.per_token_logprobs - inputs["old_logprobs"])
    clipped = torch.clamp(ratio, 0.0, 4.0)
    reference = -(clipped.detach() * output.per_token_logprobs * inputs["advantages"])[inputs["mask"]].sum()
    assert_close(output.loss, reference)


def test_gradient_matches_reference_and_survives_clipping(inputs):
    unit = TokenPartial(scale=torch.tensor(1.0))
    hidden_states = inputs["hidden_states"].clone().requires_grad_(True)
    data = {**inputs, "hidden_states": hidden_states}
    output = call_loss(
        data,
        loss_reducer=unit,
        metric_reducer=unit,
        clip_low_threshold=0.9,
        clip_high_threshold=1.1,
    )
    output.loss.backward()

    reference_hidden = inputs["hidden_states"].clone().requires_grad_(True)
    logits = reference_hidden @ inputs["weight"].T
    per_token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        inputs["labels"].reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).reshape(inputs["labels"].shape)
    logprobs = -per_token_ce
    clipped = torch.clamp(torch.exp(logprobs.detach() - inputs["old_logprobs"]), 0.9, 1.1).detach()
    reference_loss = -(clipped * logprobs * inputs["advantages"])[inputs["mask"]].sum()
    reference_loss.backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.isfinite().all()
    assert hidden_states.grad.norm() > 0
    assert_close(hidden_states.grad, reference_hidden.grad)


def test_metrics_and_kl_diagnostics(inputs):
    output = call_loss(
        inputs,
        clip_low_threshold=0.9,
        clip_high_threshold=1.1,
        compute_kl_stats=True,
    )
    assert output.metrics["clip_fraction"] > 0
    for key in (
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "kl_sample_train_k3",
        "entropy_sample",
        "valid_tokens",
    ):
        assert key in output.metrics


def test_all_ignored_is_finite():
    hidden_states = torch.randn(1, 2, 4, requires_grad=True)
    output = cispo_loss_function(
        hidden_states=hidden_states,
        weight=torch.randn(8, 4),
        labels=torch.full((1, 2), IGNORE_INDEX),
        old_logprobs=torch.zeros(1, 2),
        advantages=torch.ones(1, 2),
        ce_mode="eager",
    )
    assert output.loss.isfinite()
    assert output.loss == 0


@pytest.mark.parametrize(
    ("low", "high"),
    [(-0.1, 4.0), (2.0, 1.0)],
)
def test_invalid_bounds_rejected(inputs, low, high):
    with pytest.raises(ValueError):
        call_loss(inputs, clip_low_threshold=low, clip_high_threshold=high)
