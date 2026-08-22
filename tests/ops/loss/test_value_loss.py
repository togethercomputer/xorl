import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.lora.modules.linear import LoraLinear
from xorl.ops.loss import (
    TokenPartial,
    get_loss_function,
    value_loss_function,
    value_prediction_function,
)


pytestmark = pytest.mark.cpu

IGNORE_INDEX = -100


@pytest.fixture
def inputs():
    torch.manual_seed(7)
    batch, seq, hidden = 2, 6, 16
    hidden_states = torch.randn(batch, seq, hidden) / (hidden**0.5)
    weight = torch.randn(1, hidden)
    labels = torch.randint(0, 100, (batch, seq))
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1]],
        dtype=torch.bool,
    )
    labels[~mask] = IGNORE_INDEX
    returns = torch.randn(batch, seq)
    return {
        "hidden_states": hidden_states,
        "weight": weight,
        "labels": labels,
        "returns": returns,
        "mask": mask,
    }


def expected_values(data):
    return (data["hidden_states"].reshape(-1, data["hidden_states"].size(-1)).float() @ data["weight"].float().T).view(
        data["labels"].shape
    )


def test_registry_exposes_value_losses():
    assert get_loss_function("value_loss") is value_loss_function
    assert get_loss_function("value_prediction") is value_prediction_function


def test_value_loss_matches_masked_mse(inputs):
    unit = TokenPartial(scale=torch.tensor(1.0))
    output = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
        loss_reducer=unit,
        metric_reducer=unit,
    )
    values = expected_values(inputs)
    reference = 0.5 * ((values - inputs["returns"]) ** 2)[inputs["mask"]].sum()
    assert_close(output.loss, reference)
    # Per-token channel carries the values, not logprobs.
    assert_close(output.per_token_logprobs, values)


def test_value_loss_raw_sum_reducer_contract(inputs):
    """With scale=1 reducers, loss must be the raw masked sum (server contract)."""
    unit = TokenPartial(scale=torch.tensor(1.0))
    full = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
        loss_reducer=unit,
        metric_reducer=unit,
    )
    # Splitting the batch into two micro-batches must sum to the same loss.
    parts = []
    for row in range(2):
        parts.append(
            value_loss_function(
                hidden_states=inputs["hidden_states"][row : row + 1],
                weight=inputs["weight"],
                labels=inputs["labels"][row : row + 1],
                returns=inputs["returns"][row : row + 1],
                loss_reducer=unit,
                metric_reducer=unit,
            ).loss
        )
    assert_close(full.loss, parts[0] + parts[1])


def test_value_loss_default_reducer_is_token_mean(inputs):
    output = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
    )
    values = expected_values(inputs)
    reference = 0.5 * ((values - inputs["returns"]) ** 2)[inputs["mask"]].mean()
    assert_close(output.loss, reference)


def test_value_loss_clipping(inputs):
    unit = TokenPartial(scale=torch.tensor(1.0))
    old_values = expected_values(inputs) + torch.randn_like(inputs["returns"])
    clip_range = 0.1
    output = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
        old_values=old_values,
        clip_range=clip_range,
        loss_reducer=unit,
        metric_reducer=unit,
    )
    values = expected_values(inputs)
    clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
    reference = (
        0.5 * torch.maximum((values - inputs["returns"]) ** 2, (clipped - inputs["returns"]) ** 2)[inputs["mask"]].sum()
    )
    assert_close(output.loss, reference)
    assert "value_clip_fraction" in output.metrics


def test_value_loss_vf_coef_scales_loss(inputs):
    unit = TokenPartial(scale=torch.tensor(1.0))
    base = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
        loss_reducer=unit,
        metric_reducer=unit,
    ).loss
    scaled = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
        vf_coef=0.25,
        loss_reducer=unit,
        metric_reducer=unit,
    ).loss
    assert_close(scaled, 0.25 * base)


def test_value_loss_metrics_are_sum_composable_means(inputs):
    unit = TokenPartial(scale=torch.tensor(1.0))
    output = value_loss_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        returns=inputs["returns"],
        loss_reducer=unit,
        metric_reducer=unit,
    )
    values = expected_values(inputs)
    mask = inputs["mask"]
    assert_close(output.metrics["return_mean"], inputs["returns"][mask].sum())
    assert_close(output.metrics["value_mean"], values[mask].sum())
    assert_close(output.metrics["value_error_sq_mean"], ((values - inputs["returns"]) ** 2)[mask].sum())
    assert output.metrics["valid_tokens"] == int(mask.sum())


def test_value_loss_gradients_flow_through_lora_value_head(inputs):
    """The server consumes the head via its straight-through folded weight;
    gradients must reach the LoRA factors and the hidden states, never the
    frozen zero base weight."""
    value_head = LoraLinear(16, 1, r=4, lora_alpha=4, bias=False)
    with torch.no_grad():
        value_head.weight.zero_()
        torch.nn.init.normal_(value_head.lora_B, std=0.1)  # nonzero so grads are nontrivial
    value_head.weight.requires_grad_(False)

    hidden_states = inputs["hidden_states"].clone().requires_grad_(True)
    effective_weight = value_head.weight + value_head.get_delta_weight().to(value_head.weight.dtype)
    output = value_loss_function(
        hidden_states=hidden_states,
        weight=effective_weight,
        labels=inputs["labels"],
        returns=inputs["returns"],
        loss_reducer=TokenPartial(scale=torch.tensor(1.0)),
        metric_reducer=TokenPartial(scale=torch.tensor(1.0)),
    )
    output.loss.backward()
    assert value_head.lora_A.grad is not None and value_head.lora_A.grad.abs().sum() > 0
    assert value_head.lora_B.grad is not None and value_head.lora_B.grad.abs().sum() > 0
    assert hidden_states.grad is not None and hidden_states.grad.abs().sum() > 0
    assert value_head.weight.grad is None


def test_zero_base_lora_value_head_predicts_zero_at_init(inputs):
    """Fresh critic (lora_B = 0, base = 0) must predict V(s) = 0 everywhere."""
    value_head = LoraLinear(16, 1, r=4, lora_alpha=4, bias=False)
    with torch.no_grad():
        value_head.weight.zero_()
    effective_weight = value_head.weight + value_head.get_delta_weight().to(value_head.weight.dtype)
    output = value_prediction_function(
        hidden_states=inputs["hidden_states"],
        weight=effective_weight,
        labels=inputs["labels"],
    )
    assert torch.all(output.per_token_logprobs == 0.0)


def test_value_prediction_returns_values_everywhere(inputs):
    output = value_prediction_function(
        hidden_states=inputs["hidden_states"],
        weight=inputs["weight"],
        labels=inputs["labels"],
        metric_reducer=TokenPartial(scale=torch.tensor(1.0)),
    )
    # Values are returned for masked-out positions too (client applies its own
    # action mask for GAE).
    assert_close(output.per_token_logprobs, expected_values(inputs))
    assert output.per_token_loss is None
    assert_close(output.loss, torch.tensor(0.0))


def test_value_prediction_zero_loss_keeps_graph(inputs):
    hidden_states = inputs["hidden_states"].clone().requires_grad_(True)
    output = value_prediction_function(
        hidden_states=hidden_states,
        weight=inputs["weight"],
        labels=inputs["labels"],
    )
    output.loss.backward()  # must not raise; gradients are exactly zero
    assert_close(hidden_states.grad, torch.zeros_like(hidden_states))


def test_value_loss_rejects_bad_weight_shape(inputs):
    with pytest.raises(ValueError, match="1, hidden_size"):
        value_loss_function(
            hidden_states=inputs["hidden_states"],
            weight=torch.randn(4, 16),
            labels=inputs["labels"],
            returns=inputs["returns"],
        )
