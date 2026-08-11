import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss import TokenPartial, importance_sampling_loss_function, policy_loss_function


_IGNORE = -100
_IMPLEMENTATIONS = ("importance_sampling", "policy")


@pytest.fixture
def inputs():
    torch.manual_seed(17)
    batch, sequence, vocab, hidden = 3, 5, 12, 16
    hidden_states = torch.randn(batch, sequence, hidden) / (hidden**0.5)
    labels = torch.randint(0, vocab, (batch, sequence))
    labels[
        ~torch.tensor(
            [
                [1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
    ] = _IGNORE
    return {
        "hidden_states": hidden_states,
        "weight": torch.randn(vocab, hidden),
        "labels": labels,
        "old_logprobs": torch.randn(batch, sequence) * 0.3 - 1.5,
        "rollout_logprobs": torch.randn(batch, sequence) * 0.3 - 1.5,
        "advantages": torch.randn(batch, sequence),
    }


def _call(implementation, inputs, slc=slice(None), *, loss_reducer=None, metric_reducer=None, **kwargs):
    common = {
        "hidden_states": inputs["hidden_states"][slc],
        "weight": inputs["weight"],
        "labels": inputs["labels"][slc],
        "old_logprobs": inputs["old_logprobs"][slc],
        "advantages": inputs["advantages"][slc],
        "ignore_index": _IGNORE,
        "ce_mode": "eager",
        "loss_reducer": loss_reducer,
        "metric_reducer": metric_reducer,
        **kwargs,
    }
    if implementation == "policy":
        return policy_loss_function(rollout_logprobs=inputs["rollout_logprobs"][slc], **common)
    return importance_sampling_loss_function(**common)


def test_token_partial_denominator_composition_and_loss_identity(inputs):
    cases = (
        ("importance_sampling", {}),
        ("importance_sampling", {"compute_kl_stats": True}),
        ("policy", {}),
        ("policy", {"use_tis": True}),
        ("policy", {"icepop_beta": 1.5}),
        ("policy", {"compute_kl_stats": True}),
    )
    reducer = TokenPartial(scale=(inputs["labels"] != _IGNORE).sum())
    for implementation, extra in cases:
        legacy = _call(implementation, inputs, **extra)
        explicit = _call(implementation, inputs, loss_reducer=reducer, metric_reducer=reducer, **extra)

        assert_close(explicit.loss, legacy.loss)
        for key, expected in legacy.metrics.items():
            assert key in explicit.metrics
            assert_close(
                torch.as_tensor(explicit.metrics[key], dtype=torch.float64),
                torch.as_tensor(expected, dtype=torch.float64),
            )

        microbatches = [
            _call(
                implementation,
                inputs,
                slice(row, row + 1),
                loss_reducer=reducer,
                metric_reducer=reducer,
                **extra,
            )
            for row in range(inputs["labels"].size(0))
        ]
        assert_close(sum(result.loss for result in microbatches), explicit.loss)

        composing_metrics = {"ratio_mean", "kl_sample_train_k3", "entropy_sample"}
        if implementation == "policy":
            composing_metrics.update({"pg_clipfrac", "icepop_maskfrac", "tis_mean", "tis_clipfrac"})
        for key in composing_metrics & explicit.metrics.keys():
            assert_close(
                torch.as_tensor(sum(result.metrics[key] for result in microbatches), dtype=torch.float64),
                torch.as_tensor(explicit.metrics[key], dtype=torch.float64),
            )

    torch.manual_seed(0)
    batch, sequence = 4, 6
    values = torch.randn(batch, sequence)
    mask = torch.randint(0, 2, (batch, sequence)).float()

    reducer = TokenPartial(scale=mask.sum())
    single = reducer(values, mask)
    shares = sum(reducer(values[row : row + 1], mask[row : row + 1]) for row in range(batch))
    assert_close(shares, single)
    assert_close(TokenPartial(scale=torch.tensor(1.0))(values, mask), (values * mask).sum())

    sequence_count = torch.tensor(float(batch))
    assert_close(
        TokenPartial(scale=sequence_count)(values, mask),
        (values * mask).sum(dim=-1).sum() / sequence_count,
    )
    assert TokenPartial(scale=torch.tensor(0.0))(torch.randn(2, 4), torch.zeros(2, 4)) == 0.0


def test_behavior_k3_observability_and_temperature_policy(inputs):
    for implementation in _IMPLEMENTATIONS:
        out = _call(implementation, inputs, compute_kl_stats=True)
        valid = inputs["labels"] != _IGNORE
        log_ratio = out.per_token_logprobs - inputs["old_logprobs"]
        k3 = torch.exp(log_ratio) - log_ratio - 1.0

        assert_close(torch.as_tensor(out.metrics["kl_k3_debug_max"]), k3[valid].max())
        assert_close(torch.as_tensor(out.metrics["kl_k3_debug_logratio_min"]), log_ratio[valid].min())
        assert_close(torch.as_tensor(out.metrics["kl_k3_debug_logratio_max"]), log_ratio[valid].max())
        assert_close(torch.as_tensor(out.metrics["kl_k3_debug_abs_logratio_max"]), log_ratio[valid].abs().max())
        assert out.metric_ops["kl_k3_debug_max"] == "max"
        assert out.metric_ops["kl_k3_debug_logratio_min"] == "min"
        assert out.metric_ops["kl_k3_debug_logratio_max"] == "max"

    _assert_logprob_temperature_drives_behavior_k3(inputs)


def _assert_logprob_temperature_drives_behavior_k3(inputs):
    temperature = 0.7
    labels = inputs["labels"]
    logits = (inputs["hidden_states"].reshape(-1, inputs["hidden_states"].size(-1)) @ inputs["weight"].t()).float()
    behavior_ce = torch.nn.functional.cross_entropy(
        logits / temperature,
        labels.reshape(-1),
        reduction="none",
        ignore_index=_IGNORE,
    ).view_as(labels)
    temperature_inputs = {**inputs, "old_logprobs": -behavior_ce}

    for implementation in _IMPLEMENTATIONS:
        out = _call(
            implementation,
            temperature_inputs,
            compute_kl_stats=True,
            logprob_temperature=temperature,
        )

        assert_close(out.per_token_logprobs, -behavior_ce)
        assert_close(torch.as_tensor(out.metrics["kl_sample_train_k3"]), torch.tensor(0.0))
