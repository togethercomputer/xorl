import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss import TokenPartial, importance_sampling_loss_function


_IGNORE = -100


@pytest.fixture
def inputs():
    torch.manual_seed(11)
    B, S, V, H = 3, 5, 12, 16

    hidden_states = torch.randn(B, S, H) / (H**0.5)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))

    mask_pattern = torch.tensor(
        [
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    labels_with_mask = labels.clone()
    labels_with_mask[~mask_pattern] = _IGNORE

    return {
        "B": B,
        "hidden_states": hidden_states,
        "weight": weight,
        "labels": labels_with_mask,
        "old_logprobs": torch.randn(B, S) * 0.3 - 1.5,
        "advantages": torch.randn(B, S),
    }


def _call(d, slc, *, loss_reducer=None, metric_reducer=None, **kwargs):
    return importance_sampling_loss_function(
        hidden_states=d["hidden_states"][slc],
        weight=d["weight"],
        labels=d["labels"][slc],
        old_logprobs=d["old_logprobs"][slc],
        advantages=d["advantages"][slc],
        ignore_index=_IGNORE,
        ce_mode="eager",
        loss_reducer=loss_reducer,
        metric_reducer=metric_reducer,
        **kwargs,
    )


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="basic"),
        pytest.param({"compute_kl_stats": True}, id="kl_stats"),
    ],
)
def test_identity_against_legacy(inputs, extra):
    d = inputs
    legacy = _call(d, slice(None), **extra)

    mask = (d["labels"] != _IGNORE).float()
    reducer = TokenPartial(scale=mask.sum())
    explicit = _call(d, slice(None), loss_reducer=reducer, metric_reducer=reducer, **extra)

    assert_close(explicit.loss, legacy.loss)
    for key, expected in legacy.metrics.items():
        assert key in explicit.metrics
        assert_close(
            torch.as_tensor(explicit.metrics[key], dtype=torch.float64),
            torch.as_tensor(expected, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="basic"),
        pytest.param({"compute_kl_stats": True}, id="kl_stats"),
    ],
)
def test_microbatch_composition(inputs, extra):
    d = inputs
    B = d["B"]

    mask = (d["labels"] != _IGNORE).float()
    loss_reducer = TokenPartial(scale=mask.sum())
    metric_reducer = TokenPartial(scale=mask.sum())

    single = _call(d, slice(None), loss_reducer=loss_reducer, metric_reducer=metric_reducer, **extra)
    mbs = [
        _call(d, slice(b, b + 1), loss_reducer=loss_reducer, metric_reducer=metric_reducer, **extra) for b in range(B)
    ]

    assert_close(sum(mb.loss for mb in mbs), single.loss)

    composing = {"ratio_mean", "kl_sample_train_k3", "entropy_sample"}
    for key, expected in single.metrics.items():
        if key not in composing:
            continue
        assert_close(
            torch.as_tensor(sum(mb.metrics[key] for mb in mbs), dtype=torch.float64),
            torch.as_tensor(expected, dtype=torch.float64),
        )
