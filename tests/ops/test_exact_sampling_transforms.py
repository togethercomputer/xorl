import importlib
import math

import pytest
import torch

from xorl.ops.exact_sampling_transforms import (
    TOP_K_ALL,
    exact_sampling_support,
    exact_selected_logprob,
    normalize_exact_sampling_transforms,
)


def _rows(values):
    return torch.tensor(values, dtype=torch.float32)


def test_joint_support_matches_probability_contract_for_mixed_rows():
    logits = _rows(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [3.0, 2.0, 1.0, 0.0, -1.0],
        ]
    )
    top_ks = torch.tensor([2, TOP_K_ALL, TOP_K_ALL], dtype=torch.int64)
    top_ps = _rows([1.0, 0.7, 1.0])
    min_ps = _rows([0.0, 0.0, 0.2])
    support = exact_sampling_support(logits, top_ks, top_ps, min_ps)

    probs = logits.softmax(-1)
    expected = torch.zeros_like(support)
    for row in range(logits.shape[0]):
        ordered = sorted(range(logits.shape[1]), key=lambda token: (-float(probs[row, token]), token))
        cumulative_before = 0.0
        maximum = float(probs[row, ordered[0]])
        for rank, token in enumerate(ordered):
            probability = float(probs[row, token])
            keep = (
                rank < min(int(top_ks[row]), logits.shape[1])
                and cumulative_before <= float(top_ps[row])
                and probability >= maximum * float(min_ps[row])
            )
            expected[row, token] = keep
            cumulative_before += probability
    assert torch.equal(support, expected)


def test_equal_logits_use_token_id_as_stable_tie_break():
    logits = torch.zeros((1, 6), dtype=torch.float32)
    support = exact_sampling_support(
        logits,
        torch.tensor([3], dtype=torch.int64),
        _rows([1.0]),
        _rows([0.0]),
    )
    assert support.tolist() == [[True, True, True, False, False, False]]


def test_historical_action_outside_current_support_is_negative_infinity_with_zero_gradient():
    logits = _rows([[3.0, 2.0, 1.0]]).requires_grad_(True)
    logprob, _, selected_support, _ = exact_selected_logprob(
        logits,
        torch.tensor([2]),
        torch.tensor([1], dtype=torch.int64),
        _rows([1.0]),
        _rows([0.0]),
    )
    assert not selected_support.item()
    assert logprob.item() == -math.inf
    logprob.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_identity_row_metadata_collapses_to_no_filter_switch():
    values = normalize_exact_sampling_transforms(
        torch.full((2,), TOP_K_ALL, dtype=torch.int64),
        torch.ones(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        rows=2,
        device=torch.device("cpu"),
    )
    assert values == (None, None, None)


@pytest.mark.parametrize(
    "module_name,function_name,extra",
    [
        ("xorl.ops.loss.policy_loss", "policy_loss_function", {"compute_kl_stats": True}),
        (
            "xorl.ops.loss.importance_sampling_loss",
            "importance_sampling_loss_function",
            {"compute_kl_stats": True},
        ),
        ("xorl.ops.loss.cispo_loss", "cispo_loss_function", {"compute_kl_stats": True}),
    ],
)
def test_rl_surrogates_are_finite_and_zero_gradient_outside_current_support(
    monkeypatch, module_name, function_name, extra
):
    module = importlib.import_module(module_name)
    ce = torch.tensor([math.inf, 0.4], dtype=torch.float32, requires_grad=True)
    monkeypatch.setattr(module, "compute_per_token_ce", lambda *args, **kwargs: ce)
    function = getattr(module, function_name)
    output = function(
        hidden_states=torch.zeros((1, 2, 3)),
        weight=torch.zeros((5, 3)),
        labels=torch.tensor([[2, 1]]),
        old_logprobs=torch.tensor([[-0.3, -0.4]]),
        advantages=torch.ones((1, 2)),
        ce_mode="eager",
        **extra,
    )
    assert torch.isfinite(output.loss)
    assert output.per_token_logprobs[0, 0].item() == -math.inf
    for value in output.metrics.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
    output.loss.backward()
    assert ce.grad[0].item() == 0.0
    assert torch.isfinite(ce.grad[1])


@pytest.mark.parametrize("ratio_type", ["token", "sequence"])
def test_drgrpo_is_finite_and_zero_gradient_outside_current_support(monkeypatch, ratio_type):
    module = importlib.import_module("xorl.ops.loss.grpo_loss")
    ce = torch.tensor([math.inf, 0.4], dtype=torch.float32, requires_grad=True)
    monkeypatch.setattr(module, "compute_per_token_ce", lambda *args, **kwargs: ce)
    output = module.drgrpo_loss_function(
        hidden_states=torch.zeros((1, 2, 3)),
        weight=torch.zeros((5, 3)),
        labels=torch.tensor([[2, 1]]),
        old_logprobs=torch.tensor([[-0.3, -0.4]]),
        advantages=torch.ones((1, 2)),
        beta=0.0,
        ratio_type=ratio_type,
        ce_mode="eager",
    )
    assert torch.isfinite(output.loss)
    assert output.per_token_logprobs[0, 0].item() == -math.inf
    assert all(torch.isfinite(value).all() for value in output.metrics.values())
    output.loss.backward()
    assert ce.grad[0].item() == 0.0
    assert torch.isfinite(ce.grad[1])
