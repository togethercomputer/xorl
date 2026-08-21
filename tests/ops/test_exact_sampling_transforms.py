import importlib
import inspect
import math

import pytest
import torch

from xorl.ops.exact.sampling_transforms import (
    EXACT_FILTER_ROW_CHUNK,
    EXACT_SAMPLING_TRANSFORM_PROGRAM,
    TOP_K_ALL,
    exact_sampling_support,
    exact_selected_logprob,
    exact_selected_logprob_chunked,
    exact_support_workspace_bytes,
    normalize_exact_sampling_transforms,
)
from xorl.ops.loss.bi_fused_lm_head import _score_exact_sampling_rows


def _rows(values):
    return torch.tensor(values, dtype=torch.float32)


def _independent_support_reference(raw_logits, temperatures, top_ks, top_ps, min_ps):
    expected = torch.zeros_like(raw_logits, dtype=torch.bool)
    scaled_rows = []
    for row in range(raw_logits.shape[0]):
        scaled = [float(value) / float(temperatures[row]) for value in raw_logits[row]]
        scaled_rows.append(scaled)
        row_max = max(scaled)
        weights = [math.exp(value - row_max) for value in scaled]
        denominator = sum(weights)
        probabilities = [value / denominator for value in weights]
        ordered = sorted(
            range(len(probabilities)),
            key=lambda token: (-probabilities[token], token),
        )
        cumulative_before = 0.0
        original_max = probabilities[ordered[0]]
        for rank, token in enumerate(ordered):
            probability = probabilities[token]
            expected[row, token] = (
                rank < min(int(top_ks[row]), len(probabilities))
                and cumulative_before <= float(top_ps[row])
                and probability >= original_max * float(min_ps[row])
            )
            cumulative_before += probability
    return torch.tensor(scaled_rows, dtype=raw_logits.dtype), expected


def test_temperature_then_joint_filters_match_independent_reference():
    raw_logits = _rows(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    temperatures = _rows([0.5, 2.0, 1.0])
    top_ks = torch.tensor([4, TOP_K_ALL, 3], dtype=torch.int64)
    top_ps = _rows([0.88, 0.70, 0.60])
    min_ps = _rows([0.05, 0.80, 0.0])
    score_logits, expected = _independent_support_reference(raw_logits, temperatures, top_ks, top_ps, min_ps)

    support = exact_sampling_support(score_logits, top_ks, top_ps, min_ps)

    assert torch.equal(support, expected)
    assert EXACT_SAMPLING_TRANSFORM_PROGRAM.startswith("temperature_then_stable_token_id")


def test_equal_logits_use_token_id_as_stable_tie_break():
    logits = torch.zeros((1, 6), dtype=torch.float32)
    support = exact_sampling_support(
        logits,
        torch.tensor([3], dtype=torch.int64),
        _rows([1.0]),
        _rows([0.0]),
    )
    assert support.tolist() == [[True, True, True, False, False, False]]


def test_identity_top_p_one_is_full_support_despite_fp32_cumulative_overshoot():
    generator = torch.Generator().manual_seed(20260814)
    for _ in range(6):
        logits = torch.randn((4, 32768), generator=generator, dtype=torch.float32)
    row = logits[2:3]
    sorted_probs = torch.softmax(row, dim=-1).sort(dim=-1, descending=True, stable=True).values
    cumulative_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    assert cumulative_before[0, -1] > 1.0

    support = exact_sampling_support(
        row,
        torch.tensor([TOP_K_ALL], dtype=torch.int64),
        _rows([1.0]),
        _rows([0.0]),
    )
    assert support.all()


def test_trainer_identity_score_and_vjp_are_batch_composition_invariant():
    def native_score(logits, token_ids):
        selected = logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        # Deliberately distinguish the native reduction from the filtered one.
        lse = torch.logsumexp(logits, dim=-1) + 0.125
        return selected - lse, lse, selected

    identity_logits = _rows([[3.0, 1.0, -2.0]]).requires_grad_(True)
    identity_ids = torch.tensor([0], dtype=torch.int64)
    native_logprob, native_lse, _ = native_score(identity_logits, identity_ids)
    native_logprob.backward()
    native_grad = identity_logits.grad.detach().clone()

    mixed_logits = _rows([[3.0, 1.0, -2.0], [2.0, 0.0, -1.0]]).requires_grad_(True)
    mixed_logprob, mixed_lse, _ = _score_exact_sampling_rows(
        mixed_logits,
        torch.tensor([0, 0], dtype=torch.int64),
        torch.tensor([TOP_K_ALL, 1], dtype=torch.int64),
        _rows([1.0, 1.0]),
        _rows([0.0, 0.0]),
        native_score,
    )
    mixed_logprob[0].backward()

    assert torch.equal(mixed_logprob[0], native_logprob.detach()[0])
    assert torch.equal(mixed_lse[0], native_lse.detach()[0])
    assert torch.equal(mixed_logits.grad[0], native_grad[0])
    assert torch.equal(mixed_logits.grad[1], torch.zeros_like(mixed_logits.grad[1]))


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


def test_greedy_decision_logprob_replay_is_positive_zero_with_zero_gradient():
    logits = _rows([[3.0, 3.0, 1.0]]).requires_grad_(True)
    logprob, _, selected_support, support = exact_selected_logprob(
        logits,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        _rows([1.0]),
        _rows([0.0]),
    )

    assert support.tolist() == [[True, False, False]]
    assert selected_support.item()
    assert torch.equal(logprob, _rows([0.0]))
    assert not torch.signbit(logprob).any()

    logprob.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_chunked_selected_score_and_vjp_equal_direct_program():
    direct_logits = _rows(
        [
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 1.5, 0.5, -1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    ).requires_grad_(True)
    chunked_logits = direct_logits.detach().clone().requires_grad_(True)
    token_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    top_ks = torch.tensor([2, 3, 4], dtype=torch.int64)
    top_ps = _rows([0.9, 0.95, 1.0])
    min_ps = _rows([0.0, 0.1, 0.0])

    direct, direct_lse, direct_selected, _ = exact_selected_logprob(direct_logits, token_ids, top_ks, top_ps, min_ps)
    chunked, chunked_lse, chunked_selected = exact_selected_logprob_chunked(
        chunked_logits,
        token_ids,
        top_ks,
        top_ps,
        min_ps,
        row_chunk_size=1,
    )
    direct.sum().backward()
    chunked.sum().backward()

    assert torch.equal(chunked, direct)
    assert torch.equal(chunked_lse, direct_lse)
    assert torch.equal(chunked_selected, direct_selected)
    assert torch.equal(chunked_logits.grad, direct_logits.grad)


def test_filtered_exact_heads_do_not_save_dense_support_on_autograd_contexts():
    modules = [
        importlib.import_module("xorl.ops.loss.bi_fused_lm_head"),
        importlib.import_module("xorl.models.transformers.glm5.exact_lm_head_qlora"),
        importlib.import_module("xorl.models.transformers.deepseek_v4.exact_lm_head"),
    ]
    for module in modules:
        source = inspect.getsource(module)
        assert "ctx.sampling_support" not in source
        assert "support_chunks" not in source

    assert EXACT_FILTER_ROW_CHUNK == 32
    assert exact_support_workspace_bytes(154_880) == 9_912_320
    assert exact_support_workspace_bytes(154_880) < 10 * 1024 * 1024


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
        ("xorl.objectives.policy_loss", "policy_loss_function", {"compute_kl_stats": True}),
        (
            "xorl.objectives.importance_sampling_loss",
            "importance_sampling_loss_function",
            {"compute_kl_stats": True},
        ),
        ("xorl.objectives.cispo_loss", "cispo_loss_function", {"compute_kl_stats": True}),
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
    module = importlib.import_module("xorl.objectives.grpo_loss")
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
