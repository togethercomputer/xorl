"""Generic ce_modes score replay transforms through the shared chunked core.

These tests pin the acceptance behavior of issue #71/#60: a generic
(non-exact-family) model trains on rollouts sampled with top-k/top-p/min-p
and per-row temperature, gets logprobs on the filtered support, and never
silently scores against unfiltered support.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.exact_sampling_transforms import (
    TOP_K_ALL,
    exact_selected_logprob,
    normalize_temperature_rows,
)
from xorl.ops.loss.per_token_ce import compute_per_token_ce


pytestmark = pytest.mark.cpu

IGNORE = -100
GENERIC_CE_MODES = ("eager", "compiled", "quack_linear", "fused_quack")


def _case(seed: int = 0, rows: int = 40, hidden_size: int = 16, vocab: int = 24):
    """Rows deliberately exceed EXACT_FILTER_ROW_CHUNK so chunking is exercised."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn((rows, hidden_size), generator=generator, dtype=torch.float32)
    weight = torch.randn((vocab, hidden_size), generator=generator, dtype=torch.float32)
    labels = torch.randint(0, vocab, (rows,), generator=generator)
    labels[2] = IGNORE
    temperature = torch.linspace(0.7, 1.4, rows, dtype=torch.float32)
    top_ks = torch.randint(2, vocab, (rows,), generator=generator)
    top_ks[0] = TOP_K_ALL
    top_ps = torch.linspace(0.55, 1.0, rows, dtype=torch.float32)
    min_ps = torch.linspace(0.0, 0.2, rows, dtype=torch.float32)
    return hidden, weight, labels, temperature, top_ks, top_ps, min_ps


def _reference_filtered_ce(hidden, weight, labels, temperature, top_ks, top_ps, min_ps):
    """Dense full-graph reference through the pinned program."""

    logits = (hidden @ weight.t()).float() / temperature.unsqueeze(1)
    valid = labels != IGNORE
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    logprob, _, _, _ = exact_selected_logprob(logits, safe, top_ks, top_ps, min_ps)
    return torch.where(valid, -logprob, torch.zeros_like(logprob))


@pytest.mark.parametrize("ce_mode", GENERIC_CE_MODES)
def test_generic_modes_score_the_filtered_support(ce_mode):
    hidden, weight, labels, temperature, top_ks, top_ps, min_ps = _case()
    expected = _reference_filtered_ce(hidden, weight, labels, temperature, top_ks, top_ps, min_ps)

    ce = compute_per_token_ce(
        hidden,
        weight,
        labels,
        ignore_index=IGNORE,
        ce_mode=ce_mode,
        logprob_temperature=temperature,
        logprob_top_k=top_ks,
        logprob_top_p=top_ps,
        logprob_min_p=min_ps,
    )

    torch.testing.assert_close(ce, expected)
    assert (ce[labels == IGNORE] == 0).all()


def test_gradients_match_the_full_autograd_reference():
    hidden, weight, labels, temperature, top_ks, top_ps, min_ps = _case(seed=3)

    reference_hidden = hidden.clone().requires_grad_(True)
    reference_weight = weight.clone().requires_grad_(True)
    reference_ce = _reference_filtered_ce(
        reference_hidden, reference_weight, labels, temperature, top_ks, top_ps, min_ps
    )
    valid = labels != IGNORE
    reference_ce[valid].mean().backward()

    core_hidden = hidden.clone().requires_grad_(True)
    core_weight = weight.clone().requires_grad_(True)
    ce = compute_per_token_ce(
        core_hidden,
        core_weight,
        labels,
        ignore_index=IGNORE,
        ce_mode="eager",
        logprob_temperature=temperature,
        logprob_top_k=top_ks,
        logprob_top_p=top_ps,
        logprob_min_p=min_ps,
    )
    ce[valid].mean().backward()

    torch.testing.assert_close(core_hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(core_weight.grad, reference_weight.grad)


def test_replayed_token_outside_current_support_is_infinite_ce_with_zero_gradient():
    hidden = torch.zeros((1, 4), dtype=torch.float32).requires_grad_(True)
    weight = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).requires_grad_(True)
    hidden_live = hidden + 1.0  # non-zero logits so the argmax is token 0
    labels = torch.tensor([2], dtype=torch.int64)

    ce = compute_per_token_ce(
        hidden_live,
        weight,
        labels,
        ignore_index=IGNORE,
        ce_mode="eager",
        logprob_top_k=torch.tensor([1], dtype=torch.int64),
        logprob_top_p=torch.ones(1, dtype=torch.float32),
        logprob_min_p=torch.zeros(1, dtype=torch.float32),
    )

    assert ce.item() == math.inf
    ce.sum().backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))
    assert torch.equal(weight.grad, torch.zeros_like(weight))


def test_per_row_temperature_alone_matches_scaled_cross_entropy():
    hidden, weight, labels, temperature, _, _, _ = _case(seed=5)
    logits = (hidden @ weight.t()).float() / temperature.unsqueeze(1)
    expected = F.cross_entropy(logits, labels, reduction="none", ignore_index=IGNORE)
    expected = torch.where(labels != IGNORE, expected, torch.zeros_like(expected))

    for ce_mode in GENERIC_CE_MODES:
        ce = compute_per_token_ce(
            hidden,
            weight,
            labels,
            ignore_index=IGNORE,
            ce_mode=ce_mode,
            logprob_temperature=temperature,
        )
        torch.testing.assert_close(ce, expected)


def test_lm_head_fp32_convention_is_honored():
    hidden, weight, labels, temperature, top_ks, top_ps, min_ps = _case(seed=7)
    hidden = hidden.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16)

    ce = compute_per_token_ce(
        hidden,
        weight,
        labels,
        ignore_index=IGNORE,
        ce_mode="compiled",
        lm_head_fp32=True,
        logprob_temperature=temperature,
        logprob_top_k=top_ks,
        logprob_top_p=top_ps,
        logprob_min_p=min_ps,
    )
    expected = _reference_filtered_ce(hidden.float(), weight.float(), labels, temperature, top_ks, top_ps, min_ps)
    torch.testing.assert_close(ce, expected)


def test_scalar_identity_metadata_keeps_the_untransformed_path():
    hidden, weight, labels, _, _, _, _ = _case(seed=11)
    plain = compute_per_token_ce(hidden, weight, labels, ignore_index=IGNORE, ce_mode="eager")
    with_identity = compute_per_token_ce(
        hidden,
        weight,
        labels,
        ignore_index=IGNORE,
        ce_mode="eager",
        logprob_temperature=1.0,
        logprob_top_k=TOP_K_ALL,
        logprob_top_p=1.0,
        logprob_min_p=0.0,
    )
    assert torch.equal(plain, with_identity)


def test_identity_row_tensors_collapse_to_the_untransformed_path():
    hidden, weight, labels, _, _, _, _ = _case(seed=13)
    rows = hidden.shape[0]
    plain = compute_per_token_ce(hidden, weight, labels, ignore_index=IGNORE, ce_mode="eager")
    with_identity_rows = compute_per_token_ce(
        hidden,
        weight,
        labels,
        ignore_index=IGNORE,
        ce_mode="eager",
        logprob_top_k=torch.full((rows,), TOP_K_ALL, dtype=torch.int64),
        logprob_top_p=torch.ones(rows, dtype=torch.float32),
        logprob_min_p=torch.zeros(rows, dtype=torch.float32),
    )
    assert torch.equal(plain, with_identity_rows)


def test_fp8_lm_head_module_transform_requests_fail_loudly_in_one_place():
    hidden, weight, labels, _, top_ks, top_ps, min_ps = _case(seed=17)
    lm_head = torch.nn.Linear(hidden.shape[1], weight.shape[0], bias=False)
    with pytest.raises(NotImplementedError, match="FP8 lm_head module"):
        compute_per_token_ce(
            hidden,
            weight,
            labels,
            ignore_index=IGNORE,
            ce_mode="compiled",
            lm_head=lm_head,
            logprob_top_k=top_ks,
            logprob_top_p=top_ps,
            logprob_min_p=min_ps,
        )


def test_misaligned_transform_metadata_is_rejected_not_truncated():
    from xorl.ops.loss.sampling_transform_ce import sampling_transform_per_token_ce

    hidden, weight, labels, _, _, _, _ = _case(seed=19, rows=10)
    over_length = 15
    with pytest.raises(ValueError, match="logprob_top_ks"):
        sampling_transform_per_token_ce(
            hidden,
            weight,
            labels,
            ignore_index=IGNORE,
            temperature_rows=None,
            top_ks=torch.full((over_length,), 3, dtype=torch.int64),
            top_ps=torch.ones(over_length, dtype=torch.float32),
            min_ps=torch.zeros(over_length, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="all or none"):
        sampling_transform_per_token_ce(
            hidden,
            weight,
            labels,
            ignore_index=IGNORE,
            temperature_rows=None,
            top_ks=torch.full((10,), 3, dtype=torch.int64),
            top_ps=None,
            min_ps=None,
        )


def test_normalize_temperature_rows_contract():
    device = torch.device("cpu")
    assert normalize_temperature_rows(1.0, rows=3, device=device) is None
    rows = normalize_temperature_rows(0.7, rows=3, device=device)
    assert torch.equal(rows, torch.full((3,), 0.7, dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        normalize_temperature_rows(0.0, rows=3, device=device)
    with pytest.raises(TypeError, match="FP32"):
        normalize_temperature_rows(torch.ones(3, dtype=torch.float64), rows=3, device=device)
    with pytest.raises(ValueError, match="row-aligned"):
        normalize_temperature_rows(torch.ones(2, dtype=torch.float32), rows=3, device=device)
