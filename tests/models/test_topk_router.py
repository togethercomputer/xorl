"""Unit tests for ``xorl.models.layers.moe.router.TopKRouter``.

Covers:

* Legacy ``softmax`` path is byte-identical to the previous implementation
  (regression — no behavior change for non-V4 callers).
* DeepSeek-V4 ``sqrtsoftplus`` + ``noaux_tc`` selects experts via
  ``scores + bias`` but gathers weights from the unbiased scores.
* Hash routing via ``tid2eid`` overrides selection without touching the
  gate; routing weights still flow through ``sqrt(softplus(logits))``.
* ``routed_scaling_factor`` multiplies the post-renorm weights.
"""

import pytest
import torch
import torch.nn.functional as F

from xorl.models.layers.moe.router import TopKRouter


pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Regression: legacy softmax path
# ---------------------------------------------------------------------------


def test_softmax_path_matches_reference():
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 5, 8, 2
    logits = torch.randn(num_tokens, num_experts)

    router = TopKRouter(num_experts=num_experts, top_k=top_k, norm_topk_prob=True)
    weights, experts = router(logits, input_dtype=torch.float32)

    ref_probs = F.softmax(logits, dim=1, dtype=torch.float)
    ref_weights, ref_experts = torch.topk(ref_probs, top_k, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

    assert torch.equal(experts, ref_experts)
    torch.testing.assert_close(weights, ref_weights)


def test_softmax_path_no_renorm():
    torch.manual_seed(1)
    logits = torch.randn(3, 6)
    router = TopKRouter(num_experts=6, top_k=2, norm_topk_prob=False)
    weights, _ = router(logits, input_dtype=torch.float32)
    # When renorm is off, weights should NOT sum to 1
    sums = weights.sum(dim=-1)
    assert (sums - 1).abs().max() > 1e-3


def test_softmax_path_invariant_to_v4_kwargs():
    """Passing V4-only kwargs to a softmax router has no effect."""
    torch.manual_seed(2)
    logits = torch.randn(4, 8)
    router = TopKRouter(num_experts=8, top_k=2)
    w_a, e_a = router(logits, input_dtype=torch.float32)
    w_b, e_b = router(
        logits,
        input_dtype=torch.float32,
        expert_bias=torch.randn(8),  # ignored
        tid2eid=torch.randint(0, 8, (100, 2)),  # ignored
        input_ids=torch.zeros(4, dtype=torch.long),  # ignored
    )
    assert torch.equal(e_a, e_b)
    torch.testing.assert_close(w_a, w_b)


def test_synthetic_balanced_routing_overrides_softmax_selection(monkeypatch):
    monkeypatch.setenv("XORL_MOE_SYNTHETIC_ROUTING", "balanced")
    torch.manual_seed(7)
    num_tokens, num_experts, top_k = 7, 8, 2
    logits = torch.randn(num_tokens, num_experts)

    router = TopKRouter(num_experts=num_experts, top_k=top_k, norm_topk_prob=True)
    weights, experts = router(logits, input_dtype=torch.float32)

    # Synthetic balanced mode cycles experts as (t*top_k + [0..top_k-1]) % num_experts
    # and returns uniform routing weights (1/top_k); see balanced_synthetic_routing.
    expected_experts = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1], [2, 3], [4, 5]],
        dtype=torch.long,
    )
    assert torch.equal(experts, expected_experts)
    counts = torch.bincount(experts.flatten(), minlength=num_experts)
    assert int(counts.max() - counts.min()) <= 1

    expected_weights = torch.full((num_tokens, top_k), 1.0 / top_k)
    torch.testing.assert_close(weights, expected_weights)


# ---------------------------------------------------------------------------
# DSv4 sqrtsoftplus + noaux_tc
# ---------------------------------------------------------------------------


def test_sqrtsoftplus_noaux_selects_via_biased_scores():
    """Selection comes from ``scores + bias``, weights from unbiased scores."""
    torch.manual_seed(3)
    num_tokens, num_experts, top_k = 4, 6, 2
    logits = torch.randn(num_tokens, num_experts)
    # Heavily bias expert 0 so it always wins; expert 5 normally wins.
    bias = torch.zeros(num_experts)
    bias[0] = 100.0

    router = TopKRouter(
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
    )
    weights, experts = router(logits, input_dtype=torch.float32, expert_bias=bias)

    # Expert 0 should be selected for every token (bias dominates).
    assert (experts == 0).any(dim=-1).all(), f"expert 0 not selected everywhere: {experts}"

    # Weights should match unbiased ``sqrt(softplus(logits))``, gathered + renormed.
    unbiased = F.softplus(logits.float()).sqrt().type_as(logits)
    expected = torch.gather(unbiased, dim=1, index=experts)
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(weights, expected)


def test_sqrtsoftplus_noaux_requires_bias():
    router = TopKRouter(num_experts=4, top_k=2, scoring_func="sqrtsoftplus", topk_method="noaux_tc")
    with pytest.raises(AssertionError, match="noaux_tc requires expert_bias"):
        router(torch.randn(3, 4), input_dtype=torch.float32)


# ---------------------------------------------------------------------------
# DSv4 hash routing via tid2eid
# ---------------------------------------------------------------------------


def test_hash_routing_uses_tid2eid_for_selection():
    """Top-k indices come from ``tid2eid[input_ids]``; weights from gate."""
    torch.manual_seed(4)
    vocab_size, num_experts, top_k = 16, 8, 2
    logits = torch.randn(5, num_experts)
    # Frozen lookup: token id i -> experts (i % E, (i + 1) % E).
    table = torch.stack(
        [
            torch.arange(vocab_size) % num_experts,
            (torch.arange(vocab_size) + 1) % num_experts,
        ],
        dim=1,
    ).to(torch.int32)
    input_ids = torch.tensor([0, 3, 7, 9, 2], dtype=torch.long)

    router = TopKRouter(
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sqrtsoftplus",
    )
    weights, experts = router(logits, input_dtype=torch.float32, tid2eid=table, input_ids=input_ids)

    expected_experts = table[input_ids].to(torch.long)
    assert torch.equal(experts, expected_experts)

    # Weights come from gather(unbiased_scores, expected_experts), renormed.
    unbiased = F.softplus(logits.float()).sqrt().type_as(logits)
    expected_w = torch.gather(unbiased, dim=1, index=expected_experts)
    expected_w = expected_w / (expected_w.sum(dim=-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(weights, expected_w)


def test_hash_routing_ignores_bias():
    """When tid2eid is set, bias does not affect anything."""
    torch.manual_seed(5)
    vocab_size, num_experts, top_k = 16, 8, 2
    logits = torch.randn(3, num_experts)
    table = torch.zeros(vocab_size, top_k, dtype=torch.int32)  # all to expert 0
    input_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    router = TopKRouter(num_experts=num_experts, top_k=top_k, scoring_func="sqrtsoftplus")
    _, experts_no_bias = router(logits, input_dtype=torch.float32, tid2eid=table, input_ids=input_ids)
    big_bias = torch.full((num_experts,), 1000.0)
    big_bias[0] = -1000.0
    _, experts_with_bias = router(
        logits,
        input_dtype=torch.float32,
        tid2eid=table,
        input_ids=input_ids,
        expert_bias=big_bias,
    )
    assert torch.equal(experts_no_bias, experts_with_bias)
    assert (experts_no_bias == 0).all()


def test_hash_routing_requires_input_ids():
    table = torch.zeros(8, 2, dtype=torch.int32)
    router = TopKRouter(num_experts=4, top_k=2, scoring_func="sqrtsoftplus")
    with pytest.raises(AssertionError, match="requires input_ids"):
        router(torch.randn(3, 4), input_dtype=torch.float32, tid2eid=table)


def test_synthetic_balanced_routing_overrides_hash_and_bias(monkeypatch):
    monkeypatch.setenv("XORL_MOE_SYNTHETIC_ROUTING", "balanced")
    torch.manual_seed(8)
    num_tokens, vocab_size, num_experts, top_k = 6, 16, 8, 2
    logits = torch.randn(num_tokens, num_experts)
    table = torch.zeros(vocab_size, top_k, dtype=torch.int32)
    input_ids = torch.arange(num_tokens, dtype=torch.long)
    bias = torch.full((num_experts,), 1000.0)

    router = TopKRouter(
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
    )
    weights, experts = router(
        logits,
        input_dtype=torch.float32,
        expert_bias=bias,
        tid2eid=table,
        input_ids=input_ids,
    )

    expected_experts = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1], [2, 3]],
        dtype=torch.long,
    )
    assert torch.equal(experts, expected_experts)

    scores = F.softplus(logits.float()).sqrt().type_as(logits)
    expected_weights = torch.gather(scores, dim=1, index=expected_experts)
    expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(weights, expected_weights)


# ---------------------------------------------------------------------------
# routed_scaling_factor
# ---------------------------------------------------------------------------


def test_routed_scaling_factor_multiplies_weights():
    torch.manual_seed(6)
    logits = torch.randn(3, 6)
    bias = torch.zeros(6)
    base = TopKRouter(num_experts=6, top_k=2, scoring_func="sqrtsoftplus", topk_method="noaux_tc")
    scaled = TopKRouter(
        num_experts=6,
        top_k=2,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
    )
    w_base, _ = base(logits, input_dtype=torch.float32, expert_bias=bias)
    w_scaled, _ = scaled(logits, input_dtype=torch.float32, expert_bias=bias)
    torch.testing.assert_close(w_scaled, w_base * 1.5)


def test_routed_scaling_factor_rejected_on_softmax_path():
    """``routed_scaling_factor`` is V4-only; constructing a softmax router with
    one set raises rather than silently ignoring it.
    """
    with pytest.raises(ValueError, match="routed_scaling_factor is only used"):
        TopKRouter(num_experts=6, top_k=2, routed_scaling_factor=1.5)


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_reads_v4_fields():
    """from_config picks up sqrtsoftplus + noaux_tc + scaling_factor from a V4 config."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config  # noqa: PLC0415

    cfg = DeepseekV4Config()
    router = TopKRouter.from_config(cfg)
    assert router.scoring_func == "sqrtsoftplus"
    assert router.topk_method == "noaux_tc"
    assert router.routed_scaling_factor == cfg.routed_scaling_factor
    assert router.top_k == cfg.num_experts_per_tok
    assert router.num_experts == cfg.n_routed_experts


def test_from_config_legacy_softmax():
    """Legacy non-V4 configs land on the softmax path."""

    class _LegacyCfg:
        num_experts = 16
        num_experts_per_tok = 2
        norm_topk_prob = True

    router = TopKRouter.from_config(_LegacyCfg())
    assert router.scoring_func == "softmax"
    assert router.topk_method is None
    assert router.routed_scaling_factor is None
