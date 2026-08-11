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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xorl.models.layers.moe import moe_block as moe_block_module
from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.moe.router import TopKRouter


pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Regression: legacy softmax path
# ---------------------------------------------------------------------------


def _assert_softmax_policy_matches_reference_and_ignores_v4_inputs():
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

    # Disabling normalization preserves the selected probability mass.
    torch.manual_seed(1)
    unnormalized_logits = torch.randn(3, 6)
    unnormalized = TopKRouter(num_experts=6, top_k=2, norm_topk_prob=False)
    unnormalized_weights, _ = unnormalized(unnormalized_logits, input_dtype=torch.float32)
    sums = unnormalized_weights.sum(dim=-1)
    assert (sums - 1).abs().max() > 1e-3

    # V4-only inputs are inert on the legacy softmax path.
    torch.manual_seed(2)
    legacy_logits = torch.randn(4, 8)
    legacy = TopKRouter(num_experts=8, top_k=2)
    w_a, e_a = legacy(legacy_logits, input_dtype=torch.float32)
    w_b, e_b = legacy(
        legacy_logits,
        input_dtype=torch.float32,
        expert_bias=torch.randn(8),  # ignored
        tid2eid=torch.randint(0, 8, (100, 2)),  # ignored
        input_ids=torch.zeros(4, dtype=torch.long),  # ignored
    )
    assert torch.equal(e_a, e_b)
    torch.testing.assert_close(w_a, w_b)


def _assert_synthetic_balanced_routing_overrides_softmax_hash_and_bias(monkeypatch):
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

    table = torch.zeros(16, top_k, dtype=torch.int32)
    input_ids = torch.arange(num_tokens, dtype=torch.long)
    bias = torch.full((num_experts,), 1000.0)
    v4_router = TopKRouter(
        num_experts=num_experts,
        top_k=top_k,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
    )
    v4_weights, v4_experts = v4_router(
        logits,
        input_dtype=torch.float32,
        expert_bias=bias,
        tid2eid=table,
        input_ids=input_ids,
    )
    assert torch.equal(v4_experts, expected_experts)
    scores = F.softplus(logits.float()).sqrt().type_as(logits)
    expected_v4_weights = torch.gather(scores, dim=1, index=expected_experts)
    expected_v4_weights = expected_v4_weights / (expected_v4_weights.sum(dim=-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(v4_weights, expected_v4_weights)


def _assert_moe_block_uses_configured_router_fp32(monkeypatch):
    class RecordingGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
            self.called = False

        def forward(self, hidden_states):
            self.called = True
            return torch.zeros(hidden_states.shape[0], 2, dtype=hidden_states.dtype)

    block = MoEBlock(
        hidden_size=2,
        num_experts=2,
        top_k=1,
        intermediate_size=4,
        moe_implementation="eager",
        train_router=False,
    )
    block.config = SimpleNamespace(_router_fp32=True)
    block.gate = RecordingGate()
    calls = []

    def fake_linear(hidden_states, weight, bias=None):
        calls.append((hidden_states.dtype, weight.dtype, bias))
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(moe_block_module.F, "linear", fake_linear)

    _, selected_experts, router_logits = block.route(torch.ones(2, 2, dtype=torch.bfloat16))

    assert calls == [(torch.float32, torch.float32, None)]
    assert not block.gate.called
    assert router_logits.dtype == torch.float32
    assert torch.equal(selected_experts, torch.tensor([[1], [0]]))


def _assert_trainable_router_rejects_deepep_dispatch():
    block = MoEBlock(
        hidden_size=16,
        num_experts=4,
        top_k=2,
        intermediate_size=32,
        moe_implementation="eager",
        train_router=True,
    )
    block.experts.ep_dispatch = "deepep"

    with pytest.raises(AssertionError, match="ep_dispatch='deepep'"):
        block(torch.randn(1, 4, 16))


# ---------------------------------------------------------------------------
# DSv4 sqrtsoftplus + noaux_tc
# ---------------------------------------------------------------------------


def _assert_sqrtsoftplus_noaux_uses_unbiased_weights_and_requires_bias():
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

    missing_bias = TopKRouter(num_experts=4, top_k=2, scoring_func="sqrtsoftplus", topk_method="noaux_tc")
    with pytest.raises(AssertionError, match="noaux_tc requires expert_bias"):
        missing_bias(torch.randn(3, 4), input_dtype=torch.float32)


# ---------------------------------------------------------------------------
# DSv4 hash routing via tid2eid
# ---------------------------------------------------------------------------


def _assert_hash_routing_uses_tid2eid_ignores_bias_and_requires_input_ids():
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

    big_bias = torch.full((num_experts,), 1000.0)
    big_bias[0] = -1000.0
    weights_with_bias, experts_with_bias = router(
        logits,
        input_dtype=torch.float32,
        tid2eid=table,
        input_ids=input_ids,
        expert_bias=big_bias,
    )
    assert torch.equal(experts, experts_with_bias)
    torch.testing.assert_close(weights, weights_with_bias)

    with pytest.raises(AssertionError, match="requires input_ids"):
        router(logits, input_dtype=torch.float32, tid2eid=table)


# ---------------------------------------------------------------------------
# routed_scaling_factor
# ---------------------------------------------------------------------------


def _assert_routed_scaling_factor_multiplies_v4_and_rejects_softmax():
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

    with pytest.raises(ValueError, match="routed_scaling_factor is only used"):
        TopKRouter(num_experts=6, top_k=2, routed_scaling_factor=1.5)


def _assert_moe_block_regather_matches_router_policy():
    def make_block(scoring_func: str, routed_scaling_factor=None):
        block = MoEBlock(
            hidden_size=16,
            num_experts=8,
            top_k=2,
            intermediate_size=16,
            moe_implementation="eager",
        )
        block.router = TopKRouter(
            num_experts=8,
            top_k=2,
            norm_topk_prob=True,
            scoring_func=scoring_func,
            topk_method="noaux_tc" if scoring_func == "sqrtsoftplus" else None,
            routed_scaling_factor=routed_scaling_factor,
        )
        return block

    torch.manual_seed(0)
    block = make_block("sqrtsoftplus", routed_scaling_factor=2.5)
    router_logits = torch.randn(6, 8)
    eager_weights, eager_experts = block.router(
        router_logits,
        input_dtype=torch.float32,
        expert_bias=torch.randn(8) * 0.1,
    )

    regathered_experts, regathered_weights = block._regather_routing(
        router_logits,
        eager_experts,
        input_dtype=torch.float32,
    )

    assert torch.equal(regathered_experts, eager_experts)
    torch.testing.assert_close(regathered_weights, eager_weights, rtol=1e-5, atol=1e-6)

    block.router.routed_scaling_factor = None
    _, unscaled_weights = block._regather_routing(router_logits, eager_experts, input_dtype=torch.float32)
    torch.testing.assert_close(regathered_weights, unscaled_weights * 2.5, rtol=1e-5, atol=1e-6)
    _, bf16_weights = block._regather_routing(router_logits, eager_experts, input_dtype=torch.bfloat16)
    assert bf16_weights.dtype is torch.bfloat16

    torch.manual_seed(3)
    block = make_block("softmax")
    router_logits = torch.randn(5, 8)
    eager_weights, eager_experts = block.router(router_logits, input_dtype=torch.float32)
    _, regathered_weights = block._regather_routing(router_logits, eager_experts, input_dtype=torch.float32)
    torch.testing.assert_close(regathered_weights, eager_weights, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def _assert_from_config_selects_v4_and_legacy_policies():
    """from_config picks up sqrtsoftplus + noaux_tc + scaling_factor from a V4 config."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config  # noqa: PLC0415

    cfg = DeepseekV4Config()
    router = TopKRouter.from_config(cfg)
    assert router.scoring_func == "sqrtsoftplus"
    assert router.topk_method == "noaux_tc"
    assert router.routed_scaling_factor == cfg.routed_scaling_factor
    assert router.top_k == cfg.num_experts_per_tok
    assert router.num_experts == cfg.n_routed_experts

    class _LegacyCfg:
        num_experts = 16
        num_experts_per_tok = 2
        norm_topk_prob = True

    legacy = TopKRouter.from_config(_LegacyCfg())
    assert legacy.scoring_func == "softmax"
    assert legacy.topk_method is None
    assert legacy.routed_scaling_factor is None


def test_topk_router_legacy_and_configuration_contract(monkeypatch):
    with monkeypatch.context() as case_patch:
        _assert_synthetic_balanced_routing_overrides_softmax_hash_and_bias(case_patch)
    _assert_sqrtsoftplus_noaux_uses_unbiased_weights_and_requires_bias()
    _assert_hash_routing_uses_tid2eid_ignores_bias_and_requires_input_ids()
    _assert_softmax_policy_matches_reference_and_ignores_v4_inputs()
    _assert_routed_scaling_factor_multiplies_v4_and_rejects_softmax()
    _assert_moe_block_regather_matches_router_policy()
    _assert_from_config_selects_v4_and_legacy_policies()
    _assert_moe_block_uses_configured_router_fp32(monkeypatch)
    _assert_trainable_router_rejects_deepep_dispatch()
