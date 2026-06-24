"""Regression test for ``MoEBlock._regather_routing`` sqrtsoftplus dispatch.

DSv4 / GLM-style MoE uses ``scoring_func="sqrtsoftplus"`` (with optional
``noaux_tc`` selection-time bias and a ``routed_scaling_factor``). The
routing-replay regather path must recover the same per-token routing
weights as the eager router, otherwise gradient-checkpoint recompute
silently diverges from the forward pass.
"""

import pytest
import torch
import torch.nn.functional as F

from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.moe.router import TopKRouter


pytestmark = pytest.mark.cpu


def _make_block(scoring_func: str, routed_scaling_factor=None, num_experts=8, top_k=2):
    """Build a minimal CPU MoEBlock with a chosen TopKRouter scoring func."""
    block = MoEBlock(
        hidden_size=16,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=16,
        moe_implementation="eager",
    )
    block.router = TopKRouter(
        num_experts=num_experts,
        top_k=top_k,
        norm_topk_prob=True,
        scoring_func=scoring_func,
        topk_method="noaux_tc" if scoring_func == "sqrtsoftplus" else None,
        routed_scaling_factor=routed_scaling_factor,
    )
    return block


def test_sqrtsoftplus_regather_matches_eager_router():
    """Regather from cached top-k must equal the V4 router's eager output."""
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 6, 8, 2
    block = _make_block("sqrtsoftplus", num_experts=num_experts, top_k=top_k)

    router_logits = torch.randn(num_tokens, num_experts)
    bias = torch.randn(num_experts) * 0.1

    eager_w, eager_idx = block.router(router_logits, input_dtype=torch.float32, expert_bias=bias)

    regather_idx, regather_w = block._regather_routing(router_logits, eager_idx, input_dtype=torch.float32)

    assert torch.equal(regather_idx, eager_idx)
    torch.testing.assert_close(regather_w, eager_w, rtol=1e-5, atol=1e-6)


def test_sqrtsoftplus_regather_applies_routed_scaling_factor():
    """Regather must apply ``routed_scaling_factor`` (DSv4 sets it to 1.0+)."""
    torch.manual_seed(1)
    num_tokens, num_experts, top_k = 4, 8, 2
    scaling = 2.5
    block = _make_block("sqrtsoftplus", routed_scaling_factor=scaling, num_experts=num_experts, top_k=top_k)

    router_logits = torch.randn(num_tokens, num_experts)
    bias = torch.zeros(num_experts)
    eager_w, eager_idx = block.router(router_logits, input_dtype=torch.float32, expert_bias=bias)
    _, regather_w = block._regather_routing(router_logits, eager_idx, input_dtype=torch.float32)

    torch.testing.assert_close(regather_w, eager_w, rtol=1e-5, atol=1e-6)
    # Cross-check against an unscaled regather to make sure scaling is applied.
    block.router.routed_scaling_factor = None
    _, unscaled_w = block._regather_routing(router_logits, eager_idx, input_dtype=torch.float32)
    torch.testing.assert_close(regather_w, unscaled_w * scaling, rtol=1e-5, atol=1e-6)


def test_sqrtsoftplus_regather_preserves_input_dtype():
    """Regathered weights must come back in the requested input dtype."""
    torch.manual_seed(2)
    block = _make_block("sqrtsoftplus")
    router_logits = torch.randn(4, 8, dtype=torch.float32)
    cached = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    _, w = block._regather_routing(router_logits, cached, input_dtype=torch.bfloat16)
    assert w.dtype == torch.bfloat16


def test_softmax_regather_unchanged():
    """The softmax path must keep its existing semantics — we only added a
    sqrtsoftplus branch above it."""
    torch.manual_seed(3)
    num_tokens, num_experts, top_k = 5, 8, 2
    block = _make_block("softmax", num_experts=num_experts, top_k=top_k)

    router_logits = torch.randn(num_tokens, num_experts)
    eager_w, eager_idx = block.router(router_logits, input_dtype=torch.float32)
    _, regather_w = block._regather_routing(router_logits, eager_idx, input_dtype=torch.float32)

    # Hand-roll the expected: softmax → gather → renorm.
    probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
    expected = torch.gather(probs, 1, eager_idx)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(regather_w, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(regather_w, eager_w, rtol=1e-5, atol=1e-6)
