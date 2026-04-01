"""Tests for MoE routing cache with gradient checkpointing.

Validates:
1. context_fn mechanism: checkpoint's context_fn correctly distinguishes
   forward from recompute so the routing cache can store/replay.
2. Deque-based cache works with PP 1F1B scheduling.
3. End-to-end correctness with real MoEBlock.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


pytestmark = [pytest.mark.gpu]

try:
    from xorl.models.layers.moe.moe_block import (
        MoEBlock,
        _routing_cache_mode,
        moe_routing_context_fn,
    )
    from xorl.models.layers.moe.router import TopKRouter
except ImportError:
    pytest.skip(
        "moe_routing_context_fn / _routing_cache_mode not yet implemented",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkpoint_with_routing(fn, *args, **kwargs):
    """Checkpoint wrapper that uses moe_routing_context_fn."""
    return checkpoint(
        fn,
        *args,
        use_reentrant=False,
        context_fn=moe_routing_context_fn,
        **kwargs,
    )


class _NoisyLinear(nn.Module):
    """Linear + small random noise to simulate flash attention non-determinism."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x) + torch.randn_like(x) * 1e-4


class _SimpleDecoderLayer(nn.Module):
    """Decoder layer stub: noisy attention + MoE (eager backend)."""

    def __init__(self, hidden_size=64, num_experts=4, top_k=2, ffn_dim=128):
        super().__init__()
        self.attn = _NoisyLinear(hidden_size)
        self.mlp = MoEBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=ffn_dim,
            moe_implementation="eager",
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(hidden_states)
        moe_out, _ = self.mlp(hidden_states)
        return hidden_states + moe_out


# ===========================================================================
# 1. context_fn mechanism
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestContextFnMechanism:
    """Verify that context_fn correctly sets the routing cache mode."""

    def test_forward_sets_mode(self):
        """During checkpoint forward, mode is 'forward'."""
        import xorl.models.layers.moe.moe_block as mb

        observed = []

        class _Observer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def forward(self, x):
                observed.append(mb._routing_cache_mode)
                return self.linear(x)

        model = _Observer().cuda()
        x = torch.randn(2, 32, device="cuda", requires_grad=True)
        out = _checkpoint_with_routing(model, x)
        out.sum().backward()

        assert len(observed) == 2
        assert observed[0] == "forward"
        assert observed[1] == "recompute"

    def test_mode_none_without_context_fn(self):
        """Without context_fn, mode stays None — no caching."""
        import xorl.models.layers.moe.moe_block as mb

        observed = []

        class _Observer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def forward(self, x):
                observed.append(mb._routing_cache_mode)
                return self.linear(x)

        model = _Observer().cuda()
        x = torch.randn(2, 32, device="cuda", requires_grad=True)
        out = checkpoint(model, x, use_reentrant=False)
        out.sum().backward()

        assert len(observed) == 2
        assert observed[0] is None
        assert observed[1] is None

    def test_no_caching_in_eval(self):
        """In eval mode with context_fn, cache is not populated."""
        layer = _SimpleDecoderLayer().cuda()
        layer.eval()

        with torch.no_grad():
            layer(torch.randn(1, 8, 64, device="cuda"))

        assert len(layer.mlp._routing_cache) == 0

    def test_no_caching_without_context_fn(self):
        """Without context_fn, training forward doesn't cache routing."""
        moe = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
        ).cuda()
        moe.train()

        x = torch.randn(1, 8, 64, device="cuda")
        moe(x)
        assert len(moe._routing_cache) == 0


# ===========================================================================
# 2. Deque cache with PP scheduling
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDequeCachePP:
    """Verify deque-based routing cache works correctly with PP 1F1B."""

    def test_simple_forward_backward(self):
        """Single micro-batch: cache + reuse works."""
        layer = _SimpleDecoderLayer().cuda()
        layer.train()
        moe = layer.mlp

        x = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        out = _checkpoint_with_routing(layer, x)

        assert len(moe._routing_cache) == 1

        out.sum().backward()
        assert len(moe._routing_cache) == 0, "Cache consumed by recompute"

    def test_pp_two_forwards_two_backwards(self):
        """PP 1F1B: fwd MB1, fwd MB2, bwd MB1, bwd MB2."""
        layer = _SimpleDecoderLayer().cuda()
        layer.train()
        moe = layer.mlp

        x1 = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        x2 = torch.randn(1, 12, 64, device="cuda", requires_grad=True)

        out1 = _checkpoint_with_routing(layer, x1)
        out2 = _checkpoint_with_routing(layer, x2)

        assert len(moe._routing_cache) == 2

        # Backward MB1 (popleft gets MB1's routing)
        out1.sum().backward()
        assert len(moe._routing_cache) == 1, "MB1 consumed, MB2 remains"

        # Backward MB2 (popleft gets MB2's routing)
        out2.sum().backward()
        assert len(moe._routing_cache) == 0, "All consumed"

    def test_pp_four_microbatches_1f1b(self):
        """PP 1F1B with 4 micro-batches: warmup(2 fwd), steady(bwd+fwd), cooldown."""
        layer = _SimpleDecoderLayer().cuda()
        layer.train()
        moe = layer.mlp

        xs = [torch.randn(1, 8 + i, 64, device="cuda", requires_grad=True) for i in range(4)]
        outs = []

        # Warmup: 2 forwards
        outs.append(_checkpoint_with_routing(layer, xs[0]))
        outs.append(_checkpoint_with_routing(layer, xs[1]))
        assert len(moe._routing_cache) == 2

        # Steady state: bwd + fwd alternating
        outs[0].sum().backward()
        assert len(moe._routing_cache) == 1
        outs.append(_checkpoint_with_routing(layer, xs[2]))
        assert len(moe._routing_cache) == 2

        outs[1].sum().backward()
        assert len(moe._routing_cache) == 1
        outs.append(_checkpoint_with_routing(layer, xs[3]))
        assert len(moe._routing_cache) == 2

        # Cooldown
        outs[2].sum().backward()
        assert len(moe._routing_cache) == 1
        outs[3].sum().backward()
        assert len(moe._routing_cache) == 0

    def test_multi_layer_pp(self):
        """Multiple checkpointed decoder layers with PP scheduling."""
        hidden_size = 64

        class _Model(nn.Module):
            def __init__(self, num_layers=3):
                super().__init__()
                self.layers = nn.ModuleList([_SimpleDecoderLayer(hidden_size) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = _checkpoint_with_routing(layer, x)
                return x

        model = _Model(num_layers=3).cuda()
        model.train()

        x1 = torch.randn(1, 8, hidden_size, device="cuda", requires_grad=True)
        x2 = torch.randn(1, 10, hidden_size, device="cuda", requires_grad=True)

        # PP: two forwards
        out1 = model(x1)
        out2 = model(x2)

        for i, layer in enumerate(model.layers):
            assert len(layer.mlp._routing_cache) == 2, f"Layer {i}: expected 2 cached entries"

        # Backward MB1
        out1.sum().backward()
        for i, layer in enumerate(model.layers):
            assert len(layer.mlp._routing_cache) == 1, f"Layer {i}: expected 1 after MB1 backward"

        # Backward MB2
        out2.sum().backward()
        for i, layer in enumerate(model.layers):
            assert len(layer.mlp._routing_cache) == 0, f"Layer {i}: expected 0 after MB2 backward"

    def test_nondeterministic_attention_routing_preserved(self):
        """With non-deterministic attention, cached routing is replayed on recompute."""
        layer = _SimpleDecoderLayer().cuda()
        layer.train()
        moe = layer.mlp

        x = torch.randn(1, 16, 64, device="cuda", requires_grad=True)
        out = _checkpoint_with_routing(layer, x)

        assert len(moe._routing_cache) == 1

        # Should not raise CheckpointError — routing is deterministic via cache
        out.sum().backward()
        assert len(moe._routing_cache) == 0


# ===========================================================================
# 3. train_router flag
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTrainRouter:
    """Verify train_router controls gradient flow through expert path."""

    def test_gate_gets_grad_when_train_router_true(self):
        moe = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
            train_router=True,
        ).cuda()
        for p in moe.parameters():
            nn.init.normal_(p, std=0.01)
        moe.train()

        x = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        out, _ = moe(x)
        out.sum().backward()

        assert moe.gate.weight.grad is not None
        assert moe.gate.weight.grad.abs().sum() > 0

    def test_gate_no_grad_when_train_router_false(self):
        moe = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
            train_router=False,
        ).cuda()
        moe.train()

        x = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        out, router_logits = moe(x)
        out.sum().backward()

        assert moe.gate.weight.grad is None or moe.gate.weight.grad.abs().sum() == 0

    def test_train_router_false_checkpoint_consistent(self):
        """train_router=False works with gradient checkpointing."""
        layer = _SimpleDecoderLayer().cuda()
        layer.mlp.train_router = False
        layer.train()

        x = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        out = _checkpoint_with_routing(layer, x)
        assert len(layer.mlp._routing_cache) == 1

        # Should not raise CheckpointError
        out.sum().backward()
        assert len(layer.mlp._routing_cache) == 0

    def test_train_router_false_pp(self):
        """train_router=False works with PP scheduling + checkpointing."""
        layer = _SimpleDecoderLayer().cuda()
        layer.mlp.train_router = False
        layer.train()

        x1 = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        x2 = torch.randn(1, 12, 64, device="cuda", requires_grad=True)

        out1 = _checkpoint_with_routing(layer, x1)
        out2 = _checkpoint_with_routing(layer, x2)
        assert len(layer.mlp._routing_cache) == 2

        out1.sum().backward()
        assert len(layer.mlp._routing_cache) == 1
        out2.sum().backward()
        assert len(layer.mlp._routing_cache) == 0


# ===========================================================================
# 4. XorlPreTrainedModel integration
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBaseModelIntegration:
    """Verify that XorlPreTrainedModel.gradient_checkpointing_enable
    correctly injects context_fn for MoE models."""

    def test_context_fn_injected_for_moe(self):
        """gradient_checkpointing_enable adds context_fn when MoE blocks present."""

        class _FakeConfig:
            pass

        class _MoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = _SimpleDecoderLayer()
                self.gradient_checkpointing = False

        from xorl.models.base import XorlPreTrainedModel

        # Monkey-patch a minimal model
        model = XorlPreTrainedModel.__new__(XorlPreTrainedModel)
        nn.Module.__init__(model)
        model.config = _FakeConfig()
        model.gradient_checkpointing = False
        model.layer = _SimpleDecoderLayer()

        model.gradient_checkpointing_enable()

        assert model.gradient_checkpointing is True
        assert model._gradient_checkpointing_func is not None

        # Verify the checkpoint function uses context_fn by running it
        model.layer.train()
        model.cuda()
        x = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        out = model._gradient_checkpointing_func(model.layer, x)
        assert len(model.layer.mlp._routing_cache) == 1

        out.sum().backward()
        assert len(model.layer.mlp._routing_cache) == 0
