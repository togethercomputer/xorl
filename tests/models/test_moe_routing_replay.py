"""Tests for Megatron/SLIME-style MoE routing replay.

Validates:
1. RoutingReplay unit: record, pop_forward, pop_backward, dual-index, clear, stage.
2. MoEBlock integration: record on forward, replay on recompute via checkpoint.
3. Multi-layer replay: independent replay per MoE layer.
4. XorlPreTrainedModel integration: enable_routing_replay + gradient_checkpointing_enable.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


pytestmark = [pytest.mark.gpu]

from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.moe.routing_replay import (
    RoutingReplay,
    get_replay_stage,
    set_replay_stage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_routing_replay_state():
    """Reset all RoutingReplay state between tests."""
    yield
    set_replay_stage(None)
    RoutingReplay.clear_all()
    RoutingReplay._instances.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        self.gradient_checkpointing = False

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(hidden_states)
        moe_out, _ = self.mlp(hidden_states)
        return hidden_states + moe_out


def _run_with_replay(layer, x):
    """Run a forward + backward with routing replay stage switching."""
    set_replay_stage("record")
    out = checkpoint(layer, x, use_reentrant=False)
    set_replay_stage("replay_backward")
    out.sum().backward()
    RoutingReplay.reset_all_backward()
    return out


# ===========================================================================
# 1. RoutingReplay unit + stage management + clear/registry
# ===========================================================================


class TestRoutingReplayUnit:
    """Unit tests for RoutingReplay class, stage management, clear, and registry."""

    def test_record_pop_dual_index_clear_stage_and_registry(self):
        """Test record, forward/backward pop, dual-index, CPU pinned storage,
        clear, clear_all, reset_all_forward/backward, registry, and stage management."""
        # --- Record and pop ---
        replay = RoutingReplay()
        t1 = torch.tensor([[0, 1], [2, 3]])
        t2 = torch.tensor([[4, 5], [6, 7]])
        t3 = torch.tensor([[8, 9], [10, 11]])

        replay.record(t1)
        replay.record(t2)
        replay.record(t3)
        assert len(replay.top_indices_list) == 3

        assert torch.equal(replay.top_indices_list[0], t1)
        assert torch.equal(replay.top_indices_list[1], t2)

        # Backward index advances independently
        assert replay.backward_index == 0
        idx = replay.top_indices_list[replay.backward_index]
        replay.backward_index += 1
        assert torch.equal(idx, t1)
        idx = replay.top_indices_list[replay.backward_index]
        replay.backward_index += 1
        assert torch.equal(idx, t2)
        assert replay.backward_index == 2

        # Dual index independence
        replay2 = RoutingReplay()
        for i in range(5):
            replay2.record(torch.tensor([[i]]))
        replay2.forward_index = 3
        replay2.backward_index = 1
        assert replay2.forward_index == 3
        assert replay2.backward_index == 1

        # CPU pinned storage
        replay3 = RoutingReplay()
        t = torch.tensor([[0, 1]], device="cpu")
        replay3.record(t)
        stored = replay3.top_indices_list[0]
        assert stored.device.type == "cpu"
        assert stored.is_pinned()

        # --- Clear single instance ---
        replay4 = RoutingReplay()
        replay4.record(torch.tensor([[0]]))
        replay4.forward_index = 1
        replay4.backward_index = 1
        replay4.clear()
        assert replay4.forward_index == 0
        assert replay4.backward_index == 0
        assert len(replay4.top_indices_list) == 0

        # --- Class registry ---
        RoutingReplay._instances.clear()
        r1 = RoutingReplay()
        r2 = RoutingReplay()
        r3 = RoutingReplay()
        assert r1 in RoutingReplay._instances
        assert r2 in RoutingReplay._instances
        assert r3 in RoutingReplay._instances
        assert len(RoutingReplay._instances) == 3

        # clear_all
        r1.record(torch.tensor([[0]]))
        r2.record(torch.tensor([[1]]))
        r1.forward_index = 1
        r2.backward_index = 1
        RoutingReplay.clear_all()
        assert r1.forward_index == 0
        assert r2.backward_index == 0
        assert len(r1.top_indices_list) == 0
        assert len(r2.top_indices_list) == 0

        # reset_all_forward
        r1.forward_index = 5
        r2.forward_index = 3
        r1.backward_index = 2
        RoutingReplay.reset_all_forward()
        assert r1.forward_index == 0
        assert r2.forward_index == 0
        assert r1.backward_index == 2  # backward not touched

        # reset_all_backward
        r1.backward_index = 5
        r2.backward_index = 3
        r1.forward_index = 2
        RoutingReplay.reset_all_backward()
        assert r1.backward_index == 0
        assert r2.backward_index == 0
        assert r1.forward_index == 2  # forward not touched

        # --- Stage management ---
        assert get_replay_stage() is None
        set_replay_stage("record")
        assert get_replay_stage() == "record"
        set_replay_stage("replay_forward")
        assert get_replay_stage() == "replay_forward"
        set_replay_stage("replay_backward")
        assert get_replay_stage() == "replay_backward"
        set_replay_stage(None)
        assert get_replay_stage() is None


# ===========================================================================
# 2. MoEBlock + RoutingReplay integration
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMoEBlockReplay:
    """MoEBlock integration with routing replay: record, replay, router training, eval."""

    def test_record_replay_no_replay_and_router_training(self):
        """Test record on forward, replay on backward, no-recording conditions,
        router training/detach, and regather correctness."""
        # --- Record stores expert indices correctly ---
        moe = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
        ).cuda()
        moe.train()
        replay = RoutingReplay()
        moe._routing_replay = replay

        set_replay_stage("record")
        x = torch.randn(1, 8, 64, device="cuda")
        moe(x)

        assert len(replay.top_indices_list) == 1
        stored = replay.top_indices_list[0]
        assert stored.device.type == "cpu"
        assert stored.shape == (8, 2)

        # Full record + replay backward via checkpoint
        layer = _SimpleDecoderLayer().cuda()
        layer.train()
        replay2 = RoutingReplay()
        layer.mlp._routing_replay = replay2

        x2 = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        _run_with_replay(layer, x2)
        assert len(replay2.top_indices_list) == 1
        assert replay2.backward_index == 0

        # Checkpoint determinism with noisy attention
        layer3 = _SimpleDecoderLayer().cuda()
        layer3.train()
        replay3 = RoutingReplay()
        layer3.mlp._routing_replay = replay3
        x3 = torch.randn(1, 16, 64, device="cuda", requires_grad=True)
        _run_with_replay(layer3, x3)

        # --- No-recording conditions ---
        moe2 = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
        ).cuda()
        moe2.train()
        replay4 = RoutingReplay()
        moe2._routing_replay = replay4

        # stage=None => no recording
        set_replay_stage(None)
        moe2(torch.randn(1, 8, 64, device="cuda"))
        assert len(replay4.top_indices_list) == 0

        # No _routing_replay => stage doesn't matter
        moe3 = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
        ).cuda()
        moe3.train()
        assert moe3._routing_replay is None
        set_replay_stage("record")
        moe3(torch.randn(1, 8, 64, device="cuda"))

        # Eval mode with stage=None
        moe4 = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
        ).cuda()
        moe4.eval()
        replay5 = RoutingReplay()
        moe4._routing_replay = replay5
        set_replay_stage(None)
        with torch.no_grad():
            moe4(torch.randn(1, 8, 64, device="cuda"))
        assert len(replay5.top_indices_list) == 0

        # --- Router training and regather ---
        moe5 = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
            norm_topk_prob=True,
        ).cuda()
        moe5.train()

        x5 = torch.randn(1, 8, 64, device="cuda")
        hidden_states = x5.view(-1, 64)
        router_logits = moe5.gate(hidden_states)
        orig_weights, orig_experts = moe5.router(router_logits, x5.dtype)
        regathered_experts, regathered_weights = moe5._regather_routing(router_logits, orig_experts, x5.dtype)
        assert torch.equal(regathered_experts, orig_experts)
        assert torch.allclose(regathered_weights, orig_weights, atol=1e-6)

        # train_router=True => gate gets gradient
        layer4 = _SimpleDecoderLayer().cuda()
        layer4.mlp.train_router = True
        layer4.train()
        replay6 = RoutingReplay()
        layer4.mlp._routing_replay = replay6
        for p in layer4.parameters():
            nn.init.normal_(p, std=0.01)

        x6 = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        _run_with_replay(layer4, x6)
        assert layer4.mlp.gate.weight.grad is not None
        assert layer4.mlp.gate.weight.grad.abs().sum() > 0

        # train_router=False => no gradient from expert path
        moe6 = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
            train_router=False,
        ).cuda()
        moe6.train()
        replay7 = RoutingReplay()
        moe6._routing_replay = replay7

        set_replay_stage("record")
        x7 = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        out, _ = moe6(x7)
        out.sum().backward()
        assert moe6.gate.weight.grad is None or moe6.gate.weight.grad.abs().sum() == 0


# ===========================================================================
# 3. Multi-layer replay + PP schedule
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMultiLayerReplay:
    """Multi-layer PP tests with independent replay per MoE layer."""

    def _make_model(self, num_layers=3, hidden_size=64):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_SimpleDecoderLayer(hidden_size) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = checkpoint(layer, x, use_reentrant=False)
                return x

        model = _Model().cuda()
        model.train()

        replays = []
        for layer in model.layers:
            replay = RoutingReplay()
            layer.mlp._routing_replay = replay
            replays.append(replay)
        return model, replays

    def test_multi_layer_and_pp_schedule(self):
        """Test 3-layer 2-microbatch replay and 4-microbatch 1F1B PP schedule."""
        # 3 layers, 2 micro-batches
        model, replays = self._make_model(num_layers=3)
        x1 = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        x2 = torch.randn(1, 10, 64, device="cuda", requires_grad=True)

        set_replay_stage("record")
        out1 = model(x1)
        out2 = model(x2)

        for i, replay in enumerate(replays):
            assert len(replay.top_indices_list) == 2, f"Layer {i}: expected 2 recorded"

        set_replay_stage("replay_backward")
        out1.sum().backward()
        for i, replay in enumerate(replays):
            assert replay.backward_index == 1, f"Layer {i}: backward_index should be 1"

        out2.sum().backward()
        for i, replay in enumerate(replays):
            assert replay.backward_index == 2, f"Layer {i}: backward_index should be 2"

        # 4-microbatch 1F1B
        model2, replays2 = self._make_model(num_layers=2)
        xs = [torch.randn(1, 8, 64, device="cuda", requires_grad=True) for _ in range(4)]
        outs = []

        # Warmup: 2 forwards
        set_replay_stage("record")
        outs.append(model2(xs[0]))
        outs.append(model2(xs[1]))
        for replay in replays2:
            assert len(replay.top_indices_list) == 2

        # Steady: bwd MB0, fwd MB2
        set_replay_stage("replay_backward")
        outs[0].sum().backward()
        for replay in replays2:
            assert replay.backward_index == 1

        set_replay_stage("record")
        outs.append(model2(xs[2]))
        for replay in replays2:
            assert len(replay.top_indices_list) == 3

        # Steady: bwd MB1, fwd MB3
        set_replay_stage("replay_backward")
        outs[1].sum().backward()
        for replay in replays2:
            assert replay.backward_index == 2

        set_replay_stage("record")
        outs.append(model2(xs[3]))

        # Cooldown: bwd MB2, bwd MB3
        set_replay_stage("replay_backward")
        outs[2].sum().backward()
        for replay in replays2:
            assert replay.backward_index == 3
        outs[3].sum().backward()
        for replay in replays2:
            assert replay.backward_index == 4
            assert len(replay.top_indices_list) == 4


# ===========================================================================
# 4. XorlPreTrainedModel integration + R3 preload
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBaseModelIntegrationAndR3Preload:
    """XorlPreTrainedModel integration and R3 preload (replay_forward)."""

    def test_enable_routing_replay_checkpoint_e2e_and_r3_preload(self):
        """Test enable_routing_replay, gradient_checkpointing_enable, full e2e, and R3 preload."""
        from xorl.models.base import XorlPreTrainedModel

        class _FakeConfig:
            pass

        # enable_routing_replay creates instances
        model = XorlPreTrainedModel.__new__(XorlPreTrainedModel)
        nn.Module.__init__(model)
        model.config = _FakeConfig()
        model.gradient_checkpointing = False
        model.layer = _SimpleDecoderLayer()

        moe_layers = model.enable_routing_replay()
        assert len(moe_layers) == 1
        assert model.layer.mlp._routing_replay is not None
        assert isinstance(model.layer.mlp._routing_replay, RoutingReplay)

        # gradient_checkpointing_enable creates replay
        model2 = XorlPreTrainedModel.__new__(XorlPreTrainedModel)
        nn.Module.__init__(model2)
        model2.config = _FakeConfig()
        model2.gradient_checkpointing = False
        model2.layer = _SimpleDecoderLayer()
        model2.gradient_checkpointing_enable()
        assert model2.gradient_checkpointing is True
        assert model2.layer.mlp._routing_replay is not None

        # Full e2e with checkpoint
        model3 = XorlPreTrainedModel.__new__(XorlPreTrainedModel)
        nn.Module.__init__(model3)
        model3.config = _FakeConfig()
        model3.gradient_checkpointing = False
        model3.layer = _SimpleDecoderLayer()
        model3 = model3.cuda()
        model3.gradient_checkpointing_enable()
        model3.train()

        replay = model3.layer.mlp._routing_replay
        x = torch.randn(1, 8, 64, device="cuda", requires_grad=True)
        set_replay_stage("record")
        out = checkpoint(model3.layer, x, use_reentrant=False)
        assert len(replay.top_indices_list) == 1
        set_replay_stage("replay_backward")
        out.sum().backward()

        # Non-MoE model gets no replay instances
        model4 = XorlPreTrainedModel.__new__(XorlPreTrainedModel)
        nn.Module.__init__(model4)
        model4.config = _FakeConfig()
        model4.gradient_checkpointing = False
        model4.layer = nn.Linear(64, 64)
        moe_layers = model4.enable_routing_replay()
        assert len(moe_layers) == 0

        # --- R3 Preload ---
        moe = MoEBlock(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            intermediate_size=128,
            moe_implementation="eager",
        ).cuda()
        moe.eval()

        r3_replay = RoutingReplay()
        moe._routing_replay = r3_replay

        # Single micro-batch preload
        fake_experts = torch.randint(0, 4, (8, 2))
        r3_replay.record(fake_experts)

        set_replay_stage("replay_forward")
        x = torch.randn(1, 8, 64, device="cuda")
        with torch.no_grad():
            out, router_logits = moe(x)
        assert r3_replay.forward_index == 1
        assert out.shape == (1, 8, 64)

        # Second micro-batch
        r3_replay.record(torch.randint(0, 4, (12, 2)))
        with torch.no_grad():
            moe(torch.randn(1, 12, 64, device="cuda"))
        assert r3_replay.forward_index == 2
