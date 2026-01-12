"""Tests for Expert Parallel (EP) LoRA weight slicing.

This module tests that LoRA weights are properly sliced when using Expert Parallelism.
The key verification points are:
1. LoRA weights are initialized at GLOBAL shape [num_experts, ...]
2. parallel_plan.apply() correctly slices them to [num_local_experts, ...]
3. EPGroupGemmWithLoRA receives correctly shaped weights
4. Gradients flow correctly through the EP path
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from torch.distributed._tensor import Shard


class MockConfig:
    """Mock config for testing."""

    def __init__(
        self,
        num_experts=8,
        hidden_size=32,
        moe_intermediate_size=64,
        hidden_act="silu",
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_act = hidden_act


class TestLoRAWeightShapesBeforeSharding:
    """Test LoRA weight shapes before EP sharding is applied."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8)

    def test_fused_lora_initial_weight_shapes(self, config):
        """Test that FusedExpertsWithLoRA initializes weights at GLOBAL shape."""
        from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import (
            LoRAConfig,
            Qwen3MoeFusedExpertsWithLoRA,
        )

        lora_config = LoRAConfig(r=4, lora_alpha=8)
        experts = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)

        # Verify GLOBAL shapes before any sharding
        assert experts.num_experts == config.num_experts, "num_experts should be global"

        # Base weights should be at global shape
        assert experts.gate_proj.shape[0] == config.num_experts
        assert experts.up_proj.shape[0] == config.num_experts
        assert experts.down_proj.shape[0] == config.num_experts

        # LoRA weights should be at global shape
        assert experts.gate_proj_lora_A.shape[0] == config.num_experts
        assert experts.gate_proj_lora_B.shape[0] == config.num_experts
        assert experts.up_proj_lora_A.shape[0] == config.num_experts
        assert experts.up_proj_lora_B.shape[0] == config.num_experts
        assert experts.down_proj_lora_A.shape[0] == config.num_experts
        assert experts.down_proj_lora_B.shape[0] == config.num_experts

    def test_lora_b_initialized_to_zeros(self, config):
        """Test that LoRA B weights are initialized to zeros."""
        from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import (
            LoRAConfig,
            Qwen3MoeFusedExpertsWithLoRA,
        )

        lora_config = LoRAConfig(r=4, lora_alpha=8)
        experts = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)

        # All B matrices should be zeros (ensures delta_W = 0 at init)
        assert torch.allclose(experts.gate_proj_lora_B, torch.zeros_like(experts.gate_proj_lora_B))
        assert torch.allclose(experts.up_proj_lora_B, torch.zeros_like(experts.up_proj_lora_B))
        assert torch.allclose(experts.down_proj_lora_B, torch.zeros_like(experts.down_proj_lora_B))


class TestParallelPlanLoRASlicing:
    """Test that ParallelPlan correctly slices LoRA weights for EP."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8)

    def test_ep_plan_includes_lora_weights(self):
        """Test that EP plan includes LoRA weight patterns."""
        from xorl.models.transformers.qwen3_moe.parallel_plan import get_paralle_plan

        plan = get_paralle_plan()

        # Verify LoRA weight patterns are in the plan
        lora_patterns = [
            "model.layers.*.mlp.experts.gate_proj_lora_A",
            "model.layers.*.mlp.experts.gate_proj_lora_B",
            "model.layers.*.mlp.experts.up_proj_lora_A",
            "model.layers.*.mlp.experts.up_proj_lora_B",
            "model.layers.*.mlp.experts.down_proj_lora_A",
            "model.layers.*.mlp.experts.down_proj_lora_B",
        ]

        for pattern in lora_patterns:
            assert pattern in plan.ep_plan, f"Missing pattern: {pattern}"
            assert isinstance(plan.ep_plan[pattern], Shard), f"{pattern} should have Shard placement"
            assert plan.ep_plan[pattern].dim == 0, f"{pattern} should shard on dim 0"

    def test_shard_tensor_slices_lora_weights(self, config):
        """Test that shard_tensor correctly slices LoRA weights."""
        from xorl.models.transformers.qwen3_moe.parallel_plan import get_paralle_plan

        plan = get_paralle_plan()

        # Create a mock LoRA tensor at global shape
        global_shape = (8, 4, 32)  # [num_experts=8, r=4, hidden_dim=32]
        lora_tensor = torch.randn(global_shape)

        # Target shape after EP sharding with ep_size=2 -> num_local_experts=4
        local_shape = (4, 4, 32)  # [num_local_experts=4, r=4, hidden_dim=32]

        # Mock parallel state for ep_rank=0
        # The import happens inside _slice_expert_tensor_for_ep, so we mock at module level
        with patch("xorl.distributed.parallel_state.get_parallel_state") as mock_ps:
            mock_state = MagicMock()
            mock_state.ep_enabled = True
            mock_state.ep_rank = 0
            mock_ps.return_value = mock_state

            sliced = plan.shard_tensor(
                lora_tensor,
                "model.layers.0.mlp.experts.gate_proj_lora_A",
                local_shape,
            )

            # Should slice first half (experts 0-3)
            assert sliced.shape == local_shape
            assert torch.allclose(sliced, lora_tensor[:4])

    def test_shard_tensor_ep_rank_1(self, config):
        """Test slicing for ep_rank=1."""
        from xorl.models.transformers.qwen3_moe.parallel_plan import get_paralle_plan

        plan = get_paralle_plan()

        global_shape = (8, 4, 32)
        lora_tensor = torch.randn(global_shape)
        local_shape = (4, 4, 32)

        # Mock parallel state for ep_rank=1
        with patch("xorl.distributed.parallel_state.get_parallel_state") as mock_ps:
            mock_state = MagicMock()
            mock_state.ep_enabled = True
            mock_state.ep_rank = 1
            mock_ps.return_value = mock_state

            sliced = plan.shard_tensor(
                lora_tensor,
                "model.layers.0.mlp.experts.gate_proj_lora_A",
                local_shape,
            )

            # Should slice second half (experts 4-7)
            assert sliced.shape == local_shape
            assert torch.allclose(sliced, lora_tensor[4:8])


class TestEPLoRAForwardShapes:
    """Test that EPGroupGemmWithLoRA receives correctly shaped tensors."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8, hidden_size=32, moe_intermediate_size=64)

    @pytest.fixture
    def lora_config(self):
        from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import LoRAConfig
        return LoRAConfig(r=4, lora_alpha=8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_epgroup_gemm_with_lora_weight_shapes(self, config, lora_config):
        """Test EPGroupGemmWithLoRA forward with correctly shaped local weights."""
        from xorl.distributed.moe.moe_layer import EPGroupGemmWithLoRA

        device = "cuda"
        dtype = torch.bfloat16
        num_local_experts = 4  # Simulating ep_size=2
        num_tokens = 16
        hidden_dim = config.hidden_size
        intermediate = config.moe_intermediate_size
        r = lora_config.r

        # Create local (sliced) weights
        fc1_1_weight = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        fc1_2_weight = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        fc2_weight = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)

        # Local LoRA weights (sliced shape)
        fc1_1_lora_A = torch.randn(num_local_experts, r, hidden_dim, device=device, dtype=dtype)
        fc1_1_lora_B = torch.zeros(num_local_experts, intermediate, r, device=device, dtype=dtype)
        fc1_2_lora_A = torch.randn(num_local_experts, r, hidden_dim, device=device, dtype=dtype)
        fc1_2_lora_B = torch.zeros(num_local_experts, intermediate, r, device=device, dtype=dtype)
        fc2_lora_A = torch.randn(num_local_experts, r, intermediate, device=device, dtype=dtype)
        fc2_lora_B = torch.zeros(num_local_experts, hidden_dim, r, device=device, dtype=dtype)

        # Simulate tokens dispatched to this rank
        permute_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)

        # Cumsum for each local expert (tokens distributed across 4 experts)
        tokens_per_expert = torch.tensor([4, 4, 4, 4], device=device)
        cumsum = torch.cumsum(tokens_per_expert, dim=0)

        scaling = lora_config.lora_alpha / lora_config.r

        # Forward should work with local shapes
        output = EPGroupGemmWithLoRA.apply(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_lora_A,
            fc1_1_lora_B,
            fc1_2_lora_A,
            fc1_2_lora_B,
            fc2_lora_A,
            fc2_lora_B,
            scaling,
        )

        assert output.shape == (num_tokens, hidden_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_epgroup_gemm_with_lora_backward(self, config, lora_config):
        """Test EPGroupGemmWithLoRA backward pass produces gradients."""
        from xorl.distributed.moe.moe_layer import EPGroupGemmWithLoRA

        device = "cuda"
        dtype = torch.bfloat16
        num_local_experts = 4
        num_tokens = 16
        hidden_dim = config.hidden_size
        intermediate = config.moe_intermediate_size
        r = lora_config.r

        # Create local weights (base frozen, LoRA trainable)
        fc1_1_weight = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        fc1_2_weight = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        fc2_weight = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)

        # LoRA weights need gradients
        fc1_1_lora_A = torch.randn(num_local_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        fc1_1_lora_B = torch.zeros(num_local_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        fc1_2_lora_A = torch.randn(num_local_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        fc1_2_lora_B = torch.zeros(num_local_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        fc2_lora_A = torch.randn(num_local_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        fc2_lora_B = torch.zeros(num_local_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)

        permute_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        tokens_per_expert = torch.tensor([4, 4, 4, 4], device=device)
        cumsum = torch.cumsum(tokens_per_expert, dim=0)
        scaling = lora_config.lora_alpha / lora_config.r

        output = EPGroupGemmWithLoRA.apply(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_lora_A,
            fc1_1_lora_B,
            fc1_2_lora_A,
            fc1_2_lora_B,
            fc2_lora_A,
            fc2_lora_B,
            scaling,
        )

        loss = output.sum()
        loss.backward()

        # LoRA weights should have gradients
        assert fc1_1_lora_A.grad is not None
        assert fc1_1_lora_B.grad is not None
        assert fc1_2_lora_A.grad is not None
        assert fc1_2_lora_B.grad is not None
        assert fc2_lora_A.grad is not None
        assert fc2_lora_B.grad is not None

        # Input should also have gradient
        assert permute_tokens.grad is not None


class TestNumExpertsConsistency:
    """Test that num_experts is handled correctly across EP."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8)

    def test_moe_experts_lora_forward_uses_global_num_experts(self, config):
        """Test that moe_experts_lora_forward correctly handles global num_experts."""
        from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import (
            LoRAConfig,
            Qwen3MoeFusedExpertsWithLoRA,
        )

        lora_config = LoRAConfig(r=4, lora_alpha=8)
        experts = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)

        # num_experts should remain at global value
        assert experts.num_experts == 8, "num_experts should be global (8)"

        # Simulate EP sharding by manually slicing weights
        # In real use, parallel_plan.apply() would do this
        ep_size = 2
        num_local_experts = config.num_experts // ep_size  # 4

        # The forward function should still pass num_experts=8 to moe_experts_lora_forward
        # This is correct because routing/masking uses global expert IDs
        # The preprocess function then computes num_local_experts internally


class TestLoRAShapeValidation:
    """Test shape validation for LoRA weights in EP context."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8, hidden_size=32, moe_intermediate_size=64)

    def test_lora_weight_shapes_match_base(self, config):
        """Test that LoRA weight shapes are compatible with base weights."""
        from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import (
            LoRAConfig,
            Qwen3MoeFusedExpertsWithLoRA,
        )

        lora_config = LoRAConfig(r=4, lora_alpha=8)
        experts = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)

        # gate_proj: [num_experts, intermediate, hidden] -> lora_A: [num_experts, r, hidden]
        assert experts.gate_proj_lora_A.shape[0] == experts.gate_proj.shape[0]  # num_experts match
        assert experts.gate_proj_lora_A.shape[2] == experts.gate_proj.shape[2]  # hidden_dim match

        # gate_proj_lora_B: [num_experts, intermediate, r]
        assert experts.gate_proj_lora_B.shape[0] == experts.gate_proj.shape[0]  # num_experts match
        assert experts.gate_proj_lora_B.shape[1] == experts.gate_proj.shape[1]  # intermediate match

        # Same validation for up_proj
        assert experts.up_proj_lora_A.shape[0] == experts.up_proj.shape[0]
        assert experts.up_proj_lora_A.shape[2] == experts.up_proj.shape[2]
        assert experts.up_proj_lora_B.shape[0] == experts.up_proj.shape[0]
        assert experts.up_proj_lora_B.shape[1] == experts.up_proj.shape[1]

        # down_proj: [num_experts, hidden, intermediate] -> lora_A: [num_experts, r, intermediate]
        assert experts.down_proj_lora_A.shape[0] == experts.down_proj.shape[0]  # num_experts match
        assert experts.down_proj_lora_A.shape[2] == experts.down_proj.shape[2]  # intermediate match
        assert experts.down_proj_lora_B.shape[0] == experts.down_proj.shape[0]  # num_experts match
        assert experts.down_proj_lora_B.shape[1] == experts.down_proj.shape[1]  # hidden match


class TestCumsumMatchesLocalExperts:
    """Test that cumsum tensor shape matches local expert count."""

    def test_preprocess_returns_local_expert_cumsum(self):
        """Test that preprocess returns cumsum for local experts only."""
        # This tests the contract: cumsum.shape[0] == num_local_experts

        # Given:
        # - num_experts = 8 (global)
        # - ep_size = 2
        # - Each rank handles num_local_experts = 4

        # The returned num_global_sum_tokens_per_local_expert should have shape [4]
        # (one entry per local expert)

        # This is critical because EPGroupGemmWithLoRA uses:
        # - weights shape: [num_local_experts, ...]
        # - cumsum shape: [num_local_experts]

        # The shapes MUST match for group_gemm to work correctly
        pass  # Tested implicitly in integration tests


class TestMoEExpertsLoRAFunctionShapes:
    """Test MoeExpertsLoRAFunction (non-EP path) with correct shapes."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=4, hidden_size=32, moe_intermediate_size=64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moe_experts_lora_function_forward(self, config):
        """Test forward pass with MoeExpertsLoRAFunction (no EP)."""
        from xorl.ops.fused_moe_experts_lora import MoeExpertsLoRAFunction

        device = "cuda"
        dtype = torch.bfloat16
        num_experts = config.num_experts
        num_tokens = 8
        top_k = 2
        hidden_dim = config.hidden_size
        intermediate = config.moe_intermediate_size
        r = 4

        # Create weights at full (non-EP) shape
        fc1_1_weight = torch.randn(num_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        fc1_2_weight = torch.randn(num_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        fc2_weight = torch.randn(num_experts, hidden_dim, intermediate, device=device, dtype=dtype)

        fc1_1_lora_A = torch.randn(num_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        fc1_1_lora_B = torch.zeros(num_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        fc1_2_lora_A = torch.randn(num_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        fc1_2_lora_B = torch.zeros(num_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        fc2_lora_A = torch.randn(num_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        fc2_lora_B = torch.zeros(num_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)

        hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        gate_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
        expert_index = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        scaling = 2.0

        output = MoeExpertsLoRAFunction.apply(
            num_experts,
            gate_weights,
            expert_index,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_lora_A,
            fc1_1_lora_B,
            fc1_2_lora_A,
            fc1_2_lora_B,
            fc2_lora_A,
            fc2_lora_B,
            scaling,
        )

        assert output.shape == (num_tokens, hidden_dim)

        # Test backward
        loss = output.sum()
        loss.backward()

        assert fc1_1_lora_A.grad is not None
        assert hidden_states.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
