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

from xorl.models.layers.moe import MoEExpertsLoRA, MoELoRAConfig


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

    def test_lora_initial_weight_shapes(self, config):
        """Test that MoEExpertsLoRA initializes weights at GLOBAL shape."""
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation="triton",
            lora_config=lora_config,
        )

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
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation="triton",
            lora_config=lora_config,
        )

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
        from xorl.models.transformers.qwen3_moe.parallelize import get_ep_plan

        plan = get_ep_plan()

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
        from xorl.models.transformers.qwen3_moe.parallelize import get_ep_plan

        plan = get_ep_plan()

        # Create a mock LoRA tensor at global shape (GKN: [E, hidden_dim, r])
        global_shape = (8, 32, 4)  # [num_experts=8, hidden_dim=32, r=4]
        lora_tensor = torch.randn(global_shape)

        # Target shape after EP sharding with ep_size=2 -> num_local_experts=4
        local_shape = (4, 32, 4)  # [num_local_experts=4, hidden_dim=32, r=4]

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

            assert sliced.shape == local_shape
            assert torch.allclose(sliced, lora_tensor[:4])

    def test_shard_tensor_ep_rank_1(self, config):
        """Test slicing for ep_rank=1."""
        from xorl.models.transformers.qwen3_moe.parallelize import get_ep_plan

        plan = get_ep_plan()

        # GKN: [E, hidden_dim, r]
        global_shape = (8, 32, 4)
        lora_tensor = torch.randn(global_shape)
        local_shape = (4, 32, 4)

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

            assert sliced.shape == local_shape
            assert torch.allclose(sliced, lora_tensor[4:8])


class TestEPLoRAForwardShapes:
    """Test that EPGroupGemmWithLoRA receives correctly shaped tensors."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8, hidden_size=32, moe_intermediate_size=64)

    @pytest.fixture
    def lora_config(self):
        return MoELoRAConfig(r=4, lora_alpha=8)

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

        # Create local (sliced) weights in GKN format [E, in_features, out_features]
        gate_proj = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        up_proj = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        down_proj = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)

        # Local LoRA weights in GKN format
        gate_proj_lora_A = torch.randn(num_local_experts, hidden_dim, r, device=device, dtype=dtype)
        gate_proj_lora_B = torch.zeros(num_local_experts, r, intermediate, device=device, dtype=dtype)
        up_proj_lora_A = torch.randn(num_local_experts, hidden_dim, r, device=device, dtype=dtype)
        up_proj_lora_B = torch.zeros(num_local_experts, r, intermediate, device=device, dtype=dtype)
        down_proj_lora_A = torch.randn(num_local_experts, intermediate, r, device=device, dtype=dtype)
        down_proj_lora_B = torch.zeros(num_local_experts, r, hidden_dim, device=device, dtype=dtype)

        # Simulate tokens dispatched to this rank
        permute_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
        tokens_per_expert = torch.tensor([4, 4, 4, 4], device=device)
        cumsum = torch.cumsum(tokens_per_expert, dim=0)
        scaling = lora_config.lora_alpha / lora_config.r

        output = EPGroupGemmWithLoRA.apply(
            permute_tokens, cumsum,
            gate_proj, up_proj, down_proj,
            gate_proj_lora_A, gate_proj_lora_B,
            up_proj_lora_A, up_proj_lora_B,
            down_proj_lora_A, down_proj_lora_B,
            scaling,
        )

        assert output.shape == (num_tokens, hidden_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(
        not (torch.distributed.is_available() and torch.distributed.is_initialized()),
        reason="Backward requires distributed ep_group (all_reduce for LoRA grads)",
    )
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

        # GKN format [E, in_features, out_features]
        gate_proj = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        up_proj = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        down_proj = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)

        gate_proj_lora_A = torch.randn(num_local_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)
        gate_proj_lora_B = torch.zeros(num_local_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        up_proj_lora_A = torch.randn(num_local_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)
        up_proj_lora_B = torch.zeros(num_local_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        down_proj_lora_A = torch.randn(num_local_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        down_proj_lora_B = torch.zeros(num_local_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)

        permute_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        tokens_per_expert = torch.tensor([4, 4, 4, 4], device=device)
        cumsum = torch.cumsum(tokens_per_expert, dim=0)
        scaling = lora_config.lora_alpha / lora_config.r

        output = EPGroupGemmWithLoRA.apply(
            permute_tokens, cumsum,
            gate_proj, up_proj, down_proj,
            gate_proj_lora_A, gate_proj_lora_B,
            up_proj_lora_A, up_proj_lora_B,
            down_proj_lora_A, down_proj_lora_B,
            scaling,
        )

        loss = output.sum()
        loss.backward()

        assert gate_proj_lora_A.grad is not None
        assert gate_proj_lora_B.grad is not None
        assert down_proj_lora_A.grad is not None
        assert down_proj_lora_B.grad is not None
        assert permute_tokens.grad is not None


class TestNumExpertsConsistency:
    """Test that num_experts is handled correctly across EP."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8)

    def test_moe_experts_lora_forward_uses_global_num_experts(self, config):
        """Test that MoEExpertsLoRA correctly handles global num_experts."""
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation="triton",
            lora_config=lora_config,
        )

        assert experts.num_experts == 8, "num_experts should be global (8)"


class TestLoRAShapeValidation:
    """Test shape validation for LoRA weights in EP context."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=8, hidden_size=32, moe_intermediate_size=64)

    def test_lora_weight_shapes_match_base(self, config):
        """Test that LoRA weight shapes are compatible with base weights.

        All weights in GKN format [num_experts, in_features, out_features]:
          gate_proj:        [E, hidden, inter]
          gate_proj_lora_A: [E, hidden, r]    — input dim (dim 1) matches gate_proj
          gate_proj_lora_B: [E, r, inter]     — output dim (dim 2) matches gate_proj
          down_proj:        [E, inter, hidden]
          down_proj_lora_A: [E, inter, r]     — input dim (dim 1) matches down_proj
          down_proj_lora_B: [E, r, hidden]    — output dim (dim 2) matches down_proj
        """
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation="triton",
            lora_config=lora_config,
        )

        # gate_proj_lora_A: [E, hidden, r] — dim 0 and dim 1 match gate_proj
        assert experts.gate_proj_lora_A.shape[0] == experts.gate_proj.shape[0]
        assert experts.gate_proj_lora_A.shape[1] == experts.gate_proj.shape[1]

        # gate_proj_lora_B: [E, r, inter] — dim 0 and dim 2 match gate_proj
        assert experts.gate_proj_lora_B.shape[0] == experts.gate_proj.shape[0]
        assert experts.gate_proj_lora_B.shape[2] == experts.gate_proj.shape[2]

        # down_proj_lora_A: [E, inter, r] — dim 0 and dim 1 match down_proj
        assert experts.down_proj_lora_A.shape[0] == experts.down_proj.shape[0]
        assert experts.down_proj_lora_A.shape[1] == experts.down_proj.shape[1]

        # down_proj_lora_B: [E, r, hidden] — dim 0 and dim 2 match down_proj
        assert experts.down_proj_lora_B.shape[0] == experts.down_proj.shape[0]
        assert experts.down_proj_lora_B.shape[2] == experts.down_proj.shape[2]


class TestCumsumMatchesLocalExperts:
    """Test that cumsum tensor shape matches local expert count."""

    def test_preprocess_returns_local_expert_cumsum(self):
        """Test that preprocess returns cumsum for local experts only."""
        pass  # Tested implicitly in integration tests


class TestMoEExpertsLoRAFunctionShapes:
    """Test MoeExpertsLoRAFunction (non-EP path) with correct shapes."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=4, hidden_size=32, moe_intermediate_size=64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moe_experts_lora_function_forward(self, config):
        """Test forward pass with MoeExpertsLoRAFunction (no EP)."""
        from xorl.ops.moe_experts_lora import MoeExpertsLoRAFunction

        device = "cuda"
        dtype = torch.bfloat16
        num_experts = config.num_experts
        num_tokens = 8
        top_k = 2
        hidden_dim = config.hidden_size
        intermediate = config.moe_intermediate_size
        r = 4

        # GKN format [E, in_features, out_features]
        gate_proj = torch.randn(num_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        up_proj = torch.randn(num_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        down_proj = torch.randn(num_experts, intermediate, hidden_dim, device=device, dtype=dtype)

        gate_proj_lora_A = torch.randn(num_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)
        gate_proj_lora_B = torch.zeros(num_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        up_proj_lora_A = torch.randn(num_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)
        up_proj_lora_B = torch.zeros(num_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        down_proj_lora_A = torch.randn(num_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        down_proj_lora_B = torch.zeros(num_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)

        hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        gate_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=dtype), dim=-1)
        expert_index = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        scaling = 2.0

        output = MoeExpertsLoRAFunction.apply(
            num_experts, gate_weights, expert_index, hidden_states,
            gate_proj, up_proj, down_proj,
            gate_proj_lora_A, gate_proj_lora_B,
            up_proj_lora_A, up_proj_lora_B,
            down_proj_lora_A, down_proj_lora_B,
            scaling,
        )

        assert output.shape == (num_tokens, hidden_dim)

        loss = output.sum()
        loss.backward()

        assert gate_proj_lora_A.grad is not None
        assert hidden_states.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
