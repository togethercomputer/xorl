"""Tests for Expert Parallel (EP) LoRA weight slicing.

This module tests that LoRA weights are properly sliced when using Expert Parallelism.
The key verification points are:
1. LoRA weights are initialized at GLOBAL shape [num_experts, ...]
2. parallel_plan.apply() correctly slices them to [num_local_experts, ...]
3. TritonEPGroupGemmWithLoRA receives correctly shaped weights
4. Gradients flow correctly through the EP path
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from torch.distributed._tensor import Shard

from xorl.models.layers.moe import MoEExpertsLoRA, MoELoRAConfig

pytestmark = [pytest.mark.distributed]


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


class TestLoRAWeightInitAndShapes:
    """Test LoRA weight initialization, shapes, and compatibility with base weights."""

    def test_initial_shapes_zeros_and_base_compatibility(self):
        """LoRA weights initialized at global shape, B matrices zeroed, shapes match base weights."""
        config = MockConfig(num_experts=8, hidden_size=32, moe_intermediate_size=64)
        lora_config = MoELoRAConfig(r=4, lora_alpha=8)
        experts = MoEExpertsLoRA(
            num_experts=config.num_experts,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            moe_implementation="triton",
            lora_config=lora_config,
        )

        # Global shapes
        assert experts.num_experts == 8
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            assert getattr(experts, proj).shape[0] == config.num_experts
            assert getattr(experts, f"{proj}_lora_A").shape[0] == config.num_experts
            assert getattr(experts, f"{proj}_lora_B").shape[0] == config.num_experts

        # B matrices zeroed
        assert torch.allclose(experts.gate_proj_lora_B, torch.zeros_like(experts.gate_proj_lora_B))
        assert torch.allclose(experts.up_proj_lora_B, torch.zeros_like(experts.up_proj_lora_B))
        assert torch.allclose(experts.down_proj_lora_B, torch.zeros_like(experts.down_proj_lora_B))

        # LoRA A/B shapes compatible with base weights (GKN format)
        assert experts.gate_proj_lora_A.shape[1] == experts.gate_proj.shape[1]  # input dim
        assert experts.gate_proj_lora_B.shape[2] == experts.gate_proj.shape[2]  # output dim
        assert experts.down_proj_lora_A.shape[1] == experts.down_proj.shape[1]
        assert experts.down_proj_lora_B.shape[2] == experts.down_proj.shape[2]


class TestParallelPlanLoRASlicing:
    """Test that ParallelPlan correctly includes and slices LoRA weights for EP."""

    def test_ep_plan_and_shard_tensor(self):
        """EP plan includes all LoRA patterns with Shard(0); shard_tensor slices by ep_rank."""
        from xorl.models.transformers.qwen3_moe.parallelize import get_ep_plan

        plan = get_ep_plan()

        # Verify all LoRA patterns are in the plan with Shard(dim=0)
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
            assert isinstance(plan.ep_plan[pattern], Shard)
            assert plan.ep_plan[pattern].dim == 0

        # Shard tensor for ep_rank=0
        global_shape = (8, 32, 4)
        local_shape = (4, 32, 4)
        lora_tensor = torch.randn(global_shape)

        for ep_rank, expected_slice in [(0, slice(0, 4)), (1, slice(4, 8))]:
            with patch("xorl.distributed.parallel_state.get_parallel_state") as mock_ps:
                mock_state = MagicMock()
                mock_state.ep_enabled = True
                mock_state.ep_rank = ep_rank
                mock_ps.return_value = mock_state

                sliced = plan.shard_tensor(
                    lora_tensor,
                    "model.layers.0.mlp.experts.gate_proj_lora_A",
                    local_shape,
                )
                assert sliced.shape == local_shape
                assert torch.allclose(sliced, lora_tensor[expected_slice])


class TestEPLoRAForwardAndGradients:
    """Test forward pass shapes and gradient flow for both EP and non-EP paths."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ep_and_non_ep_forward_with_gradients(self):
        """Both EP and non-EP LoRA forward produce correct output shapes with gradient flow."""
        from xorl.ops.moe.triton_lora import TritonEPGroupGemmWithLoRA, TritonMoeExpertsLoRAFunction

        device = "cuda"
        dtype = torch.bfloat16
        hidden_dim, intermediate, r = 32, 64, 4

        # --- EP path: TritonEPGroupGemmWithLoRA ---
        num_local_experts = 4
        num_tokens = 16
        gate_proj = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        up_proj = torch.randn(num_local_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        down_proj = torch.randn(num_local_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        gate_A = torch.randn(num_local_experts, hidden_dim, r, device=device, dtype=dtype)
        gate_B = torch.zeros(num_local_experts, r, intermediate, device=device, dtype=dtype)
        up_A = torch.randn(num_local_experts, hidden_dim, r, device=device, dtype=dtype)
        up_B = torch.zeros(num_local_experts, r, intermediate, device=device, dtype=dtype)
        down_A = torch.randn(num_local_experts, intermediate, r, device=device, dtype=dtype)
        down_B = torch.zeros(num_local_experts, r, hidden_dim, device=device, dtype=dtype)

        permute_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
        tokens_per_expert = torch.tensor([4, 4, 4, 4], device=device)
        cumsum = torch.cumsum(tokens_per_expert, dim=0)
        scaling = 8.0 / 4.0

        output = TritonEPGroupGemmWithLoRA.apply(
            permute_tokens, cumsum,
            gate_proj, up_proj, down_proj,
            gate_A, gate_B, up_A, up_B, down_A, down_B,
            scaling,
        )
        assert output.shape == (num_tokens, hidden_dim)

        # --- Non-EP path: TritonMoeExpertsLoRAFunction with gradient check ---
        num_experts = 4
        num_tokens2 = 8
        top_k = 2
        gate_proj2 = torch.randn(num_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        up_proj2 = torch.randn(num_experts, hidden_dim, intermediate, device=device, dtype=dtype)
        down_proj2 = torch.randn(num_experts, intermediate, hidden_dim, device=device, dtype=dtype)
        gate_A2 = torch.randn(num_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)
        gate_B2 = torch.zeros(num_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        up_A2 = torch.randn(num_experts, hidden_dim, r, device=device, dtype=dtype, requires_grad=True)
        up_B2 = torch.zeros(num_experts, r, intermediate, device=device, dtype=dtype, requires_grad=True)
        down_A2 = torch.randn(num_experts, intermediate, r, device=device, dtype=dtype, requires_grad=True)
        down_B2 = torch.zeros(num_experts, r, hidden_dim, device=device, dtype=dtype, requires_grad=True)

        hidden_states = torch.randn(num_tokens2, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        gate_weights = torch.softmax(torch.randn(num_tokens2, top_k, device=device, dtype=dtype), dim=-1)
        expert_index = torch.randint(0, num_experts, (num_tokens2, top_k), device=device)

        output2 = TritonMoeExpertsLoRAFunction.apply(
            num_experts, gate_weights, expert_index, hidden_states,
            gate_proj2, up_proj2, down_proj2,
            gate_A2, gate_B2, up_A2, up_B2, down_A2, down_B2,
            scaling,
        )
        assert output2.shape == (num_tokens2, hidden_dim)

        loss = output2.sum()
        loss.backward()
        assert gate_A2.grad is not None
        assert hidden_states.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
