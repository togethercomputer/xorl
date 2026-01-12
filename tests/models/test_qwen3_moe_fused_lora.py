"""Tests for Qwen3MoeFusedExperts and Qwen3MoeFusedExpertsWithLoRA."""

import pytest
import torch
import torch.nn as nn

from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeFusedExperts,
    Qwen3MoeSparseExperts,
)
from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import (
    LoRAConfig,
    Qwen3MoeFusedExpertsWithLoRA,
    Qwen3MoeSparseExpertsWithLoRA,
)
from xorl.lora.mapping import can_apply_lora, get_lora_class_for_module


class MockConfig:
    """Mock config for testing."""
    def __init__(
        self,
        num_experts=4,
        hidden_size=64,
        moe_intermediate_size=128,
        hidden_act="silu",
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_act = hidden_act


class TestQwen3MoeFusedExperts:
    """Tests for base Qwen3MoeFusedExperts."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.fixture
    def fused_experts(self, config):
        experts = Qwen3MoeFusedExperts(config)
        # Initialize weights with small random values
        nn.init.xavier_normal_(experts.gate_proj.data)
        nn.init.xavier_normal_(experts.up_proj.data)
        nn.init.xavier_normal_(experts.down_proj.data)
        return experts

    def test_init(self, config):
        """Test that FusedExperts initializes correctly."""
        experts = Qwen3MoeFusedExperts(config)

        assert experts.num_experts == config.num_experts
        assert experts.hidden_dim == config.hidden_size
        assert experts.intermediate_size == config.moe_intermediate_size

        # Check weight shapes
        assert experts.gate_proj.shape == (config.num_experts, config.moe_intermediate_size, config.hidden_size)
        assert experts.up_proj.shape == (config.num_experts, config.moe_intermediate_size, config.hidden_size)
        assert experts.down_proj.shape == (config.num_experts, config.hidden_size, config.moe_intermediate_size)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused MoE")
    def test_forward(self, fused_experts, config):
        """Test forward pass."""
        device = "cuda"
        fused_experts = fused_experts.to(device).to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        num_tokens = batch_size * seq_len
        top_k = 2

        # Create inputs
        hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected_experts = torch.randint(0, config.num_experts, (num_tokens, top_k), device=device)

        # Forward pass
        output = fused_experts(hidden_states, routing_weights, selected_experts)

        # Check output shape
        assert output.shape == hidden_states.shape
        assert output.dtype == torch.bfloat16

    def test_lora_mapping_registered(self, fused_experts):
        """Test that FusedExperts is registered in LoRA mapping."""
        assert can_apply_lora(fused_experts)
        lora_cls = get_lora_class_for_module(fused_experts)
        assert lora_cls == Qwen3MoeFusedExpertsWithLoRA


class TestQwen3MoeFusedExpertsWithLoRA:
    """Tests for Qwen3MoeFusedExpertsWithLoRA."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    @pytest.fixture
    def lora_config(self):
        return LoRAConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])

    @pytest.fixture
    def fused_experts_lora(self, config, lora_config):
        experts = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)
        # Initialize base weights
        nn.init.xavier_normal_(experts.gate_proj.data)
        nn.init.xavier_normal_(experts.up_proj.data)
        nn.init.xavier_normal_(experts.down_proj.data)
        return experts

    def test_init(self, config, lora_config):
        """Test that FusedExpertsWithLoRA initializes correctly."""
        experts = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)

        assert experts.num_experts == config.num_experts
        assert experts.hidden_dim == config.hidden_size
        assert experts.intermediate_size == config.moe_intermediate_size
        assert experts.lora_config == lora_config

        # Check base weight shapes
        assert experts.gate_proj.shape == (config.num_experts, config.moe_intermediate_size, config.hidden_size)
        assert experts.up_proj.shape == (config.num_experts, config.moe_intermediate_size, config.hidden_size)
        assert experts.down_proj.shape == (config.num_experts, config.hidden_size, config.moe_intermediate_size)

        # Check LoRA weight shapes
        r = lora_config.r
        assert experts.gate_proj_lora_A.shape == (config.num_experts, r, config.hidden_size)
        assert experts.gate_proj_lora_B.shape == (config.num_experts, config.moe_intermediate_size, r)
        assert experts.up_proj_lora_A.shape == (config.num_experts, r, config.hidden_size)
        assert experts.up_proj_lora_B.shape == (config.num_experts, config.moe_intermediate_size, r)
        assert experts.down_proj_lora_A.shape == (config.num_experts, r, config.moe_intermediate_size)
        assert experts.down_proj_lora_B.shape == (config.num_experts, config.hidden_size, r)

    def test_base_weights_frozen(self, fused_experts_lora):
        """Test that base weights are frozen (requires_grad=False)."""
        assert not fused_experts_lora.gate_proj.requires_grad
        assert not fused_experts_lora.up_proj.requires_grad
        assert not fused_experts_lora.down_proj.requires_grad

    def test_lora_weights_trainable(self, fused_experts_lora, lora_config):
        """Test that LoRA weights are trainable."""
        for name in lora_config.target_modules:
            lora_A = getattr(fused_experts_lora, f"{name}_lora_A")
            lora_B = getattr(fused_experts_lora, f"{name}_lora_B")
            assert isinstance(lora_A, nn.Parameter)
            assert isinstance(lora_B, nn.Parameter)
            assert lora_A.requires_grad
            assert lora_B.requires_grad

    def test_lora_initialization(self, fused_experts_lora, lora_config):
        """Test that LoRA B is initialized to zeros."""
        for name in lora_config.target_modules:
            lora_B = getattr(fused_experts_lora, f"{name}_lora_B")
            assert torch.allclose(lora_B, torch.zeros_like(lora_B))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused MoE")
    def test_forward(self, fused_experts_lora, config):
        """Test forward pass."""
        device = "cuda"
        fused_experts_lora = fused_experts_lora.to(device).to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        num_tokens = batch_size * seq_len
        top_k = 2

        # Create inputs
        hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected_experts = torch.randint(0, config.num_experts, (num_tokens, top_k), device=device)

        # Forward pass
        output = fused_experts_lora(hidden_states, routing_weights, selected_experts)

        # Check output shape
        assert output.shape == hidden_states.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused MoE")
    def test_backward(self, fused_experts_lora, config):
        """Test backward pass computes gradients for LoRA weights only."""
        device = "cuda"
        fused_experts_lora = fused_experts_lora.to(device).to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        num_tokens = batch_size * seq_len
        top_k = 2

        # Create inputs
        hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
        routing_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16), dim=-1)
        selected_experts = torch.randint(0, config.num_experts, (num_tokens, top_k), device=device)

        # Forward pass
        output = fused_experts_lora(hidden_states, routing_weights, selected_experts)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check LoRA weights have gradients
        assert fused_experts_lora.gate_proj_lora_A.grad is not None
        assert fused_experts_lora.gate_proj_lora_B.grad is not None
        assert fused_experts_lora.up_proj_lora_A.grad is not None
        assert fused_experts_lora.up_proj_lora_B.grad is not None
        assert fused_experts_lora.down_proj_lora_A.grad is not None
        assert fused_experts_lora.down_proj_lora_B.grad is not None

        # Check base weights have no gradients (frozen)
        assert fused_experts_lora.gate_proj.grad is None
        assert fused_experts_lora.up_proj.grad is None
        assert fused_experts_lora.down_proj.grad is None

    def test_from_module(self, config, lora_config):
        """Test from_module class method."""
        # Create base module
        base_experts = Qwen3MoeFusedExperts(config)
        nn.init.xavier_normal_(base_experts.gate_proj.data)
        nn.init.xavier_normal_(base_experts.up_proj.data)
        nn.init.xavier_normal_(base_experts.down_proj.data)

        # Create LoRA module from base
        lora_experts = Qwen3MoeFusedExpertsWithLoRA.from_module(
            base_experts,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
        )

        # Check dimensions match
        assert lora_experts.num_experts == base_experts.num_experts
        assert lora_experts.hidden_dim == base_experts.hidden_dim
        assert lora_experts.intermediate_size == base_experts.intermediate_size

        # Check base weights are copied
        assert torch.allclose(lora_experts.gate_proj, base_experts.gate_proj)
        assert torch.allclose(lora_experts.up_proj, base_experts.up_proj)
        assert torch.allclose(lora_experts.down_proj, base_experts.down_proj)

        # Check LoRA weights are created
        assert hasattr(lora_experts, "gate_proj_lora_A")
        assert hasattr(lora_experts, "gate_proj_lora_B")

    def test_extra_repr(self, fused_experts_lora, config, lora_config):
        """Test extra_repr includes LoRA info."""
        repr_str = fused_experts_lora.extra_repr()
        assert f"num_experts={config.num_experts}" in repr_str
        assert f"r={lora_config.r}" in repr_str
        assert f"lora_alpha={lora_config.lora_alpha}" in repr_str


class TestLoRAInjection:
    """Test LoRA injection into models."""

    @pytest.fixture
    def config(self):
        return MockConfig()

    def test_inject_lora_into_fused_experts(self, config):
        """Test inject_lora_into_model works with FusedExperts."""
        from xorl.lora import inject_lora_into_model

        # Create a simple model with FusedExperts
        class SimpleModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.experts = Qwen3MoeFusedExperts(config)
                nn.init.xavier_normal_(self.experts.gate_proj.data)
                nn.init.xavier_normal_(self.experts.up_proj.data)
                nn.init.xavier_normal_(self.experts.down_proj.data)

        model = SimpleModel(config)

        # Check original type
        assert isinstance(model.experts, Qwen3MoeFusedExperts)

        # Inject LoRA
        inject_lora_into_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["experts"],  # Target the experts module directly
        )

        # Check type changed to LoRA variant
        assert isinstance(model.experts, Qwen3MoeFusedExpertsWithLoRA)

        # Check LoRA weights exist
        assert hasattr(model.experts, "gate_proj_lora_A")
        assert hasattr(model.experts, "gate_proj_lora_B")

    def test_inject_lora_indirect_matching(self, config):
        """Test inject_lora_into_model works with indirect target matching (gate_proj)."""
        from xorl.lora import inject_lora_into_model

        # Create a simple model with FusedExperts
        class SimpleModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.experts = Qwen3MoeFusedExperts(config)
                nn.init.xavier_normal_(self.experts.gate_proj.data)
                nn.init.xavier_normal_(self.experts.up_proj.data)
                nn.init.xavier_normal_(self.experts.down_proj.data)

        model = SimpleModel(config)

        # Inject LoRA using indirect targets (gate_proj is an attribute of experts)
        inject_lora_into_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )

        # Check type changed to LoRA variant
        assert isinstance(model.experts, Qwen3MoeFusedExpertsWithLoRA)


class TestSparseVsFusedConsistency:
    """Test consistency between sparse and fused implementations."""

    @pytest.fixture
    def config(self):
        return MockConfig(num_experts=4, hidden_size=32, moe_intermediate_size=64)

    @pytest.fixture
    def lora_config(self):
        return LoRAConfig(r=4, lora_alpha=8, target_modules=["gate_proj", "up_proj", "down_proj"])

    def test_lora_weight_shapes_match(self, config, lora_config):
        """Test that LoRA weight shapes match between sparse and fused."""
        sparse = Qwen3MoeSparseExpertsWithLoRA(config, lora_config)
        fused = Qwen3MoeFusedExpertsWithLoRA(config, lora_config)

        # Check all LoRA weight shapes match
        for name in lora_config.target_modules:
            sparse_A = getattr(sparse, f"{name}_lora_A")
            fused_A = getattr(fused, f"{name}_lora_A")
            sparse_B = getattr(sparse, f"{name}_lora_B")
            fused_B = getattr(fused, f"{name}_lora_B")

            assert sparse_A.shape == fused_A.shape, f"{name}_lora_A shape mismatch"
            assert sparse_B.shape == fused_B.shape, f"{name}_lora_B shape mismatch"


class TestLoRAInjectionErrors:
    """Test error handling in LoRA injection."""

    def test_error_no_matching_modules(self):
        """Test that error is raised when no modules match target_modules."""
        from xorl.lora import inject_lora_into_model

        # Create a simple model without matching modules
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(64, 64)
                self.layer2 = nn.Linear(64, 64)

        model = SimpleModel()

        # Try to inject LoRA with non-existent target modules
        with pytest.raises(ValueError, match="No modules found matching target_modules"):
            inject_lora_into_model(
                model,
                r=8,
                lora_alpha=16,
                target_modules=["nonexistent_proj", "another_nonexistent"],
            )

    def test_error_no_lora_support(self):
        """Test that error is raised when matched modules have no LoRA support."""
        from xorl.lora import inject_lora_into_model

        # Create a model with modules that don't have LoRA support
        class UnsupportedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(64, 64))

            def forward(self, x):
                return x @ self.weight

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_layer = UnsupportedModule()

        model = SimpleModel()

        # Try to inject LoRA - should fail because UnsupportedModule has no LoRA class registered
        # The error will be "No modules found" because _find_target_modules only returns
        # modules where can_apply_lora() is True
        with pytest.raises(ValueError, match="No modules found matching target_modules"):
            inject_lora_into_model(
                model,
                r=8,
                lora_alpha=16,
                target_modules=["custom_layer"],
            )

    def test_success_with_valid_target(self):
        """Test that injection succeeds with valid targets."""
        from xorl.lora import inject_lora_into_model

        # Create a model with nn.Linear which has LoRA support
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        model = SimpleModel()

        # Should succeed
        inject_lora_into_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        # Check that modules were replaced
        from xorl.lora import LoraLinear
        assert isinstance(model.q_proj, LoraLinear)
        assert isinstance(model.v_proj, LoraLinear)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
