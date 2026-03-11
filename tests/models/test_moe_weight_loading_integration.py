"""
Integration tests for MoE weight auto-merging during model loading.

These tests simulate the full flow of loading per-expert HuggingFace weights
into a model that expects fused (stacked) expert format.
"""

import os
import tempfile
from typing import Dict, Iterator, Tuple

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.cpu]

# Re-implement the core logic to test without full xorl dependencies
import re
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple


# =============================================================================
# Copy of core implementation for testing
# =============================================================================

_EXPERT_KEY_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)

_FUSED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate|up|down)_proj$"
)


def parse_expert_key(key: str) -> Optional[Tuple[int, int, str]]:
    match = _EXPERT_KEY_PATTERN.match(key)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def _model_needs_expert_merging(parameter_names: Set[str]) -> bool:
    for name in parameter_names:
        if _FUSED_EXPERT_PATTERN.match(name):
            return True
    return False


class ExpertWeightBuffer:
    """Buffer with streaming copy (copy on add, not batch copy)."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self._stacked_buffers: Dict[Tuple[int, str], torch.Tensor] = {}
        self._filled_experts: Dict[Tuple[int, str], Set[int]] = defaultdict(set)

    def add(self, layer_idx: int, expert_idx: int, proj: str, tensor: torch.Tensor) -> None:
        key = (layer_idx, proj)
        # Transpose from HF nn.Linear [out, in] to (K, N) = [in, out] format
        tensor = tensor.t().contiguous()
        if key not in self._stacked_buffers:
            stacked_shape = (self.num_experts,) + tensor.shape
            self._stacked_buffers[key] = torch.empty(
                stacked_shape, dtype=tensor.dtype, device="cpu"
            )
        # Copy directly into the slice (streaming)
        self._stacked_buffers[key][expert_idx].copy_(tensor)
        self._filled_experts[key].add(expert_idx)

    def is_complete(self, layer_idx: int, proj: str) -> bool:
        key = (layer_idx, proj)
        return len(self._filled_experts.get(key, set())) == self.num_experts

    def pop_stacked(self, layer_idx: int, proj: str) -> torch.Tensor:
        key = (layer_idx, proj)
        if key not in self._stacked_buffers:
            raise KeyError(f"No buffered experts for layer {layer_idx}, projection {proj}")
        filled = self._filled_experts.pop(key)
        if len(filled) != self.num_experts:
            raise ValueError(
                f"Incomplete experts for layer {layer_idx}, {proj}_proj: "
                f"got {len(filled)}, expected {self.num_experts}"
            )
        return self._stacked_buffers.pop(key)

    @staticmethod
    def get_fused_name(layer_idx: int, proj: str) -> str:
        return f"model.layers.{layer_idx}.mlp.experts.{proj}_proj"

    def get_pending_counts(self) -> Dict[Tuple[int, str], int]:
        return {key: len(experts) for key, experts in self._filled_experts.items()}


# =============================================================================
# Mock model classes for testing
# =============================================================================


class MockFusedMoeExperts(nn.Module):
    """Mock MoE experts module with fused/stacked weight format."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        # Fused format (G, K, N): [num_experts, in_features, out_features]
        self.gate_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )


class MockMoeLayer(nn.Module):
    """Mock MoE layer."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.experts = MockFusedMoeExperts(num_experts, hidden_size, intermediate_size)


class MockModelInner(nn.Module):
    """Inner model component that holds the layers."""

    def __init__(self, num_layers: int, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mlp": MockMoeLayer(num_experts, hidden_size, intermediate_size)
            })
            for _ in range(num_layers)
        ])


class MockMoeModel(nn.Module):
    """Mock model with MoE layers expecting fused expert format.

    Structure matches HuggingFace: model.layers.{i}.mlp.experts.{proj}_proj
    """

    def __init__(self, num_layers: int, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        # Use 'model' as the attribute name to match HuggingFace structure
        self.model = MockModelInner(num_layers, num_experts, hidden_size, intermediate_size)

    class Config:
        def __init__(self, num_experts):
            self.num_experts = num_experts

    @property
    def config(self):
        return self.Config(self.num_experts)


# =============================================================================
# Test utilities
# =============================================================================


def create_per_expert_state_dict(
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Dict[str, torch.Tensor]:
    """Create a state dict in per-expert HuggingFace format."""
    state_dict = {}
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            # gate_proj and up_proj: [intermediate_size, hidden_size]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = \
                torch.randn(intermediate_size, hidden_size)
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = \
                torch.randn(intermediate_size, hidden_size)
            # down_proj: [hidden_size, intermediate_size]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = \
                torch.randn(hidden_size, intermediate_size)
    return state_dict


def simulate_load_model_weights(
    model: nn.Module,
    state_dict_iterator: Iterator[Tuple[str, torch.Tensor]],
    num_experts: int,
) -> Dict[str, torch.Tensor]:
    """
    Simulate the load_model_weights function with auto-merge logic.

    Returns the loaded state dict (fused format) for verification.
    """
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    loaded_weights = {}

    # Check if model needs merging
    expert_buffer = None
    if _model_needs_expert_merging(parameter_names_to_load):
        expert_buffer = ExpertWeightBuffer(num_experts)

    for name, tensor in state_dict_iterator:
        # Check if this is a per-expert key that needs merging
        if expert_buffer is not None:
            parsed = parse_expert_key(name)
            if parsed is not None:
                layer_idx, expert_idx, proj = parsed
                expert_buffer.add(layer_idx, expert_idx, proj, tensor)

                # If all experts collected, stack and dispatch
                if expert_buffer.is_complete(layer_idx, proj):
                    fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                    stacked = expert_buffer.pop_stacked(layer_idx, proj)

                    if fused_name in parameter_names_to_load:
                        parameter_names_to_load.remove(fused_name)
                        loaded_weights[fused_name] = stacked
                continue

        # Normal key handling
        if name in parameter_names_to_load:
            parameter_names_to_load.remove(name)
            loaded_weights[name] = tensor

    # Check for incomplete buffers
    if expert_buffer is not None:
        pending = expert_buffer.get_pending_counts()
        if pending:
            raise RuntimeError(f"Incomplete expert weights: {pending}")

    return loaded_weights


# =============================================================================
# Integration Tests
# =============================================================================


class TestMoeWeightLoadingIntegration:
    """Integration tests for MoE weight loading with auto-merge."""

    def test_load_and_verify_shapes_and_order(self):
        """Test loading per-expert weights, verifying fused shapes, expert order, and projection sizes."""
        num_layers = 2
        num_experts = 4
        hidden_size = 32
        intermediate_size = 128

        model = MockMoeModel(num_layers, num_experts, hidden_size, intermediate_size)
        per_expert_state_dict = create_per_expert_state_dict(
            num_layers, num_experts, hidden_size, intermediate_size
        )

        loaded_weights = simulate_load_model_weights(
            model, iter(per_expert_state_dict.items()), num_experts
        )

        # All fused parameters created with correct shapes
        for layer_idx in range(num_layers):
            for proj in ["gate", "up", "down"]:
                param_name = f"model.layers.{layer_idx}.mlp.experts.{proj}_proj"
                assert param_name in loaded_weights, f"Missing {param_name}"

            gate = loaded_weights[f"model.layers.{layer_idx}.mlp.experts.gate_proj"]
            up = loaded_weights[f"model.layers.{layer_idx}.mlp.experts.up_proj"]
            down = loaded_weights[f"model.layers.{layer_idx}.mlp.experts.down_proj"]

            # gate and up: [num_experts, hidden_size, intermediate_size]
            assert gate.shape == (num_experts, hidden_size, intermediate_size)
            assert up.shape == (num_experts, hidden_size, intermediate_size)
            # down: [num_experts, intermediate_size, hidden_size]
            assert down.shape == (num_experts, intermediate_size, hidden_size)

        # Expert order preserved with identifiable values
        model2 = MockMoeModel(1, num_experts, 8, 16)
        state_dict2 = {}
        for expert_idx in range(num_experts):
            state_dict2[f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = \
                torch.full((16, 8), float(expert_idx))
            state_dict2[f"model.layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = \
                torch.randn(16, 8)
            state_dict2[f"model.layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = \
                torch.randn(8, 16)

        import random
        items = list(state_dict2.items())
        random.seed(42)
        random.shuffle(items)

        loaded2 = simulate_load_model_weights(model2, iter(items), num_experts)
        gate_proj = loaded2["model.layers.0.mlp.experts.gate_proj"]
        for expert_idx in range(num_experts):
            assert torch.all(gate_proj[expert_idx] == float(expert_idx)), \
                f"Expert {expert_idx} not in correct position"

    def test_streaming_and_edge_cases(self):
        """Test sharded streaming load, large expert count, single expert, and random order."""
        # Streaming across shards
        num_layers = 2
        num_experts = 8
        hidden_size = 16
        intermediate_size = 32

        model = MockMoeModel(num_layers, num_experts, hidden_size, intermediate_size)
        state_dict = create_per_expert_state_dict(
            num_layers, num_experts, hidden_size, intermediate_size
        )
        items = list(state_dict.items())
        shard1 = [(k, v) for k, v in items if any(f".experts.{i}." in k for i in range(4))]
        shard2 = [(k, v) for k, v in items if any(f".experts.{i}." in k for i in range(4, 8))]
        combined_iterator = iter(shard1 + shard2)

        loaded_weights = simulate_load_model_weights(model, combined_iterator, num_experts)
        for layer_idx in range(num_layers):
            gate = loaded_weights[f"model.layers.{layer_idx}.mlp.experts.gate_proj"]
            assert gate.shape == (num_experts, hidden_size, intermediate_size)

        # Large expert count (128)
        model_large = MockMoeModel(1, 128, 16, 32)
        state_dict_large = create_per_expert_state_dict(1, 128, 16, 32)
        loaded_large = simulate_load_model_weights(
            model_large, iter(state_dict_large.items()), 128
        )
        assert loaded_large["model.layers.0.mlp.experts.gate_proj"].shape == (128, 16, 32)

        # Single expert
        model_single = MockMoeModel(1, 1, 8, 16)
        state_dict_single = create_per_expert_state_dict(1, 1, 8, 16)
        loaded_single = simulate_load_model_weights(
            model_single, iter(state_dict_single.items()), 1
        )
        assert loaded_single["model.layers.0.mlp.experts.gate_proj"].shape == (1, 8, 16)

        # Random order loading
        import random
        model_rand = MockMoeModel(2, 4, 8, 16)
        state_dict_rand = create_per_expert_state_dict(2, 4, 8, 16)
        items_rand = list(state_dict_rand.items())
        random.seed(123)
        random.shuffle(items_rand)
        loaded_rand = simulate_load_model_weights(model_rand, iter(items_rand), 4)
        for layer_idx in range(2):
            for proj in ["gate", "up", "down"]:
                assert f"model.layers.{layer_idx}.mlp.experts.{proj}_proj" in loaded_rand

    def test_incomplete_experts_raises_error(self):
        """Test that incomplete expert set raises an error."""
        model = MockMoeModel(1, 4, 8, 16)

        state_dict = {}
        for expert_idx in range(3):  # Missing expert 3
            state_dict[f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = \
                torch.randn(16, 8)
            state_dict[f"model.layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = \
                torch.randn(16, 8)
            state_dict[f"model.layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = \
                torch.randn(8, 16)

        with pytest.raises(RuntimeError, match="Incomplete expert weights"):
            simulate_load_model_weights(model, iter(state_dict.items()), 4)
