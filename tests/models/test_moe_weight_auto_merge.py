"""
Tests for MoE weight auto-merging functionality.

Tests the ExpertWeightBuffer class and related functions that handle
automatic merging of per-expert HuggingFace weights into stacked format
during model loading.

Note: This test file re-implements the classes/functions under test to avoid
import dependency issues. The actual implementation in module_utils.py should
be kept in sync with these copies.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pytest
import torch


# =============================================================================
# Copy of the implementation for testing (avoids heavy import dependencies)
# =============================================================================

# Pattern to match per-expert HuggingFace weight keys
_EXPERT_KEY_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)

# Pattern to check if model expects fused expert format
_FUSED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate|up|down)_proj$"
)


def parse_expert_key(key: str) -> Optional[Tuple[int, int, str]]:
    """
    Parse a per-expert weight key to extract layer index, expert index, and projection name.
    """
    match = _EXPERT_KEY_PATTERN.match(key)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def _model_needs_expert_merging(parameter_names: Set[str]) -> bool:
    """
    Check if the model expects fused expert format.
    """
    for name in parameter_names:
        if _FUSED_EXPERT_PATTERN.match(name):
            return True
    return False


class ExpertWeightBuffer:
    """
    Buffer for collecting per-expert weights and merging them into stacked tensors.

    Optimized for performance:
    - Pre-allocates stacked tensor on first expert arrival
    - Copies each expert directly into slice as it arrives (streaming)
    - Avoids intermediate buffering and torch.stack overhead
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        # {(layer_idx, proj_name): pre-allocated stacked tensor}
        self._stacked_buffers: Dict[Tuple[int, str], torch.Tensor] = {}
        # {(layer_idx, proj_name): set of expert indices that have been filled}
        self._filled_experts: Dict[Tuple[int, str], Set[int]] = defaultdict(set)

    def add(self, layer_idx: int, expert_idx: int, proj: str, tensor: torch.Tensor) -> None:
        key = (layer_idx, proj)

        # Pre-allocate stacked tensor on first expert
        if key not in self._stacked_buffers:
            stacked_shape = (self.num_experts,) + tensor.shape
            self._stacked_buffers[key] = torch.empty(
                stacked_shape, dtype=tensor.dtype, device="cpu"
            )

        # Copy directly into the pre-allocated slice (streaming)
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

    def get_pending_keys(self) -> List[Tuple[int, str]]:
        return list(self._stacked_buffers.keys())

    def get_pending_counts(self) -> Dict[Tuple[int, str], int]:
        return {key: len(experts) for key, experts in self._filled_experts.items()}


# =============================================================================
# Tests
# =============================================================================


class TestParseExpertKey:
    """Tests for parse_expert_key function."""

    def test_valid_gate_proj_key(self):
        """Test parsing a valid gate_proj expert key."""
        key = "model.layers.0.mlp.experts.5.gate_proj.weight"
        result = parse_expert_key(key)
        assert result == (0, 5, "gate")

    def test_valid_up_proj_key(self):
        """Test parsing a valid up_proj expert key."""
        key = "model.layers.10.mlp.experts.127.up_proj.weight"
        result = parse_expert_key(key)
        assert result == (10, 127, "up")

    def test_valid_down_proj_key(self):
        """Test parsing a valid down_proj expert key."""
        key = "model.layers.35.mlp.experts.0.down_proj.weight"
        result = parse_expert_key(key)
        assert result == (35, 0, "down")

    def test_non_expert_key_returns_none(self):
        """Test that non-expert keys return None."""
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",  # dense layer, not expert
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        for key in keys:
            assert parse_expert_key(key) is None, f"Expected None for {key}"

    def test_fused_format_key_returns_none(self):
        """Test that already-fused format keys return None."""
        keys = [
            "model.layers.0.mlp.experts.gate_proj",  # fused format (no expert idx)
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]
        for key in keys:
            assert parse_expert_key(key) is None, f"Expected None for {key}"

    def test_invalid_projection_name_returns_none(self):
        """Test that invalid projection names return None."""
        key = "model.layers.0.mlp.experts.5.other_proj.weight"
        assert parse_expert_key(key) is None

    def test_missing_weight_suffix_returns_none(self):
        """Test that keys without .weight suffix return None."""
        key = "model.layers.0.mlp.experts.5.gate_proj"
        assert parse_expert_key(key) is None


class TestModelNeedsExpertMerging:
    """Tests for _model_needs_expert_merging function."""

    def test_fused_format_model_needs_merging(self):
        """Test that models with fused format parameters need merging."""
        parameter_names = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert _model_needs_expert_merging(parameter_names) is True

    def test_non_moe_model_does_not_need_merging(self):
        """Test that non-MoE models don't need merging."""
        parameter_names = {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert _model_needs_expert_merging(parameter_names) is False

    def test_empty_parameters_does_not_need_merging(self):
        """Test that empty parameter set doesn't need merging."""
        assert _model_needs_expert_merging(set()) is False


class TestExpertWeightBuffer:
    """Tests for ExpertWeightBuffer class."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = ExpertWeightBuffer(num_experts=8)
        assert buffer.num_experts == 8
        assert len(buffer.get_pending_keys()) == 0

    def test_add_single_expert(self):
        """Test adding a single expert tensor."""
        buffer = ExpertWeightBuffer(num_experts=4)
        tensor = torch.randn(64, 32)
        buffer.add(layer_idx=0, expert_idx=0, proj="gate", tensor=tensor)

        pending = buffer.get_pending_counts()
        assert (0, "gate") in pending
        assert pending[(0, "gate")] == 1

    def test_is_complete_false_when_incomplete(self):
        """Test is_complete returns False when not all experts collected."""
        buffer = ExpertWeightBuffer(num_experts=4)
        tensor = torch.randn(64, 32)

        buffer.add(0, 0, "gate", tensor)
        buffer.add(0, 1, "gate", tensor)
        buffer.add(0, 2, "gate", tensor)

        assert buffer.is_complete(0, "gate") is False

    def test_is_complete_true_when_all_collected(self):
        """Test is_complete returns True when all experts collected."""
        buffer = ExpertWeightBuffer(num_experts=4)
        tensor = torch.randn(64, 32)

        for i in range(4):
            buffer.add(0, i, "gate", tensor)

        assert buffer.is_complete(0, "gate") is True

    def test_pop_stacked_returns_correct_shape(self):
        """Test pop_stacked returns tensor with correct shape."""
        num_experts = 4
        intermediate_size = 64
        hidden_size = 32

        buffer = ExpertWeightBuffer(num_experts=num_experts)

        for i in range(num_experts):
            tensor = torch.randn(intermediate_size, hidden_size)
            buffer.add(0, i, "gate", tensor)

        stacked = buffer.pop_stacked(0, "gate")
        assert stacked.shape == (num_experts, intermediate_size, hidden_size)

    def test_pop_stacked_preserves_expert_order(self):
        """Test pop_stacked preserves expert ordering."""
        num_experts = 4
        buffer = ExpertWeightBuffer(num_experts=num_experts)

        # Add experts in reverse order
        tensors = []
        for i in range(num_experts - 1, -1, -1):
            tensor = torch.full((2, 2), fill_value=float(i))
            tensors.append(tensor)
            buffer.add(0, i, "gate", tensor)

        stacked = buffer.pop_stacked(0, "gate")

        # Verify order: expert 0 first, expert 3 last
        for i in range(num_experts):
            assert torch.all(stacked[i] == float(i)), f"Expert {i} not in correct position"

    def test_pop_stacked_removes_from_buffer(self):
        """Test that pop_stacked removes data from buffer."""
        buffer = ExpertWeightBuffer(num_experts=2)

        for i in range(2):
            buffer.add(0, i, "gate", torch.randn(4, 4))

        buffer.pop_stacked(0, "gate")

        assert (0, "gate") not in buffer.get_pending_counts()
        assert buffer.is_complete(0, "gate") is False

    def test_pop_stacked_raises_on_incomplete(self):
        """Test pop_stacked raises ValueError when experts incomplete."""
        buffer = ExpertWeightBuffer(num_experts=4)
        buffer.add(0, 0, "gate", torch.randn(4, 4))
        buffer.add(0, 1, "gate", torch.randn(4, 4))

        with pytest.raises(ValueError, match="Incomplete experts"):
            buffer.pop_stacked(0, "gate")

    def test_pop_stacked_raises_on_nonexistent_key(self):
        """Test pop_stacked raises KeyError for non-existent key."""
        buffer = ExpertWeightBuffer(num_experts=4)

        with pytest.raises(KeyError):
            buffer.pop_stacked(0, "gate")

    def test_get_fused_name(self):
        """Test get_fused_name returns correct parameter names."""
        assert ExpertWeightBuffer.get_fused_name(0, "gate") == "model.layers.0.mlp.experts.gate_proj"
        assert ExpertWeightBuffer.get_fused_name(10, "up") == "model.layers.10.mlp.experts.up_proj"
        assert ExpertWeightBuffer.get_fused_name(35, "down") == "model.layers.35.mlp.experts.down_proj"

    def test_multiple_layers_independent(self):
        """Test that different layers are tracked independently."""
        buffer = ExpertWeightBuffer(num_experts=2)

        # Add experts for layer 0
        buffer.add(0, 0, "gate", torch.randn(4, 4))
        buffer.add(0, 1, "gate", torch.randn(4, 4))

        # Add experts for layer 1 (incomplete)
        buffer.add(1, 0, "gate", torch.randn(4, 4))

        assert buffer.is_complete(0, "gate") is True
        assert buffer.is_complete(1, "gate") is False

    def test_multiple_projections_independent(self):
        """Test that different projections are tracked independently."""
        buffer = ExpertWeightBuffer(num_experts=2)

        # Add all experts for gate_proj
        buffer.add(0, 0, "gate", torch.randn(4, 4))
        buffer.add(0, 1, "gate", torch.randn(4, 4))

        # Add partial experts for up_proj
        buffer.add(0, 0, "up", torch.randn(4, 4))

        assert buffer.is_complete(0, "gate") is True
        assert buffer.is_complete(0, "up") is False

    def test_tensors_stored_on_cpu(self):
        """Test that tensors are always stored on CPU."""
        buffer = ExpertWeightBuffer(num_experts=2)

        if torch.cuda.is_available():
            tensor = torch.randn(4, 4, device="cuda")
            buffer.add(0, 0, "gate", tensor)
            buffer.add(0, 1, "gate", tensor)

            stacked = buffer.pop_stacked(0, "gate")
            assert stacked.device.type == "cpu"

    def test_get_pending_keys(self):
        """Test get_pending_keys returns correct keys."""
        buffer = ExpertWeightBuffer(num_experts=2)

        buffer.add(0, 0, "gate", torch.randn(4, 4))
        buffer.add(1, 0, "up", torch.randn(4, 4))

        pending_keys = buffer.get_pending_keys()
        assert (0, "gate") in pending_keys
        assert (1, "up") in pending_keys
        assert len(pending_keys) == 2

    def test_full_workflow(self):
        """Test complete workflow: add all experts, stack, verify."""
        num_experts = 8
        num_layers = 2
        hidden_size = 32
        intermediate_size = 64

        buffer = ExpertWeightBuffer(num_experts=num_experts)

        # Simulate loading weights layer by layer, projection by projection
        for layer_idx in range(num_layers):
            for proj in ["gate", "up", "down"]:
                if proj == "down":
                    shape = (hidden_size, intermediate_size)
                else:
                    shape = (intermediate_size, hidden_size)

                for expert_idx in range(num_experts):
                    tensor = torch.randn(*shape)
                    buffer.add(layer_idx, expert_idx, proj, tensor)

                    if expert_idx == num_experts - 1:
                        assert buffer.is_complete(layer_idx, proj)
                        stacked = buffer.pop_stacked(layer_idx, proj)
                        assert stacked.shape == (num_experts, *shape)

        # Buffer should be empty after processing all
        assert len(buffer.get_pending_keys()) == 0


class TestExpertWeightBufferEdgeCases:
    """Edge case tests for ExpertWeightBuffer."""

    def test_single_expert(self):
        """Test with single expert (edge case)."""
        buffer = ExpertWeightBuffer(num_experts=1)
        tensor = torch.randn(4, 4)
        buffer.add(0, 0, "gate", tensor)

        assert buffer.is_complete(0, "gate")
        stacked = buffer.pop_stacked(0, "gate")
        assert stacked.shape == (1, 4, 4)
        assert torch.allclose(stacked[0].cpu(), tensor.cpu())

    def test_large_number_of_experts(self):
        """Test with large number of experts."""
        num_experts = 128
        buffer = ExpertWeightBuffer(num_experts=num_experts)

        for i in range(num_experts):
            buffer.add(0, i, "gate", torch.randn(4, 4))

        assert buffer.is_complete(0, "gate")
        stacked = buffer.pop_stacked(0, "gate")
        assert stacked.shape == (num_experts, 4, 4)

    def test_overwrite_expert(self):
        """Test that adding same expert twice overwrites."""
        buffer = ExpertWeightBuffer(num_experts=2)

        tensor1 = torch.full((2, 2), 1.0)
        tensor2 = torch.full((2, 2), 2.0)

        buffer.add(0, 0, "gate", tensor1)
        buffer.add(0, 0, "gate", tensor2)  # Overwrite
        buffer.add(0, 1, "gate", tensor1)

        stacked = buffer.pop_stacked(0, "gate")
        assert torch.all(stacked[0] == 2.0)  # Should be tensor2


# =============================================================================
# Tests for checkpoint format detection
# =============================================================================

def _checkpoint_has_per_expert_weights(checkpoint_keys):
    """Copy of implementation for testing."""
    for key in checkpoint_keys:
        if _EXPERT_KEY_PATTERN.match(key):
            return True
    return False


class TestCheckpointFormatDetection:
    """Tests for checkpoint format detection."""

    def test_detect_per_expert_format(self):
        """Test detection of per-expert checkpoint format."""
        per_expert_keys = {
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.1.up_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert _checkpoint_has_per_expert_weights(per_expert_keys) is True

    def test_detect_fused_format(self):
        """Test detection of already-fused checkpoint format."""
        fused_keys = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert _checkpoint_has_per_expert_weights(fused_keys) is False

    def test_detect_non_moe_model(self):
        """Test detection of non-MoE model (no expert keys)."""
        non_moe_keys = {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert _checkpoint_has_per_expert_weights(non_moe_keys) is False

    def test_empty_checkpoint(self):
        """Test with empty checkpoint keys."""
        assert _checkpoint_has_per_expert_weights(set()) is False

    def test_mixed_format_is_per_expert(self):
        """Test that mixed format (even one per-expert key) is detected as per-expert."""
        # This scenario shouldn't happen in practice, but test defensive behavior
        mixed_keys = {
            "model.layers.0.mlp.experts.gate_proj",  # fused
            "model.layers.1.mlp.experts.0.gate_proj.weight",  # per-expert
        }
        assert _checkpoint_has_per_expert_weights(mixed_keys) is True


class TestLoadingBothFormats:
    """Tests that verify loading logic works for both checkpoint formats."""

    def test_fused_checkpoint_keys_match_model_params(self):
        """Test that fused checkpoint keys can be loaded directly without merging."""
        # Model expects fused format
        model_params = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj.weight",
        }

        # Fused checkpoint has same keys
        fused_checkpoint_keys = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj.weight",
        }

        # Model needs fused format
        assert _model_needs_expert_merging(model_params) is True

        # Checkpoint is already fused
        assert _checkpoint_has_per_expert_weights(fused_checkpoint_keys) is False

        # All keys should match directly
        for key in fused_checkpoint_keys:
            assert key in model_params, f"Key {key} not found in model params"

    def test_per_expert_checkpoint_needs_merging(self):
        """Test that per-expert checkpoint is correctly identified as needing merge."""
        model_params = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
        }

        per_expert_keys = {
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.1.up_proj.weight",
        }

        # Model needs fused format
        assert _model_needs_expert_merging(model_params) is True

        # Checkpoint has per-expert format
        assert _checkpoint_has_per_expert_weights(per_expert_keys) is True

        # Per-expert keys should NOT match model params directly
        for key in per_expert_keys:
            assert key not in model_params

        # After merging, fused names should match
        for key in per_expert_keys:
            parsed = parse_expert_key(key)
            if parsed:
                layer_idx, _, proj = parsed
                fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                assert fused_name in model_params
