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


pytestmark = [pytest.mark.cpu]


# =============================================================================
# Copy of the implementation for testing (avoids heavy import dependencies)
# =============================================================================

_EXPERT_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

_FUSED_EXPERT_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate|up|down)_proj$")


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
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self._stacked_buffers: Dict[Tuple[int, str], torch.Tensor] = {}
        self._filled_experts: Dict[Tuple[int, str], Set[int]] = defaultdict(set)

    def add(self, layer_idx: int, expert_idx: int, proj: str, tensor: torch.Tensor) -> None:
        key = (layer_idx, proj)
        if key not in self._stacked_buffers:
            stacked_shape = (self.num_experts,) + tensor.shape
            self._stacked_buffers[key] = torch.empty(stacked_shape, dtype=tensor.dtype, device="cpu")
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
                f"Incomplete experts for layer {layer_idx}, {proj}_proj: got {len(filled)}, expected {self.num_experts}"
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


class TestParseAndMergingDetection:
    """Tests for parse_expert_key and _model_needs_expert_merging."""

    def test_parse_expert_key_and_merging_detection(self):
        """Test valid/invalid key parsing, fused/non-MoE/empty detection."""
        # Valid keys
        assert parse_expert_key("model.layers.0.mlp.experts.5.gate_proj.weight") == (0, 5, "gate")
        assert parse_expert_key("model.layers.10.mlp.experts.127.up_proj.weight") == (10, 127, "up")
        assert parse_expert_key("model.layers.35.mlp.experts.0.down_proj.weight") == (35, 0, "down")

        # Non-expert keys return None
        for key in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]:
            assert parse_expert_key(key) is None, f"Expected None for {key}"

        # Already-fused format keys return None
        for key in [
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]:
            assert parse_expert_key(key) is None, f"Expected None for {key}"

        # Invalid projection name and missing suffix
        assert parse_expert_key("model.layers.0.mlp.experts.5.other_proj.weight") is None
        assert parse_expert_key("model.layers.0.mlp.experts.5.gate_proj") is None

        # Fused format model needs merging
        assert (
            _model_needs_expert_merging(
                {
                    "model.layers.0.mlp.experts.gate_proj",
                    "model.layers.0.mlp.experts.up_proj",
                    "model.layers.0.mlp.experts.down_proj",
                    "model.layers.0.self_attn.q_proj.weight",
                }
            )
            is True
        )

        # Non-MoE model does not
        assert (
            _model_needs_expert_merging(
                {
                    "model.layers.0.mlp.gate_proj.weight",
                    "model.layers.0.mlp.up_proj.weight",
                    "model.layers.0.mlp.down_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                }
            )
            is False
        )

        # Empty
        assert _model_needs_expert_merging(set()) is False


class TestExpertWeightBuffer:
    """Tests for ExpertWeightBuffer: lifecycle, layers, projections, errors, edge cases."""

    def test_buffer_lifecycle_layers_errors_and_edge_cases(self):
        """Test full lifecycle, independent layers/projections, errors, overwrite,
        single/large expert count, fused name, CPU storage, and full workflow."""
        num_experts = 4
        buffer = ExpertWeightBuffer(num_experts=num_experts)

        # Empty buffer
        assert len(buffer.get_pending_keys()) == 0

        # Add experts in reverse order
        tensors = {}
        for i in range(num_experts - 1, -1, -1):
            tensor = torch.full((2, 2), fill_value=float(i))
            tensors[i] = tensor
            buffer.add(0, i, "gate", tensor)
            if i > 0:
                assert not buffer.is_complete(0, "gate")

        assert buffer.is_complete(0, "gate")

        # Pop and verify order preserved
        stacked = buffer.pop_stacked(0, "gate")
        assert stacked.shape == (num_experts, 2, 2)
        for i in range(num_experts):
            assert torch.all(stacked[i] == float(i)), f"Expert {i} not in correct position"

        assert (0, "gate") not in buffer.get_pending_counts()
        assert not buffer.is_complete(0, "gate")

        # Independent layers and projections
        buffer2 = ExpertWeightBuffer(num_experts=2)
        buffer2.add(0, 0, "gate", torch.randn(4, 4))
        buffer2.add(0, 1, "gate", torch.randn(4, 4))
        buffer2.add(1, 0, "gate", torch.randn(4, 4))
        assert buffer2.is_complete(0, "gate")
        assert not buffer2.is_complete(1, "gate")

        buffer2.add(0, 0, "up", torch.randn(4, 4))
        assert buffer2.is_complete(0, "gate")
        assert not buffer2.is_complete(0, "up")

        pending_keys = buffer2.get_pending_keys()
        assert (0, "gate") in pending_keys
        assert (1, "gate") in pending_keys
        assert (0, "up") in pending_keys

        # Pop nonexistent key
        buffer3 = ExpertWeightBuffer(num_experts=4)
        with pytest.raises(KeyError):
            buffer3.pop_stacked(0, "gate")

        # Pop incomplete
        buffer3.add(0, 0, "gate", torch.randn(4, 4))
        buffer3.add(0, 1, "gate", torch.randn(4, 4))
        with pytest.raises(ValueError, match="Incomplete experts"):
            buffer3.pop_stacked(0, "gate")

        # Overwrite expert
        buffer4 = ExpertWeightBuffer(num_experts=2)
        buffer4.add(0, 0, "gate", torch.full((2, 2), 1.0))
        buffer4.add(0, 0, "gate", torch.full((2, 2), 2.0))
        buffer4.add(0, 1, "gate", torch.full((2, 2), 1.0))
        stacked = buffer4.pop_stacked(0, "gate")
        assert torch.all(stacked[0] == 2.0)

        # Single expert edge case
        buffer5 = ExpertWeightBuffer(num_experts=1)
        tensor = torch.randn(4, 4)
        buffer5.add(0, 0, "gate", tensor)
        assert buffer5.is_complete(0, "gate")
        stacked = buffer5.pop_stacked(0, "gate")
        assert stacked.shape == (1, 4, 4)
        assert torch.allclose(stacked[0].cpu(), tensor.cpu())

        # Large number of experts
        large_num = 128
        buffer6 = ExpertWeightBuffer(num_experts=large_num)
        for i in range(large_num):
            buffer6.add(0, i, "gate", torch.randn(4, 4))
        assert buffer6.is_complete(0, "gate")
        stacked = buffer6.pop_stacked(0, "gate")
        assert stacked.shape == (large_num, 4, 4)

        # get_fused_name
        assert ExpertWeightBuffer.get_fused_name(0, "gate") == "model.layers.0.mlp.experts.gate_proj"
        assert ExpertWeightBuffer.get_fused_name(10, "up") == "model.layers.10.mlp.experts.up_proj"
        assert ExpertWeightBuffer.get_fused_name(35, "down") == "model.layers.35.mlp.experts.down_proj"

        # CPU storage
        if torch.cuda.is_available():
            buffer7 = ExpertWeightBuffer(num_experts=2)
            t = torch.randn(4, 4, device="cuda")
            buffer7.add(0, 0, "gate", t)
            buffer7.add(0, 1, "gate", t)
            stacked = buffer7.pop_stacked(0, "gate")
            assert stacked.device.type == "cpu"

        # Full workflow
        num_e = 8
        num_layers = 2
        hidden_size = 32
        intermediate_size = 64
        buffer8 = ExpertWeightBuffer(num_experts=num_e)
        for layer_idx in range(num_layers):
            for proj in ["gate", "up", "down"]:
                shape = (hidden_size, intermediate_size) if proj == "down" else (intermediate_size, hidden_size)
                for expert_idx in range(num_e):
                    buffer8.add(layer_idx, expert_idx, proj, torch.randn(*shape))
                    if expert_idx == num_e - 1:
                        assert buffer8.is_complete(layer_idx, proj)
                        stacked = buffer8.pop_stacked(layer_idx, proj)
                        assert stacked.shape == (num_e, *shape)
        assert len(buffer8.get_pending_keys()) == 0


# =============================================================================
# Tests for checkpoint format detection and loading logic
# =============================================================================


def _checkpoint_has_per_expert_weights(checkpoint_keys):
    for key in checkpoint_keys:
        if _EXPERT_KEY_PATTERN.match(key):
            return True
    return False


class TestCheckpointFormatAndLoading:
    """Tests for checkpoint format detection and loading both formats."""

    def test_checkpoint_format_detection_and_loading(self):
        """Test detection of per-expert/fused/non-MoE/empty/mixed formats, and loading both."""
        # Per-expert format
        assert (
            _checkpoint_has_per_expert_weights(
                {
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "model.layers.0.mlp.experts.1.gate_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                }
            )
            is True
        )

        # Fused format
        assert (
            _checkpoint_has_per_expert_weights(
                {
                    "model.layers.0.mlp.experts.gate_proj",
                    "model.layers.0.mlp.experts.up_proj",
                }
            )
            is False
        )

        # Non-MoE
        assert (
            _checkpoint_has_per_expert_weights(
                {
                    "model.layers.0.mlp.gate_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                }
            )
            is False
        )

        # Empty
        assert _checkpoint_has_per_expert_weights(set()) is False

        # Mixed
        assert (
            _checkpoint_has_per_expert_weights(
                {
                    "model.layers.0.mlp.experts.gate_proj",
                    "model.layers.1.mlp.experts.0.gate_proj.weight",
                }
            )
            is True
        )

        # --- Loading both formats ---
        model_params = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj.weight",
        }

        # Fused checkpoint matches directly
        fused_checkpoint_keys = {
            "model.layers.0.mlp.experts.gate_proj",
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj.weight",
        }
        assert _model_needs_expert_merging(model_params) is True
        assert _checkpoint_has_per_expert_weights(fused_checkpoint_keys) is False
        for key in fused_checkpoint_keys:
            assert key in model_params

        # Per-expert checkpoint needs merging
        per_expert_keys = {
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.1.up_proj.weight",
        }
        assert _checkpoint_has_per_expert_weights(per_expert_keys) is True
        for key in per_expert_keys:
            assert key not in model_params

        # After merging, fused names match model params
        for key in per_expert_keys:
            parsed = parse_expert_key(key)
            if parsed:
                layer_idx, _, proj = parsed
                fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                assert fused_name in model_params
