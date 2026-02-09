"""Reusable weight merge/split buffers for checkpoint handlers.

These utility classes handle common weight transformation patterns:
- ExpertWeightBuffer: merges per-expert HF weights into fused [num_experts, ...] format
- GateUpMergeBuffer: merges gate_proj + up_proj into gate_up_proj
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch


# =============================================================================
# Regex patterns for checkpoint key matching
# =============================================================================

# Per-expert HuggingFace weight keys
# e.g., model.layers.0.mlp.experts.5.gate_proj.weight
EXPERT_KEY_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)

# Model parameter names in fused expert format
# e.g., model.layers.0.mlp.experts.gate_proj
FUSED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate|up|down)_proj$"
)

# Dense MLP gate/up projection weight keys
# e.g., model.layers.0.mlp.gate_proj.weight or model.layers.0.mlp.shared_expert.up_proj.weight
DENSE_GATE_UP_PATTERN = re.compile(
    r"^(.*)\.(gate|up)_proj\.weight$"
)

# Attention QKV projection weight/bias keys
# e.g., model.layers.0.self_attn.q_proj.weight
QKV_PROJ_PATTERN = re.compile(
    r"^(.*\.self_attn)\.(q|k|v)_proj\.(weight|bias)$"
)


# =============================================================================
# Key parsing and detection helpers
# =============================================================================

def parse_expert_key(key: str) -> Optional[Tuple[int, int, str]]:
    """Parse a per-expert weight key to extract layer index, expert index, and projection name.

    Args:
        key: Weight key like "model.layers.0.mlp.experts.5.gate_proj.weight"

    Returns:
        Tuple of (layer_idx, expert_idx, proj_name) or None if not a per-expert key.
        proj_name is one of "gate", "up", "down".
    """
    match = EXPERT_KEY_PATTERN.match(key)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def checkpoint_has_per_expert_weights(checkpoint_keys: Set[str]) -> bool:
    """Check if checkpoint has per-expert weight format (needs merging)."""
    for key in checkpoint_keys:
        if EXPERT_KEY_PATTERN.match(key):
            return True
    return False


def model_needs_expert_merging(parameter_names: Set[str]) -> bool:
    """Check if the model expects fused expert format (stacked [num_experts, ...] tensors)."""
    for name in parameter_names:
        if FUSED_EXPERT_PATTERN.match(name):
            return True
    return False


def model_has_gate_up_merged(parameter_names: Set[str]) -> bool:
    """Check if the model uses merged gate_up_proj layers."""
    return any(".gate_up_proj." in name for name in parameter_names)


def checkpoint_has_separate_gate_up(checkpoint_keys: Set[str]) -> bool:
    """Check if checkpoint has separate gate_proj/up_proj weights (needs merging into gate_up_proj)."""
    for key in checkpoint_keys:
        if DENSE_GATE_UP_PATTERN.match(key):
            return True
    return False


# =============================================================================
# ExpertWeightBuffer
# =============================================================================

class ExpertWeightBuffer:
    """Buffer for collecting per-expert weights and merging them into stacked tensors.

    Optimized for performance:
    - Pre-allocates stacked tensor on first expert arrival
    - Copies each expert directly into slice as it arrives (streaming)
    - EP-aware: when ep_size > 1, only buffers experts belonging to this EP rank
    """

    def __init__(self, num_experts: int, ep_rank: int = 0, ep_size: int = 1):
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size

        self.local_num_experts = num_experts // ep_size
        self.expert_start = ep_rank * self.local_num_experts
        self.expert_end = self.expert_start + self.local_num_experts

        self._stacked_buffers: Dict[Tuple[int, str], torch.Tensor] = {}
        self._filled_experts: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        self._total_seen: Dict[Tuple[int, str], int] = defaultdict(int)

    def add(self, layer_idx: int, expert_idx: int, proj: str, tensor: torch.Tensor) -> None:
        """Add a per-expert tensor. Experts outside this EP rank's range are counted but not buffered."""
        key = (layer_idx, proj)
        self._total_seen[key] += 1

        if expert_idx < self.expert_start or expert_idx >= self.expert_end:
            return

        if key not in self._stacked_buffers:
            stacked_shape = (self.local_num_experts,) + tensor.shape
            self._stacked_buffers[key] = torch.empty(
                stacked_shape, dtype=tensor.dtype, device="cpu"
            )

        local_idx = expert_idx - self.expert_start
        self._stacked_buffers[key][local_idx].copy_(tensor)
        self._filled_experts[key].add(expert_idx)

    def is_complete(self, layer_idx: int, proj: str) -> bool:
        """Check if all num_experts keys have been seen for this (layer, proj)."""
        key = (layer_idx, proj)
        return self._total_seen.get(key, 0) == self.num_experts

    def pop_stacked(self, layer_idx: int, proj: str) -> torch.Tensor:
        """Return and remove the completed stacked tensor for this EP rank."""
        key = (layer_idx, proj)
        if key not in self._stacked_buffers:
            raise KeyError(f"No buffered experts for layer {layer_idx}, projection {proj}")

        filled = self._filled_experts.pop(key)
        self._total_seen.pop(key, None)
        if len(filled) != self.local_num_experts:
            raise ValueError(
                f"Incomplete experts for layer {layer_idx}, {proj}_proj: "
                f"got {len(filled)}, expected {self.local_num_experts}"
            )

        return self._stacked_buffers.pop(key)

    @staticmethod
    def get_fused_name(layer_idx: int, proj: str) -> str:
        """Get the fused parameter name, e.g., 'model.layers.0.mlp.experts.gate_proj'."""
        return f"model.layers.{layer_idx}.mlp.experts.{proj}_proj"

    def get_pending_keys(self) -> List[Tuple[int, str]]:
        """Get list of (layer_idx, proj) combinations that have partial data."""
        return list(self._stacked_buffers.keys())

    def get_pending_counts(self) -> Dict[Tuple[int, str], int]:
        """Get counts of collected experts for each pending (layer, proj)."""
        return {key: len(experts) for key, experts in self._filled_experts.items()}


# =============================================================================
# GateUpMergeBuffer
# =============================================================================

class GateUpMergeBuffer:
    """Buffer for merging separate gate_proj and up_proj checkpoint weights into gate_up_proj.

    When a model uses a merged gate_up_proj linear layer but the checkpoint has
    separate gate_proj and up_proj weights, this buffer collects both and concatenates
    them into the merged format.
    """

    def __init__(self):
        self._pending: Dict[str, Dict[str, torch.Tensor]] = {}

    def add(self, key: str, tensor: torch.Tensor) -> Optional[Tuple[str, torch.Tensor]]:
        """Try to add a gate/up weight. Returns (merged_key, merged_tensor) when both are available."""
        match = DENSE_GATE_UP_PATTERN.match(key)
        if match is None:
            return None

        prefix = match.group(1)
        proj_type = match.group(2)  # "gate" or "up"

        if prefix not in self._pending:
            self._pending[prefix] = {}
        self._pending[prefix][proj_type] = tensor

        if "gate" in self._pending[prefix] and "up" in self._pending[prefix]:
            gate_tensor = self._pending[prefix]["gate"]
            up_tensor = self._pending[prefix]["up"]
            del self._pending[prefix]
            merged = torch.cat([gate_tensor, up_tensor], dim=0)
            return f"{prefix}.gate_up_proj.weight", merged

        return None

    def is_gate_up_key(self, key: str) -> bool:
        """Check if a key matches the gate/up pattern."""
        return DENSE_GATE_UP_PATTERN.match(key) is not None

    def get_pending(self) -> Dict[str, List[str]]:
        """Get pending (incomplete) merge pairs for debugging."""
        return {prefix: list(projs.keys()) for prefix, projs in self._pending.items()}


# =============================================================================
# QKVMergeBuffer
# =============================================================================

class QKVMergeBuffer:
    """Buffer for merging separate q_proj, k_proj, v_proj checkpoint weights into qkv_proj.

    When a model uses a merged qkv_proj linear layer but the checkpoint has
    separate q_proj, k_proj, v_proj weights, this buffer collects all three
    and concatenates them into the merged format (q, k, v along dim=0).
    Handles both weight and bias parameters.
    """

    def __init__(self):
        # _pending: {(prefix, param_type): {"q": tensor, "k": tensor, "v": tensor}}
        self._pending: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

    def add(self, key: str, tensor: torch.Tensor) -> Optional[Tuple[str, torch.Tensor]]:
        """Try to add a q/k/v weight or bias. Returns merged result when all three are available."""
        match = QKV_PROJ_PATTERN.match(key)
        if match is None:
            return None

        prefix = match.group(1)       # e.g., "model.layers.0.self_attn"
        proj_type = match.group(2)    # "q", "k", or "v"
        param_type = match.group(3)   # "weight" or "bias"

        buf_key = (prefix, param_type)
        if buf_key not in self._pending:
            self._pending[buf_key] = {}
        self._pending[buf_key][proj_type] = tensor

        if all(p in self._pending[buf_key] for p in ("q", "k", "v")):
            parts = self._pending.pop(buf_key)
            merged = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            return f"{prefix}.qkv_proj.{param_type}", merged

        return None

    def is_qkv_key(self, key: str) -> bool:
        """Check if a key matches the q/k/v projection pattern."""
        return QKV_PROJ_PATTERN.match(key) is not None

    def get_pending(self) -> Dict[Tuple[str, str], List[str]]:
        """Get pending (incomplete) merge groups for debugging."""
        return {key: list(projs.keys()) for key, projs in self._pending.items()}
