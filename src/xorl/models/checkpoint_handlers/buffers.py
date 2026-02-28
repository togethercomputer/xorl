"""Reusable weight merge/split buffers for checkpoint handlers.

These utility classes handle common weight transformation patterns:
- ExpertWeightBuffer: merges per-expert HF weights into fused [num_experts, ...] format
- GateUpMergeBuffer: merges gate_proj + up_proj into gate_up_proj
"""

import json
import os
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

# Per-expert quantized auxiliary keys (NVFP4 / modelopt format)
# e.g., model.layers.0.mlp.experts.5.gate_proj.weight_scale
#       model.layers.0.mlp.experts.5.gate_proj.weight_scale_2
#       model.layers.0.mlp.experts.5.gate_proj.input_scale
EXPERT_QUANT_AUX_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\."
    r"(weight_scale|weight_scale_2|input_scale)$"
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

# Attention o_proj weight key
# e.g., model.layers.0.self_attn.o_proj.weight
OPROJ_WEIGHT_PATTERN = re.compile(
    r"^.*\.self_attn\.o_proj\.weight$"
)

# Dense/shared-expert down_proj weight key (non-expert)
# e.g., model.layers.0.mlp.down_proj.weight or model.layers.0.mlp.shared_expert.down_proj.weight
DENSE_DOWN_PROJ_PATTERN = re.compile(
    r"^.*\.mlp\.(?:shared_expert\.)?down_proj\.weight$"
)

# Attention QKV projection weight/bias keys
# e.g., model.layers.0.self_attn.q_proj.weight
QKV_PROJ_PATTERN = re.compile(
    r"^(.*\.self_attn)\.(q|k|v)_proj\.(weight|bias)$"
)

# Quantized auxiliary suffixes used by modelopt NVFP4 checkpoints.
# Matches any key ending in .weight_scale, .weight_scale_2, or .input_scale
QUANT_AUX_SUFFIX_PATTERN = re.compile(
    r"\.(weight_scale|weight_scale_2|input_scale)$"
)

# Block FP8 auxiliary suffix (HuggingFace FP8 checkpoint format)
# Matches any key ending in .weight_scale_inv
FP8_AUX_SUFFIX_PATTERN = re.compile(r"\.weight_scale_inv$")


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


def _resolve_weights_path(weights_path: Optional[str]) -> Optional[str]:
    """Resolve a HF hub model ID or local path to a local directory.

    If ``weights_path`` is already a local directory, returns it as-is.
    Otherwise, tries to resolve it as a HF hub model ID via the local cache
    (using ``cached_file`` to locate ``config.json``).

    Returns:
        Local directory path, or None if resolution fails.
    """
    if not weights_path:
        return None
    if os.path.isdir(weights_path):
        return weights_path
    # Try resolving as a HF hub model ID
    try:
        from transformers.utils import cached_file
        config_path = cached_file(weights_path, "config.json", _raise_exceptions_for_missing_entries=False)
        if config_path and os.path.isfile(config_path):
            return os.path.dirname(config_path)
    except Exception:
        pass
    return None


def detect_prequantized_checkpoint(weights_path: Optional[str]) -> bool:
    """Detect whether a checkpoint contains pre-quantized weights (NVFP4 modelopt format).

    Checks multiple signals in order:
    1. ``hf_quant_config.json`` with ``quant_algo == "NVFP4"``
    2. ``quantization_config.quant_algo == "NVFP4"`` in ``config.json``
    3. Presence of ``weight_scale`` keys in safetensors index (some NVFP4
       checkpoints, e.g. ``nvidia/Qwen3-235B-A22B-NVFP4``, omit the config
       metadata but still store weights in NVFP4 packed format)

    Args:
        weights_path: Path to HF model directory or HF hub model ID.
            Returns False if None or unresolvable.
    """
    weights_path = _resolve_weights_path(weights_path)
    if weights_path is None:
        return False

    # Check hf_quant_config.json
    # nvidia/modelopt format nests under "quantization": {"quant_algo": "NVFP4", ...}
    quant_config_path = os.path.join(weights_path, "hf_quant_config.json")
    if os.path.isfile(quant_config_path):
        try:
            with open(quant_config_path) as f:
                quant_config = json.load(f)
            algo = (
                quant_config.get("quantization", {}).get("quant_algo")
                or quant_config.get("quant_algo")
            )
            if algo == "NVFP4":
                return True
        except (json.JSONDecodeError, OSError):
            pass

    # Check config.json quantization_config
    config_path = os.path.join(weights_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("quantization_config", {}).get("quant_algo") == "NVFP4":
                return True
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: check safetensors index for weight_scale keys (NVFP4 indicator)
    index_path = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            has_weight_scale = any(k.endswith(".weight_scale") for k in weight_map)
            has_weight_scale_2 = any(k.endswith(".weight_scale_2") for k in weight_map)
            if has_weight_scale and has_weight_scale_2:
                return True
        except (json.JSONDecodeError, OSError):
            pass

    return False


def detect_prequantized_block_fp8_checkpoint(weights_path: Optional[str]) -> bool:
    """Detect whether a checkpoint contains pre-quantized block FP8 weights.

    Checks multiple signals:
    1. ``quantization_config`` in ``config.json`` with ``quant_method == "fp8"``
       and ``weight_block_size == [128, 128]``
    2. Presence of ``weight_scale_inv`` keys in safetensors index (and NO
       ``weight_scale`` keys to distinguish from NVFP4)

    Args:
        weights_path: Path to HF model directory or HF hub model ID.
            Returns False if None or unresolvable.
    """
    weights_path = _resolve_weights_path(weights_path)
    if weights_path is None:
        return False

    # Check config.json quantization_config
    config_path = os.path.join(weights_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            qc = config.get("quantization_config", {})
            if (qc.get("quant_method") == "fp8"
                    and qc.get("weight_block_size") == [128, 128]):
                return True
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: check safetensors index for weight_scale_inv keys
    index_path = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            has_weight_scale_inv = any(k.endswith(".weight_scale_inv") for k in weight_map)
            # Distinguish from NVFP4: NVFP4 has .weight_scale, FP8 has .weight_scale_inv
            has_weight_scale = any(k.endswith(".weight_scale") for k in weight_map)
            if has_weight_scale_inv and not has_weight_scale:
                return True
        except (json.JSONDecodeError, OSError):
            pass

    return False


def get_prequantized_exclude_modules(weights_path: Optional[str]) -> Set[str]:
    """Read exclude_modules from pre-quantized checkpoint quantization config.

    Pre-quantized checkpoints (e.g., nvidia/modelopt NVFP4) may list modules
    that were NOT quantized — their weights remain in bf16/fp16. This function
    reads that list so callers can skip QLoRA injection and checkpoint key
    skipping for those modules.

    Checks (in order):
    1. ``hf_quant_config.json`` — modelopt nested format
       ``{"quantization": {"exclude_modules": [...]}}`` or flat
       ``{"exclude_modules": [...]}``
    2. ``config.json`` — HuggingFace format
       ``{"quantization_config": {"exclude_modules": [...]}}``
       Also checks ``modules_to_not_convert`` (used by some HF FP8 checkpoints).

    Args:
        weights_path: Path to HF model directory or HF hub model ID.

    Returns:
        Set of module name suffixes (e.g., ``{"lm_head", "gate"}``).
        Empty set if none found.
    """
    weights_path = _resolve_weights_path(weights_path)
    if weights_path is None:
        return set()

    # 1. hf_quant_config.json (modelopt nested or flat format)
    quant_config_path = os.path.join(weights_path, "hf_quant_config.json")
    if os.path.isfile(quant_config_path):
        try:
            with open(quant_config_path) as f:
                qc = json.load(f)
            exclude = (
                qc.get("quantization", {}).get("exclude_modules")
                or qc.get("exclude_modules")
            )
            if exclude:
                return set(exclude)
        except (json.JSONDecodeError, OSError):
            pass

    # 2. config.json quantization_config
    config_path = os.path.join(weights_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            qc = config.get("quantization_config", {})
            exclude = qc.get("exclude_modules") or qc.get("modules_to_not_convert")
            if exclude:
                return set(exclude)
        except (json.JSONDecodeError, OSError):
            pass

    return set()


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
    """Buffer for collecting per-expert weights and merging them into stacked (G,K,N) tensors.

    HuggingFace checkpoints store nn.Linear weights as [out_features, in_features].
    This buffer transposes each expert weight to [in_features, out_features] during
    stacking, producing the (G, K, N) format expected by the MoE expert kernels.

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
        """Add a per-expert tensor. Experts outside this EP rank's range are counted but not buffered.

        Each expert weight is transposed from HF [out_features, in_features] to
        [in_features, out_features] for (G, K, N) format.
        """
        key = (layer_idx, proj)
        self._total_seen[key] += 1

        if expert_idx < self.expert_start or expert_idx >= self.expert_end:
            return

        # Transpose from HF nn.Linear [out, in] to (K, N) = [in, out] format
        tensor = tensor.t().contiguous()

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

    def count_skipped(self, layer_idx: int, proj: str) -> None:
        """Count an expert key as seen without buffering tensor data.

        Used by EP-aware filtered loading: out-of-range expert tensors are not
        read from disk, but the handler still needs to know how many expert keys
        have been seen so ``is_complete`` triggers correctly.
        """
        key = (layer_idx, proj)
        self._total_seen[key] += 1

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
