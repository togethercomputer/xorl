"""Reusable weight merge/split buffers for checkpoint handlers.

These utility classes handle common weight transformation patterns:
- ExpertWeightBuffer: merges per-expert HF weights into fused [num_experts, ...] format
- GateUpMergeBuffer: merges gate_proj + up_proj into gate_up_proj
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.distributed._tensor import DTensor
from transformers.utils import cached_file

from xorl.ops.quantize import (
    block_fp8_dequantize_gkn,
    block_fp8_quantize_gkn,
    nvfp4_dequantize,
    nvfp4_quantize,
)
from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX
from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.moe_experts import (
    BlockFP8QLoRAMoeExperts,
    QLoRAMoeExperts,
)


# =============================================================================
# Regex patterns for checkpoint key matching
# =============================================================================

# Per-expert HuggingFace weight keys
# e.g., model.layers.0.mlp.experts.5.gate_proj.weight
EXPERT_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

# Per-expert quantized auxiliary keys (NVFP4 / modelopt format)
# e.g., model.layers.0.mlp.experts.5.gate_proj.weight_scale
#       model.layers.0.mlp.experts.5.gate_proj.weight_scale_2
#       model.layers.0.mlp.experts.5.gate_proj.input_scale
EXPERT_QUANT_AUX_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\."
    r"(weight_scale|weight_scale_2|input_scale)$"
)

# Generic expert key: matches any suffix (weight, weight_scale, weight_scale_inv, etc.)
_EXPERT_ANY_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(.+)$")

# Model parameter names in fused expert format
# e.g., model.layers.0.mlp.experts.gate_up_proj
FUSED_EXPERT_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up|gate|up|down)_proj$")

# Dense MLP gate/up projection weight keys
# e.g., model.layers.0.mlp.gate_proj.weight or model.layers.0.mlp.shared_expert.up_proj.weight
DENSE_GATE_UP_PATTERN = re.compile(r"^(.*)\.(gate|up)_proj\.weight$")

# Attention o_proj weight key
# e.g., model.layers.0.self_attn.o_proj.weight
OPROJ_WEIGHT_PATTERN = re.compile(r"^.*\.self_attn\.o_proj\.weight$")

# Dense/shared-expert down_proj weight key (non-expert)
# e.g., model.layers.0.mlp.down_proj.weight or model.layers.0.mlp.shared_expert.down_proj.weight
DENSE_DOWN_PROJ_PATTERN = re.compile(r"^.*\.mlp\.(?:shared_expert\.)?down_proj\.weight$")

# Attention QKV projection weight/bias keys
# e.g., model.layers.0.self_attn.q_proj.weight
QKV_PROJ_PATTERN = re.compile(r"^(.*\.self_attn)\.(q|k|v)_proj\.(weight|bias)$")

# Quantized auxiliary suffixes used by modelopt NVFP4 checkpoints.
# Matches any key ending in .weight_scale, .weight_scale_2, or .input_scale
QUANT_AUX_SUFFIX_PATTERN = re.compile(r"\.(weight_scale|weight_scale_2|input_scale)$")

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


def parse_expert_full_key(key: str) -> Optional[Tuple[int, int, str, str]]:
    """Parse any expert key → (layer_idx, expert_idx, proj, suffix) or None.

    Unlike ``parse_expert_key`` which only matches ``.weight``, this matches
    any suffix (weight, weight_scale, weight_scale_inv, etc.).
    """
    match = _EXPERT_ANY_KEY_PATTERN.match(key)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)
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
            algo = quant_config.get("quantization", {}).get("quant_algo") or quant_config.get("quant_algo")
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
            if qc.get("quant_method") == "fp8" and qc.get("weight_block_size") == [128, 128]:
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
            exclude = qc.get("quantization", {}).get("exclude_modules") or qc.get("exclude_modules")
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
            exclude = qc.get("exclude_modules") or qc.get("modules_to_not_convert") or qc.get("ignore")
            if exclude:
                # Normalize full FQNs (e.g. "model.layers.0.mlp.gate")
                # to short suffixes (e.g. "gate", "lm_head")
                exclude = {name.split(".")[-1] for name in exclude}
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
    return any(".gate_up_proj." in name or name.endswith(".gate_up_proj") for name in parameter_names)


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
    - When ``device`` is a CUDA device, stacking happens on GPU: each expert is
      pin+DMA'd to GPU, transposed on GPU, and stacked in a GPU buffer. This is
      ~13x faster than CPU transpose for large MoE models (e.g. 30B).
    """

    def __init__(
        self,
        num_experts: int,
        ep_rank: int = 0,
        ep_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self._device = device

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

        # GPU fast path: pin+DMA to GPU, transpose on GPU (~13x faster than CPU)
        if self._device is not None and self._device.type == "cuda":
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(dtype=torch.bfloat16)
            tensor = tensor.pin_memory().to(device=self._device, non_blocking=True)
            tensor = tensor.t().contiguous()

            if key not in self._stacked_buffers:
                stacked_shape = (self.local_num_experts,) + tensor.shape
                self._stacked_buffers[key] = torch.empty(stacked_shape, dtype=tensor.dtype, device=self._device)
        else:
            # CPU path: transpose on CPU
            tensor = tensor.t().contiguous()

            if key not in self._stacked_buffers:
                stacked_shape = (self.local_num_experts,) + tensor.shape
                self._stacked_buffers[key] = torch.empty(stacked_shape, dtype=tensor.dtype, device="cpu")

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

    @staticmethod
    def get_gate_up_name(layer_idx: int) -> str:
        """Get the fused gate/up parameter name."""
        return f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"

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

        prefix = match.group(1)  # e.g., "model.layers.0.self_attn"
        proj_type = match.group(2)  # "q", "k", or "v"
        param_type = match.group(3)  # "weight" or "bias"

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


# =============================================================================
# QLoRAWeightBuffer — inline quantized weight loading
# =============================================================================

# Known checkpoint suffixes per quantization format
_QLORA_SUFFIXES_NVFP4 = frozenset({"weight", "weight_scale", "weight_scale_2"})
_QLORA_SUFFIXES_FP8 = frozenset({"weight", "weight_scale_inv"})
_QLORA_ALL_SUFFIXES = _QLORA_SUFFIXES_NVFP4 | _QLORA_SUFFIXES_FP8 | {"input_scale"}


class _SourceInfo:
    """Mapping from a checkpoint key prefix to a target module."""

    __slots__ = ("module_fqn", "src_proj")

    def __init__(self, module_fqn: str, src_proj: Optional[str]):
        self.module_fqn = module_fqn
        self.src_proj = src_proj


class _ModuleAccum:
    """Accumulator for collecting quantized weight pieces for one QLoRALinear module."""

    __slots__ = (
        "module",
        "target_format",
        "target_group_size",
        "expected_pieces",
        "merge_sources",
        "received",
        "received_count",
        "ema_amax",
        "scale_dtypes",
    )

    def __init__(
        self,
        module: "torch.nn.Module",
        target_format: str,
        target_group_size: int,
        expected_pieces: int,
        merge_sources: Optional[Tuple[str, ...]],
    ):
        self.module = module
        self.target_format = target_format
        self.target_group_size = target_group_size
        self.expected_pieces = expected_pieces
        self.merge_sources = merge_sources
        self.received: Dict[Tuple[str, str], torch.Tensor] = {}
        self.received_count = 0
        self.ema_amax: Optional[torch.Tensor] = None
        self.scale_dtypes: Optional[Dict[str, torch.dtype]] = None


class _QLoRAWeightBufferBase:
    """Base class for format-specific inline QLoRA weight buffers.

    Routes quantized checkpoint keys to the appropriate QLoRALinear modules,
    handles merged module assembly (qkv_proj, gate_up_proj), and emits packed
    dispatch pairs compatible with _dispatch_parameter / _dispatch_buffer.

    Subclasses implement ``_suffixes`` and ``_pack()``.
    """

    # Subclasses override: the set of checkpoint key suffixes this format owns.
    _suffixes: frozenset

    def __init__(self, model: "torch.nn.Module"):
        self._prefix_map: Dict[str, _SourceInfo] = {}
        self._accums: Dict[str, _ModuleAccum] = {}

        keys_per_source = len(self._suffixes)

        for fqn, module in model.named_modules():
            if not isinstance(module, QLoRALinear) or not module._is_prequantized:
                continue

            merge_sources = module._merge_sources
            num_sources = len(merge_sources) if merge_sources is not None else 1

            if merge_sources is not None:
                for src_proj in merge_sources:
                    prefix = f"{module._source_fqn}.{src_proj}"
                    self._prefix_map[prefix] = _SourceInfo(
                        module_fqn=fqn,
                        src_proj=src_proj,
                    )
            else:
                self._prefix_map[module._source_fqn] = _SourceInfo(
                    module_fqn=fqn,
                    src_proj=None,
                )

            self._accums[fqn] = _ModuleAccum(
                module=module,
                target_format=module.quant_format,
                target_group_size=module.quant_group_size,
                expected_pieces=num_sources * keys_per_source,
                merge_sources=merge_sources,
            )

    # ----- public API (shared) -----

    def try_consume(self, key: str, tensor: "torch.Tensor") -> Optional[List[Tuple[str, "torch.Tensor"]]]:
        """Try to consume a checkpoint key as a QLoRA quantized weight.

        Returns:
            None   — not a QLoRA key (caller handles normally)
            []     — consumed, waiting for more pieces
            [(...)] — all pieces received → dispatch pairs
        """
        dot_pos = key.rfind(".")
        if dot_pos < 0:
            return None
        prefix, suffix = key[:dot_pos], key[dot_pos + 1 :]

        if prefix not in self._prefix_map:
            return None
        if suffix not in self._suffixes:
            return None

        info = self._prefix_map[prefix]
        accum = self._accums[info.module_fqn]

        accum.received[(info.src_proj or "", suffix)] = tensor
        accum.received_count += 1

        if accum.received_count >= accum.expected_pieces:
            result = self._pack(info.module_fqn, accum)
            # Cross-format conversion if source ≠ target
            if accum.target_format != self._source_format:
                result = self._convert_format(info.module_fqn, accum, result)
            accum.received.clear()
            return result
        return []

    def is_qlora_key(self, key: str) -> bool:
        """Quick check if a key belongs to a QLoRA quantized module."""
        dot_pos = key.rfind(".")
        if dot_pos < 0:
            return False
        prefix, suffix = key[:dot_pos], key[dot_pos + 1 :]
        return prefix in self._prefix_map and suffix in self._suffixes

    def set_inline_metadata(self) -> None:
        """Finalize: set _scale_dtypes, _ema_amax, _inline_loaded on each module."""
        for fqn, accum in self._accums.items():
            if accum.received_count < accum.expected_pieces:
                continue
            module = accum.module

            if accum.scale_dtypes:
                module._scale_dtypes = accum.scale_dtypes
            else:
                module._scale_dtypes = self._default_scale_dtypes(accum.target_format)

            if accum.ema_amax is not None:
                module._ema_amax = accum.ema_amax.to(module.packed_weight_f32.device)
            else:
                module._ema_amax = None

            module._aqn_step_cache = None
            # For block_fp8, inline-loaded packed_weight_f32 may be corrupted by
            # FSDP mixed-precision bf16 cast. Mark as NOT inline-loaded so the
            # deferred path can re-load from disk with correct dtype handling.
            if accum.target_format == "block_fp8":
                module._inline_loaded = False
            else:
                module._inline_loaded = True

    # ----- subclass interface -----

    # Source format string ("nvfp4" or "block_fp8").
    _source_format: str

    def _pack(self, fqn: str, accum: _ModuleAccum) -> List[Tuple[str, "torch.Tensor"]]:
        raise NotImplementedError

    # ----- shared helpers -----

    @staticmethod
    def _default_scale_dtypes(fmt: str) -> Dict[str, "torch.dtype"]:
        if fmt == "block_fp8":
            return {"weight_block_scales": torch.float32}
        return {
            "weight_block_scales": torch.float32,
            "weight_global_scale": torch.float32,
        }

    def _convert_format(
        self,
        fqn: str,
        accum: _ModuleAccum,
        dispatch_pairs: List[Tuple[str, "torch.Tensor"]],
    ) -> List[Tuple[str, "torch.Tensor"]]:
        """Dequantize from source format, re-quantize in target format."""

        module = accum.module
        M, K = module.out_features, module.in_features

        packed_f32 = scales_uint8 = gs_uint8 = None
        for k, t in dispatch_pairs:
            if k.endswith(".packed_weight_f32"):
                packed_f32 = t
            elif k.endswith(".weight_block_scales"):
                scales_uint8 = t
            elif k.endswith(".weight_global_scale"):
                gs_uint8 = t

        # Dequantize from source
        uint8_data = packed_f32.view(torch.uint8)
        if self._source_format == "block_fp8":
            fp8_w = uint8_data.view(torch.float8_e4m3fn).reshape(M, K)
            scales = scales_uint8.contiguous().view(torch.float32)
            w = block_fp8_dequantize_gkn(fp8_w, scales, 128)
        else:
            bs = scales_uint8.contiguous().view(torch.float32)
            gs = gs_uint8.contiguous().view(torch.float32)
            w = nvfp4_dequantize(uint8_data, bs, gs, M * K, 16).reshape(M, K)

        # Re-quantize into target
        target_fmt = accum.target_format
        target_gs = accum.target_group_size
        if target_fmt == "block_fp8":
            fp8_w, scales = block_fp8_quantize_gkn(w.float(), target_gs)
            new_packed = QLoRALinear._to_uint8(fp8_w).contiguous().view(torch.float32)
            new_packed = new_packed.reshape(module.packed_weight_f32.shape)
            accum.ema_amax = None
            accum.scale_dtypes = {"weight_block_scales": torch.float32}
            return [
                (f"{fqn}.packed_weight_f32", new_packed),
                (f"{fqn}.weight_block_scales", QLoRALinear._to_uint8(scales)),
            ]
        else:
            ema_amax = w.float().abs().max().reshape(1)
            packed, block_scales, global_scale = nvfp4_quantize(
                w,
                target_gs,
                global_amax=ema_amax,
            )
            new_packed = QLoRALinear._to_uint8(packed).contiguous().view(torch.float32)
            new_packed = new_packed.reshape(module.packed_weight_f32.shape)
            accum.ema_amax = ema_amax
            accum.scale_dtypes = {
                "weight_block_scales": torch.float32,
                "weight_global_scale": torch.float32,
            }
            return [
                (f"{fqn}.packed_weight_f32", new_packed),
                (f"{fqn}.weight_block_scales", QLoRALinear._to_uint8(block_scales)),
                (f"{fqn}.weight_global_scale", QLoRALinear._to_uint8(global_scale)),
            ]


class NvFP4WeightBuffer(_QLoRAWeightBufferBase):
    """Inline buffer for NVFP4 pre-quantized checkpoints.

    Checkpoint keys per projection: ``weight``, ``weight_scale``, ``weight_scale_2``.
    Absorbs per-source global_scale into block_scales, tracks EMA amax.
    """

    _suffixes = _QLORA_SUFFIXES_NVFP4
    _source_format = "nvfp4"

    def _pack(self, fqn: str, accum: _ModuleAccum) -> List[Tuple[str, "torch.Tensor"]]:
        module = accum.module
        merge_sources = accum.merge_sources

        if merge_sources is not None:
            packed_parts, bs_parts, amax_vals = [], [], []
            for src in merge_sources:
                packed = accum.received[(src, "weight")]
                block_scales = accum.received[(src, "weight_scale")]
                global_scale = accum.received[(src, "weight_scale_2")]

                amax_vals.append(global_scale.float().item() * 6.0 * 448.0)
                bs_parts.append(block_scales.float() * global_scale.float())
                packed_parts.append(packed)

            merged_packed = torch.cat(packed_parts, dim=0)
            merged_bs = torch.cat(bs_parts, dim=0)
            merged_gs = torch.tensor([1.0], dtype=torch.float32)
            max_amax = max(amax_vals)
        else:
            packed = accum.received[("", "weight")]
            block_scales = accum.received[("", "weight_scale")]
            global_scale = accum.received[("", "weight_scale_2")]

            max_amax = global_scale.float().item() * 6.0 * 448.0
            merged_packed = packed
            merged_bs = block_scales.float() * global_scale.float()
            merged_gs = torch.tensor([1.0], dtype=torch.float32)

        accum.ema_amax = torch.tensor([max_amax], dtype=torch.float32)

        packed_f32 = QLoRALinear._to_uint8(merged_packed).contiguous().view(torch.float32)
        packed_f32 = packed_f32.reshape(module.packed_weight_f32.shape)

        return [
            (f"{fqn}.packed_weight_f32", packed_f32),
            (f"{fqn}.weight_block_scales", QLoRALinear._to_uint8(merged_bs)),
            (f"{fqn}.weight_global_scale", QLoRALinear._to_uint8(merged_gs)),
        ]


class BlockFP8WeightBuffer(_QLoRAWeightBufferBase):
    """Inline buffer for block FP8 pre-quantized checkpoints.

    Checkpoint keys per projection: ``weight`` (float8_e4m3fn), ``weight_scale_inv`` (float32).
    No EMA amax — block FP8 uses per-block scales only.
    """

    _suffixes = _QLORA_SUFFIXES_FP8
    _source_format = "block_fp8"

    def _pack(self, fqn: str, accum: _ModuleAccum) -> List[Tuple[str, "torch.Tensor"]]:
        module = accum.module
        merge_sources = accum.merge_sources

        if merge_sources is not None:
            packed_parts, scales_parts = [], []
            for src in merge_sources:
                packed_parts.append(accum.received[(src, "weight")])
                scales_parts.append(accum.received[(src, "weight_scale_inv")])
            merged_packed = torch.cat(packed_parts, dim=0)
            merged_scales = torch.cat(scales_parts, dim=0)
        else:
            merged_packed = accum.received[("", "weight")]
            merged_scales = accum.received[("", "weight_scale_inv")]

        accum.ema_amax = None

        packed_f32 = QLoRALinear._to_uint8(merged_packed).contiguous().view(torch.float32)
        packed_f32 = packed_f32.reshape(module.packed_weight_f32.shape)

        return [
            (f"{fqn}.packed_weight_f32", packed_f32),
            (f"{fqn}.weight_block_scales", QLoRALinear._to_uint8(merged_scales.float())),
        ]


def QLoRAWeightBuffer(model: "torch.nn.Module") -> _QLoRAWeightBufferBase:
    """Factory: return the appropriate weight buffer for the model's source format.

    Inspects the first prequantized QLoRALinear module to determine whether the
    checkpoint is NVFP4 or block FP8, then returns the matching buffer class.
    """

    for _fqn, module in model.named_modules():
        if isinstance(module, QLoRALinear) and module._is_prequantized:
            source_format = module._source_quant_format or module.quant_format
            if source_format == "block_fp8":
                return BlockFP8WeightBuffer(model)
            return NvFP4WeightBuffer(model)

    # No prequantized modules — return a no-op NVFP4 buffer (will never match keys)
    return NvFP4WeightBuffer(model)


# =============================================================================
# QLoRAExpertBuffer — inline quantized expert weight loading
# =============================================================================


class _QLoRAExpertBufferBase:
    """Inline buffer for pre-quantized MoE expert weights.

    Accumulates expert weight + scale checkpoint keys during the main loading pass.
    When all keys for one expert+proj arrive, processes them (transpose, scale absorption).
    When all experts for a (layer, proj) complete, stacks and writes directly to module.

    No dispatch pairs returned — writes to QLoRAMoeExperts buffers in-place.
    """

    _suffixes: frozenset  # expected suffixes per expert (subclass defines)

    def __init__(
        self,
        model: "torch.nn.Module",
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ):
        self._num_experts = num_experts
        self._local_num = num_experts // ep_size
        self._expert_start = ep_rank * self._local_num
        self._expert_end = self._expert_start + self._local_num
        self._num_suffixes = len(self._suffixes)

        # Map layer_idx -> QLoRAMoeExperts module
        self._layer_modules: Dict[int, "QLoRAMoeExperts"] = {}
        for fqn, m in model.named_modules():
            if isinstance(m, QLoRAMoeExperts) and not m._weights_loaded:
                layer_match = re.search(r"layers\.(\d+)", fqn)
                if layer_match:
                    self._layer_modules[int(layer_match.group(1))] = m

        # Per-expert accumulator: (layer, expert_idx, proj) -> {suffix -> tensor}
        self._accums: Dict[Tuple[int, int, str], Dict[str, torch.Tensor]] = {}

        # Per-(layer, proj): processed expert data lists
        self._packed_lists: Dict[Tuple[int, str], List[Optional[torch.Tensor]]] = {}
        self._scales_lists: Dict[Tuple[int, str], List[Optional[torch.Tensor]]] = {}
        self._meta_lists: Dict[Tuple[int, str], List] = {}

        # Completion: total keys seen per (layer, proj) — includes EP-skipped
        self._keys_seen: Dict[Tuple[int, str], int] = defaultdict(int)
        self._keys_expected_per_proj = num_experts * self._num_suffixes

        # Track how many local experts have been fully processed per (layer, proj)
        self._local_experts_filled: Dict[Tuple[int, str], int] = defaultdict(int)

        # Track finalized projections
        self._finalized: Set[Tuple[int, str]] = set()

    @property
    def expert_start(self) -> int:
        return self._expert_start

    @property
    def expert_end(self) -> int:
        return self._expert_end

    def has_modules(self) -> bool:
        return len(self._layer_modules) > 0

    def try_consume(self, key: str, tensor: torch.Tensor) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """Try to consume an expert quantized key.

        Returns None if not an expert key.
        Returns [] (writes directly to module when complete).
        """
        parsed = parse_expert_full_key(key)
        if parsed is None:
            return None

        layer, expert_idx, proj, suffix = parsed

        if layer not in self._layer_modules:
            return None

        # Drop input_scale and unknown suffixes
        if suffix not in self._suffixes:
            return []

        lp = (layer, proj)
        self._keys_seen[lp] += 1

        # EP: skip out-of-range experts (counted but not processed)
        if expert_idx < self._expert_start or expert_idx >= self._expert_end:
            self._maybe_finalize(layer, proj)
            return []

        # Accumulate
        key_3 = (layer, expert_idx, proj)
        if key_3 not in self._accums:
            self._accums[key_3] = {}
        self._accums[key_3][suffix] = tensor

        # Check if this expert is complete
        if len(self._accums[key_3]) >= self._num_suffixes:
            local_idx = expert_idx - self._expert_start
            self._process_expert(layer, proj, local_idx, self._accums.pop(key_3))
            self._local_experts_filled[lp] += 1

        self._maybe_finalize(layer, proj)
        return []

    def count_skipped(self, key: str) -> List[Tuple[str, torch.Tensor]]:
        """Count a skipped expert key for completion tracking."""
        parsed = parse_expert_full_key(key)
        if parsed is None:
            return []

        layer, _, proj, suffix = parsed
        if suffix not in self._suffixes or layer not in self._layer_modules:
            return []

        self._keys_seen[(layer, proj)] += 1
        self._maybe_finalize(layer, proj)
        return []

    def _maybe_finalize(self, layer: int, proj: str) -> None:
        """Check if all experts for (layer, proj) are complete, finalize if so."""
        lp = (layer, proj)
        if lp in self._finalized:
            return
        # Need all global keys seen AND all local experts fully processed
        if self._keys_seen[lp] < self._keys_expected_per_proj:
            return
        if self._local_experts_filled[lp] < self._local_num:
            return

        self._finalize_proj(layer, proj)
        self._finalized.add(lp)

        # Check if all 3 projections for this layer are done
        if all((layer, p) in self._finalized for p in ("gate", "up", "down")):
            self._layer_modules[layer]._weights_loaded = True

    def get_pending(self) -> List[Tuple[int, str]]:
        """Return (layer, proj) pairs that haven't been finalized."""
        pending = []
        for layer in self._layer_modules:
            for proj in ("gate", "up", "down"):
                lp = (layer, proj)
                if lp not in self._finalized and self._keys_seen[lp] > 0:
                    pending.append(lp)
        return pending

    def set_inline_metadata(self) -> None:
        """Finalize inline-loaded modules: materialize LoRA params from meta device."""

        for module in self._layer_modules.values():
            if not module._weights_loaded:
                continue
            for name, param in list(module.named_parameters()):
                if param.device.type != "meta":
                    continue
                is_dtensor = isinstance(param, DTensor)
                if is_dtensor:
                    local_shape = param.to_local().shape
                    placement = param.placements
                    mesh = param.device_mesh
                    local_data = torch.zeros(
                        local_shape,
                        dtype=param.dtype,
                        device="cuda",
                    )
                    materialized = torch.nn.Parameter(
                        DTensor.from_local(local_data, mesh, placement, run_check=False),
                        requires_grad=param.requires_grad,
                    )
                else:
                    materialized = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype, device="cuda"),
                        requires_grad=param.requires_grad,
                    )
                    parts = name.split("_")
                    if parts[-1] == "A":
                        for i in range(materialized.shape[0]):
                            torch.nn.init.kaiming_uniform_(materialized.data[i], a=math.sqrt(5))
                setattr(module, name, materialized)

    def _process_expert(self, layer: int, proj: str, local_idx: int, pieces: Dict[str, torch.Tensor]) -> None:
        """Process one expert's data (transpose, scale absorption). Subclass implements."""
        raise NotImplementedError

    def _finalize_proj(self, layer: int, proj: str) -> None:
        """Stack all local experts, convert to uint8, write to module. Subclass implements."""
        raise NotImplementedError


class NvFP4QLoRAExpertBuffer(_QLoRAExpertBufferBase):
    """Inline buffer for NVFP4 pre-quantized MoE expert weights."""

    _suffixes = _QLORA_SUFFIXES_NVFP4  # {"weight", "weight_scale", "weight_scale_2"}

    def _process_expert(self, layer, proj, local_idx, pieces):
        lp = (layer, proj)
        packed = pieces["weight"]  # [N, K//2] uint8
        block_scales = pieces["weight_scale"]  # [N, K//bs] fp8
        global_scale = pieces["weight_scale_2"]  # [1] fp32

        # Transpose HF [N, K] -> GKN [K, N], absorb global_scale into block_scales
        packed_gkn = packed.T.contiguous()
        block_scales_gkn = (block_scales.float() * global_scale.float()).T.contiguous()
        amax = global_scale.float().item() * FP4_E2M1_MAX * FP8_E4M3_MAX

        if lp not in self._packed_lists:
            self._packed_lists[lp] = [None] * self._local_num
            self._scales_lists[lp] = [None] * self._local_num
            self._meta_lists[lp] = [None] * self._local_num

        self._packed_lists[lp][local_idx] = QLoRAMoeExperts._to_uint8(packed_gkn)
        self._scales_lists[lp][local_idx] = QLoRAMoeExperts._to_uint8(block_scales_gkn)
        self._meta_lists[lp][local_idx] = amax

    def _finalize_proj(self, layer, proj):
        lp = (layer, proj)
        module = self._layer_modules[layer]
        device = torch.device("cuda")

        stacked_packed = torch.stack(self._packed_lists.pop(lp)).to(device)
        stacked_scales = torch.stack(self._scales_lists.pop(lp)).to(device)
        amax_list = self._meta_lists.pop(lp)

        setattr(module, f"{proj}_packed", stacked_packed)
        setattr(module, f"{proj}_block_scales", stacked_scales)

        # Global scale absorbed into block_scales — store 1.0 per expert
        gs = QLoRAMoeExperts._to_uint8(torch.ones(self._local_num, 1, dtype=torch.float32)).to(device)
        setattr(module, f"{proj}_global_scale", gs)

        module._scale_dtypes[proj] = {
            "weight_block_scales": torch.float32,
            "weight_global_scale": torch.float32,
        }
        module._ema_amax[proj] = torch.tensor(amax_list, dtype=torch.float32, device=device)


class BlockFP8QLoRAExpertBuffer(_QLoRAExpertBufferBase):
    """Inline buffer for block FP8 pre-quantized MoE expert weights."""

    _suffixes = _QLORA_SUFFIXES_FP8  # {"weight", "weight_scale_inv"}

    def _process_expert(self, layer, proj, local_idx, pieces):
        lp = (layer, proj)
        fp8_w = pieces["weight"]  # [N, K] float8_e4m3fn
        scales = pieces["weight_scale_inv"]  # [N//128, K//128] f32

        # Transpose HF [N, K] -> GKN [K, N]
        fp8_w_gkn = fp8_w.T.contiguous()
        scales_gkn = scales.float().T.contiguous()

        if lp not in self._packed_lists:
            self._packed_lists[lp] = [None] * self._local_num
            self._scales_lists[lp] = [None] * self._local_num

        self._packed_lists[lp][local_idx] = QLoRAMoeExperts._to_uint8(fp8_w_gkn)
        self._scales_lists[lp][local_idx] = QLoRAMoeExperts._to_uint8(scales_gkn)

    def _finalize_proj(self, layer, proj):
        lp = (layer, proj)
        module = self._layer_modules[layer]
        device = torch.device("cuda")

        stacked_packed = torch.stack(self._packed_lists.pop(lp)).to(device)
        stacked_scales = torch.stack(self._scales_lists.pop(lp)).to(device)

        setattr(module, f"{proj}_packed", stacked_packed)
        setattr(module, f"{proj}_block_scales", stacked_scales)

        module._scale_dtypes[proj] = {"weight_block_scales": torch.float32}


def QLoRAExpertBuffer(
    model: "torch.nn.Module",
    ep_rank: int,
    ep_size: int,
    num_experts: int,
) -> Optional[_QLoRAExpertBufferBase]:
    """Factory: create format-specific QLoRA expert buffer, or None if no MoE experts."""

    for _, m in model.named_modules():
        if isinstance(m, QLoRAMoeExperts) and not m._weights_loaded:
            cls = BlockFP8QLoRAExpertBuffer if isinstance(m, BlockFP8QLoRAMoeExperts) else NvFP4QLoRAExpertBuffer
            buf = cls(model, ep_rank, ep_size, num_experts)
            return buf if buf.has_modules() else None

    return None
