"""Checkpoint handler for Qwen3_5 MoE models."""

import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from ...checkpoint_handlers.base import CheckpointHandler
from ...checkpoint_handlers.buffers import (
    DENSE_DOWN_PROJ_PATTERN,
    DENSE_GATE_UP_PATTERN,
    EXPERT_QUANT_AUX_PATTERN,
    FP8_AUX_SUFFIX_PATTERN,
    OPROJ_WEIGHT_PATTERN,
    QKV_PROJ_PATTERN,
    QUANT_AUX_SUFFIX_PATTERN,
    ExpertWeightBuffer,
    GateUpMergeBuffer,
    QKVMergeBuffer,
    parse_expert_key,
)
from ..qwen3_5_shared import (
    is_excluded_module_key,
    map_qwen3_5_linear_attention_weight,
)


class Qwen3_5MoeCheckpointHandler(CheckpointHandler):
    """Checkpoint handler for Qwen3_5 MoE models.

    Load transforms:
    1. Per-expert weights -> fused [num_experts, ...] stacked tensors
    2. Shared expert / dense layer gate_proj + up_proj -> gate_up_proj
    3. q_proj + k_proj + v_proj -> qkv_proj

    Save transforms:
    1. gate_up_proj -> gate_proj + up_proj (shared expert / dense layers)
    2. qkv_proj -> q_proj + k_proj + v_proj
    3. Expert tensors pass through as-is (fused format)

    EP handling:
    - direct-load path: pass real ep_rank/ep_size. ExpertWeightBuffer filters per rank.
    - broadcast path: pass ep_rank=0, ep_size=1. Rank 0 buffers all experts.
      EP slicing is handled later by ParallelPlan.shard_tensor().

    TP handling:
    - When ``skip_qkv_merge=True``, QKV keys pass through unmerged
      (model has separate q_proj/k_proj/v_proj after unfuse_for_tp).
    - When ``skip_gate_up_merge=True``, gate/up keys pass through unmerged
      (dense MLP layers have separate gate_proj/up_proj after unfuse_for_tp).
    - Expert merging is always active (stacking per-expert HF weights).
    """

    def __init__(
        self,
        num_experts: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        linear_key_dim: int,
        linear_value_dim: int,
        ep_rank: int = 0,
        ep_size: int = 1,
        checkpoint_has_per_expert: bool = True,
        skip_qkv_merge: bool = False,
        skip_gate_up_merge: bool = False,
        is_prequantized: bool = False,
        exclude_modules: Optional[Set[str]] = None,
    ):
        self._expert_buffer: Optional[ExpertWeightBuffer] = None
        # Disable expert buffer when pre-quantized: expert weights are loaded
        # directly by QLoRAMoeExperts, not merged through the checkpoint handler.
        if checkpoint_has_per_expert and not is_prequantized:
            self._expert_buffer = ExpertWeightBuffer(num_experts, ep_rank=ep_rank, ep_size=ep_size)
        self._qkv_buffer: Optional[QKVMergeBuffer] = None
        if not skip_qkv_merge:
            self._qkv_buffer = QKVMergeBuffer()
        self._gate_up_buffer: Optional[GateUpMergeBuffer] = None
        if not skip_gate_up_merge:
            self._gate_up_buffer = GateUpMergeBuffer()
        self._q_dim = num_attention_heads * head_dim
        self._kv_dim = num_key_value_heads * head_dim
        self._linear_key_dim = linear_key_dim
        self._linear_value_dim = linear_value_dim
        self._is_prequantized = is_prequantized
        self._exclude_modules = exclude_modules or set()
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._local_num_experts = num_experts // ep_size
        self._expert_start = ep_rank * self._local_num_experts
        self._expert_end = self._expert_start + self._local_num_experts
        self._stacked_gate_up_pending: Dict[int, Dict[str, torch.Tensor]] = {}

    _STACKED_EXPERT_SPLIT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj$")
    _HF_FUSED_EXPERT_GATE_UP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj\.weight$")
    _HF_FUSED_EXPERT_DOWN_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj\.weight$")
    _INTERNAL_FUSED_EXPERT_GATE_UP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
    _INTERNAL_FUSED_EXPERT_DOWN_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")

    def _slice_expert_tensor_for_ep(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._ep_size == 1:
            return tensor
        return tensor[self._expert_start : self._expert_end].contiguous()

    def _maybe_finalize_per_expert_merge(
        self,
        layer_idx: int,
        proj: str,
    ) -> List[Tuple[str, torch.Tensor]]:
        if self._expert_buffer is None:
            return []

        if proj in {"gate", "up"}:
            if not (
                self._expert_buffer.is_complete(layer_idx, "gate") and self._expert_buffer.is_complete(layer_idx, "up")
            ):
                return []
            gate = self._expert_buffer.pop_stacked(layer_idx, "gate")
            up = self._expert_buffer.pop_stacked(layer_idx, "up")
            return [
                (
                    ExpertWeightBuffer.get_gate_up_name(layer_idx),
                    torch.cat([gate, up], dim=2),
                )
            ]

        if proj == "down" and self._expert_buffer.is_complete(layer_idx, "down"):
            return [
                (
                    ExpertWeightBuffer.get_fused_name(layer_idx, "down"),
                    self._expert_buffer.pop_stacked(layer_idx, "down"),
                )
            ]

        return []

    def _handle_fused_expert_weights(self, key: str, tensor: torch.Tensor) -> Optional[List[Tuple[str, torch.Tensor]]]:
        internal_gate_up_match = self._INTERNAL_FUSED_EXPERT_GATE_UP_PATTERN.match(key)
        if internal_gate_up_match is not None:
            return [(key, self._slice_expert_tensor_for_ep(tensor))]

        internal_down_match = self._INTERNAL_FUSED_EXPERT_DOWN_PATTERN.match(key)
        if internal_down_match is not None:
            return [(key, self._slice_expert_tensor_for_ep(tensor))]

        gate_up_match = self._HF_FUSED_EXPERT_GATE_UP_PATTERN.match(key)
        if gate_up_match is not None:
            layer_idx = int(gate_up_match.group(1))
            tensor = self._slice_expert_tensor_for_ep(tensor)
            gate_up = tensor.transpose(1, 2).contiguous()
            return [
                (f"model.layers.{layer_idx}.mlp.experts.gate_up_proj", gate_up),
            ]

        down_match = self._HF_FUSED_EXPERT_DOWN_PATTERN.match(key)
        if down_match is not None:
            layer_idx = int(down_match.group(1))
            tensor = self._slice_expert_tensor_for_ep(tensor)
            down = tensor.transpose(1, 2).contiguous()
            return [
                (f"model.layers.{layer_idx}.mlp.experts.down_proj", down),
            ]

        split_match = self._STACKED_EXPERT_SPLIT_PATTERN.match(key)
        if split_match is not None:
            layer_idx = int(split_match.group(1))
            proj = split_match.group(2)
            tensor = self._slice_expert_tensor_for_ep(tensor)
            if proj == "down":
                return [(f"model.layers.{layer_idx}.mlp.experts.down_proj", tensor)]

            pending = self._stacked_gate_up_pending.setdefault(layer_idx, {})
            pending[proj] = tensor
            if "gate" in pending and "up" in pending:
                gate = pending.pop("gate")
                up = pending.pop("up")
                del self._stacked_gate_up_pending[layer_idx]
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                        torch.cat([gate, up], dim=2),
                    )
                ]
            return []

        return None

    def _handle_linear_attention_weights(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        return map_qwen3_5_linear_attention_weight(
            key,
            tensor,
            linear_key_dim=self._linear_key_dim,
            linear_value_dim=self._linear_value_dim,
        )

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        """Return predicate to skip keys during loading.

        Skips:
        - Out-of-range expert keys (EP loading)
        - All quantized auxiliary keys when is_prequantized=True (weight_scale,
          weight_scale_2, input_scale) — these are loaded directly by QLoRA modules
        """
        has_ep_filter = self._expert_buffer is not None and not (
            self._expert_buffer.expert_start == 0 and self._expert_buffer.expert_end == self._expert_buffer.num_experts
        )

        if not has_ep_filter and not self._is_prequantized:
            return None

        ep_start = self._expert_buffer.expert_start if has_ep_filter else 0
        ep_end = self._expert_buffer.expert_end if has_ep_filter else 0
        is_prequantized = self._is_prequantized
        exclude_modules = self._exclude_modules

        def _should_skip(key: str) -> bool:
            if is_prequantized:
                # Don't skip keys belonging to excluded modules — they're bf16, load normally.
                if exclude_modules:
                    module_fqn = key.rsplit(".", 1)[0] if "." in key else key
                    module_short_name = module_fqn.rsplit(".", 1)[-1]
                    if module_short_name in exclude_modules:
                        return False

                # Skip all quantized auxiliary keys — loaded by QLoRA modules directly
                if QUANT_AUX_SUFFIX_PATTERN.search(key):
                    return True
                # Skip block FP8 scale keys
                if FP8_AUX_SUFFIX_PATTERN.search(key):
                    return True
                # Skip all expert weight keys — loaded by QLoRAMoeExperts directly
                if parse_expert_key(key) is not None:
                    return True
                # Skip expert quantized auxiliary keys
                if EXPERT_QUANT_AUX_PATTERN.match(key) is not None:
                    return True
                # Skip all linear projection weight keys that are loaded by QLoRALinear
                # directly. Bias keys (.bias) are NOT skipped (merged normally).
                if key.endswith(".weight"):
                    if (
                        QKV_PROJ_PATTERN.match(key)
                        or DENSE_GATE_UP_PATTERN.match(key)
                        or OPROJ_WEIGHT_PATTERN.match(key)
                        or DENSE_DOWN_PROJ_PATTERN.match(key)
                    ):
                        return True
            # Skip out-of-range expert keys for EP (non-prequantized path)
            if has_ep_filter:
                parsed = parse_expert_key(key)
                if parsed is not None:
                    _, expert_idx, _ = parsed
                    return expert_idx < ep_start or expert_idx >= ep_end
            return False

        return _should_skip

    def _is_excluded_module(self, key: str) -> bool:
        """Check if a key belongs to a module excluded from quantization."""
        return is_excluded_module_key(key, self._exclude_modules)

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        # 0. Pre-quantized: skip quantized auxiliary keys and quantized weight keys
        #    that weren't caught by get_skip_key_fn (e.g., when skip_key_fn wasn't used)
        #    Excluded modules pass through as bf16 — don't drop them.
        if self._is_prequantized and not self._is_excluded_module(key):
            if QUANT_AUX_SUFFIX_PATTERN.search(key):
                return []
            if FP8_AUX_SUFFIX_PATTERN.search(key):
                return []
            if parse_expert_key(key) is not None:
                return []
            if EXPERT_QUANT_AUX_PATTERN.match(key) is not None:
                return []
            # Skip o_proj and down_proj weight keys — loaded by QLoRALinear directly
            if key.endswith(".weight"):
                if OPROJ_WEIGHT_PATTERN.match(key) or DENSE_DOWN_PROJ_PATTERN.match(key):
                    return []

        # 1. Check fused expert tensors from HF Qwen3.5 checkpoints
        fused_expert_results = self._handle_fused_expert_weights(key, tensor)
        if fused_expert_results is not None:
            return fused_expert_results

        # 2. Check linear attention tensors from HF Qwen3.5 checkpoints
        linear_attn_results = self._handle_linear_attention_weights(key, tensor)
        if linear_attn_results is not None:
            return linear_attn_results

        # 3. Check expert merge (disabled when prequantized — expert_buffer is None)
        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, expert_idx, proj = parsed
                self._expert_buffer.add(layer_idx, expert_idx, proj, tensor)
                return self._maybe_finalize_per_expert_merge(layer_idx, proj)

        # 4. Check QKV merge (skipped when unfused for TP)
        if self._qkv_buffer is not None:
            # When pre-quantized, QKV .weight keys are uint8 packed —
            # skip standard merging, they'll be loaded by QLoRALinear
            if self._is_prequantized and key.endswith(".weight"):
                if self._qkv_buffer.is_qkv_key(key):
                    return []

            qkv_result = self._qkv_buffer.add(key, tensor)
            if qkv_result is not None:
                return [qkv_result]
            if self._qkv_buffer.is_qkv_key(key):
                return []

        # 5. Check gate/up merge (skipped when unfused for TP)
        if self._gate_up_buffer is not None:
            # When pre-quantized, gate/up .weight keys are uint8 packed —
            # skip standard merging, they'll be loaded by QLoRALinear
            if self._is_prequantized and key.endswith(".weight"):
                if self._gate_up_buffer.is_gate_up_key(key):
                    return []

            merge_result = self._gate_up_buffer.add(key, tensor)
            if merge_result is not None:
                return [merge_result]
            if self._gate_up_buffer.is_gate_up_key(key):
                return []

        # 6. Passthrough
        return [(key, tensor)]

    def on_skip_weight(self, key: str) -> List[Tuple[str, torch.Tensor]]:
        """Count a skipped expert key so completion tracking stays correct."""
        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, _expert_idx, proj = parsed
                self._expert_buffer.count_skipped(layer_idx, proj)
                return self._maybe_finalize_per_expert_merge(layer_idx, proj)
        return []

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
        import warnings

        if self._expert_buffer is not None:
            pending = self._expert_buffer.get_pending_counts()
            if pending:
                warnings.warn(f"Incomplete expert weights after loading: {pending}")
        if self._stacked_gate_up_pending:
            warnings.warn(f"Incomplete stacked expert gate/up merges after loading: {self._stacked_gate_up_pending}")
        if self._gate_up_buffer is not None:
            pending_gu = self._gate_up_buffer.get_pending()
            if pending_gu:
                warnings.warn(f"Incomplete gate/up merge pairs after loading: {pending_gu}")
        if self._qkv_buffer is not None:
            pending_qkv = self._qkv_buffer.get_pending()
            if pending_qkv:
                warnings.warn(f"Incomplete QKV merge groups after loading: {pending_qkv}")
        return []

    def on_save_weight(self, param_name: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        # Split fused MoE experts into per-expert HF weights.
        if param_name.endswith(".mlp.experts.gate_up_proj"):
            prefix = param_name.rsplit(".gate_up_proj", 1)[0]
            half = tensor.shape[2] // 2
            gate = tensor[:, :, :half].transpose(1, 2).contiguous()
            up = tensor[:, :, half:].transpose(1, 2).contiguous()
            result = []
            for expert_idx in range(tensor.shape[0]):
                result.append((f"{prefix}.{expert_idx}.gate_proj.weight", gate[expert_idx]))
                result.append((f"{prefix}.{expert_idx}.up_proj.weight", up[expert_idx]))
            return result

        if param_name.endswith(".mlp.experts.down_proj"):
            prefix = param_name.rsplit(".down_proj", 1)[0]
            down = tensor.transpose(1, 2).contiguous()
            return [
                (f"{prefix}.{expert_idx}.down_proj.weight", down[expert_idx]) for expert_idx in range(tensor.shape[0])
            ]

        # Split gate_up_proj -> gate_proj + up_proj (shared expert / dense layers)
        if ".gate_up_proj." in param_name:
            prefix, suffix = param_name.rsplit(".gate_up_proj.", 1)
            half = tensor.shape[0] // 2
            return [
                (f"{prefix}.gate_proj.{suffix}", tensor[:half]),
                (f"{prefix}.up_proj.{suffix}", tensor[half:]),
            ]

        # Split qkv_proj -> q_proj + k_proj + v_proj
        if ".qkv_proj." in param_name:
            prefix, suffix = param_name.rsplit(".qkv_proj.", 1)
            q, k, v = tensor.split([self._q_dim, self._kv_dim, self._kv_dim], dim=0)
            return [
                (f"{prefix}.q_proj.{suffix}", q),
                (f"{prefix}.k_proj.{suffix}", k),
                (f"{prefix}.v_proj.{suffix}", v),
            ]

        return [(param_name, tensor)]
