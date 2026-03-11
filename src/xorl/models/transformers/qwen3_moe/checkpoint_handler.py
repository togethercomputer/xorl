"""Checkpoint handler for Qwen3 MoE models."""

from typing import Callable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from ...checkpoint_handlers.base import CheckpointHandler
from ...checkpoint_handlers.buffers import (
    ExpertWeightBuffer, GateUpMergeBuffer, QKVMergeBuffer,
    QLoRAWeightBuffer, QLoRAExpertBuffer,
    parse_expert_key, parse_expert_full_key,
    EXPERT_QUANT_AUX_PATTERN, QUANT_AUX_SUFFIX_PATTERN,
    FP8_AUX_SUFFIX_PATTERN,
    QKV_PROJ_PATTERN, DENSE_GATE_UP_PATTERN,
    OPROJ_WEIGHT_PATTERN, DENSE_DOWN_PROJ_PATTERN,
)


class Qwen3MoeCheckpointHandler(CheckpointHandler):
    """Checkpoint handler for Qwen3 MoE models.

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
        ep_rank: int = 0,
        ep_size: int = 1,
        checkpoint_has_per_expert: bool = True,
        skip_qkv_merge: bool = False,
        skip_gate_up_merge: bool = False,
        is_prequantized: bool = False,
        exclude_modules: Optional[Set[str]] = None,
        device: Optional["torch.device"] = None,
        model: Optional[nn.Module] = None,
    ):
        self._expert_buffer: Optional[ExpertWeightBuffer] = None
        # Non-QLoRA expert stacking (disabled when pre-quantized — use QLoRAExpertBuffer instead)
        if checkpoint_has_per_expert and not is_prequantized:
            self._expert_buffer = ExpertWeightBuffer(
                num_experts, ep_rank=ep_rank, ep_size=ep_size, device=device,
            )
        self._qkv_buffer: Optional[QKVMergeBuffer] = None
        if not skip_qkv_merge:
            self._qkv_buffer = QKVMergeBuffer()
        self._gate_up_buffer: Optional[GateUpMergeBuffer] = None
        if not skip_gate_up_merge:
            self._gate_up_buffer = GateUpMergeBuffer()
        self._q_dim = num_attention_heads * head_dim
        self._kv_dim = num_key_value_heads * head_dim
        self._is_prequantized = is_prequantized
        self._exclude_modules = exclude_modules or set()
        # Inline QLoRA weight loading for dense modules (attention, shared expert)
        self._qlora_buffer: Optional[QLoRAWeightBuffer] = None
        if is_prequantized and model is not None:
            self._qlora_buffer = QLoRAWeightBuffer(model)
        # Inline QLoRA expert weight loading (MoE experts)
        self._qlora_expert_buffer = None
        if is_prequantized and model is not None:
            self._qlora_expert_buffer = QLoRAExpertBuffer(
                model, ep_rank=ep_rank, ep_size=ep_size, num_experts=num_experts,
            )

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        """Return predicate to skip keys during loading.

        Skips:
        - Out-of-range expert keys (EP loading — both prequantized and normal)
        - All expert keys when prequantized WITHOUT inline expert buffer (deferred path)
        - Dense quantized keys only when dense buffer is NOT active (deferred path)

        When inline buffers are active, in-range keys flow through on_load_weight.
        """
        has_ep_filter = (
            self._expert_buffer is not None
            and not (self._expert_buffer.expert_start == 0
                     and self._expert_buffer.expert_end == self._expert_buffer.num_experts)
        )
        has_expert_ep_filter = (
            self._qlora_expert_buffer is not None
            and not (self._qlora_expert_buffer.expert_start == 0
                     and self._qlora_expert_buffer.expert_end == self._qlora_expert_buffer._num_experts)
        )

        if not has_ep_filter and not has_expert_ep_filter and not self._is_prequantized:
            return None

        ep_start = self._expert_buffer.expert_start if has_ep_filter else 0
        ep_end = self._expert_buffer.expert_end if has_ep_filter else 0
        is_prequantized = self._is_prequantized
        exclude_modules = self._exclude_modules
        has_qlora_buffer = self._qlora_buffer is not None
        has_qlora_expert_buffer = self._qlora_expert_buffer is not None
        if has_qlora_expert_buffer:
            qe_start = self._qlora_expert_buffer.expert_start
            qe_end = self._qlora_expert_buffer.expert_end

        def _should_skip(key: str) -> bool:
            if is_prequantized:
                # Don't skip keys belonging to excluded modules — they're bf16, load normally.
                if exclude_modules:
                    module_fqn = key.rsplit(".", 1)[0] if "." in key else key
                    module_short_name = module_fqn.rsplit(".", 1)[-1]
                    if module_short_name in exclude_modules:
                        return False

                # Expert keys
                if has_qlora_expert_buffer:
                    # Inline path: EP-filter expert keys (skip out-of-range, load in-range)
                    parsed = parse_expert_full_key(key)
                    if parsed is not None:
                        _, expert_idx, _, suffix = parsed
                        if suffix == "input_scale":
                            return True
                        return expert_idx < qe_start or expert_idx >= qe_end
                else:
                    # Deferred path: skip ALL expert keys
                    if parse_expert_key(key) is not None:
                        return True
                    if EXPERT_QUANT_AUX_PATTERN.match(key) is not None:
                        return True

                # Dense quantized keys: skip only when buffer is NOT active (deferred path)
                if not has_qlora_buffer:
                    if QUANT_AUX_SUFFIX_PATTERN.search(key):
                        return True
                    if FP8_AUX_SUFFIX_PATTERN.search(key):
                        return True
                    if key.endswith(".weight"):
                        if (QKV_PROJ_PATTERN.match(key) or DENSE_GATE_UP_PATTERN.match(key)
                                or OPROJ_WEIGHT_PATTERN.match(key) or DENSE_DOWN_PROJ_PATTERN.match(key)):
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
        if not self._exclude_modules:
            return False
        module_fqn = key.rsplit(".", 1)[0] if "." in key else key
        module_short_name = module_fqn.rsplit(".", 1)[-1]
        return module_short_name in self._exclude_modules

    def on_load_weight(
        self, key: str, tensor: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        # Drop input_scale (unused by our quantization)
        if key.endswith(".input_scale"):
            return []

        # 0a. QLoRA expert buffer: route expert quantized keys for inline loading
        if self._qlora_expert_buffer is not None and not self._is_excluded_module(key):
            result = self._qlora_expert_buffer.try_consume(key, tensor)
            if result is not None:
                return result  # [] always (writes directly to module)

        # 0b. QLoRA buffer: route dense quantized keys for inline loading
        if self._qlora_buffer is not None and not self._is_excluded_module(key):
            result = self._qlora_buffer.try_consume(key, tensor)
            if result is not None:
                return result  # [] = buffered, or list of dispatch pairs

        # Pre-quantized safety net: drop quantized keys not consumed by buffers
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

        # 1. Check expert merge (disabled when prequantized — expert_buffer is None)
        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, expert_idx, proj = parsed
                self._expert_buffer.add(layer_idx, expert_idx, proj, tensor)
                if self._expert_buffer.is_complete(layer_idx, proj):
                    fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                    stacked = self._expert_buffer.pop_stacked(layer_idx, proj)
                    return [(fused_name, stacked)]
                return []

        # 2. Check QKV merge (skipped when unfused for TP)
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

        # 3. Check gate/up merge (skipped when unfused for TP)
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

        # 4. Passthrough
        return [(key, tensor)]

    def on_skip_weight(
        self, key: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Count a skipped expert key so completion tracking stays correct."""
        # QLoRA expert buffer: count skipped out-of-range expert keys
        if self._qlora_expert_buffer is not None:
            self._qlora_expert_buffer.count_skipped(key)

        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, _expert_idx, proj = parsed
                self._expert_buffer.count_skipped(layer_idx, proj)
                if self._expert_buffer.is_complete(layer_idx, proj):
                    fused_name = ExpertWeightBuffer.get_fused_name(layer_idx, proj)
                    stacked = self._expert_buffer.pop_stacked(layer_idx, proj)
                    return [(fused_name, stacked)]
        return []

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
        import warnings
        if self._expert_buffer is not None:
            pending = self._expert_buffer.get_pending_counts()
            if pending:
                warnings.warn(f"Incomplete expert weights after loading: {pending}")
        if self._gate_up_buffer is not None:
            pending_gu = self._gate_up_buffer.get_pending()
            if pending_gu:
                warnings.warn(f"Incomplete gate/up merge pairs after loading: {pending_gu}")
        if self._qkv_buffer is not None:
            pending_qkv = self._qkv_buffer.get_pending()
            if pending_qkv:
                warnings.warn(f"Incomplete QKV merge groups after loading: {pending_qkv}")
        # Finalize inline-loaded QLoRA modules
        if self._qlora_buffer is not None:
            self._qlora_buffer.set_inline_metadata()
        if self._qlora_expert_buffer is not None:
            pending_exp = self._qlora_expert_buffer.get_pending()
            if pending_exp:
                warnings.warn(
                    f"Incomplete QLoRA expert weights after loading "
                    f"(will fall back to deferred loading): {pending_exp}"
                )
            self._qlora_expert_buffer.set_inline_metadata()
        return []

    def on_save_weight(
        self, param_name: str, tensor: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
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
