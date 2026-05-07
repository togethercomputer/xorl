"""Checkpoint handler for GLM-4 MoE models.

Reuses the same buffer infrastructure as Qwen3 MoE:
- ExpertWeightBuffer: stacks per-expert HF weights into [num_experts, ...] tensors
- QKVMergeBuffer: merges q_proj + k_proj + v_proj -> qkv_proj
- GateUpMergeBuffer: merges gate_proj + up_proj -> gate_up_proj (dense + shared experts)

GLM-specific: ``gate.e_score_correction_bias`` passes through directly (checkpoint
path matches model path).
"""

import warnings
from typing import Callable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

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
    QLoRAExpertBuffer,
    QLoRAWeightBuffer,
    parse_expert_full_key,
    parse_expert_key,
)


class Glm4MoeCheckpointHandler(CheckpointHandler):
    """Checkpoint handler for GLM-4 MoE models.

    Load transforms:
    1. Per-expert weights -> fused [num_experts, ...] stacked tensors
    2. Dense layer / shared expert gate_proj + up_proj -> gate_up_proj
    3. q_proj + k_proj + v_proj -> qkv_proj

    Save transforms:
    1. gate_up_proj -> gate_proj + up_proj
    2. qkv_proj -> q_proj + k_proj + v_proj
    3. Expert tensors pass through as-is (fused format)
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
        num_hidden_layers: Optional[int] = None,
    ):
        self._expert_buffer: Optional[ExpertWeightBuffer] = None
        if checkpoint_has_per_expert and not is_prequantized:
            self._expert_buffer = ExpertWeightBuffer(
                num_experts,
                ep_rank=ep_rank,
                ep_size=ep_size,
                device=device,
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
        # MTP (multi-token prediction) layer remapping: GLM-4.7 stores embedding,
        # output norm, and LM head under model.layers.{num_hidden_layers} in the
        # checkpoint, but the model expects them at top-level positions.
        self._mtp_layer_prefix = f"model.layers.{num_hidden_layers}."
        self._mtp_remap = {
            f"model.layers.{num_hidden_layers}.embed_tokens.weight": "model.embed_tokens.weight",
            f"model.layers.{num_hidden_layers}.shared_head.norm.weight": "model.norm.weight",
            f"model.layers.{num_hidden_layers}.shared_head.head.weight": "lm_head.weight",
        }
        self._qlora_buffer: Optional[QLoRAWeightBuffer] = None
        if is_prequantized and model is not None:
            self._qlora_buffer = QLoRAWeightBuffer(model)
        self._qlora_expert_buffer = None
        if is_prequantized and model is not None:
            self._qlora_expert_buffer = QLoRAExpertBuffer(
                model,
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_experts=num_experts,
            )

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        has_ep_filter = self._expert_buffer is not None and not (
            self._expert_buffer.expert_start == 0 and self._expert_buffer.expert_end == self._expert_buffer.num_experts
        )
        has_expert_ep_filter = self._qlora_expert_buffer is not None and not (
            self._qlora_expert_buffer.expert_start == 0
            and self._qlora_expert_buffer.expert_end == self._qlora_expert_buffer._num_experts
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
                if exclude_modules:
                    module_fqn = key.rsplit(".", 1)[0] if "." in key else key
                    module_short_name = module_fqn.rsplit(".", 1)[-1]
                    if module_short_name in exclude_modules:
                        return False

                if has_qlora_expert_buffer:
                    parsed = parse_expert_full_key(key)
                    if parsed is not None:
                        _, expert_idx, _, suffix = parsed
                        if suffix == "input_scale":
                            return True
                        return expert_idx < qe_start or expert_idx >= qe_end
                else:
                    if parse_expert_key(key) is not None:
                        return True
                    if EXPERT_QUANT_AUX_PATTERN.match(key) is not None:
                        return True

                if not has_qlora_buffer:
                    if QUANT_AUX_SUFFIX_PATTERN.search(key):
                        return True
                    if FP8_AUX_SUFFIX_PATTERN.search(key):
                        return True
                    if key.endswith(".weight"):
                        if (
                            QKV_PROJ_PATTERN.match(key)
                            or DENSE_GATE_UP_PATTERN.match(key)
                            or OPROJ_WEIGHT_PATTERN.match(key)
                            or DENSE_DOWN_PROJ_PATTERN.match(key)
                        ):
                            return True

            if has_ep_filter:
                parsed = parse_expert_key(key)
                if parsed is not None:
                    _, expert_idx, _ = parsed
                    return expert_idx < ep_start or expert_idx >= ep_end
            return False

        return _should_skip

    def _maybe_finalize_per_expert_merge(
        self,
        layer_idx: int,
        proj: str,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Finalize expert weight merging: fuse gate+up into gate_up_proj."""
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

    def _is_excluded_module(self, key: str) -> bool:
        if not self._exclude_modules:
            return False
        module_fqn = key.rsplit(".", 1)[0] if "." in key else key
        module_short_name = module_fqn.rsplit(".", 1)[-1]
        return module_short_name in self._exclude_modules

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        if key.endswith(".input_scale"):
            return []

        # QLoRA expert buffer
        if self._qlora_expert_buffer is not None and not self._is_excluded_module(key):
            result = self._qlora_expert_buffer.try_consume(key, tensor)
            if result is not None:
                return result

        # QLoRA buffer
        if self._qlora_buffer is not None and not self._is_excluded_module(key):
            result = self._qlora_buffer.try_consume(key, tensor)
            if result is not None:
                return result

        if self._is_prequantized and not self._is_excluded_module(key):
            if QUANT_AUX_SUFFIX_PATTERN.search(key):
                return []
            if FP8_AUX_SUFFIX_PATTERN.search(key):
                return []
            if parse_expert_key(key) is not None:
                return []
            if EXPERT_QUANT_AUX_PATTERN.match(key) is not None:
                return []
            if key.endswith(".weight"):
                if OPROJ_WEIGHT_PATTERN.match(key) or DENSE_DOWN_PROJ_PATTERN.match(key):
                    return []

        # 1. MTP layer remapping: remap shared output components, skip MTP-only weights.
        #    Must be checked first to prevent MTP layer expert/QKV/gate_up keys from
        #    entering the merge buffers (which would emit fused keys for a nonexistent layer).
        if key.startswith(self._mtp_layer_prefix):
            if key in self._mtp_remap:
                return [(self._mtp_remap[key], tensor)]
            # Skip MTP-only components (eh_proj, enorm, hnorm, and MTP decoder layer)
            return []

        # 2. Expert merge
        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, expert_idx, proj = parsed
                self._expert_buffer.add(layer_idx, expert_idx, proj, tensor)
                result = self._maybe_finalize_per_expert_merge(layer_idx, proj)
                return result if result else []

        # 3. QKV merge
        if self._qkv_buffer is not None:
            if self._is_prequantized and key.endswith(".weight"):
                if self._qkv_buffer.is_qkv_key(key):
                    return []

            qkv_result = self._qkv_buffer.add(key, tensor)
            if qkv_result is not None:
                return [qkv_result]
            if self._qkv_buffer.is_qkv_key(key):
                return []

        # 4. Gate/up merge
        if self._gate_up_buffer is not None:
            if self._is_prequantized and key.endswith(".weight"):
                if self._gate_up_buffer.is_gate_up_key(key):
                    return []

            merge_result = self._gate_up_buffer.add(key, tensor)
            if merge_result is not None:
                return [merge_result]
            if self._gate_up_buffer.is_gate_up_key(key):
                return []

        # 5. Passthrough (includes gate.e_score_correction_bias)
        return [(key, tensor)]

    def on_skip_weight(self, key: str) -> List[Tuple[str, torch.Tensor]]:
        if self._qlora_expert_buffer is not None:
            self._qlora_expert_buffer.count_skipped(key)

        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, _expert_idx, proj = parsed
                self._expert_buffer.count_skipped(layer_idx, proj)
                result = self._maybe_finalize_per_expert_merge(layer_idx, proj)
                if result:
                    return result
        return []

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
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
        if self._qlora_buffer is not None:
            self._qlora_buffer.set_inline_metadata()
        if self._qlora_expert_buffer is not None:
            pending_exp = self._qlora_expert_buffer.get_pending()
            if pending_exp:
                warnings.warn(
                    f"Incomplete QLoRA expert weights after loading (will fall back to deferred loading): {pending_exp}"
                )
            self._qlora_expert_buffer.set_inline_metadata()
        return []

    def on_save_weight(self, param_name: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        # Split gate_up_proj -> gate_proj + up_proj (dense / shared experts)
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
