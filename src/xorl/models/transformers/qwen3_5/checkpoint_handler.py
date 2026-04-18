"""Checkpoint handler for dense Qwen3_5 models."""

import warnings
from typing import Callable, List, Optional, Set, Tuple

import torch

from ...checkpoint_handlers.base import CheckpointHandler
from ...checkpoint_handlers.buffers import (
    DENSE_DOWN_PROJ_PATTERN,
    DENSE_GATE_UP_PATTERN,
    FP8_AUX_SUFFIX_PATTERN,
    OPROJ_WEIGHT_PATTERN,
    QKV_PROJ_PATTERN,
    QUANT_AUX_SUFFIX_PATTERN,
    GateUpMergeBuffer,
    QKVMergeBuffer,
)
from ..qwen3_5_shared import (
    is_excluded_module_key,
    map_qwen3_5_linear_attention_weight,
)


class Qwen3_5CheckpointHandler(CheckpointHandler):
    """Checkpoint handler for dense Qwen3_5 models.

    Load: merge gate_proj + up_proj -> gate_up_proj
          merge q_proj + k_proj + v_proj -> qkv_proj
    Save: split gate_up_proj -> gate_proj + up_proj
          split qkv_proj -> q_proj + k_proj + v_proj

    When ``is_prequantized=True``, quantized auxiliary keys (weight_scale,
    weight_scale_2, input_scale, weight_scale_inv) and linear projection
    ``.weight`` keys are skipped - they are loaded directly by QLoRA modules.
    Bias keys still flow through the normal merge buffers.
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        linear_key_dim: int,
        linear_value_dim: int,
        skip_qkv_merge: bool = False,
        is_prequantized: bool = False,
        exclude_modules: Optional[Set[str]] = None,
    ):
        self._gate_up_buffer = GateUpMergeBuffer()
        self._qkv_buffer = None if skip_qkv_merge else QKVMergeBuffer()
        self._q_dim = num_attention_heads * head_dim
        self._kv_dim = num_key_value_heads * head_dim
        self._linear_key_dim = linear_key_dim
        self._linear_value_dim = linear_value_dim
        self._is_prequantized = is_prequantized
        self._exclude_modules = exclude_modules or set()

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

        When prequantized, skips all quantized auxiliary keys and linear
        projection ``.weight`` keys - these are loaded by QLoRA modules.
        """
        if not self._is_prequantized:
            return None

        exclude_modules = self._exclude_modules

        def _should_skip(key: str) -> bool:
            # Don't skip keys belonging to excluded modules - they're bf16, load normally.
            if exclude_modules:
                # Extract module FQN: "model.layers.0.mlp.gate.weight" -> "model.layers.0.mlp.gate"
                module_fqn = key.rsplit(".", 1)[0] if "." in key else key
                # Check if the short module name (last component) is excluded
                module_short_name = module_fqn.rsplit(".", 1)[-1]
                if module_short_name in exclude_modules:
                    return False

            # Skip quantized auxiliary keys (weight_scale, weight_scale_2, input_scale)
            if QUANT_AUX_SUFFIX_PATTERN.search(key):
                return True
            # Skip block FP8 scale keys (weight_scale_inv)
            if FP8_AUX_SUFFIX_PATTERN.search(key):
                return True
            # Skip linear projection weight keys loaded by QLoRALinear directly.
            # Bias keys (.bias) are NOT skipped - they merge normally.
            if key.endswith(".weight"):
                if (
                    QKV_PROJ_PATTERN.match(key)
                    or DENSE_GATE_UP_PATTERN.match(key)
                    or OPROJ_WEIGHT_PATTERN.match(key)
                    or DENSE_DOWN_PROJ_PATTERN.match(key)
                ):
                    return True
            return False

        return _should_skip

    def _is_excluded_module(self, key: str) -> bool:
        """Check if a key belongs to a module excluded from quantization."""
        return is_excluded_module_key(key, self._exclude_modules)

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        # Pre-quantized: drop any quant auxiliary or weight keys that weren't
        # caught by get_skip_key_fn (safety net).
        # Excluded modules pass through as bf16 - don't drop them.
        if self._is_prequantized and not self._is_excluded_module(key):
            if QUANT_AUX_SUFFIX_PATTERN.search(key):
                return []
            if FP8_AUX_SUFFIX_PATTERN.search(key):
                return []
            if key.endswith(".weight"):
                if OPROJ_WEIGHT_PATTERN.match(key) or DENSE_DOWN_PROJ_PATTERN.match(key):
                    return []

        linear_attn_results = self._handle_linear_attention_weights(key, tensor)
        if linear_attn_results is not None:
            return linear_attn_results

        # QKV merge (skipped when unfused for TP)
        if self._qkv_buffer is not None:
            if self._is_prequantized and key.endswith(".weight"):
                if self._qkv_buffer.is_qkv_key(key):
                    return []
            else:
                qkv_result = self._qkv_buffer.add(key, tensor)
                if qkv_result is not None:
                    return [qkv_result]
                if self._qkv_buffer.is_qkv_key(key):
                    return []

        # Gate/up merge
        if self._is_prequantized and key.endswith(".weight"):
            # Packed uint8 gate/up weights - skip standard merging
            if self._gate_up_buffer.is_gate_up_key(key):
                return []
        else:
            merge_result = self._gate_up_buffer.add(key, tensor)
            if merge_result is not None:
                return [merge_result]
            if self._gate_up_buffer.is_gate_up_key(key):
                return []

        return [(key, tensor)]

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
        pending_gu = self._gate_up_buffer.get_pending()
        if pending_gu:
            warnings.warn(f"Incomplete gate/up merge pairs after loading: {pending_gu}")
        if self._qkv_buffer is not None:
            pending_qkv = self._qkv_buffer.get_pending()
            if pending_qkv:
                warnings.warn(f"Incomplete QKV merge groups after loading: {pending_qkv}")
        return []

    def on_save_weight(self, param_name: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        # Split gate_up_proj -> gate_proj + up_proj
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
