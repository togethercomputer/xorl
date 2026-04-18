"""Checkpoint handler for Llama 3 models."""

import warnings
from typing import Callable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

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
    QLoRAWeightBuffer,
)


class Llama3CheckpointHandler(CheckpointHandler):
    """Checkpoint handler for Llama 3 models.

    Load: merge gate_proj + up_proj -> gate_up_proj
          merge q_proj + k_proj + v_proj -> qkv_proj
    Save: split gate_up_proj -> gate_proj + up_proj
          split qkv_proj -> q_proj + k_proj + v_proj

    When ``is_prequantized=True`` and a model is provided, quantized weights are
    loaded inline via QLoRAWeightBuffer (single-pass I/O). When model is not
    provided, falls back to skipping quantized keys (deferred loading).
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        is_prequantized: bool = False,
        exclude_modules: Optional[Set[str]] = None,
        model: Optional[nn.Module] = None,
    ):
        self._gate_up_buffer = GateUpMergeBuffer()
        self._qkv_buffer = QKVMergeBuffer()
        self._q_dim = num_attention_heads * head_dim
        self._kv_dim = num_key_value_heads * head_dim
        self._is_prequantized = is_prequantized
        self._exclude_modules = exclude_modules or set()
        self._qlora_buffer: Optional[QLoRAWeightBuffer] = None
        if is_prequantized and model is not None:
            self._qlora_buffer = QLoRAWeightBuffer(model)

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        if not self._is_prequantized:
            return None

        if self._qlora_buffer is not None:
            return None

        exclude_modules = self._exclude_modules

        def _should_skip(key: str) -> bool:
            if exclude_modules:
                module_fqn = key.rsplit(".", 1)[0] if "." in key else key
                module_short_name = module_fqn.rsplit(".", 1)[-1]
                if module_short_name in exclude_modules:
                    return False

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
            return False

        return _should_skip

    def _is_excluded_module(self, key: str) -> bool:
        if not self._exclude_modules:
            return False
        module_fqn = key.rsplit(".", 1)[0] if "." in key else key
        module_short_name = module_fqn.rsplit(".", 1)[-1]
        return module_short_name in self._exclude_modules

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        # Drop input_scale (unused by our quantization)
        if key.endswith(".input_scale"):
            return []

        # QLoRA buffer: route quantized keys for inline loading
        if self._qlora_buffer is not None and not self._is_excluded_module(key):
            result = self._qlora_buffer.try_consume(key, tensor)
            if result is not None:
                return result

        # Pre-quantized without buffer (deferred path): drop quantized keys
        if self._is_prequantized and self._qlora_buffer is None and not self._is_excluded_module(key):
            if QUANT_AUX_SUFFIX_PATTERN.search(key):
                return []
            if FP8_AUX_SUFFIX_PATTERN.search(key):
                return []
            if key.endswith(".weight"):
                if OPROJ_WEIGHT_PATTERN.match(key) or DENSE_DOWN_PROJ_PATTERN.match(key):
                    return []

        # QKV merge
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
        pending_qkv = self._qkv_buffer.get_pending()
        if pending_qkv:
            warnings.warn(f"Incomplete QKV merge groups after loading: {pending_qkv}")
        if self._qlora_buffer is not None:
            self._qlora_buffer.set_inline_metadata()
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
