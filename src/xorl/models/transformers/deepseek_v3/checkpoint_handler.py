"""Checkpoint handler for DeepseekV3 / Kimi-K2.5."""

import math
import re
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from ...checkpoint_handlers.base import CheckpointHandler
from ...checkpoint_handlers.buffers import (
    ExpertWeightBuffer,
    parse_expert_key,
)


class DeepseekV3CheckpointHandler(CheckpointHandler):
    _STACKED_EXPERT_SPLIT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj$")
    _FUSED_EXPERT_GATE_UP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
    _FUSED_EXPERT_DOWN_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")
    _COMPRESSED_EXPERT_PATTERN = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight_packed|weight_scale|weight_shape)$"
    )
    _COMPRESSED_EXPERT_SUFFIXES = frozenset({"weight_packed", "weight_scale", "weight_shape"})

    def __init__(
        self,
        num_experts: int,
        ep_rank: int = 0,
        ep_size: int = 1,
        checkpoint_has_per_expert: bool = True,
        packed_expert_num_bits: int = 4,
        packed_expert_group_size: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._dequant_device = None if device is None or device.type == "meta" else device
        self._output_dtype = dtype
        self._expert_buffer: Optional[ExpertWeightBuffer] = None
        if checkpoint_has_per_expert:
            self._expert_buffer = ExpertWeightBuffer(
                num_experts,
                ep_rank=ep_rank,
                ep_size=ep_size,
                device=self._dequant_device,
            )
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._local_num_experts = num_experts // ep_size
        self._expert_start = ep_rank * self._local_num_experts
        self._expert_end = self._expert_start + self._local_num_experts
        self._stacked_gate_up_pending: Dict[int, Dict[str, torch.Tensor]] = {}
        self._packed_expert_num_bits = packed_expert_num_bits
        self._packed_expert_group_size = packed_expert_group_size
        self._packed_expert_pending: Dict[Tuple[int, int, str], Dict[str, torch.Tensor]] = {}
        self._skipped_packed_expert_suffixes: Dict[Tuple[int, int, str], Set[str]] = defaultdict(set)

    def _normalize_key(self, key: str) -> Optional[str]:
        if key.startswith("vision_tower.") or key.startswith("mm_projector."):
            return None
        if key.startswith("language_model."):
            return key.removeprefix("language_model.")
        return key

    def _slice_expert_tensor_for_ep(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._ep_size == 1:
            return tensor
        return tensor[self._expert_start : self._expert_end].contiguous()

    def _parse_compressed_expert_key(self, key: str) -> Optional[Tuple[int, int, str, str]]:
        match = self._COMPRESSED_EXPERT_PATTERN.match(key)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)

    def _unpack_packed_int32_tensor(self, tensor: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        if tensor.dtype != torch.int32:
            tensor = tensor.to(torch.int32)
        if tensor.ndim != 2:
            raise ValueError(f"Expected packed expert tensor to be rank-2, got shape {tuple(tensor.shape)}")

        num_bits = self._packed_expert_num_bits
        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1
        rows, cols = original_shape

        bit_shifts = torch.arange(pack_factor, device=tensor.device, dtype=torch.int32) * num_bits
        unpacked = ((tensor.unsqueeze(-1) >> bit_shifts) & mask).reshape(rows, -1)
        unpacked = unpacked[:, :cols]

        offset = 1 << (num_bits - 1)
        return (unpacked - offset).to(torch.int8)

    def _dequantize_packed_expert_weight(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        if self._dequant_device is not None:
            packed = packed.to(self._dequant_device)
            scale = scale.to(self._dequant_device)
            shape = shape.to(self._dequant_device)

        original_shape = tuple(int(dim) for dim in shape.flatten().tolist())
        if len(original_shape) != 2:
            raise ValueError(f"Expected rank-2 expert weight shape, got {original_shape}")

        rows, cols = original_shape
        groups = math.ceil(cols / self._packed_expert_group_size)
        unpacked = self._unpack_packed_int32_tensor(packed, (rows, cols))
        compute_dtype = scale.dtype if scale.is_floating_point() else torch.float32
        dequantized = unpacked.to(compute_dtype)

        if scale.ndim == 0:
            result = dequantized * scale.to(compute_dtype)
            return result.to(self._output_dtype) if self._output_dtype is not None else result

        if scale.ndim == 1:
            if scale.numel() == rows:
                result = dequantized * scale.to(compute_dtype).view(rows, 1)
                return result.to(self._output_dtype) if self._output_dtype is not None else result
            if scale.numel() == groups:
                scale = scale.to(compute_dtype).view(1, groups)
            else:
                raise ValueError(
                    f"Unsupported packed expert scale shape {tuple(scale.shape)} for original shape {original_shape}"
                )
        elif scale.ndim == 2:
            if scale.shape == (rows, 1):
                result = dequantized * scale.to(compute_dtype)
                return result.to(self._output_dtype) if self._output_dtype is not None else result
            if scale.shape[0] in (1, rows) and scale.shape[1] == groups:
                scale = scale.to(compute_dtype)
            elif scale.shape[1] in (1, rows) and scale.shape[0] == groups:
                scale = scale.t().contiguous().to(compute_dtype)
            else:
                raise ValueError(
                    f"Unsupported packed expert scale shape {tuple(scale.shape)} for original shape {original_shape}"
                )
        else:
            raise ValueError(f"Unsupported packed expert scale rank {scale.ndim} for shape {original_shape}")

        padded_cols = groups * self._packed_expert_group_size
        if padded_cols > cols:
            dequantized = torch.nn.functional.pad(dequantized, (0, padded_cols - cols))
        dequantized = dequantized.unflatten(1, (groups, self._packed_expert_group_size))

        if scale.shape[0] == 1 and rows != 1:
            scale = scale.expand(rows, -1)

        result = (dequantized * scale.unsqueeze(-1)).flatten(1)[:, :cols].contiguous()
        if self._output_dtype is not None and result.dtype != self._output_dtype:
            result = result.to(self._output_dtype)
        return result

    def _handle_compressed_expert_weight(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        parsed = self._parse_compressed_expert_key(key)
        if parsed is None:
            return None

        layer_idx, expert_idx, proj, suffix = parsed
        buffer_key = (layer_idx, expert_idx, proj)
        pending = self._packed_expert_pending.setdefault(buffer_key, {})
        pending[suffix] = tensor

        if not self._COMPRESSED_EXPERT_SUFFIXES.issubset(pending):
            return []

        dense_weight = self._dequantize_packed_expert_weight(
            packed=pending["weight_packed"],
            scale=pending["weight_scale"],
            shape=pending["weight_shape"],
        )
        del self._packed_expert_pending[buffer_key]

        if self._expert_buffer is None:
            return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}_proj.weight", dense_weight)]

        self._expert_buffer.add(layer_idx, expert_idx, proj, dense_weight)
        return self._maybe_finalize_per_expert_merge(layer_idx, proj)

    def _maybe_finalize_per_expert_merge(self, layer_idx: int, proj: str) -> List[Tuple[str, torch.Tensor]]:
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

    def _handle_stacked_expert_weights(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        gate_up_match = self._FUSED_EXPERT_GATE_UP_PATTERN.match(key)
        if gate_up_match is not None:
            # xorl checkpoints already store fused expert weights in native
            # [experts, hidden, 2 * intermediate] layout. Only EP slicing is needed.
            return [(key, self._slice_expert_tensor_for_ep(tensor))]

        down_match = self._FUSED_EXPERT_DOWN_PATTERN.match(key)
        if down_match is not None:
            # xorl checkpoints already store fused expert weights in native
            # [experts, intermediate, hidden] layout. Only EP slicing is needed.
            return [(key, self._slice_expert_tensor_for_ep(tensor))]

        split_match = self._STACKED_EXPERT_SPLIT_PATTERN.match(key)
        if split_match is None:
            return None

        layer_idx = int(split_match.group(1))
        proj = split_match.group(2)
        tensor = self._slice_expert_tensor_for_ep(tensor)

        if proj == "down":
            return [(f"model.layers.{layer_idx}.mlp.experts.down_proj", tensor.transpose(1, 2).contiguous())]

        pending = self._stacked_gate_up_pending.setdefault(layer_idx, {})
        pending[proj] = tensor
        if "gate" in pending and "up" in pending:
            gate = pending.pop("gate")
            up = pending.pop("up")
            del self._stacked_gate_up_pending[layer_idx]
            return [
                (
                    f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                    torch.cat([gate.transpose(1, 2), up.transpose(1, 2)], dim=2).contiguous(),
                )
            ]
        return []

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        has_ep_filter = self._expert_buffer is not None and not (
            self._expert_buffer.expert_start == 0 and self._expert_buffer.expert_end == self._expert_buffer.num_experts
        )
        if not has_ep_filter:
            return None

        ep_start = self._expert_buffer.expert_start
        ep_end = self._expert_buffer.expert_end

        def _should_skip(key: str) -> bool:
            normalized_key = self._normalize_key(key)
            if normalized_key is None:
                return True

            compressed = self._parse_compressed_expert_key(normalized_key)
            if compressed is not None:
                _, expert_idx, _, _ = compressed
                return expert_idx < ep_start or expert_idx >= ep_end

            parsed = parse_expert_key(normalized_key)
            if parsed is None:
                return False
            _, expert_idx, _ = parsed
            return expert_idx < ep_start or expert_idx >= ep_end

        return _should_skip

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        key = self._normalize_key(key)
        if key is None:
            return []

        if key.endswith(".input_scale"):
            return []

        compressed_expert_results = self._handle_compressed_expert_weight(key, tensor)
        if compressed_expert_results is not None:
            return compressed_expert_results

        stacked_expert_results = self._handle_stacked_expert_weights(key, tensor)
        if stacked_expert_results is not None:
            return stacked_expert_results

        if self._expert_buffer is not None:
            parsed = parse_expert_key(key)
            if parsed is not None:
                layer_idx, expert_idx, proj = parsed
                self._expert_buffer.add(layer_idx, expert_idx, proj, tensor)
                return self._maybe_finalize_per_expert_merge(layer_idx, proj)

        return [(key, tensor)]

    def on_skip_weight(self, key: str) -> List[Tuple[str, torch.Tensor]]:
        if self._expert_buffer is None:
            return []
        key = self._normalize_key(key)
        if key is None:
            return []

        compressed = self._parse_compressed_expert_key(key)
        if compressed is not None:
            layer_idx, expert_idx, proj, suffix = compressed
            skipped = self._skipped_packed_expert_suffixes[(layer_idx, expert_idx, proj)]
            skipped.add(suffix)
            if self._COMPRESSED_EXPERT_SUFFIXES.issubset(skipped):
                del self._skipped_packed_expert_suffixes[(layer_idx, expert_idx, proj)]
                self._expert_buffer.count_skipped(layer_idx, proj)
                return self._maybe_finalize_per_expert_merge(layer_idx, proj)
            return []

        parsed = parse_expert_key(key)
        if parsed is None:
            return []
        layer_idx, _expert_idx, proj = parsed
        self._expert_buffer.count_skipped(layer_idx, proj)
        return self._maybe_finalize_per_expert_merge(layer_idx, proj)

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
        if self._expert_buffer is not None:
            pending = self._expert_buffer.get_pending_counts()
            if pending:
                warnings.warn(f"Incomplete expert weights after loading: {pending}")
        if self._stacked_gate_up_pending:
            warnings.warn(f"Incomplete stacked expert gate/up merges after loading: {self._stacked_gate_up_pending}")
        if self._packed_expert_pending:
            pending_triplets = {key: sorted(value.keys()) for key, value in self._packed_expert_pending.items()}
            warnings.warn(f"Incomplete packed expert triplets after loading: {pending_triplets}")
        if self._skipped_packed_expert_suffixes:
            pending_skips = {key: sorted(value) for key, value in self._skipped_packed_expert_suffixes.items()}
            warnings.warn(f"Incomplete skipped packed expert triplets after loading: {pending_skips}")
        return []

    def on_save_weight(self, param_name: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
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

        return [(param_name, tensor)]
