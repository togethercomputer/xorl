"""Checkpoint transforms for MiniMax M3 text-only xorl support."""

from __future__ import annotations

import re
import warnings
from typing import Callable

import torch

from xorl.models.checkpoint_handlers.base import CheckpointHandler
from xorl.models.checkpoint_handlers.buffers import ExpertWeightBuffer, GateUpMergeBuffer


_NON_LANGUAGE_PREFIXES = (
    "vision_tower.",
    "model.vision_tower.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
    "patch_merge_mlp.",
    "model.patch_merge_mlp.",
)

_MM_STANDALONE_KEYS = {
    "image_newline",
    "model.image_newline",
}

_MINIMAX_EXPERT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w([123])\.weight$")
_MINIMAX_GATE_PATTERN = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$")
_MINIMAX_BIAS_PATTERN = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.e_score_correction_bias$")
_MINIMAX_SHARED_PATTERN = re.compile(
    r"^(model\.layers\.(\d+)\.block_sparse_moe\.shared_experts)\.(gate|up|down)_proj\.weight$"
)
_MINIMAX_INTERNAL_EXPERT_GATE_UP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_MINIMAX_INTERNAL_EXPERT_DOWN_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")

_EXPERT_PROJ = {"1": "gate", "2": "down", "3": "up"}


def normalize_minimax_m3_checkpoint_key(key: str) -> str:
    if key.startswith("model.language_model.model."):
        return "model." + key.removeprefix("model.language_model.model.")
    if key.startswith("language_model.model."):
        return "model." + key.removeprefix("language_model.model.")
    if key.startswith("model.language_model.lm_head."):
        return "lm_head." + key.removeprefix("model.language_model.lm_head.")
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key.removeprefix("language_model.lm_head.")
    if key.startswith("model.language_model."):
        return key.removeprefix("model.language_model.")
    if key.startswith("language_model."):
        return key.removeprefix("language_model.")
    return key


def is_minimax_m3_non_language_key(key: str) -> bool:
    key = normalize_minimax_m3_checkpoint_key(key)
    return key in _MM_STANDALONE_KEYS or key.startswith(_NON_LANGUAGE_PREFIXES)


class MiniMaxM3CheckpointHandler(CheckpointHandler):
    """Load MiniMax-M3 language weights into the xorl text-only module.

    MiniMax publishes the language model under ``language_model.*`` and stores
    sparse MoE experts as ``block_sparse_moe.experts.{i}.w{1,2,3}.weight``.
    xorl owns the text tensors directly under ``model.*`` and uses fused
    expert tensors in GKN layout.
    """

    def __init__(self, num_experts: int, ep_rank: int = 0, ep_size: int = 1):
        self._expert_buffer = ExpertWeightBuffer(num_experts, ep_rank=ep_rank, ep_size=ep_size)
        self._gate_up_buffer = GateUpMergeBuffer()
        self._shared_gate_up_buffer = GateUpMergeBuffer()
        self._num_experts = num_experts
        self._ep_rank = ep_rank
        self._ep_size = ep_size

    def _maybe_finalize_expert_merge(self, layer_idx: int, proj: str) -> list[tuple[str, torch.Tensor]]:
        if proj in {"gate", "up"}:
            if not (
                self._expert_buffer.is_complete(layer_idx, "gate") and self._expert_buffer.is_complete(layer_idx, "up")
            ):
                return []
            gate = self._expert_buffer.pop_stacked(layer_idx, "gate")
            up = self._expert_buffer.pop_stacked(layer_idx, "up")
            return [(ExpertWeightBuffer.get_gate_up_name(layer_idx), torch.cat([gate, up], dim=2))]

        if proj == "down" and self._expert_buffer.is_complete(layer_idx, "down"):
            return [
                (
                    ExpertWeightBuffer.get_fused_name(layer_idx, "down"),
                    self._expert_buffer.pop_stacked(layer_idx, "down"),
                )
            ]

        return []

    def _handle_minimax_expert(self, key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        match = _MINIMAX_EXPERT_PATTERN.match(key)
        if match is None:
            return None
        layer_idx = int(match.group(1))
        expert_idx = int(match.group(2))
        proj = _EXPERT_PROJ[match.group(3)]
        self._expert_buffer.add(layer_idx, expert_idx, proj, tensor)
        return self._maybe_finalize_expert_merge(layer_idx, proj)

    def _handle_minimax_shared(self, key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        match = _MINIMAX_SHARED_PATTERN.match(key)
        if match is None:
            return None
        old_prefix = match.group(1)
        layer_idx = int(match.group(2))
        proj = match.group(3)
        new_prefix = f"model.layers.{layer_idx}.mlp.shared_experts"
        if proj == "down":
            return [(f"{new_prefix}.down_proj.weight", tensor)]
        mapped = key.replace(old_prefix, new_prefix, 1)
        result = self._shared_gate_up_buffer.add(mapped, tensor)
        if result is not None:
            return [result]
        return []

    def get_skip_key_fn(self) -> Callable[[str], bool]:
        has_ep_filter = not (
            self._expert_buffer.expert_start == 0 and self._expert_buffer.expert_end == self._num_experts
        )
        expert_start = self._expert_buffer.expert_start
        expert_end = self._expert_buffer.expert_end

        def _should_skip(key: str) -> bool:
            key = normalize_minimax_m3_checkpoint_key(key)
            if is_minimax_m3_non_language_key(key):
                return True
            if has_ep_filter:
                match = _MINIMAX_EXPERT_PATTERN.match(key)
                if match is not None:
                    expert_idx = int(match.group(2))
                    return expert_idx < expert_start or expert_idx >= expert_end
            return False

        return _should_skip

    def on_skip_weight(self, key: str) -> list[tuple[str, torch.Tensor]]:
        key = normalize_minimax_m3_checkpoint_key(key)
        match = _MINIMAX_EXPERT_PATTERN.match(key)
        if match is None:
            return []
        layer_idx = int(match.group(1))
        proj = _EXPERT_PROJ[match.group(3)]
        self._expert_buffer.count_skipped(layer_idx, proj)
        return self._maybe_finalize_expert_merge(layer_idx, proj)

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        key = normalize_minimax_m3_checkpoint_key(key)
        if is_minimax_m3_non_language_key(key):
            return []

        expert_results = self._handle_minimax_expert(key, tensor)
        if expert_results is not None:
            return expert_results

        gate_match = _MINIMAX_GATE_PATTERN.match(key)
        if gate_match is not None:
            return [(f"model.layers.{gate_match.group(1)}.mlp.gate.weight", tensor)]

        bias_match = _MINIMAX_BIAS_PATTERN.match(key)
        if bias_match is not None:
            return [(f"model.layers.{bias_match.group(1)}.mlp.e_score_correction_bias", tensor)]

        shared_results = self._handle_minimax_shared(key, tensor)
        if shared_results is not None:
            return shared_results

        dense_result = self._gate_up_buffer.add(key, tensor)
        if dense_result is not None:
            return [dense_result]
        if self._gate_up_buffer.is_gate_up_key(key):
            return []

        return [(key, tensor)]

    def on_load_complete(self) -> list[tuple[str, torch.Tensor]]:
        pending = self._expert_buffer.get_pending_counts()
        if pending:
            warnings.warn(f"Incomplete MiniMax M3 expert weights after loading: {pending}")
        pending_gate_up = self._gate_up_buffer.get_pending()
        if pending_gate_up:
            warnings.warn(f"Incomplete MiniMax M3 dense gate/up merges after loading: {pending_gate_up}")
        pending_shared = self._shared_gate_up_buffer.get_pending()
        if pending_shared:
            warnings.warn(f"Incomplete MiniMax M3 shared expert gate/up merges after loading: {pending_shared}")
        return []

    def on_save_weight(self, param_name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        gate_up_match = _MINIMAX_INTERNAL_EXPERT_GATE_UP_PATTERN.match(param_name)
        if gate_up_match is not None:
            layer_idx = gate_up_match.group(1)
            half = tensor.shape[2] // 2
            gate = tensor[:, :, :half].transpose(1, 2).contiguous()
            up = tensor[:, :, half:].transpose(1, 2).contiguous()
            result = []
            for expert_idx in range(tensor.shape[0]):
                result.append(
                    (
                        f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight",
                        gate[expert_idx],
                    )
                )
                result.append(
                    (
                        f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight",
                        up[expert_idx],
                    )
                )
            return result

        down_match = _MINIMAX_INTERNAL_EXPERT_DOWN_PATTERN.match(param_name)
        if down_match is not None:
            layer_idx = down_match.group(1)
            down = tensor.transpose(1, 2).contiguous()
            return [
                (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight", down[expert_idx])
                for expert_idx in range(tensor.shape[0])
            ]

        if param_name.endswith(".mlp.gate.weight"):
            layer_prefix = param_name.rsplit(".mlp.gate.weight", 1)[0]
            return [(f"{layer_prefix}.block_sparse_moe.gate.weight", tensor)]

        if param_name.endswith(".mlp.e_score_correction_bias"):
            layer_prefix = param_name.rsplit(".mlp.e_score_correction_bias", 1)[0]
            return [(f"{layer_prefix}.block_sparse_moe.e_score_correction_bias", tensor)]

        if ".mlp.shared_experts.gate_up_proj." in param_name:
            prefix, suffix = param_name.rsplit(".gate_up_proj.", 1)
            half = tensor.shape[0] // 2
            hf_prefix = prefix.replace(".mlp.shared_experts", ".block_sparse_moe.shared_experts", 1)
            return [
                (f"{hf_prefix}.gate_proj.{suffix}", tensor[:half]),
                (f"{hf_prefix}.up_proj.{suffix}", tensor[half:]),
            ]

        if ".mlp.shared_experts.down_proj." in param_name:
            return [(param_name.replace(".mlp.shared_experts.", ".block_sparse_moe.shared_experts.", 1), tensor)]

        if ".mlp.gate_up_proj." in param_name:
            prefix, suffix = param_name.rsplit(".gate_up_proj.", 1)
            half = tensor.shape[0] // 2
            return [
                (f"{prefix}.gate_proj.{suffix}", tensor[:half]),
                (f"{prefix}.up_proj.{suffix}", tensor[half:]),
            ]

        return [(param_name, tensor)]


__all__ = ["MiniMaxM3CheckpointHandler", "is_minimax_m3_non_language_key", "normalize_minimax_m3_checkpoint_key"]
