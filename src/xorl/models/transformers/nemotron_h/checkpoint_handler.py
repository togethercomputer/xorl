"""Checkpoint handler for NVIDIA Nemotron-3-Ultra (nemotron_h).

HF checkpoints store the module tree under ``backbone.*`` (plus ``lm_head`` and
the ignored ``mtp.*`` MTP head); xorl uses ``model.*`` internally (see the
modeling module docstring). On load this handler renames ``backbone.`` →
``model.``, drops ``mtp.*``, and stacks the 512 per-expert non-gated weights

    backbone.layers.{i}.mixer.experts.{e}.up_proj.weight   [I, latent]
    backbone.layers.{i}.mixer.experts.{e}.down_proj.weight [latent, I]

into the GKN ``[num_experts, K_in, N_out]`` parameters

    model.layers.{i}.mixer.experts.gate_up_proj  [E, latent, I]
    model.layers.{i}.mixer.experts.down_proj     [E, I, latent]

(``ExpertWeightBuffer`` transposes each HF ``[out, in]`` weight to
``[in, out]`` while stacking). The stacked 3D expert layout used by the
transformers 5.x in-memory state dict (``experts.up_proj`` ``[E, out, in]``)
is supported too. ``on_save_weight`` emits the per-expert ``backbone.*``
layout, the exact inverse of the per-expert load path, so DCP→HF export and
weight sync produce HF-checkpoint-compatible names.
"""

import re
import warnings
from typing import Callable, List, Optional, Set, Tuple

import torch

from ....utils import logging
from ...checkpoint_handlers.base import CheckpointHandler
from ...checkpoint_handlers.buffers import ExpertWeightBuffer


logger = logging.get_logger(__name__)

# Per-expert HF weight keys (after backbone. -> model. normalization).
PER_EXPERT_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")
# Stacked HF NemotronHExperts 3D params ([E, out, in] layout, transformers >= 5.x in-memory format).
STACKED_HF_EXPERT_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mixer\.experts\.(up|down)_proj$")
# xorl GKN fused expert key (only gate_up_proj is unambiguous: a bare ``down_proj``
# 3D key always means the HF stacked layout — our save side emits per-expert keys).
FUSED_GATE_UP_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mixer\.experts\.gate_up_proj$")
# Fused expert parameter names on the save side.
_SAVE_GATE_UP_SUFFIX = ".mixer.experts.gate_up_proj"
_SAVE_DOWN_SUFFIX = ".mixer.experts.down_proj"


def checkpoint_has_per_expert_nemotron_weights(checkpoint_keys: Set[str]) -> bool:
    """True when the checkpoint stores experts as per-expert ``up/down_proj`` weights."""
    for key in checkpoint_keys:
        normalized = _normalize_prefix(key)
        if normalized is not None and PER_EXPERT_KEY_PATTERN.match(normalized):
            return True
    return False


def _normalize_prefix(key: str) -> Optional[str]:
    """Map raw checkpoint keys to xorl ``model.*`` names; None = drop the key."""
    if key.startswith("mtp."):
        return None
    if key.startswith("backbone."):
        return "model." + key.removeprefix("backbone.")
    return key


def denormalize_param_name(param_name: str) -> str:
    """Map a xorl ``model.*`` param name to the published HF ``backbone.*`` name."""
    if param_name.startswith("model."):
        return "backbone." + param_name.removeprefix("model.")
    return param_name


class NemotronHCheckpointHandler(CheckpointHandler):
    def __init__(
        self,
        num_experts: int,
        ep_rank: int = 0,
        ep_size: int = 1,
        checkpoint_has_per_expert: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        del dtype  # bf16 checkpoints load as-is; kept for handler-interface parity
        stacking_device = None if device is None or device.type == "meta" else device
        self._expert_buffer: Optional[ExpertWeightBuffer] = None
        if checkpoint_has_per_expert:
            self._expert_buffer = ExpertWeightBuffer(
                num_experts,
                ep_rank=ep_rank,
                ep_size=ep_size,
                device=stacking_device,
            )
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._local_num_experts = num_experts // ep_size
        self._expert_start = ep_rank * self._local_num_experts
        self._expert_end = self._expert_start + self._local_num_experts
        self._mtp_skip_logged = False

    # ------------------------------------------------------------------
    # Load side
    # ------------------------------------------------------------------

    def _normalize_key(self, key: str) -> Optional[str]:
        normalized = _normalize_prefix(key)
        if normalized is None and not self._mtp_skip_logged:
            logger.info_rank0("Ignoring mtp.* keys in NemotronH checkpoint (multi-token-prediction head unused).")
            self._mtp_skip_logged = True
        return normalized

    def _slice_expert_tensor_for_ep(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._ep_size == 1:
            return tensor
        return tensor[self._expert_start : self._expert_end].contiguous()

    def _maybe_finalize_expert_stack(self, layer_idx: int, proj: str) -> List[Tuple[str, torch.Tensor]]:
        if self._expert_buffer is None or not self._expert_buffer.is_complete(layer_idx, proj):
            return []
        stacked = self._expert_buffer.pop_stacked(layer_idx, proj)
        param = "gate_up_proj" if proj == "up" else "down_proj"
        return [(f"model.layers.{layer_idx}.mixer.experts.{param}", stacked)]

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        key = self._normalize_key(key)
        if key is None:
            return []

        per_expert_match = PER_EXPERT_KEY_PATTERN.match(key)
        if per_expert_match is not None and self._expert_buffer is not None:
            layer_idx = int(per_expert_match.group(1))
            expert_idx = int(per_expert_match.group(2))
            proj = per_expert_match.group(3)
            self._expert_buffer.add(layer_idx, expert_idx, proj, tensor)
            return self._maybe_finalize_expert_stack(layer_idx, proj)

        stacked_match = STACKED_HF_EXPERT_KEY_PATTERN.match(key)
        if stacked_match is not None:
            # HF stacked 3D layout [E, out, in] -> GKN [E, in, out].
            layer_idx = int(stacked_match.group(1))
            proj = stacked_match.group(2)
            tensor = self._slice_expert_tensor_for_ep(tensor).transpose(1, 2).contiguous()
            param = "gate_up_proj" if proj == "up" else "down_proj"
            return [(f"model.layers.{layer_idx}.mixer.experts.{param}", tensor)]

        if FUSED_GATE_UP_KEY_PATTERN.match(key) is not None:
            # xorl-internal GKN checkpoint; only EP slicing is needed.
            return [(key, self._slice_expert_tensor_for_ep(tensor))]

        return [(key, tensor)]

    def on_skip_weight(self, key: str) -> List[Tuple[str, torch.Tensor]]:
        key = self._normalize_key(key)
        if key is None or self._expert_buffer is None:
            return []
        per_expert_match = PER_EXPERT_KEY_PATTERN.match(key)
        if per_expert_match is None:
            return []
        layer_idx = int(per_expert_match.group(1))
        proj = per_expert_match.group(3)
        self._expert_buffer.count_skipped(layer_idx, proj)
        return self._maybe_finalize_expert_stack(layer_idx, proj)

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        has_ep_filter = self._expert_buffer is not None and not (
            self._expert_buffer.expert_start == 0 and self._expert_buffer.expert_end == self._expert_buffer.num_experts
        )
        if not has_ep_filter:
            return None

        ep_start = self._expert_buffer.expert_start
        ep_end = self._expert_buffer.expert_end

        def _should_skip(key: str) -> bool:
            normalized = _normalize_prefix(key)
            if normalized is None:
                return True
            match = PER_EXPERT_KEY_PATTERN.match(normalized)
            if match is None:
                return False
            expert_idx = int(match.group(2))
            return expert_idx < ep_start or expert_idx >= ep_end

        return _should_skip

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
        if self._expert_buffer is not None:
            pending = self._expert_buffer.get_pending_counts()
            if pending:
                warnings.warn(f"Incomplete NemotronH expert weights after loading: {pending}")
        return []

    # ------------------------------------------------------------------
    # Save side (exact inverse of the load mapping)
    # ------------------------------------------------------------------

    def _denormalize_key(self, param_name: str) -> str:
        return denormalize_param_name(param_name)

    def on_save_weight(self, param_name: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        if param_name.endswith(_SAVE_GATE_UP_SUFFIX) or param_name.endswith(_SAVE_DOWN_SUFFIX):
            proj = "up" if param_name.endswith(_SAVE_GATE_UP_SUFFIX) else "down"
            prefix = self._denormalize_key(param_name.rsplit(".", 1)[0])
            unstacked = tensor.transpose(1, 2).contiguous()
            return [
                (f"{prefix}.{expert_idx}.{proj}_proj.weight", unstacked[expert_idx])
                for expert_idx in range(tensor.shape[0])
            ]
        return [(self._denormalize_key(param_name), tensor)]


__all__ = [
    "NemotronHCheckpointHandler",
    "checkpoint_has_per_expert_nemotron_weights",
    "denormalize_param_name",
]
