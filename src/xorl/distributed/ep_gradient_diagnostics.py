"""Opt-in diagnostics for replicated EP gradient state."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import torch


logger = logging.getLogger(__name__)


def gradient_trace_enabled() -> bool:
    """Return whether expensive stage-by-stage replica tracing is enabled."""

    return os.environ.get("XORL_TRACE_ADAPTER_REPLICA_GRADIENTS") == "1"


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _record_value(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"present": False, "shape": None, "dtype": None, "digest": None, "value": None}
    value = value.detach().to(device="cpu").contiguous()
    return {
        "present": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "digest": _tensor_digest(value),
        "value": value,
    }


def _difference(values: list[torch.Tensor]) -> tuple[float | None, float | None, bool]:
    if not values:
        return None, None, True
    reference = values[0]
    exact = all(torch.equal(reference, value) for value in values[1:])
    if exact:
        return 0.0, 0.0, True
    reference_fp32 = reference.to(torch.float32)
    max_abs = 0.0
    max_rel = 0.0
    for value in values[1:]:
        if tuple(value.shape) != tuple(reference.shape):
            return float("inf"), float("inf"), False
        difference = (value.to(torch.float32) - reference_fp32).abs()
        max_abs = max(max_abs, float(difference.max().item()) if difference.numel() else 0.0)
        denominator = reference_fp32.abs().maximum(torch.full_like(reference_fp32, 1e-12))
        relative = difference / denominator
        max_rel = max(max_rel, float(relative.max().item()) if relative.numel() else 0.0)
    return max_abs, max_rel, False


@torch.no_grad()
def trace_replicated_gradient_stage(
    *,
    stage: str,
    values: Mapping[str, torch.Tensor | None],
    ep_group: torch.distributed.ProcessGroup,
) -> None:
    """Gather and log per-replica hashes and numerical differences.

    This is intentionally expensive and disabled by default. It gathers the
    small replicated adapter rectangles as CPU objects so the trace can report
    participating ranks, exact equality, and max absolute/relative differences
    without introducing another production synchronization path.
    """

    if not gradient_trace_enabled():
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if torch.distributed.get_world_size(ep_group) <= 1:
        return

    local = {name: _record_value(value) for name, value in sorted(values.items())}
    gathered: list[dict[str, dict[str, Any]] | None] = [None] * torch.distributed.get_world_size(ep_group)
    torch.distributed.all_gather_object(gathered, local, group=ep_group)
    if torch.distributed.get_rank(ep_group) != 0:
        return

    payload = []
    for name in sorted({name for rank_values in gathered if rank_values for name in rank_values}):
        records = [rank_values[name] for rank_values in gathered if rank_values is not None and name in rank_values]
        present = [index for index, record in enumerate(records) if record["present"]]
        tensors = [record["value"] for record in records if record["present"]]
        max_abs, max_rel, exact_equal = _difference(tensors)
        payload.append(
            {
                "name": name,
                "participating_ranks": present,
                "hashes": [record["digest"] for record in records],
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "exact_equal": exact_equal and len(present) == len(records),
            }
        )

    logger.info("EP_REPLICA_GRADIENT_TRACE %s", json.dumps({"stage": stage, "values": payload}, sort_keys=True))
