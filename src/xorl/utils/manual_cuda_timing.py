"""Lightweight CUDA-event scopes for model-specific phase timing.

This module complements trainer-level phase timing and hook-based component
timing. It lets model code add fine-grained CUDA-event scopes without depending
on the Trainer class. Timings are accumulated per process and drained by the
trainer at the end of a step.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import torch

from xorl.utils.device import get_device_type


_enabled = False
_mode = "idle"
_pairs: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)


def set_manual_cuda_timing_enabled(enabled: bool) -> None:
    """Enable or disable manual CUDA-event timing for the current process."""
    global _enabled
    _enabled = bool(enabled) and get_device_type() == "cuda"
    if not _enabled:
        reset_manual_cuda_timing()


def set_manual_cuda_timing_mode(mode: str) -> None:
    """Set the current timing mode: ``fwd``, ``bwd``/recompute, or ``idle``."""
    global _mode
    if mode not in ("fwd", "bwd", "idle"):
        raise ValueError(f"invalid manual CUDA timing mode: {mode}")
    _mode = mode


def reset_manual_cuda_timing() -> None:
    """Clear all accumulated events."""
    _pairs.clear()


def _phase_name(name: str) -> str | None:
    if not _enabled or _mode == "idle" or get_device_type() != "cuda":
        return None
    if _mode == "fwd":
        return f"fwd_{name}"
    if _mode == "bwd":
        # During activation checkpointing, model forward code runs under
        # backward. These scopes therefore describe recompute work.
        return f"recompute_{name}"
    return None


@contextmanager
def manual_cuda_timing_scope(name: str) -> Iterator[None]:
    """Record a CUDA-event scope under the current manual timing mode."""
    phase = _phase_name(name)
    if phase is None:
        yield
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _pairs[phase].append((start, end))


def drain_manual_cuda_timing(*, synchronize: bool = True) -> Dict[str, float]:
    """Return accumulated timings in seconds and clear the accumulator."""
    if not _pairs:
        return {}
    if synchronize and get_device_type() == "cuda":
        torch.cuda.synchronize()

    result: Dict[str, float] = {}
    for phase, events in _pairs.items():
        total_ms = 0.0
        valid_events = 0
        for start, end in events:
            try:
                total_ms += start.elapsed_time(end)
            except ValueError as exc:
                if "Both events must be recorded" not in str(exc):
                    raise
                continue
            valid_events += 1
        if valid_events:
            result[phase] = total_ms / 1000.0
    reset_manual_cuda_timing()
    return result
