"""Non-invasive bubble/memory/P2P profiling for torch.distributed.pipelining schedules.

``PPBubbleProfiler`` wraps an already-built pipeline schedule and monkey-patches its
PipelineStage compute entry points (``forward_one_chunk``, ``backward_one_chunk``,
``backward_weight_one_chunk`` -- the methods every torch 2.10 schedule dispatches
compute through) with CUDA-event bookended wrappers. This measures per-rank busy
time, bubble fraction, peak memory, and estimated P2P bytes for any schedule step
without modifying xorl or torch source.

Usage:
    profiler = PPBubbleProfiler(schedule)
    for _ in range(warmup_steps):
        schedule.step(...)
    with profiler.step_scope():
        schedule.step(...)
    stats = profiler.report()
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Optional

import torch


__all__ = [
    "PPBubbleProfiler",
    "estimate_p2p_bytes_per_step",
    "merge_busy_intervals",
]


# Stage compute entry points called by all torch 2.10 schedules (single-stage,
# multi-stage, and the zero-bubble _PipelineScheduleRuntime paths).
_STAGE_COMPUTE_METHODS = ("forward_one_chunk", "backward_one_chunk", "backward_weight_one_chunk")
_ACTION_TYPES = ("fwd", "bwd_full", "bwd_input", "bwd_weight")


def merge_busy_intervals(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length of the union of ``(start, end)`` intervals; overlaps counted once."""
    total = 0.0
    cur_start: Optional[float] = None
    cur_end = 0.0
    for start, end in sorted((s, e) for s, e in intervals if e > s):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def estimate_p2p_bytes_per_step(stages: Iterable[Any], n_microbatches: int) -> Optional[int]:
    """Estimated P2P bytes sent+received by this rank per schedule step.

    Derived from each local stage's recv-buffer sizes (``args_recv_info``) and output
    metas (``act_send_info`` x ``get_outputs_meta``), times ``n_microbatches``; grad
    traffic is assumed to mirror activation traffic in reverse. Same-rank
    adjacent-stage handoffs (the V-schedule local special case) are excluded via
    ``stage_index_to_group_rank``.

    Limitations: assumes every activation/grad tensor is exchanged for every
    microbatch, ignores init-time shape-exchange P2P and non-tensor metadata, and
    requires the stage infra populated by the first ``step()``. Returns None when
    the estimate is unavailable (e.g. before the first step).
    """
    total = 0
    try:
        for stage in stages:
            # A one-stage pipeline has no neighboring rank and therefore no
            # activation/gradient P2P. Torch does not necessarily populate
            # args_recv_info/act_send_info for this degenerate case, so avoid
            # treating absent shape metadata as an unavailable estimate.
            if stage.is_first and stage.is_last:
                continue
            stage_to_rank = stage.stage_index_to_group_rank
            prev_local = not stage.is_first and stage_to_rank.get(stage.stage_index - 1) == stage.group_rank
            next_local = not stage.is_last and stage_to_rank.get(stage.stage_index + 1) == stage.group_rank
            fwd_recv = sum(_nbytes(info.buffer) for info in stage.args_recv_info[0] if hasattr(info, "buffer"))
            outputs_meta = stage.get_outputs_meta()
            fwd_send = 0
            for idx, dst_list in stage.act_send_info.items():
                fwd_send += _nbytes(outputs_meta[idx]) * len([d for d in dst_list if d is not None])
            stage_bytes = (0 if prev_local else fwd_recv) + (0 if next_local else fwd_send)
            if stage.has_backward:
                stage_bytes += (0 if next_local else fwd_send) + (0 if prev_local else fwd_recv)
            total += stage_bytes * n_microbatches
    except (AttributeError, KeyError, IndexError, RuntimeError):
        return None
    return total


def _classify_action(method_name: str, args: tuple, kwargs: dict) -> str:
    """Map a stage compute call to fwd/bwd_full/bwd_input/bwd_weight."""
    if method_name == "forward_one_chunk":
        return "fwd"
    if method_name == "backward_weight_one_chunk":
        return "bwd_weight"
    # backward_one_chunk(bwd_chunk_id, loss=None, full_backward=True, last_backward=False)
    full_backward = kwargs.get("full_backward", args[2] if len(args) > 2 else True)
    return "bwd_full" if full_backward else "bwd_input"


class PPBubbleProfiler:
    """Per-rank pipeline bubble/memory/P2P profiler for a built pipeline schedule.

    Correct with multiple stages per rank: each compute action records a
    (start, end) CUDA-event pair; at report time all pairs are placed on a common
    timeline via elapsed times from a shared step-start event and their union is
    merged on CPU, so busy time is never double-counted.
    ``bubble_fraction = 1 - busy_time / step_time``.
    """

    def __init__(self, schedule: Any) -> None:
        self.schedule = schedule
        self.stages = self._stages_from_schedule(schedule)
        self._patched: list[tuple[Any, str]] = []
        self._active = False
        self._actions: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._pending: list[dict[str, Any]] = []
        self._reports: list[dict[str, Any]] = []
        self._patch()

    @staticmethod
    def _stages_from_schedule(schedule: Any) -> list[Any]:
        if hasattr(schedule, "_stages"):
            return list(schedule._stages)
        if hasattr(schedule, "_stage"):
            return [schedule._stage]
        raise ValueError(f"Cannot locate PipelineStage(s) on schedule of type {type(schedule).__name__}")

    def _patch(self) -> None:
        for stage in self.stages:
            for name in _STAGE_COMPUTE_METHODS:
                if name in stage.__dict__:
                    raise RuntimeError(f"'{name}' is already instance-patched on stage {stage.stage_index}")
                setattr(stage, name, self._make_wrapper(getattr(stage, name), name))
                self._patched.append((stage, name))

    def _make_wrapper(self, orig: Callable, method_name: str) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self._active:
                return orig(*args, **kwargs)
            action = _classify_action(method_name, args, kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return orig(*args, **kwargs)
            finally:
                end.record()
                self._actions.append((action, start, end))

        return wrapper

    def close(self) -> None:
        """Restore the original stage compute methods."""
        for stage, name in self._patched:
            stage.__dict__.pop(name, None)
        self._patched.clear()

    @contextlib.contextmanager
    def step_scope(self) -> Iterator[None]:
        """Instrument one ``schedule.step(...)`` call; results retrieved via ``report()``."""
        if not torch.cuda.is_available():
            raise RuntimeError("PPBubbleProfiler.step_scope requires CUDA")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        self._actions = []
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        self._active = True
        try:
            yield
        finally:
            self._active = False
            step_end.record()
            # Allocator peaks are updated host-side at enqueue time, so reading here is safe.
            self._pending.append(
                {
                    "step_start": step_start,
                    "step_end": step_end,
                    "actions": self._actions,
                    "mem_before": mem_before,
                    "peak_mem": torch.cuda.max_memory_allocated(),
                }
            )
            self._actions = []

    def _finalize_pending(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        n_microbatches = getattr(self.schedule, "_n_microbatches", None)
        p2p_bytes = estimate_p2p_bytes_per_step(self.stages, n_microbatches) if n_microbatches else None
        for rec in self._pending:
            step_start = rec["step_start"]
            step_time_s = step_start.elapsed_time(rec["step_end"]) / 1e3
            intervals = []
            counts = dict.fromkeys(_ACTION_TYPES, 0)
            times = dict.fromkeys(_ACTION_TYPES, 0.0)
            for action, start, end in rec["actions"]:
                t0 = step_start.elapsed_time(start) / 1e3
                t1 = step_start.elapsed_time(end) / 1e3
                intervals.append((t0, t1))
                counts[action] += 1
                times[action] += max(0.0, t1 - t0)
            busy_time_s = merge_busy_intervals(intervals)
            self._reports.append(
                {
                    "busy_time_s": busy_time_s,
                    "step_time_s": step_time_s,
                    "bubble_fraction": (1.0 - busy_time_s / step_time_s) if step_time_s > 0 else None,
                    "actions": {a: {"count": counts[a], "time_s": times[a]} for a in _ACTION_TYPES},
                    "peak_memory_bytes": rec["peak_mem"] - rec["mem_before"],
                    "p2p_bytes": p2p_bytes,
                    "num_local_stages": len(self.stages),
                }
            )
        self._pending.clear()

    def report(self) -> dict[str, Any]:
        """Stats dict for the most recent instrumented step."""
        self._finalize_pending()
        if not self._reports:
            raise RuntimeError("No instrumented steps; run schedule.step() inside profiler.step_scope() first")
        return self._reports[-1]

    def report_all(self) -> list[dict[str, Any]]:
        """Stats dicts for all instrumented steps, in order."""
        self._finalize_pending()
        return list(self._reports)

    def aggregate(self, warmup: int = 0) -> dict[str, Any]:
        """Mean busy/step/bubble over instrumented steps, skipping the first ``warmup``."""
        reports = self.report_all()[warmup:]
        if not reports:
            raise RuntimeError("No instrumented steps to aggregate")
        n = len(reports)
        busy = sum(r["busy_time_s"] for r in reports) / n
        step = sum(r["step_time_s"] for r in reports) / n
        return {
            "num_steps": n,
            "busy_time_s": busy,
            "step_time_s": step,
            "bubble_fraction": (1.0 - busy / step) if step > 0 else None,
            "actions": {
                a: {
                    "count": reports[-1]["actions"][a]["count"],
                    "time_s": sum(r["actions"][a]["time_s"] for r in reports) / n,
                }
                for a in _ACTION_TYPES
            },
            "peak_memory_bytes": max(r["peak_memory_bytes"] for r in reports),
            "p2p_bytes": reports[-1]["p2p_bytes"],
            "num_local_stages": reports[-1]["num_local_stages"],
        }
