"""Real-NCCL tests for the IS-metric cross-rank reduction primitives.

Targets the three helpers in ``xorl.server.runner.model_runner`` that
``dist.all_reduce`` IS metrics across process groups:

- ``_sp_allreduce_kl_metrics`` — per-mb CP/Ulysses reduction.
- ``ModelRunner._accumulate_is_metrics`` — cross-mb accumulation.
- ``ModelRunner._finalize_is_metrics`` — cross-DP reduction + finalization.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

from xorl.server.runner import model_runner as mr
from xorl.utils.device import get_nccl_backend


pytestmark = [pytest.mark.distributed]


def _setup_dist() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    return torch.device("cuda", local_rank)


def _make_metrics(device: torch.device, **values) -> dict:
    """Plain scalars → the {valid_tokens: int, others: float64-tensor} shape
    that the IS-metric helpers consume in production."""
    return {
        k: v if k == "valid_tokens" else torch.as_tensor(v, dtype=torch.float64, device=device)
        for k, v in values.items()
    }


def _finalize_with_world_dp(accumulated: dict, result: dict) -> None:
    """``_finalize_is_metrics`` resolves the DP group via ``get_parallel_state()``.
    Spinning up the full mesh for a unit test is overkill, so stub it to
    treat WORLD as the DP group for the duration of the call."""
    ps = SimpleNamespace(dp_enabled=True, dp_group=dist.group.WORLD)
    with patch.object(mr, "get_parallel_state", lambda: ps):
        mr.ModelRunner._finalize_is_metrics(accumulated, result)


# ---------------------------------------------------------------------------
# Case: per-mb CP/SP reduction (_sp_allreduce_kl_metrics)
# ---------------------------------------------------------------------------


def _case_sp_partial_sum(device: torch.device) -> None:
    """Catches a re-introduction of the ``v * local_n / total_n`` weighting bug:
    each rank's contribution must be its raw partial sum, not a per-rank mean.
    Rank-0 numbers chosen so per-rank-mean averaging would give a wrong answer."""
    rank = dist.get_rank()
    if rank == 0:
        n, ratio_sum, clipfrac_sum, lo, hi = 2, 2.0, 0.0, 0.9, 1.1
    else:
        n, ratio_sum, clipfrac_sum, lo, hi = 6, 7.5, 2.0, 0.5, 1.8

    metrics = _make_metrics(
        device,
        valid_tokens=n,
        ratio_mean=ratio_sum,
        pg_clipfrac=clipfrac_sum,
        ratio_min=lo,
        ratio_max=hi,
    )
    metric_ops = {"ratio_min": "min", "ratio_max": "max"}

    mr._sp_allreduce_kl_metrics(metrics, metric_ops, dist.group.WORLD)

    total_n = 2 + 6
    expected_ratio_mean = (2.0 + 7.5) / total_n
    expected_clipfrac = (0.0 + 2.0) / total_n

    assert metrics["valid_tokens"] == total_n
    got_ratio = metrics["ratio_mean"].item() / metrics["valid_tokens"]
    got_clip = metrics["pg_clipfrac"].item() / metrics["valid_tokens"]
    assert math.isclose(got_ratio, expected_ratio_mean, rel_tol=1e-12), (
        f"[rank {rank}] ratio_mean: got {got_ratio}, expected {expected_ratio_mean}"
    )
    assert math.isclose(got_clip, expected_clipfrac, rel_tol=1e-12), (
        f"[rank {rank}] pg_clipfrac: got {got_clip}, expected {expected_clipfrac}"
    )
    assert metrics["ratio_min"].item() == 0.5
    assert metrics["ratio_max"].item() == 1.8


# ---------------------------------------------------------------------------
# Case: cross-mb + cross-DP (_accumulate_is_metrics + _finalize_is_metrics)
# ---------------------------------------------------------------------------


# Asymmetric valid_tokens per mb verifies (sum, count) bookkeeping under
# uneven contributions. First two mbs go to rank 0, last two to rank 1.
_DP_MBS = [
    {"valid_tokens": 3, "ratio_mean": 3.6, "pg_clipfrac": 1.0, "ratio_min": 0.7, "ratio_max": 1.5},
    {"valid_tokens": 4, "ratio_mean": 4.0, "pg_clipfrac": 0.0, "ratio_min": 0.9, "ratio_max": 1.2},
    {"valid_tokens": 2, "ratio_mean": 2.5, "pg_clipfrac": 2.0, "ratio_min": 0.4, "ratio_max": 1.8},
    {"valid_tokens": 5, "ratio_mean": 4.5, "pg_clipfrac": 1.0, "ratio_min": 1.1, "ratio_max": 1.3},
]


def _case_dp_accumulate_finalize(device: torch.device) -> None:
    rank = dist.get_rank()
    my_mbs = _DP_MBS[:2] if rank == 0 else _DP_MBS[2:]
    metric_ops = {"ratio_min": "min", "ratio_max": "max"}

    accumulated = {}
    for mb in my_mbs:
        mr.ModelRunner._accumulate_is_metrics(accumulated, _make_metrics(device, **mb), metric_ops)

    result = {}
    _finalize_with_world_dp(accumulated, result)

    total_n = sum(mb["valid_tokens"] for mb in _DP_MBS)
    expected = {
        "is_ratio_mean": sum(mb["ratio_mean"] for mb in _DP_MBS) / total_n,
        "is_pg_clipfrac": sum(mb["pg_clipfrac"] for mb in _DP_MBS) / total_n,
        # valid_tokens is itself accumulated as a mean, but its per-mb count
        # is +=1 (not +=n_tokens), so the finalized value is total_n / num_mbs.
        "is_valid_tokens": total_n / len(_DP_MBS),
        "is_ratio_min": min(mb["ratio_min"] for mb in _DP_MBS),
        "is_ratio_max": max(mb["ratio_max"] for mb in _DP_MBS),
    }
    for key, want in expected.items():
        got = result[key]
        assert math.isclose(got, want, rel_tol=1e-12), f"[rank {rank}] {key}: got {got}, expected {want}"


# ---------------------------------------------------------------------------
# Case: empty-rank min/max identity
# ---------------------------------------------------------------------------


def _case_empty_rank_one_empty(device: torch.device) -> None:
    """Rank 0 empty (all IGNORE_INDEX → ±inf identity); rank 1 has real values.
    The empty rank must not leak into min/max, and the global mean must
    reflect rank 1's contribution alone."""
    rank = dist.get_rank()
    if rank == 0:
        new_metrics = _make_metrics(
            device, valid_tokens=0, ratio_mean=0.0, ratio_min=float("inf"), ratio_max=float("-inf")
        )
    else:
        new_metrics = _make_metrics(device, valid_tokens=5, ratio_mean=6.25, ratio_min=0.6, ratio_max=1.7)

    accumulated = {}
    mr.ModelRunner._accumulate_is_metrics(accumulated, new_metrics, {"ratio_min": "min", "ratio_max": "max"})
    result = {}
    _finalize_with_world_dp(accumulated, result)

    assert math.isclose(result["is_ratio_mean"], 1.25, rel_tol=1e-12)
    assert math.isclose(result["is_ratio_min"], 0.6, rel_tol=1e-12)
    assert math.isclose(result["is_ratio_max"], 1.7, rel_tol=1e-12)


def _case_empty_rank_all_empty(device: torch.device) -> None:
    """All ranks empty. Mean keys with global count == 0 are dropped; min/max
    with non-finite reductions fall back to 1.0."""
    new_metrics = _make_metrics(device, valid_tokens=0, ratio_mean=0.0, ratio_min=float("inf"), ratio_max=float("-inf"))

    accumulated = {}
    mr.ModelRunner._accumulate_is_metrics(accumulated, new_metrics, {"ratio_min": "min", "ratio_max": "max"})
    result = {}
    _finalize_with_world_dp(accumulated, result)

    assert result["is_ratio_min"] == 1.0
    assert result["is_ratio_max"] == 1.0
    assert "is_ratio_mean" not in result, f"empty-rank fallback should drop mean keys: {result}"


# ---------------------------------------------------------------------------
# Subprocess dispatch
# ---------------------------------------------------------------------------


_CASES = {
    "sp_partial_sum": [_case_sp_partial_sum],
    "dp_accumulate_finalize": [_case_dp_accumulate_finalize],
    "empty_rank": [_case_empty_rank_one_empty, _case_empty_rank_all_empty],
}


def _main() -> None:
    case_name = os.environ["XORL_TEST_CASE"]
    device = _setup_dist()
    try:
        for fn in _CASES[case_name]:
            fn(device)
    finally:
        dist.destroy_process_group()


def _launch(case: str):
    return run_distributed_script(__file__, num_gpus=2, timeout=120, extra_env={"XORL_TEST_CASE": case})


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_sp_allreduce_kl_metrics_under_cp():
        _launch("sp_partial_sum").assert_success("CP _sp_allreduce_kl_metrics partial-sum reduction")

    @skip_if_gpu_count_less_than(2)
    def test_accumulate_finalize_under_dp():
        _launch("dp_accumulate_finalize").assert_success("DP _accumulate_is_metrics + _finalize_is_metrics")

    @skip_if_gpu_count_less_than(2)
    def test_empty_rank_min_max_identity():
        _launch("empty_rank").assert_success("Empty-rank min/max identity fallback")


if __name__ == "__main__":
    _main()
