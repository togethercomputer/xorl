"""Real-NCCL coverage for per-micro-batch CP/Ulysses loss-metric reduction."""

from __future__ import annotations

import math
import os

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

    mr._sp_allreduce_kl_metrics(metrics, dist.group.WORLD, metric_ops)

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


_CASES = {"sp_partial_sum": [_case_sp_partial_sum]}


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


if __name__ == "__main__":
    _main()
