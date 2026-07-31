"""Mesh checks for lm-head-only TP composed with expert parallelism (ep>1).

EP is a separate re-grouping of the same ranks for the experts only; it is not an
axis of the main device_mesh. So the lm_head_mesh (replica x lm_head_tp, carved from
the CP-innermost main mesh) must be IDENTICAL whether or not ep>1. This test pins
that: ep=2 gives the same lm_head_tp / replica group membership as the ep-independent
construction, and EP is genuinely active alongside it.
"""

import os
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_case() -> None:
    dist.init_process_group(backend="gloo")
    try:
        # 4 ranks: dp_shard=2 x ulysses(CP)=2, ep=2 overlaid, lm_head_tp=2.
        # Main mesh [dp_shard=2, ulysses=2] (row-major) -> rank r: dp=r//2, cp=r%2.
        # cp_replica = cp_size/lm_head_tp = 1, so lm_head_tp groups are the CP groups
        # {0,1},{2,3} and replica groups span DP: {0,2},{1,3}.
        init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            ulysses_size=2,
            ep_size=2,
            lm_head_tp_size=2,
            device_type="cpu",
        )
        ps = get_parallel_state()
        rank = dist.get_rank()

        # EP is genuinely active.
        assert ps.ep_size == 2, ps.ep_size
        assert ps.ep_enabled
        assert ps.ep_fsdp_device_mesh is not None

        # Sizes.
        assert dist.get_world_size(ps.lm_head_tp_group) == 2
        assert dist.get_world_size(ps.lm_head_tp_replica_group) == 2

        # Exact membership (ep-independent expectation).
        expected_tp = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}[rank]
        expected_replica = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}[rank]
        assert dist.get_process_group_ranks(ps.lm_head_tp_group) == expected_tp, (
            rank,
            dist.get_process_group_ranks(ps.lm_head_tp_group),
        )
        assert dist.get_process_group_ranks(ps.lm_head_tp_replica_group) == expected_replica, (
            rank,
            dist.get_process_group_ranks(ps.lm_head_tp_replica_group),
        )
        print(f"rank{rank} OK tp={expected_tp} replica={expected_replica} ep_size={ps.ep_size}")
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_lm_head_tp_ep_parallel_state_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120)
        result.assert_success()


if __name__ == "__main__":
    _run_case()
