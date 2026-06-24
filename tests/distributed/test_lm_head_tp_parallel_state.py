"""Distributed mesh checks for lm-head-only tensor parallelism."""

import os
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_lm_head_tp_parallel_state_case() -> None:
    dist.init_process_group(backend="gloo")
    try:
        init_parallel_state(
            dp_size=1,
            dp_shard_size=1,
            ulysses_size=4,
            lm_head_tp_size=2,
            device_type="cpu",
        )
        ps = get_parallel_state()
        assert ps.cp_size == 4
        assert ps.fsdp_size == 4
        assert dist.get_world_size(ps.sp_group) == 4
        assert dist.get_world_size(ps.ulysses_group) == 4
        assert dist.get_world_size(ps.fsdp_group) == 4
        assert dist.get_world_size(ps.lm_head_tp_group) == 2
        assert dist.get_world_size(ps.lm_head_tp_replica_group) == 2
        assert ps.ulysses_rank == dist.get_rank()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_lm_head_tp_parallel_state_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120)
        result.assert_success()


if __name__ == "__main__":
    _run_lm_head_tp_parallel_state_case()
