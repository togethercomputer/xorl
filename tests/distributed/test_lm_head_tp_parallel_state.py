"""Distributed mesh checks for lm-head-only tensor parallelism."""

import os
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.distributed.parallel_state import (  # noqa: E402
    _build_stage_local_lm_head_mesh,
    get_parallel_state,
    init_parallel_state,
)


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_lm_head_tp_parallel_state_case() -> None:
    dist.init_process_group(backend="gloo")
    try:
        cfg = os.environ.get("XORL_LMHEAD_PS_CFG", "cp")
        if cfg == "cp":
            init_parallel_state(
                dp_size=1,
                dp_shard_size=1,
                ulysses_size=4,
                lm_head_tp_size=2,
                device_type="cpu",
            )
            expected_tp = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
            expected_replica = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
        elif cfg == "dp":
            init_parallel_state(
                dp_size=4,
                dp_shard_size=4,
                lm_head_tp_size=2,
                device_type="cpu",
            )
            expected_tp = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
            expected_replica = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
        elif cfg == "hsdp":
            init_parallel_state(
                dp_size=4,
                dp_replicate_size=2,
                dp_shard_size=2,
                lm_head_tp_size=2,
                device_type="cpu",
            )
            expected_tp = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
            expected_replica = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
        elif cfg == "pp":
            init_parallel_state(
                pp_size=2,
                dp_size=2,
                dp_shard_size=2,
                lm_head_tp_size=2,
                device_type="cpu",
            )
            expected_tp = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
            expected_replica = {rank: [rank] for rank in range(4)}
        else:
            raise ValueError(f"Unknown XORL_LMHEAD_PS_CFG={cfg!r}")
        ps = get_parallel_state()
        rank = dist.get_rank()
        if cfg == "cp":
            assert ps.cp_size == 4
            assert dist.get_world_size(ps.sp_group) == 4
            assert dist.get_world_size(ps.ulysses_group) == 4
            assert ps.ulysses_rank == rank
        else:
            assert ps.cp_size == 1
            assert ps.sp_group is None
            assert ps.ulysses_group is None
        assert dist.get_world_size(ps.fsdp_group) == (2 if cfg == "pp" else 4)
        assert dist.get_world_size(ps.lm_head_tp_group) == 2
        assert dist.get_world_size(ps.lm_head_tp_replica_group) == (1 if cfg == "pp" else 2)
        assert dist.get_process_group_ranks(ps.lm_head_tp_group) == expected_tp[rank]
        assert dist.get_process_group_ranks(ps.lm_head_tp_replica_group) == expected_replica[rank]
        if cfg == "hsdp":
            assert tuple(ps.fsdp_mesh.mesh.shape) == (2, 2)
            assert tuple(ps.fsdp_mesh.mesh_dim_names) == ("dp_replicate", "dp_shard")
            assert tuple(ps.dp_replicate_mesh.mesh.shape) == (2,)
            assert tuple(ps.dp_shard_mesh.mesh.shape) == (2,)
            assert ps.lm_head_mesh.mesh.tolist() == [[0, 1], [2, 3]]
        if cfg == "pp":
            expected_stage = [[0, 1]] if rank < 2 else [[2, 3]]
            assert ps.lm_head_mesh.mesh.tolist() == expected_stage
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_lm_head_tp_parallel_state_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120, extra_env={"XORL_LMHEAD_PS_CFG": "cp"})
        result.assert_success()

    def test_lm_head_tp_parallel_state_nocp_dp_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120, extra_env={"XORL_LMHEAD_PS_CFG": "dp"})
        result.assert_success()

    def test_lm_head_tp_parallel_state_nocp_hsdp_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120, extra_env={"XORL_LMHEAD_PS_CFG": "hsdp"})
        result.assert_success()

    def test_lm_head_tp_parallel_state_isolated_per_pp_stage_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120, extra_env={"XORL_LMHEAD_PS_CFG": "pp"})
        result.assert_success()


@pytest.mark.parametrize("dp_size,cp_size", [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)])
def test_glm52_lm_head_tp_spans_composed_dp_cp_plane(dp_size: int, cp_size: int):
    mesh = _build_stage_local_lm_head_mesh(
        world_size=16,
        pp_size=1,
        dp_size=dp_size,
        cp_size=cp_size,
        tp_size=1,
        lm_head_tp_size=16,
    )
    assert mesh.tolist() == [[list(range(16))]]


def test_glm52_lm_head_tp_groups_never_cross_pp_stages():
    mesh = _build_stage_local_lm_head_mesh(
        world_size=32,
        pp_size=2,
        dp_size=4,
        cp_size=4,
        tp_size=1,
        lm_head_tp_size=16,
    )
    assert mesh.tolist() == [[list(range(16))], [list(range(16, 32))]]


if __name__ == "__main__":
    _run_lm_head_tp_parallel_state_case()
