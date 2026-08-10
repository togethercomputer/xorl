"""Distributed failure semantics for adapter optimizer restore."""

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from xorl.server.runner.adapters.optimizer_reshard import (
    clone_state_to_cpu,
    commit_optimizer_state,
    coordinate_restore_error,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _RejectTargetOnceOptimizer:
    """Minimal optimizer double that can fail after partially mutating state."""

    def __init__(
        self,
        state: dict[str, Any],
        *,
        reject_target_once: bool,
        reject_rollback: bool = False,
    ) -> None:
        self._state = clone_state_to_cpu(state)
        self._reject_target_once = reject_target_once
        self._reject_rollback = reject_rollback

    def state_dict(self) -> dict[str, Any]:
        return clone_state_to_cpu(self._state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state = clone_state_to_cpu(state)
        value = self._state["state"][0]["exp_avg"]
        if self._reject_target_once and torch.equal(value, torch.tensor([99.0])):
            self._reject_target_once = False
            raise RuntimeError("injected post-mutation rejection")
        if self._reject_rollback and not torch.equal(value, torch.tensor([99.0])):
            raise RuntimeError("injected rollback rejection")


def test_real_gloo_optimizer_restore_rolls_back_every_rank_on_one_rank_rejection():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_ADAPTER_OPTIMIZER_RESTORE_WORKER": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    result.assert_success("real two-rank Gloo adapter optimizer restore transaction")


def _run_optimizer_restore_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2

    resident = {
        "state": {0: {"step": torch.tensor(7.0), "exp_avg": torch.tensor([float(rank + 1)])}},
        "param_groups": [{"params": [0], "lr": 1e-3}],
    }
    target = {
        "state": {0: {"step": torch.tensor(8.0), "exp_avg": torch.tensor([99.0])}},
        "param_groups": [{"params": [0], "lr": 2e-3}],
    }
    optimizer = _RejectTargetOnceOptimizer(resident, reject_target_once=rank == 1)
    adapter_state = SimpleNamespace(optimizer=optimizer)

    # Rank-local validation errors are agreed before any optimizer mutation.
    before_preflight = optimizer.state_dict()
    try:
        coordinate_restore_error(
            RuntimeError("injected preflight rejection") if rank == 1 else None,
            phase="preflight",
        )
    except RuntimeError as exc:
        assert "rank 1: RuntimeError: injected preflight rejection" in str(exc)
    else:
        raise AssertionError("rank-asymmetric preflight rejection was not coordinated")
    assert optimizer.state_dict()["state"][0]["step"].item() == before_preflight["state"][0]["step"].item()

    try:
        commit_optimizer_state(adapter_state, target)
    except RuntimeError as exc:
        assert "rank 1: RuntimeError: injected post-mutation rejection" in str(exc)
    else:
        raise AssertionError("rank-asymmetric optimizer rejection was not coordinated")

    restored = optimizer.state_dict()
    assert torch.equal(restored["state"][0]["exp_avg"], resident["state"][0]["exp_avg"])
    assert restored["state"][0]["step"].item() == 7
    assert restored["param_groups"][0]["lr"] == 1e-3

    restored_values: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(restored_values, float(restored["state"][0]["exp_avg"].item()))
    assert restored_values == [1.0, 2.0]

    # A rollback rejection is not recoverable in-process: all ranks must fail
    # the restore rather than claim that the optimizer transaction completed.
    dist.barrier()
    fatal_optimizer = _RejectTargetOnceOptimizer(
        resident,
        reject_target_once=rank == 1,
        reject_rollback=rank == 1,
    )
    fatal_state = SimpleNamespace(optimizer=fatal_optimizer)
    try:
        commit_optimizer_state(fatal_state, target)
    except RuntimeError as exc:
        assert "rollback was not globally successful" in str(exc)
    else:
        raise AssertionError("rollback rejection was not treated as fatal")
    dist.destroy_process_group()


if os.environ.get("XORL_ADAPTER_OPTIMIZER_RESTORE_WORKER") == "1":
    _run_optimizer_restore_worker()
