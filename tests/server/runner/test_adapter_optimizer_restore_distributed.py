"""Distributed failure semantics for adapter optimizer restore."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.lora import LoraLinear
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.adapters.optimizer_reshard import (
    clone_state_to_cpu,
    commit_optimizer_state,
    coordinate_restore_error,
    same_optimizer_value,
)
from xorl.server.runner.checkpoint.manager import CheckpointManager
from xorl.server.runner.model_runner import ModelRunner


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


def test_real_gloo_physical_pp_adapter_checkpoint_restores_each_stage(tmp_path: Path):
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=180,
        extra_env={
            "XORL_PHYSICAL_PP_ADAPTER_RESUME_WORKER": "1",
            "XORL_PHYSICAL_PP_ADAPTER_RESUME_DIR": str(tmp_path),
            "XORL_SERVER_ARTIFACT_ROOT": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    result.assert_success("real two-rank Gloo physical-PP adapter checkpoint save and restore")


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


class _PhysicalPPAdapterStage(nn.Module):
    def __init__(self, stage_ordinal: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([None] * (stage_ordinal + 1))
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = LoraLinear(4, 4, r=2, lora_alpha=4)
        self.model.layers[stage_ordinal] = layer


def _physical_pp_session_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_model": "synthetic/physical-pp",
        "is_lora": True,
        "lora_config": {
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_init_seed": 17,
        },
        "optimizer_config": {
            "type": "adamw",
            "learning_rate": 1e-2,
            "weight_decay": 0.01,
            "optimizer_dtype": "fp32",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "optimizer_kwargs": {},
        },
    }


def _build_physical_pp_manager(root: Path, rank: int) -> tuple[_PhysicalPPAdapterStage, LoRAAdapterManager]:
    model = _PhysicalPPAdapterStage(rank)
    manager = LoRAAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(root / f"target-rank{rank}" / "adapters"),
        auto_save_on_eviction=False,
        lora_config={"lora_rank": 2, "lora_alpha": 4, "pipeline_parallel_size": 2},
        optimizer_type="adamw",
        optimizer_dtype="fp32",
        optimizer_fused=False,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    manager.register_adapter(
        "physical-pp",
        session_spec=_physical_pp_session_spec(),
        initialize_fresh=True,
    )
    runner = object.__new__(ModelRunner)
    runner.model = model
    runner.model_parts = [model]
    runner.rank = rank
    runner.train_config = {"tensor_parallel_size": 1}
    runner._adapter_manager = manager
    runner._compile_registered_adapter_gradient_ownership("physical-pp")
    return model, manager


def _physical_pp_checkpoint_manager(
    model: nn.Module,
    manager: LoRAAdapterManager,
    rank: int,
) -> CheckpointManager:
    checkpoint_manager = CheckpointManager(
        model=model,
        optimizer=None,
        checkpointer=None,
        lora_config={"lora_rank": 2, "lora_alpha": 4, "pipeline_parallel_size": 2},
        model_config={"model_path": "synthetic/physical-pp"},
        train_config={"tensor_parallel_size": 1},
        rank=rank,
        local_rank=rank,
        adapter_manager=manager,
    )
    checkpoint_manager.lora_target_modules = ["o_proj"]
    checkpoint_manager.lora_alpha_value = 4
    return checkpoint_manager


def _run_physical_pp_adapter_resume_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    root = Path(os.environ["XORL_PHYSICAL_PP_ADAPTER_RESUME_DIR"])
    checkpoint_path = root / "physical-pp-checkpoint"

    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=2,
        ringattn_size=1,
        ulysses_size=1,
        dp_mode="none",
        device_type="cpu",
        cp_fsdp_mode="none",
    )
    assert get_parallel_state().pp_rank == rank
    model, source = _build_physical_pp_manager(root / "source", rank)
    source_state = source.get_adapter_state("physical-pp")
    assert source_state.pipeline_stage_ordinal == rank
    assert source_state.layout_world_size == 1
    assert source_state.layout_group_ranks == (rank,)

    for ordinal, parameter in enumerate(source_state.local_params.values(), start=1):
        parameter.grad = torch.full_like(parameter, float((rank + 1) * 10 + ordinal))
    source_state.optimizer.step()
    source_state.optimizer.zero_grad(set_to_none=True)
    source_state.global_step = 1
    source_weights = {name: value.detach().clone() for name, value in source_state.local_params.items()}
    source_optimizer = clone_state_to_cpu(source_state.optimizer.state_dict())
    assert source_optimizer["state"]

    checkpoint = _physical_pp_checkpoint_manager(model, source, rank)
    checkpoint.save_adapter_state("physical-pp", str(checkpoint_path), save_optimizer=True)
    if rank == 0:
        manifest = json.loads((checkpoint_path / "optimizer_shards.json").read_text(encoding="utf-8"))
        assert manifest["world_size"] == 2
        assert manifest["per_rank_layout_world_size"] == [1, 1]
        assert manifest["per_rank_layout_group_ranks"] == [[0], [1]]
        stage_records = manifest["optimizer_restore_contracts_by_stage"]
        assert [record["pipeline_stage_ordinal"] for record in stage_records] == [0, 1]
        assert all(record["optimizer_restore_contract"] is not None for record in stage_records)
        assert stage_records[0]["parameter_fqns"] != stage_records[1]["parameter_fqns"]
        metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["gradient_ownership"]["optimizer_restore_contract"] is None
        assert metadata["gradient_ownership"]["optimizer_restore_contracts_by_stage"] == stage_records
    dist.barrier()

    _target_model, target = _build_physical_pp_manager(root / "target", rank)
    target.load_adapter_state("physical-pp", str(checkpoint_path), load_optimizer=True)
    target_state = target.get_adapter_state("physical-pp")
    assert set(target_state.local_params) == set(source_weights)
    for name, expected in source_weights.items():
        assert torch.equal(target_state.local_params[name], expected), name
    assert same_optimizer_value(target_state.optimizer.state_dict(), source_optimizer)

    before_rejected_load_weights = {name: value.detach().clone() for name, value in target_state.local_params.items()}
    before_rejected_load_optimizer = clone_state_to_cpu(target_state.optimizer.state_dict())
    if rank == 0:
        metadata_path = checkpoint_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stage_one = next(
            record
            for record in metadata["gradient_ownership"]["optimizer_restore_contracts_by_stage"]
            if record["pipeline_stage_ordinal"] == 1
        )
        stage_one["pipeline_stage_ordinal"] = 7
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dist.barrier()

    try:
        target.load_adapter_state("physical-pp", str(checkpoint_path), load_optimizer=True)
    except RuntimeError as exc:
        message = str(exc)
        assert "rank 1" in message
        assert "no optimizer restore contract for this physical PP stage or layout topology" in message
    else:
        raise AssertionError("Wrong-stage optimizer restore metadata was accepted")
    for name, expected in before_rejected_load_weights.items():
        assert torch.equal(target_state.local_params[name], expected), name
    assert same_optimizer_value(target_state.optimizer.state_dict(), before_rejected_load_optimizer)
    dist.destroy_process_group()


if os.environ.get("XORL_ADAPTER_OPTIMIZER_RESTORE_WORKER") == "1":
    _run_optimizer_restore_worker()

if os.environ.get("XORL_PHYSICAL_PP_ADAPTER_RESUME_WORKER") == "1":
    _run_physical_pp_adapter_resume_worker()
