"""Tests for multi-rank adapter load coordination."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

import xorl.server.runner.adapters.adapter_coordinator as adapter_coordinator_module
from xorl.lora.utils import convert_peft_lora_state_dict
from xorl.server.protocol.operations import AdapterStateData, RegisterAdapterData, RegisterSessionData
from xorl.server.runner.adapters.adapter_coordinator import AdapterCoordinator
from xorl.server.runner.adapters.sharded_state import AdapterTensorLayout, pack_logical_tensor


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeAdapterState:
    def __init__(self, lr: float = 1e-5):
        self.global_step = 0
        self.global_forward_backward_step = 0
        self.lr = lr
        self.local_params = {}
        self.tensor_layouts = {}
        self.last_access_time = time.time()


class _FakeAdapterManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = str(checkpoint_dir)
        self.adapters = {}
        self.current_adapter_id = None
        self.max_adapters = 8
        self.model = None

    @staticmethod
    def _canonical_lora_param_name(name: str) -> str:
        if name.endswith(".weight"):
            return name[: -len(".weight")]
        return name

    def get_adapter_state(self, model_id: str):
        return self.adapters[model_id]

    def has_adapter(self, model_id: str) -> bool:
        return model_id in self.adapters

    def remove_adapter(self, model_id: str) -> None:
        self.adapters.pop(model_id, None)

    def list_adapters(self):
        return list(self.adapters.keys())

    def set_lr(self, model_id: str, lr: float) -> None:
        self.adapters[model_id].lr = lr

    def _pack_logical_state_dict(self, state, state_dict):
        expected_shapes = {name: layout.logical_shape for name, layout in state.tensor_layouts.items()}
        converted = convert_peft_lora_state_dict(state_dict, expected_shapes=expected_shapes)
        return {name: pack_logical_tensor(state.tensor_layouts[name], converted[name]) for name in state.local_params}


class _FakeTrainer:
    def __init__(self, checkpoint_dir: Path, *, adapter_state_load_mode: str = "all_ranks"):
        self.adapter_manager = _FakeAdapterManager(checkpoint_dir)
        self.register_calls = []
        self.load_calls = []
        self.save_calls = []
        self.lora_config = {"adapter_state_load_mode": adapter_state_load_mode}
        self.train_config = {"pipeline_parallel_size": 1}
        self.lora_session_specs = {}
        self.fail_load_with = None

    @staticmethod
    def _session_spec(lr: float = 1e-5):
        return {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
            "lora_config": {"lora_rank": 4, "lora_alpha": 8},
            "optimizer_config": {
                "type": "adamw",
                "learning_rate": lr,
                "weight_decay": 0.01,
                "optimizer_dtype": "bf16",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "optimizer_kwargs": {},
            },
        }

    def get_lora_session_spec(self, model_id: str):
        return self.lora_session_specs[model_id]

    def register_session(
        self, model_id: str, session_spec: dict, materialize: bool = False, initialize_fresh: bool = True
    ):
        self.lora_session_specs[model_id] = session_spec
        if materialize:
            self.register_lora_adapter(model_id, session_spec["optimizer_config"]["learning_rate"])
        return {"registered": True, "model_id": model_id, "materialized": materialize}

    def register_lora_adapter(self, model_id: str, lr: float):
        self.lora_session_specs.setdefault(model_id, self._session_spec(lr))
        self.register_calls.append((model_id, lr))
        self.adapter_manager.adapters[model_id] = _FakeAdapterState(lr=lr)

    def register_adapter(self, model_id: str, lr: float):
        self.register_lora_adapter(model_id, lr)
        return {"registered": True, "model_id": model_id, "lr": lr}

    def load_adapter_state(self, model_id: str, path: str, load_optimizer: bool = True, lr: float | None = None):
        if self.fail_load_with is not None:
            raise self.fail_load_with
        self.load_calls.append(
            {
                "model_id": model_id,
                "path": path,
                "load_optimizer": load_optimizer,
                "lr": lr,
            }
        )
        state = self.adapter_manager.adapters.setdefault(model_id, _FakeAdapterState())
        if lr is not None:
            state.lr = lr
        state.global_step = 7
        return {"success": True, "model_id": model_id, "step": state.global_step}

    def save_adapter_state(self, model_id: str, path: str, save_optimizer: bool = True):
        self.save_calls.append(
            {
                "model_id": model_id,
                "path": path,
                "save_optimizer": save_optimizer,
            }
        )
        return {"success": True, "model_id": model_id, "path": path}


def test_adapter_load_lifecycle_and_transport_policy(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "adapters"
    evicted_path = checkpoint_dir / "evicted" / "policy-a"
    evicted_path.mkdir(parents=True)

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-a", trainer._session_spec(1e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    was_auto_loaded, checkpoint_path = coordinator.auto_load_if_evicted("policy-a")

    assert was_auto_loaded is True
    assert checkpoint_path == str(evicted_path)
    assert trainer.register_calls == [("policy-a", 1e-5)]
    assert trainer.load_calls == [
        {
            "model_id": "policy-a",
            "path": str(evicted_path),
            "load_optimizer": True,
            "lr": None,
        }
    ]
    coordinator.broadcast_adapter_state.assert_not_called()

    for name, assertion in (
        ("fresh", _assert_auto_load_broadcasts_fresh_adapter),
        ("materialize-failure", _assert_auto_load_syncs_materialization_failure),
        ("fresh-disabled", _assert_auto_load_rejects_missing_checkpoint_when_fresh_disabled),
        ("restore-failure", _assert_auto_load_rolls_back_when_restore_fails),
    ):
        case_root = tmp_path / name
        case_root.mkdir()
        assertion(case_root)

    explicit_root = tmp_path / "explicit-load"
    explicit_root.mkdir()
    _assert_handle_load_adapter_state_and_path_admission_policy(explicit_root)
    with monkeypatch.context() as case_patch:
        _assert_rank0_broadcast_load_policy(case_patch, tmp_path / "rank0-broadcast")
    registration_root = tmp_path / "registration"
    registration_root.mkdir()
    _assert_adapter_registration_policy(registration_root)
    save_root = tmp_path / "save"
    save_root.mkdir()
    _assert_handle_save_adapter_state_requires_evicted_checkpoint(save_root)


def _assert_auto_load_broadcasts_fresh_adapter(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-new", trainer._session_spec(1e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    was_auto_loaded, checkpoint_path = coordinator.auto_load_if_evicted("policy-new")

    assert was_auto_loaded is True
    assert checkpoint_path is None
    assert trainer.register_calls == [("policy-new", 1e-5)]
    assert trainer.load_calls == []
    coordinator.broadcast_adapter_state.assert_called_once_with("policy-new", 1e-5)


def _assert_auto_load_syncs_materialization_failure(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-fail", trainer._session_spec(1e-5), materialize=False)

    def _fail_register(model_id, lr):
        raise RuntimeError("capacity full")

    trainer.register_lora_adapter = _fail_register
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()
    coordinator._sync_collective_error = Mock(return_value="rank 1: capacity full")

    with pytest.raises(RuntimeError, match="rank 1: capacity full"):
        coordinator.auto_load_if_evicted("policy-fail")

    coordinator._sync_collective_error.assert_called_once()
    coordinator.broadcast_adapter_state.assert_not_called()


def _assert_auto_load_rejects_missing_checkpoint_when_fresh_disabled(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-missing", trainer._session_spec(2e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=1, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    with pytest.raises(FileNotFoundError, match="Refusing to recreate fresh state"):
        coordinator.auto_load_if_evicted("policy-missing", allow_fresh_materialization=False)

    assert trainer.register_calls == []
    coordinator.broadcast_adapter_state.assert_not_called()


def _assert_auto_load_rolls_back_when_restore_fails(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    evicted_path = checkpoint_dir / "evicted" / "policy-bad"
    evicted_path.mkdir(parents=True)

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.fail_load_with = RuntimeError("corrupt checkpoint")
    trainer.register_session("policy-bad", trainer._session_spec(2e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=1, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    with pytest.raises(RuntimeError, match="Failed to auto-load adapter 'policy-bad'"):
        coordinator.auto_load_if_evicted("policy-bad")

    assert trainer.register_calls == [("policy-bad", 2e-5)]
    assert trainer.load_calls == []
    assert not trainer.adapter_manager.has_adapter("policy-bad")
    coordinator.broadcast_adapter_state.assert_not_called()


def _assert_handle_load_adapter_state_and_path_admission_policy(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-b", trainer._session_spec(3e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    payload = AdapterStateData(
        model_id="policy-b",
        path=str(adapter_path),
        load_optimizer=True,
        lr=3e-5,
    )

    result = asyncio.run(coordinator.handle_load_adapter_state({"payload": payload}))

    assert result == {"success": True, "model_id": "policy-b"}
    assert trainer.register_calls == [("policy-b", 3e-5)]
    assert trainer.load_calls == [
        {
            "model_id": "policy-b",
            "path": str(adapter_path),
            "load_optimizer": True,
            "lr": 3e-5,
        }
    ]
    coordinator.broadcast_adapter_state.assert_not_called()

    path_root = tmp_path / "path-admission"
    path_root.mkdir()
    _assert_adapter_state_paths_stay_within_output_root(path_root)

    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    _assert_handle_load_adapter_state_failure_policy(failure_root)


def _assert_adapter_state_paths_stay_within_output_root(tmp_path):
    checkpoint_dir = tmp_path / "output" / "adapters"
    checkpoint_dir.mkdir(parents=True)
    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-b", trainer._session_spec(3e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=1, cpu_group=None)

    payload = AdapterStateData(
        model_id="policy-b",
        path=str(tmp_path / "outside" / "checkpoint"),
        load_optimizer=True,
        lr=3e-5,
    )

    result = asyncio.run(coordinator.handle_load_adapter_state({"payload": payload}))

    assert result["success"] is False
    assert "must be inside the server output directory" in result["error"]
    assert trainer.register_calls == []
    assert trainer.load_calls == []

    with pytest.raises(ValueError, match="must be inside the server output directory"):
        coordinator._find_evicted_checkpoint("../../../outside")

    trainer.register_lora_adapter("policy-save", 4e-5)
    with pytest.raises(RuntimeError, match="must be inside the server output directory"):
        asyncio.run(
            coordinator.handle_save_adapter_state(
                {
                    "payload": AdapterStateData(
                        model_id="policy-save",
                        path=str(tmp_path / "outside" / "save-target"),
                        save_optimizer=True,
                    )
                }
            )
        )
    assert trainer.save_calls == []


def _assert_rank0_broadcast_load_policy(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    evicted_path = checkpoint_dir / "evicted" / "policy-c"
    evicted_path.mkdir(parents=True)

    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.register_session("policy-c", trainer._session_spec(1e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator._restore_rank0_broadcast_adapter_state = Mock(
        return_value={"success": True, "model_id": "policy-c", "step": 0}
    )
    coordinator.broadcast_adapter_state = Mock()

    was_auto_loaded, checkpoint_path = coordinator.auto_load_if_evicted("policy-c")

    assert was_auto_loaded is True
    assert checkpoint_path == str(evicted_path)
    assert trainer.register_calls == [("policy-c", 1e-5)]
    assert trainer.load_calls == []
    coordinator._restore_rank0_broadcast_adapter_state.assert_called_once()
    coordinator.broadcast_adapter_state.assert_not_called()

    handle_root = tmp_path / "handle-load"
    handle_root.mkdir()
    _assert_handle_load_uses_rank0_broadcast_mode(handle_root)

    sharded_root = tmp_path / "sharded-restore"
    sharded_root.mkdir()
    _assert_rank0_broadcast_sharded_restore_policy(monkeypatch, sharded_root)


def _assert_handle_load_uses_rank0_broadcast_mode(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()

    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.register_session("policy-d", trainer._session_spec(2e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator._restore_rank0_broadcast_adapter_state = Mock(
        return_value={"success": True, "model_id": "policy-d", "step": 0}
    )
    coordinator.broadcast_adapter_state = Mock()

    payload = AdapterStateData(
        model_id="policy-d",
        path=str(adapter_path),
        load_optimizer=True,
        lr=2e-5,
    )

    result = asyncio.run(coordinator.handle_load_adapter_state({"payload": payload}))

    assert result == {"success": True, "model_id": "policy-d"}
    assert trainer.register_calls == [("policy-d", 2e-5)]
    assert trainer.load_calls == []
    coordinator._restore_rank0_broadcast_adapter_state.assert_called_once()
    coordinator.broadcast_adapter_state.assert_not_called()


def _assert_rank0_broadcast_sharded_restore_policy(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()

    param_name = "model.layers.0.mlp.experts.down_proj_lora_A"
    full_tensor = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 5, 2)
    save_safetensors_file(
        {
            f"base_model.model.model.layers.0.mlp.experts.{expert_idx}.down_proj.lora_A.weight": full_tensor[expert_idx]
            .transpose(0, 1)
            .contiguous()
            for expert_idx in range(4)
        },
        str(adapter_path / "adapter_model.safetensors"),
    )
    (adapter_path / "metadata.json").write_text(
        json.dumps({"global_step": 11, "global_forward_backward_step": 13, "lr": 3e-5}),
        encoding="utf-8",
    )

    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.register_session("policy-ep", trainer._session_spec(3e-5), materialize=False)
    trainer.register_lora_adapter("policy-ep", 3e-5)
    trainer.adapter_manager.model = object()
    state = trainer.adapter_manager.get_adapter_state("policy-ep")
    state.local_params = {param_name: torch.nn.Parameter(torch.empty(2, 5, 2))}
    state.tensor_layouts = {
        param_name: AdapterTensorLayout(
            fqn=param_name,
            dtype=torch.float32,
            rank_dim=0,
            substrate_shape=(4, 5, 2),
            logical_shape=(4, 5, 2),
            local_substrate_shape=(2, 5, 2),
            local_logical_offset=(2, 0, 0),
            local_logical_shape=(2, 5, 2),
            active_local_slices=(slice(0, 2), slice(0, 5), slice(0, 2)),
            active_storage_shape=(2, 5, 2),
            replica_key=(param_name, (4, 5, 2), (2, 0, 0), (2, 5, 2), "float32"),
        )
    }
    monkeypatch.setattr(
        adapter_coordinator_module.dist,
        "broadcast_object_list",
        lambda payload, src=0, group=None: None,
    )

    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    result = coordinator._restore_adapter_state(
        model_id="policy-ep",
        path=str(adapter_path),
        load_optimizer=False,
        lr=None,
        default_lr=3e-5,
    )

    assert result["success"] is True
    assert result["step"] == 11
    assert torch.equal(state.local_params[param_name].detach(), full_tensor[2:4])
    assert state.global_forward_backward_step == 13
    assert state.lr == 3e-5
    assert trainer.load_calls == []
    coordinator.broadcast_adapter_state.assert_not_called()

    mismatch_root = tmp_path / "session-mismatch"
    mismatch_root.mkdir()
    _assert_rank0_broadcast_restore_rejects_session_spec_mismatch(monkeypatch, mismatch_root)
    optimizer_root = tmp_path / "optimizer-rejection"
    optimizer_root.mkdir()
    _assert_rank0_broadcast_optimizer_rejection_is_transactional(monkeypatch, optimizer_root)


def _assert_rank0_broadcast_restore_rejects_session_spec_mismatch(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()

    param_name = "model.layers.0.mlp.experts.down_proj_lora_A"
    full_tensor = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 5, 2)
    save_safetensors_file(
        {
            f"base_model.model.model.layers.0.mlp.experts.{expert_idx}.down_proj.lora_A.weight": full_tensor[expert_idx]
            .transpose(0, 1)
            .contiguous()
            for expert_idx in range(4)
        },
        str(adapter_path / "adapter_model.safetensors"),
    )
    (adapter_path / "metadata.json").write_text(
        json.dumps({"global_step": 11, "global_forward_backward_step": 13, "lr": 3e-5}),
        encoding="utf-8",
    )
    checkpoint_session_spec = _FakeTrainer._session_spec(3e-5)
    checkpoint_session_spec["lora_config"]["lora_alpha"] = 16
    (adapter_path / "session_spec.json").write_text(json.dumps(checkpoint_session_spec), encoding="utf-8")

    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.register_session("policy-ep", trainer._session_spec(3e-5), materialize=False)
    trainer.register_lora_adapter("policy-ep", 3e-5)
    trainer.adapter_manager.model = object()
    state = trainer.adapter_manager.get_adapter_state("policy-ep")
    original_tensor = torch.zeros(2, 5, 2)
    state.local_params = {param_name: torch.nn.Parameter(original_tensor.clone())}
    state.tensor_layouts = {
        param_name: AdapterTensorLayout(
            fqn=param_name,
            dtype=torch.float32,
            rank_dim=0,
            substrate_shape=(4, 5, 2),
            logical_shape=(4, 5, 2),
            local_substrate_shape=(2, 5, 2),
            local_logical_offset=(2, 0, 0),
            local_logical_shape=(2, 5, 2),
            active_local_slices=(slice(0, 2), slice(0, 5), slice(0, 2)),
            active_storage_shape=(2, 5, 2),
            replica_key=(param_name, (4, 5, 2), (2, 0, 0), (2, 5, 2), "float32"),
        )
    }
    monkeypatch.setattr(
        adapter_coordinator_module.dist,
        "broadcast_object_list",
        lambda payload, src=0, group=None: None,
    )

    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=2, cpu_group=None)

    with pytest.raises(ValueError, match="Checkpoint session spec does not match"):
        coordinator._restore_rank0_broadcast_adapter_state(
            model_id="policy-ep",
            path=str(adapter_path),
            load_optimizer=True,
            lr=None,
        )

    assert torch.equal(state.local_params[param_name].detach(), original_tensor)


def _assert_rank0_broadcast_optimizer_rejection_is_transactional(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.register_session("policy", trainer._session_spec(3e-5), materialize=False)
    trainer.register_lora_adapter("policy", 3e-5)
    state = trainer.adapter_manager.get_adapter_state("policy")
    state.global_step = 7
    state.global_forward_backward_step = 9
    original_lr = state.lr
    original_last_access = state.last_access_time
    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=2, cpu_group=None)
    coordinator._rank0_load_adapter_checkpoint_payload = Mock(
        return_value={"optimizer_present": True, "weights": {}, "metadata": {"global_step": 99}}
    )

    with pytest.raises(RuntimeError, match="cannot restore topology-specific adapter optimizer shards"):
        coordinator._restore_rank0_broadcast_adapter_state(
            model_id="policy",
            path=str(tmp_path / "checkpoint"),
            load_optimizer=True,
            lr=None,
        )

    assert state.global_step == 7
    assert state.global_forward_backward_step == 9
    assert state.lr == original_lr
    assert state.last_access_time == original_last_access


def _assert_handle_load_adapter_state_failure_policy(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()

    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.register_session("policy-sync-fail", trainer._session_spec(2e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()
    coordinator._sync_collective_error = Mock(side_effect=[None, None, "rank 0: Adapter state restore failed"])

    payload = AdapterStateData(
        model_id="policy-sync-fail",
        path=str(adapter_path),
        load_optimizer=True,
        lr=2e-5,
    )

    result = asyncio.run(coordinator.handle_load_adapter_state({"payload": payload}))

    assert result == {
        "success": False,
        "error": "Adapter state load failed: rank 0: Adapter state restore failed",
    }
    assert trainer.register_calls == [("policy-sync-fail", 2e-5)]
    assert not trainer.adapter_manager.has_adapter("policy-sync-fail")
    coordinator.broadcast_adapter_state.assert_not_called()

    pp_root = tmp_path / "pipeline-rejection"
    pp_root.mkdir()
    _assert_handle_load_rejects_pipeline_multi_adapter_lora(pp_root)
    session_root = tmp_path / "session-rollback"
    session_root.mkdir()
    _assert_handle_load_rolls_back_auto_registered_session(session_root)


def _assert_handle_load_rejects_pipeline_multi_adapter_lora(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()

    trainer = _FakeTrainer(checkpoint_dir, adapter_state_load_mode="rank0_broadcast")
    trainer.train_config["pipeline_parallel_size"] = 2
    trainer.register_session("policy-pp", trainer._session_spec(2e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)

    payload = AdapterStateData(
        model_id="policy-pp",
        path=str(adapter_path),
        load_optimizer=True,
        lr=2e-5,
    )

    result = asyncio.run(coordinator.handle_load_adapter_state({"payload": payload}))

    assert result == {
        "success": False,
        "error": (
            "Adapter state load failed: pipeline_parallel_size > 1 is not supported with multi-adapter LoRA "
            "server training. Adapter coordination currently assumes identical local LoRA layouts on every rank."
        ),
    }
    assert "policy-pp" in trainer.lora_session_specs
    assert trainer.load_calls == []


def _assert_handle_load_rolls_back_auto_registered_session(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()
    adapter_path = tmp_path / "checkpoint"
    adapter_path.mkdir()
    (adapter_path / "session_spec.json").write_text(json.dumps(_FakeTrainer._session_spec(4e-5)), encoding="utf-8")

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.fail_load_with = RuntimeError("corrupt checkpoint")
    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=1, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    payload = AdapterStateData(
        model_id="policy-checkpoint-only",
        path=str(adapter_path),
        load_optimizer=True,
    )

    result = asyncio.run(coordinator.handle_load_adapter_state({"payload": payload}))

    assert result["success"] is False
    assert result["error"] == (
        "Adapter state load failed: Adapter state restore failed for model_id=policy-checkpoint-only: corrupt checkpoint"
    )
    assert "policy-checkpoint-only" not in trainer.lora_session_specs
    assert not trainer.adapter_manager.has_adapter("policy-checkpoint-only")
    coordinator.broadcast_adapter_state.assert_not_called()


def _assert_adapter_registration_policy(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()

    result = asyncio.run(coordinator.handle_register_adapter({"payload": RegisterAdapterData("policy-e", 4e-5)}))
    assert result == {}

    session_spec = trainer._session_spec(5e-5)
    result = asyncio.run(
        coordinator.handle_register_session(
            {"payload": RegisterSessionData(model_id="policy-f", session_spec=session_spec, materialize=True)}
        )
    )

    assert result == {}
    assert trainer.lora_session_specs["policy-f"] == session_spec
    assert trainer.register_calls == [("policy-e", 4e-5), ("policy-f", 5e-5)]
    assert coordinator.broadcast_adapter_state.call_args_list == [
        call("policy-e", 4e-5),
        call("policy-f", 5e-5),
    ]

    rollback_root = tmp_path / "rollback"
    rollback_root.mkdir()
    _assert_register_session_rolls_back_on_cross_rank_failure(rollback_root)
    worker_root = tmp_path / "worker-failure"
    worker_root.mkdir()
    _assert_register_session_raises_on_worker_failure(worker_root)


def _assert_register_session_rolls_back_on_cross_rank_failure(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)
    coordinator.broadcast_adapter_state = Mock()
    coordinator._sync_collective_error = Mock(return_value="rank 0: pending gradients")

    session_spec = trainer._session_spec(5e-5)
    with pytest.raises(RuntimeError, match="Session registration failed: rank 0: pending gradients"):
        asyncio.run(
            coordinator.handle_register_session(
                {"payload": RegisterSessionData(model_id="policy-fail", session_spec=session_spec, materialize=True)}
            )
        )

    assert "policy-fail" not in trainer.lora_session_specs
    assert not trainer.adapter_manager.has_adapter("policy-fail")
    coordinator.broadcast_adapter_state.assert_not_called()


def _assert_register_session_raises_on_worker_failure(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)

    def _fail_register(*args, **kwargs):
        raise ValueError("boom")

    trainer.register_session = _fail_register
    coordinator = AdapterCoordinator(trainer=trainer, rank=1, world_size=2, cpu_group=None)

    with pytest.raises(RuntimeError, match="Session registration failed: boom"):
        asyncio.run(
            coordinator.handle_register_session(
                {
                    "payload": RegisterSessionData(
                        model_id="policy-g",
                        session_spec=trainer._session_spec(3e-5),
                        materialize=False,
                    )
                }
            )
        )


def _assert_handle_save_adapter_state_requires_evicted_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "adapters"
    checkpoint_dir.mkdir()

    trainer = _FakeTrainer(checkpoint_dir)
    trainer.register_session("policy-save", trainer._session_spec(4e-5), materialize=False)
    coordinator = AdapterCoordinator(trainer=trainer, rank=0, world_size=1, cpu_group=None)

    with pytest.raises(RuntimeError, match="Adapter state save failed: .*Refusing to recreate fresh state"):
        asyncio.run(
            coordinator.handle_save_adapter_state(
                {
                    "payload": AdapterStateData(
                        model_id="policy-save",
                        path=str(tmp_path / "save-target"),
                        save_optimizer=True,
                    )
                }
            )
        )

    assert trainer.register_calls == []
    assert trainer.save_calls == []
