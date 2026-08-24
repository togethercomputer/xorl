"""Tests for checkpoint-manager save failure handling."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xorl.models.exact_contract import set_glm52_exact_active_lora
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.session_spec import normalize_session_spec


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "checkpoint" / "manager.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_checkpoint_manager", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
CheckpointManager = _MODULE.CheckpointManager


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeOptimizer:
    def state_dict(self):
        return {"state": {}, "param_groups": []}


class _FakeAdapterState:
    def __init__(self):
        self.global_step = 7
        self.global_forward_backward_step = 11
        self.lr = 2e-5
        self.optimizer = _FakeOptimizer()
        self.local_params = {"adapter.weight.lora_A": nn.Parameter(torch.ones(1, 1))}
        self.tensor_layouts = {}
        self.layout_fingerprint = "f" * 64
        self.gradient_ownership_plan = None
        self.session_spec = {
            "lora_config": {
                "lora_rank": 4,
                "lora_alpha": 16,
            },
            "optimizer_config": {
                "type": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "optimizer_dtype": "bf16",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "optimizer_kwargs": {},
            },
        }

    @property
    def lora_params(self):
        return self.local_params


class _FakeAdapterManager:
    def __init__(self):
        self.checkpoint_dir = "/tmp/adapters"
        self.current_adapter_id = "policy-a"
        self.adapters = {"policy-a": _FakeAdapterState()}

    def get_adapter_state(self, model_id: str):
        return self.adapters[model_id]

    def get_global_step(self, model_id: str) -> int:
        return self.adapters[model_id].global_step

    def get_adapter_session_spec(self, model_id: str):
        return self.adapters[model_id].session_spec

    def validate_weight_publication(self, model_id: str) -> None:
        assert model_id in self.adapters

    def validate_strict_checkpoint_publication(self, model_id: str) -> None:
        assert model_id in self.adapters

    def switch_adapter(self, model_id: str, auto_register: bool = False) -> bool:
        return model_id in self.adapters


class _DummyLoRALayer(nn.Module):
    def __init__(self, *, max_rank: int = 4) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(max_rank, 8))
        self.lora_B = nn.Parameter(torch.zeros(8, max_rank))
        self.active_r = max_rank
        self.active_lora_alpha = 16

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha


class _DummyLoRAModel(nn.Module):
    def __init__(self, *, max_rank: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.o_proj = _DummyLoRALayer(max_rank=max_rank)


class _ExactActiveLoRAComponent(nn.Module):
    _glm52_exact_active_lora_component = True


class _Dsv4ExactActiveLoRAComponent(nn.Module):
    _dsv4_flash_exact_active_lora_component = True


def _build_checkpoint_manager() -> CheckpointManager:
    manager = object.__new__(CheckpointManager)
    manager.rank = 0
    manager.local_rank = 0
    manager.lora_config = {"enable_lora": True}
    manager._adapter_manager = _FakeAdapterManager()
    return manager


def test_detached_single_tenant_publication_uses_live_pp_publisher() -> None:
    manager = object.__new__(CheckpointManager)
    manager._adapter_manager = None
    expected = {"model.layers.7.self_attn.q_proj.lora_A": torch.ones(2, 3)}

    class _Publisher:
        def materialize_live_model_logical_state_dict(self, *, destination_rank):
            assert destination_rank == 0
            return expected

    manager._detached_adapter_publisher = _Publisher()

    assert manager._gather_adapter_lora_params("default") is expected


def _build_fast_save_manager(tmp_path: Path) -> CheckpointManager:
    model = _DummyLoRAModel(max_rank=4)
    adapter_manager = LoRAAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        lora_config={
            "base_model": "Qwen/Qwen3-8B",
            "lora_rank": 4,
            "lora_alpha": 16,
        },
    )
    adapter_manager.register_adapter(
        "policy-a",
        session_spec=normalize_session_spec(
            base_model="Qwen/Qwen3-8B",
            raw_lora_config={
                "lora_rank": 4,
                "lora_alpha": 16,
            },
            raw_optimizer_config={
                "type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "optimizer_dtype": "bf16",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "optimizer_kwargs": {},
            },
            default_rank=4,
            default_alpha=16,
            max_lora_rank=4,
            default_optimizer_type="adamw",
            default_learning_rate=1e-4,
            default_weight_decay=0.01,
            default_optimizer_dtype="bf16",
            default_optimizer_kwargs={},
        ),
        initialize_fresh=True,
    )

    manager = object.__new__(CheckpointManager)
    manager.rank = 0
    manager.local_rank = 0
    manager.model = model
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager.lora_config = {
        "enable_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    manager._adapter_manager = adapter_manager
    return manager


def test_save_adapter_state_raises_before_barrier_when_rank0_write_fails(monkeypatch, tmp_path):
    manager = _build_checkpoint_manager()
    manager._save_lora_weights = lambda path, model_id, **kwargs: None
    manager._sync_collective_error = lambda error: error

    def _fail_write(*args, **kwargs):
        raise PermissionError("disk full")

    monkeypatch.setattr(manager, "_write_adapter_training_artifacts", _fail_write)
    monkeypatch.setattr(
        _MODULE.dist, "barrier", lambda: (_ for _ in ()).throw(AssertionError("barrier should not run"))
    )

    with pytest.raises(RuntimeError, match="Adapter state save failed: disk full"):
        manager.save_adapter_state("policy-a", path=str(tmp_path / "adapter-save"), save_optimizer=True)


def test_exact_active_lora_rejects_full_snapshot_paths_before_downstream_work(monkeypatch, tmp_path):
    manager = object.__new__(CheckpointManager)
    manager.model = nn.Sequential(_ExactActiveLoRAComponent())

    monkeypatch.setattr(
        _MODULE,
        "get_parallel_state",
        lambda: (_ for _ in ()).throw(AssertionError("parallel-state resolution must not run")),
    )
    monkeypatch.setattr(
        _MODULE,
        "ckpt_to_state_dict",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("checkpoint conversion must not run")),
    )

    with pytest.raises(RuntimeError, match="factor-only adapter publication"):
        manager.save_full_weights(str(tmp_path / "full"))
    with pytest.raises(RuntimeError, match="factor-only adapter publication"):
        manager.save_weights_for_sampler(str(tmp_path / "checkpoint"), str(tmp_path / "sampler"))


def test_factor_only_snapshot_guard_leaves_ordinary_models_unrestricted():
    manager = object.__new__(CheckpointManager)
    manager.model = nn.Linear(2, 2)

    manager._require_factor_only_exact_active_lora("ordinary save")


def test_fullparam_rejects_every_legacy_serving_export_before_downstream_work(monkeypatch, tmp_path):
    manager = object.__new__(CheckpointManager)
    manager.model = nn.Linear(2, 2)
    manager.train_config = {"glm52_fullparam_fp8_training": True}

    monkeypatch.setattr(
        _MODULE,
        "get_parallel_state",
        lambda: (_ for _ in ()).throw(AssertionError("parallel-state resolution must not run")),
    )
    monkeypatch.setattr(
        _MODULE,
        "get_model_state_dict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("state extraction must not run")),
    )
    monkeypatch.setattr(
        _MODULE,
        "ckpt_to_state_dict",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("checkpoint conversion must not run")),
    )

    with pytest.raises(RuntimeError, match="checksummed step-boundary payload"):
        manager.extract_model_weights()
    with pytest.raises(RuntimeError, match="checksummed step-boundary payload"):
        manager.extract_full_weights_with_ep()
    with pytest.raises(RuntimeError, match="checksummed step-boundary payload"):
        manager.save_full_weights(str(tmp_path / "full"))
    with pytest.raises(RuntimeError, match="checksummed step-boundary payload"):
        manager.save_weights_for_sampler(str(tmp_path / "checkpoint"), str(tmp_path / "sampler"))


def test_exact_dsv4_adapter_export_rejects_non_bank_format():
    manager = object.__new__(CheckpointManager)
    manager.model = nn.Sequential(_Dsv4ExactActiveLoRAComponent())
    manager.lora_config = {"lora_export_format": "peft"}

    with pytest.raises(RuntimeError, match="dsv4_expert_banks"):
        manager._resolve_lora_export_format()


@pytest.mark.parametrize("rank", [0, 1])
def test_single_writer_full_weight_save_has_one_collective_completion(monkeypatch, tmp_path, rank):
    manager = object.__new__(CheckpointManager)
    manager.rank = rank
    manager.model = nn.Linear(2, 2)
    manager.extract_full_weights_with_ep = lambda: {"weight": torch.ones(2, 2)} if rank == 0 else {}

    save_calls = []
    barrier_calls = []
    monkeypatch.setattr(_MODULE, "save_model_weights", lambda **kwargs: save_calls.append(kwargs))
    monkeypatch.setattr(_MODULE.dist, "barrier", lambda: barrier_calls.append(True))

    result = manager._save_full_weights_single_writer(
        str(tmp_path / "full"),
        "bfloat16",
        base_model_path=None,
    )

    assert len(barrier_calls) == 1
    if rank == 0:
        assert len(save_calls) == 1
        assert save_calls[0]["global_rank"] is None
        assert result["status"] == "success"
    else:
        assert save_calls == []
        assert result == {"status": "skipped", "reason": "non-rank-0"}


@pytest.mark.parametrize("rank", [0, 1])
def test_single_writer_full_weight_save_syncs_writer_failure_before_barrier(monkeypatch, tmp_path, rank):
    manager = object.__new__(CheckpointManager)
    manager.rank = rank
    manager.model = nn.Linear(2, 2)
    manager.extract_full_weights_with_ep = lambda: {"weight": torch.ones(2, 2)} if rank == 0 else {}
    local_errors = []

    def _fail_writer(**_kwargs):
        raise PermissionError("disk full")

    def _sync_error(local_error):
        local_errors.append(local_error)
        return "rank 0: disk full"

    manager._sync_collective_error = _sync_error
    monkeypatch.setattr(_MODULE, "save_model_weights", _fail_writer)
    monkeypatch.setattr(
        _MODULE.dist, "barrier", lambda: (_ for _ in ()).throw(AssertionError("barrier should not run"))
    )

    with pytest.raises(RuntimeError, match="Full-weight save failed: rank 0: disk full"):
        manager._save_full_weights_single_writer(
            str(tmp_path / "full"),
            "bfloat16",
            base_model_path=None,
        )

    assert local_errors == (["disk full"] if rank == 0 else [None])


def test_save_adapter_state_requests_dtype_preserving_lora_checkpoint(monkeypatch, tmp_path):
    manager = _build_checkpoint_manager()
    captured = {}

    def _capture_save_lora_weights(path, model_id, **kwargs):
        captured["path"] = path
        captured["model_id"] = model_id
        captured.update(kwargs)

    monkeypatch.setattr(manager, "_save_lora_weights", _capture_save_lora_weights)
    monkeypatch.setattr(manager, "_write_adapter_training_artifacts", lambda *args, **kwargs: None)

    manager.save_adapter_state("policy-a", path=str(tmp_path / "adapter-save"), save_optimizer=True)

    assert captured["model_id"] == "policy-a"
    assert captured["preserve_lora_dtype"] is True


def test_save_lora_only_raises_before_barrier_when_rank0_write_fails(monkeypatch, tmp_path):
    manager = _build_checkpoint_manager()
    manager._sync_collective_error = lambda error: error

    def _fail_save(*args, **kwargs):
        raise PermissionError("peft write failed")

    monkeypatch.setattr(manager, "_save_lora_weights", _fail_save)
    monkeypatch.setattr(
        _MODULE.dist, "barrier", lambda: (_ for _ in ()).throw(AssertionError("barrier should not run"))
    )

    with pytest.raises(RuntimeError, match="LoRA-only save failed: peft write failed"):
        manager.save_lora_only(str(tmp_path / "adapter-export"), model_id="policy-a")


def test_fast_lora_save_uses_live_adapter_target_modules_not_requested_config(tmp_path):
    manager = _build_fast_save_manager(tmp_path)

    export_dir = tmp_path / "adapter-export"
    manager._save_lora_weights(str(export_dir), "policy-a")

    adapter_config = json.loads((export_dir / "adapter_config.json").read_text(encoding="utf-8"))

    assert sorted(adapter_config["target_modules"]) == ["o_proj"]


def test_moe_lora_save_uses_collective_gather_even_with_adapter_manager(monkeypatch, tmp_path):
    manager = _build_checkpoint_manager()
    manager.model = nn.Module()
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager.lora_config = {
        "enable_lora": True,
        "moe_hybrid_shared_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_target_modules": ["gate_proj", "up_proj", "down_proj"],
    }
    collective_state = {
        "model.layers.0.mlp.experts.gate_proj_lora_A": torch.arange(64, dtype=torch.float32).reshape(1, 8, 8),
        "model.layers.0.mlp.experts.gate_proj_lora_B": torch.arange(128, dtype=torch.float32).reshape(1, 8, 16),
    }
    manager._gather_adapter_lora_params = lambda model_id: collective_state

    captured = {}

    def _capture_save_lora_checkpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(_MODULE, "save_lora_checkpoint", _capture_save_lora_checkpoint)

    manager._save_lora_weights(str(tmp_path / "moe-export"), "policy-a")

    exported_state = captured["lora_state_dict"]
    assert tuple(exported_state["model.layers.0.mlp.experts.gate_proj_lora_A"].shape) == (1, 8, 4)
    assert tuple(exported_state["model.layers.0.mlp.experts.gate_proj_lora_B"].shape) == (1, 4, 16)
    torch.testing.assert_close(
        exported_state["model.layers.0.mlp.experts.gate_proj_lora_A"],
        collective_state["model.layers.0.mlp.experts.gate_proj_lora_A"][..., :4],
    )
    torch.testing.assert_close(
        exported_state["model.layers.0.mlp.experts.gate_proj_lora_B"],
        collective_state["model.layers.0.mlp.experts.gate_proj_lora_B"][:, :4, :],
    )
    assert captured["r"] == 4


def test_glm52_active_lora_save_publishes_frozen_router_bundle(monkeypatch, tmp_path):
    manager = _build_fast_save_manager(tmp_path)
    manager.global_step = 0
    manager._adapter_manager.get_adapter_state("policy-a").global_step = 7
    manager.model.exact_component = _ExactActiveLoRAComponent()
    manager.model.config = type(
        "Config",
        (),
        {
            "train_router": False,
            "first_k_dense_replace": 3,
            "num_hidden_layers": 5,
        },
    )()

    router_state = {
        "layer.3.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "layer.4.weight": torch.full((2, 2), 2, dtype=torch.bfloat16),
    }
    calls = {}

    def _gather_router(model, *, destination_rank):
        calls["gather"] = (model, destination_rank)
        return router_state

    def _save_router(directory, state, *, weight_step, expected_layer_ids):
        calls["save"] = (directory, state, weight_step, expected_layer_ids)
        return {
            "schema": "xorl.glm52_router_bundle.v1",
            "tensor_file": "xorl_router/xorl_glm52_router.safetensors",
            "sha256": "a" * 64,
            "router_count": 2,
            "layer_ids": [3, 4],
            "weight_step": weight_step,
        }

    monkeypatch.setattr(_MODULE, "gather_glm52_router_weights_across_ranks", _gather_router)
    monkeypatch.setattr(_MODULE, "save_glm52_router_bundle", _save_router)
    monkeypatch.setattr(
        _MODULE,
        "mark_adapter_config_with_glm52_router_bundle",
        lambda directory, manifest: calls.setdefault("mark", (directory, manifest)),
    )

    export_dir = tmp_path / "glm52-export"
    manager._save_lora_weights(str(export_dir), "policy-a")

    assert calls["gather"] == (manager.model, 0)
    assert calls["save"] == (str(export_dir), router_state, 7, [3, 4])
    assert calls["mark"][0] == str(export_dir)
    assert calls["mark"][1]["router_count"] == 2


def test_glm52_active_lora_publication_uses_replicated_config_stamp_on_dense_stage():
    manager = object.__new__(CheckpointManager)
    manager.model = nn.Sequential(nn.Linear(2, 2))
    manager.model.config = type("Config", (), {})()
    set_glm52_exact_active_lora(manager.model.config, enabled=True)

    assert manager._has_glm52_exact_active_lora()


def test_moe_lora_save_uses_resolved_target_modules_for_detection(monkeypatch, tmp_path):
    manager = _build_checkpoint_manager()

    class _ModelWithStackedMoELoRA:
        def named_parameters(self):
            yield "model.layers.0.mlp.experts.gate_proj_lora_A", nn.Parameter(torch.ones(1, 8, 4))

    manager.model = _ModelWithStackedMoELoRA()
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager.lora_config = {
        "enable_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
    }
    manager.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    manager.lora_alpha_value = 16
    collective_state = {"model.layers.0.mlp.experts.gate_proj_lora_A": torch.ones(1, 8, 4)}
    manager._gather_adapter_lora_params = lambda model_id: collective_state

    captured = {}

    def _capture_save_lora_checkpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(_MODULE, "save_lora_checkpoint", _capture_save_lora_checkpoint)

    manager._save_lora_weights(str(tmp_path / "moe-export"), "policy-a")

    assert captured["lora_state_dict"].keys() == collective_state.keys()
    torch.testing.assert_close(
        captured["lora_state_dict"]["model.layers.0.mlp.experts.gate_proj_lora_A"],
        collective_state["model.layers.0.mlp.experts.gate_proj_lora_A"],
    )
    assert captured["r"] == 4


def test_lora_save_forwards_export_format(monkeypatch, tmp_path):
    manager = object.__new__(CheckpointManager)
    manager.rank = 0
    manager.local_rank = 0
    manager.global_step = 0
    manager.model = nn.Module()
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager._adapter_manager = None
    manager.lora_target_modules = ["gate_proj", "up_proj", "down_proj"]
    manager.lora_alpha_value = 16
    manager.lora_config = {
        "enable_lora": True,
        "moe_hybrid_shared_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_export_format": "sglang_shared_outer",
    }

    monkeypatch.setattr(_MODULE, "get_lora_state_dict", lambda model: {})

    captured = {}

    def _capture_save_lora_checkpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(_MODULE, "save_lora_checkpoint", _capture_save_lora_checkpoint)

    manager._save_lora_weights(str(tmp_path / "sglang-export"), "default")

    assert captured["lora_export_format"] == "sglang_shared_outer"


def test_adapter_training_artifacts_include_strict_target_manifest(tmp_path):
    manager = _build_checkpoint_manager()
    manifest = {
        "schema_version": 1,
        "config_rank": 4,
        "config_alpha": 16.0,
        "target_modules": ["o_proj"],
        "expected_modules": [{"pattern": "model.layers.*.self_attn.o_proj", "count": 1, "rank": 4}],
        "allow_unlisted": False,
        "source_lora_key_fingerprint": "a" * 64,
        "source_lora_shape_fingerprint": "b" * 64,
    }
    manager.lora_config["lora_target_manifest"] = manifest

    manager._write_adapter_training_artifacts(
        str(tmp_path),
        "policy-a",
        manager._adapter_manager.get_adapter_state("policy-a"),
        save_optimizer=True,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["optimizer_format"] == "sharded_v3"
    assert json.loads((tmp_path / "lora_target_manifest.json").read_text()) == manifest
