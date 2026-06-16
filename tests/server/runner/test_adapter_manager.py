"""Tests for adapter-manager optimizer integration."""

import asyncio
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from xorl.optim import SignSGD
from xorl.server.protocol.operations import AdapterStateData
from xorl.server.runner.adapters.adapter_coordinator import AdapterCoordinator
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.session_spec import normalize_session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server]


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


def _build_manager(tmp_path: Path, **kwargs) -> LoRAAdapterManager:
    max_rank = kwargs.pop("max_rank", 4)
    return LoRAAdapterManager(
        _DummyLoRAModel(max_rank=max_rank),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        **kwargs,
    )


class _CoordinatorTrainer:
    def __init__(self, adapter_manager: LoRAAdapterManager) -> None:
        self.adapter_manager = adapter_manager
        self.lora_session_specs = {}

    def register_session(self, model_id: str, session_spec: dict, materialize: bool = False, **kwargs):
        self.lora_session_specs[model_id] = session_spec
        if materialize and not self.adapter_manager.has_adapter(model_id):
            self.adapter_manager.register_adapter(
                model_id=model_id,
                session_spec=session_spec,
                initialize_fresh=kwargs.get("initialize_fresh", True),
            )
        return {"registered": True, "model_id": model_id}

    def get_lora_session_spec(self, model_id: str) -> dict:
        if model_id in self.lora_session_specs:
            return self.lora_session_specs[model_id]
        return self.adapter_manager.get_adapter_session_spec(model_id)

    def register_lora_adapter(self, model_id: str, lr=None):
        session_spec = dict(self.get_lora_session_spec(model_id))
        if lr is not None:
            session_spec.setdefault("optimizer_config", {})["learning_rate"] = lr
        self.adapter_manager.register_adapter(model_id=model_id, session_spec=session_spec, initialize_fresh=True)
        return {"registered": True, "model_id": model_id}

    def load_adapter_state(self, model_id: str, path: str, load_optimizer: bool = True, lr=None):
        return self.adapter_manager.load_adapter_state(
            model_id=model_id,
            path=path,
            load_optimizer=load_optimizer,
            lr=lr,
        )


def _session_spec(*, rank: int, alpha: int, optimizer_type: str, lr: float, weight_decay: float = 0.0) -> dict:
    return {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": True,
        "lora_config": {
            "lora_rank": rank,
            "lora_alpha": alpha,
        },
        "optimizer_config": {
            "type": optimizer_type,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer_dtype": "bf16",
            "betas": None if optimizer_type in {"sgd", "signsgd"} else [0.9, 0.95],
            "eps": None if optimizer_type in {"sgd", "signsgd"} else 1e-8,
            "optimizer_kwargs": {},
        },
    }


def test_register_adapter_uses_shared_optimizer_factory_and_checkpoint_dir(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="signsgd", weight_decay=0.25)

    manager.register_adapter("policy-a", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-a")

    assert isinstance(state.optimizer, SignSGD)

    save_result = manager.save_adapter_state("policy-a")
    save_path = Path(save_result["path"])
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))

    assert save_path == tmp_path / "adapters" / "policy-a"
    assert metadata["optimizer"]["type"] == "signsgd"
    assert metadata["optimizer"]["weight_decay"] == pytest.approx(0.25)
    assert metadata["optimizer"]["betas"] == [0.9, 0.95]
    assert metadata["optimizer"]["eps"] == pytest.approx(1e-8)


def test_save_adapter_state_preserves_lora_weight_dtype(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="adamw")
    manager.register_adapter("policy-fp32", lr=0.1, initialize_fresh=True)

    save_path = Path(manager.save_adapter_state("policy-fp32")["path"])
    weights = safetensors_load_file(str(save_path / "adapter_model.safetensors"))

    assert weights["base_model.model.model.layers.0.self_attn.o_proj.lora_A"].dtype == torch.float32
    assert weights["base_model.model.model.layers.0.self_attn.o_proj.lora_B"].dtype == torch.float32


def test_load_adapter_state_uses_checkpoint_optimizer_contract_for_fresh_session(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="signsgd")
    source_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    result = target_manager.load_adapter_state("policy-b", checkpoint_path, load_optimizer=True)

    assert result["model_id"] == "policy-b"
    assert isinstance(target_manager.get_adapter_state("policy-b").optimizer, SignSGD)
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["type"] == "signsgd"


def test_adapter_coordinator_loads_checkpoint_without_placeholder_spec_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="signsgd")
    source_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    coordinator = AdapterCoordinator(
        trainer=_CoordinatorTrainer(target_manager),
        rank=0,
        world_size=1,
        cpu_group=None,
    )

    result = asyncio.run(
        coordinator.handle_load_adapter_state(
            {
                "payload": AdapterStateData(
                    model_id="policy-b",
                    path=checkpoint_path,
                    load_optimizer=True,
                )
            }
        )
    )

    assert result["model_id"] == "policy-b"
    assert isinstance(target_manager.get_adapter_state("policy-b").optimizer, SignSGD)
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["learning_rate"] == pytest.approx(
        0.1
    )


def test_adapter_coordinator_auto_load_evicted_uses_checkpoint_session_spec(tmp_path):
    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    source_manager = _build_manager(tmp_path / "source", optimizer_type="signsgd")
    source_manager.register_adapter("policy-evicted", lr=0.2, initialize_fresh=True)
    checkpoint_path = Path(target_manager.checkpoint_dir) / "evicted" / "policy-evicted"
    source_manager.save_adapter_state("policy-evicted", str(checkpoint_path))

    coordinator = AdapterCoordinator(
        trainer=_CoordinatorTrainer(target_manager),
        rank=0,
        world_size=1,
        cpu_group=None,
    )

    was_loaded, loaded_path = coordinator.auto_load_if_evicted("policy-evicted")

    assert was_loaded is True
    assert loaded_path == str(checkpoint_path)
    assert isinstance(target_manager.get_adapter_state("policy-evicted").optimizer, SignSGD)
    assert target_manager.get_adapter_session_spec("policy-evicted")["optimizer_config"]["learning_rate"] == (
        pytest.approx(0.2)
    )


def test_load_adapter_state_rejects_registered_session_spec_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="signsgd")
    source_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    target_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)

    with pytest.raises(ValueError, match="Checkpoint session spec does not match"):
        target_manager.load_adapter_state("policy-b", checkpoint_path, load_optimizer=True)


def test_load_adapter_state_allows_lr_override_for_registered_session(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1, weight_decay=0.01)
    source_manager.register_adapter("policy-lr-override", session_spec=source_spec, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-lr-override")["path"]

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    target_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)
    target_manager.register_adapter("policy-lr-override", session_spec=target_spec, initialize_fresh=True)

    result = target_manager.load_adapter_state(
        "policy-lr-override",
        checkpoint_path,
        load_optimizer=True,
        lr=0.2,
    )

    target_state = target_manager.get_adapter_state("policy-lr-override")
    assert result["model_id"] == "policy-lr-override"
    assert target_state.lr == pytest.approx(0.2)
    assert target_state.optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    assert target_manager.get_adapter_session_spec("policy-lr-override")["optimizer_config"][
        "learning_rate"
    ] == pytest.approx(0.2)


def test_load_adapter_state_allows_weights_only_optimizer_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="signsgd")
    source_spec = _session_spec(rank=4, alpha=16, optimizer_type="signsgd", lr=0.1)
    source_manager.register_adapter("policy-b", session_spec=source_spec, initialize_fresh=True)
    source_state = source_manager.get_adapter_state("policy-b")
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(1.25)
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(0.5)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    target_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)
    target_manager.register_adapter("policy-b", session_spec=target_spec, initialize_fresh=True)

    result = target_manager.load_adapter_state("policy-b", checkpoint_path, load_optimizer=False)

    target_state = target_manager.get_adapter_state("policy-b")
    assert result["model_id"] == "policy-b"
    assert isinstance(target_state.optimizer, torch.optim.AdamW)
    assert target_state.lr == pytest.approx(0.05)
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["type"] == "adamw"
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["learning_rate"] == pytest.approx(
        0.05
    )
    assert target_state.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"],
        torch.full((4, 8), 1.25),
    )
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"],
        torch.full((8, 4), 0.5),
    )


def test_load_adapter_state_weights_only_restores_checkpoint_lr_for_same_optimizer_contract(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1, weight_decay=0.01)
    source_manager.register_adapter("policy-lr-restore", session_spec=source_spec, initialize_fresh=True)
    source_state = source_manager.get_adapter_state("policy-lr-restore")
    for param in source_state.lora_params.values():
        param.grad = torch.ones_like(param)
    source_manager.optim_step("policy-lr-restore", lr=0.25)
    checkpoint_path = source_manager.save_adapter_state("policy-lr-restore")["path"]

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    target_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)
    target_manager.register_adapter("policy-lr-restore", session_spec=target_spec, initialize_fresh=True)

    target_manager.load_adapter_state("policy-lr-restore", checkpoint_path, load_optimizer=False)

    target_state = target_manager.get_adapter_state("policy-lr-restore")
    assert target_state.lr == pytest.approx(0.25)
    assert target_state.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
    assert target_manager.get_adapter_session_spec("policy-lr-restore")["optimizer_config"][
        "learning_rate"
    ] == pytest.approx(0.25)


def test_load_adapter_state_rejects_checkpoint_target_module_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="adamw")
    source_manager.register_adapter("policy-structure", lr=0.1, initialize_fresh=True)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-structure")["path"])

    adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config["target_modules"] = ["q_proj"]
    (checkpoint_path / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    with pytest.raises(ValueError, match="target_modules"):
        target_manager.load_adapter_state("policy-structure", str(checkpoint_path), load_optimizer=True)


def test_load_adapter_state_rejects_checkpoint_with_missing_lora_tensors(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="adamw")
    source_manager.register_adapter("policy-missing", lr=0.1, initialize_fresh=True)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-missing")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    weights.pop("base_model.model.model.layers.0.self_attn.o_proj.lora_B")
    safetensors_save_file(weights, str(weights_path))

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    with pytest.raises(ValueError, match="parameter set does not match"):
        target_manager.load_adapter_state("policy-missing", str(checkpoint_path), load_optimizer=True)


def test_load_adapter_state_rolls_back_freshly_registered_adapter_on_failure(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="adamw")
    source_manager.register_adapter("policy-rollback", lr=0.1, initialize_fresh=True)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-rollback")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    weights.pop("base_model.model.model.layers.0.self_attn.o_proj.lora_B")
    safetensors_save_file(weights, str(weights_path))

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    assert "policy-rollback" not in target_manager.adapters

    with pytest.raises(ValueError, match="parameter set does not match"):
        target_manager.load_adapter_state("policy-rollback", str(checkpoint_path), load_optimizer=True)

    assert "policy-rollback" not in target_manager.adapters


def test_load_adapter_state_accepts_weight_suffixed_checkpoint_tensor_names(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_manager.register_adapter("policy-weight-suffix", lr=0.1, initialize_fresh=True)
    source_state = source_manager.get_adapter_state("policy-weight-suffix")
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(1.5)
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(0.75)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-weight-suffix")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    renamed_weights = {}
    for key, value in weights.items():
        renamed_weights[f"{key}.weight"] = value
    safetensors_save_file(renamed_weights, str(weights_path))

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    result = target_manager.load_adapter_state("policy-weight-suffix", str(checkpoint_path), load_optimizer=True)

    target_state = target_manager.get_adapter_state("policy-weight-suffix")
    assert result["model_id"] == "policy-weight-suffix"
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"],
        torch.full((4, 8), 1.5),
    )
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"],
        torch.full((8, 4), 0.75),
    )


def test_load_adapter_state_rejects_checkpoint_rank_exceeding_model_capacity(tmp_path):
    source_manager = _build_manager(
        tmp_path / "source",
        optimizer_type="adamw",
        max_rank=8,
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 8, "lora_alpha": 16},
    )
    source_manager.register_adapter(
        "policy-r8",
        session_spec=_session_spec(rank=8, alpha=16, optimizer_type="adamw", lr=0.1),
        initialize_fresh=True,
    )
    checkpoint_path = source_manager.save_adapter_state("policy-r8")["path"]

    target_manager = _build_manager(
        tmp_path / "target",
        optimizer_type="adamw",
        max_rank=4,
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )
    with pytest.raises(ValueError, match="exceeds live model LoRA capacity"):
        target_manager.load_adapter_state("policy-r8", checkpoint_path, load_optimizer=True)


def test_register_adapter_refuses_to_evict_dirty_adapter(tmp_path):
    manager = _build_manager(tmp_path, max_adapters=1, optimizer_type="adamw")
    manager.register_adapter("policy-a", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1))

    dirty_state = manager.get_adapter_state("policy-a")
    dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].grad = torch.ones_like(
        dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"]
    )

    with pytest.raises(RuntimeError, match="pending gradients"):
        manager.register_adapter(
            "policy-b", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.2)
        )

    assert manager.has_adapter("policy-a")
    assert not manager.has_adapter("policy-b")


def test_register_adapter_evicts_clean_adapter_before_dirty_one(tmp_path):
    manager = _build_manager(tmp_path, max_adapters=2, optimizer_type="adamw")
    manager.register_adapter("policy-a", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1))
    manager.register_adapter("policy-b", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.2))

    dirty_state = manager.get_adapter_state("policy-a")
    clean_state = manager.get_adapter_state("policy-b")
    dirty_state.last_access_time = 1.0
    clean_state.last_access_time = 2.0
    dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].grad = torch.ones_like(
        dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"]
    )

    manager.register_adapter("policy-c", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.3))

    assert manager.has_adapter("policy-a")
    assert not manager.has_adapter("policy-b")
    assert manager.has_adapter("policy-c")


def test_multi_adapter_manager_supports_mixed_ranks_and_optimizers(tmp_path):
    manager = _build_manager(
        tmp_path,
        optimizer_type="adamw",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )

    small_spec = _session_spec(rank=2, alpha=8, optimizer_type="signsgd", lr=0.2)
    large_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)

    manager.register_adapter("policy-small", session_spec=small_spec, initialize_fresh=True)
    manager.register_adapter("policy-large", session_spec=large_spec, initialize_fresh=True)

    small_state = manager.get_adapter_state("policy-small")
    large_state = manager.get_adapter_state("policy-large")
    layer = manager.model.model.layers[0].self_attn.o_proj

    assert isinstance(small_state.optimizer, SignSGD)
    assert isinstance(large_state.optimizer, torch.optim.AdamW)
    assert tuple(small_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].shape) == (2, 8)
    assert tuple(small_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].shape) == (8, 2)
    assert tuple(large_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].shape) == (4, 8)
    assert tuple(large_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].shape) == (8, 4)

    small_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(1.5)
    small_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(2.5)
    large_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(3.5)
    large_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(4.5)

    manager.prepare_forward("policy-small")
    assert layer.active_r == 2
    assert layer.active_lora_alpha == 8
    assert torch.allclose(layer.lora_A[:2], torch.full((2, 8), 1.5))
    assert torch.count_nonzero(layer.lora_A[2:]) == 0
    assert torch.allclose(layer.lora_B[:, :2], torch.full((8, 2), 2.5))
    assert torch.count_nonzero(layer.lora_B[:, 2:]) == 0

    layer.lora_A.grad = torch.full_like(layer.lora_A, 1.0)
    layer.lora_B.grad = torch.full_like(layer.lora_B, 2.0)
    manager.capture_gradients("policy-small")
    assert tuple(small_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].grad.shape) == (2, 8)
    assert tuple(small_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].grad.shape) == (8, 2)
    small_grad_norm = manager.optim_step("policy-small", lr=0.2)
    assert small_grad_norm > 0
    assert manager.get_global_step("policy-small") == 1

    manager.prepare_forward("policy-large")
    assert layer.active_r == 4
    assert layer.active_lora_alpha == 16
    assert torch.allclose(layer.lora_A, torch.full((4, 8), 3.5))
    assert torch.allclose(layer.lora_B, torch.full((8, 4), 4.5))

    layer.lora_A.grad = torch.full_like(layer.lora_A, 3.0)
    layer.lora_B.grad = torch.full_like(layer.lora_B, 4.0)
    manager.capture_gradients("policy-large")
    assert tuple(large_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].grad.shape) == (4, 8)
    assert tuple(large_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].grad.shape) == (8, 4)
    large_grad_norm = manager.optim_step("policy-large", lr=0.05)
    assert large_grad_norm > 0
    assert manager.get_global_step("policy-large") == 1
    assert large_state.optimizer.state

    small_checkpoint = manager.save_adapter_state("policy-small")["path"]
    large_checkpoint = manager.save_adapter_state("policy-large")["path"]

    reloaded_manager = _build_manager(
        tmp_path,
        optimizer_type="sgd",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )
    reloaded_manager.load_adapter_state("policy-small", small_checkpoint, load_optimizer=True)
    reloaded_manager.load_adapter_state("policy-large", large_checkpoint, load_optimizer=True)

    assert isinstance(reloaded_manager.get_adapter_state("policy-small").optimizer, SignSGD)
    assert isinstance(reloaded_manager.get_adapter_state("policy-large").optimizer, torch.optim.AdamW)
    assert reloaded_manager.get_adapter_session_spec("policy-small")["lora_config"]["lora_rank"] == 2
    assert reloaded_manager.get_adapter_session_spec("policy-large")["lora_config"]["lora_rank"] == 4


def test_save_adapter_state_persists_current_learning_rate(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="adamw")
    session_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1, weight_decay=0.01)
    manager.register_adapter("policy-lr", session_spec=session_spec, initialize_fresh=True)

    state = manager.get_adapter_state("policy-lr")
    for param in state.lora_params.values():
        param.grad = torch.ones_like(param)

    manager.optim_step("policy-lr", lr=0.25)
    checkpoint_path = Path(manager.save_adapter_state("policy-lr")["path"])
    session_spec_json = json.loads((checkpoint_path / "session_spec.json").read_text(encoding="utf-8"))
    metadata_json = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))

    assert manager.get_adapter_session_spec("policy-lr")["optimizer_config"]["learning_rate"] == pytest.approx(0.25)
    assert session_spec_json["optimizer_config"]["learning_rate"] == pytest.approx(0.25)
    assert metadata_json["lr"] == pytest.approx(0.25)
    assert metadata_json["optimizer"]["learning_rate"] == pytest.approx(0.25)


def test_register_adapter_hoists_common_adam_hparams_out_of_optimizer_kwargs(tmp_path):
    manager = _build_manager(
        tmp_path,
        optimizer_type="adamw",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )

    session_spec = normalize_session_spec(
        base_model="Qwen/Qwen3-8B",
        raw_lora_config={"lora_rank": 4, "lora_alpha": 16},
        raw_optimizer_config={
            "type": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.02,
            "optimizer_kwargs": {
                "betas": [0.8, 0.88],
                "eps": 1e-7,
                "capturable": True,
            },
        },
        default_rank=4,
        default_alpha=16,
        max_lora_rank=16,
        default_optimizer_type="adamw",
        default_learning_rate=1e-5,
        default_weight_decay=0.01,
        default_optimizer_dtype="bf16",
        default_optimizer_kwargs={},
        server_lora_config={"enable_lora": True, "lora_rank": 4, "lora_alpha": 16, "max_lora_rank": 16},
    )

    manager.register_adapter("policy-adam", session_spec=session_spec, initialize_fresh=True)
    state = manager.get_adapter_state("policy-adam")

    assert isinstance(state.optimizer, torch.optim.AdamW)
    assert state.optimizer.defaults["betas"] == (0.8, 0.88)
    assert state.optimizer.defaults["eps"] == pytest.approx(1e-7)
    assert state.optimizer.defaults["capturable"] is True
    assert manager.get_adapter_session_spec("policy-adam")["optimizer_config"]["optimizer_kwargs"] == {
        "capturable": True
    }


def test_muon_set_lr_preserves_muon_param_group_lr(tmp_path):
    manager = _build_manager(
        tmp_path,
        optimizer_type="muon",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )
    session_spec = _session_spec(rank=4, alpha=16, optimizer_type="muon", lr=1e-4, weight_decay=0.01)
    session_spec["optimizer_config"]["optimizer_kwargs"] = {
        "muon_lr": 0.02,
        "muon_ns_use_quack_kernels": False,
    }

    manager.register_adapter("policy-muon", session_spec=session_spec, initialize_fresh=True)
    state = manager.get_adapter_state("policy-muon")
    muon_groups = [param_group for param_group in state.optimizer.param_groups if param_group.get("use_muon", False)]
    assert muon_groups
    assert all(param_group["lr"] == pytest.approx(0.02) for param_group in muon_groups)

    manager.set_lr("policy-muon", 2e-4)
    assert state.lr == pytest.approx(2e-4)
    assert all(param_group["lr"] == pytest.approx(0.02) for param_group in muon_groups)

    manager.optim_step("policy-muon", lr=3e-4)
    assert state.lr == pytest.approx(3e-4)
    assert all(param_group["lr"] == pytest.approx(0.02) for param_group in muon_groups)
