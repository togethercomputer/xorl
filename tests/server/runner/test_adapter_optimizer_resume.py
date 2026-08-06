"""Adapter-checkpoint resume validation: optimizer moments must survive save/load.

Regression tests for the rank-0-only ``optimizer.pt`` defect: the legacy pickle
format saved only rank 0's Adam moments, and loading it on every rank of an EP
run silently assigned rank-0 moments to other ranks' expert parameters.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xorl.server.runner.adapters import manager as adapter_manager_module
from xorl.server.runner.adapters.manager import (
    OPTIMIZER_SHARD_MANIFEST_FILENAME,
    AdapterState,
    LoRAAdapterManager,
    _adapter_param_structure_fingerprint,
    _descriptor_structure_fingerprint,
    _layout_descriptor_fingerprint,
    _optimizer_shard_filename,
    _reshard_adapter_optimizer_state,
    _save_optimizer_state_safetensors,
    load_adapter_optimizer_shards,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    clone_state_to_cpu as _clone_state_to_cpu,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    same_optimizer_value as _same_optimizer_value,
)
from xorl.server.runner.adapters.sharded_state import AdapterTensorLayout

from .test_adapter_manager import _build_manager, _session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server]


@pytest.fixture(autouse=True)
def _trusted_server_artifact_root(tmp_path: Path, monkeypatch):
    """Keep cross-manager resume fixtures inside the Foundation trust root."""
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(tmp_path))


def _register(manager: LoRAAdapterManager, model_id: str, lr: float = 1e-2) -> None:
    manager.register_adapter(
        model_id=model_id,
        session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=lr),
        initialize_fresh=True,
    )


def _fill_params(manager: LoRAAdapterManager, model_id: str, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    state = manager.get_adapter_state(model_id)
    for param in state.lora_params.values():
        param.data.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))


def _apply_step(manager: LoRAAdapterManager, model_id: str, seed: int, lr: float = 1e-2) -> None:
    generator = torch.Generator().manual_seed(seed)
    state = manager.get_adapter_state(model_id)
    for param in state.lora_params.values():
        param.grad = torch.randn(param.shape, generator=generator, dtype=param.dtype)
    manager.optim_step(model_id, lr=lr)


def _moment_tensors(manager: LoRAAdapterManager, model_id: str) -> dict:
    state_dict = manager.get_adapter_state(model_id).optimizer.state_dict()
    return {
        f"{param_id}.{key}": value
        for param_id, param_state in state_dict["state"].items()
        for key, value in param_state.items()
        if isinstance(value, torch.Tensor)
    }


def test_adapter_optimizer_parameter_map_is_canonical_ordered():
    parameters = {
        "_orig_mod.z.lora_B": torch.nn.Parameter(torch.ones(2)),
        "a.lora_A": torch.nn.Parameter(torch.ones(2)),
    }
    ordered = LoRAAdapterManager._optimizer_parameter_map(parameters)
    assert list(ordered) == ["a.lora_A", "_orig_mod.z.lora_B"]


def test_adapter_optimizer_fingerprint_is_canonical_and_ignores_wrappers():
    first = {
        "_orig_mod.z.lora_B": torch.nn.Parameter(torch.ones(2)),
        "a.lora_A": torch.nn.Parameter(torch.ones(2)),
    }
    equivalent = {
        "_fsdp_wrapped_module.a.lora_A": torch.nn.Parameter(torch.ones(2)),
        "z.lora_B": torch.nn.Parameter(torch.ones(2)),
    }
    assert _adapter_param_structure_fingerprint(first) == _adapter_param_structure_fingerprint(equivalent)


def test_transaction_snapshot_recursively_clones_tensor_state_to_cpu():
    source = torch.arange(4, dtype=torch.float32)
    snapshot = _clone_state_to_cpu(
        {
            "tensor": source,
            "nested": [source.view(2, 2), (torch.tensor(7),)],
            "scalar": 3,
        }
    )

    source.add_(10)
    assert snapshot["tensor"].device.type == "cpu"
    assert snapshot["nested"][0].device.type == "cpu"
    assert snapshot["nested"][1][0].device.type == "cpu"
    assert torch.equal(snapshot["tensor"], torch.arange(4, dtype=torch.float32))
    assert snapshot["tensor"].data_ptr() != source.data_ptr()
    assert snapshot["scalar"] == 3


def test_live_optimizer_binding_rejects_noncanonical_parameter_object_before_state_read():
    name = "layer.lora_A"
    layout = _one_dimensional_layout(name, offset=0, size=4, logical_size=4)
    state = _target_adapter_state(name, layout)
    imposter = torch.nn.Parameter(torch.zeros_like(state.local_params[name]))
    state.optimizer.param_groups[0]["params"][0] = imposter

    with pytest.raises(RuntimeError, match="parameter objects are not in canonical"):
        adapter_manager_module._validate_live_optimizer_binding(state)


def test_collective_failure_after_optimizer_mutation_is_fatal_without_rollback(monkeypatch):
    resident = {"state": {0: {"exp_avg": torch.tensor([1.0])}}, "param_groups": [{"params": [0]}]}
    target = {"state": {0: {"exp_avg": torch.tensor([2.0])}}, "param_groups": [{"params": [0]}]}

    class RecordingOptimizer:
        def __init__(self):
            self.value = _clone_state_to_cpu(resident)
            self.loads = 0

        def state_dict(self):
            return _clone_state_to_cpu(self.value)

        def load_state_dict(self, value):
            self.loads += 1
            self.value = _clone_state_to_cpu(value)

    optimizer = RecordingOptimizer()
    state = SimpleNamespace(optimizer=optimizer)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("communicator failed")),
    )

    with pytest.raises(RuntimeError, match="must terminate"):
        adapter_manager_module._commit_optimizer_state_transactionally(state, target)

    assert optimizer.loads == 1, "must not attempt rollback coordination on a failed process group"
    assert _same_optimizer_value(optimizer.value, target)


def test_save_writes_sharded_optimizer_with_manifest(tmp_path: Path):
    manager = _build_manager(tmp_path)
    _register(manager, "resume-a")
    _apply_step(manager, "resume-a", seed=1)
    result = manager.save_adapter_state("resume-a")
    path = result["path"]

    assert os.path.exists(os.path.join(path, _optimizer_shard_filename(0)))
    assert not os.path.exists(os.path.join(path, "optimizer.pt"))
    with open(os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME)) as f:
        manifest = json.load(f)
    assert manifest["format_version"] == 3
    assert manifest["world_size"] == 1
    assert manifest["session_rank"] == 4
    assert manifest["per_rank_optimizer_parameter_order"] == [manifest["optimizer_parameter_order"]]
    assert manifest["per_rank_layout_fingerprint"] == [manager.get_adapter_state("resume-a").layout_fingerprint]
    assert manifest["optimizer_parameter_order"] == sorted(
        adapter_manager_module.canonical_parameter_name(name)
        for name, param in manager.get_adapter_state("resume-a").local_params.items()
        if param.numel() > 0
    )
    state = manager.get_adapter_state("resume-a")
    assert manifest["per_rank_param_structure_sha256"] == [_adapter_param_structure_fingerprint(state.lora_params)]


@pytest.mark.parametrize(
    ("manifest_update", "error_match"),
    [
        ({"session_rank": 2}, "saved for session_rank"),
        ({"per_rank_layout_fingerprint": ["0" * 64]}, "Layout fingerprint does not match descriptors"),
        ({"per_rank_optimizer_parameter_order": [["wrong.fqn"]]}, "different optimizer parameter order"),
    ],
)
def test_manifest_identity_contract_refuses_incompatible_optimizer_shards(tmp_path, manifest_update, error_match):
    source = _build_manager(tmp_path / "source")
    _register(source, "resume-contract")
    _apply_step(source, "resume-contract", seed=3)
    path = source.save_adapter_state("resume-contract")["path"]
    manifest_path = os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.update(manifest_update)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    target = _build_manager(tmp_path / "target")
    with pytest.raises(RuntimeError, match=error_match):
        target.load_adapter_state(model_id="resume-contract", path=path, load_optimizer=True)


def test_load_restores_optimizer_moments_and_step_bitwise(tmp_path: Path):
    source = _build_manager(tmp_path)
    _register(source, "resume-b")
    _fill_params(source, "resume-b", seed=7)
    _apply_step(source, "resume-b", seed=11)
    _apply_step(source, "resume-b", seed=13)
    path = source.save_adapter_state("resume-b")["path"]
    source_moments = _moment_tensors(source, "resume-b")
    assert source_moments, "expected non-empty Adam state after two steps"

    target = _build_manager(tmp_path / "target")
    target.load_adapter_state(model_id="resume-b", path=path, load_optimizer=True)
    target_moments = _moment_tensors(target, "resume-b")

    assert set(target_moments) == set(source_moments)
    for key, tensor in source_moments.items():
        assert torch.equal(target_moments[key], tensor), f"moment mismatch: {key}"


def test_resume_matches_uninterrupted_run_bitwise(tmp_path: Path):
    torch.use_deterministic_algorithms(True)
    try:
        uninterrupted = _build_manager(tmp_path / "full")
        _register(uninterrupted, "resume-c")
        _fill_params(uninterrupted, "resume-c", seed=23)
        for seed in (31, 37, 41, 43):
            _apply_step(uninterrupted, "resume-c", seed=seed)

        first_half = _build_manager(tmp_path / "half")
        _register(first_half, "resume-c")
        _fill_params(first_half, "resume-c", seed=23)
        for seed in (31, 37):
            _apply_step(first_half, "resume-c", seed=seed)
        path = first_half.save_adapter_state("resume-c")["path"]

        resumed = _build_manager(tmp_path / "resumed")
        resumed.load_adapter_state(model_id="resume-c", path=path, load_optimizer=True)
        for seed in (41, 43):
            _apply_step(resumed, "resume-c", seed=seed)

        full_state = uninterrupted.get_adapter_state("resume-c")
        resumed_state = resumed.get_adapter_state("resume-c")
        for name, param in full_state.lora_params.items():
            assert torch.equal(resumed_state.lora_params[name].data, param.data), (
                f"resumed weights diverge from uninterrupted run: {name}"
            )
        full_moments = _moment_tensors(uninterrupted, "resume-c")
        resumed_moments = _moment_tensors(resumed, "resume-c")
        for key, tensor in full_moments.items():
            assert torch.equal(resumed_moments[key], tensor), f"moment diverges: {key}"
    finally:
        torch.use_deterministic_algorithms(False)


def test_weights_only_resume_would_diverge(tmp_path: Path):
    """Control: without optimizer state the continuation differs, proving the
    bitwise equality above is a real signal and not step-invariance."""
    full = _build_manager(tmp_path / "full")
    _register(full, "resume-d")
    _fill_params(full, "resume-d", seed=23)
    for seed in (31, 37, 41):
        _apply_step(full, "resume-d", seed=seed)

    half = _build_manager(tmp_path / "half")
    _register(half, "resume-d")
    _fill_params(half, "resume-d", seed=23)
    for seed in (31, 37):
        _apply_step(half, "resume-d", seed=seed)
    path = half.save_adapter_state("resume-d")["path"]

    weights_only = _build_manager(tmp_path / "weights-only")
    weights_only.load_adapter_state(model_id="resume-d", path=path, load_optimizer=False)
    _apply_step(weights_only, "resume-d", seed=41)

    diverged = any(
        not torch.equal(
            weights_only.get_adapter_state("resume-d").lora_params[name].data,
            param.data,
        )
        for name, param in full.get_adapter_state("resume-d").lora_params.items()
    )
    assert diverged, "weights-only resume unexpectedly matched the moment-resumed run"


def test_legacy_pickle_checkpoint_is_always_refused(tmp_path: Path, monkeypatch):
    source = _build_manager(tmp_path)
    _register(source, "resume-e")
    _apply_step(source, "resume-e", seed=3)
    path = source.save_adapter_state("resume-e")["path"]
    # Rewrite as a legacy checkpoint: single optimizer.pt, no manifest.
    state_dict = source.get_adapter_state("resume-e").optimizer.state_dict()
    torch.save(state_dict, os.path.join(path, "optimizer.pt"))
    os.remove(os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME))
    os.remove(os.path.join(path, _optimizer_shard_filename(0)))

    target = _build_manager(tmp_path / "target")
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 8))
    with pytest.raises(RuntimeError, match="legacy pickle-backed optimizer.pt"):
        target.load_adapter_state(model_id="resume-e", path=path, load_optimizer=True)

    # The pickle-backed optimizer is also refused single-rank; weights-only
    # remains the explicit migration path.
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 1))
    with pytest.raises(RuntimeError, match="legacy pickle-backed optimizer.pt"):
        target.load_adapter_state(model_id="resume-e", path=path, load_optimizer=True)
    target.load_adapter_state(model_id="resume-e", path=path, load_optimizer=False)


def test_optimizer_shard_without_manifest_is_refused(tmp_path: Path):
    source = _build_manager(tmp_path)
    _register(source, "resume-incomplete")
    _apply_step(source, "resume-incomplete", seed=3)
    path = source.save_adapter_state("resume-incomplete")["path"]
    os.remove(os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME))

    target = _build_manager(tmp_path / "target")
    with pytest.raises(RuntimeError, match="per-rank optimizer shards but no"):
        target.load_adapter_state(model_id="resume-incomplete", path=path, load_optimizer=True)
    assert not target.has_adapter("resume-incomplete")


def test_declared_optimizer_checkpoint_without_artifacts_is_refused(tmp_path: Path):
    source = _build_manager(tmp_path)
    _register(source, "resume-missing")
    path = source.save_adapter_state("resume-missing")["path"]
    os.remove(os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME))
    os.remove(os.path.join(path, _optimizer_shard_filename(0)))

    target = _build_manager(tmp_path / "target")
    with pytest.raises(RuntimeError, match="declares saved optimizer state"):
        target.load_adapter_state(model_id="resume-missing", path=path, load_optimizer=True)
    assert not target.has_adapter("resume-missing")


def test_incomplete_optimizer_restore_does_not_mutate_resident_adapter(tmp_path: Path):
    source = _build_manager(tmp_path / "source")
    _register(source, "resume-resident")
    _fill_params(source, "resume-resident", seed=41)
    _apply_step(source, "resume-resident", seed=43)
    path = source.save_adapter_state("resume-resident")["path"]
    os.remove(os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME))
    os.remove(os.path.join(path, _optimizer_shard_filename(0)))

    target = _build_manager(tmp_path / "target")
    _register(target, "resume-resident")
    _fill_params(target, "resume-resident", seed=47)
    _apply_step(target, "resume-resident", seed=53)
    resident_state = target.get_adapter_state("resume-resident")
    resident_weights = {name: param.detach().clone() for name, param in resident_state.lora_params.items()}
    resident_moments = {name: tensor.clone() for name, tensor in _moment_tensors(target, "resume-resident").items()}
    assert any(
        not torch.equal(param, source.get_adapter_state("resume-resident").lora_params[name])
        for name, param in resident_weights.items()
    )

    with pytest.raises(RuntimeError, match="declares saved optimizer state"):
        target.load_adapter_state(model_id="resume-resident", path=path, load_optimizer=True)

    assert target.has_adapter("resume-resident")
    restored_state = target.get_adapter_state("resume-resident")
    for name, tensor in resident_weights.items():
        assert torch.equal(restored_state.lora_params[name], tensor)
    restored_moments = _moment_tensors(target, "resume-resident")
    assert restored_moments.keys() == resident_moments.keys()
    for name, tensor in resident_moments.items():
        assert torch.equal(restored_moments[name], tensor)


def test_world_size_mismatch_refused(tmp_path: Path):
    source = _build_manager(tmp_path)
    _register(source, "resume-f")
    _apply_step(source, "resume-f", seed=5)
    path = source.save_adapter_state("resume-f")["path"]
    manifest_path = os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["world_size"] = 8
    manifest["per_rank_param_structure_sha256"] = manifest["per_rank_param_structure_sha256"] * 8
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    target = _build_manager(tmp_path / "target")
    with pytest.raises(RuntimeError, match="world_size=8"):
        target.load_adapter_state(model_id="resume-f", path=path, load_optimizer=True)


def _one_dimensional_layout(name: str, *, offset: int, size: int, logical_size: int) -> AdapterTensorLayout:
    return AdapterTensorLayout(
        fqn=name,
        dtype=torch.float32,
        rank_dim=0,
        substrate_shape=(logical_size,),
        logical_shape=(logical_size,),
        local_substrate_shape=(size,),
        local_logical_offset=(offset,),
        local_logical_shape=(size,),
        active_local_slices=(slice(0, size),),
        active_storage_shape=(size,),
    )


def _two_dimensional_layout(
    name: str,
    *,
    offset: tuple[int, int],
    shape: tuple[int, int],
    logical_shape: tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> AdapterTensorLayout:
    return AdapterTensorLayout(
        fqn=name,
        dtype=dtype,
        rank_dim=0,
        substrate_shape=logical_shape,
        logical_shape=logical_shape,
        local_substrate_shape=shape,
        local_logical_offset=offset,
        local_logical_shape=shape,
        active_local_slices=tuple(slice(0, size) for size in shape),
        active_storage_shape=shape,
    )


def _write_two_rank_optimizer_checkpoint(
    path: Path,
    *,
    name: str,
    layouts: list[AdapterTensorLayout],
    moments: list[torch.Tensor],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for rank, moment in enumerate(moments):
        state = {
            "state": {
                0: {
                    "step": torch.tensor(8.0),
                    "exp_avg": moment,
                    "exp_avg_sq": moment.square(),
                }
            },
            "param_groups": [
                {
                    "params": [0],
                    "lr": 1e-2,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "decoupled_weight_decay": True,
                }
            ],
        }
        _save_optimizer_state_safetensors(state, str(path / _optimizer_shard_filename(rank)))
    manifest = {
        "format_version": 3,
        "world_size": 2,
        "per_rank_layout_descriptors": [[layout.to_json_dict()] for layout in layouts],
        "session_rank": 4,
        "optimizer_parameter_order": [name],
        "per_rank_optimizer_parameter_order": [[name], [name]],
    }
    manifest["per_rank_layout_fingerprint"] = [
        _layout_descriptor_fingerprint(descriptors, world_size=2)
        for descriptors in manifest["per_rank_layout_descriptors"]
    ]
    manifest["per_rank_param_structure_sha256"] = [
        _descriptor_structure_fingerprint(descriptors, [name])
        for descriptors in manifest["per_rank_layout_descriptors"]
    ]
    (path / OPTIMIZER_SHARD_MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")


def _target_adapter_state(name: str, layout: AdapterTensorLayout) -> AdapterState:
    parameter = torch.nn.Parameter(torch.zeros(layout.active_storage_shape, dtype=layout.dtype))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        fused=False,
    )
    return AdapterState(
        model_id="reshard",
        session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=1e-2),
        local_params={name: parameter},
        tensor_layouts={name: layout},
        layout_fingerprint="target",
        optimizer=optimizer,
    )


def test_world_size_change_reshards_adam_rectangles_exactly(tmp_path: Path, monkeypatch):
    name = "layer.lora_A"
    source_layouts = [
        _one_dimensional_layout(name, offset=0, size=2, logical_size=4),
        _one_dimensional_layout(name, offset=2, size=2, logical_size=4),
    ]
    _write_two_rank_optimizer_checkpoint(
        tmp_path,
        name=name,
        layouts=source_layouts,
        moments=[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
    )
    target = _target_adapter_state(name, _one_dimensional_layout(name, offset=0, size=4, logical_size=4))
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 1))

    assert load_adapter_optimizer_shards(target, str(tmp_path), torch.device("cpu"))

    parameter = target.local_params[name]
    assert torch.equal(target.optimizer.state[parameter]["exp_avg"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(target.optimizer.state[parameter]["exp_avg_sq"], torch.tensor([1.0, 4.0, 9.0, 16.0]))
    assert target.optimizer.state[parameter]["step"].item() == 8


def test_world_size_change_refuses_divergent_replicated_moments(tmp_path: Path, monkeypatch):
    name = "layer.lora_A"
    replicated = _one_dimensional_layout(name, offset=0, size=4, logical_size=4)
    _write_two_rank_optimizer_checkpoint(
        tmp_path,
        name=name,
        layouts=[replicated, replicated],
        moments=[torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.0, 2.0, 3.0, 5.0])],
    )
    target = _target_adapter_state(name, replicated)
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 1))

    with pytest.raises(RuntimeError, match="Replicated optimizer field layer.lora_A.exp_avg differs"):
        load_adapter_optimizer_shards(target, str(tmp_path), torch.device("cpu"))


def test_topology_change_reshards_multidimensional_disjoint_rectangles(tmp_path: Path, monkeypatch):
    name = "layer.lora_A"
    source_layouts = [
        _two_dimensional_layout(name, offset=(0, 0), shape=(2, 4), logical_shape=(4, 4)),
        _two_dimensional_layout(name, offset=(2, 0), shape=(2, 4), logical_shape=(4, 4)),
    ]
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    _write_two_rank_optimizer_checkpoint(
        tmp_path,
        name=name,
        layouts=source_layouts,
        moments=[source[:2].clone(), source[2:].clone()],
    )
    target_layout = _two_dimensional_layout(name, offset=(1, 1), shape=(2, 2), logical_shape=(4, 4))
    target = _target_adapter_state(name, target_layout)
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 1))

    assert load_adapter_optimizer_shards(target, str(tmp_path), torch.device("cpu"))

    parameter = target.local_params[name]
    torch.testing.assert_close(target.optimizer.state[parameter]["exp_avg"], source[1:3, 1:3], rtol=0, atol=0)
    torch.testing.assert_close(
        target.optimizer.state[parameter]["exp_avg_sq"],
        source[1:3, 1:3].square(),
        rtol=0,
        atol=0,
    )


def test_topology_change_slices_replicated_rectangle(tmp_path: Path, monkeypatch):
    name = "layer.lora_A"
    replicated = _two_dimensional_layout(name, offset=(0, 0), shape=(4, 4), logical_shape=(4, 4))
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    _write_two_rank_optimizer_checkpoint(
        tmp_path,
        name=name,
        layouts=[replicated, replicated],
        moments=[source.clone(), source.clone()],
    )
    target_layout = _two_dimensional_layout(name, offset=(2, 1), shape=(2, 2), logical_shape=(4, 4))
    target = _target_adapter_state(name, target_layout)
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 1))

    assert load_adapter_optimizer_shards(target, str(tmp_path), torch.device("cpu"))

    parameter = target.local_params[name]
    torch.testing.assert_close(target.optimizer.state[parameter]["exp_avg"], source[2:4, 1:3], rtol=0, atol=0)


def test_same_world_layout_change_uses_logical_reshard(tmp_path: Path, monkeypatch):
    name = "layer.lora_A"
    source_layouts = [
        _one_dimensional_layout(name, offset=0, size=2, logical_size=4),
        _one_dimensional_layout(name, offset=2, size=2, logical_size=4),
    ]
    _write_two_rank_optimizer_checkpoint(
        tmp_path,
        name=name,
        layouts=source_layouts,
        moments=[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
    )
    target = _target_adapter_state(name, _one_dimensional_layout(name, offset=0, size=4, logical_size=4))
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 2))

    assert load_adapter_optimizer_shards(target, str(tmp_path), torch.device("cpu"))

    parameter = target.local_params[name]
    torch.testing.assert_close(
        target.optimizer.state[parameter]["exp_avg"], torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("defect", ["hole", "overlap", "dtype", "logical_shape", "step"])
def test_topology_change_rejects_invalid_source_without_mutation(tmp_path: Path, monkeypatch, defect: str):
    name = "layer.lora_A"
    layouts = [
        _one_dimensional_layout(name, offset=0, size=2, logical_size=4),
        _one_dimensional_layout(name, offset=2, size=2, logical_size=4),
    ]
    moments = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    _write_two_rank_optimizer_checkpoint(tmp_path, name=name, layouts=layouts, moments=moments)
    manifest_path = tmp_path / OPTIMIZER_SHARD_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if defect == "hole":
        manifest["per_rank_layout_descriptors"][1][0]["local_logical_offset"] = [3]
    elif defect == "overlap":
        manifest["per_rank_layout_descriptors"][1][0]["local_logical_offset"] = [1]
    elif defect == "dtype":
        manifest["per_rank_layout_descriptors"][1][0]["dtype"] = "float64"
    elif defect == "logical_shape":
        manifest["per_rank_layout_descriptors"][1][0]["logical_shape"] = [5]
    else:
        state_path = tmp_path / _optimizer_shard_filename(1)
        state = adapter_manager_module._load_optimizer_state_safetensors(str(state_path), torch.device("cpu"))
        state["state"][0]["step"] = torch.tensor(9.0)
        _save_optimizer_state_safetensors(state, str(state_path))
    manifest["per_rank_layout_fingerprint"] = [
        _layout_descriptor_fingerprint(descriptors, world_size=2)
        for descriptors in manifest["per_rank_layout_descriptors"]
    ]
    manifest["per_rank_param_structure_sha256"] = [
        _descriptor_structure_fingerprint(descriptors, [name])
        for descriptors in manifest["per_rank_layout_descriptors"]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target = _target_adapter_state(name, _one_dimensional_layout(name, offset=0, size=4, logical_size=4))
    parameter = target.local_params[name]
    parameter.grad = torch.tensor([0.5, 0.25, -0.25, -0.5])
    target.optimizer.step()
    before = _clone_state_to_cpu(target.optimizer.state_dict())
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 1))

    with pytest.raises(RuntimeError):
        load_adapter_optimizer_shards(target, str(tmp_path), torch.device("cpu"))

    after = target.optimizer.state_dict()
    assert _same_optimizer_value(before, after)


def test_topology_change_rejects_rank_without_active_optimizer_parameters(tmp_path: Path):
    manifest = {
        "world_size": 1,
        "per_rank_layout_descriptors": [[]],
        "per_rank_optimizer_parameter_order": [[]],
    }
    state = AdapterState(
        model_id="empty",
        session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=1e-2),
        local_params={},
        tensor_layouts={},
        layout_fingerprint="empty",
        optimizer=None,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="coordinator ranks with no active local optimizer parameters"):
        _reshard_adapter_optimizer_state(state, str(tmp_path), manifest, rank=0, world=1)


@pytest.mark.parametrize("defect", ["group_metadata", "state_shape"])
def test_staged_optimizer_contract_is_validated_before_resident_mutation(tmp_path: Path, defect: str):
    source = _build_manager(tmp_path / "source")
    _register(source, "staged-contract")
    _apply_step(source, "staged-contract", seed=17)
    path = source.save_adapter_state("staged-contract")["path"]
    shard_path = os.path.join(path, _optimizer_shard_filename(0))
    staged = adapter_manager_module._load_optimizer_state_safetensors(shard_path, torch.device("cpu"))
    if defect == "group_metadata":
        staged["param_groups"][0].pop("eps")
    else:
        first_state = next(iter(staged["state"].values()))
        first_spatial_name = next(
            name for name, value in first_state.items() if isinstance(value, torch.Tensor) and value.ndim > 0
        )
        first_state[first_spatial_name] = first_state[first_spatial_name].reshape(-1)[:-1]
    _save_optimizer_state_safetensors(staged, shard_path)

    target = _build_manager(tmp_path / "target")
    _register(target, "staged-contract")
    _apply_step(target, "staged-contract", seed=19)
    resident = target.get_adapter_state("staged-contract")
    before = _clone_state_to_cpu(resident.optimizer.state_dict())

    with pytest.raises(RuntimeError):
        load_adapter_optimizer_shards(resident, path, torch.device("cpu"))

    assert _same_optimizer_value(before, resident.optimizer.state_dict())


def test_param_structure_mismatch_refused(tmp_path: Path):
    source = _build_manager(tmp_path)
    _register(source, "resume-g")
    _apply_step(source, "resume-g", seed=9)
    path = source.save_adapter_state("resume-g")["path"]
    manifest_path = os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["per_rank_param_structure_sha256"] = ["0" * 64]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    target = _build_manager(tmp_path / "target")
    with pytest.raises(RuntimeError, match="Parameter fingerprint does not match descriptors"):
        target.load_adapter_state(model_id="resume-g", path=path, load_optimizer=True)
