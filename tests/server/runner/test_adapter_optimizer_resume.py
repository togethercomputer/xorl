"""Adapter-checkpoint resume validation: optimizer moments must survive save/load.

Regression tests for the rank-0-only ``optimizer.pt`` defect: the legacy pickle
format saved only rank 0's Adam moments, and loading it on every rank of an EP
run silently assigned rank-0 moments to other ranks' expert parameters.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from xorl.server.runner.adapters import manager as adapter_manager_module
from xorl.server.runner.adapters.manager import (
    OPTIMIZER_SHARD_MANIFEST_FILENAME,
    LoRAAdapterManager,
    _adapter_param_structure_fingerprint,
    _clone_state_to_cpu,
    _optimizer_shard_filename,
)

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
    assert manifest["per_rank_layout_fingerprint"] == [manager.get_adapter_state("resume-a").layout_fingerprint]
    assert manifest["optimizer_parameter_order"] == sorted(
        name for name, param in manager.get_adapter_state("resume-a").local_params.items() if param.numel() > 0
    )
    state = manager.get_adapter_state("resume-a")
    assert manifest["per_rank_param_structure_sha256"] == [_adapter_param_structure_fingerprint(state.lora_params)]


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
    with pytest.raises(RuntimeError, match="different local parameter structure"):
        target.load_adapter_state(model_id="resume-g", path=path, load_optimizer=True)
