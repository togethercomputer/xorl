import json
from types import SimpleNamespace

import pytest
import torch

from xorl.checkpoint import checkpointer
from xorl.optim import multi_optimizer
from xorl.qarl import QARLLinear


pytestmark = [pytest.mark.cpu]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)
        self.register_buffer("persistent_buf", torch.ones(3))
        self.register_buffer("scratch_buf", torch.zeros(2), persistent=False)


class _TinyQARLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = QARLLinear(4, 3, weight_block_size=(2, 2))


class _TinyPlainLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)


def _assert_checkpoint_compatibility_detects_qarl_buffer_mismatch(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True)
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=False))
    monkeypatch.setattr(checkpointer.dist, "get_rank", lambda: 0)
    checkpointer._save_checkpoint_metadata(str(tmp_path), _TinyQARLModel())

    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parameter_keys"] == ["proj.bias", "proj.weight"]
    assert metadata["num_buffers"] == 5
    assert metadata["buffer_keys"] == [
        "proj.qarl_forward_count",
        "proj.qarl_input_amax",
        "proj.qarl_input_scale_inv",
        "proj.qarl_weight_amax",
        "proj.qarl_weight_scale_inv",
    ]

    with pytest.raises(RuntimeError, match="Unexpected buffers"):
        checkpointer._validate_checkpoint_compatibility(str(tmp_path), _TinyPlainLinearModel(), strict=True)

    result = checkpointer._validate_checkpoint_compatibility(str(tmp_path), _TinyPlainLinearModel(), strict=False)

    assert result["compatible"] is False
    assert result["missing_in_checkpoint"] == []
    assert result["unexpected_in_checkpoint"] == []
    assert set(result["unexpected_buffers_in_checkpoint"]) == {
        "proj.qarl_forward_count",
        "proj.qarl_input_amax",
        "proj.qarl_input_scale_inv",
        "proj.qarl_weight_amax",
        "proj.qarl_weight_scale_inv",
    }


def _assert_checkpoint_model_key_and_compatibility_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=False))
    monkeypatch.setattr(
        checkpointer.dist,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected collective")),
    )

    parameter_keys, buffer_keys, pipeline_key_union = checkpointer._get_checkpoint_model_keys(_TinyModel())

    assert parameter_keys == ["linear.weight"]
    assert buffer_keys == ["persistent_buf"]
    assert pipeline_key_union is False

    _assert_checkpoint_key_contract_unions_pipeline_stage_keys(monkeypatch)
    _assert_checkpoint_compatibility_detects_qarl_buffer_mismatch(tmp_path / "qarl", monkeypatch)
    _assert_checkpoint_metadata_uses_pipeline_stage_key_union(tmp_path / "metadata", monkeypatch)
    _assert_pipeline_lora_checkpoint_compatibility_policy(tmp_path, monkeypatch)


def _assert_checkpoint_key_contract_unions_pipeline_stage_keys(monkeypatch):
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=True))
    monkeypatch.setattr(checkpointer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpointer.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather_object(output, local_keys, group=None):
        output[:] = [
            local_keys,
            (["layers.1.weight", "norm.weight"], ["layers.1.cache"]),
        ]

    monkeypatch.setattr(checkpointer.dist, "all_gather_object", fake_all_gather_object)

    parameter_keys, buffer_keys, pipeline_key_union = checkpointer._get_checkpoint_model_keys(
        _TinyModel(), process_group=object()
    )

    assert parameter_keys == ["layers.1.weight", "linear.weight", "norm.weight"]
    assert buffer_keys == ["layers.1.cache", "persistent_buf"]
    assert pipeline_key_union is True


def _assert_checkpoint_metadata_uses_pipeline_stage_key_union(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True)
    monkeypatch.setattr(checkpointer.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        checkpointer,
        "_get_checkpoint_model_keys",
        lambda model, process_group=None: (
            ["embed.weight", "layers.1.weight", "norm.weight"],
            ["layers.1.cache"],
            True,
        ),
    )

    checkpointer._save_checkpoint_metadata(str(tmp_path), _TinyModel(), process_group=object())

    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert metadata["pipeline_parallel_key_union"] is True
    assert metadata["num_parameters"] == 3
    assert metadata["parameter_keys"] == ["embed.weight", "layers.1.weight", "norm.weight"]
    assert metadata["num_buffers"] == 1
    assert metadata["buffer_keys"] == ["layers.1.cache"]

    result = checkpointer._validate_checkpoint_compatibility(
        str(tmp_path), _TinyModel(), strict=True, process_group=object()
    )

    assert result["compatible"] is True
    assert result["pipeline_parallel_key_union"] is True
    assert result["model_parameter_count"] == 3
    assert result["model_buffer_count"] == 1


def _assert_pipeline_lora_checkpoint_compatibility_policy(tmp_path, monkeypatch):
    metadata = {
        "parameter_keys": ["stage_0.weight", "stage_1.weight"],
        "buffer_keys": [],
    }
    (tmp_path / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        checkpointer,
        "_get_checkpoint_model_keys",
        lambda model, process_group=None: (
            ["stage_0.lora_A", "stage_0.weight", "stage_1.lora_A", "stage_1.weight"],
            [],
            True,
        ),
    )

    result = checkpointer._validate_checkpoint_compatibility(
        str(tmp_path), _TinyModel(), strict=True, process_group=object()
    )

    assert result["compatible"] is True
    assert result["load_mode"] == "base_to_lora"
    assert set(result["missing_lora_keys"]) == {"stage_0.lora_A", "stage_1.lora_A"}

    _assert_pipeline_lora_only_checkpoint_compatibility(tmp_path / "lora-only", monkeypatch)


def _assert_pipeline_lora_only_checkpoint_compatibility(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True)
    metadata = {
        "parameter_keys": ["stage_0.lora_A", "stage_1.lora_A"],
        "buffer_keys": [],
        "save_lora_only": True,
    }
    (tmp_path / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        checkpointer,
        "_get_checkpoint_model_keys",
        lambda model, process_group=None: (
            ["stage_0.lora_A", "stage_0.weight", "stage_1.lora_A", "stage_1.weight"],
            [],
            True,
        ),
    )

    result = checkpointer._validate_checkpoint_compatibility(
        str(tmp_path), _TinyModel(), strict=True, process_group=object()
    )

    assert result["compatible"] is True
    assert result["load_mode"] == "lora_only"
    assert set(result["missing_non_lora_keys"]) == {"stage_0.weight", "stage_1.weight"}


def test_distributed_checkpointer_io_policy(tmp_path, monkeypatch):
    with monkeypatch.context() as case_patch:
        _assert_distributed_checkpointer_process_group_selection_policy(case_patch)
    compatibility_root = tmp_path / "compatibility"
    compatibility_root.mkdir()
    with monkeypatch.context() as case_patch:
        _assert_checkpoint_model_key_and_compatibility_policy(compatibility_root, case_patch)
    optimizer_root = tmp_path / "optimizer"
    optimizer_root.mkdir()
    with monkeypatch.context() as case_patch:
        _assert_optimizer_state_checkpoint_filtering_policy(optimizer_root, case_patch)
    with monkeypatch.context() as case_patch:
        _assert_distributed_checkpointer_metadata_admission_policy(tmp_path / "metadata", case_patch)
    with monkeypatch.context() as case_patch:
        _assert_pipeline_load_uses_custom_dcp_group(tmp_path / "load", case_patch)
    with monkeypatch.context() as case_patch:
        _assert_distributed_checkpointer_save_group_policy(tmp_path / "save", case_patch)


def _assert_distributed_checkpointer_process_group_selection_policy(monkeypatch):
    created = []
    fake_group = object()

    monkeypatch.setattr(
        checkpointer,
        "dist",
        SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_backend=lambda: "nccl",
            new_group=lambda backend: created.append(backend) or fake_group,
        ),
    )
    monkeypatch.setattr(checkpointer.DistributedCheckpointer, "_sync_process_group", None)

    assert checkpointer.DistributedCheckpointer._get_sync_process_group() is fake_group
    assert checkpointer.DistributedCheckpointer._get_sync_process_group() is fake_group
    assert created == ["gloo"]

    monkeypatch.setattr(
        checkpointer,
        "dist",
        SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_backend=lambda: "gloo",
        ),
    )
    checkpointer.DistributedCheckpointer._sync_process_group = None
    assert checkpointer.DistributedCheckpointer._get_sync_process_group() is None

    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=False))
    monkeypatch.setattr(
        checkpointer.DistributedCheckpointer,
        "_get_sync_process_group",
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("unexpected process group"))),
    )
    assert checkpointer.DistributedCheckpointer._get_metadata_process_group() is None
    assert checkpointer.DistributedCheckpointer._get_metadata_process_group(object()) is None

    custom_group = object()
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=True))
    monkeypatch.setattr(
        checkpointer.DistributedCheckpointer,
        "_get_sync_process_group",
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("unexpected global process group"))),
    )
    assert checkpointer.DistributedCheckpointer._get_metadata_process_group(custom_group) is custom_group

    metadata_group = object()
    monkeypatch.setattr(
        checkpointer.DistributedCheckpointer,
        "_get_sync_process_group",
        classmethod(lambda cls: metadata_group),
    )
    assert checkpointer.DistributedCheckpointer._get_metadata_process_group() is metadata_group


def _assert_distributed_checkpointer_metadata_admission_policy(tmp_path, monkeypatch):
    source = _TinyModel()
    source.linear.weight.data.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    checkpointer.dcp.save({"model": source.state_dict()}, checkpoint_id=str(tmp_path))

    target = _TinyModel()
    optimizer = torch.optim.Adam(target.parameters())
    monkeypatch.setenv("XORL_DCP_LOAD_NO_DIST", "1")
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=False))

    state = {"model": target, "optimizer": optimizer}
    result = checkpointer.DistributedCheckpointer.load(str(tmp_path), state)

    assert result is state
    assert torch.equal(target.linear.weight, source.linear.weight)
    assert optimizer.state == {}


def _assert_pipeline_load_uses_custom_dcp_group(tmp_path, monkeypatch):
    captured = {}
    custom_group = object()

    class _FakeReader:
        def __init__(self, path):
            self.path = path

        def read_metadata(self):
            return SimpleNamespace(state_dict_metadata={"model.linear.weight": object()})

    def fake_validate(checkpoint_dir, model, strict=True, process_group=None):
        captured["validation_process_group"] = process_group
        return {"validated": False, "reason": "test"}

    def fake_dcp_load(state_dict, storage_reader, process_group=None, planner=None, no_dist=False):
        captured["dcp_process_group"] = process_group

    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=True))
    monkeypatch.setattr(
        checkpointer.DistributedCheckpointer,
        "_get_sync_process_group",
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("unexpected global process group"))),
    )
    monkeypatch.setattr(checkpointer, "_validate_checkpoint_compatibility", fake_validate)
    monkeypatch.setattr(checkpointer, "FileSystemReader", _FakeReader)
    monkeypatch.setattr(checkpointer.dcp, "load", fake_dcp_load)

    checkpointer.DistributedCheckpointer.load(
        str(tmp_path),
        {"model": _TinyModel()},
        process_group=custom_group,
    )

    assert captured["validation_process_group"] is custom_group
    assert captured["dcp_process_group"] is custom_group


def _assert_distributed_checkpointer_save_group_policy(tmp_path, monkeypatch):
    captured = {}
    sync_group = object()

    monkeypatch.setattr(
        checkpointer,
        "get_parallel_state",
        lambda: SimpleNamespace(pp_enabled=True, dp_mode="none"),
    )
    monkeypatch.setattr(
        checkpointer.DistributedCheckpointer,
        "_get_sync_process_group",
        classmethod(lambda cls: sync_group),
    )
    monkeypatch.setattr(checkpointer, "FileSystemWriter", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        checkpointer.dcp,
        "save",
        lambda state_dict, storage_writer, process_group=None: captured.update(dcp_process_group=process_group),
    )
    monkeypatch.setattr(
        checkpointer,
        "_save_checkpoint_metadata",
        lambda checkpoint_dir, model, has_lora=False, save_lora_only=False, process_group=None: captured.update(
            metadata_process_group=process_group
        ),
    )
    monkeypatch.setattr(checkpointer.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpointer.torch.cuda, "synchronize", lambda: None)

    checkpointer.DistributedCheckpointer.save(str(tmp_path), {"model": _TinyModel()})

    assert captured["dcp_process_group"] is sync_group
    assert captured["metadata_process_group"] is sync_group

    _assert_async_save_non_pipeline_avoids_second_gloo(tmp_path, monkeypatch)


def _assert_async_save_non_pipeline_avoids_second_gloo(tmp_path, monkeypatch):
    captured = {}
    async_group = object()

    monkeypatch.setattr(
        checkpointer,
        "get_parallel_state",
        lambda: SimpleNamespace(pp_enabled=False, dp_mode="none"),
    )
    monkeypatch.setattr(checkpointer.dist, "new_group", lambda backend: async_group)
    monkeypatch.setattr(
        checkpointer.DistributedCheckpointer,
        "_get_sync_process_group",
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("unexpected second Gloo group"))),
    )
    monkeypatch.setattr(checkpointer, "FileSystemWriter", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        checkpointer.dcp,
        "async_save",
        lambda state_dict, storage_writer, process_group=None: (
            captured.update(dcp_process_group=process_group) or object()
        ),
    )
    monkeypatch.setattr(
        checkpointer,
        "_save_checkpoint_metadata",
        lambda checkpoint_dir, model, has_lora=False, save_lora_only=False, process_group=None: captured.update(
            metadata_process_group=process_group
        ),
    )
    monkeypatch.setattr(checkpointer.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpointer.torch.cuda, "synchronize", lambda: None)
    checkpointer.DistributedCheckpointer._async_process_group = None
    checkpointer.DistributedCheckpointer.dcp_save_future = None

    checkpointer.DistributedCheckpointer.save(
        str(tmp_path),
        {"model": _TinyModel()},
        save_async=True,
    )

    assert captured["dcp_process_group"] is async_group
    assert captured["metadata_process_group"] is None


def _assert_optimizer_state_checkpoint_filtering_policy(tmp_path, monkeypatch):
    class _FakeMultiOptimizer:
        _is_multi_optimizer = True

        def state_dict(self):
            return {
                "state.model.layers.0.mlp.gate.weight.step": object(),
                "state.model.layers.0.mlp.experts.gate_up_proj.step": object(),
                "param_groups.0.lr": object(),
            }

        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state_dict = state_dict
            self.loaded_strict = strict

    optimizer = _FakeMultiOptimizer()
    optimizer_state = checkpointer.OptimizerState(
        _TinyModel(),
        optimizer,
        load_keys={
            "state.model.layers.0.mlp.experts.gate_up_proj.step",
            "param_groups.0.lr",
        },
    )

    state_dict = optimizer_state.state_dict()

    assert set(state_dict) == {
        "state.model.layers.0.mlp.experts.gate_up_proj.step",
        "param_groups.0.lr",
    }
    optimizer_state.load_state_dict(state_dict)
    assert optimizer.loaded_state_dict is state_dict
    assert optimizer.loaded_strict is False

    with monkeypatch.context() as case_patch:
        _assert_distributed_load_passes_optimizer_metadata_keys(tmp_path / "metadata", case_patch)
    with monkeypatch.context() as case_patch:
        _assert_multi_optimizer_load_filters_state_per_child(case_patch)


def _assert_distributed_load_passes_optimizer_metadata_keys(tmp_path, monkeypatch):
    captured = {}

    class _FakeReader:
        def __init__(self, path):
            self.path = path

        def read_metadata(self):
            return SimpleNamespace(
                state_dict_metadata={
                    "model.linear.weight": object(),
                    "optimizer.state.model.layers.0.mlp.experts.gate_up_proj.step": object(),
                    "optimizer.param_groups.0.lr": object(),
                }
            )

    def fake_dcp_load(state_dict, storage_reader, process_group=None, planner=None, no_dist=False):
        captured["optimizer"] = state_dict["optimizer"]

    monkeypatch.setattr(checkpointer, "FileSystemReader", _FakeReader)
    monkeypatch.setattr(checkpointer.dcp, "load", fake_dcp_load)

    state = {"model": _TinyModel(), "optimizer": object()}
    result = checkpointer.DistributedCheckpointer.load(str(tmp_path), state)

    assert result is state
    assert captured["optimizer"].load_keys == {
        "state.model.layers.0.mlp.experts.gate_up_proj.step",
        "param_groups.0.lr",
    }


def _assert_multi_optimizer_load_filters_state_per_child(monkeypatch):
    ep_optimizer = object()
    non_ep_optimizer = object()
    calls = []

    def fake_get_optimizer_state_dict(model, optimizer, options):
        if optimizer is ep_optimizer:
            return {
                "state.model.layers.0.mlp.experts.gate_up_proj.step": object(),
                "state.model.layers.0.mlp.experts.gate_up_proj.exp_avg": object(),
                "state.model.layers.0.mlp.gate.weight.step": object(),
            }
        if optimizer is non_ep_optimizer:
            return {
                "state.model.embed_tokens.weight.step": object(),
                "state.model.layers.0.self_attn.q_proj.weight.step": object(),
            }
        raise AssertionError(f"unexpected optimizer: {optimizer}")

    def fake_set_optimizer_state_dict(model, optimizers, optim_state_dict, options):
        calls.append((optimizers, set(optim_state_dict), options.strict, optim_state_dict))

    monkeypatch.setattr(multi_optimizer, "get_optimizer_state_dict", fake_get_optimizer_state_dict)
    monkeypatch.setattr(multi_optimizer, "set_optimizer_state_dict", fake_set_optimizer_state_dict)

    optimizer = multi_optimizer.MultiOptimizer(
        _TinyModel(),
        {"ep": ep_optimizer, "non_ep": non_ep_optimizer},
        key_names=["ep", "non_ep"],
    )
    optimizer.load_state_dict(
        {
            "state.model.layers.0.mlp.experts.gate_up_proj.step": object(),
            "state.model.embed_tokens.weight.step": object(),
            "state.unrelated.weight.step": object(),
        },
        strict=False,
    )

    assert len(calls) == 2
    assert calls[0][:3] == (
        ep_optimizer,
        {
            "state.model.layers.0.mlp.experts.gate_up_proj.step",
            "state.model.layers.0.mlp.experts.gate_up_proj.exp_avg",
            "state.model.layers.0.mlp.gate.weight.step",
        },
        False,
    )
    assert calls[1][:3] == (
        non_ep_optimizer,
        {
            "state.model.embed_tokens.weight.step",
            "state.model.layers.0.self_attn.q_proj.weight.step",
        },
        False,
    )
