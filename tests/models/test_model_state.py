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


def test_reference_state_dict_bypasses_dcp_state_dict_and_skips_nonpersistent_buffers(monkeypatch):
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(dp_mode="none"))
    monkeypatch.setattr(
        checkpointer,
        "get_model_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected DCP state_dict call")),
    )

    model_state = checkpointer.ModelState(_TinyModel())
    state_dict = model_state.reference_state_dict()

    assert "linear.weight" in state_dict
    assert "persistent_buf" in state_dict
    assert "scratch_buf" not in state_dict


def test_reference_state_dict_includes_qarl_persistent_buffers(monkeypatch):
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(dp_mode="none"))
    monkeypatch.setattr(
        checkpointer,
        "get_model_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected DCP state_dict call")),
    )

    model = _TinyQARLModel()
    model.proj(torch.randn(2, 4))
    state_dict = checkpointer.ModelState(model).reference_state_dict()

    assert "proj.weight" in state_dict
    assert "proj.qarl_input_amax" in state_dict
    assert "proj.qarl_weight_amax" in state_dict
    assert "proj.qarl_input_scale_inv" in state_dict
    assert "proj.qarl_weight_scale_inv" in state_dict
    assert "proj.qarl_forward_count" in state_dict
    assert state_dict["proj.qarl_weight_scale_inv"].shape == (2, 2)
    assert state_dict["proj.qarl_forward_count"].item() == 1


def test_checkpoint_metadata_records_qarl_persistent_buffers(tmp_path, monkeypatch):
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


def test_checkpoint_compatibility_detects_qarl_buffer_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointer.dist, "get_rank", lambda: 0)
    checkpointer._save_checkpoint_metadata(str(tmp_path), _TinyQARLModel())

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


def test_distributed_checkpointer_load_skips_missing_optimizer_state(tmp_path, monkeypatch):
    captured = {}

    class _FakeReader:
        def __init__(self, path):
            self.path = path

        def read_metadata(self):
            return SimpleNamespace(state_dict_metadata={"model.linear.weight": object()})

    def fake_dcp_load(state_dict, storage_reader, process_group=None, planner=None, no_dist=False):
        captured["state_keys"] = set(state_dict)
        captured["storage_reader"] = storage_reader
        captured["process_group"] = process_group
        captured["planner"] = planner
        captured["no_dist"] = no_dist

    monkeypatch.setattr(checkpointer, "FileSystemReader", _FakeReader)
    monkeypatch.setattr(checkpointer.dcp, "load", fake_dcp_load)

    state = {"model": _TinyModel(), "optimizer": object()}
    result = checkpointer.DistributedCheckpointer.load(str(tmp_path), state)

    assert result is state
    assert captured["state_keys"] == {"model"}
    assert isinstance(captured["storage_reader"], _FakeReader)
    assert captured["planner"] is not None
    assert captured["no_dist"] is False


def test_optimizer_state_filters_load_target_to_checkpoint_keys():
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


def test_distributed_checkpointer_load_passes_optimizer_metadata_keys(tmp_path, monkeypatch):
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


def test_multi_optimizer_load_filters_state_per_child_optimizer(monkeypatch):
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
