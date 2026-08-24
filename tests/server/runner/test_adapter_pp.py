from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import xorl.server.runner.model_runner as model_runner_impl
from xorl.server.runner.adapters import manager as manager_impl
from xorl.server.runner.adapters.manager import LocalModelPartsView, LoRAAdapterManager
from xorl.server.runner.adapters.sharded_state import AdapterTensorLayout, discover_adapter_layouts
from xorl.server.runner.model_runner import ModelRunner
from xorl.trainers.model_builder import build_training_model


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_server_virtual_pp_rejects_before_foundation_model_construction(monkeypatch) -> None:
    monkeypatch.setattr(
        "xorl.trainers.model_builder.build_foundation_model",
        lambda **_kwargs: pytest.fail("virtual server PP must reject before model construction"),
    )

    with pytest.raises(NotImplementedError, match="checkpoint, publication, and optimizer mutation"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            server_training=True,
            pp_virtual_stages=2,
        )


def _bare_manager(*, pp_size: int, stage_group: object) -> LoRAAdapterManager:
    manager = object.__new__(LoRAAdapterManager)
    manager.model = nn.Module()
    manager._pipeline_parallel_size = pp_size
    manager._adapter_process_group = stage_group
    manager.prepare_forward = lambda _model_id: None
    return manager


def test_pp2_publication_merges_one_stage_leader_payload_without_changing_bytes(monkeypatch) -> None:
    stage_group = object()
    manager = _bare_manager(pp_size=2, stage_group=stage_group)
    stage0 = {
        "model.layers.0.self_attn.q_proj.lora_A": torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16),
    }
    # This represents an already reconstructed EP factor. Its replica sends
    # None, so the global writer receives exactly one logical tensor.
    stage1_expert = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    stage1 = {"model.layers.7.mlp.experts.gate_proj_lora_B": stage1_expert}
    monkeypatch.setattr("xorl.lora.utils.get_lora_state_dict", lambda _model: stage0)
    monkeypatch.setattr(manager_impl, "_optimizer_shard_rank_world", lambda: (0, 4))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: (0, 1))
    captured = {}

    def _gather_object(payload, object_gather_list, *, dst):
        captured["payload"] = payload
        captured["dst"] = dst
        object_gather_list[:] = [stage0, None, stage1, None]

    monkeypatch.setattr(torch.distributed, "gather_object", _gather_object)

    merged = manager.materialize_logical_state_dict("policy", destination_rank=0)

    assert captured == {"payload": stage0, "dst": 0}
    assert tuple(merged) == (*stage0, *stage1)
    for name, expected in {**stage0, **stage1}.items():
        assert merged[name].dtype is expected.dtype
        assert (
            merged[name].contiguous().view(torch.uint8).numpy().tobytes()
            == expected.contiguous().view(torch.uint8).numpy().tobytes()
        )


def test_pp_publication_rejects_duplicate_stage_fqn(monkeypatch) -> None:
    stage_group = object()
    manager = _bare_manager(pp_size=2, stage_group=stage_group)
    state = {"model.layers.0.proj.lora_A": torch.ones(1, 2)}
    monkeypatch.setattr("xorl.lora.utils.get_lora_state_dict", lambda _model: state)
    monkeypatch.setattr(manager_impl, "_optimizer_shard_rank_world", lambda: (0, 2))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: (0,))

    def _gather_object(payload, object_gather_list, *, dst):
        object_gather_list[:] = [payload, {next(iter(state)): torch.zeros(1, 2)}]

    monkeypatch.setattr(torch.distributed, "gather_object", _gather_object)
    with pytest.raises(RuntimeError, match="duplicate parameter 'model.layers.0.proj.lora_A'"):
        manager.materialize_logical_state_dict("policy", destination_rank=0)


def test_pp1_publication_keeps_direct_path(monkeypatch) -> None:
    manager = _bare_manager(pp_size=1, stage_group=object())
    state = {"lm_head.lora_B": torch.tensor([[3.0]], dtype=torch.float32)}
    monkeypatch.setattr("xorl.lora.utils.get_lora_state_dict", lambda _model: state)
    monkeypatch.setattr(manager_impl, "_optimizer_shard_rank_world", lambda: (0, 2))
    monkeypatch.setattr(
        torch.distributed,
        "gather_object",
        lambda *_args, **_kwargs: pytest.fail("PP1 must not gather stage dictionaries"),
    )
    assert manager.materialize_logical_state_dict("policy", destination_rank=0) is state


def test_live_model_publication_does_not_restore_stale_adapter_slots(monkeypatch) -> None:
    manager = _bare_manager(pp_size=1, stage_group=object())
    live_state = {"lm_head.lora_B": torch.tensor([[7.0]], dtype=torch.float32)}
    manager.prepare_forward = lambda _model_id: pytest.fail(
        "detached publication must not overwrite shared-optimizer updates"
    )
    monkeypatch.setattr("xorl.lora.utils.get_lora_state_dict", lambda _model: live_state)
    monkeypatch.setattr(manager_impl, "_optimizer_shard_rank_world", lambda: (0, 1))

    publisher = manager.make_live_model_lora_publisher()
    published = publisher.materialize_live_model_logical_state_dict(destination_rank=0)

    assert published is live_state
    assert not hasattr(publisher, "adapters")


def test_pp_load_filters_a_combined_checkpoint_to_the_local_stage() -> None:
    local_name = "model.layers.7.self_attn.q_proj.lora_A"
    other_stage_name = "model.layers.1.self_attn.q_proj.lora_A"
    layout = AdapterTensorLayout(
        fqn=local_name,
        dtype=torch.float32,
        rank_dim=0,
        substrate_shape=(1, 2),
        logical_shape=(1, 2),
        local_substrate_shape=(1, 2),
        local_logical_offset=(0, 0),
        local_logical_shape=(1, 2),
        active_local_slices=(slice(0, 1), slice(0, 2)),
        active_storage_shape=(1, 2),
    )
    state = SimpleNamespace(
        local_params={local_name: nn.Parameter(torch.zeros(1, 2))},
        tensor_layouts={local_name: layout},
    )
    checkpoint = {
        local_name: torch.tensor([[3.0, 4.0]]),
        other_stage_name: torch.tensor([[1.0, 2.0]]),
    }
    manager = _bare_manager(pp_size=2, stage_group=object())

    packed = manager._pack_logical_state_dict(state, checkpoint)

    assert set(packed) == {local_name}
    torch.testing.assert_close(packed[local_name], checkpoint[local_name])

    manager._pipeline_parallel_size = 1
    with pytest.raises(ValueError, match="unexpected=.*model.layers.1"):
        manager._pack_logical_state_dict(state, checkpoint)


def test_stage_local_layout_discovery_keeps_global_replica_rank_ids(monkeypatch) -> None:
    model = nn.Module()
    model.lora_A = nn.Parameter(torch.ones(2, 4))
    metadata = {"lora_A": {"shape": (2, 4), "dtype": torch.float32, "rank_dim": 0}}
    group = object()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 16 if group is None else 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 8 if group is None else 0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda actual: (8, 9))

    def _all_gather_object(outputs, payload, *, group):
        outputs[:] = [payload, payload]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _all_gather_object)
    layouts, _fingerprint, _memberships = discover_adapter_layouts(
        model,
        metadata,
        active_rank=2,
        process_group=group,
    )

    assert layouts["lora_A"].replica_ranks == (8, 9)
    assert layouts["lora_A"].replica_count == 2


class _VirtualStageLoRA(nn.Module):
    adapter_gradient_producer_family = "module_managed"

    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2, 3))
        self.lora_B = nn.Parameter(torch.zeros(3, 2))
        self.active_r = 2
        self.active_lora_alpha = 4

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha


class _VirtualStagePart(nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.config = SimpleNamespace()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([None] * (layer_index + 1))
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = _VirtualStageLoRA()
        self.model.layers[layer_index] = layer


def _live_virtual_stage_lora(parts) -> dict[str, torch.Tensor]:
    view = LocalModelPartsView(parts)
    return {
        name: parameter.detach().clone()
        for name, parameter in view.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def test_virtual_pp_adapter_spans_all_local_parts_for_switch_publication_and_load(
    tmp_path,
    monkeypatch,
) -> None:
    source_parts = [_VirtualStagePart(0), _VirtualStagePart(3)]
    manager = LoRAAdapterManager(
        source_parts,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "source-adapters"),
        auto_save_on_eviction=False,
        lora_config={"lora_rank": 2, "lora_alpha": 4},
        optimizer_type="sgd",
    )
    assert isinstance(manager.model, LocalModelPartsView)
    expected_names = {
        "model.layers.0.self_attn.o_proj.lora_A",
        "model.layers.0.self_attn.o_proj.lora_B",
        "model.layers.3.self_attn.o_proj.lora_A",
        "model.layers.3.self_attn.o_proj.lora_B",
    }
    assert set(manager._lora_param_names) == expected_names

    manager.register_adapter("policy", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy")
    optimizer_parameters = {id(parameter) for group in state.optimizer.param_groups for parameter in group["params"]}
    assert optimizer_parameters == {id(parameter) for parameter in state.local_params.values()}

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = source_parts[0]
    runner.model_parts = source_parts
    runner._adapter_manager = manager
    runner.rank = 0
    parallel_state = SimpleNamespace(
        sp_grad_sync_group=None,
        lm_head_tp_replica_group=None,
        lm_head_tp_group=None,
        ep_group=None,
        ep_enabled=False,
        ep_size=1,
    )
    monkeypatch.setattr(model_runner_impl, "get_parallel_state", lambda: parallel_state)
    runner._compile_registered_adapter_gradient_ownership("policy", group_memberships={})
    assert {item.fqn for item in state.gradient_ownership_plan.parameters} == expected_names

    with torch.no_grad():
        for ordinal, name in enumerate(sorted(state.local_params), start=1):
            state.local_params[name].fill_(ordinal)
    manager.switch_adapter("policy")
    switched = _live_virtual_stage_lora(source_parts)
    assert set(switched) == expected_names
    for name, parameter in state.local_params.items():
        torch.testing.assert_close(switched[name], parameter)

    published = manager.materialize_logical_state_dict("policy")
    assert set(published) == expected_names
    for name, parameter in state.local_params.items():
        torch.testing.assert_close(published[name], parameter)

    checkpoint = manager.save_adapter_state("policy", save_optimizer=False)["path"]
    target_parts = [_VirtualStagePart(0), _VirtualStagePart(3)]
    restored = LoRAAdapterManager(
        target_parts,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "target-adapters"),
        auto_save_on_eviction=False,
        lora_config={"lora_rank": 2, "lora_alpha": 4},
        optimizer_type="sgd",
    )
    restored.load_adapter_state("restored", checkpoint, load_optimizer=False, lr=0.1)
    restored.switch_adapter("restored")
    reloaded = _live_virtual_stage_lora(target_parts)
    assert set(reloaded) == expected_names
    for name in expected_names:
        torch.testing.assert_close(reloaded[name], published[name])
