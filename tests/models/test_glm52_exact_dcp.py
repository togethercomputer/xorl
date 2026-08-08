import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed._tensor import Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

from xorl.checkpoint import checkpointer
from xorl.models.exact_contract import GLM52_EXACT_ACTIVE_LORA_FLAGS
from xorl.models.transformers.glm5.exact_dcp import (
    Glm52ExactBaseDcpLoadProjection,
    _empty_sharded_half,
    _fuse_sharded_halves,
)
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.trainers import model_builder


pytestmark = [pytest.mark.cpu]


class _OneDenseExactModel(nn.Module):
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__()
        self.mlp = Glm52ExactTP1DenseMLP(128, 128, device=device)


class _FullMismatchFamilyModel(nn.Module):
    """Small tensors with the production 3-dense/315-scale module inventory."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.ModuleList([Glm52ExactTP1DenseMLP(128, 128, device="meta") for _ in range(3)])
        # The 78 attention layers have four ordinary exact QLoRA projections;
        # kv_b is the separate absorbed native module. Dense down adds three.
        self.attention = nn.ModuleList(
            [Glm52ExactTP1BlockFP8QLoRALinear(128, 128, device=torch.device("meta")) for _ in range(4 * 78)]
        )


def _persistent_buffer_keys(model: nn.Module) -> list[str]:
    modules = dict(model.named_modules())
    keys = []
    for name, buffer in model.named_buffers():
        module_name, _, buffer_name = name.rpartition(".")
        if buffer_name not in modules[module_name]._non_persistent_buffers_set:
            keys.append(name)
    return keys


def test_exact_base_dcp_contract_exhausts_three_dense_and_315_scale_aliases(tmp_path, monkeypatch) -> None:
    model = _FullMismatchFamilyModel()
    projection = Glm52ExactBaseDcpLoadProjection(model)
    parameter_keys = [name for name, _ in model.named_parameters()]
    buffer_keys = _persistent_buffer_keys(model)

    projected_parameters, projected_buffers = projection.project_key_contract(parameter_keys, buffer_keys)

    assert len(projection.dense_roots) == 3
    assert len(projection.scale_roots) == 315
    assert len(projected_parameters) == len(parameter_keys) + 321
    assert len(buffer_keys) == 315
    assert projected_buffers == []
    for root in projection.dense_roots:
        assert f"{root}.packed_weight_f32" not in projected_parameters
        assert f"{root}.weight_scale_inv" not in projected_parameters
        for member in ("gate_proj", "up_proj"):
            assert f"{root}.{member}.packed_weight_f32" in projected_parameters
            assert f"{root}.{member}.weight_scale_inv" in projected_parameters

    base_parameters = [name for name in projected_parameters if "lora_" not in name]
    metadata = {
        "has_lora": False,
        "save_lora_only": False,
        "parameter_keys": base_parameters,
        "buffer_keys": projected_buffers,
    }
    (tmp_path / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(pp_enabled=False))

    result = checkpointer._validate_checkpoint_compatibility(str(tmp_path), model, strict=True)

    assert result["glm52_exact_base_dcp_projection"] is True
    assert result["load_mode"] == "base_to_lora"
    assert len(result["missing_lora_keys"]) == 642
    assert result["missing_non_lora_keys"] == []
    assert result["unexpected_in_checkpoint"] == []
    assert result["missing_buffers_in_checkpoint"] == []
    assert result["unexpected_buffers_in_checkpoint"] == []


def test_distributed_checkpointer_loads_official_base_dcp_keys_into_exact_runtime_state(tmp_path, monkeypatch) -> None:
    source_model = _OneDenseExactModel()
    source_projection = Glm52ExactBaseDcpLoadProjection(source_model)
    source_state = {
        name: tensor.clone()
        for name, tensor in source_projection.project_state(source_model.state_dict()).items()
        if "lora_" not in name
    }
    source_state["mlp.gate_proj.packed_weight_f32"].fill_(1)
    source_state["mlp.up_proj.packed_weight_f32"].fill_(2)
    source_state["mlp.gate_proj.weight_scale_inv"].fill_(3)
    source_state["mlp.up_proj.weight_scale_inv"].fill_(4)
    source_state["mlp.down_proj.weight_scale_inv"].fill_(5)
    source_state["mlp.down_proj.packed_weight_f32"].fill_(6)
    dcp.save({"model": source_state}, checkpoint_id=str(tmp_path))
    metadata = {
        "has_lora": False,
        "save_lora_only": False,
        "parameter_keys": sorted(source_state),
        "buffer_keys": [],
    }
    (tmp_path / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    target = _OneDenseExactModel()
    initial_lora = target.mlp.down_proj.lora_A.detach().clone()
    monkeypatch.setenv("XORL_DCP_LOAD_NO_DIST", "1")
    monkeypatch.setattr(
        checkpointer,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_mode="none", pp_enabled=False),
    )

    checkpointer.DistributedCheckpointer.load(str(tmp_path), {"model": target})

    assert torch.all(target.mlp.packed_weight_f32[:128] == 1)
    assert torch.all(target.mlp.packed_weight_f32[128:] == 2)
    assert torch.equal(target.mlp.weight_scale_inv, torch.tensor([[3.0], [4.0]]))
    assert torch.all(target.mlp.down_proj.weight_block_scales.view(torch.float32) == 5)
    assert torch.all(target.mlp.down_proj.packed_weight_f32 == 6)
    assert torch.equal(target.mlp.down_proj.lora_A, initial_lora)
    assert target.mlp._exact_gate_up_base_loaded is True
    assert target.mlp.down_proj._inline_loaded is True


def _run_sharded_dense_restore(rank: int, world_size: int, rendezvous_path: str) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{rendezvous_path}",
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        target = distribute_tensor(torch.zeros(8, 4), mesh, [Shard(0)])
        gate = _empty_sharded_half(target, name="mlp.packed_weight_f32")
        up = _empty_sharded_half(target, name="mlp.packed_weight_f32")
        gate_full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        up_full = torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)
        gate.to_local().copy_(gate_full.chunk(world_size, dim=0)[rank])
        up.to_local().copy_(up_full.chunk(world_size, dim=0)[rank])

        _fuse_sharded_halves(gate, up, target, name="mlp.packed_weight_f32")

        assert torch.equal(target.full_tensor(), torch.cat((gate_full, up_full), dim=0))
    finally:
        dist.destroy_process_group()


def test_exact_base_dcp_dense_staging_fuses_into_four_rank_shard() -> None:
    handle, rendezvous_path = tempfile.mkstemp(prefix="xorl-exact-dcp-")
    os.close(handle)
    mp.start_processes(
        _run_sharded_dense_restore,
        args=(4, rendezvous_path),
        nprocs=4,
        join=True,
        start_method="fork",
    )


def test_skip_mode_defers_base_loading_to_dcp_but_keeps_fsdp_deregistration(monkeypatch) -> None:
    model = nn.Module()
    model.config = SimpleNamespace(**dict.fromkeys(GLM52_EXACT_ACTIVE_LORA_FLAGS, True))
    calls = []
    monkeypatch.setattr(
        model_builder,
        "_deregister_qlora_weights_from_fsdp",
        lambda candidate, param_names: calls.append((candidate, param_names)) or 7,
    )
    monkeypatch.setattr(
        model_builder,
        "maybe_load_prequantized_qlora",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected HF shard read")),
    )
    monkeypatch.setattr(model_builder.torch.cuda, "empty_cache", lambda: None)

    model_builder._deferred_qlora_quantize(model, "/dcp-only", load_weights_mode="skip")

    assert calls == [(model, ("packed_weight_f32",))]


def test_skip_mode_rejects_non_exact_qlora_model() -> None:
    model = nn.Module()
    model.config = SimpleNamespace()

    with pytest.raises(ValueError, match="complete GLM-5.2 exact active-LoRA model"):
        model_builder._deferred_qlora_quantize(model, "/dcp-only", load_weights_mode="skip")
