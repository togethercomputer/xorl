from types import SimpleNamespace

import pytest
import torch
from torch.distributed._tensor import Shard

from xorl.checkpoint import checkpointer
from xorl.checkpoint.checkpointer import ModelState, _get_ep_checkpoint_mesh


pytestmark = pytest.mark.cpu


class _FakeDeviceMesh:
    def __init__(self, shape, mesh_dim_names, selections=None):
        self.shape = tuple(shape)
        self.mesh_dim_names = tuple(mesh_dim_names)
        self.ndim = len(self.shape)
        self.selections = [] if selections is None else selections

    def __getitem__(self, mesh_dim_names):
        names = (mesh_dim_names,) if isinstance(mesh_dim_names, str) else tuple(mesh_dim_names)
        self.selections.append(names)
        if names == self.mesh_dim_names:
            return self
        shape = tuple(self.shape[self.mesh_dim_names.index(name)] for name in names)
        return _FakeDeviceMesh(shape, names, self.selections)


def test_legacy_2d_ep_checkpoint_mesh_is_unchanged():
    mesh = _FakeDeviceMesh((8, 2), ("ep", "ep_fsdp"))

    selected = _get_ep_checkpoint_mesh(mesh)

    assert selected is mesh
    assert selected.shape == (8, 2)
    assert mesh.selections == [("ep", "ep_fsdp")]


def test_pp2_ep8_cp8_parent_selects_stage_local_checkpoint_mesh():
    # CP8 lives in the primary training mesh. The separate EP mesh represents
    # those same eight ranks per PP stage as EP8 x expert-FSDP1.
    mesh = _FakeDeviceMesh((2, 8, 1), ("_pp_ep", "ep", "ep_fsdp"))

    selected = _get_ep_checkpoint_mesh(mesh)

    assert selected.mesh_dim_names == ("ep", "ep_fsdp")
    assert selected.shape == (8, 1)
    assert mesh.selections == [("ep", "ep_fsdp")]


@pytest.mark.parametrize(
    "mesh_dim_names",
    [
        ("_pp_ep", "ep"),
        ("_pp_ep", "ep_fsdp"),
        ("ep", "ep", "ep_fsdp"),
    ],
)
def test_ep_checkpoint_mesh_rejects_missing_or_ambiguous_dimensions(mesh_dim_names):
    mesh = _FakeDeviceMesh((1,) * len(mesh_dim_names), mesh_dim_names)

    with pytest.raises(RuntimeError, match="exactly one 'ep' and one 'ep_fsdp'"):
        _get_ep_checkpoint_mesh(mesh)


@pytest.mark.parametrize(
    ("parent_shape", "parent_names"),
    [
        ((8, 2), ("ep", "ep_fsdp")),
        ((2, 8, 1), ("_pp_ep", "ep", "ep_fsdp")),
    ],
)
def test_model_state_restores_ep_dim_from_named_legacy_or_pp_mesh(monkeypatch, parent_shape, parent_names):
    parent_mesh = _FakeDeviceMesh(parent_shape, parent_names)
    state = ModelState.__new__(ModelState)
    state.ep_fqn2spec_info = {"experts.weight": SimpleNamespace(placement=Shard(0), ep_fsdp_mesh=parent_mesh)}
    restored = object()

    def fake_restore(tensor, mesh):
        assert tensor is original
        assert _get_ep_checkpoint_mesh(mesh).mesh_dim_names == ("ep", "ep_fsdp")
        return restored

    monkeypatch.setattr(checkpointer, "_restore_ep_dim", fake_restore)
    original = torch.ones(1)

    result = state.get_state_dict_with_ep_dim({"experts.weight": original})

    assert result["experts.weight"] is restored


def test_restore_ep_dim_uses_only_named_checkpoint_dimensions(monkeypatch):
    parent_mesh = _FakeDeviceMesh((2, 8, 1), ("_pp_ep", "ep", "ep_fsdp"))
    local_tensor = torch.ones(1)
    origin = type("DTensor", (), {"_local_tensor": local_tensor})()
    restored = object()
    call = {}

    def fake_from_local(tensor, *, device_mesh, placements):
        call.update(tensor=tensor, device_mesh=device_mesh, placements=placements)
        return restored

    monkeypatch.setattr(checkpointer, "DTensor", SimpleNamespace(from_local=fake_from_local))

    result = checkpointer._restore_ep_dim(origin, parent_mesh)

    assert result is restored
    assert call["tensor"] is local_tensor
    assert call["device_mesh"].mesh_dim_names == ("ep", "ep_fsdp")
    assert [placement.dim for placement in call["placements"]] == [0, 1]


def test_drop_ep_dim_uses_stage_local_expert_fsdp_dimension(monkeypatch):
    parent_mesh = _FakeDeviceMesh((2, 8, 1), ("_pp_ep", "ep", "ep_fsdp"))
    local_tensor = torch.ones(1)
    loaded = SimpleNamespace(_local_tensor=local_tensor, placements=(Shard(0), Shard(1)))
    dropped = object()
    call = {}

    def fake_from_local(tensor, *, device_mesh, placements):
        call.update(tensor=tensor, device_mesh=device_mesh, placements=placements)
        return dropped

    monkeypatch.setattr(checkpointer, "DTensor", SimpleNamespace(from_local=fake_from_local))

    result = checkpointer._drop_ep_dim(loaded, parent_mesh)

    assert result is dropped
    assert call["tensor"] is local_tensor
    assert call["device_mesh"].mesh_dim_names == ("ep_fsdp",)
    assert call["device_mesh"].shape == (1,)
    assert [placement.dim for placement in call["placements"]] == [1]
