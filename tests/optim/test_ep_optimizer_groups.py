"""Contracts for EP optimizer parameter-group construction."""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor, Shard

import xorl.optim.optimizer as optimizer_module


pytestmark = pytest.mark.cpu


def test_mixed_dtensor_parameters_are_split_for_fused_optimizer(monkeypatch):
    class FakeDTensor:
        pass

    dtensor = FakeDTensor()
    local = object()
    monkeypatch.setattr(optimizer_module, "DTensor", FakeDTensor)

    groups = optimizer_module._split_dtensor_parameter_groups([local, dtensor])

    assert groups == [[dtensor], [local]]


def test_homogeneous_parameters_preserve_one_group(monkeypatch):
    class FakeDTensor:
        pass

    first = object()
    second = object()
    monkeypatch.setattr(optimizer_module, "DTensor", FakeDTensor)

    assert optimizer_module._split_dtensor_parameter_groups([first, second]) == [[first, second]]


def test_split_groups_execute_real_fused_adamw_with_dtensor(tmp_path):
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'single_rank_pg'}",
            rank=0,
            world_size=1,
        )
        initialized_here = True
    elif dist.get_world_size() != 1:
        pytest.skip("single-rank DTensor optimizer smoke test requires a single-rank process group")

    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp",))
        local = nn.Parameter(torch.ones(2, 3))
        distributed = nn.Parameter(DTensor.from_local(torch.ones(2, 3), mesh, [Shard(0)], run_check=False))
        local.grad = torch.full_like(local, 0.5)
        distributed.grad = DTensor.from_local(torch.full((2, 3), 0.5), mesh, [Shard(0)], run_check=False)

        split = optimizer_module._split_dtensor_parameter_groups([local, distributed])
        optimizer = torch.optim.AdamW(
            [{"params": parameters} for parameters in split],
            lr=0.1,
            fused=True,
            foreach=False,
        )
        optimizer.step()

        assert torch.allclose(local, torch.full_like(local, 0.899))
        assert torch.allclose(distributed.to_local(), torch.full((2, 3), 0.899))
    finally:
        if initialized_here:
            dist.destroy_process_group()
