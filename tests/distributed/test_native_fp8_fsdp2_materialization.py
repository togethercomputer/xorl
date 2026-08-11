"""FSDP2 meta-materialization regressions for frozen native-FP8 state.

The worker uses two CPU/gloo ranks so the assertions cover real DTensor local
shards.  It also nests the native module under a separately fully-sharded root,
matching the model materialization lifecycle that exposed the regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
from xorl.ops.block_fp8_native import NativeBlockFP8Linear


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _make_native_module(family: str, device: torch.device | str) -> nn.Module:
    if family == "linear":
        return NativeBlockFP8Linear(128, 128, device=device)
    if family == "experts":
        return Glm52NativeBlockFP8Experts(2, 256, 256, device=device)
    raise AssertionError(f"unknown native-FP8 family {family!r}")


def _assert_frozen_fp32_parameters(module: nn.Module, *, device: torch.device) -> None:
    parameters = dict(module.named_parameters(recurse=False))
    assert parameters
    for parameter in parameters.values():
        assert parameter.dtype is torch.float32
        assert parameter.requires_grad is False
        assert parameter.device == device


class _NestedNativeModel(nn.Module):
    def __init__(self, family: str, device: torch.device | str) -> None:
        super().__init__()
        self.native = _make_native_module(family, device)


def _run_fsdp2_materialization_regression() -> None:
    # Exercise the replacement path that originally corrupted frozen DTensor
    # parameters instead of assuming the process-wide default still selects it.
    torch.__future__.set_swap_module_params_on_conversion(False)
    dist.init_process_group(backend="gloo")
    try:
        mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
        assert mesh.size() == 2

        for family in ("linear", "experts"):
            model = _NestedNativeModel(family, "meta")
            if family == "experts":
                fully_shard(model.native, mesh=mesh, shard_placement_fn=lambda parameter: Shard(1))
            else:
                fully_shard(model.native, mesh=mesh)
            fully_shard(model, mesh=mesh)

            before = {}
            saw_strict_shard = False
            for name, parameter in model.native.named_parameters(recurse=False):
                assert isinstance(parameter, DTensor)
                before[name] = {
                    "mesh": parameter.device_mesh,
                    "placements": tuple(parameter.placements),
                    "global_shape": tuple(parameter.shape),
                    "local_shape": tuple(parameter.to_local().shape),
                }
                assert parameter.is_meta
                assert parameter.to_local().is_meta
                saw_strict_shard |= parameter.to_local().numel() < parameter.numel()
                if family == "experts":
                    assert tuple(parameter.placements) == (Shard(1),)
            assert saw_strict_shard

            model.to_empty(device=torch.device("cpu"))

            parameters = dict(model.native.named_parameters(recurse=False))
            assert set(parameters) == set(before)
            for name, parameter in parameters.items():
                expected = before[name]
                assert isinstance(parameter, DTensor)
                assert parameter.device_mesh == expected["mesh"]
                assert tuple(parameter.placements) == expected["placements"]
                assert tuple(parameter.shape) == expected["global_shape"]
                assert tuple(parameter.to_local().shape) == expected["local_shape"]
                assert parameter.is_meta is False
                assert parameter.to_local().is_meta is False
                assert parameter.to_local().device == torch.device("cpu")
                assert parameter.to_local().is_contiguous()
                assert parameter.dtype is torch.float32
                assert parameter.requires_grad is False

            state = model.state_dict()
            assert set(state) == {f"native.{name}" for name in before}
            for name, tensor in state.items():
                parameter = parameters[name.removeprefix("native.")]
                assert isinstance(tensor, DTensor)
                assert tensor.device_mesh == parameter.device_mesh
                assert tuple(tensor.placements) == tuple(parameter.placements)
                assert tuple(tensor.shape) == tuple(parameter.shape)
                assert tuple(tensor.to_local().shape) == tuple(parameter.to_local().shape)
                assert tensor.to_local().is_meta is False
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_native_fp8_plain_apply_keeps_frozen_fp32_state() -> None:
    """The DTensor fix must not change ordinary device/dtype conversion."""

    for family in ("linear", "experts"):
        module = _make_native_module(family, "meta")
        module.to_empty(device=torch.device("cpu"))
        _assert_frozen_fp32_parameters(module, device=torch.device("cpu"))
        assert all(not isinstance(parameter, DTensor) for parameter in module.parameters())
        before = {
            name: parameter.detach().view(torch.uint8).clone()
            for name, parameter in module.named_parameters(recurse=False)
        }

        module.to(device=torch.device("cpu"), dtype=torch.bfloat16)

        _assert_frozen_fp32_parameters(module, device=torch.device("cpu"))
        for name, parameter in module.named_parameters(recurse=False):
            assert torch.equal(parameter.detach().view(torch.uint8), before[name]), family


if __name__ != "__main__":

    def test_native_fp8_meta_materialization_preserves_fsdp2_dtensors() -> None:
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success("native-FP8 meta materialization must preserve FSDP2 DTensor parameters")


if __name__ == "__main__":
    _run_fsdp2_materialization_regression()
