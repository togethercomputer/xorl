"""FSDP2 ownership regression for the exact GLM-5.2 shared-expert root.

The shared root reads projection factors directly instead of invoking the
projection children.  Consequently, wrapping those children leaves their
FSDP pre-forward hooks dormant.  This test records that unsafe boundary
without launching the SGLang kernels, then proves that wrapping the shared
root presents all six FP32 factors fully materialized to the same boundary.
"""

from __future__ import annotations

import gc
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    GLM52_SHARED_EXPERT_HIDDEN_SIZE,
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]

_EXPECTED_FACTOR_SHAPES = {
    "gate_proj.lora_A": (1, 6144),
    "gate_proj.lora_B": (2048, 1),
    "up_proj.lora_A": (1, 6144),
    "up_proj.lora_B": (2048, 1),
    "down_proj.lora_A": (1, 2048),
    "down_proj.lora_B": (6144, 1),
}
_RANK_ONE_A_FACTORS = (
    "gate_proj.lora_A",
    "up_proj.lora_A",
    "down_proj.lora_A",
)


def _factor_state(module: Glm52ExactTP16SharedExpertBlockFP8QLoRA) -> dict[str, dict[str, object]]:
    state = {}
    for fqn in module.logical_factor_names:
        projection_name, factor_name = fqn.split(".")
        factor = getattr(getattr(module, projection_name), factor_name)
        local = factor.to_local() if isinstance(factor, DTensor) else factor
        state[fqn] = {
            "is_dtensor": isinstance(factor, DTensor),
            "dtype": factor.dtype,
            "shape": tuple(factor.shape),
            "local_shape": tuple(local.shape),
            "local_numel": local.numel(),
        }
    return state


class _SharedRootBoundaryWitness(Glm52ExactTP16SharedExpertBlockFP8QLoRA):
    """Record the real root boundary while deliberately skipping CUDA kernels."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__(device=device)
        self.forward_factor_states: list[dict[str, dict[str, object]]] = []

    def forward(self, input: torch.Tensor, contributor_ordinal: int) -> torch.Tensor:
        self._validate_ordinal(contributor_ordinal)
        self.forward_factor_states.append(_factor_state(self))
        return input.clone()


class _SharedExpertParent(nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.shared = _SharedRootBoundaryWitness(device=device)
        self.parent_marker = nn.Parameter(torch.ones(1, dtype=torch.float32, device=device), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.shared(input, contributor_ordinal=dist.get_rank())


def _wrap_parent(parent: _SharedExpertParent, mesh) -> None:
    fully_shard(
        parent,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        reshard_after_forward=True,
    )


def _assert_resharded_factor_state(module: _SharedRootBoundaryWitness) -> None:
    state = _factor_state(module)
    assert set(state) == set(_EXPECTED_FACTOR_SHAPES)
    assert all(item["is_dtensor"] for item in state.values())
    assert all(item["dtype"] is torch.float32 for item in state.values())
    assert {name: item["shape"] for name, item in state.items()} == _EXPECTED_FACTOR_SHAPES
    assert any(item["local_numel"] < math.prod(item["shape"]) for item in state.values())


def _run_two_rank_shared_root_lifecycle() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    try:
        assert dist.get_world_size() == 2
        device = torch.device("cuda", local_rank)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp_shard",))
        input_values = torch.zeros((1, GLM52_SHARED_EXPERT_HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

        # Reproduce the old policy safely: each projection is an FSDP child,
        # but the shared root reads its factor attributes without calling any
        # child.  Its direct-forward boundary therefore still sees DTensors.
        child_wrapped = _SharedExpertParent(device=device)
        fully_shard(child_wrapped.shared.gate_proj, mesh=mesh, reshard_after_forward=True)
        fully_shard(child_wrapped.shared.up_proj, mesh=mesh, reshard_after_forward=True)
        fully_shard(child_wrapped.shared.down_proj, mesh=mesh, reshard_after_forward=True)
        _wrap_parent(child_wrapped, mesh)
        _assert_resharded_factor_state(child_wrapped.shared)

        child_output = child_wrapped(input_values)
        assert torch.equal(child_output, input_values)
        assert len(child_wrapped.shared.forward_factor_states) == 1
        unsafe_state = child_wrapped.shared.forward_factor_states[0]
        assert all(item["is_dtensor"] for item in unsafe_state.values())
        assert any(item["local_numel"] < math.prod(item["shape"]) for item in unsafe_state.values())
        if dist.get_rank() == 1:
            assert all(unsafe_state[name]["local_numel"] == 0 for name in _RANK_ONE_A_FACTORS)
        _assert_resharded_factor_state(child_wrapped.shared)

        dist.barrier()
        del child_output, child_wrapped
        gc.collect()
        torch.cuda.empty_cache()

        # Correct ownership: one FSDP unit covers the shared root.  Its own
        # pre-forward unshard runs before direct factor access, while the BF16
        # outer unit cannot cast these nested FP32 master factors.
        root_wrapped = _SharedExpertParent(device=device)
        fully_shard(root_wrapped.shared, mesh=mesh, reshard_after_forward=True)
        _wrap_parent(root_wrapped, mesh)
        _assert_resharded_factor_state(root_wrapped.shared)

        root_output = root_wrapped(input_values)
        assert torch.equal(root_output, input_values)
        assert len(root_wrapped.shared.forward_factor_states) == 1
        safe_state = root_wrapped.shared.forward_factor_states[0]
        assert all(not item["is_dtensor"] for item in safe_state.values())
        assert all(item["dtype"] is torch.float32 for item in safe_state.values())
        assert {name: item["shape"] for name, item in safe_state.items()} == _EXPECTED_FACTOR_SHAPES
        assert all(item["local_shape"] == item["shape"] for item in safe_state.values())
        assert all(item["local_numel"] == math.prod(item["shape"]) for item in safe_state.values())
        _assert_resharded_factor_state(root_wrapped.shared)
        assert isinstance(root_wrapped.parent_marker, DTensor)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_exact_shared_expert_root_owns_two_rank_fsdp2_factor_lifecycle() -> None:
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success("exact shared-expert root FSDP must unshard all six FP32 factors before direct access")


if __name__ == "__main__":
    _run_two_rank_shared_root_lifecycle()
