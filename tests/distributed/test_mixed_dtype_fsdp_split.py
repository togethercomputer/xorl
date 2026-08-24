"""Real two-rank FSDP2 coverage for a composite with BF16 compute and FP32 state."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.torch_parallelize import (
    _bf16_mixed_precision_policy,
    _fully_shard_declared_mixed_dtype_unit,
)
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]


class _MixedComposite(nn.Module):
    fsdp_full_precision_parameter_names = ("A_log", "dt_bias")

    def __init__(self) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.linspace(-0.4, 0.2, 4, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.linspace(0.1, 0.4, 4, dtype=torch.float32))
        self.left = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.right = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.right(torch.nn.functional.silu(self.left(inputs)))
        state = (self.A_log.exp() + self.dt_bias).to(torch.bfloat16)
        return hidden + state


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixed = _MixedComposite()
        self.out = nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out(self.mixed(inputs))


def _build(device: torch.device) -> _Model:
    torch.manual_seed(1234)
    return _Model().to(device).train()


def _run_split() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    try:
        world_size = dist.get_world_size()
        assert world_size == 2
        device = torch.device("cuda", local_rank)
        init_parallel_state(dp_size=world_size, dp_shard_size=world_size, device_type="cuda")
        mesh = get_parallel_state().dp_shard_mesh

        reference = _build(device)
        sharded = _build(device)
        _fully_shard_declared_mixed_dtype_unit(
            sharded.mixed,
            compute_kwargs={"mesh": mesh, "mp_policy": _bf16_mixed_precision_policy()},
            full_precision_kwargs={"mesh": mesh},
        )
        fully_shard(sharded, mesh=mesh, mp_policy=_bf16_mixed_precision_policy())

        inputs = torch.arange(8, device=device, dtype=torch.bfloat16).reshape(2, 4) + local_rank
        reference_loss = reference(inputs).float().square().mean()
        sharded_loss = sharded(inputs).float().square().mean()
        reference_loss.backward()
        sharded_loss.backward()

        for parameter in reference.parameters():
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)

        reference_parameters = dict(reference.named_parameters())
        sharded_parameters = dict(sharded.named_parameters())
        assert set(reference_parameters) == set(sharded_parameters)
        for name, reference_parameter in reference_parameters.items():
            sharded_parameter = sharded_parameters[name]
            assert isinstance(sharded_parameter, DTensor), name
            assert isinstance(sharded_parameter.grad, DTensor), name
            full_gradient = sharded_parameter.grad.full_tensor()
            torch.testing.assert_close(full_gradient, reference_parameter.grad, rtol=0, atol=0)
            assert full_gradient.dtype == reference_parameter.dtype

        assert sharded_parameters["mixed.A_log"].dtype == torch.float32
        assert sharded_parameters["mixed.dt_bias"].dtype == torch.float32
        assert sharded_parameters["mixed.left.weight"].dtype == torch.bfloat16
    finally:
        dist.destroy_process_group()


@skip_if_gpu_count_less_than(2)
def test_real_two_rank_mixed_dtype_fsdp_split() -> None:
    if os.environ.get("RUN_MIXED_DTYPE_FSDP_SPLIT") == "1":
        _run_split()
        return
    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=180,
        extra_env={"RUN_MIXED_DTYPE_FSDP_SPLIT": "1"},
    )
    result.assert_success("mixed-dtype composite should form two sharded FSDP groups")
