"""FSDP2 lifecycle gate for the exact GLM-5.2 TP1 dense-MLP composite."""

from __future__ import annotations

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

from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]


class _DenseParent(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.mlp = Glm52ExactTP1DenseMLP(256, 128, device=device)
        self.parent_marker = nn.Parameter(torch.ones(1, dtype=torch.float32, device=device), requires_grad=False)
        self.parent_forward_state = []

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.parent_forward_state.append((isinstance(self.parent_marker, DTensor), self.parent_marker.dtype))
        return self.mlp(input)


def _fp8_pattern(shape: tuple[int, ...], device: torch.device, offset: int) -> torch.Tensor:
    values = torch.arange(torch.tensor(shape).prod().item(), device=device, dtype=torch.int32)
    return ((values + offset) % 31 - 15).reshape(shape).float().to(torch.float8_e4m3fn)


def _load_state(model: _DenseParent, device: torch.device) -> None:
    gate_weight = _fp8_pattern((128, 256), device, 0)
    up_weight = _fp8_pattern((128, 256), device, 7)
    gate_scale = torch.arange(2, device=device, dtype=torch.float32).reshape(1, 2).add_(1).div_(16)
    up_scale = torch.arange(2, device=device, dtype=torch.float32).reshape(1, 2).add_(3).div_(16)
    model.mlp.load_gate_up_prequantized(gate_weight, gate_scale, up_weight, up_scale)

    down_weight = _fp8_pattern((256, 128), device, 11)
    down_scale = torch.arange(2, device=device, dtype=torch.float32).reshape(2, 1).add_(5).div_(16)
    model.mlp.down_proj._source_fqn = "mlp.down_proj"
    model.mlp.down_proj._load_prequantized(lambda name: down_weight if name == "mlp.down_proj.weight" else down_scale)
    with torch.no_grad():
        factors = (
            model.mlp.gate_proj.lora_A,
            model.mlp.gate_proj.lora_B,
            model.mlp.up_proj.lora_A,
            model.mlp.up_proj.lora_B,
            model.mlp.down_proj.lora_A,
            model.mlp.down_proj.lora_B,
        )
        for index, factor in enumerate(factors):
            factor.copy_(
                torch.linspace(
                    -0.5 + index / 32,
                    0.5 - index / 64,
                    factor.numel(),
                    device=device,
                    dtype=torch.float32,
                ).reshape_as(factor)
            )


def _run_fsdp2_lifecycle() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    try:
        world_size = dist.get_world_size()
        assert world_size in (1, 2)
        device = torch.device("cuda", local_rank)
        reference = _DenseParent(device)
        sharded = _DenseParent(device)
        _load_state(reference, device)
        _load_state(sharded, device)

        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        fully_shard(sharded.mlp, mesh=mesh, reshard_after_forward=True)
        fully_shard(
            sharded,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )
        assert isinstance(sharded.mlp.packed_weight_f32, DTensor)
        assert isinstance(sharded.mlp.down_proj.packed_weight_f32, DTensor)
        assert isinstance(sharded.mlp.gate_proj.lora_A, DTensor)
        assert isinstance(sharded.mlp.down_proj.lora_B, DTensor)
        assert _deregister_qlora_weights_from_fsdp(sharded, param_names=("packed_weight_f32",)) == 1

        forward_state = []

        def record_unsharded_state(module, _args) -> None:
            forward_state.append(
                {
                    "gate_A": isinstance(module.gate_proj.lora_A, DTensor),
                    "up_B": isinstance(module.up_proj.lora_B, DTensor),
                    "down_A": isinstance(module.down_proj.lora_A, DTensor),
                    "outer_packed": isinstance(module.packed_weight_f32, DTensor),
                    "down_packed": isinstance(module.down_proj.packed_weight_f32, DTensor),
                    "factor_dtypes": {
                        module.gate_proj.lora_A.dtype,
                        module.up_proj.lora_B.dtype,
                        module.down_proj.lora_A.dtype,
                    },
                }
            )

        hook = sharded.mlp.register_forward_pre_hook(record_unsharded_state)
        values = torch.linspace(-1, 1, 17 * 256, device=device, dtype=torch.bfloat16).reshape(17, 256)
        reference_input = values.detach().clone().requires_grad_(True)
        sharded_input = values.detach().clone().requires_grad_(True)
        grad_output = torch.linspace(-0.75, 0.75, 17 * 256, device=device, dtype=torch.bfloat16).reshape(17, 256)

        reference_output = reference(reference_input)
        sharded_output = sharded(sharded_input)
        assert reference.parent_forward_state == [(False, torch.float32)]
        assert sharded.parent_forward_state == [(False, torch.bfloat16)]
        assert forward_state == [
            {
                "gate_A": False,
                "up_B": False,
                "down_A": False,
                "outer_packed": False,
                "down_packed": True,
                "factor_dtypes": {torch.float32},
            }
        ]
        assert torch.equal(sharded_output.view(torch.uint8), reference_output.view(torch.uint8))

        reference_output.backward(grad_output)
        sharded_output.backward(grad_output)
        assert torch.equal(sharded_input.grad, reference_input.grad)
        reference_parameters = dict(reference.mlp.named_parameters())
        sharded_parameters = dict(sharded.mlp.named_parameters())
        for name in sharded.mlp.logical_factor_names:
            sharded_grad = sharded_parameters[name].grad
            assert isinstance(sharded_grad, DTensor)
            assert torch.equal(sharded_grad.full_tensor(), reference_parameters[name].grad), name
        assert isinstance(sharded.mlp.packed_weight_f32, DTensor)
        assert isinstance(sharded.mlp.down_proj.packed_weight_f32, DTensor)
        hook.remove()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_exact_dense_mlp_composes_with_two_rank_fsdp2_lifecycle() -> None:
        pytest.importorskip("sglang")
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success("exact dense MLP should preserve all six gradients through two-rank FSDP2")


if __name__ == "__main__":
    _run_fsdp2_lifecycle()
