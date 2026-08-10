"""FSDP2 lifecycle gate for the GLM-5.2 exact TP1 QLoRA wrapper."""

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

from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]


class _ExactTP1Parent(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.projection = Glm52ExactTP1BlockFP8QLoRALinear(256, 576, device=device)
        self.parent_marker = nn.Parameter(torch.ones(1, dtype=torch.float32, device=device), requires_grad=False)
        self.parent_forward_state = []

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.parent_forward_state.append(
            (
                isinstance(self.parent_marker, DTensor),
                self.parent_marker.dtype,
            )
        )
        return self.projection(input)


def _load_state(model: _ExactTP1Parent, device: torch.device) -> None:
    weight_values = torch.arange(576 * 256, device=device, dtype=torch.int32)
    weight = ((weight_values % 31) - 15).reshape(576, 256).float().to(torch.float8_e4m3fn)
    scales = torch.arange(10, device=device, dtype=torch.float32).reshape(5, 2).add_(1).div_(16)
    model.projection._source_fqn = "projection"
    model.projection._load_prequantized(lambda name: weight if name == "projection.weight" else scales)
    with torch.no_grad():
        model.projection.lora_A.copy_(torch.linspace(-0.5, 0.5, 256, device=device).unsqueeze(0))
        model.projection.lora_B.copy_(torch.linspace(-0.25, 0.25, 576, device=device).unsqueeze(1))


def _run_fsdp2_lifecycle() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    try:
        world_size = dist.get_world_size()
        assert world_size in (1, 2)
        device = torch.device("cuda", local_rank)
        reference = _ExactTP1Parent(device)
        sharded = _ExactTP1Parent(device)
        _load_state(reference, device)
        _load_state(sharded, device)

        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        # Match production ownership: exact modules are child FSDP units with
        # no MP policy, nested below a BF16 decoder FSDP unit. The parent
        # policy must not cast the wrapper's FP32 master factors.
        fully_shard(sharded.projection, mesh=mesh, reshard_after_forward=True)
        fully_shard(
            sharded,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
            reshard_after_forward=True,
        )
        assert isinstance(sharded.projection.lora_A, DTensor)
        assert isinstance(sharded.projection.packed_weight_f32, DTensor)
        assert isinstance(sharded.parent_marker, DTensor)
        if world_size == 2:
            assert sharded.projection.lora_B.to_local().numel() < sharded.projection.lora_B.numel()
        assert _deregister_qlora_weights_from_fsdp(sharded, param_names=("packed_weight_f32",)) == 1

        forward_state = []

        def record_unsharded_factor_state(module, _args) -> None:
            forward_state.append(
                (
                    isinstance(module.lora_A, DTensor),
                    isinstance(module.lora_B, DTensor),
                    isinstance(module.packed_weight_f32, DTensor),
                    module.lora_A.dtype,
                    module.lora_B.dtype,
                )
            )

        hook = sharded.projection.register_forward_pre_hook(record_unsharded_factor_state)
        input_values = torch.linspace(-1, 1, 17 * 256, device=device, dtype=torch.bfloat16).reshape(17, 256)
        reference_input = input_values.detach().clone().requires_grad_(True)
        sharded_input = input_values.detach().clone().requires_grad_(True)
        grad_output = torch.linspace(-0.75, 0.75, 17 * 576, device=device, dtype=torch.bfloat16).reshape(17, 576)

        reference_output = reference(reference_input)
        sharded_output = sharded(sharded_input)
        assert reference.parent_forward_state == [(False, torch.float32)]
        assert sharded.parent_forward_state == [(False, torch.bfloat16)]
        assert forward_state == [(False, False, True, torch.float32, torch.float32)]
        assert torch.equal(sharded_output.view(torch.uint8), reference_output.view(torch.uint8))
        assert isinstance(sharded.projection.lora_A, DTensor)
        assert isinstance(sharded.projection.packed_weight_f32, DTensor)
        assert isinstance(sharded.parent_marker, DTensor)

        reference_output.backward(grad_output)
        sharded_output.backward(grad_output)

        assert torch.equal(sharded_input.grad, reference_input.grad)
        assert isinstance(sharded.projection.lora_A.grad, DTensor)
        assert isinstance(sharded.projection.lora_B.grad, DTensor)
        if world_size == 2:
            assert sharded.projection.lora_B.grad.to_local().numel() < sharded.projection.lora_B.grad.numel()
        full_lora_A_grad = sharded.projection.lora_A.grad.full_tensor()
        full_lora_B_grad = sharded.projection.lora_B.grad.full_tensor()
        assert torch.equal(full_lora_A_grad, reference.projection.lora_A.grad)
        assert torch.equal(full_lora_B_grad, reference.projection.lora_B.grad)
        assert torch.isfinite(sharded_input.grad).all()
        hook.remove()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(1)
    def test_exact_tp1_qlora_composes_with_production_fsdp2_lifecycle() -> None:
        pytest.importorskip("sglang")
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
        result = run_distributed_script(__file__, num_gpus=1, timeout=180)
        result.assert_success("exact TP1 QLoRA should survive FSDP2 reshard and packed-state deregistration")

    @skip_if_gpu_count_less_than(2)
    def test_exact_tp1_qlora_composes_with_two_rank_fsdp2_lifecycle() -> None:
        pytest.importorskip("sglang")
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success(
            "exact TP1 QLoRA should preserve full factor gradients through two-rank FSDP2 collectives"
        )


if __name__ == "__main__":
    _run_fsdp2_lifecycle()
