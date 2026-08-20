"""FSDP2 lifecycle gate for the exact GLM-5.2 absorbed kv_b child."""

from __future__ import annotations

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

from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]

_NUM_HEADS = 64
_QK_NOPE_HEAD_DIM = 192
_V_HEAD_DIM = 256
_KV_LORA_RANK = 512
_OUT_FEATURES = _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM)


def _child_state(module: Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA) -> dict[str, object]:
    state = {
        "lora_A": module.lora_A,
        "lora_B": module.lora_B,
        "packed_weight_f32": module.packed_weight_f32,
        "weight_scale_inv": module.weight_scale_inv,
    }
    return {
        "dtensors": {name: isinstance(parameter, DTensor) for name, parameter in state.items()},
        "dtypes": {name: parameter.dtype for name, parameter in state.items()},
    }


class _AbsorbedKvBParent(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.projection = Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(device=device)
        self.parent_marker = nn.Parameter(torch.ones(1, dtype=torch.float32, device=device), requires_grad=False)
        self.parent_forward_state: list[tuple[bool, torch.dtype]] = []
        self.between_child_calls: list[dict[str, object]] = []

    def forward(
        self,
        q_nope: torch.Tensor,
        attn_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.parent_forward_state.append((isinstance(self.parent_marker, DTensor), self.parent_marker.dtype))
        q_value = self.projection(q_nope, branch="q")
        self.between_child_calls.append(_child_state(self.projection))
        v_value = self.projection(attn_latent, branch="v")
        return q_value, v_value


def _fp8_pattern(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = torch.arange(math.prod(shape), dtype=torch.int32, device=device)
    return ((values % 31) - 15).reshape(shape).float().to(torch.float8_e4m3fn)


def _bf16_pattern(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    modulus: int,
    center: int,
    divisor: int,
) -> torch.Tensor:
    return (
        torch.arange(math.prod(shape), dtype=torch.float32, device=device)
        .reshape(shape)
        .remainder_(modulus)
        .sub_(center)
        .div_(divisor)
        .to(torch.bfloat16)
    )


def _load_state(model: _AbsorbedKvBParent, device: torch.device) -> None:
    weight = _fp8_pattern((_OUT_FEATURES, _KV_LORA_RANK), device)
    scales = (
        torch.arange(model.projection.weight_scale_inv.numel(), dtype=torch.float32, device=device)
        .remainder_(13)
        .add_(1)
        .div_(64)
        .reshape_as(model.projection.weight_scale_inv)
    )
    model.projection.load_prequantized(weight, scales)
    with torch.no_grad():
        model.projection.lora_A.copy_(
            torch.arange(model.projection.lora_A.numel(), dtype=torch.float32, device=device)
            .reshape_as(model.projection.lora_A)
            .remainder_(257)
            .sub_(128)
            .div_(1021)
        )
        model.projection.lora_B.copy_(
            torch.arange(model.projection.lora_B.numel(), dtype=torch.float32, device=device)
            .reshape_as(model.projection.lora_B)
            .remainder_(251)
            .sub_(125)
            .div_(2053)
        )


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype is expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _assert_child_is_resharded(module: Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA) -> None:
    state = _child_state(module)
    assert state["dtensors"] == {
        "lora_A": True,
        "lora_B": True,
        "packed_weight_f32": True,
        "weight_scale_inv": True,
    }
    assert state["dtypes"] == {
        "lora_A": torch.float32,
        "lora_B": torch.float32,
        "packed_weight_f32": torch.float32,
        "weight_scale_inv": torch.float32,
    }


def _run_fsdp2_lifecycle() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    try:
        world_size = dist.get_world_size()
        assert world_size in (1, 2)
        device = torch.device("cuda", local_rank)
        reference = _AbsorbedKvBParent(device)
        sharded = _AbsorbedKvBParent(device)
        _load_state(reference, device)
        _load_state(sharded, device)

        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        # This Native child keeps every frozen packed-base parameter registered.
        # Its child FSDP unit has no MP policy and must all-gather the base on
        # each q/v Module.__call__ below the BF16 parent unit.
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
        _assert_child_is_resharded(sharded.projection)
        assert isinstance(sharded.parent_marker, DTensor)
        if world_size == 2:
            assert sharded.projection.lora_B.to_local().numel() < sharded.projection.lora_B.numel()
            assert (
                sharded.projection.packed_weight_f32.to_local().numel() < sharded.projection.packed_weight_f32.numel()
            )

        prehook_state: list[dict[str, object]] = []

        def record_unsharded_child_state(
            module: Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
            _args,
        ) -> None:
            prehook_state.append(_child_state(module))

        hook = sharded.projection.register_forward_pre_hook(record_unsharded_child_state)
        batch_size, sequence_length = 1, 2
        q_backing = _bf16_pattern(
            (batch_size, sequence_length, _NUM_HEADS, _QK_NOPE_HEAD_DIM + 64),
            device,
            modulus=127,
            center=63,
            divisor=67,
        )
        q_values = q_backing[..., :_QK_NOPE_HEAD_DIM]
        expected_q_stride = (
            sequence_length * _NUM_HEADS * (_QK_NOPE_HEAD_DIM + 64),
            _NUM_HEADS * (_QK_NOPE_HEAD_DIM + 64),
            _QK_NOPE_HEAD_DIM + 64,
            1,
        )
        assert q_values.stride() == expected_q_stride
        assert not q_values.is_contiguous()
        v_values = _bf16_pattern(
            (batch_size, sequence_length, _NUM_HEADS, _KV_LORA_RANK),
            device,
            modulus=113,
            center=56,
            divisor=79,
        )
        reference_q_input = q_values.detach().requires_grad_(True)
        reference_v_input = v_values.detach().clone().requires_grad_(True)
        sharded_q_input = q_backing.detach().clone()[..., :_QK_NOPE_HEAD_DIM].requires_grad_(True)
        sharded_v_input = v_values.detach().clone().requires_grad_(True)
        assert reference_q_input.stride() == sharded_q_input.stride() == expected_q_stride
        grad_q = _bf16_pattern(
            (batch_size, sequence_length, _NUM_HEADS, _KV_LORA_RANK),
            device,
            modulus=97,
            center=48,
            divisor=71,
        )
        grad_v = _bf16_pattern(
            (batch_size, sequence_length, _NUM_HEADS, _V_HEAD_DIM),
            device,
            modulus=89,
            center=44,
            divisor=73,
        )

        reference_q, reference_v = reference(reference_q_input, reference_v_input)
        sharded_q, sharded_v = sharded(sharded_q_input, sharded_v_input)
        assert reference.parent_forward_state == [(False, torch.float32)]
        assert sharded.parent_forward_state == [(False, torch.bfloat16)]
        expected_unsharded = {
            "dtensors": {
                "lora_A": False,
                "lora_B": False,
                "packed_weight_f32": False,
                "weight_scale_inv": False,
            },
            "dtypes": {
                "lora_A": torch.float32,
                "lora_B": torch.float32,
                "packed_weight_f32": torch.float32,
                "weight_scale_inv": torch.float32,
            },
        }
        assert prehook_state == [expected_unsharded, expected_unsharded]
        assert reference.between_child_calls == [
            {
                "dtensors": {
                    "lora_A": False,
                    "lora_B": False,
                    "packed_weight_f32": False,
                    "weight_scale_inv": False,
                },
                "dtypes": expected_unsharded["dtypes"],
            }
        ]
        assert sharded.between_child_calls == [
            {
                "dtensors": {
                    "lora_A": True,
                    "lora_B": True,
                    "packed_weight_f32": True,
                    "weight_scale_inv": True,
                },
                "dtypes": expected_unsharded["dtypes"],
            }
        ]
        _assert_child_is_resharded(sharded.projection)
        assert isinstance(sharded.parent_marker, DTensor)
        _assert_same_bytes(sharded_q, reference_q)
        _assert_same_bytes(sharded_v, reference_v)

        torch.autograd.backward((reference_q, reference_v), (grad_q, grad_v))
        torch.autograd.backward((sharded_q, sharded_v), (grad_q, grad_v))
        assert reference_q_input.grad is not None and sharded_q_input.grad is not None
        assert reference_v_input.grad is not None and sharded_v_input.grad is not None
        _assert_same_bytes(sharded_q_input.grad, reference_q_input.grad)
        _assert_same_bytes(sharded_v_input.grad, reference_v_input.grad)

        assert isinstance(sharded.projection.lora_A.grad, DTensor)
        assert isinstance(sharded.projection.lora_B.grad, DTensor)
        if world_size == 2:
            assert sharded.projection.lora_B.grad.to_local().numel() < sharded.projection.lora_B.grad.numel()
        full_lora_A_grad = sharded.projection.lora_A.grad.full_tensor()
        full_lora_B_grad = sharded.projection.lora_B.grad.full_tensor()
        assert reference.projection.lora_A.grad is not None
        assert reference.projection.lora_B.grad is not None
        _assert_same_bytes(full_lora_A_grad, reference.projection.lora_A.grad)
        _assert_same_bytes(full_lora_B_grad, reference.projection.lora_B.grad)
        assert torch.isfinite(full_lora_A_grad).all() and torch.count_nonzero(full_lora_A_grad)
        assert torch.isfinite(full_lora_B_grad).all() and torch.count_nonzero(full_lora_B_grad)
        _assert_child_is_resharded(sharded.projection)
        hook.remove()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_exact_absorbed_kv_b_composes_with_two_rank_fsdp2_lifecycle() -> None:
        pytest.importorskip("sglang")
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("the qualified exact GLM-5.2 absorbed component requires Hopper")
        result = run_distributed_script(__file__, num_gpus=2, timeout=300)
        result.assert_success("exact absorbed kv_b should preserve both branch gradients through two-rank child FSDP2")


if __name__ == "__main__":
    _run_fsdp2_lifecycle()
