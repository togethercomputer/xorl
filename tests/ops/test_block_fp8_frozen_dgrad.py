"""Component gates for the frozen NativeBlockFP8Linear activation dgrad.

The GLM-5.2 full-param trainable set backpropagates THROUGH frozen native
block-FP8 projections (attention q_a/q_b/kv_a/o, shared experts).  The
admitted mechanism keeps the forward VALUE program byte-identical and adds
ONLY an activation gradient: ``grad_output @ dequant(cache)`` in the
declared BF16 linear program — the same base-branch treatment the trainable
full-param composites use — with NO wgrad and NO byte mutation.

Gates:
- value bytes with gradient engagement == scoring-only value bytes (bitwise,
  full range and block-aligned partitions);
- dgrad == the reference dequant-matmul autograd (bitwise), and == the
  trainable composite's dgrad on identical bytes (the "same treatment"
  claim, cross-implementation);
- negatives: an unadmitted module still refuses with the phase-one
  scoring-only error; frozen parameters receive no gradient and no byte
  changes; engagement logs exactly once per process.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.exact.block_fp8_native import (
    NATIVE_BLOCK_FP8_FROZEN_DGRAD_CONTRACT_VERSION,
    NativeBlockFP8Linear,
)


_IN, _OUT = 256, 384


def _hopper_or_skip() -> torch.device:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    return torch.device("cuda")


def _seeded_module(device: torch.device) -> NativeBlockFP8Linear:
    module = NativeBlockFP8Linear(_IN, _OUT, device=device)
    values = torch.arange(_OUT * _IN, dtype=torch.float32, device=device)
    weight = (((values * 3 + 1) % 29) - 14).reshape(_OUT, _IN).div(16.0).to(torch.float8_e4m3fn)
    scales = (torch.arange(6, dtype=torch.float32, device=device).reshape(3, 2) + 1.0) / 7.0
    module.load_prequantized(weight, scales)
    return module


def _input(device: torch.device, rows: int = 8, width: int = _IN) -> torch.Tensor:
    values = torch.arange(rows * width, dtype=torch.float32, device=device)
    return ((values % 23) - 11).reshape(rows, width).div(8.0).to(torch.bfloat16).contiguous()


def _reference_dequant_matmul_dgrad(
    module: NativeBlockFP8Linear,
    input: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    output_range: tuple[int, int],
    input_range: tuple[int, int],
) -> torch.Tensor:
    """The reference: autograd through F.linear on the dequantized slice."""

    from xorl.ops.quantize import block_fp8_dequantize_gkn

    out_start, out_end = output_range
    in_start, in_end = input_range
    weight = module.fp8_weight()[out_start:out_end, in_start:in_end].contiguous()
    scale = module.weight_scale_inv[
        out_start // 128 : (out_end + 127) // 128,
        in_start // 128 : (in_end + 127) // 128,
    ].contiguous()
    dequantized = block_fp8_dequantize_gkn(weight, scale, 128).to(torch.float32).to(torch.bfloat16)
    with torch.enable_grad():
        reference_input = input.detach().requires_grad_(True)
        output = F.linear(reference_input, dequantized)
        (reference_grad,) = torch.autograd.grad(output, reference_input, grad_outputs=grad_output)
    return reference_grad


def test_unadmitted_module_still_refuses_with_the_phase_one_error() -> None:
    module = NativeBlockFP8Linear(128, 128)
    assert module._frozen_dgrad_admitted is False
    with pytest.raises(RuntimeError, match="scoring-only; activation backward requires a validated kernel"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16, requires_grad=True))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_value_bytes_identical_with_and_without_grad_engagement() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    module = _seeded_module(device)
    module.enable_frozen_activation_dgrad()
    cases = [
        (None, None, _IN),
        ((128, 384), None, _IN),
        ((0, 128), (128, 256), 128),
    ]
    for output_range, input_range, width in cases:
        input = _input(device, width=width)
        with torch.no_grad():
            scoring = module(input, output_range=output_range, input_range=input_range)
        engaged = module(
            input.detach().requires_grad_(True),
            output_range=output_range,
            input_range=input_range,
        )
        assert engaged.requires_grad
        assert torch.equal(engaged.detach().view(torch.uint8), scoring.view(torch.uint8)), (
            f"value bytes changed under gradient engagement for ranges {output_range}/{input_range}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_dgrad_matches_reference_and_composite_and_mutates_nothing(caplog) -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    NativeBlockFP8Linear._frozen_dgrad_engagement_logged = False
    module = _seeded_module(device)
    packed_before = module.packed_weight_f32.detach().view(torch.uint8).clone()
    scales_before = module.weight_scale_inv.detach().clone()

    with caplog.at_level(logging.INFO, logger="xorl.ops.exact.block_fp8_native"):
        module.enable_frozen_activation_dgrad()
        module.enable_frozen_activation_dgrad()  # idempotent
    engagement = [record for record in caplog.records if "frozen-trunk activation dgrad engaged" in record.message]
    assert len(engagement) == 1
    assert NATIVE_BLOCK_FP8_FROZEN_DGRAD_CONTRACT_VERSION in engagement[0].message

    # Full range, 2D and 3D activations, plus a block-aligned partition.
    for shape_rows, ranges in ((8, None), ((2, 4), None), (8, ((128, 384), (0, 128)))):
        output_range, input_range = ranges if ranges is not None else (None, None)
        width = _IN if input_range is None else input_range[1] - input_range[0]
        rows = shape_rows if isinstance(shape_rows, int) else int(torch.tensor(shape_rows).prod())
        input = _input(device, rows=rows, width=width)
        if not isinstance(shape_rows, int):
            input = input.reshape(*shape_rows, width)
        input = input.detach().requires_grad_(True)

        output = module(input, output_range=output_range, input_range=input_range)
        grad_output = (
            torch.arange(output.numel(), dtype=torch.float32, device=device)
            .remainder(13)
            .sub(6)
            .div(4.0)
            .reshape(output.shape)
            .to(torch.bfloat16)
        )
        output.backward(grad_output)

        reference = _reference_dequant_matmul_dgrad(
            module,
            input,
            grad_output,
            output_range=output_range or (0, _OUT),
            input_range=input_range or (0, _IN),
        )
        assert input.grad is not None and input.grad.dtype is torch.bfloat16
        assert torch.equal(input.grad.view(torch.uint8), reference.view(torch.uint8)), (
            f"frozen dgrad diverged from the reference dequant-matmul autograd for {ranges}"
        )

    # Same treatment as the trainable composite on identical bytes (full range).
    from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
        Glm52ExactTP1BlockFP8FullParamLinear,
    )

    composite = Glm52ExactTP1BlockFP8FullParamLinear(_IN, _OUT, device=device)
    composite.load_prequantized(module.fp8_weight(), module.weight_scale_inv.detach())
    frozen_input = _input(device).detach().requires_grad_(True)
    composite_input = frozen_input.detach().clone().requires_grad_(True)
    grad_output = _input(device, width=_OUT)
    module(frozen_input).backward(grad_output)
    composite(composite_input).backward(grad_output)
    assert composite_input.grad is not None and frozen_input.grad is not None
    assert torch.equal(
        frozen_input.grad.view(torch.uint8),
        composite_input.grad.to(torch.bfloat16).view(torch.uint8),
    ), "frozen dgrad diverged from the trainable composite's dgrad on identical bytes"

    # Frozen means frozen: no parameter gradients, no byte movement.
    assert module.packed_weight_f32.grad is None
    assert module.weight_scale_inv.grad is None
    assert not module.packed_weight_f32.requires_grad
    assert not module.weight_scale_inv.requires_grad
    assert torch.equal(module.packed_weight_f32.detach().view(torch.uint8), packed_before)
    assert torch.equal(module.weight_scale_inv.detach(), scales_before)
