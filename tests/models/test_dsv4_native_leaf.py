"""Exact DSV4 routed/shared leaf arithmetic at the BF16 transport boundary."""

import torch

from xorl.models.transformers.deepseek_v4.native_payload import (
    dsv4_join_routed_shared_partial,
)


def test_dsv4_leaf_uses_one_round_fp32_fma_and_propagates_gradients():
    """Pin FMA, not a separately rounded FP32 multiply followed by an add."""

    shared = torch.tensor([-0.01165771484375], dtype=torch.bfloat16, requires_grad=True)
    routed = torch.tensor([209920.0], dtype=torch.bfloat16, requires_grad=True)
    scale = 1.1
    scale_fp32 = torch.tensor(scale, dtype=torch.float32)

    result = dsv4_join_routed_shared_partial(
        routed,
        shared,
        routed_scaling_factor=scale,
    )

    # For this selected finite witness, the double-precision expression rounds
    # to the exact one-round FP32-FMA result. Materializing the product as FP32
    # first is the deliberately different two-round program.
    fma_fp32 = torch.tensor(
        float(shared.item()) + float(routed.item()) * float(scale_fp32.item()),
        dtype=torch.float32,
    )
    expected = fma_fp32.to(torch.bfloat16).reshape_as(result)
    separate_multiply_add = (shared.float() + routed.float() * scale_fp32).to(torch.bfloat16)

    assert result.dtype is torch.bfloat16
    assert expected.item() == 231424.0
    assert separate_multiply_add.item() == 230400.0
    assert torch.equal(result, expected)
    assert not torch.equal(result, separate_multiply_add)

    result.float().sum().backward()
    assert torch.equal(shared.grad, torch.ones_like(shared))
    assert torch.equal(routed.grad, torch.full_like(routed, scale))


def test_dsv4_leaf_production_scale_avoids_bf16_intermediate_rounding():
    shared = torch.tensor([0.5], dtype=torch.bfloat16)
    routed = torch.tensor([-121.0], dtype=torch.bfloat16)

    result = dsv4_join_routed_shared_partial(
        routed,
        shared,
        routed_scaling_factor=1.5,
    )
    retired_bf16_leaf = torch.add(shared, routed, alpha=1.5)

    assert torch.equal(result, torch.tensor([-181.0], dtype=torch.bfloat16))
    assert torch.equal(retired_bf16_leaf, torch.tensor([-182.0], dtype=torch.bfloat16))
    assert not torch.equal(result, retired_bf16_leaf)
