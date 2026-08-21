"""Fail-closed checks for the Class-B RoPE input contract."""

import pytest
import torch

from xorl.ops.exact.rope_class_b import (
    build_class_b_cos_sin,
    class_b_apply_rotary_pos_emb,
)


pytestmark = pytest.mark.cpu


def test_class_b_admission_shape_backward_and_table_layout_contract():
    q = torch.zeros((1, 1, 1, 4), dtype=torch.bfloat16)
    cos = torch.ones((1, 1, 4), dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)

    with pytest.raises(RuntimeError, match="requires fp32 cos/sin"):
        class_b_apply_rotary_pos_emb(q, q, cos, sin)

    for shape in ((2, 1, 8, 8), (3, 2, 12, 8), (1, 1, 16, 8)):
        _assert_class_b_shape_and_partial_rotary_backward(*shape)

    cos_half = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    sin_half = -cos_half
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    cos_flat, sin_flat = build_class_b_cos_sin(cos, sin)
    assert cos_flat.shape == sin_flat.shape == (3, 4)
    assert torch.equal(cos_flat, cos_half.view(3, 4))
    assert torch.equal(sin_flat, sin_half.view(3, 4))


def _assert_class_b_shape_and_partial_rotary_backward(q_heads, k_heads, head_dim, rotary_dim):
    torch.manual_seed(17)
    q = torch.randn((1, 3, q_heads, head_dim), dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((1, 3, k_heads, head_dim), dtype=torch.bfloat16, requires_grad=True)
    half = rotary_dim // 2
    angles = torch.randn((1, 3, half), dtype=torch.float32)
    cos_half, sin_half = angles.cos(), angles.sin()
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    q_out, k_out = class_b_apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    if rotary_dim < head_dim:
        assert torch.equal(q_out[..., rotary_dim:], q[..., rotary_dim:])
        assert torch.equal(k_out[..., rotary_dim:], k[..., rotary_dim:])

    q_grad = torch.randn_like(q_out)
    k_grad = torch.randn_like(k_out)
    dq, dk = torch.autograd.grad((q_out, k_out), (q, k), grad_outputs=(q_grad, k_grad))
    assert dq.shape == q.shape and dk.shape == k.shape
    assert torch.isfinite(dq.float()).all()
    assert torch.isfinite(dk.float()).all()
    if rotary_dim < head_dim:
        assert torch.equal(dq[..., rotary_dim:], q_grad[..., rotary_dim:])
        assert torch.equal(dk[..., rotary_dim:], k_grad[..., rotary_dim:])
