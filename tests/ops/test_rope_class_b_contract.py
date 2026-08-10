"""Fail-closed checks for the Class-B RoPE input contract."""

import pytest
import torch

from xorl.ops.rope_class_b import class_b_apply_rotary_pos_emb


pytestmark = pytest.mark.cpu


def test_class_b_rejects_bf16_table_before_kernel_dispatch():
    q = torch.zeros((1, 1, 1, 4), dtype=torch.bfloat16)
    cos = torch.ones((1, 1, 4), dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)

    with pytest.raises(RuntimeError, match="requires fp32 cos/sin"):
        class_b_apply_rotary_pos_emb(q, q, cos, sin)
