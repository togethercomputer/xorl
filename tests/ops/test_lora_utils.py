"""Tests for stacked LoRA helper utilities."""

import pytest
import torch

from xorl.ops.group_gemm.kernel.lora_utils import (
    get_lora_delta_weight_stacked,
    init_lora_weights_stacked,
    merge_lora_weights_stacked,
    unmerge_lora_weights_stacked,
)


pytestmark = [pytest.mark.cpu]


def test_stacked_lora_helpers_use_gkn_layout():
    lora_A, lora_B = init_lora_weights_stacked(
        num_experts=2,
        r=3,
        in_features=4,
        out_features=5,
    )

    assert lora_A.shape == (2, 4, 3)
    assert lora_B.shape == (2, 3, 5)
    assert torch.equal(lora_B, torch.zeros_like(lora_B))

    lora_A = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    lora_B = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    base = torch.ones(2, 4, 5, dtype=torch.float32)
    scaling = 0.25
    expected_delta = torch.bmm(lora_A, lora_B) * scaling

    delta = get_lora_delta_weight_stacked(lora_A, lora_B, scaling)
    merged = merge_lora_weights_stacked(base, lora_A, lora_B, scaling)
    unmerged = unmerge_lora_weights_stacked(merged, lora_A, lora_B, scaling)

    assert torch.equal(delta, expected_delta)
    assert torch.equal(merged, base + expected_delta)
    assert torch.equal(unmerged, base)
