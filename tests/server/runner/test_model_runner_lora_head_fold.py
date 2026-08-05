"""LM-head LoRA folding on the server loss path."""

from types import SimpleNamespace

import pytest
import torch

from xorl.lora.fold import canonical_lora_fold_linear
from xorl.lora.modules.linear import LoraLinear
from xorl.server.runner.model_runner import ModelRunner


pytestmark = pytest.mark.cpu


def test_effective_lm_head_uses_canonical_merged_weight_with_adapter_gradients():
    head = LoraLinear(5, 7, r=2, lora_alpha=4, bias=False, dtype=torch.bfloat16)
    head.exact_merged_forward = True
    with torch.no_grad():
        head.weight.copy_(torch.linspace(-1.0, 1.0, head.weight.numel()).reshape_as(head.weight))
        head.lora_A.copy_(torch.linspace(-0.25, 0.25, head.lora_A.numel()).reshape_as(head.lora_A))
        head.lora_B.copy_(torch.linspace(0.125, -0.125, head.lora_B.numel()).reshape_as(head.lora_B))

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(lm_head=head)
    effective = runner._get_effective_lm_head_weight()
    expected = canonical_lora_fold_linear(
        head.weight,
        head.lora_A,
        head.lora_B,
        head._active_scaling(),
    )

    assert torch.equal(effective.detach().view(torch.int16), expected.view(torch.int16))
    effective.float().sum().backward()
    assert head.weight.grad is None
    assert head.lora_A.grad is not None and torch.count_nonzero(head.lora_A.grad) > 0
    assert head.lora_B.grad is not None and torch.count_nonzero(head.lora_B.grad) > 0


def test_effective_lm_head_keeps_legacy_unmerged_formula_without_contract():
    head = LoraLinear(3, 4, r=2, lora_alpha=2, bias=False, dtype=torch.bfloat16)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(lm_head=head)

    expected = head.weight + head.get_delta_weight().to(head.weight.dtype)
    assert torch.equal(runner._get_effective_lm_head_weight(), expected)
