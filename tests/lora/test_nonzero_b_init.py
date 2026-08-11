import copy

import pytest
import torch

from xorl.lora.modules.linear import LoraLinear
from xorl.lora.utils import initialize_lora_b_nonzero


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(LoraLinear(8, 6, r=4, lora_alpha=8), LoraLinear(6, 3, r=4, lora_alpha=8))


def test_nonzero_lora_b_init_is_opt_in_and_deterministic() -> None:
    default = _model()
    assert initialize_lora_b_nonzero(default, std=0.0, seed=17) == 0
    assert all(torch.count_nonzero(module.lora_B) == 0 for module in default if isinstance(module, LoraLinear))

    first = _model()
    second = copy.deepcopy(first)
    a_before = [module.lora_A.detach().clone() for module in first if isinstance(module, LoraLinear)]
    assert initialize_lora_b_nonzero(first, std=1e-3, seed=17) == 2
    assert initialize_lora_b_nonzero(second, std=1e-3, seed=17) == 2

    for module_first, module_second, original_a in zip(first, second, a_before, strict=True):
        assert torch.equal(module_first.lora_A, original_a)
        assert torch.equal(module_first.lora_B, module_second.lora_B)
        assert torch.count_nonzero(module_first.lora_B) > 0


def test_nonzero_lora_b_init_rejects_negative_std_and_missing_adapters() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        initialize_lora_b_nonzero(_model(), std=-1.0)
    with pytest.raises(RuntimeError, match="no lora_B"):
        initialize_lora_b_nonzero(torch.nn.Linear(2, 2), std=1e-3)
