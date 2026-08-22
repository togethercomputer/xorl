"""Value-head factors must stay out of sampler adapters but survive training saves."""

import json
import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

from xorl.lora.modules.linear import LoraLinear
from xorl.lora.utils import save_lora_checkpoint


pytestmark = pytest.mark.cpu


class _TinyLoraModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.q_proj = LoraLinear(hidden, hidden, r=2, lora_alpha=2, bias=False)
        self.value_head = LoraLinear(hidden, 1, r=2, lora_alpha=2, bias=False)
        with torch.no_grad():
            self.value_head.weight.zero_()
        self.value_head.weight.requires_grad_(False)
        self.config = None


def _saved_keys(save_path):
    tensors = load_file(os.path.join(save_path, "adapter_model.safetensors"))
    return set(tensors.keys())


def test_sampler_export_drops_value_head(tmp_path):
    model = _TinyLoraModel()
    save_path = str(tmp_path / "sampler_adapter")
    save_lora_checkpoint(model, save_path, base_model_name="tiny", exclude_value_head=True)

    keys = _saved_keys(save_path)
    assert any("q_proj" in key for key in keys)
    assert not any("value_head" in key for key in keys)

    with open(os.path.join(save_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    assert "value_head" not in adapter_config["target_modules"]


def test_training_save_keeps_value_head(tmp_path):
    model = _TinyLoraModel()
    save_path = str(tmp_path / "training_adapter")
    save_lora_checkpoint(model, save_path, base_model_name="tiny", preserve_lora_dtype=True)

    keys = _saved_keys(save_path)
    assert any("value_head" in key and "lora_A" in key for key in keys)
    assert any("value_head" in key and "lora_B" in key for key in keys)
