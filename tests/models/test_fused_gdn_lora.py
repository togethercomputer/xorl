from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from xorl.lora.fold import canonical_lora_fold_linear
from xorl.lora.modules.delta_linear import LoraDeltaLinear
from xorl.lora.target_manifest import collect_lora_runtime_modules
from xorl.lora.utils import (
    inject_lora_into_model,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from xorl.ops.linear_attention.layers.gated_deltanet import GatedDeltaNet


class _SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(4, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = GatedDeltaNet(
            hidden_size=8,
            expand_v=1,
            head_dim=2,
            num_heads=2,
            num_v_heads=2,
            use_short_conv=False,
            layer_idx=0,
        )
        self.mlp = nn.Module()
        self.mlp.shared_expert = _SharedExpert()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])


def _manifest():
    return {
        "schema_version": 1,
        "target_modules": ["down_proj", "in_proj_qkvz", "out_proj"],
        "expected_modules": [
            {
                "pattern": "model.layers.*.linear_attn.in_proj_qkvz",
                "count": 1,
                "rank": 2,
            },
            {
                "pattern": "model.layers.*.linear_attn.out_proj",
                "count": 1,
                "rank": 2,
            },
        ],
        "allow_unlisted": False,
    }


def test_river_fused_gdn_geometry_and_manifest_path_filter(tmp_path):
    model = _Model()
    inject_lora_into_model(model, r=2, lora_alpha=4, target_manifest=_manifest())
    gdn = model.model.layers[0].linear_attn

    assert isinstance(gdn.in_proj_qkvz, LoraDeltaLinear)
    assert isinstance(gdn.out_proj, LoraDeltaLinear)
    assert isinstance(model.model.layers[0].mlp.shared_expert.down_proj, nn.Linear)
    assert collect_lora_runtime_modules(model) == {
        "model.layers.0.linear_attn.in_proj_qkvz": 2,
        "model.layers.0.linear_attn.out_proj": 2,
    }

    with torch.no_grad():
        gdn.in_proj_qkvz.lora_B.normal_()
        gdn.out_proj.lora_B.normal_()
    hidden = torch.randn(2, 3, 8)
    fused_delta = gdn.in_proj_qkvz(hidden)
    split_delta = gdn._fused_qkvz_lora_delta(hidden)
    assert split_delta is not None
    assert torch.equal(torch.cat(split_delta, dim=-1), fused_delta)

    output_input = torch.randn(2, 3, 4)
    expected_output = gdn.o_proj(output_input) + gdn.out_proj(output_input)
    assert torch.equal(gdn._project_output_linear(output_input), expected_output)

    save_lora_checkpoint(
        model,
        str(tmp_path),
        base_model_name="test",
        r=2,
        lora_alpha=4,
        preserve_lora_dtype=True,
    )
    weights = load_file(tmp_path / "adapter_model.safetensors")
    assert set(weights) == {
        "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight",
        "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_B.weight",
        "base_model.model.model.layers.0.linear_attn.out_proj.lora_A.weight",
        "base_model.model.model.layers.0.linear_attn.out_proj.lora_B.weight",
    }
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert sorted(config["target_modules"]) == ["in_proj_qkvz", "out_proj"]


def test_delta_linear_matches_explicit_low_rank_product():
    module = LoraDeltaLinear(8, 12, r=2, lora_alpha=4)
    with torch.no_grad():
        module.lora_B.normal_()
    inputs = torch.randn(5, 8)
    expected = F.linear(F.linear(inputs, module.lora_A), module.lora_B) * 2
    assert torch.allclose(module(inputs), expected)
    assert torch.allclose(module.get_delta_weight(), module.lora_B @ module.lora_A * 2)


def test_fused_gdn_delta_merged_forward_uses_canonical_fold_and_keeps_gradients(monkeypatch):
    monkeypatch.setenv("XORL_LORA_MERGED_FORWARD", "1")
    module = LoraDeltaLinear(8, 12, r=2, lora_alpha=4, dtype=torch.float32)
    with torch.no_grad():
        module.lora_B.normal_()
    base = torch.randn(5, 8)
    inputs = torch.randn(3, 8)

    folded = module.merged_weight_for_forward(base, output_start=2, output_end=7)
    expected = canonical_lora_fold_linear(base, module.lora_A, module.lora_B[2:7], 2.0)
    assert torch.equal(folded.detach(), expected)

    F.linear(inputs, folded).sum().backward()
    assert module.lora_A.grad is not None
    assert module.lora_B.grad is not None
    assert torch.count_nonzero(module.lora_A.grad) > 0
    assert torch.count_nonzero(module.lora_B.grad[2:7]) > 0
    assert torch.count_nonzero(module.lora_B.grad[:2]) == 0
    assert torch.count_nonzero(module.lora_B.grad[7:]) == 0


def test_fused_gdn_merged_weight_cache_is_bounded_by_current_slices(monkeypatch):
    monkeypatch.setenv("XORL_LORA_MERGED_FORWARD", "1")
    module = LoraDeltaLinear(8, 12, r=2, lora_alpha=4, dtype=torch.float32)
    first_base = torch.randn(5, 8)
    second_base = torch.randn(7, 8)

    previous_weights = []
    for _ in range(5):
        first = module._merged_weight(first_base, output_start=0, output_end=5)
        second = module._merged_weight(second_base, output_start=5, output_end=12)
        assert module._merged_weight(first_base, output_start=0, output_end=5) is first
        assert len(module._merged_weight_cache["slices"]) == 2
        previous_weights.extend((first, second))
        with torch.no_grad():
            module.lora_B.add_(1)

    current_weights = [entry["weight"] for entry in module._merged_weight_cache["slices"].values()]
    assert all(weight is not retained for weight in previous_weights[:-2] for retained in current_weights)


def test_fused_gdn_output_projection_merged_forward_matches_canonical_fold(monkeypatch):
    monkeypatch.setenv("XORL_LORA_MERGED_FORWARD", "1")
    model = _Model()
    inject_lora_into_model(model, r=2, lora_alpha=4, target_manifest=_manifest())
    gdn = model.model.layers[0].linear_attn
    with torch.no_grad():
        gdn.out_proj.lora_B.normal_()
    inputs = torch.randn(2, 3, gdn.o_proj.in_features)
    expected_weight = canonical_lora_fold_linear(
        gdn.o_proj.weight,
        gdn.out_proj.lora_A,
        gdn.out_proj.lora_B,
        2.0,
    )
    expected = F.linear(inputs, expected_weight, gdn.o_proj.bias)
    actual = gdn._project_output_linear(inputs)
    assert torch.equal(actual, expected)


def test_sharded_peft_checkpoint_loads_into_fused_gdn(tmp_path):
    source = _Model()
    inject_lora_into_model(source, r=2, lora_alpha=4, target_manifest=_manifest())
    with torch.no_grad():
        source.model.layers[0].linear_attn.in_proj_qkvz.lora_B.normal_()
        source.model.layers[0].linear_attn.out_proj.lora_B.normal_()
    expected = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    save_lora_checkpoint(
        source,
        str(tmp_path),
        base_model_name="test",
        r=2,
        lora_alpha=4,
        preserve_lora_dtype=True,
    )
    exported = load_file(tmp_path / "adapter_model.safetensors")
    items = sorted(exported.items())
    midpoint = len(items) // 2
    shard_names = (
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
    )
    save_file(dict(items[:midpoint]), tmp_path / shard_names[0])
    save_file(dict(items[midpoint:]), tmp_path / shard_names[1])
    weight_map = {
        key: shard_names[0] if index < midpoint else shard_names[1] for index, (key, _value) in enumerate(items)
    }
    (tmp_path / "adapter_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "adapter_model.safetensors").unlink()

    target = _Model()
    inject_lora_into_model(target, r=2, lora_alpha=4, target_manifest=_manifest())
    load_lora_checkpoint(target, str(tmp_path), strict=True)

    actual = {
        name: parameter.detach()
        for name, parameter in target.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name])
