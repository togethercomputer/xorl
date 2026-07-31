from __future__ import annotations

import pytest
import torch.nn as nn

from xorl.lora.modules import LoraLinear
from xorl.lora.target_manifest import (
    collect_lora_runtime_modules,
    load_lora_target_manifest,
    resolve_lora_target_modules,
    validate_lora_target_manifest,
)
from xorl.lora.utils import inject_lora_into_model


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])


def _manifest(*, q_count=2, include_o=False, rank=4):
    expected = [{"pattern": "model.layers.*.self_attn.q_proj", "count": q_count, "rank": rank}]
    targets = ["q_proj"]
    if include_o:
        expected.append({"pattern": "model.layers.*.self_attn.o_proj", "count": 2, "rank": rank})
        targets.append("o_proj")
    return {
        "schema_version": 1,
        "target_modules": targets,
        "expected_modules": expected,
        "allow_unlisted": False,
    }


def test_manifest_drives_targets_and_validates_runtime_coverage():
    model = _Model()
    inject_lora_into_model(model, r=4, lora_alpha=8, target_manifest=_manifest())

    modules = collect_lora_runtime_modules(model)
    assert modules == {
        "model.layers.0.self_attn.q_proj": 4,
        "model.layers.1.self_attn.q_proj": 4,
    }
    assert isinstance(model.model.layers[0].self_attn.q_proj, LoraLinear)
    assert isinstance(model.model.layers[0].self_attn.o_proj, nn.Linear)


def test_manifest_fails_closed_on_count_mismatch():
    with pytest.raises(ValueError, match="matched 2 modules, expected 3"):
        inject_lora_into_model(_Model(), r=4, lora_alpha=8, target_manifest=_manifest(q_count=3))


def test_manifest_fails_closed_on_rank_mismatch():
    model = _Model()
    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=["q_proj"])
    with pytest.raises(ValueError, match="rank mismatch"):
        validate_lora_target_manifest(model, _manifest(rank=2))


def test_configured_targets_must_match_manifest():
    with pytest.raises(ValueError, match="do not match"):
        resolve_lora_target_modules(["q_proj", "o_proj"], _manifest())


def test_manifest_rejects_unlisted_lora_modules():
    model = _Model()
    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"])
    with pytest.raises(ValueError, match="unlisted LoRA modules"):
        validate_lora_target_manifest(model, _manifest())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("allow_unlisted", "false", "allow_unlisted must be a Boolean"),
    ],
)
def test_manifest_rejects_non_exact_scalar_types(field, value, message):
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match=message):
        load_lora_target_manifest(manifest)


@pytest.mark.parametrize(("field", "value"), [("count", True), ("rank", True)])
def test_manifest_rejects_booleans_for_integer_fields(field, value):
    manifest = _manifest()
    manifest["expected_modules"][0][field] = value
    with pytest.raises(ValueError, match=field):
        load_lora_target_manifest(manifest)
