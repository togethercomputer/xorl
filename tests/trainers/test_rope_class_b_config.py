"""Configuration guards for the certified Class-B RoPE lane."""

import pytest
import torch
from transformers import PretrainedConfig

from xorl.models.auto import (
    ResolvedModelNumericalProgram,
    _resolve_rope_modes,
    resolve_cross_entropy_mode,
    resolve_model_numerical_program,
)
from xorl.models.layers.rope import RotaryEmbedding, rope_class_b_enabled, set_rope_class_b
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.trainers.model_builder import build_training_model


pytestmark = pytest.mark.cpu


def test_class_b_requires_serving_table_provenance():
    with pytest.raises(ValueError, match="rope_class_b=True requires rope_native=True"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            rope_class_b=True,
            rope_native=False,
        )


def test_class_b_selector_can_be_reset():
    set_rope_class_b(True)
    assert rope_class_b_enabled()
    set_rope_class_b(False)
    assert not rope_class_b_enabled()


def test_canonical_glm_resolves_native_class_b_without_environment(monkeypatch):
    monkeypatch.delenv("XORL_ROPE_CLASS_B", raising=False)
    config = Glm5Config(indexer_types=["full"])
    assert _resolve_rope_modes(config, rope_native=None, rope_class_b=None) == (True, True)

    set_rope_class_b(False)
    rotary = RotaryEmbedding(config)
    cos, sin = rotary(
        torch.zeros((1, 1, config.hidden_size), dtype=torch.bfloat16),
        torch.tensor([[0, 1]], dtype=torch.long),
    )
    assert cos.dtype is sin.dtype is torch.float32


@pytest.mark.parametrize("name", ["rope_native", "rope_class_b"])
def test_canonical_glm_rejects_explicit_class_b_opt_out(name):
    config = Glm5Config(indexer_types=["full"])
    kwargs = {"rope_native": None, "rope_class_b": None, name: False}
    with pytest.raises(ValueError, match="Canonical GLM-5.2 requires native Class-B RoPE"):
        _resolve_rope_modes(config, **kwargs)


def test_non_glm_rope_defaults_and_opt_in_are_unchanged():
    config = PretrainedConfig()
    assert _resolve_rope_modes(config, rope_native=None, rope_class_b=None) == (False, False)
    assert _resolve_rope_modes(config, rope_native=True, rope_class_b=True) == (True, True)


def test_canonical_glm_minimal_config_resolves_complete_exact_program(monkeypatch):
    for name in (
        "XORL_BATCH_INVARIANT_OPS",
        "XORL_BATCH_INVARIANT_MATMUL",
        "XORL_FAMILIES_V2",
        "XORL_MOE_BI_ROUTER",
        "SGLANG_BATCH_INVARIANT_OPS",
        "SGLANG_FAMILIES_V2",
    ):
        monkeypatch.delenv(name, raising=False)

    config = Glm5Config(indexer_types=["full"])
    program = resolve_model_numerical_program(
        config,
        attn_implementation=None,
        non_glm_attn_default="flash_attention_3",
        router_fp32=None,
        lm_head_fp32=None,
        rmsnorm_mode=None,
        activation_native=False,
        rope_native=None,
        rope_class_b=None,
        attention_cast_bf16=False,
        sparse_mla_enabled=None,
        sparse_mla_backend="auto",
    )
    assert program == ResolvedModelNumericalProgram(
        attn_implementation="flash_attention_4",
        router_fp32=True,
        lm_head_fp32=True,
        rmsnorm_mode="sglang_fused",
        activation_native=False,
        rope_native=True,
        rope_class_b=True,
        attention_cast_bf16=False,
        sparse_mla_enabled=True,
        sparse_mla_backend="flashmla",
    )
    assert resolve_cross_entropy_mode(config, None) == "bi_fused"


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("attn_implementation", "flash_attention_3"),
        ("router_fp32", False),
        ("lm_head_fp32", False),
        ("rmsnorm_mode", "native"),
        ("activation_native", True),
        ("attention_cast_bf16", True),
        ("sparse_mla_enabled", False),
        ("sparse_mla_backend", "torch"),
    ],
)
def test_canonical_glm_rejects_incompatible_numerical_override(override, value):
    config = Glm5Config(indexer_types=["full"])
    kwargs = {
        "attn_implementation": None,
        "non_glm_attn_default": "flash_attention_3",
        "router_fp32": None,
        "lm_head_fp32": None,
        "rmsnorm_mode": None,
        "activation_native": False,
        "rope_native": None,
        "rope_class_b": None,
        "attention_cast_bf16": False,
        "sparse_mla_enabled": None,
        "sparse_mla_backend": "auto",
        override: value,
    }
    with pytest.raises(ValueError, match="rejects incompatible numerical overrides"):
        resolve_model_numerical_program(config, **kwargs)


def test_canonical_glm_rejects_incompatible_ce_override():
    config = Glm5Config(indexer_types=["full"])
    with pytest.raises(ValueError, match="requires ce_mode='bi_fused'"):
        resolve_cross_entropy_mode(config, "compiled")


def test_non_glm_numerical_defaults_are_preserved():
    config = PretrainedConfig()
    program = resolve_model_numerical_program(
        config,
        attn_implementation=None,
        non_glm_attn_default="flash_attention_3",
        router_fp32=None,
        lm_head_fp32=None,
        rmsnorm_mode=None,
        activation_native=False,
        rope_native=None,
        rope_class_b=None,
        attention_cast_bf16=False,
        sparse_mla_enabled=None,
        sparse_mla_backend=None,
    )
    assert program == ResolvedModelNumericalProgram(
        attn_implementation="flash_attention_3",
        router_fp32=True,
        lm_head_fp32=True,
        rmsnorm_mode="native",
        activation_native=False,
        rope_native=False,
        rope_class_b=False,
        attention_cast_bf16=False,
        sparse_mla_enabled=False,
        sparse_mla_backend="auto",
    )
    assert resolve_cross_entropy_mode(config, None) == "compiled"
