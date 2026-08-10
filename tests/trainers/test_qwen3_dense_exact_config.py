"""Configuration tests for the exact Qwen3-8B trainer program."""

import pytest

from xorl.models.auto import (
    _resolve_rope_modes,
    _validate_exact_qwen3_dense_model_scope,
    resolve_cross_entropy_mode,
    resolve_model_numerical_program,
)
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config


pytestmark = pytest.mark.cpu


def _config(**overrides):
    values = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "max_position_embeddings": 40960,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "use_sliding_window": False,
    }
    values.update(overrides)
    config = Qwen3Config(**values)
    config._qwen3_dense_exact_contract = True
    return config


def _program(config, **overrides):
    values = {
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
        "sparse_mla_backend": None,
    }
    values.update(overrides)
    return resolve_model_numerical_program(config, **values)


def test_qwen3_8b_resolves_shared_exact_program():
    config = _config()
    _validate_exact_qwen3_dense_model_scope(config)
    program = _program(config)

    assert program.attn_implementation == "flash_attention_4"
    assert program.lm_head_fp32
    assert program.rmsnorm_mode == "sglang_fused"
    assert not program.activation_native
    assert program.rope_native
    assert program.rope_class_b
    assert resolve_cross_entropy_mode(config, None) == "bi_fused"
    assert _resolve_rope_modes(
        config,
        rope_native=None,
        rope_class_b=None,
    ) == (True, True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("attn_implementation", "flash_attention_3"),
        ("lm_head_fp32", False),
        ("rmsnorm_mode", "native"),
        ("activation_native", True),
        ("rope_native", False),
        ("rope_class_b", False),
    ],
)
def test_qwen3_8b_rejects_numerical_opt_out(name, value):
    with pytest.raises(ValueError):
        _program(_config(), **{name: value})


def test_qwen3_8b_rejects_geometry_drift():
    with pytest.raises(ValueError, match="official model geometry"):
        _validate_exact_qwen3_dense_model_scope(_config(hidden_size=2048))


def test_qwen3_8b_accepts_transformers_v5_rope_parameters():
    config = _config()
    config.rope_theta = None
    config.rope_parameters = {"rope_theta": 1_000_000}

    _validate_exact_qwen3_dense_model_scope(config)
