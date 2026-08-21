"""Configuration tests for the exact dense Qwen3 trainer program."""

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
        "attention_dropout": 0.0,
        "rope_scaling": None,
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
        "rope_fp32_single_round": None,
        "attention_cast_bf16": False,
        "sparse_mla_enabled": None,
        "sparse_mla_backend": None,
    }
    values.update(overrides)
    return resolve_model_numerical_program(config, **values)


def test_dense_qwen3_resolves_shared_exact_program():
    config = _config()
    _validate_exact_qwen3_dense_model_scope(config)
    program = _program(config)

    assert program.attn_implementation == "flash_attention_4"
    assert program.lm_head_fp32
    assert program.rmsnorm_mode == "sglang_fused"
    assert not program.activation_native
    assert program.rope_native
    assert program.rope_fp32_single_round
    assert resolve_cross_entropy_mode(config, None) == "bi_fused"
    assert _resolve_rope_modes(
        config,
        rope_native=None,
        rope_fp32_single_round=None,
    ) == (True, True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("attn_implementation", "flash_attention_3"),
        ("lm_head_fp32", False),
        ("rmsnorm_mode", "native"),
        ("activation_native", True),
        ("rope_native", False),
        ("rope_fp32_single_round", False),
    ],
)
def test_dense_qwen3_rejects_numerical_opt_out(name, value):
    with pytest.raises(ValueError):
        _program(_config(), **{name: value})


@pytest.mark.parametrize(
    "geometry",
    [
        {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "tie_word_embeddings": True,
        },
        {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "tie_word_embeddings": True,
        },
        {
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rope_theta": 5_000_000,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": True,
        },
        {
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
        },
    ],
)
def test_dense_qwen3_accepts_family_geometries(geometry):
    _validate_exact_qwen3_dense_model_scope(_config(**geometry))


@pytest.mark.parametrize(
    "override",
    [
        {"hidden_act": "gelu"},
        {"head_dim": 64},
        {"attention_bias": True},
        {"attention_dropout": 0.1},
        {"use_sliding_window": True},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"num_key_value_heads": 7},
        {"hidden_size": 0},
    ],
)
def test_dense_qwen3_rejects_unsupported_capabilities(override):
    with pytest.raises(ValueError, match="does not support this architecture configuration"):
        _validate_exact_qwen3_dense_model_scope(_config(**override))


def test_dense_qwen3_accepts_transformers_v5_rope_parameters():
    config = _config()
    config.rope_theta = None
    config.rope_parameters = {"rope_theta": 1_000_000}

    _validate_exact_qwen3_dense_model_scope(config)
