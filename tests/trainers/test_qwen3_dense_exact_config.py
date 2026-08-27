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
        "rope_class_b": None,
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


def test_dense_qwen3_pairs_with_one_round_swiglu():
    """Exact dense Qwen3 must select serving's one-round FP32 SwiGLU.

    Serving applies fp32_silu_and_mul universally under a resolved exact
    contract (SiluAndMul.forward_exact); a two-round trainer activation
    diverges at bf16 scale in every MLP (first seen as a layer-0 MLP
    mismatch during Qwen3-8B K3 qualification).
    """
    from xorl.models.auto import _resolve_exact_one_round_swiglu

    dense = _config()
    assert _resolve_exact_one_round_swiglu(dense)

    generic = _config()
    generic._qwen3_dense_exact_contract = False
    assert not _resolve_exact_one_round_swiglu(generic)


def test_dense_qwen3_mlp_dispatches_one_round_swiglu():
    import torch
    import torch.nn.functional as F

    from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3MLP
    from xorl.ops.fused_silu_and_mul import exact_fp32_silu_and_mul

    exact_config = _config(hidden_size=8, intermediate_size=16)
    exact_config._exact_one_round_swiglu = True
    exact_mlp = Qwen3MLP(exact_config)
    assert exact_mlp._exact_one_round
    gate_up = torch.randn(2, 32, dtype=torch.bfloat16)
    # dispatch identity: the exact path routes through the one-round program
    assert torch.equal(exact_mlp._fused_act(gate_up), exact_fp32_silu_and_mul(gate_up))

    legacy_config = _config(hidden_size=8, intermediate_size=16)
    legacy_mlp = Qwen3MLP(legacy_config)
    assert not legacy_mlp._exact_one_round

    # The two programs are genuinely different byte streams on bf16 inputs;
    # this guards against either side silently collapsing into the other.
    torch.manual_seed(0)
    bf16 = torch.randn(64, 128, dtype=torch.bfloat16) * 3
    gate, up = bf16.chunk(2, dim=-1)
    one_round = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    two_round = (F.silu(gate.float()).to(torch.bfloat16) * up).to(torch.bfloat16)
    assert torch.equal(exact_fp32_silu_and_mul(bf16), one_round)
    assert not torch.equal(one_round, two_round)
