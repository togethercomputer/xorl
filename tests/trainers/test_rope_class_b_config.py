"""Configuration guards for exact model numerical programs."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from xorl.models.auto import (
    ResolvedModelNumericalProgram,
    _resolve_rope_modes,
    _validate_canonical_glm52_model_scope,
    _validate_exact_qwen35_model_scope,
    _validate_exact_qwen35_moe_program,
    _validate_exact_qwen35_topology,
    resolve_cross_entropy_mode,
    resolve_model_numerical_program,
)
from xorl.models.layers.rope import RotaryEmbedding, rope_class_b_enabled, set_rope_class_b
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.trainers.model_builder import build_training_model


pytestmark = pytest.mark.cpu


def _exact_glm52_config() -> Glm5Config:
    indexer_types = ["full" if layer_idx < 3 or (layer_idx - 2) % 4 == 0 else "shared" for layer_idx in range(78)]
    return Glm5Config(
        indexer_types=indexer_types,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
        index_topk_freq=4,
    )


def _exact_qwen35_dense_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        max_position_embeddings=262144,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


def _exact_qwen35_moe_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        num_experts=256,
        head_dim=256,
        max_position_embeddings=262144,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )


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
    config = _exact_glm52_config()
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
    config = _exact_glm52_config()
    kwargs = {"rope_native": None, "rope_class_b": None, name: False}
    with pytest.raises(ValueError, match="Canonical GLM-5.2 requires native Class-B RoPE"):
        _resolve_rope_modes(config, **kwargs)


def test_non_glm_rope_defaults_and_opt_in_are_unchanged():
    config = PretrainedConfig()
    assert _resolve_rope_modes(config, rope_native=None, rope_class_b=None) == (False, False)
    assert _resolve_rope_modes(config, rope_native=True, rope_class_b=True) == (True, True)


def test_canonical_glm_resolves_complete_exact_program(monkeypatch):
    for name in (
        "XORL_BATCH_INVARIANT_OPS",
        "XORL_BATCH_INVARIANT_MATMUL",
        "XORL_FAMILIES_V2",
        "SGLANG_BATCH_INVARIANT_OPS",
        "SGLANG_FAMILIES_V2",
    ):
        monkeypatch.delenv(name, raising=False)

    config = _exact_glm52_config()
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
        qwen35_rmsnorm_family=None,
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
    config = _exact_glm52_config()
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
    config = _exact_glm52_config()
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
        qwen35_rmsnorm_family=None,
        activation_native=False,
        rope_native=False,
        rope_class_b=False,
        attention_cast_bf16=False,
        sparse_mla_enabled=False,
        sparse_mla_backend="auto",
    )
    assert resolve_cross_entropy_mode(config, None) == "compiled"


@pytest.mark.parametrize("config_factory", [_exact_qwen35_dense_config, _exact_qwen35_moe_config])
def test_exact_qwen35_resolves_the_certified_numerical_program(config_factory):
    config = config_factory()
    config._qwen35_exact_contract = True

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
        attn_implementation="flash_attention_4",
        router_fp32=True,
        lm_head_fp32=True,
        rmsnorm_mode="sglang_fused",
        qwen35_rmsnorm_family="v2",
        activation_native=True,
        rope_native=True,
        rope_class_b=True,
        attention_cast_bf16=True,
        sparse_mla_enabled=False,
        sparse_mla_backend="auto",
    )
    assert resolve_cross_entropy_mode(config, None) == "bi_fused"


@pytest.mark.parametrize("config_factory", [_exact_qwen35_dense_config, _exact_qwen35_moe_config])
def test_exact_qwen35_resolves_class_b_and_rejects_opt_out(config_factory):
    config = config_factory()
    config._qwen35_exact_contract = True

    default_modes = _resolve_rope_modes(config, rope_native=None, rope_class_b=None)
    assert default_modes == (True, True)
    with pytest.raises(ValueError, match="requires native Class-B RoPE"):
        _resolve_rope_modes(config, rope_native=None, rope_class_b=False)

    program = resolve_model_numerical_program(
        config,
        attn_implementation=None,
        non_glm_attn_default="flash_attention_3",
        router_fp32=None,
        lm_head_fp32=None,
        rmsnorm_mode=None,
        activation_native=False,
        rope_native=None,
        rope_class_b=True,
        attention_cast_bf16=False,
        sparse_mla_enabled=None,
        sparse_mla_backend=None,
    )
    assert program.rope_native is True
    assert program.rope_class_b is True


@pytest.mark.parametrize("config_factory", [_exact_qwen35_dense_config, _exact_qwen35_moe_config])
def test_exact_qwen35_rmsnorm_v2_is_the_architecture_scoped_default(config_factory):
    config = config_factory()
    config._qwen35_exact_contract = True
    kwargs = {
        "attn_implementation": None,
        "non_glm_attn_default": "flash_attention_3",
        "router_fp32": None,
        "lm_head_fp32": None,
        "rmsnorm_mode": None,
        "qwen35_rmsnorm_family": None,
        "activation_native": False,
        "rope_native": None,
        "rope_class_b": None,
        "attention_cast_bf16": False,
        "sparse_mla_enabled": None,
        "sparse_mla_backend": None,
    }

    program = resolve_model_numerical_program(config, **kwargs)

    assert program.qwen35_rmsnorm_family == "v2"
    assert program.rmsnorm_mode == "sglang_fused"


def test_non_qwen_rejects_qwen35_rmsnorm_v2():
    with pytest.raises(ValueError, match="supported only by exact Qwen3.5/3.6"):
        resolve_model_numerical_program(
            PretrainedConfig(),
            attn_implementation=None,
            non_glm_attn_default="flash_attention_3",
            router_fp32=None,
            lm_head_fp32=None,
            rmsnorm_mode=None,
            qwen35_rmsnorm_family="v2",
            activation_native=False,
            rope_native=None,
            rope_class_b=None,
            attention_cast_bf16=False,
            sparse_mla_enabled=None,
            sparse_mla_backend=None,
        )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("attn_implementation", "eager"),
        ("router_fp32", False),
        ("lm_head_fp32", False),
        ("rmsnorm_mode", "native"),
        ("rope_native", False),
        ("qwen35_rmsnorm_family", "v1"),
    ],
)
def test_exact_qwen35_rejects_incompatible_numerical_override(override, value):
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
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
        "sparse_mla_backend": None,
        override: value,
    }
    with pytest.raises(ValueError, match="Exact Qwen3.5-family server training"):
        resolve_model_numerical_program(config, **kwargs)


def test_exact_qwen35_rejects_incompatible_ce_override():
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    with pytest.raises(ValueError, match="requires ce_mode='bi_fused'"):
        resolve_cross_entropy_mode(config, "compiled")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moe_implementation": "eager", "ep_dispatch": "alltoall", "deepep_async_combine": False},
        {"moe_implementation": "triton", "ep_dispatch": "deepep", "deepep_async_combine": False},
        {"moe_implementation": "triton", "ep_dispatch": "alltoall", "deepep_async_combine": True},
    ],
)
def test_exact_qwen35_moe_rejects_noncertified_moe_program(kwargs):
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    with pytest.raises(ValueError, match="rejects incompatible MoE overrides"):
        _validate_exact_qwen35_moe_program(config, **kwargs)


def test_exact_qwen35_moe_admits_structural_defaults():
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    _validate_exact_qwen35_moe_program(
        config,
        moe_implementation=None,
        ep_dispatch="alltoall",
        deepep_async_combine=False,
    )


def _qwen35_topology(**overrides):
    fields = {
        "world_size": 16,
        "dp_size": 16,
        "dp_replicate_size": 2,
        "dp_shard_size": 8,
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 8,
        "cp_size": 1,
        "ringattn_size": 1,
        "ulysses_size": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_exact_qwen35_moe_admits_world16_ep8_topology():
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    _validate_exact_qwen35_topology(config, _qwen35_topology())


def test_exact_qwen35_moe_admits_world8_ep8_topology():
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    _validate_exact_qwen35_topology(
        config,
        _qwen35_topology(
            world_size=8,
            dp_size=8,
            dp_replicate_size=1,
            dp_shard_size=8,
        ),
    )


def test_exact_qwen35_moe_admits_world8_ep8_ulysses8_topology():
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    _validate_exact_qwen35_topology(
        config,
        _qwen35_topology(
            world_size=8,
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            cp_size=8,
            ulysses_size=8,
        ),
    )


def test_exact_qwen35_dense_admits_single_gpu_topology():
    config = _exact_qwen35_dense_config()
    config._qwen35_exact_contract = True
    _validate_exact_qwen35_topology(
        config,
        _qwen35_topology(
            world_size=1,
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            ep_size=1,
        ),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"world_size": 8},
        {"dp_size": 8},
        {"dp_replicate_size": 1, "dp_shard_size": 16},
        {"dp_replicate_size": 4, "dp_shard_size": 4},
        {"ep_size": 16},
        {"tp_size": 2},
        {"pp_size": 2},
        {"cp_size": 2},
        {"ringattn_size": 2},
        {"ulysses_size": 2},
    ],
)
def test_exact_qwen35_moe_rejects_noncertified_topology(override):
    config = _exact_qwen35_moe_config()
    config._qwen35_exact_contract = True
    with pytest.raises(ValueError, match="admitted only"):
        _validate_exact_qwen35_topology(config, _qwen35_topology(**override))


def test_exact_glm52_model_scope_accepts_only_official_geometry():
    _validate_canonical_glm52_model_scope(_exact_glm52_config())
    config = _exact_glm52_config()
    config.indexer_types[6] = "shared"
    with pytest.raises(ValueError, match="official model geometry.*indexer_types"):
        _validate_canonical_glm52_model_scope(config)


@pytest.mark.parametrize("config_factory", [_exact_qwen35_dense_config, _exact_qwen35_moe_config])
def test_exact_qwen35_model_scope_accepts_certified_snapshots(config_factory):
    config = config_factory()
    config._qwen35_exact_contract = True
    _validate_exact_qwen35_model_scope(config)


def test_exact_qwen35_model_scope_accepts_hf_outer_config_without_materialized_layer_types():
    text_config = _exact_qwen35_moe_config()
    del text_config.layer_types
    config = SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=text_config,
        _qwen35_exact_contract=True,
    )
    _validate_exact_qwen35_model_scope(config)


def test_exact_qwen35_model_scope_rejects_wrong_materialized_layer_types():
    config = _exact_qwen35_moe_config()
    config.layer_types[3] = "linear_attention"
    config._qwen35_exact_contract = True
    with pytest.raises(ValueError, match="certified only.*layer_types"):
        _validate_exact_qwen35_model_scope(config)


@pytest.mark.parametrize("config_factory", [_exact_qwen35_dense_config, _exact_qwen35_moe_config])
def test_exact_qwen35_model_scope_rejects_nearby_geometry(config_factory):
    config = config_factory()
    config.hidden_size += 1
    config._qwen35_exact_contract = True
    with pytest.raises(ValueError, match="certified only.*hidden_size"):
        _validate_exact_qwen35_model_scope(config)
