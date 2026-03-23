from types import SimpleNamespace

from xorl.models.registry import get_registry
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig


def test_qwen3_5_conditional_generation_registered():
    registry = get_registry()
    assert "Qwen3_5ForConditionalGeneration" in registry.supported_models
    assert "Qwen3_5MoeForConditionalGeneration" in registry.supported_models


def test_qwen3_5_moe_config_from_hf_config():
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
    }
    text_config = SimpleNamespace(
        vocab_size=248320,
        hidden_size=2048,
        intermediate_size=2048,
        shared_expert_intermediate_size=512,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=["linear_attention", "full_attention"],
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
        rope_parameters=rope_parameters,
    )
    hf_config = SimpleNamespace(
        text_config=text_config,
        tie_word_embeddings=False,
    )

    config = Qwen3_5MoeConfig.from_hf_config(hf_config)

    assert config.layer_types == ["linear_attention", "full_attention"]
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == 32
    assert config.linear_key_head_dim == 128
    assert config.linear_value_head_dim == 128
    assert config.mrope_interleaved is True
    assert config.num_experts == 256


def test_qwen3_5_config_from_hf_config():
    text_config = SimpleNamespace(
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000},
    )
    hf_config = SimpleNamespace(text_config=text_config, tie_word_embeddings=False)

    config = Qwen3_5Config.from_hf_config(hf_config)

    assert config.vocab_size == 248320
    assert config.hidden_size == 4096
    assert config.head_dim == 256
