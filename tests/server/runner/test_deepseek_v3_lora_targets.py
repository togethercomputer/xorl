import json

import pytest

from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _write_kimi_config(config_dir):
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k25",
                "text_config": {
                    "model_type": "kimi_k2",
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "vocab_size": 163840,
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 32,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "kv_lora_rank": 16,
                    "q_lora_rank": 32,
                    "qk_nope_head_dim": 16,
                    "qk_rope_head_dim": 8,
                    "v_head_dim": 16,
                    "n_routed_experts": 8,
                    "n_shared_experts": 2,
                    "n_group": 4,
                    "topk_group": 2,
                    "num_experts_per_tok": 2,
                    "first_k_dense_replace": 1,
                    "routed_scaling_factor": 1.25,
                    "norm_topk_prob": True,
                    "hidden_act": "silu",
                    "max_position_embeddings": 4096,
                    "initializer_range": 0.02,
                    "rms_norm_eps": 1e-6,
                    "use_cache": True,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "output_router_logits": True,
                    "router_aux_loss_coef": 0.001,
                    "topk_method": "noaux_tc",
                    "scoring_func": "sigmoid",
                    "rope_scaling": {"rope_type": "default", "rope_theta": 1000000.0},
                },
                "vision_config": {"hidden_size": 1024},
                "tie_word_embeddings": False,
            }
        )
    )


def _make_runner(config_path, lora_config):
    runner = object.__new__(ModelRunner)
    runner.model_config = {"config_path": str(config_path)}
    runner.lora_config = lora_config
    return runner


def test_model_runner_resolves_kimi_defaults_from_wrapper_config(tmp_path):
    config_dir = tmp_path / "kimi-k25"
    _write_kimi_config(config_dir)
    runner = _make_runner(
        config_dir,
        {
            "train_attn": True,
            "train_mlp": False,
            "train_unembed": True,
        },
    )

    assert runner._resolve_lora_target_modules() == [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "lm_head",
    ]


def test_model_runner_prefers_explicit_lora_targets(tmp_path):
    config_dir = tmp_path / "kimi-k25"
    _write_kimi_config(config_dir)
    runner = _make_runner(
        config_dir,
        {
            "lora_target_modules": ["q_b_proj", "o_proj"],
            "train_attn": False,
            "train_mlp": False,
            "train_unembed": False,
        },
    )

    assert runner._resolve_lora_target_modules() == ["q_b_proj", "o_proj"]
