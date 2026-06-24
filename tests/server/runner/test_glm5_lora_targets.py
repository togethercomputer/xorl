import json

import pytest

from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _write_glm_config(config_dir):
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
                "vocab_size": 151552,
                "hidden_size": 4096,
                "intermediate_size": 10944,
                "moe_intermediate_size": 1408,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 192,
                "qk_rope_head_dim": 64,
                "v_head_dim": 256,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "index_head_dim": 128,
                "index_n_heads": 64,
                "index_topk": 2048,
            }
        )
    )


def _make_runner(config_path, lora_config):
    runner = object.__new__(ModelRunner)
    runner.model_config = {"config_path": str(config_path)}
    runner.lora_config = lora_config
    return runner


def test_model_runner_resolves_glm_defaults_from_raw_hf_config(tmp_path):
    config_dir = tmp_path / "glm-5"
    _write_glm_config(config_dir)
    runner = _make_runner(
        config_dir,
        {
            "train_attn": True,
            "train_mlp": False,
            "train_unembed": False,
        },
    )

    assert runner._resolve_lora_target_modules() == [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    ]


def test_model_runner_prefers_explicit_lora_targets(tmp_path):
    config_dir = tmp_path / "glm-5"
    _write_glm_config(config_dir)
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
