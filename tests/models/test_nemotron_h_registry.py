import json

import pytest

from xorl.models.auto import _load_local_xorl_config
from xorl.models.registry import get_registry
from xorl.models.transformers.nemotron_h.configuration_nemotron_h import NemotronHConfig
from xorl.models.transformers.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM


pytestmark = [pytest.mark.cpu]


def _ultra_style_config_dict() -> dict:
    return {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": 131072,
        "hidden_size": 64,
        "layers_block_type": ["mamba", "moe", "attention", "moe"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "mamba_num_heads": 8,
        "mamba_head_dim": 16,
        "n_groups": 4,
        "ssm_state_size": 32,
        "conv_kernel": 4,
        "chunk_size": 64,
        "mlp_hidden_act": "relu2",
        "mamba_hidden_act": "silu",
        "n_routed_experts": 16,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 48,
        "moe_shared_expert_intermediate_size": 96,
        "moe_latent_size": 32,
        "routed_scaling_factor": 5.0,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "time_step_floor": 1e-4,
        "time_step_min": 1e-3,
        "time_step_max": 0.1,
        "time_step_limit": [1e-4, 0.1],
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "rescale_prenorm_residual": True,
        "tie_word_embeddings": False,
    }


def test_nemotron_h_registered():
    registry = get_registry()
    assert "NemotronHForCausalLM" in registry.supported_models
    assert registry.get_model_cls_from_model_arch("NemotronHForCausalLM") is NemotronHForCausalLM


def test_local_auto_config_builds_nemotron_h_config(tmp_path):
    config_dir = tmp_path / "nemotron-3-ultra"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(_ultra_style_config_dict()))

    config = _load_local_xorl_config(str(config_dir), {})

    assert isinstance(config, NemotronHConfig)
    assert config.model_type == "nemotron_h"
    assert config.architectures == ["NemotronHForCausalLM"]
    assert config.layers_block_type == ["mamba", "moe", "attention", "moe"]
    assert config.num_hidden_layers == 4
    assert config.moe_latent_size == 32
    assert config.routed_scaling_factor == 5.0
    assert config.time_step_limit == (1e-4, 0.1)
    assert config.n_routed_experts == 16
    assert config.tie_word_embeddings is False
