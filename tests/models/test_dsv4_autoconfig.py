"""Verify ``transformers.AutoConfig.from_pretrained`` resolves to our
vendored ``DeepseekV4Config`` for the upstream HF ``model_type =
"deepseek_v4"`` declared in Flash's ``config.json``.

Without this dispatch, the xorl ``train`` CLI (which uses AutoConfig)
can't drive DSv4 training from the standard HF snapshot layout.
"""

import json
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu


_FLASH_SHAPE_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "qk_rope_head_dim": 4,
    "max_position_embeddings": 256,
    "q_lora_rank": 16,
    "o_groups": 1,
    "o_lora_rank": 8,
    "sliding_window": 8,
    "moe_intermediate_size": 16,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "num_hash_layers": 0,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "compress_ratios": [0, 0],
    "compress_rope_theta": 160000,
    "swiglu_limit": 0.0,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 128,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    },
    "num_nextn_predict_layers": 0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


def test_autoconfig_dispatches_to_deepseekv4_config():
    """``AutoConfig.from_pretrained(snapshot_with_model_type=deepseek_v4)``
    returns an instance of our ``DeepseekV4Config``."""
    from transformers import AutoConfig

    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config  # noqa: F401  registers

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        with cfg_path.open("w") as f:
            json.dump(_FLASH_SHAPE_CONFIG, f)
        cfg = AutoConfig.from_pretrained(tmp)

    assert isinstance(cfg, DeepseekV4Config)
    assert cfg.model_type == "deepseek_v4"
    assert cfg.num_hidden_layers == 2
    assert cfg.n_routed_experts == 4


def test_automodel_dispatches_to_for_causal_lm():
    """``AutoModelForCausalLM`` knows how to instantiate from our config.

    We don't actually call ``from_config`` here (it would materialize
    a model on cuda); we just check the registry entry resolved.
    """
    from transformers import AutoModelForCausalLM

    from xorl.models.transformers.deepseek_v4 import (  # noqa: F401  registers
        DeepseekV4Config,
        DeepseekV4ForCausalLM,
    )

    cls = AutoModelForCausalLM._model_mapping.get(DeepseekV4Config, None)
    assert cls is DeepseekV4ForCausalLM


def test_build_foundation_model_uses_xorl_registry():
    """The normal xorl train/server builder can instantiate DSv4."""
    from xorl.models import build_foundation_model
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        with cfg_path.open("w") as f:
            json.dump(_FLASH_SHAPE_CONFIG, f)

        model = build_foundation_model(
            tmp,
            init_device="meta",
            moe_implementation="eager",
            attn_implementation="flash_attention_3",
        )

    assert isinstance(model, DeepseekV4ForCausalLM)
