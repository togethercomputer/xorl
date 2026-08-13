import json

import pytest

from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_model_runner_resolves_qwen35_projection_defaults(tmp_path):
    config_dir = tmp_path / "qwen3-5-moe"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))

    runner = object.__new__(ModelRunner)
    runner.model_config = {"config_path": str(config_dir)}
    runner.lora_config = {
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }

    assert runner._resolve_lora_target_modules() == [
        "q_proj",
        "k_proj",
        "v_proj",
        "g_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
