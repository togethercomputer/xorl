import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.server.runner.model_runner import ModelRunner


def test_model_runner_defaults_fp8_training_to_fail_fast_fallback(monkeypatch):
    captured = {}

    def fake_build_training_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=nn.Linear(1, 1),
            model_config=object(),
            pp_enabled=False,
            pp_stages=None,
            model_parts=None,
            has_first_stage=True,
            has_last_stage=True,
            optimizer_pre_hook_fn=None,
            is_prequantized=False,
            checkpoint_quant_format=None,
            exclude_modules=set(),
        )

    monkeypatch.setattr("xorl.server.runner.model_runner.build_training_model", fake_build_training_model)

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model_config = {
        "model_path": "Qwen/Qwen3-8B",
        "config_path": "Qwen/Qwen3-8B",
    }
    runner.train_config = {
        "enable_fp8_training": True,
        "enable_mixed_precision": False,
        "init_device": "cpu",
    }
    runner.lora_config = {}

    ModelRunner._initialize_model(runner)

    assert captured["enable_fp8_training"] is True
    assert captured["fp8_training_allow_bf16_fallback"] is False


def test_model_runner_threads_qarl_config_to_model_builder(monkeypatch):
    captured = {}

    def fake_build_training_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=nn.Linear(1, 1),
            model_config=object(),
            pp_enabled=False,
            pp_stages=None,
            model_parts=None,
            has_first_stage=True,
            has_last_stage=True,
            optimizer_pre_hook_fn=None,
            is_prequantized=False,
            checkpoint_quant_format=None,
            exclude_modules=set(),
        )

    monkeypatch.setattr("xorl.server.runner.model_runner.build_training_model", fake_build_training_model)

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model_config = {
        "model_path": "Qwen/Qwen3-8B",
        "config_path": "Qwen/Qwen3-8B",
    }
    runner.train_config = {
        "enable_qarl": True,
        "qarl_quant_cfg": {"format": "fp8_e4m3", "activation": False},
        "qarl_calib_data": "/tmp/qarl-calib.json",
        "qarl_calib_size": 4,
        "qarl_quant_sequence_length": 16,
        "qarl_sync_format": "fp8",
        "qarl_target_modules": ["q_proj"],
        "qarl_exclude_modules": ["lm_head"],
        "enable_mixed_precision": False,
        "init_device": "cpu",
    }
    runner.lora_config = {}

    ModelRunner._initialize_model(runner)

    assert captured["enable_qarl"] is True
    assert captured["qarl_quant_cfg"] == {"format": "fp8_e4m3", "activation": False}
    assert captured["qarl_calib_data"] == "/tmp/qarl-calib.json"
    assert captured["qarl_calib_size"] == 4
    assert captured["qarl_quant_sequence_length"] == 16
    assert captured["qarl_sync_format"] == "fp8"
    assert captured["qarl_target_modules"] == ["q_proj"]
    assert captured["qarl_exclude_modules"] == ["lm_head"]


def test_model_runner_causallm_loss_returns_raw_token_sum():
    class TinyCausalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()

        def forward(self, input_ids, **_kwargs):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            return SimpleNamespace(last_hidden_state=hidden)

    runner = object.__new__(ModelRunner)
    runner.model = TinyCausalModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False

    loss, per_token_outputs, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        {
            "input_ids": torch.tensor([[0, 0]]),
            "labels": torch.tensor([[0, 1]]),
        },
        "causallm_loss",
        {},
    )

    assert loss.item() == pytest.approx(2 * math.log(2.0))
    assert per_token_outputs["loss"].reshape(-1).tolist() == pytest.approx([math.log(2.0), math.log(2.0)])
