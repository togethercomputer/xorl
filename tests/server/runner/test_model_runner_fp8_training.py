import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.server.runner.model_runner import ModelRunner


def _fake_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="tiny",
        _resolved_numerical_program={
            "attn_implementation": "eager",
            "router_fp32": False,
            "lm_head_fp32": False,
            "rmsnorm_mode": "eager",
            "activation_native": False,
            "rope_native": False,
            "rope_class_b": False,
            "attention_cast_bf16": False,
            "sparse_mla_enabled": False,
            "sparse_mla_backend": None,
        },
    )


def test_model_runner_defaults_fp8_training_to_fail_fast_fallback(monkeypatch):
    captured = {}

    def fake_build_training_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=nn.Linear(1, 1),
            model_config=_fake_model_config(),
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
    runner.ce_mode = None
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
            model_config=_fake_model_config(),
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
    runner.ce_mode = None
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


def test_model_runner_threads_glm52_block_fp8_qlora_mode(monkeypatch):
    captured = {}

    def fake_build_training_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=nn.Linear(1, 1),
            model_config=_fake_model_config(),
            pp_enabled=False,
            pp_stages=None,
            model_parts=None,
            has_first_stage=True,
            has_last_stage=True,
            optimizer_pre_hook_fn=None,
            is_prequantized=True,
            checkpoint_quant_format="block_fp8",
            exclude_modules=set(),
        )

    monkeypatch.setattr("xorl.server.runner.model_runner.build_training_model", fake_build_training_model)

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model_config = {
        "model_path": "zai-org/GLM-5.2-FP8",
        "config_path": "zai-org/GLM-5.2-FP8",
        "moe_implementation": "triton",
        "ep_dispatch": "deepep",
    }
    runner.train_config = {
        "enable_mixed_precision": False,
        "freeze_router": True,
        "init_device": "cpu",
    }
    runner.ce_mode = None
    runner.lora_config = {
        "enable_lora": True,
        "enable_qlora": True,
        "block_fp8_qlora_training": True,
        "quant_format": "block_fp8",
        "quant_group_size": 128,
        "moe_hybrid_shared_lora": True,
    }

    ModelRunner._initialize_model(runner)

    assert captured["enable_qlora"] is True
    assert captured["block_fp8_qlora_training"] is True
    assert captured["lora_target_modules"] is None
    assert captured["quant_format"] == "block_fp8"
    assert captured["quant_group_size"] == 128
    assert captured["moe_hybrid_shared_lora"] is True
    assert set(runner.lora_target_modules) == {
        "down_proj",
        "gate_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "lm_head",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "up_proj",
    }


def test_model_runner_threads_sharded_lm_head_loss_to_model_builder(monkeypatch):
    captured = {}

    def fake_build_training_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=nn.Linear(1, 1),
            model_config=_fake_model_config(),
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
        "fsdp_sharded_lm_head_loss": True,
        "enable_mixed_precision": False,
        "init_device": "cpu",
    }
    runner.ce_mode = None
    runner.lora_config = {}

    ModelRunner._initialize_model(runner)

    assert captured["fsdp_sharded_lm_head_loss"] is True


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


def test_dsv4_forward_only_reshards_external_lm_head_compute_view() -> None:
    class _Head:
        def __init__(self):
            self.calls = 0

        def reshard(self):
            self.calls += 1

    runner = object.__new__(ModelRunner)
    head = _Head()
    runner.model = SimpleNamespace(
        config=SimpleNamespace(_dsv4_flash_exact_mode=True),
        lm_head=head,
    )

    runner._reshard_exact_forward_only_lm_head()

    assert head.calls == 1


def test_exact_dsv4_auto_selects_complete_bank_adapter_export() -> None:
    runner = object.__new__(ModelRunner)
    runner.model_config_obj = SimpleNamespace(_dsv4_flash_exact_active_lora=True)
    runner.lora_config = {"lora_export_format": "peft"}

    runner._select_exact_dsv4_lora_export_format()

    assert runner.lora_config["lora_export_format"] == "dsv4_expert_banks"


def test_dsv4_forward_only_requires_fsdp_managed_lm_head() -> None:
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(_dsv4_flash_exact_mode=True),
        lm_head=nn.Module(),
    )

    with pytest.raises(RuntimeError, match="FSDP-managed lm_head"):
        runner._reshard_exact_forward_only_lm_head()


def test_dsv4_forward_only_reshards_lm_head_when_forward_fails() -> None:
    class _Head:
        def __init__(self):
            self.calls = 0

        def reshard(self):
            self.calls += 1

    runner = object.__new__(ModelRunner)
    head = _Head()
    runner.model = SimpleNamespace(
        config=SimpleNamespace(_dsv4_flash_exact_mode=True, vocab_size=2),
        lm_head=head,
    )
    runner.is_sleeping = False
    runner.lora_config = {"enable_lora": False}
    runner._adapter_manager = None
    runner._active_session_id = None
    runner.pp_enabled = False
    runner._routing_handler = SimpleNamespace(setup=lambda *_args, **_kwargs: False)

    def fail_forward(*_args, **_kwargs):
        raise ValueError("forward failed")

    runner._forward_loop = fail_forward

    with pytest.raises(ValueError, match="forward failed"):
        runner.forward([{"input_ids": torch.tensor([[0]])}])

    assert head.calls == 1
