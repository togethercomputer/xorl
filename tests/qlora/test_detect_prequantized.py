"""Tests for pre-quantized checkpoint detection (NVFP4 and block FP8)
and dense Qwen3 checkpoint handler prequantized key skipping.

Verifies that detect_prequantized_checkpoint() correctly identifies NVFP4
checkpoints, detect_prequantized_block_fp8_checkpoint() identifies block
FP8 checkpoints, and that the dense Qwen3CheckpointHandler correctly
skips quantized keys when is_prequantized=True.
"""

import json
import os
import tempfile

import pytest
import torch

pytestmark = [pytest.mark.cpu]

from xorl.models.checkpoint_handlers.buffers import (
    detect_prequantized_checkpoint,
    detect_prequantized_block_fp8_checkpoint,
    get_prequantized_exclude_modules,
)
from xorl.models.transformers.qwen3.checkpoint_handler import Qwen3CheckpointHandler
from xorl.models.transformers.qwen3_moe.checkpoint_handler import Qwen3MoeCheckpointHandler


class TestDetectPrequantizedCheckpoint:
    """Test NVFP4 and block FP8 detection from config files."""

    def test_nvfp4_detection(self, tmp_path):
        """All valid NVFP4 formats detected; non-NVFP4/missing/None/empty/malformed return False; precedence correct."""
        # -- Positive cases --
        # modelopt nested format
        with open(tmp_path / "hf_quant_config.json", "w") as f:
            json.dump({
                "producer": {"name": "modelopt", "version": "0.34.1"},
                "quantization": {"quant_algo": "NVFP4", "group_size": 16, "exclude_modules": ["lm_head"]},
            }, f)
        assert detect_prequantized_checkpoint(str(tmp_path)) is True

        # flat format
        d2 = tmp_path / "flat"
        d2.mkdir()
        with open(d2 / "hf_quant_config.json", "w") as f:
            json.dump({"quant_algo": "NVFP4"}, f)
        assert detect_prequantized_checkpoint(str(d2)) is True

        # config.json quantization_config
        d3 = tmp_path / "cfgjson"
        d3.mkdir()
        with open(d3 / "config.json", "w") as f:
            json.dump({"quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"}}, f)
        assert detect_prequantized_checkpoint(str(d3)) is True

        # hf_quant_config takes precedence
        d4 = tmp_path / "prec"
        d4.mkdir()
        with open(d4 / "hf_quant_config.json", "w") as f:
            json.dump({"quantization": {"quant_algo": "NVFP4"}}, f)
        with open(d4 / "config.json", "w") as f:
            json.dump({"quantization_config": {"quant_algo": "INT8"}}, f)
        assert detect_prequantized_checkpoint(str(d4)) is True

        # -- Negative cases --
        d5 = tmp_path / "int8"
        d5.mkdir()
        with open(d5 / "hf_quant_config.json", "w") as f:
            json.dump({"quantization": {"quant_algo": "INT8"}}, f)
        assert detect_prequantized_checkpoint(str(d5)) is False

        d6 = tmp_path / "empty_dir"
        d6.mkdir()
        assert detect_prequantized_checkpoint(str(d6)) is False
        assert detect_prequantized_checkpoint(None) is False
        assert detect_prequantized_checkpoint("/nonexistent/path") is False

        d7 = tmp_path / "empty_cfg"
        d7.mkdir()
        with open(d7 / "hf_quant_config.json", "w") as f:
            json.dump({}, f)
        assert detect_prequantized_checkpoint(str(d7)) is False

        d8 = tmp_path / "malformed"
        d8.mkdir()
        with open(d8 / "hf_quant_config.json", "w") as f:
            f.write("not valid json {{{")
        assert detect_prequantized_checkpoint(str(d8)) is False

    def test_block_fp8_detection(self, tmp_path):
        """Block FP8 detected via config.json and safetensors index; NVFP4/wrong-size/plain/None return False."""
        # config.json with fp8 + [128, 128]
        d1 = tmp_path / "fp8"
        d1.mkdir()
        with open(d1 / "config.json", "w") as f:
            json.dump({"quantization_config": {
                "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
            }}, f)
        assert detect_prequantized_block_fp8_checkpoint(str(d1)) is True

        # safetensors index with weight_scale_inv
        d2 = tmp_path / "safe"
        d2.mkdir()
        with open(d2 / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "s.safetensors",
                "model.layers.0.self_attn.q_proj.weight_scale_inv": "s.safetensors",
            }}, f)
        assert detect_prequantized_block_fp8_checkpoint(str(d2)) is True

        # NVFP4 index (weight_scale + weight_scale_2) not FP8
        d3 = tmp_path / "nvfp4"
        d3.mkdir()
        with open(d3 / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {
                "m.q_proj.weight": "s.safetensors",
                "m.q_proj.weight_scale": "s.safetensors",
                "m.q_proj.weight_scale_2": "s.safetensors",
            }}, f)
        assert detect_prequantized_block_fp8_checkpoint(str(d3)) is False

        d4 = tmp_path / "wrong_bs"
        d4.mkdir()
        with open(d4 / "config.json", "w") as f:
            json.dump({"quantization_config": {"quant_method": "fp8", "weight_block_size": [64, 64]}}, f)
        assert detect_prequantized_block_fp8_checkpoint(str(d4)) is False

        d5 = tmp_path / "bf16"
        d5.mkdir()
        with open(d5 / "config.json", "w") as f:
            json.dump({"model_type": "qwen3"}, f)
        assert detect_prequantized_block_fp8_checkpoint(str(d5)) is False
        assert detect_prequantized_block_fp8_checkpoint(None) is False
        assert detect_prequantized_block_fp8_checkpoint("/nonexistent") is False


class TestQwen3DenseCheckpointHandlerPrequantized:
    """Test Qwen3CheckpointHandler skip_fn, on_load_weight, QKV merge, and bias merge."""

    _skip_keys = [
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.q_proj.weight_scale_2",
        "model.layers.0.self_attn.q_proj.input_scale",
        "model.layers.0.mlp.gate_proj.weight_scale",
        "model.layers.0.mlp.gate_proj.weight_scale_2",
        "model.layers.0.mlp.down_proj.input_scale",
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.layers.0.mlp.gate_proj.weight_scale_inv",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]

    _passthrough_keys = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    def _make_handler(self, is_prequantized, **kwargs):
        return Qwen3CheckpointHandler(
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            is_prequantized=is_prequantized, **kwargs,
        )

    def test_prequantized_skip_and_on_load(self):
        """skip_fn skips quant keys, passes norms; on_load_weight matches; normal mode QKV merge works; bias merge works."""
        handler = self._make_handler(is_prequantized=True)
        skip_fn = handler.get_skip_key_fn()
        assert skip_fn is not None

        for key in self._skip_keys:
            assert skip_fn(key), f"Expected to skip: {key}"
        for key in self._passthrough_keys:
            assert not skip_fn(key), f"Should NOT skip: {key}"

        dummy = torch.zeros(1)
        for key in self._skip_keys:
            assert handler.on_load_weight(key, dummy) == []
        for key in self._passthrough_keys:
            result = handler.on_load_weight(key, torch.randn(64))
            assert len(result) == 1 and result[0][0] == key

        # Normal mode: skip_fn is None, QKV merge works
        handler_normal = self._make_handler(is_prequantized=False)
        assert handler_normal.get_skip_key_fn() is None
        q, k, v = torch.randn(8192, 5120), torch.randn(1024, 5120), torch.randn(1024, 5120)
        assert handler_normal.on_load_weight("model.layers.0.self_attn.q_proj.weight", q) == []
        assert handler_normal.on_load_weight("model.layers.0.self_attn.k_proj.weight", k) == []
        r3 = handler_normal.on_load_weight("model.layers.0.self_attn.v_proj.weight", v)
        assert len(r3) == 1 and "qkv_proj" in r3[0][0] and r3[0][1].shape[0] == 10240

        # Prequantized bias merge
        handler_preq = self._make_handler(is_prequantized=True)
        qb, kb, vb = torch.randn(8192), torch.randn(1024), torch.randn(1024)
        assert handler_preq.on_load_weight("model.layers.0.self_attn.q_proj.bias", qb) == []
        assert handler_preq.on_load_weight("model.layers.0.self_attn.k_proj.bias", kb) == []
        r3b = handler_preq.on_load_weight("model.layers.0.self_attn.v_proj.bias", vb)
        assert len(r3b) == 1 and "qkv_proj" in r3b[0][0] and r3b[0][1].shape[0] == 10240


class TestGetPrequantizedExcludeModules:
    """Test get_prequantized_exclude_modules: all config formats and edge cases."""

    def test_exclude_modules_all_cases(self, tmp_path):
        """All config formats return correct modules; empty/missing/None/malformed return empty set; precedence correct."""
        # modelopt nested
        d1 = tmp_path / "nested"
        d1.mkdir()
        with open(d1 / "hf_quant_config.json", "w") as f:
            json.dump({"quantization": {"quant_algo": "NVFP4", "exclude_modules": ["lm_head", "gate"]}}, f)
        assert get_prequantized_exclude_modules(str(d1)) == {"lm_head", "gate"}

        # flat
        d2 = tmp_path / "flat"
        d2.mkdir()
        with open(d2 / "hf_quant_config.json", "w") as f:
            json.dump({"quant_algo": "NVFP4", "exclude_modules": ["lm_head"]}, f)
        assert get_prequantized_exclude_modules(str(d2)) == {"lm_head"}

        # config.json
        d3 = tmp_path / "cfg"
        d3.mkdir()
        with open(d3 / "config.json", "w") as f:
            json.dump({"quantization_config": {"exclude_modules": ["lm_head", "gate"]}}, f)
        assert get_prequantized_exclude_modules(str(d3)) == {"lm_head", "gate"}

        # modules_to_not_convert (HF FP8)
        d4 = tmp_path / "fp8"
        d4.mkdir()
        with open(d4 / "config.json", "w") as f:
            json.dump({"quantization_config": {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}}, f)
        assert get_prequantized_exclude_modules(str(d4)) == {"lm_head"}

        # Precedence
        d5 = tmp_path / "prec"
        d5.mkdir()
        with open(d5 / "hf_quant_config.json", "w") as f:
            json.dump({"quantization": {"exclude_modules": ["lm_head", "gate"]}}, f)
        with open(d5 / "config.json", "w") as f:
            json.dump({"quantization_config": {"exclude_modules": ["different"]}}, f)
        assert get_prequantized_exclude_modules(str(d5)) == {"lm_head", "gate"}

        # No key, no files, None, nonexistent, malformed -> empty set
        d6 = tmp_path / "nokey"
        d6.mkdir()
        with open(d6 / "hf_quant_config.json", "w") as f:
            json.dump({"quantization": {"quant_algo": "NVFP4"}}, f)
        assert get_prequantized_exclude_modules(str(d6)) == set()
        d7 = tmp_path / "empty_dir"
        d7.mkdir()
        assert get_prequantized_exclude_modules(str(d7)) == set()
        assert get_prequantized_exclude_modules(None) == set()
        assert get_prequantized_exclude_modules("/nonexistent") == set()
        d8 = tmp_path / "bad"
        d8.mkdir()
        with open(d8 / "hf_quant_config.json", "w") as f:
            f.write("not json")
        assert get_prequantized_exclude_modules(str(d8)) == set()


class TestCheckpointHandlerExcludeModules:
    """Test dense and MoE checkpoint handlers with exclude_modules."""

    def test_dense_handler_exclude_modules(self):
        """Excluded modules not skipped in skip_fn and on_load_weight; non-excluded skipped; empty/no exclude normal."""
        handler = Qwen3CheckpointHandler(
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            is_prequantized=True, exclude_modules={"down_proj"},
        )
        skip_fn = handler.get_skip_key_fn()
        assert not skip_fn("model.layers.0.mlp.down_proj.weight")
        assert skip_fn("model.layers.0.self_attn.o_proj.weight")
        assert skip_fn("model.layers.0.self_attn.o_proj.weight_scale")

        tensor = torch.randn(1024, 512)
        result = handler.on_load_weight("model.layers.0.mlp.down_proj.weight", tensor)
        assert len(result) == 1 and result[0][0] == "model.layers.0.mlp.down_proj.weight"
        assert handler.on_load_weight("model.layers.0.self_attn.o_proj.weight", tensor) == []

        # Aux keys for excluded module pass through
        handler2 = Qwen3CheckpointHandler(
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            is_prequantized=True, exclude_modules={"o_proj"},
        )
        sfn2 = handler2.get_skip_key_fn()
        assert not sfn2("model.layers.0.self_attn.o_proj.weight")
        assert not sfn2("model.layers.0.self_attn.o_proj.weight_scale")
        assert not sfn2("model.layers.0.self_attn.o_proj.weight_scale_inv")
        assert sfn2("model.layers.0.mlp.down_proj.weight")

        # Empty/no exclude_modules: all quant weight keys skipped
        for kwargs in [{"exclude_modules": set()}, {}]:
            h = Qwen3CheckpointHandler(
                num_attention_heads=64, num_key_value_heads=8, head_dim=128,
                is_prequantized=True, **kwargs,
            )
            assert h.get_skip_key_fn()("model.layers.0.mlp.down_proj.weight")

    def test_moe_handler_exclude_modules(self):
        """MoE handler: excluded keys pass through, non-excluded skipped; on_load_weight consistent."""
        handler = Qwen3MoeCheckpointHandler(
            num_experts=64, num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            is_prequantized=True, exclude_modules={"gate"},
        )
        assert not handler.get_skip_key_fn()("model.layers.0.mlp.gate.weight")
        assert handler.get_skip_key_fn()("model.layers.0.self_attn.o_proj.weight")

        handler2 = Qwen3MoeCheckpointHandler(
            num_experts=64, num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            is_prequantized=True, exclude_modules={"down_proj"},
        )
        tensor = torch.randn(1024, 512)
        result = handler2.on_load_weight("model.layers.0.mlp.shared_expert.down_proj.weight", tensor)
        assert len(result) == 1 and result[0][0] == "model.layers.0.mlp.shared_expert.down_proj.weight"
        assert handler2.on_load_weight("model.layers.0.self_attn.o_proj.weight", tensor) == []
