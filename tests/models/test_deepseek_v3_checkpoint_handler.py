import json
import math

import pytest
import torch

from xorl.models.transformers.deepseek_v3.checkpoint_handler import DeepseekV3CheckpointHandler
from xorl.models.transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from xorl.models.transformers.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM


pytestmark = [pytest.mark.cpu]


def _expert_weight(expert_idx: int, proj: str) -> torch.Tensor:
    hidden_size = 2
    intermediate_size = 3
    value = float(expert_idx * 10 + {"gate": 1, "up": 2, "down": 3}[proj])
    if proj == "down":
        return torch.full((hidden_size, intermediate_size), value)
    return torch.full((intermediate_size, hidden_size), value)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.int8:
        raise ValueError(f"Expected int8 values to pack, got {values.dtype}")
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor to pack, got {tuple(values.shape)}")

    num_bits = 4
    pack_factor = 32 // num_bits
    unsigned = (values + (1 << (num_bits - 1))).to(torch.uint8)
    pad_cols = (-values.shape[1]) % pack_factor
    if pad_cols:
        unsigned = torch.nn.functional.pad(unsigned, (0, pad_cols))
    reshaped = unsigned.view(values.shape[0], -1, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, dtype=torch.int32) * num_bits
    return (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)


def _packed_expert_weight(expert_idx: int, proj: str) -> dict[str, torch.Tensor]:
    dense_weight = _expert_weight(expert_idx, proj)
    quantized = torch.ones_like(dense_weight, dtype=torch.int8)
    num_groups = max(1, math.ceil(dense_weight.shape[1] / 32))
    scales = torch.full((dense_weight.shape[0], num_groups), dense_weight.flatten()[0].item(), dtype=torch.float32)
    return {
        "weight_packed": _pack_int4(quantized),
        "weight_scale": scales,
        "weight_shape": torch.tensor(dense_weight.shape, dtype=torch.int64),
    }


def _tiny_config() -> DeepseekV3Config:
    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
    )
    config._attn_implementation = "eager"
    config._activation_native = True
    return config


def test_checkpoint_handler_merges_language_model_experts_and_skips_multimodal_keys():
    handler = DeepseekV3CheckpointHandler(num_experts=4)
    loaded = {}

    for expert_idx in range(4):
        for proj in ("gate", "up", "down"):
            loaded.update(
                handler.on_load_weight(
                    f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.weight",
                    _expert_weight(expert_idx, proj),
                )
            )

    loaded.update(handler.on_load_weight("language_model.model.layers.0.self_attn.o_proj.weight", torch.eye(2)))
    assert handler.on_load_weight("vision_tower.encoder.weight", torch.ones(1)) == []
    assert handler.on_load_weight("mm_projector.weight", torch.ones(1)) == []

    gate_up = dict(loaded)["model.layers.0.mlp.experts.gate_up_proj"]
    down = dict(loaded)["model.layers.0.mlp.experts.down_proj"]

    assert gate_up.shape == (4, 2, 6)
    assert down.shape == (4, 3, 2)
    assert torch.all(gate_up[0, :, :3] == 1.0)
    assert torch.all(gate_up[0, :, 3:] == 2.0)
    assert torch.all(gate_up[3, :, :3] == 31.0)
    assert torch.all(gate_up[3, :, 3:] == 32.0)
    assert torch.all(down[1] == 13.0)
    assert torch.equal(dict(loaded)["model.layers.0.self_attn.o_proj.weight"], torch.eye(2))


def test_checkpoint_handler_splits_internal_fused_experts_on_save():
    handler = DeepseekV3CheckpointHandler(num_experts=2)
    gate = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    up = gate + 100.0
    gate_up = torch.cat([gate, up], dim=2)
    down = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

    split_gate_up = dict(handler.on_save_weight("model.layers.0.mlp.experts.gate_up_proj", gate_up))
    split_down = dict(handler.on_save_weight("model.layers.0.mlp.experts.down_proj", down))

    assert torch.equal(split_gate_up["model.layers.0.mlp.experts.0.gate_proj.weight"], gate[0].transpose(0, 1))
    assert torch.equal(split_gate_up["model.layers.0.mlp.experts.1.up_proj.weight"], up[1].transpose(0, 1))
    assert torch.equal(split_down["model.layers.0.mlp.experts.0.down_proj.weight"], down[0].transpose(0, 1))
    assert torch.equal(split_down["model.layers.0.mlp.experts.1.down_proj.weight"], down[1].transpose(0, 1))


def test_checkpoint_handler_keeps_internal_fused_expert_layout_on_load():
    handler = DeepseekV3CheckpointHandler(num_experts=2)
    gate_up = torch.arange(2 * 2 * 6, dtype=torch.float32).reshape(2, 2, 6)
    down = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

    loaded_gate_up = dict(handler.on_load_weight("model.layers.0.mlp.experts.gate_up_proj", gate_up))
    loaded_down = dict(handler.on_load_weight("model.layers.0.mlp.experts.down_proj", down))

    assert torch.equal(loaded_gate_up["model.layers.0.mlp.experts.gate_up_proj"], gate_up)
    assert torch.equal(loaded_down["model.layers.0.mlp.experts.down_proj"], down)


def test_checkpoint_handler_ep_slices_to_local_experts():
    handler = DeepseekV3CheckpointHandler(num_experts=4, ep_rank=1, ep_size=2)
    skip_key = handler.get_skip_key_fn()
    loaded = {}

    for expert_idx in range(4):
        for proj in ("gate", "up", "down"):
            key = f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.weight"
            if skip_key is not None and skip_key(key):
                loaded.update(handler.on_skip_weight(key))
            else:
                loaded.update(handler.on_load_weight(key, _expert_weight(expert_idx, proj)))

    gate_up = dict(loaded)["model.layers.0.mlp.experts.gate_up_proj"]
    down = dict(loaded)["model.layers.0.mlp.experts.down_proj"]

    assert gate_up.shape == (2, 2, 6)
    assert down.shape == (2, 3, 2)
    assert gate_up[:, 0, 0].tolist() == [21.0, 31.0]
    assert down[:, 0, 0].tolist() == [23.0, 33.0]


def test_checkpoint_handler_loads_packed_expert_weights():
    handler = DeepseekV3CheckpointHandler(num_experts=4)
    loaded = {}

    for expert_idx in range(4):
        for proj in ("gate", "up", "down"):
            for suffix, tensor in _packed_expert_weight(expert_idx, proj).items():
                loaded.update(
                    handler.on_load_weight(
                        f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.{suffix}",
                        tensor,
                    )
                )

    gate_up = dict(loaded)["model.layers.0.mlp.experts.gate_up_proj"]
    down = dict(loaded)["model.layers.0.mlp.experts.down_proj"]

    assert gate_up.shape == (4, 2, 6)
    assert down.shape == (4, 3, 2)
    assert torch.all(gate_up[0, :, :3] == 1.0)
    assert torch.all(gate_up[0, :, 3:] == 2.0)
    assert torch.all(gate_up[3, :, :3] == 31.0)
    assert torch.all(gate_up[3, :, 3:] == 32.0)
    assert torch.all(down[1] == 13.0)


def test_checkpoint_handler_loads_packed_expert_weights_in_requested_dtype():
    handler = DeepseekV3CheckpointHandler(num_experts=4, device=torch.device("cpu"), dtype=torch.bfloat16)
    loaded = {}

    for expert_idx in range(4):
        for proj in ("gate", "up", "down"):
            for suffix, tensor in _packed_expert_weight(expert_idx, proj).items():
                loaded.update(
                    handler.on_load_weight(
                        f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.{suffix}",
                        tensor,
                    )
                )

    gate_up = dict(loaded)["model.layers.0.mlp.experts.gate_up_proj"]
    down = dict(loaded)["model.layers.0.mlp.experts.down_proj"]

    assert handler._expert_buffer is not None
    assert handler._expert_buffer._device == torch.device("cpu")
    assert gate_up.dtype == torch.bfloat16
    assert down.dtype == torch.bfloat16
    assert torch.all(gate_up[0, :, :3] == torch.tensor(1.0, dtype=torch.bfloat16))
    assert torch.all(down[1] == torch.tensor(13.0, dtype=torch.bfloat16))


def test_checkpoint_handler_ep_slices_packed_experts_to_local_experts():
    handler = DeepseekV3CheckpointHandler(num_experts=4, ep_rank=1, ep_size=2)
    skip_key = handler.get_skip_key_fn()
    loaded = {}

    for expert_idx in range(4):
        for proj in ("gate", "up", "down"):
            for suffix, tensor in _packed_expert_weight(expert_idx, proj).items():
                key = f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.{suffix}"
                if skip_key is not None and skip_key(key):
                    loaded.update(handler.on_skip_weight(key))
                else:
                    loaded.update(handler.on_load_weight(key, tensor))

    gate_up = dict(loaded)["model.layers.0.mlp.experts.gate_up_proj"]
    down = dict(loaded)["model.layers.0.mlp.experts.down_proj"]

    assert gate_up.shape == (2, 2, 6)
    assert down.shape == (2, 3, 2)
    assert gate_up[:, 0, 0].tolist() == [21.0, 31.0]
    assert down[:, 0, 0].tolist() == [23.0, 33.0]


def test_model_checkpoint_handler_accepts_official_packed_expert_layout():
    model = DeepseekV3ForCausalLM(_tiny_config())
    handler = model.get_checkpoint_handler(
        checkpoint_keys={"language_model.model.layers.0.mlp.experts.0.gate_proj.weight_packed"},
    )
    assert isinstance(handler, DeepseekV3CheckpointHandler)


def test_model_checkpoint_handler_reads_packed_quant_config_from_text_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k25",
                "text_config": {
                    "quantization_config": {
                        "quant_method": "compressed-tensors",
                        "format": "pack-quantized",
                        "config_groups": {
                            "group_0": {
                                "weights": {
                                    "group_size": 64,
                                    "num_bits": 8,
                                }
                            }
                        },
                    }
                },
            }
        )
    )

    model = DeepseekV3ForCausalLM(_tiny_config())
    handler = model.get_checkpoint_handler(
        checkpoint_keys={"language_model.model.layers.0.mlp.experts.0.gate_proj.weight_packed"},
        weights_path=str(tmp_path),
    )

    assert handler._packed_expert_group_size == 64
    assert handler._packed_expert_num_bits == 8
