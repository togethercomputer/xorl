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


def _pack_quantized(values: torch.Tensor, *, num_bits: int) -> torch.Tensor:
    if values.dtype != torch.int8:
        raise ValueError(f"Expected int8 values to pack, got {values.dtype}")
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor to pack, got {tuple(values.shape)}")

    pack_factor = 32 // num_bits
    unsigned = (values + (1 << (num_bits - 1))).to(torch.uint8)
    pad_cols = (-values.shape[1]) % pack_factor
    if pad_cols:
        unsigned = torch.nn.functional.pad(unsigned, (0, pad_cols))
    reshaped = unsigned.view(values.shape[0], -1, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, dtype=torch.int32) * num_bits
    return (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)


def _packed_expert_weight(
    expert_idx: int,
    proj: str,
    *,
    num_bits: int = 4,
    group_size: int = 32,
) -> dict[str, torch.Tensor]:
    dense_weight = _expert_weight(expert_idx, proj)
    quantized = torch.ones_like(dense_weight, dtype=torch.int8)
    num_groups = max(1, math.ceil(dense_weight.shape[1] / group_size))
    scales = torch.full((dense_weight.shape[0], num_groups), dense_weight.flatten()[0].item(), dtype=torch.float32)
    return {
        "weight_packed": _pack_quantized(quantized, num_bits=num_bits),
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


def _load_external_experts(
    handler: DeepseekV3CheckpointHandler,
    *,
    packed: bool = False,
    packed_num_bits: int = 4,
    packed_group_size: int = 32,
) -> dict[str, torch.Tensor]:
    loaded = {}
    skip_key = handler.get_skip_key_fn()
    for expert_idx in range(4):
        for proj in ("gate", "up", "down"):
            weights = (
                _packed_expert_weight(
                    expert_idx,
                    proj,
                    num_bits=packed_num_bits,
                    group_size=packed_group_size,
                )
                if packed
                else {"weight": _expert_weight(expert_idx, proj)}
            )
            for suffix, tensor in weights.items():
                key = f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.{suffix}"
                if skip_key is not None and skip_key(key):
                    loaded.update(handler.on_skip_weight(key))
                else:
                    loaded.update(handler.on_load_weight(key, tensor))
    return dict(loaded)


def test_checkpoint_handler_expert_layout_ep_and_packed_policy(tmp_path):
    handler = DeepseekV3CheckpointHandler(num_experts=4)
    loaded = _load_external_experts(handler)

    loaded.update(handler.on_load_weight("language_model.model.layers.0.self_attn.o_proj.weight", torch.eye(2)))
    assert handler.on_load_weight("vision_tower.encoder.weight", torch.ones(1)) == []
    assert handler.on_load_weight("mm_projector.weight", torch.ones(1)) == []

    gate_up = loaded["model.layers.0.mlp.experts.gate_up_proj"]
    down = loaded["model.layers.0.mlp.experts.down_proj"]

    assert gate_up.shape == (4, 2, 6)
    assert down.shape == (4, 3, 2)
    assert torch.all(gate_up[0, :, :3] == 1.0)
    assert torch.all(gate_up[0, :, 3:] == 2.0)
    assert torch.all(gate_up[3, :, :3] == 31.0)
    assert torch.all(gate_up[3, :, 3:] == 32.0)
    assert torch.all(down[1] == 13.0)
    assert torch.equal(loaded["model.layers.0.self_attn.o_proj.weight"], torch.eye(2))

    internal_handler = DeepseekV3CheckpointHandler(num_experts=2)
    gate = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    up = gate + 100.0
    internal_gate_up = torch.cat([gate, up], dim=2)
    internal_down = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

    loaded_gate_up = dict(internal_handler.on_load_weight("model.layers.0.mlp.experts.gate_up_proj", internal_gate_up))
    loaded_down = dict(internal_handler.on_load_weight("model.layers.0.mlp.experts.down_proj", internal_down))
    assert torch.equal(loaded_gate_up["model.layers.0.mlp.experts.gate_up_proj"], internal_gate_up)
    assert torch.equal(loaded_down["model.layers.0.mlp.experts.down_proj"], internal_down)

    split_gate_up = dict(internal_handler.on_save_weight("model.layers.0.mlp.experts.gate_up_proj", internal_gate_up))
    split_down = dict(internal_handler.on_save_weight("model.layers.0.mlp.experts.down_proj", internal_down))

    assert torch.equal(split_gate_up["model.layers.0.mlp.experts.0.gate_proj.weight"], gate[0].transpose(0, 1))
    assert torch.equal(split_gate_up["model.layers.0.mlp.experts.1.up_proj.weight"], up[1].transpose(0, 1))
    assert torch.equal(split_down["model.layers.0.mlp.experts.0.down_proj.weight"], internal_down[0].transpose(0, 1))
    assert torch.equal(split_down["model.layers.0.mlp.experts.1.down_proj.weight"], internal_down[1].transpose(0, 1))

    _assert_checkpoint_handler_ep_slices_dense_and_packed_experts()
    _assert_checkpoint_handler_loads_packed_expert_weights_in_requested_dtype_and_config(tmp_path)


def _assert_checkpoint_handler_ep_slices_dense_and_packed_experts():
    for packed in (False, True):
        handler = DeepseekV3CheckpointHandler(num_experts=4, ep_rank=1, ep_size=2)
        loaded = _load_external_experts(handler, packed=packed)
        gate_up = loaded["model.layers.0.mlp.experts.gate_up_proj"]
        down = loaded["model.layers.0.mlp.experts.down_proj"]

        assert gate_up.shape == (2, 2, 6)
        assert down.shape == (2, 3, 2)
        assert gate_up[:, 0, 0].tolist() == [21.0, 31.0]
        assert down[:, 0, 0].tolist() == [23.0, 33.0]


def _assert_checkpoint_handler_loads_packed_expert_weights_in_requested_dtype_and_config(tmp_path):
    model = DeepseekV3ForCausalLM(_tiny_config())
    checkpoint_keys = {"language_model.model.layers.0.mlp.experts.0.gate_proj.weight_packed"}
    default_handler = model.get_checkpoint_handler(checkpoint_keys=checkpoint_keys)
    assert isinstance(default_handler, DeepseekV3CheckpointHandler)

    default_loaded = _load_external_experts(default_handler, packed=True)
    default_gate_up = default_loaded["model.layers.0.mlp.experts.gate_up_proj"]
    default_down = default_loaded["model.layers.0.mlp.experts.down_proj"]
    assert default_gate_up.shape == (4, 2, 6)
    assert default_down.shape == (4, 3, 2)
    assert torch.all(default_gate_up[0, :, :3] == 1.0)
    assert torch.all(default_gate_up[3, :, 3:] == 32.0)
    assert torch.all(default_down[1] == 13.0)

    handler = DeepseekV3CheckpointHandler(num_experts=4, device=torch.device("cpu"), dtype=torch.bfloat16)
    loaded = _load_external_experts(handler, packed=True)
    gate_up = loaded["model.layers.0.mlp.experts.gate_up_proj"]
    down = loaded["model.layers.0.mlp.experts.down_proj"]

    assert handler._expert_buffer is not None
    assert handler._expert_buffer._device == torch.device("cpu")
    assert gate_up.dtype == torch.bfloat16
    assert down.dtype == torch.bfloat16
    assert torch.all(gate_up[0, :, :3] == torch.tensor(1.0, dtype=torch.bfloat16))
    assert torch.all(down[1] == torch.tensor(13.0, dtype=torch.bfloat16))

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

    configured_handler = model.get_checkpoint_handler(
        checkpoint_keys=checkpoint_keys,
        weights_path=str(tmp_path),
    )
    configured_loaded = _load_external_experts(
        configured_handler,
        packed=True,
        packed_num_bits=8,
        packed_group_size=64,
    )

    configured_gate_up = configured_loaded["model.layers.0.mlp.experts.gate_up_proj"]
    configured_down = configured_loaded["model.layers.0.mlp.experts.down_proj"]
    assert torch.all(configured_gate_up[0, :, :3] == 1.0)
    assert torch.all(configured_gate_up[3, :, 3:] == 32.0)
    assert torch.all(configured_down[1] == 13.0)
