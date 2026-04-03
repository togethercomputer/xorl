import pytest
import torch

from xorl.models.layers.moe import MoEExperts, MoEExpertsLoRA, MoELoRAConfig
from xorl.models.transformers.qwen3_5_moe.checkpoint_handler import Qwen3_5MoeCheckpointHandler
from xorl.models.transformers.qwen3_moe.checkpoint_handler import Qwen3MoeCheckpointHandler


pytestmark = [pytest.mark.cpu]


def test_moe_experts_register_only_fused_gate_up_proj():
    experts = MoEExperts(num_experts=3, hidden_dim=4, intermediate_size=5, moe_implementation="eager")

    named_params = dict(experts.named_parameters())
    assert set(named_params) == {"gate_up_proj", "down_proj"}
    assert named_params["gate_up_proj"].shape == (3, 4, 10)
    assert experts.gate_proj.shape == (3, 4, 5)
    assert experts.up_proj.shape == (3, 4, 5)

    with torch.no_grad():
        experts.gate_up_proj.zero_()
        experts.gate_proj.fill_(1.0)
        experts.up_proj.fill_(2.0)

    torch.testing.assert_close(experts.gate_up_proj[..., :5], torch.ones(3, 4, 5))
    torch.testing.assert_close(experts.gate_up_proj[..., 5:], torch.full((3, 4, 5), 2.0))


def test_moe_experts_lora_registers_fused_base_weight():
    experts = MoEExpertsLoRA(
        num_experts=3,
        hidden_dim=4,
        intermediate_size=5,
        moe_implementation="eager",
        lora_config=MoELoRAConfig(r=2, lora_alpha=4),
    )

    named_params = dict(experts.named_parameters())
    assert "gate_up_proj" in named_params
    assert "gate_proj" not in named_params
    assert "up_proj" not in named_params
    assert named_params["gate_up_proj"].shape == (3, 4, 10)
    assert experts.gate_proj.shape == (3, 4, 5)
    assert experts.up_proj.shape == (3, 4, 5)


def test_qwen3_moe_checkpoint_handler_round_trips_fused_experts():
    hidden_size = 4
    intermediate_size = 3
    handler = Qwen3MoeCheckpointHandler(
        num_experts=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
    )

    gate_0 = torch.arange(0, 12, dtype=torch.float32).view(intermediate_size, hidden_size)
    gate_1 = torch.arange(12, 24, dtype=torch.float32).view(intermediate_size, hidden_size)
    up_0 = torch.arange(24, 36, dtype=torch.float32).view(intermediate_size, hidden_size)
    up_1 = torch.arange(36, 48, dtype=torch.float32).view(intermediate_size, hidden_size)
    down_0 = torch.arange(48, 60, dtype=torch.float32).view(hidden_size, intermediate_size)
    down_1 = torch.arange(60, 72, dtype=torch.float32).view(hidden_size, intermediate_size)

    results = []
    for key, tensor in [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate_0),
        ("model.layers.0.mlp.experts.1.gate_proj.weight", gate_1),
        ("model.layers.0.mlp.experts.0.up_proj.weight", up_0),
        ("model.layers.0.mlp.experts.1.up_proj.weight", up_1),
        ("model.layers.0.mlp.experts.0.down_proj.weight", down_0),
        ("model.layers.0.mlp.experts.1.down_proj.weight", down_1),
    ]:
        results.extend(handler.on_load_weight(key, tensor))

    loaded = dict(results)
    expected_gate = torch.stack([gate_0.t(), gate_1.t()], dim=0)
    expected_up = torch.stack([up_0.t(), up_1.t()], dim=0)
    expected_gate_up = torch.cat([expected_gate, expected_up], dim=2)
    expected_down = torch.stack([down_0.t(), down_1.t()], dim=0)

    assert set(loaded) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    torch.testing.assert_close(loaded["model.layers.0.mlp.experts.gate_up_proj"], expected_gate_up)
    torch.testing.assert_close(loaded["model.layers.0.mlp.experts.down_proj"], expected_down)

    saved = dict(handler.on_save_weight("model.layers.0.mlp.experts.gate_up_proj", expected_gate_up))
    saved.update(handler.on_save_weight("model.layers.0.mlp.experts.down_proj", expected_down))

    torch.testing.assert_close(saved["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_0)
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.1.gate_proj.weight"], gate_1)
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.0.up_proj.weight"], up_0)
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.1.up_proj.weight"], up_1)
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.0.down_proj.weight"], down_0)
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.1.down_proj.weight"], down_1)


def test_qwen3_5_moe_checkpoint_handler_round_trips_fused_experts():
    hidden_size = 4
    intermediate_size = 3
    gate_up_weight = torch.arange(0, 48, dtype=torch.float32).view(2, 2 * intermediate_size, hidden_size)
    down_weight = torch.arange(48, 72, dtype=torch.float32).view(2, hidden_size, intermediate_size)

    handler = Qwen3_5MoeCheckpointHandler(
        num_experts=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        linear_key_dim=2,
        linear_value_dim=2,
    )

    loaded = dict(
        handler.on_load_weight("model.layers.0.mlp.experts.gate_up_proj.weight", gate_up_weight)
        + handler.on_load_weight("model.layers.0.mlp.experts.down_proj.weight", down_weight)
    )

    expected_gate_up = gate_up_weight.transpose(1, 2).contiguous()
    expected_down = down_weight.transpose(1, 2).contiguous()
    torch.testing.assert_close(loaded["model.layers.0.mlp.experts.gate_up_proj"], expected_gate_up)
    torch.testing.assert_close(loaded["model.layers.0.mlp.experts.down_proj"], expected_down)

    saved = dict(handler.on_save_weight("model.layers.0.mlp.experts.gate_up_proj", expected_gate_up))
    saved.update(handler.on_save_weight("model.layers.0.mlp.experts.down_proj", expected_down))

    gate, up = gate_up_weight.split(intermediate_size, dim=1)
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.0.gate_proj.weight"], gate[0])
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.1.gate_proj.weight"], gate[1])
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.0.up_proj.weight"], up[0])
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.1.up_proj.weight"], up[1])
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.0.down_proj.weight"], down_weight[0])
    torch.testing.assert_close(saved["model.layers.0.mlp.experts.1.down_proj.weight"], down_weight[1])
