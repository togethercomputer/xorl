from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

import xorl.models.transformers.glm5.exact_dense_mlp as exact_dense_mlp_module
from xorl.lora.utils import get_lora_state_dict, load_lora_state_dict, save_lora_checkpoint
from xorl.models.transformers.glm5.exact_dense_mlp import (
    GLM52_EXACT_TP1_DENSE_MLP_CONTRACT_VERSION,
    Glm52ExactTP1DenseMLP,
)
from xorl.models.transformers.glm5.exact_gate_up_qlora import (
    Glm52ExactTP1FusedGateUpBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear


def _module() -> Glm52ExactTP1DenseMLP:
    module = Glm52ExactTP1DenseMLP(8, 128, device=torch.device("cpu"))
    with torch.no_grad():
        module.gate_proj.lora_A.copy_(
            torch.tensor([[0.1001, -0.2002, 0.3003, -0.4004, 0.5005, -0.6006, 0.7007, -0.8008]])
        )
        module.up_proj.lora_A.copy_(
            torch.tensor([[-0.8108, 0.7107, -0.6106, 0.5105, -0.4104, 0.3103, -0.2102, 0.1101]])
        )
        module.gate_proj.lora_B.copy_(torch.arange(128, dtype=torch.float32).sub_(47).div_(311).unsqueeze(1))
        module.up_proj.lora_B.copy_(torch.arange(128, dtype=torch.float32).sub_(73).div_(277).neg_().unsqueeze(1))
        module.down_proj.lora_A.copy_(torch.arange(128, dtype=torch.float32).sub_(61).div_(389).unsqueeze(0))
        module.down_proj.lora_B.copy_(torch.arange(8, dtype=torch.float32).sub_(3).div_(173).unsqueeze(1))
    return module


def _literal_linear_value(
    input: torch.Tensor,
    base_weight: torch.Tensor,
    factor_A: torch.Tensor,
    factor_B: torch.Tensor,
) -> torch.Tensor:
    base = F.linear(input.float(), base_weight.float()).to(torch.bfloat16)
    lora_A_output = F.linear(input.float(), factor_A.float()).to(torch.bfloat16)
    lora_delta = F.linear(lora_A_output.float(), factor_B.float()).to(torch.bfloat16)
    return (base + lora_delta).to(torch.bfloat16)


def _literal_gate_up_value(
    input: torch.Tensor,
    base_weight: torch.Tensor,
    gate_A: torch.Tensor,
    gate_B: torch.Tensor,
    up_A: torch.Tensor,
    up_B: torch.Tensor,
) -> torch.Tensor:
    base = F.linear(input.float(), base_weight.float()).to(torch.bfloat16)
    gate_A_output = F.linear(input.float(), gate_A.float()).to(torch.bfloat16)
    gate_delta = F.linear(gate_A_output.float(), gate_B.float()).to(torch.bfloat16)
    up_A_output = F.linear(input.float(), up_A.float()).to(torch.bfloat16)
    up_delta = F.linear(up_A_output.float(), up_B.float()).to(torch.bfloat16)
    return (base + torch.cat((gate_delta, up_delta), dim=-1)).to(torch.bfloat16)


def test_dense_mlp_root_preserves_six_canonical_unique_factor_paths_without_aliases() -> None:
    module = _module()
    module.bind_checkpoint_sources("model.layers.0.mlp")

    assert module.contract_version == GLM52_EXACT_TP1_DENSE_MLP_CONTRACT_VERSION
    assert isinstance(module, Glm52ExactTP1FusedGateUpBlockFP8QLoRA)
    assert isinstance(module.down_proj, Glm52ExactTP1BlockFP8QLoRALinear)
    assert module.get_submodule("gate_proj") is module.gate_proj
    assert module.get_submodule("up_proj") is module.up_proj
    assert module.get_submodule("down_proj") is module.down_proj
    trainable = {name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert tuple(trainable) == module.logical_factor_names
    assert len({id(parameter) for parameter in trainable.values()}) == 6
    assert all(parameter.dtype is torch.float32 for parameter in trainable.values())

    expected_state_paths = {
        "packed_weight_f32",
        "weight_scale_inv",
        "gate_proj.lora_A",
        "gate_proj.lora_B",
        "up_proj.lora_A",
        "up_proj.lora_B",
        "down_proj.lora_A",
        "down_proj.lora_B",
        "down_proj.packed_weight_f32",
        "down_proj.weight_block_scales",
    }
    state_paths = tuple(module.state_dict())
    assert set(state_paths) == expected_state_paths
    assert len(state_paths) == len(set(state_paths))
    assert not any(path.startswith(("gate_up.", "base.", "down_proj.base.")) for path in state_paths)

    state_objects = [
        *module.named_parameters(remove_duplicate=False),
        *module.named_buffers(remove_duplicate=False),
    ]
    assert len({name for name, _ in state_objects}) == len(state_objects)
    assert len({id(value) for _, value in state_objects}) == len(state_objects)
    assert module._exact_gate_source_fqn == "model.layers.0.mlp.gate_proj"
    assert module._exact_up_source_fqn == "model.layers.0.mlp.up_proj"
    assert module.down_proj._source_fqn == "model.layers.0.mlp.down_proj"
    assert not hasattr(module, "_source_fqn")
    module.bind_checkpoint_sources("model.layers.0.mlp")
    with pytest.raises(RuntimeError, match="immutable once bound"):
        module.bind_checkpoint_sources("model.layers.1.mlp")


def test_dense_mlp_forward_composes_fused_gate_up_production_activation_and_exact_down(monkeypatch) -> None:
    module = _module()
    fused_base = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8).sub_(719).div_(1543).to(torch.bfloat16)
    down_base = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128).sub_(401).div_(1291).to(torch.bfloat16)
    events = []

    def gate_up_value(input, gate_A, gate_B, up_A, up_B):
        events.append(("gate_up", tuple(input.shape)))
        return _literal_gate_up_value(input, fused_base, gate_A, gate_B, up_A, up_B)

    def activation_value(gate_up):
        events.append(("activation", tuple(gate_up.shape)))
        return F.silu(gate_up[..., :128]) * gate_up[..., 128:]

    def down_value(input, factor_A, factor_B):
        events.append(("down", tuple(input.shape)))
        return _literal_linear_value(input, down_base, factor_A, factor_B)

    monkeypatch.setattr(module, "_exact_forward_value", gate_up_value)
    monkeypatch.setattr(exact_dense_mlp_module, "fused_silu_and_mul", activation_value)
    monkeypatch.setattr(module.down_proj, "_exact_forward_value", down_value)
    input = torch.arange(24, dtype=torch.float32).reshape(3, 8).sub_(7).div_(53).to(torch.bfloat16)

    actual = module(input)

    effective_gate_A = module.gate_proj.lora_A.detach().to(torch.bfloat16)
    effective_gate_B = module.gate_proj.lora_B.detach().to(torch.bfloat16)
    effective_up_A = module.up_proj.lora_A.detach().to(torch.bfloat16)
    effective_up_B = module.up_proj.lora_B.detach().to(torch.bfloat16)
    effective_down_A = module.down_proj.lora_A.detach().to(torch.bfloat16)
    effective_down_B = module.down_proj.lora_B.detach().to(torch.bfloat16)
    expected_gate_up = _literal_gate_up_value(
        input,
        fused_base,
        effective_gate_A,
        effective_gate_B,
        effective_up_A,
        effective_up_B,
    )
    expected_activation = F.silu(expected_gate_up[..., :128]) * expected_gate_up[..., 128:]
    expected = _literal_linear_value(
        expected_activation,
        down_base,
        effective_down_A,
        effective_down_B,
    )
    assert events == [
        ("gate_up", (3, 8)),
        ("activation", (3, 256)),
        ("down", (3, 128)),
    ]
    assert torch.equal(actual, expected)


def test_dense_mlp_runtime_rank_alpha_contract_is_atomic_and_fails_before_forward() -> None:
    rank_three = Glm52ExactTP1DenseMLP(8, 128, r=3, lora_alpha=7)
    assert rank_three.down_proj.lora_A.shape == (3, 128)
    assert rank_three.down_proj.lora_B.shape == (8, 3)
    with pytest.raises(ValueError, match="positive integer rank"):
        Glm52ExactTP1DenseMLP(8, 128, r=0, lora_alpha=1)

    module = _module()
    before = (
        module.active_r,
        module.active_lora_alpha,
        module.down_proj.active_r,
        module.down_proj.active_lora_alpha,
    )
    with pytest.raises(ValueError, match="positive integer alpha"):
        module.set_runtime_lora_config(1, 0)
    assert (
        module.active_r,
        module.active_lora_alpha,
        module.down_proj.active_r,
        module.down_proj.active_lora_alpha,
    ) == before

    input = torch.zeros(1, 8, dtype=torch.bfloat16)
    module.down_proj.active_lora_alpha = 2
    with pytest.raises(RuntimeError, match="one consistent adapter contract"):
        module(input)
    module.set_runtime_lora_config(1, 1)
    module.active_r = 2
    with pytest.raises(RuntimeError, match="one consistent adapter contract"):
        module(input)


def test_dense_mlp_roundtrips_six_canonical_factors_through_xorl_and_peft_export(tmp_path) -> None:
    source = _module()
    state = get_lora_state_dict(source)
    assert tuple(state) == source.logical_factor_names

    target = Glm52ExactTP1DenseMLP(8, 128, device=torch.device("cpu"))
    load_lora_state_dict(target, state)
    assert all(
        torch.equal(dict(target.named_parameters())[name], dict(source.named_parameters())[name])
        for name in source.logical_factor_names
    )

    checkpoint = tmp_path / "adapter"
    save_lora_checkpoint(source, str(checkpoint))
    exported = load_safetensors_file(str(checkpoint / "adapter_model.safetensors"))
    assert set(exported) == {f"base_model.model.{name}.weight" for name in source.logical_factor_names}
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    assert config["r"] == config["lora_alpha"] == 1
    assert set(config["target_modules"]) == {"gate_proj", "up_proj", "down_proj"}
    for name in source.logical_factor_names:
        assert torch.equal(
            exported[f"base_model.model.{name}.weight"],
            dict(source.named_parameters())[name].detach().to(torch.bfloat16),
        )
