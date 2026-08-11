import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

from xorl.lora.modules.linear import LoraLinear
from xorl.lora.utils import (
    LoraTensorShardSpec,
    convert_peft_lora_state_dict,
    get_lora_state_dict,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from xorl.models.layers.moe import MoEExpertsLoRA, MoELoRAConfig
from xorl.qlora.modules.moe_experts import NF4QLoRAMoeExperts
from xorl.server.runner.adapters.manager import LoRAAdapterManager


pytestmark = [pytest.mark.cpu, pytest.mark.server]

_TARGET_MODULES = ["q_proj", "gate_proj", "up_proj", "down_proj"]


class _TinyAttention(nn.Module):
    def __init__(self, r: int = 2, lora_alpha: int = 4):
        super().__init__()
        self.q_proj = LoraLinear.from_module(nn.Linear(8, 8, bias=False), r=r, lora_alpha=lora_alpha)


class _TinyLayer(nn.Module):
    def __init__(self, r: int = 2, lora_alpha: int = 4, hybrid_shared: bool = True):
        super().__init__()
        self.self_attn = _TinyAttention(r=r, lora_alpha=lora_alpha)
        self.mlp = nn.Module()
        self.mlp.experts = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=8,
            intermediate_size=16,
            moe_implementation="eager",
            lora_config=MoELoRAConfig(r=r, lora_alpha=lora_alpha, hybrid_shared=hybrid_shared),
        )


class _TinyInnerModel(nn.Module):
    def __init__(self, r: int = 2, lora_alpha: int = 4, hybrid_shared: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(r=r, lora_alpha=lora_alpha, hybrid_shared=hybrid_shared)])


class _TinyMoELoraModel(nn.Module):
    def __init__(self, r: int = 2, lora_alpha: int = 4, hybrid_shared: bool = True):
        super().__init__()
        self.model = _TinyInnerModel(r=r, lora_alpha=lora_alpha, hybrid_shared=hybrid_shared)


class _TinyQLoRALayer(nn.Module):
    def __init__(self, r: int = 2, lora_alpha: int = 4):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = NF4QLoRAMoeExperts(
            num_local_experts=4,
            num_experts=4,
            hidden_size=64,
            intermediate_size=64,
            r=r,
            lora_alpha=lora_alpha,
            device=torch.device("cpu"),
            moe_implementation="quack",
            hybrid_shared=True,
            target_modules=["gate_proj", "down_proj"],
        )


class _TinyExpertQLoRAModel(nn.Module):
    def __init__(self, r: int = 2, lora_alpha: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyQLoRALayer(r=r, lora_alpha=lora_alpha)])


def _iter_lora_parameters(module: nn.Module):
    for name, param in module.named_parameters():
        if "lora_" in name:
            yield name, param


def _assign_distinct_lora_values(module: nn.Module) -> None:
    with torch.no_grad():
        for offset, (_, param) in enumerate(_iter_lora_parameters(module), start=1):
            values = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) + offset
            param.copy_(values.to(param.dtype))


def _expected_saved_lora_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().to(torch.bfloat16).to(param.dtype).clone()
        for name, param in _iter_lora_parameters(module)
    }


def _actual_lora_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in _iter_lora_parameters(module)}


def test_convert_peft_moe_lora_slices_global_experts_for_ep_shard():
    prefix = "model.layers.0"
    for proj_name, lora_type, per_expert_shape in (
        ("down_proj", "A", (5, 2)),
        ("gate_proj", "B", (2, 5)),
    ):
        internal_name = f"{prefix}.mlp.experts.{proj_name}_lora_{lora_type}"
        global_tensor = torch.arange(8 * per_expert_shape[0] * per_expert_shape[1], dtype=torch.float32).reshape(
            8, *per_expert_shape
        )
        checkpoint_state = {
            f"base_model.model.{prefix}.mlp.experts.{expert_idx}.{proj_name}.lora_{lora_type}.weight": global_tensor[
                expert_idx
            ]
            .transpose(0, 1)
            .contiguous()
            for expert_idx in range(8)
        }

        converted = convert_peft_lora_state_dict(
            checkpoint_state,
            expected_shapes={internal_name: torch.Size((2, *per_expert_shape))},
            expected_shard_specs={internal_name: LoraTensorShardSpec(dim=0, index=2, size=4)},
        )

        assert set(converted) == {internal_name}
        assert torch.equal(converted[internal_name], global_tensor[4:6]), (proj_name, lora_type)


def _assert_runtime_rank_lora_export_slices_weights_and_config(tmp_path):
    source = _TinyMoELoraModel(r=4, lora_alpha=8)
    _assign_distinct_lora_values(source)
    for module in source.modules():
        setter = getattr(module, "set_runtime_lora_config", None)
        if callable(setter):
            setter(lora_rank=2, lora_alpha=6)

    state = get_lora_state_dict(source)
    assert state["model.layers.0.self_attn.q_proj.lora_A"].shape == (2, 8)
    assert state["model.layers.0.self_attn.q_proj.lora_B"].shape == (8, 2)
    assert state["model.layers.0.mlp.experts.gate_proj_lora_A"].shape == (1, 8, 2)
    assert state["model.layers.0.mlp.experts.gate_proj_lora_B"].shape == (4, 2, 16)
    assert state["model.layers.0.mlp.experts.down_proj_lora_A"].shape == (4, 16, 2)
    assert state["model.layers.0.mlp.experts.down_proj_lora_B"].shape == (1, 2, 8)

    checkpoint_dir = tmp_path / "checkpoint"
    save_lora_checkpoint(
        model=source,
        save_path=str(checkpoint_dir),
        target_modules=_TARGET_MODULES,
        r=4,
        lora_alpha=8,
        moe_hybrid_shared_lora=True,
    )

    weights = load_safetensors_file(str(checkpoint_dir / "adapter_model.safetensors"))
    cfg = json.loads((checkpoint_dir / "adapter_config.json").read_text())
    prefix = "base_model.model.model.layers.0"

    assert cfg["r"] == 2
    assert cfg["lora_alpha"] == 6
    assert weights[f"{prefix}.self_attn.q_proj.lora_A.weight"].shape == (2, 8)
    assert weights[f"{prefix}.self_attn.q_proj.lora_B.weight"].shape == (8, 2)
    assert weights[f"{prefix}.mlp.experts.shared.gate_proj.lora_A.weight"].shape == (2, 8)
    assert weights[f"{prefix}.mlp.experts.0.gate_proj.lora_B.weight"].shape == (16, 2)
    assert weights[f"{prefix}.mlp.experts.0.down_proj.lora_A.weight"].shape == (2, 16)
    assert weights[f"{prefix}.mlp.experts.shared.down_proj.lora_B.weight"].shape == (8, 2)


def test_unquantized_lora_checkpoint_export_and_roundtrip_policy(tmp_path):
    _assert_runtime_rank_lora_export_slices_weights_and_config(tmp_path / "runtime-rank")

    source = _TinyMoELoraModel()
    _assign_distinct_lora_values(source)

    checkpoint_dir = tmp_path / "checkpoint"
    save_lora_checkpoint(
        model=source,
        save_path=str(checkpoint_dir),
        moe_hybrid_shared_lora=True,
    )

    weights = load_safetensors_file(str(checkpoint_dir / "adapter_model.safetensors"))
    with open(checkpoint_dir / "adapter_config.json", "r") as f:
        adapter_config = json.load(f)

    moe = source.model.layers[0].mlp.experts

    gate_proj_shared_a = weights["base_model.model.model.layers.0.mlp.experts.shared.gate_proj.lora_A.weight"]
    up_proj_shared_a = weights["base_model.model.model.layers.0.mlp.experts.shared.up_proj.lora_A.weight"]
    gate_proj_expert_b = weights["base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight"]
    down_proj_expert_a = weights["base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_A.weight"]
    down_proj_shared_b = weights["base_model.model.model.layers.0.mlp.experts.shared.down_proj.lora_B.weight"]

    assert gate_proj_shared_a.shape == (2, 8)
    assert up_proj_shared_a.shape == (2, 8)
    assert gate_proj_expert_b.shape == (16, 2)
    assert down_proj_expert_a.shape == (2, 16)
    assert down_proj_shared_b.shape == (8, 2)

    assert torch.equal(
        gate_proj_shared_a,
        moe.gate_proj_lora_A.detach().cpu()[0].transpose(0, 1).contiguous().to(torch.bfloat16),
    )
    assert torch.equal(
        up_proj_shared_a,
        moe.up_proj_lora_A.detach().cpu()[0].transpose(0, 1).contiguous().to(torch.bfloat16),
    )
    assert torch.equal(
        gate_proj_expert_b,
        moe.gate_proj_lora_B.detach().cpu()[0].transpose(0, 1).contiguous().to(torch.bfloat16),
    )
    assert torch.equal(
        down_proj_expert_a,
        moe.down_proj_lora_A.detach().cpu()[0].transpose(0, 1).contiguous().to(torch.bfloat16),
    )
    assert torch.equal(
        down_proj_shared_b,
        moe.down_proj_lora_B.detach().cpu()[0].transpose(0, 1).contiguous().to(torch.bfloat16),
    )
    assert adapter_config["r"] == 2
    assert adapter_config["moe_hybrid_shared_lora"] is True

    loaded = _TinyMoELoraModel()
    load_lora_checkpoint(loaded, str(checkpoint_dir), strict=True)

    expected = _expected_saved_lora_state(source)
    actual = _actual_lora_state(loaded)

    assert set(actual) == set(expected)
    for name, expected_tensor in expected.items():
        assert torch.equal(actual[name], expected_tensor), name

    _assert_save_and_load_sglang_shared_outer_hybrid_shared_roundtrip(tmp_path / "sglang")
    _assert_adapter_manager_load_roundtrips_both_expert_ownership_modes(tmp_path / "manager")


def _assert_save_and_load_sglang_shared_outer_hybrid_shared_roundtrip(tmp_path):
    source = _TinyMoELoraModel()
    _assign_distinct_lora_values(source)

    checkpoint_dir = tmp_path / "checkpoint"
    save_lora_checkpoint(
        model=source,
        save_path=str(checkpoint_dir),
        target_modules=_TARGET_MODULES,
        r=2,
        lora_alpha=4,
        moe_hybrid_shared_lora=True,
        lora_export_format="sglang_shared_outer",
    )

    cfg = json.loads((checkpoint_dir / "adapter_config.json").read_text())
    assert cfg["_sglang_lora_format"] == "shared_outer"
    assert cfg["moe_hybrid_shared_lora"] is True
    tensors = load_safetensors_file(str(checkpoint_dir / "adapter_model.safetensors"))
    layer_prefix = "base_model.model.model.layers.0.mlp.experts"
    expected_shapes = {
        f"{layer_prefix}.w1.lora_A.weight": (1, 2, 8),
        f"{layer_prefix}.w1.lora_B.weight": (4, 16, 2),
        f"{layer_prefix}.w2.lora_A.weight": (4, 2, 16),
        f"{layer_prefix}.w2.lora_B.weight": (1, 8, 2),
        f"{layer_prefix}.w3.lora_A.weight": (1, 2, 8),
        f"{layer_prefix}.w3.lora_B.weight": (4, 16, 2),
    }
    assert {key for key in tensors if ".mlp.experts." in key} == set(expected_shapes)
    for key, shape in expected_shapes.items():
        assert tuple(tensors[key].shape) == shape, key

    loaded = _TinyMoELoraModel()
    load_lora_checkpoint(loaded, str(checkpoint_dir), strict=True)

    expected = _expected_saved_lora_state(source)
    actual = _actual_lora_state(loaded)

    assert set(actual) == set(expected)
    for name, expected_tensor in expected.items():
        assert torch.equal(actual[name], expected_tensor), name

    with pytest.raises(ValueError, match="moe_hybrid_shared_lora=True"):
        save_lora_checkpoint(
            model=source,
            save_path=str(tmp_path / "invalid-checkpoint"),
            target_modules=_TARGET_MODULES,
            r=2,
            lora_alpha=4,
            moe_hybrid_shared_lora=False,
            lora_export_format="sglang_shared_outer",
        )


def _assert_adapter_manager_load_roundtrips_both_expert_ownership_modes(tmp_path):
    for hybrid_shared, model_id in ((True, "hybrid-shared"), (False, "all-owner")):
        source = _TinyMoELoraModel(hybrid_shared=hybrid_shared)
        _assign_distinct_lora_values(source)
        checkpoint_dir = tmp_path / f"{model_id}-checkpoint"
        save_lora_checkpoint(
            model=source,
            save_path=str(checkpoint_dir),
            target_modules=_TARGET_MODULES,
            r=2,
            lora_alpha=4,
            moe_hybrid_shared_lora=hybrid_shared,
        )

        manager = LoRAAdapterManager(
            model=_TinyMoELoraModel(hybrid_shared=hybrid_shared),
            device=torch.device("cpu"),
            checkpoint_dir=str(tmp_path / f"{model_id}-adapters"),
            auto_save_on_eviction=False,
            lora_config={"moe_hybrid_shared_lora": hybrid_shared},
        )
        result = manager.load_adapter_state(
            model_id=model_id,
            path=str(checkpoint_dir),
            load_optimizer=False,
            lr=1e-4,
        )

        actual = {
            name: parameter.detach().cpu().clone()
            for name, parameter in manager.get_adapter_state(model_id).local_params.items()
        }
        expected = _expected_saved_lora_state(source)
        assert result["model_id"] == model_id
        assert set(actual) == set(expected)
        for name, expected_tensor in expected.items():
            assert torch.equal(actual[name], expected_tensor), name


def test_quantized_expert_projection_subset_checkpoint_roundtrip(tmp_path):
    source = _TinyExpertQLoRAModel()
    _assign_distinct_lora_values(source)
    checkpoint_dir = tmp_path / "quantized-expert-checkpoint"

    save_lora_checkpoint(
        model=source,
        save_path=str(checkpoint_dir),
        target_modules=["gate_proj", "down_proj"],
        r=2,
        lora_alpha=4,
        moe_hybrid_shared_lora=True,
    )

    loaded = _TinyExpertQLoRAModel()
    load_lora_checkpoint(loaded, str(checkpoint_dir), strict=True)

    expected = _expected_saved_lora_state(source)
    actual = _actual_lora_state(loaded)
    assert set(actual) == set(expected)
    assert all("up_proj_lora" not in name for name in actual)
    for name, expected_tensor in expected.items():
        assert torch.equal(actual[name], expected_tensor), name
