import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xorl.lora.modules.linear import LoraLinear
from xorl.lora.utils import load_lora_checkpoint, save_lora_checkpoint
from xorl.models.layers.moe import MoEExpertsLoRA, MoELoRAConfig


pytestmark = [pytest.mark.cpu, pytest.mark.server]

_TARGET_MODULES = ["q_proj", "gate_proj", "up_proj", "down_proj"]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "xorl_server_runner_adapters_manager",
    _REPO_ROOT / "src/xorl/server/runner/adapters/manager.py",
)
assert _MANAGER_SPEC is not None and _MANAGER_SPEC.loader is not None
_MANAGER_MODULE = importlib.util.module_from_spec(_MANAGER_SPEC)
_MANAGER_SPEC.loader.exec_module(_MANAGER_MODULE)
LoRAAdapterManager = _MANAGER_MODULE.LoRAAdapterManager


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = LoraLinear.from_module(nn.Linear(8, 8, bias=False), r=2, lora_alpha=4)


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttention()
        self.mlp = nn.Module()
        self.mlp.experts = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=8,
            intermediate_size=16,
            moe_implementation="eager",
            lora_config=MoELoRAConfig(r=2, lora_alpha=4, hybrid_shared=True),
        )


class _TinyInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer()])


class _TinyMoELoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyInnerModel()


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


def test_load_lora_checkpoint_roundtrip_supports_hybrid_shared(tmp_path):
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
    )

    loaded = _TinyMoELoraModel()
    load_lora_checkpoint(loaded, str(checkpoint_dir), strict=True)

    expected = _expected_saved_lora_state(source)
    actual = _actual_lora_state(loaded)

    assert set(actual) == set(expected)
    for name, expected_tensor in expected.items():
        assert torch.equal(actual[name], expected_tensor), name


def test_adapter_manager_load_adapter_state_roundtrip_supports_hybrid_shared(tmp_path):
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
    )

    manager = LoRAAdapterManager(
        model=_TinyMoELoraModel(),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        lora_config={"moe_hybrid_shared_lora": True},
    )

    result = manager.load_adapter_state(
        model_id="adapter-1",
        path=str(checkpoint_dir),
        load_optimizer=False,
        lr=1e-4,
    )

    state = manager.get_adapter_state("adapter-1")
    actual = {name: param.detach().cpu().clone() for name, param in state.lora_params.items()}
    expected = _expected_saved_lora_state(source)

    assert result["model_id"] == "adapter-1"
    assert set(actual) == set(expected)
    for name, expected_tensor in expected.items():
        assert torch.equal(actual[name], expected_tensor), name
