from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from xorl.lora.modules.delta_linear import LoraDeltaLinear
from xorl.lora.utils import _get_default_target_modules, inject_lora_into_model
from xorl.models.layers.fused_projection_lora import project_fused_linear_with_lora
from xorl.models.transformers.glm4_moe.modeling_glm4_moe import Glm4MoeMLP
from xorl.models.transformers.llama3.modeling_llama3 import LlamaMLP
from xorl.models.transformers.olmo2.modeling_olmo2 import Olmo2MLP
from xorl.models.transformers.qwen2.modeling_qwen2 import Qwen2MLP
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3MLP
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from xorl.server.weight_sync.handler import WeightSyncHandler


pytestmark = [pytest.mark.cpu]


class _AuditedFusedBlock(nn.Module):
    _supports_fused_qkv_lora = True
    _supports_fused_gate_up_lora = True

    def __init__(self) -> None:
        super().__init__()
        self.q_dim = 6
        self.kv_dim = 2
        self.intermediate_size = 5
        self.qkv_proj = nn.Linear(4, self.q_dim + 2 * self.kv_dim, bias=False)
        self.gate_up_proj = nn.Linear(4, 2 * self.intermediate_size, bias=False)

    def qkv(self, inputs: torch.Tensor) -> torch.Tensor:
        return project_fused_linear_with_lora(
            self,
            inputs,
            base_name="qkv_proj",
            projection_names=("q_proj", "k_proj", "v_proj"),
            projection_sizes=(self.q_dim, self.kv_dim, self.kv_dim),
        )

    def gate_up(self, inputs: torch.Tensor) -> torch.Tensor:
        return project_fused_linear_with_lora(
            self,
            inputs,
            base_name="gate_up_proj",
            projection_names=("gate_proj", "up_proj"),
            projection_sizes=(self.intermediate_size, self.intermediate_size),
        )


class _AuditedModel(nn.Module):
    def __init__(self, model_type: str = "qwen3") -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.block = _AuditedFusedBlock()


def _seed_nonzero_adapters(model: nn.Module) -> None:
    with torch.no_grad():
        for index, module in enumerate(
            (child for child in model.modules() if isinstance(child, LoraDeltaLinear)),
            start=1,
        ):
            module.lora_A.copy_(torch.arange(module.lora_A.numel()).reshape_as(module.lora_A) / (10 + index))
            module.lora_B.copy_(torch.arange(1, module.lora_B.numel() + 1).reshape_as(module.lora_B) / (20 + index))


def test_split_targets_keep_fused_base_projections_and_independent_factors() -> None:
    model = _AuditedModel()
    inputs = torch.randn(3, 4)
    qkv_base = model.block.qkv_proj(inputs)
    gate_up_base = model.block.gate_up_proj(inputs)

    inject_lora_into_model(
        model,
        r=2,
        lora_alpha=2,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
    )

    assert isinstance(model.block.qkv_proj, nn.Linear)
    assert isinstance(model.block.gate_up_proj, nn.Linear)
    assert all(
        isinstance(getattr(model.block, name), LoraDeltaLinear)
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    )
    assert torch.equal(model.block.qkv(inputs), qkv_base)
    assert torch.equal(model.block.gate_up(inputs), gate_up_base)


@pytest.mark.parametrize(
    "mlp_class",
    [Qwen2MLP, Qwen3MLP, Qwen3MoeMLP, Qwen3_5MLP, Qwen3_5MoeMLP, LlamaMLP, Olmo2MLP, Glm4MoeMLP],
)
def test_audited_fused_mlp_implementations_retain_base_program(mlp_class: type[nn.Module]) -> None:
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=6,
        hidden_act="gelu",
        mlp_bias=False,
        _activation_native=True,
    )
    mlp = mlp_class(config)
    inputs = torch.randn(2, 3, config.hidden_size)
    expected = mlp(inputs)
    inject_lora_into_model(mlp, r=2, lora_alpha=2, target_modules=["gate_proj", "up_proj", "down_proj"])

    assert isinstance(mlp.gate_up_proj, nn.Linear)
    assert isinstance(mlp.gate_proj, LoraDeltaLinear)
    assert isinstance(mlp.up_proj, LoraDeltaLinear)
    assert torch.equal(mlp(inputs), expected)


@pytest.mark.parametrize(
    ("method", "base_name", "projection_names", "sizes"),
    [
        ("qkv", "qkv_proj", ("q_proj", "k_proj", "v_proj"), (6, 2, 2)),
        ("gate_up", "gate_up_proj", ("gate_proj", "up_proj"), (5, 5)),
    ],
)
def test_fused_projection_dynamic_and_exact_merged_programs(
    method: str,
    base_name: str,
    projection_names: tuple[str, ...],
    sizes: tuple[int, ...],
) -> None:
    model = _AuditedModel()
    inject_lora_into_model(model, r=2, lora_alpha=2, target_modules=list(projection_names))
    _seed_nonzero_adapters(model)
    inputs = torch.randn(2, 3, 4)
    block = model.block
    base = getattr(block, base_name)

    base_parts = list(base(inputs).split(sizes, dim=-1))
    for index, name in enumerate(projection_names):
        base_parts[index] = base_parts[index] + getattr(block, name)(inputs)
    expected_dynamic = torch.cat(base_parts, dim=-1)
    assert torch.equal(getattr(block, method)(inputs), expected_dynamic)

    folded_parts = []
    for base_part, name in zip(base.weight.split(sizes, dim=0), projection_names, strict=True):
        adapter = getattr(block, name)
        adapter.exact_merged_forward = True
        folded_parts.append(adapter.merged_weight_for_forward(base_part))
    expected_merged = F.linear(inputs, torch.cat(folded_parts, dim=0))
    assert torch.equal(getattr(block, method)(inputs), expected_merged)


def test_weight_sync_folds_logical_factors_into_fused_base_weights() -> None:
    model = _AuditedModel()
    inject_lora_into_model(
        model,
        r=2,
        lora_alpha=2,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
    )
    _seed_nonzero_adapters(model)
    for module in model.modules():
        if isinstance(module, LoraDeltaLinear):
            module.exact_merged_forward = True

    class _FakeDTensor:
        pass

    extracted = dict(WeightSyncHandler._extract_params_for_sync(model, "(root)", _FakeDTensor))
    for base_name, projection_names, sizes in (
        ("qkv_proj", ("q_proj", "k_proj", "v_proj"), (6, 2, 2)),
        ("gate_up_proj", ("gate_proj", "up_proj"), (5, 5)),
    ):
        base = getattr(model.block, base_name)
        expected = torch.cat(
            [
                getattr(model.block, name)._merged_weight(base_part).to(torch.bfloat16)
                for name, base_part in zip(projection_names, base.weight.split(sizes, dim=0), strict=True)
            ],
            dim=0,
        )
        assert torch.equal(extracted[f"block.{base_name}.weight"], expected)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("xorl_llama3", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ("qwen2", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ("xorl_qwen3", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ("qwen3_moe", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        (
            "xorl_qwen3_5_moe",
            ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        ("olmo2", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ("glm4_moe", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        (
            "deepseek_v3",
            [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        (
            "kimi_k25",
            [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        (
            "xorl_glm5",
            [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        ("deepseek_v4", ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]),
        ("gpt_oss", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("minimax_m3", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("nemotron_h", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ],
)
def test_audited_model_family_defaults(model_type: str, expected: list[str]) -> None:
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    assert _get_default_target_modules(model) == expected


def test_unknown_model_family_requires_explicit_targets() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(model_type="new_unreviewed_architecture")
    with pytest.raises(ValueError, match="No audited default LoRA targets"):
        _get_default_target_modules(model)


def test_unmatched_projection_target_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing_proj"):
        inject_lora_into_model(_AuditedModel(), target_modules=["q_proj", "missing_proj"])


def test_checked_in_plain_lora_configs_cover_audited_projection_sets() -> None:
    root = Path(__file__).resolve().parents[2]
    config_paths = sorted((root / "examples").glob("**/configs/lora/*.yaml"))
    assert config_paths
    split_qwen_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    for path in config_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = payload.get("model", payload)
        lora = payload.get("lora", payload)
        model_path = model["model_path"].lower()
        targets = lora["lora_target_modules"]
        assert len(targets) == len(set(targets)), path

        if "qwen3.5" in model_path:
            assert split_qwen_targets | {"g_proj"} <= set(targets), path
        elif "qwen3" in model_path:
            assert split_qwen_targets <= set(targets), path
        elif "llama" in model_path:
            assert {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"} <= set(targets), path
