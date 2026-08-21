from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from xorl.lora.fold import canonical_lora_fold_linear
from xorl.lora.modules.base import LoraModule
from xorl.lora.modules.delta_linear import LoraDeltaLinear
from xorl.lora.modules.linear import LoraLinear
from xorl.lora.target_manifest import collect_lora_runtime_modules
from xorl.lora.utils import (
    _get_default_target_modules,
    inject_lora_into_model,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from xorl.models.layers.gated_deltanet import GatedDeltaNet
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
from xorl.ops.batch_invariant_ops import set_trunk_linear_contract, wrap_trunk_linears_batch_invariant


RANK = 16


def _mlp_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=6,
        hidden_act="silu",
        _activation_native=True,
    )


class _Layer(nn.Module):
    def __init__(self, *, with_gdn: bool = True) -> None:
        super().__init__()
        if with_gdn:
            self.linear_attn = GatedDeltaNet(
                hidden_size=8,
                expand_v=1,
                head_dim=2,
                num_heads=2,
                num_v_heads=2,
                use_short_conv=False,
                layer_idx=0,
            )
        self.mlp = nn.Module()
        self.mlp.shared_expert = Qwen3_5MoeMLP(_mlp_config())


class _Model(nn.Module):
    def __init__(self, layers: int = 1, *, with_gdn: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="xorl_qwen3_5_moe")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(with_gdn=with_gdn) for _ in range(layers)])


def _projection_targets() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "g_proj", "gate_proj", "up_proj", "down_proj"]


def test_qwen35_gdn_and_shared_expert_projection_topology_and_gradients(tmp_path) -> None:
    torch.manual_seed(7)
    model = _Model()
    assert _get_default_target_modules(model) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "g_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    inject_lora_into_model(model, r=RANK, lora_alpha=RANK, target_modules=_projection_targets())

    gdn = model.model.layers[0].linear_attn
    assert not hasattr(gdn, "in_proj_qkvz")
    gdn_inputs = (gdn.q_proj, gdn.k_proj, gdn.v_proj, gdn.g_proj)
    assert all(isinstance(module, LoraLinear) and module.r == RANK for module in gdn_inputs)
    assert len({module.lora_A.data_ptr() for module in gdn_inputs}) == 4

    shared = model.model.layers[0].mlp.shared_expert
    assert isinstance(shared.gate_proj, LoraDeltaLinear)
    assert isinstance(shared.up_proj, LoraDeltaLinear)
    assert isinstance(shared.down_proj, LoraLinear)
    assert all(module.r == RANK for module in (shared.gate_proj, shared.up_proj, shared.down_proj))

    with torch.no_grad():
        for module in (*gdn_inputs, shared.gate_proj, shared.up_proj, shared.down_proj):
            module.lora_B.normal_()
            module.exact_merged_forward = True

    gate_base, up_base = shared.gate_up_proj.weight.chunk(2, dim=0)
    gate_weight = canonical_lora_fold_linear(gate_base, shared.gate_proj.lora_A, shared.gate_proj.lora_B, 1.0)
    up_weight = canonical_lora_fold_linear(up_base, shared.up_proj.lora_A, shared.up_proj.lora_B, 1.0)
    down_weight = canonical_lora_fold_linear(
        shared.down_proj.weight,
        shared.down_proj.lora_A,
        shared.down_proj.lora_B,
        1.0,
    )
    inputs = torch.randn(2, 3, 8)
    gate, up = F.linear(inputs, torch.cat((gate_weight, up_weight), dim=0)).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, down_weight)
    actual = shared(inputs)
    assert torch.equal(actual, expected)

    actual.sum().backward()
    for module in (shared.gate_proj, shared.up_proj, shared.down_proj):
        assert module.lora_A.grad is not None and torch.count_nonzero(module.lora_A.grad) > 0
        assert module.lora_B.grad is not None and torch.count_nonzero(module.lora_B.grad) > 0

    save_lora_checkpoint(
        model,
        str(tmp_path),
        base_model_name="Qwen/Qwen3.6-35B-A3B",
        r=RANK,
        lora_alpha=RANK,
        preserve_lora_dtype=True,
    )
    restored = _Model()
    inject_lora_into_model(restored, r=RANK, lora_alpha=RANK, target_modules=_projection_targets())
    load_lora_checkpoint(restored, str(tmp_path), strict=True)
    expected_state = {name: value for name, value in model.state_dict().items() if "lora_" in name}
    actual_state = {name: value for name, value in restored.state_dict().items() if "lora_" in name}
    assert actual_state.keys() == expected_state.keys()
    for name, expected_value in expected_state.items():
        assert torch.equal(actual_state[name], expected_value)


def test_qwen36_forty_layer_shared_expert_inventory() -> None:
    model = _Model(layers=40, with_gdn=False)
    inject_lora_into_model(
        model,
        r=RANK,
        lora_alpha=RANK,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    inventory = collect_lora_runtime_modules(model)
    assert len(inventory) == 40 * 3
    for layer_idx in range(40):
        prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
        assert inventory[f"{prefix}.gate_proj"] == RANK
        assert inventory[f"{prefix}.up_proj"] == RANK
        assert inventory[f"{prefix}.down_proj"] == RANK


def test_qwen_shared_expert_separate_factors_compose_with_exact_trunk_wrap() -> None:
    model = _Model()
    inject_lora_into_model(model, r=RANK, lora_alpha=RANK, target_modules=_projection_targets())
    for module in model.modules():
        if isinstance(module, LoraModule):
            module.exact_merged_forward = True
    try:
        wrapped = wrap_trunk_linears_batch_invariant(model)
        shared = model.model.layers[0].mlp.shared_expert
        assert wrapped["gate_up_proj"] == 1
        assert getattr(shared.gate_up_proj, "_xorl_bi_trunk_wrapped", False)
        assert getattr(shared.down_proj, "_xorl_bi_trunk_wrapped", False)
        assert not getattr(shared.gate_proj, "_xorl_bi_trunk_wrapped", False)
        assert not getattr(shared.up_proj, "_xorl_bi_trunk_wrapped", False)
    finally:
        set_trunk_linear_contract(False)


def test_qwen_shared_expert_supports_independent_fused_gate_up_adapters() -> None:
    for target in ("gate_proj", "up_proj"):
        model = _Model(with_gdn=False)
        inject_lora_into_model(model, r=RANK, lora_alpha=RANK, target_modules=[target])
        shared = model.model.layers[0].mlp.shared_expert
        assert isinstance(getattr(shared, target), LoraDeltaLinear)
        other = "up_proj" if target == "gate_proj" else "gate_proj"
        assert not hasattr(shared, other)
