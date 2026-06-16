"""Tests for the fp32 cast-once merge variant on both LoraLinear and MoEExpertsLoRA.

Invariants:
  - With zero LoRA (B=0), merge must be bit-exact: W_merged == W (no change).
  - After merge, ``merged_weight`` equals the fp32 reference
    ``(W.to(fp32) + B@A*s).to(W.dtype)`` bit-for-bit (by construction).
  - Merged weight is >= as faithful as the naive ``W + Δ.to(W.dtype)`` variant —
    i.e., the fp32-sum-then-cast distance to the true fp32 merged value is ≤
    the naive-merge distance.
"""

import pytest
import torch


pytestmark = [pytest.mark.gpu]


def _naive_merge(weight, delta):
    """Old behavior for comparison: round Δ per-element, then add."""
    return weight + delta.to(weight.dtype)


def _fp32_merge(weight, delta):
    """New behavior: add in fp32, cast once."""
    return (weight.to(torch.float32) + delta).to(weight.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_lora_linear_merge_zero_b_is_bitexact(dtype):
    """merge_weights() on a fresh LoraLinear (B=0) must leave weight untouched."""
    from xorl.lora import LoraLinear

    torch.manual_seed(0)
    layer = LoraLinear(128, 64, r=8, lora_alpha=16, device="cuda", dtype=dtype)
    layer.weight.data.normal_(std=0.05)
    before = layer.weight.detach().clone()
    layer.merge_weights()
    assert torch.equal(layer.weight, before), "zero-LoRA merge must be bit-exact"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lora_linear_merge_matches_fp32_reference(dtype):
    """The merged weight must equal ``(W.to(fp32) + B@A*s).to(W.dtype)`` exactly."""
    from xorl.lora import LoraLinear

    torch.manual_seed(0)
    layer = LoraLinear(128, 64, r=8, lora_alpha=16, device="cuda", dtype=dtype)
    layer.weight.data.normal_(std=0.05)
    # non-zero LoRA
    layer.lora_B.data.normal_(std=0.02)
    w_before = layer.weight.detach().clone()
    delta_fp32 = (layer.lora_B @ layer.lora_A) * layer.scaling
    expected = _fp32_merge(w_before, delta_fp32)

    layer.merge_weights()
    assert torch.equal(layer.weight, expected), "merge_weights must match fp32-cast-once reference"


def test_lora_linear_merge_strictly_ge_naive_precision():
    """For any B, fp32-cast-once ≤ naive in distance to true fp32 merged value."""
    from xorl.lora import LoraLinear

    torch.manual_seed(42)
    layer = LoraLinear(128, 64, r=8, lora_alpha=16, device="cuda", dtype=torch.bfloat16)
    layer.weight.data.normal_(std=0.05)
    layer.lora_B.data.normal_(std=0.02)

    w = layer.weight.detach().clone()
    delta_fp32 = (layer.lora_B @ layer.lora_A) * layer.scaling
    true_fp32 = w.to(torch.float32) + delta_fp32  # reference in fp32
    naive = _naive_merge(w, delta_fp32).to(torch.float32)  # rounds Δ then adds
    fp32_cast_once = _fp32_merge(w, delta_fp32).to(torch.float32)  # fp32 sum, cast once

    naive_err = (naive - true_fp32).abs().max().item()
    fp32_err = (fp32_cast_once - true_fp32).abs().max().item()
    assert fp32_err <= naive_err + 1e-9, (
        f"fp32-cast-once should be ≤ naive precision: fp32={fp32_err:.3e}  naive={naive_err:.3e}"
    )


def _tiny_moe_experts_with_lora(dtype):
    from xorl.lora import MoEExpertsLoRA, MoELoRAConfig

    cfg = MoELoRAConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])
    e = (
        MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=24,
            hidden_act="silu",
            moe_implementation="eager",
            lora_config=cfg,
        )
        .to(dtype)
        .cuda()
    )
    e.gate_up_proj.data.normal_(std=0.05)
    e.down_proj.data.normal_(std=0.05)
    return e


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_merge_zero_b_is_bitexact(dtype):
    e = _tiny_moe_experts_with_lora(dtype)
    gu_before = e.gate_up_proj.detach().clone()
    dn_before = e.down_proj.detach().clone()
    e.merge_weights()
    assert torch.equal(e.gate_up_proj, gu_before), "zero-LoRA MoE merge must not change gate_up_proj"
    assert torch.equal(e.down_proj, dn_before), "zero-LoRA MoE merge must not change down_proj"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_merge_matches_fp32_reference(dtype):
    e = _tiny_moe_experts_with_lora(dtype)
    # perturb all lora_B
    torch.manual_seed(7)
    for name in ("gate_proj", "up_proj", "down_proj"):
        getattr(e, f"{name}_lora_B").data.normal_(std=0.02)

    # expected = fp32 sum then cast, computed from SNAPSHOTS of current state
    gu_before = e.gate_up_proj.detach().clone()
    dn_before = e.down_proj.detach().clone()
    expected_updates = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        delta = e._compute_proj_delta(proj)  # fp32, [E, in, out]
        expected_updates[proj] = delta

    # gate/up land in the fused gate_up_proj via views, each shape (E, H, I)
    I = e.intermediate_size
    gate_expected = gu_before.clone()
    gate_expected[..., :I] = _fp32_merge(gu_before[..., :I], expected_updates["gate_proj"])
    gate_expected[..., I:] = _fp32_merge(gu_before[..., I:], expected_updates["up_proj"])

    down_expected = _fp32_merge(dn_before, expected_updates["down_proj"])

    e.merge_weights()

    assert torch.equal(e.gate_up_proj, gate_expected), "MoE gate+up merge must match fp32 reference"
    assert torch.equal(e.down_proj, down_expected), "MoE down merge must match fp32 reference"
