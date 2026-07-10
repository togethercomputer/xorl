"""Routing-weight position (before vs after the down GEMM) in ``TritonEPGroupGemm``.

Covers the ``moe_routing_weights_before_down`` config knob and its
``XORL_MOE_ROUTING_WEIGHTS_BEFORE_DOWN`` env override:

- Gradient parity of BOTH positions against an fp64 eager reference. The two
  positions are mathematically identical (a per-row scalar commutes through the
  linear down projection) but are NOT bit-equal to each other (different bf16/fp32
  rounding points); the gate is that both sit in the same error class vs fp64.
- The lazily-read knob API (config setter + env force-on).
- The regime-aware ``auto`` default: resolves on only for ``train_router=True`` +
  alltoall dispatch with the ``XORL_MOE_SGLANG_FUSED_EXPERTS`` parity opt-in
  inactive; explicit true/false override the regime check.
"""

import pytest
import torch
import torch.nn.functional as F

import xorl.ops.moe.triton as moe_triton
from tests.ops.test_ep_routing_scores import _counts_from_cumsum, _patch_ep_kernels
from xorl.arguments import ModelArguments


pytestmark = pytest.mark.cpu


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    num_local_experts = 3
    hidden_dim = 16
    intermediate_size = 12
    counts = torch.tensor([4, 1, 3])
    cumsum = torch.cumsum(counts, dim=0)
    num_tokens = int(cumsum[-1].item())

    permute_tokens = torch.randn(num_tokens, hidden_dim)
    gate_up_proj = torch.randn(num_local_experts, hidden_dim, 2 * intermediate_size) * hidden_dim**-0.5
    down_proj = torch.randn(num_local_experts, intermediate_size, hidden_dim) * intermediate_size**-0.5
    expert_scores = torch.rand(num_tokens) * 0.5 + 0.01
    upstream = torch.randn(num_tokens, hidden_dim)
    return permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores, upstream


def _fp64_reference(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores, upstream):
    x = permute_tokens.detach().double().requires_grad_(True)
    gup = gate_up_proj.detach().double().requires_grad_(True)
    down = down_proj.detach().double().requires_grad_(True)
    scores = expert_scores.detach().double().requires_grad_(True)

    outputs = []
    start = 0
    for expert_idx, count in enumerate(_counts_from_cumsum(cumsum)):
        end = start + count
        xs = x[start:end]
        gate_up = xs @ gup[expert_idx]
        h = F.silu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
        h = h * scores[start:end].unsqueeze(-1)
        outputs.append(h @ down[expert_idx])
        start = end
    out = torch.cat(outputs, dim=0)
    out.backward(upstream.double())
    return out.detach(), {"dX": x.grad, "d_gate_up": gup.grad, "d_down": down.grad, "d_expert_scores": scores.grad}


def _run_position(module, before_down, inputs, scores_require_grad=True):
    permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores, upstream = inputs
    module.set_routing_weights_before_down(before_down)
    x = permute_tokens.detach().clone().requires_grad_(True)
    gup = gate_up_proj.detach().clone().requires_grad_(True)
    down = down_proj.detach().clone().requires_grad_(True)
    scores = expert_scores.detach().clone().requires_grad_(scores_require_grad)

    out = module.TritonEPGroupGemm.apply(x, cumsum, gup, down, intermediate_size, scores, "silu", 0.0, True, 0)
    out.backward(upstream)
    grads = {"dX": x.grad, "d_gate_up": gup.grad, "d_down": down.grad}
    if scores_require_grad:
        grads["d_expert_scores"] = scores.grad
    return out.detach(), grads


def test_both_routing_positions_match_fp64_reference(monkeypatch):
    module = _patch_ep_kernels(monkeypatch, "xorl.ops.moe.triton")
    inputs = _make_inputs()
    ref_out, ref_grads = _fp64_reference(*inputs)

    per_position = {}
    for before_down in (False, True):
        out, grads = _run_position(module, before_down, inputs)
        torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-4)
        errs = {}
        for key, ref in ref_grads.items():
            got = grads[key].double()
            torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-4)
            errs[key] = (got - ref).abs().mean().item()
        per_position[before_down] = errs

    # Same error class: neither position systematically worse vs fp64.
    for key in ref_grads:
        ea = max(per_position[False][key], 1e-30)
        eb = max(per_position[True][key], 1e-30)
        assert max(ea / eb, eb / ea) < 3.0, f"{key}: error class diverged (after={ea:.3e}, before={eb:.3e})"


def test_before_down_without_score_grad_matches_fp64_reference(monkeypatch):
    """The in-place score fold on the recomputed intermediate (no router grad) is safe."""
    module = _patch_ep_kernels(monkeypatch, "xorl.ops.moe.triton")
    inputs = _make_inputs(seed=7)
    _, ref_grads = _fp64_reference(*inputs)

    _, grads = _run_position(module, True, inputs, scores_require_grad=False)
    for key in ("dX", "d_gate_up", "d_down"):
        torch.testing.assert_close(grads[key].double(), ref_grads[key], rtol=1e-3, atol=1e-4)


def test_routing_weight_position_knob(monkeypatch):
    monkeypatch.delenv("XORL_MOE_ROUTING_WEIGHTS_BEFORE_DOWN", raising=False)
    monkeypatch.setattr(moe_triton, "_ROUTING_WEIGHTS_BEFORE_DOWN_CONFIG", False)
    assert moe_triton.routing_weights_before_down() is False

    moe_triton.set_routing_weights_before_down(True)
    assert moe_triton.routing_weights_before_down() is True
    moe_triton.set_routing_weights_before_down(False)
    assert moe_triton.routing_weights_before_down() is False

    # Env var force-enables regardless of the config default (read lazily, so it
    # keeps working when set after import).
    monkeypatch.setenv("XORL_MOE_ROUTING_WEIGHTS_BEFORE_DOWN", "1")
    assert moe_triton.routing_weights_before_down() is True


def test_model_arguments_field_default():
    args = ModelArguments(model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    assert args.moe_routing_weights_before_down == "auto"


@pytest.mark.parametrize(
    ("train_router", "ep_dispatch", "expected"),
    [
        (True, "alltoall", True),  # the measured-win regime
        (True, "deepep", False),
        (False, "alltoall", False),
        (False, "deepep", False),
    ],
)
def test_auto_resolution_regimes(monkeypatch, train_router, ep_dispatch, expected):
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS", raising=False)
    resolved = moe_triton.resolve_routing_weights_before_down(
        "auto", train_router=train_router, ep_dispatch=ep_dispatch
    )
    assert resolved is expected


def test_auto_resolution_disabled_under_parity_opt_in(monkeypatch):
    """The XORL_MOE_SGLANG_FUSED_EXPERTS parity lane keeps the historical after-down tree."""
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS", "1")
    assert moe_triton.resolve_routing_weights_before_down("auto", train_router=True, ep_dispatch="alltoall") is False


@pytest.mark.parametrize("setting", [True, "true", "1"])
def test_explicit_true_overrides_regime(monkeypatch, setting):
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS", "1")
    assert moe_triton.resolve_routing_weights_before_down(setting, train_router=False, ep_dispatch="deepep") is True


@pytest.mark.parametrize("setting", [False, "false", "0"])
def test_explicit_false_overrides_regime(monkeypatch, setting):
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS", raising=False)
    assert moe_triton.resolve_routing_weights_before_down(setting, train_router=True, ep_dispatch="alltoall") is False


def test_invalid_setting_raises():
    with pytest.raises(ValueError, match="moe_routing_weights_before_down"):
        moe_triton.resolve_routing_weights_before_down("maybe", train_router=True, ep_dispatch="alltoall")
