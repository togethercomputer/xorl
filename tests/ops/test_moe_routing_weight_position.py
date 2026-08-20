"""Routing-weight position (before vs after the down GEMM) in ``TritonEPGroupGemm``.

Covers the ``moe_routing_weights_before_down`` config knob:

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
from tests._helpers.moe import counts_from_cumsum, patch_ep_kernels


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
    for expert_idx, count in enumerate(counts_from_cumsum(cumsum)):
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


def test_routing_weight_position_numerical_contract(monkeypatch):
    patch_ep_kernels(monkeypatch, moe_triton)
    module = moe_triton
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

    _assert_before_down_without_score_grad_matches_reference(monkeypatch)
    with monkeypatch.context() as config_patch:
        _assert_routing_weight_position_configuration_policy(config_patch)


def _assert_before_down_without_score_grad_matches_reference(monkeypatch):
    """The in-place score fold on the recomputed intermediate (no router grad) is safe."""
    patch_ep_kernels(monkeypatch, moe_triton)
    module = moe_triton
    inputs = _make_inputs(seed=7)
    _, ref_grads = _fp64_reference(*inputs)

    _, grads = _run_position(module, True, inputs, scores_require_grad=False)
    for key in ("dX", "d_gate_up", "d_down"):
        torch.testing.assert_close(grads[key].double(), ref_grads[key], rtol=1e-3, atol=1e-4)


def _assert_routing_weight_position_configuration_policy(monkeypatch):
    monkeypatch.setattr(moe_triton, "_ROUTING_WEIGHTS_BEFORE_DOWN_CONFIG", False)
    assert moe_triton.routing_weights_before_down() is False

    moe_triton.set_routing_weights_before_down(True)
    assert moe_triton.routing_weights_before_down() is True
    moe_triton.set_routing_weights_before_down(False)
    assert moe_triton.routing_weights_before_down() is False

    _assert_auto_resolution_regimes(monkeypatch)
    _assert_auto_resolution_disabled_under_parity_opt_in(monkeypatch)
    _assert_explicit_true_overrides_regime(monkeypatch)
    _assert_explicit_false_overrides_regime(monkeypatch)
    _assert_invalid_setting_raises()


def _assert_auto_resolution_regimes(monkeypatch):
    # Pin the stock expert tree. Unset is an auto mode that enables the serving
    # kernel on supported EP1 CUDA lanes, so it is not equivalent to opt-out.
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS", "0")
    for train_router, ep_dispatch, expected in (
        (True, "alltoall", True),  # the measured-win regime
        (True, "deepep", False),
        (False, "alltoall", False),
        (False, "deepep", False),
    ):
        resolved = moe_triton.resolve_routing_weights_before_down(
            "auto", train_router=train_router, ep_dispatch=ep_dispatch
        )
        assert resolved is expected


def _assert_auto_resolution_disabled_under_parity_opt_in(monkeypatch):
    """The XORL_MOE_SGLANG_FUSED_EXPERTS parity lane keeps the historical after-down tree."""
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS", "1")
    assert moe_triton.resolve_routing_weights_before_down("auto", train_router=True, ep_dispatch="alltoall") is False


def _assert_explicit_true_overrides_regime(monkeypatch):
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS", "1")
    for setting in (True, "true"):
        assert moe_triton.resolve_routing_weights_before_down(setting, train_router=False, ep_dispatch="deepep") is True


def _assert_explicit_false_overrides_regime(monkeypatch):
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS", raising=False)
    for setting in (False, "false"):
        assert (
            moe_triton.resolve_routing_weights_before_down(setting, train_router=True, ep_dispatch="alltoall") is False
        )


def _assert_invalid_setting_raises():
    with pytest.raises(ValueError, match="moe_routing_weights_before_down"):
        moe_triton.resolve_routing_weights_before_down("maybe", train_router=True, ep_dispatch="alltoall")
