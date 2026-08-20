"""Tests that EP adapter wrappers in backend/__init__.py correctly forward expert_scores.

Bug: _quack_ep_fused / _native_ep_fused accepted expert_scores but silently dropped it.

These tests monkeypatch the downstream implementations to verify the adapters pass
expert_scores through, and compare output to a naive reference.
"""

import inspect

import pytest
import torch
import torch.nn.functional as F

import xorl.models.layers.moe.backend as moe_backend
from tests._helpers.moe import counts_from_cumsum, patch_ep_kernels
from xorl.models.layers.moe.backend import EP_EXPERT_COMPUTE, EP_EXPERT_COMPUTE_MOE_ACT
from xorl.ops.moe import quack as quack_moe
from xorl.ops.moe import triton as triton_moe


pytestmark = pytest.mark.cpu


def _assert_quack_ep_registers_without_optional_moe_act():
    """The base Quack EP path must not depend on the optional MoE-act class."""
    assert hasattr(moe_backend, "_QuackEPGroupGemm")
    assert not hasattr(moe_backend, "_QuackEPGroupGemmMoeAct")
    assert "quack" in EP_EXPERT_COMPUTE
    assert "quack" not in EP_EXPERT_COMPUTE_MOE_ACT


# ---------------------------------------------------------------------------
# Signature contract tests — replaces the old source-grep regression test.
# Inspects live function signatures instead of pattern-matching on source text,
# so formatting changes, variable renames, and line rewrapping can't fool it.
# ---------------------------------------------------------------------------

# Required explicit parameters for every EP_EXPERT_COMPUTE entry.
_REQUIRED_EP_PARAMS = ("expert_scores", "hidden_act", "activation_native", "swiglu_limit")


def _assert_ep_registry_signature_contracts():
    """Every registered EP function exposes the shared forward-compatible API."""
    registries = {
        "EP_EXPERT_COMPUTE": EP_EXPERT_COMPUTE,
        "EP_EXPERT_COMPUTE_MOE_ACT": EP_EXPERT_COMPUTE_MOE_ACT,
    }
    for registry_name, registry in registries.items():
        for name, fn in registry.items():
            sig = inspect.signature(fn)
            params = sig.parameters

            for required in _REQUIRED_EP_PARAMS:
                assert required in params, (
                    f"{registry_name}['{name}'] is missing explicit '{required}' param. Signature: {sig}"
                )

            has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in params.values())
            assert has_var_keyword, (
                f"{registry_name}['{name}'] has no **kwargs — new extras like "
                f"gate_up_bias will break callers. Signature: {sig}"
            )


def _assert_native_ep_adapter_consumes_fp8_kwargs_when_fp8_compute_is_disabled():
    """Native EP is a BF16 fallback path but receives the common EP FP8 kwargs."""

    fn = EP_EXPERT_COMPUTE["native"]

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    out = fn(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size=4,
        expert_scores=None,
        hidden_act="silu",
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
    )

    assert out.shape == permute_tokens.shape


def _assert_native_ep_adapter_rejects_fp8_expert_compute_explicitly():
    fn = EP_EXPERT_COMPUTE["native"]

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    with pytest.raises(NotImplementedError, match="native EP backend does not support FP8 expert compute"):
        fn(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            intermediate_size=4,
            expert_scores=None,
            hidden_act="silu",
            fp8_compute=True,
            fp8_grouped_backend="triton_grouped",
            fp8_block_size=128,
        )


def _assert_triton_ep_adapter_consumes_fp8_kwargs_when_fp8_compute_is_disabled(monkeypatch):
    fn = EP_EXPERT_COMPUTE["triton"]

    def fake_apply(*args):
        return args[0].new_empty(args[0].shape)

    monkeypatch.setattr(moe_backend.TritonEPGroupGemm, "apply", staticmethod(fake_apply))

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    out = fn(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size=4,
        expert_scores=None,
        hidden_act="silu",
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
    )

    assert out.shape == permute_tokens.shape


def _assert_triton_ep_adapter_rejects_fp8_expert_compute_explicitly():
    fn = EP_EXPERT_COMPUTE["triton"]

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    with pytest.raises(NotImplementedError, match="triton EP backend does not support FP8 expert compute"):
        fn(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            intermediate_size=4,
            expert_scores=None,
            hidden_act="silu",
            fp8_compute=True,
            fp8_grouped_backend="triton_grouped",
            fp8_block_size=128,
            gate_up_bias=None,
            down_bias=None,
        )


def _assert_triton_moe_act_ep_adapter_consumes_common_kwargs(monkeypatch):
    fn = EP_EXPERT_COMPUTE_MOE_ACT["triton"]

    def fake_apply(*args):
        return args[0].new_empty(args[0].shape)

    monkeypatch.setattr(moe_backend.TritonEPGroupGemmMoeAct, "apply", staticmethod(fake_apply))

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    out = fn(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size=4,
        expert_scores=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        swiglu_limit=0.0,
        gate_up_bias=None,
        down_bias=None,
        gated=True,
    )

    assert out.shape == permute_tokens.shape


def _assert_triton_moe_act_ep_adapter_rejects_fp8_expert_compute_explicitly():
    fn = EP_EXPERT_COMPUTE_MOE_ACT["triton"]

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    with pytest.raises(NotImplementedError, match="triton moe_act EP backend does not support FP8 expert compute"):
        fn(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            intermediate_size=4,
            expert_scores=None,
            hidden_act="silu",
            activation_native=False,
            fp8_compute=True,
            fp8_grouped_backend="triton_grouped",
            fp8_block_size=128,
            gate_up_bias=None,
            down_bias=None,
        )


def _assert_quack_ep_adapter_forwards_activation_native(monkeypatch):
    fn = EP_EXPERT_COMPUTE["quack"]
    quack_cls = moe_backend._QuackEPGroupGemm

    seen = {}

    def fake_apply(*args):
        seen["activation_native"] = args[7]
        return args[0].new_empty(args[0].shape)

    monkeypatch.setattr(quack_cls, "apply", staticmethod(fake_apply))

    permute_tokens = torch.empty(0, 4)
    cumsum = torch.zeros(1, dtype=torch.int32)
    gate_up_proj = torch.empty(1, 4, 8)
    down_proj = torch.empty(1, 4, 4)

    out = fn(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size=4,
        expert_scores=None,
        hidden_act="silu",
        activation_native=True,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
    )

    assert out.shape == permute_tokens.shape
    assert seen["activation_native"] is True


def _reference_ep_forward(permute_tokens, cumsum, gate_proj, up_proj, down_proj, expert_scores):
    outputs = []
    start = 0
    for expert_idx, count in enumerate(counts_from_cumsum(cumsum)):
        end = start + count
        x = permute_tokens[start:end]
        hidden = F.silu(x @ gate_proj[expert_idx]) * (x @ up_proj[expert_idx])
        hidden = hidden * expert_scores[start:end].to(hidden.dtype).unsqueeze(-1)
        outputs.append(hidden @ down_proj[expert_idx])
        start = end
    return torch.cat(outputs, dim=0)


def _assert_ep_group_gemm_propagates_routing_score_gradients(monkeypatch, module, class_name):
    patch_ep_kernels(monkeypatch, module)
    fn = getattr(module, class_name)

    torch.manual_seed(0)
    num_local_experts = 2
    hidden_dim = 8
    intermediate_size = 12
    counts = torch.tensor([3, 2])
    cumsum = torch.cumsum(counts, dim=0)
    num_tokens = int(cumsum[-1].item())

    permute_tokens = torch.randn(num_tokens, hidden_dim)
    gate_proj = torch.randn(num_local_experts, hidden_dim, intermediate_size)
    up_proj = torch.randn(num_local_experts, hidden_dim, intermediate_size)
    down_proj = torch.randn(num_local_experts, intermediate_size, hidden_dim)
    expert_scores = torch.rand(num_tokens, requires_grad=True)
    upstream = torch.randn(num_tokens, hidden_dim)

    gate_up_proj = torch.cat([gate_proj, up_proj], dim=-1)
    output = fn.apply(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores)
    output.backward(upstream)
    grad_scores = expert_scores.grad.detach().clone()

    expert_scores_ref = expert_scores.detach().clone().requires_grad_(True)
    ref_output = _reference_ep_forward(
        permute_tokens,
        cumsum,
        gate_proj,
        up_proj,
        down_proj,
        expert_scores_ref,
    )
    ref_output.backward(upstream)

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(grad_scores, expert_scores_ref.grad)


def test_ep_adapter_registry_backend_arguments_and_fp8_boundary_contract(monkeypatch):
    _assert_quack_ep_registers_without_optional_moe_act()
    _assert_ep_registry_signature_contracts()
    assert {"native", "triton", "quack"} <= EP_EXPERT_COMPUTE.keys()
    assert "triton" in EP_EXPERT_COMPUTE_MOE_ACT
    assert hasattr(moe_backend, "TritonEPGroupGemm")
    assert hasattr(moe_backend, "TritonEPGroupGemmMoeAct")

    for module, class_name in (
        (triton_moe, "TritonEPGroupGemm"),
        (quack_moe, "QuackEPGroupGemm"),
    ):
        with monkeypatch.context() as kernel_patch:
            _assert_ep_group_gemm_propagates_routing_score_gradients(kernel_patch, module, class_name)

    _assert_native_ep_adapter_consumes_fp8_kwargs_when_fp8_compute_is_disabled()
    _assert_native_ep_adapter_rejects_fp8_expert_compute_explicitly()
    _assert_triton_ep_adapter_consumes_fp8_kwargs_when_fp8_compute_is_disabled(monkeypatch)
    _assert_triton_ep_adapter_rejects_fp8_expert_compute_explicitly()
    _assert_triton_moe_act_ep_adapter_consumes_common_kwargs(monkeypatch)
    _assert_triton_moe_act_ep_adapter_rejects_fp8_expert_compute_explicitly()
    _assert_quack_ep_adapter_forwards_activation_native(monkeypatch)
