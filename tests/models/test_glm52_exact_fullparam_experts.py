"""Tests for the GLM-5.2 full-parameter block-FP8 routed-expert bank.

Mirrors the exact routed-experts test idioms: the 128/128/16 rank-local
geometry, distinguishable byte fixtures, monkeypatched value programs on CPU,
and Hopper GPU gates for the load-bearing byte contracts (value equality
against the frozen serving bank on identical bytes, step-0 preservation,
refresh/publication identity, straight-through gradients, payload apply).
"""

from __future__ import annotations

import pytest
import torch

from xorl.models.transformers.glm5.exact_fullparam_experts import (
    GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION,
    Glm52FullParamBlockFP8RoutedExperts,
)


_HIDDEN = 128
_INTERMEDIATE = 128
_LOCAL_EXPERTS = 16


def _bank(device: torch.device | str = "cpu") -> Glm52FullParamBlockFP8RoutedExperts:
    return Glm52FullParamBlockFP8RoutedExperts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)


def _routing_fixture(device: torch.device):
    """Each local expert receives exactly one row; one row carries a sentinel."""

    rows = _LOCAL_EXPERTS
    hidden = (
        torch.arange(rows * _HIDDEN, dtype=torch.float32, device=device)
        .reshape(rows, _HIDDEN)
        .remainder_(17)
        .add_(1)
        .div_(32)
        .to(torch.bfloat16)
    )
    local_ids = torch.arange(rows, dtype=torch.int32, device=device).reshape(rows, 1).contiguous()
    routing = ((local_ids.float() + 1) / 32).contiguous()
    return hidden, routing, local_ids


# ---------------------------------------------------------------------------
# Construction and admission (CPU)
# ---------------------------------------------------------------------------


def test_bank_construction_declares_contract_and_fails_closed() -> None:
    module = _bank()
    assert module.contract_version == GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION
    assert module.glm52_fullparam_payload_kind == "block_fp8_expert_bank"
    assert module.fsdp_requires_full_precision is True
    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert trainable == {"gate_up_weight_master", "down_weight_master"}
    assert module.gate_up_weight_master.shape == (_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE)
    assert module.down_weight_master.shape == (_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN)
    assert module.gate_up_weight_master.dtype is torch.float32

    with pytest.raises(ValueError, match="divide the 128x128 block shape"):
        Glm52FullParamBlockFP8RoutedExperts(2, 96, 128)
    with pytest.raises(ValueError, match="only support silu"):
        Glm52FullParamBlockFP8RoutedExperts(2, 128, 128, hidden_act="gelu")


def test_bank_forward_and_publication_fail_closed_before_seed_and_after_mutation(monkeypatch) -> None:
    module = _bank()
    hidden, routing, local_ids = _routing_fixture(torch.device("cpu"))

    with pytest.raises(RuntimeError, match="before the quantized cache was seeded"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(RuntimeError, match="before the quantized cache was seeded"):
        module.publishable_expert_bytes()

    module._record_master_identity()
    monkeypatch.setattr(
        module,
        "_sampler_value",
        lambda h, r, ids, routed_scaling_factor: torch.zeros(h.shape[0], _HIDDEN, dtype=torch.bfloat16),
    )
    module(hidden, routing, sglang_ep_native_local_ids=local_ids)

    with torch.no_grad():
        module.gate_up_weight_master.add_(1.0)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module.publishable_expert_bytes()

    module._record_master_identity()
    module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    with torch.no_grad():
        module.down_weight_master.mul_(2.0)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)


def test_bank_engaged_contract_rejects_bad_inputs_before_any_kernel(monkeypatch) -> None:
    module = _bank()
    module._record_master_identity()
    monkeypatch.setattr(
        module,
        "_sampler_value",
        lambda h, r, ids, routed_scaling_factor: torch.zeros(h.shape[0], _HIDDEN, dtype=torch.bfloat16),
    )
    hidden, routing, local_ids = _routing_fixture(torch.device("cpu"))

    with pytest.raises(RuntimeError, match="Global ids must not bypass"):
        module(hidden, routing, selected_experts=local_ids.to(torch.int64))
    with pytest.raises(RuntimeError, match="require canonical rank-local IDs"):
        module(hidden, routing)
    with pytest.raises(TypeError, match="Unexpected"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids, topk=2)
    with pytest.raises(ValueError, match="scaling factor"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids, routed_scaling_factor=0.0)
    with pytest.raises(TypeError, match="BF16"):
        module(hidden.float(), routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(TypeError, match="FP32 \\[rows, topk\\]"):
        module(hidden, routing.to(torch.bfloat16), sglang_ep_native_local_ids=local_ids)
    with pytest.raises(TypeError, match="int32"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids.to(torch.int64))
    with pytest.raises(ValueError, match="sentinel"):
        module(hidden, routing, sglang_ep_native_local_ids=(local_ids + _LOCAL_EXPERTS).contiguous())
    with pytest.raises(ValueError, match="non-empty and contiguous"):
        module(
            torch.zeros(_LOCAL_EXPERTS, 2 * _HIDDEN, dtype=torch.bfloat16)[:, ::2],
            routing,
            sglang_ep_native_local_ids=local_ids,
        )

    module.gate_up_weight_master.requires_grad_(False)
    with pytest.raises(RuntimeError, match="must be trainable"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    module.gate_up_weight_master.requires_grad_(True)
    module.gate_up_packed_weight_f32.requires_grad_(True)
    with pytest.raises(RuntimeError, match="must remain frozen"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)


def test_bank_cpu_autograd_wiring_matches_direct_vjp(monkeypatch) -> None:
    torch.manual_seed(11)
    module = _bank()
    module._record_master_identity()

    gate_up_deq = torch.randn(_LOCAL_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN).to(torch.bfloat16)
    down_deq = torch.randn(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE).to(torch.bfloat16)
    monkeypatch.setattr(module, "_dequantized_cached_experts", lambda: (gate_up_deq, down_deq))

    def sampler_value(hidden, routing, ids, *, routed_scaling_factor):
        return module._surrogate_program(
            hidden.float(),
            routing.float(),
            ids,
            gate_up_deq.float(),
            down_deq.float(),
            routed_scaling_factor=routed_scaling_factor,
        ).to(torch.bfloat16)

    monkeypatch.setattr(module, "_sampler_value", sampler_value)

    hidden, routing, local_ids = _routing_fixture(torch.device("cpu"))
    # Route only experts 3 and 7; expert 7 twice, one sentinel row.
    local_ids = torch.tensor([[3], [7], [7], [-1]], dtype=torch.int32)
    routing = torch.tensor([[0.25], [0.5], [0.125], [1.0]], dtype=torch.float32)
    hidden = hidden[:4].contiguous().requires_grad_(True)
    routing_leaf = routing.clone().requires_grad_(True)

    output = module(hidden, routing_leaf, sglang_ep_native_local_ids=local_ids, routed_scaling_factor=2.0)
    grad_output = torch.randn_like(output.float()).to(torch.bfloat16)
    output.backward(grad_output)

    expected = module._straight_through_vjp(
        hidden.detach(),
        routing,
        local_ids,
        grad_output=grad_output,
        routed_scaling_factor=2.0,
        needs_input_grad=(True, True, True, True),
    )
    assert torch.equal(hidden.grad, expected[0].to(torch.bfloat16))
    assert torch.equal(routing_leaf.grad, expected[1])
    assert module.gate_up_weight_master.grad is not None
    assert module.gate_up_weight_master.grad.dtype is torch.float32
    assert module.gate_up_weight_master.grad.shape == module.gate_up_weight_master.shape
    assert torch.equal(module.gate_up_weight_master.grad, expected[2])
    assert torch.equal(module.down_weight_master.grad, expected[3])

    # Expert-boundary locality: only routed experts carry gradient.
    routed = {3, 7}
    for expert_index in range(_LOCAL_EXPERTS):
        gate_up_grad = module.gate_up_weight_master.grad[expert_index]
        down_grad = module.down_weight_master.grad[expert_index]
        if expert_index in routed:
            assert torch.count_nonzero(gate_up_grad) > 0
            assert torch.count_nonzero(down_grad) > 0
        else:
            assert torch.count_nonzero(gate_up_grad) == 0
            assert torch.count_nonzero(down_grad) == 0


def test_bank_engaged_contract_refuses_understored_rows_before_any_kernel(monkeypatch) -> None:
    """Stored-rows companion to the frozen bank's admission.

    The full-param local-id bound checks against the DECLARED EP-local
    size; that bound is only safe if the storage actually holds that many
    rows.  A bank whose masters/caches were sliced after construction must
    refuse before any kernel."""

    module = _bank()
    module._record_master_identity()
    monkeypatch.setattr(
        module,
        "_sampler_value",
        lambda h, r, ids, routed_scaling_factor: torch.zeros(h.shape[0], _HIDDEN, dtype=torch.bfloat16),
    )
    hidden, routing, local_ids = _routing_fixture(torch.device("cpu"))

    # Simulate the corrupt construction: one stored expert row, 16 declared.
    for name in ("gate_up_weight_master", "down_weight_master"):
        sliced = getattr(module, name).detach()[:1].clone()
        setattr(module, name, torch.nn.Parameter(sliced, requires_grad=True))
    for name in (
        "gate_up_packed_weight_f32",
        "gate_up_weight_scale_inv",
        "down_packed_weight_f32",
        "down_weight_scale_inv",
    ):
        setattr(module, name, getattr(module, name).detach()[:1].clone())

    with pytest.raises(RuntimeError, match="corrupt construction"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)
