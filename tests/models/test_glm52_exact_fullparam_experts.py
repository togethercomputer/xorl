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
from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    quantize_expert_masters_to_serving_bytes,
)
from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
from xorl.server.weight_sync.glm52_fullparam_payload import (
    Glm52ExpectedPayloadField,
    Glm52ExpectedPayloadInventory,
    Glm52ExpectedPayloadItem,
    Glm52WeightVersionGuard,
    apply_glm52_fullparam_payload,
    publish_glm52_fullparam_payload,
)


def _expected_inventory(payload) -> Glm52ExpectedPayloadInventory:
    return Glm52ExpectedPayloadInventory(
        items=tuple(
            Glm52ExpectedPayloadItem(
                target=item.target,
                kind=item.kind,
                contract_version=item.contract_version,
                fields=tuple(Glm52ExpectedPayloadField(field.name, field.dtype, field.shape) for field in item.fields),
            )
            for item in payload.items
        )
    )


_HIDDEN = 128
_INTERMEDIATE = 128
_LOCAL_EXPERTS = 16


def _bank(device: torch.device | str = "cpu") -> Glm52FullParamBlockFP8RoutedExperts:
    return Glm52FullParamBlockFP8RoutedExperts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)


def _checkpoint_bytes(device: torch.device):
    gate_up = torch.empty(_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, dtype=torch.float8_e4m3fn, device=device)
    gate_up[..., :_INTERMEDIATE] = 0.015625
    gate_up[..., _INTERMEDIATE:] = 0.03125
    gate_up_scale = torch.ones(_LOCAL_EXPERTS, 1, 2, dtype=torch.float32, device=device)
    down = torch.full(
        (_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN),
        0.015625,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    down_scale = torch.ones(_LOCAL_EXPERTS, 1, 1, dtype=torch.float32, device=device)
    return gate_up, gate_up_scale, down, down_scale


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


# ---------------------------------------------------------------------------
# Hopper CUDA byte contracts
# ---------------------------------------------------------------------------


def _hopper_or_skip() -> torch.device:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    return torch.device("cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_value_bytes_match_frozen_serving_bank_on_identical_bytes() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    module = _bank(device)
    checkpoint = _checkpoint_bytes(device)
    module.load_prequantized(*checkpoint)

    # Step-0 identity: publication returns the checkpoint bytes verbatim.
    published = module.publishable_expert_bytes()
    assert torch.equal(published[0].view(torch.uint8), checkpoint[0].view(torch.uint8))
    assert torch.equal(published[1], checkpoint[1])
    assert torch.equal(published[2].view(torch.uint8), checkpoint[2].view(torch.uint8))
    assert torch.equal(published[3], checkpoint[3])

    frozen = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    frozen.load_prequantized(*checkpoint)

    hidden, routing, local_ids = _routing_fixture(device)
    trainer_value = module(
        hidden.clone().requires_grad_(True),
        routing.clone().requires_grad_(True),
        sglang_ep_native_local_ids=local_ids,
    )
    frozen_value = frozen(
        hidden,
        routing,
        sglang_ep_native_local_ids=local_ids,
    )
    assert trainer_value.dtype is torch.bfloat16
    assert torch.equal(trainer_value.detach(), frozen_value)
    assert torch.count_nonzero(trainer_value.detach()) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_refresh_publishes_per_expert_quantization_and_staleness_trips() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    torch.manual_seed(13)
    module = _bank(device)
    with torch.no_grad():
        module.gate_up_weight_master.copy_(torch.randn(_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device))
        module.down_weight_master.copy_(torch.randn(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN, device=device))
    module.refresh_quantized_cache()

    published = module.publishable_expert_bytes()
    reference_gate_up, reference_gate_up_scale = quantize_expert_masters_to_serving_bytes(module.gate_up_weight_master)
    reference_down, reference_down_scale = quantize_expert_masters_to_serving_bytes(module.down_weight_master)
    assert torch.equal(published[0].view(torch.uint8), reference_gate_up.view(torch.uint8))
    assert torch.equal(published[1], reference_gate_up_scale)
    assert torch.equal(published[2].view(torch.uint8), reference_down.view(torch.uint8))
    assert torch.equal(published[3], reference_down_scale)

    hidden, routing, local_ids = _routing_fixture(device)
    optimizer = torch.optim.SGD([module.gate_up_weight_master, module.down_weight_master], lr=5.0)
    module(hidden, routing, sglang_ep_native_local_ids=local_ids).float().sum().backward()
    optimizer.step()
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module.publishable_expert_bytes()
    module.refresh_quantized_cache()
    module(hidden, routing, sglang_ep_native_local_ids=local_ids)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_straight_through_gradients_match_direct_vjp_with_expert_locality() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    module = _bank(device)
    module.load_prequantized(*_checkpoint_bytes(device))

    rows = 5
    hidden = (
        torch.arange(rows * _HIDDEN, dtype=torch.float32, device=device)
        .reshape(rows, _HIDDEN)
        .remainder_(23)
        .sub_(11)
        .div_(64)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    local_ids = torch.tensor([[2], [9], [9], [14], [-1]], dtype=torch.int32, device=device)
    routing = torch.tensor([[0.75], [0.5], [0.25], [1.0], [1.0]], dtype=torch.float32, device=device).requires_grad_(
        True
    )

    output = module(hidden, routing, sglang_ep_native_local_ids=local_ids, routed_scaling_factor=1.5)
    grad_output = torch.ones_like(output)
    output.backward(grad_output)

    expected = module._straight_through_vjp(
        hidden.detach(),
        routing.detach(),
        local_ids,
        grad_output=grad_output,
        routed_scaling_factor=1.5,
        needs_input_grad=(True, True, True, True),
    )
    assert torch.equal(hidden.grad, expected[0].to(torch.bfloat16))
    assert torch.equal(routing.grad, expected[1])
    assert torch.equal(module.gate_up_weight_master.grad, expected[2])
    assert torch.equal(module.down_weight_master.grad, expected[3])

    routed = {2, 9, 14}
    for expert_index in range(_LOCAL_EXPERTS):
        nonzero = torch.count_nonzero(module.gate_up_weight_master.grad[expert_index])
        if expert_index in routed:
            assert nonzero > 0
        else:
            assert nonzero == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_ep_rank_local_banks_partition_values_and_own_master_gradients() -> None:
    """Verify EP ownership and normalization.

    Production canonical-EP semantics: every rank's bank sees the GATHERED
    token set with foreign slots mapped to the -1 sentinel and routing
    weights normalized ONCE globally (sigmoid -> topk -> norm -> scale,
    never renormalized over the local expert subset); the combine SUMS the
    rank partials.  Gate: two EP-local half banks seeded from the same
    bytes as a 16-expert reference bank must (a) emit exact zeros for
    fully-foreign rows, (b) sum to the reference value, and (c) own
    exactly their slice of the master gradients, bitwise.
    """

    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    torch.manual_seed(29)

    reference = _bank(device)  # 16 experts: the single-rank reference
    with torch.no_grad():
        reference.gate_up_weight_master.copy_(torch.randn(_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device))
        reference.down_weight_master.copy_(torch.randn(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN, device=device))
    reference.refresh_quantized_cache()
    published = reference.publishable_expert_bytes()

    half = _LOCAL_EXPERTS // 2
    rank_banks: list[Glm52FullParamBlockFP8RoutedExperts] = []
    for rank in (0, 1):
        bank = Glm52FullParamBlockFP8RoutedExperts(half, _HIDDEN, _INTERMEDIATE, device=device)
        rows = slice(rank * half, (rank + 1) * half)
        bank.load_prequantized(published[0][rows], published[1][rows], published[2][rows], published[3][rows])
        bank.assign_global_expert_range(rank * half, _LOCAL_EXPERTS)
        rank_banks.append(bank)
    # Expert-boundary-preserving quantization: the half banks hold the
    # reference's exact bytes, so any value drift below is EP mechanics.
    for rank, bank in enumerate(rank_banks):
        rows = slice(rank * half, (rank + 1) * half)
        assert torch.equal(
            bank.gate_up_packed_weight_f32.view(torch.uint8),
            reference.gate_up_packed_weight_f32[rows].view(torch.uint8),
        )

    # Gathered token set, topk=2: one row per global expert (sentinel second
    # slot with a NONZERO weight that must be ignored), one cross-rank row,
    # one expert-multiplicity row. Routing is production-shaped: sigmoid
    # scores, normalized over the row's REAL experts once, globally.
    rows = _LOCAL_EXPERTS + 2
    cross_row, multiplicity_row = _LOCAL_EXPERTS, _LOCAL_EXPERTS + 1
    hidden_values = (
        torch.arange(rows * _HIDDEN, dtype=torch.float32, device=device)
        .reshape(rows, _HIDDEN)
        .remainder_(19)
        .sub_(9)
        .div_(64)
        .to(torch.bfloat16)
    )
    global_ids = torch.full((rows, 2), -1, dtype=torch.int32, device=device)
    global_ids[:_LOCAL_EXPERTS, 0] = torch.arange(_LOCAL_EXPERTS, dtype=torch.int32, device=device)
    global_ids[cross_row] = torch.tensor([3, 11], dtype=torch.int32, device=device)
    global_ids[multiplicity_row, 0] = 9
    scores = torch.sigmoid(torch.randn(rows, 2, device=device, dtype=torch.float32))
    real = (global_ids >= 0).float()
    routing = (scores * real) / ((scores * real).sum(dim=-1, keepdim=True) + 1e-20)
    routing = routing + 0.33 * (1.0 - real)  # sentinel-slot weights must be dead
    routing = routing.contiguous()

    def local_ids_for(rank: int) -> torch.Tensor:
        owned = (global_ids >= rank * half) & (global_ids < (rank + 1) * half)
        return torch.where(owned, global_ids - rank * half, global_ids.new_full((), -1)).contiguous()

    grad_output = (
        torch.arange(rows * _HIDDEN, dtype=torch.float32, device=device)
        .reshape(rows, _HIDDEN)
        .remainder_(13)
        .sub_(6)
        .div_(32)
        .to(torch.bfloat16)
    )

    def run(bank: Glm52FullParamBlockFP8RoutedExperts, ids: torch.Tensor):
        hidden_leaf = hidden_values.clone().requires_grad_(True)
        routing_leaf = routing.clone().requires_grad_(True)
        value = bank(hidden_leaf, routing_leaf, sglang_ep_native_local_ids=ids, routed_scaling_factor=1.5)
        value.backward(grad_output)
        return value.detach(), hidden_leaf.grad, routing_leaf.grad

    reference_value, reference_hidden_grad, reference_routing_grad = run(reference, global_ids)
    partials = [run(bank, local_ids_for(rank)) for rank, bank in enumerate(rank_banks)]

    # (a) fully-foreign rows are EXACT zeros (the combine adds partials).
    for rank, (value, _, _) in enumerate(partials):
        foreign = (local_ids_for(rank) < 0).all(dim=-1)
        assert bool(foreign.any())
        assert torch.count_nonzero(value[foreign]) == 0, f"rank {rank} leaked non-zero foreign rows"

    # (b) the combine reproduces the reference value: bitwise wherever a row
    # is served by ONE rank; the cross-rank row differs from a single-rank
    # reference by exactly the combine's extra BF16 rounding (the reference
    # kernel sums both expert contributions before the cast — one rounding;
    # the EP combine adds two already-rounded partials — two roundings; the
    # production NCCL combine has the same property). The bound below is
    # expressed in terms of the rounded operands.
    combined = partials[0][0] + partials[1][0]
    single_owner = torch.ones(rows, dtype=torch.bool, device=device)
    single_owner[cross_row] = False
    assert torch.equal(combined[single_owner], reference_value[single_owner])
    cross_diff = (combined[cross_row].float() - reference_value[cross_row].float()).abs()
    # Rounding error of the two-partial combine is bounded by the OPERAND
    # magnitudes (cancellation makes a result-relative bound wrong).
    cross_bound = (partials[0][0][cross_row].float().abs() + partials[1][0][cross_row].float().abs()).clamp(
        min=1.0
    ) * 2**-7
    assert bool((cross_diff <= cross_bound).all()), "cross-rank row exceeded one BF16 combine rounding"

    # (c) master-gradient ownership: each rank owns exactly its slice.
    assert torch.count_nonzero(reference.gate_up_weight_master.grad) > 0
    for rank, bank in enumerate(rank_banks):
        rows_slice = slice(rank * half, (rank + 1) * half)
        assert torch.equal(bank.gate_up_weight_master.grad, reference.gate_up_weight_master.grad[rows_slice])
        assert torch.equal(bank.down_weight_master.grad, reference.down_weight_master.grad[rows_slice])

    # Routing-weight gradients partition per (row, slot): owner slots carry
    # the reference gradient, foreign and sentinel slots are exactly zero.
    assert torch.equal(partials[0][2] + partials[1][2], reference_routing_grad)
    sentinel_slots = global_ids < 0
    assert torch.count_nonzero(reference_routing_grad[sentinel_slots]) == 0

    # Hidden-state gradients: single-owner rows are bitwise; the cross-rank
    # row sums two BF16 partials (two roundings) against the reference's
    # single FP32 accumulation (one rounding) — compared at BF16 tolerance.
    hidden_grad_sum = partials[0][1].float() + partials[1][1].float()
    assert torch.equal(hidden_grad_sum[single_owner].to(torch.bfloat16), reference_hidden_grad[single_owner])
    assert torch.allclose(
        hidden_grad_sum[cross_row].to(torch.bfloat16).float(),
        reference_hidden_grad[cross_row].float(),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_production_optimizer_covers_only_masters_and_post_update_equality() -> None:
    """Verify optimizer coverage and post-update equality.

    A production-style optimizer built from ``parameters()`` with the
    requires_grad filter must cover exactly the two FP32 masters; a real
    step must leave the byte caches untouched (bytes move only at refresh),
    trip the staleness gate, and after refresh the published bytes must be
    the quantization of the UPDATED masters — with the frozen receiver
    byte- and value-equal through BOTH transport forms (fused bank payload
    and per-expert checkpoint items).
    """

    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeExpertSlotReceiver

    torch.manual_seed(31)
    module = _bank(device)
    module.load_prequantized(*_checkpoint_bytes(device))
    module.assign_global_expert_range(0, _LOCAL_EXPERTS)

    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    assert {id(p) for p in trainable} == {id(module.gate_up_weight_master), id(module.down_weight_master)}
    optimizer = torch.optim.AdamW(trainable, lr=0.05)

    cache_before = tuple(
        getattr(module, name).detach().clone()
        for name in (
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        )
    )
    masters_before = (
        module.gate_up_weight_master.detach().clone(),
        module.down_weight_master.detach().clone(),
    )

    hidden, routing, local_ids = _routing_fixture(device)
    value = module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    value.float().square().sum().backward()
    optimizer.step()

    # The step moved ONLY the masters; the consumed bytes are untouched.
    for name, before in zip(
        (
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        ),
        cache_before,
        strict=True,
    ):
        assert torch.equal(getattr(module, name).detach(), before), f"optimizer step mutated {name}"
    assert not torch.equal(module.gate_up_weight_master.detach(), masters_before[0])
    assert not torch.equal(module.down_weight_master.detach(), masters_before[1])
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)

    module.refresh_quantized_cache()
    published = module.publishable_expert_bytes()
    # The refreshed cache is the quantization of the UPDATED masters.
    expected_gate_up, expected_gate_up_scale = quantize_expert_masters_to_serving_bytes(module.gate_up_weight_master)
    assert torch.equal(published[0].view(torch.uint8), expected_gate_up.view(torch.uint8))
    assert torch.equal(published[1], expected_gate_up_scale)
    # ... and it really moved off the step-0 checkpoint bytes.
    assert not torch.equal(published[0].view(torch.uint8), _checkpoint_bytes(device)[0].view(torch.uint8))

    # Post-update equality through BOTH transport forms.
    payload = publish_glm52_fullparam_payload(
        [
            ("experts", module),
            *(
                (f"experts_ckpt.{global_id}", publication)
                for global_id, publication in module.checkpoint_publications()
            ),
        ],
        weight_version="post-step-1",
        weight_step=1,
    )
    fused_receiver = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    slot_receiver_bank = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)

    def resolver(target: str, kind: str):
        if kind == "block_fp8_expert_bank":
            return fused_receiver
        assert kind == "block_fp8_expert"
        return Glm52NativeExpertSlotReceiver(slot_receiver_bank, int(target.rsplit(".", 1)[1]))

    apply_glm52_fullparam_payload(
        payload,
        resolver,
        expected_inventory=_expected_inventory(payload),
        version_guard=Glm52WeightVersionGuard(),
    )

    for receiver in (fused_receiver, slot_receiver_bank):
        assert torch.equal(
            receiver.gate_up_packed_weight_f32.view(torch.uint8),
            module.gate_up_packed_weight_f32.view(torch.uint8),
        )
        assert torch.equal(
            receiver.down_packed_weight_f32.view(torch.uint8),
            module.down_packed_weight_f32.view(torch.uint8),
        )
        receiver_value = receiver(hidden, routing, sglang_ep_native_local_ids=local_ids)
        trainer_value = module(hidden, routing, sglang_ep_native_local_ids=local_ids)
        assert torch.equal(trainer_value.detach(), receiver_value)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_payload_applies_expert_bank_to_frozen_receiver_with_byte_equality() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    torch.manual_seed(17)
    module = _bank(device)
    with torch.no_grad():
        module.gate_up_weight_master.copy_(torch.randn(_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device))
        module.down_weight_master.copy_(torch.randn(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN, device=device))
    module.refresh_quantized_cache()

    payload = publish_glm52_fullparam_payload([("experts", module)], weight_version="step-2")

    receiver = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)

    def resolver(target: str, kind: str):
        assert (target, kind) == ("experts", "block_fp8_expert_bank")
        return receiver

    apply_glm52_fullparam_payload(
        payload,
        resolver,
        expected_inventory=_expected_inventory(payload),
        version_guard=Glm52WeightVersionGuard(),
    )

    hidden, routing, local_ids = _routing_fixture(device)
    trainer_value = module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    receiver_value = receiver(hidden, routing, sglang_ep_native_local_ids=local_ids)
    assert torch.equal(trainer_value.detach(), receiver_value)


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
