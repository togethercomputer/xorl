"""Component gates for the GLM-5.2 frozen-trunk backward mechanisms.

Covers the two mechanisms the full-param admission engages so trainable-set
gradients can traverse the frozen trunk:

1. the frozen expert BANK activation backward (out-of-scope layers under
   ``trainable_expert_layers``): hidden + routing gradients through the
   dequantized frozen bytes — the identical checked program the trainable
   bank differentiates, minus master gradients and minus any mutation;
2. the trainer-owned ROUTING-WEIGHT surrogate: serving top-k values verbatim
   in the forward, analytic regather vjp into the router logits in the
   backward (full-param mode trains routers; the serving top-k programs are
   gradient-opaque).

The full-depth integration of both is gated by
tests/models/test_glm52_fullparam_reduced_backward_gate.py.
"""

from __future__ import annotations

import logging

import pytest
import torch

from xorl.models.transformers.glm5.exact_fullparam_experts import (
    Glm52FullParamBlockFP8RoutedExperts,
)
from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    GLM52_FULLPARAM_ROUTING_SURROGATE_CONTRACT_VERSION,
    glm52_fullparam_routing_weights_with_grad,
)
from xorl.models.transformers.glm5.native_fp8 import (
    GLM52_NATIVE_EXPERTS_FROZEN_DGRAD_CONTRACT_VERSION,
    Glm52NativeBlockFP8Experts,
)


_HIDDEN = 128
_INTERMEDIATE = 128
_LOCAL_EXPERTS = 16


def _hopper_or_skip() -> torch.device:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    return torch.device("cuda")


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


def _grad_fixture(device: torch.device):
    """Multi-token routing with expert locality and one -1 sentinel slot."""

    rows = 5
    hidden = (
        torch.arange(rows * _HIDDEN, dtype=torch.float32, device=device)
        .reshape(rows, _HIDDEN)
        .remainder_(23)
        .sub_(11)
        .div_(64)
        .to(torch.bfloat16)
    )
    local_ids = torch.tensor([[2], [9], [9], [14], [-1]], dtype=torch.int32, device=device)
    routing = torch.tensor([[0.75], [0.5], [0.25], [1.0], [1.0]], dtype=torch.float32, device=device)
    return hidden, routing, local_ids


# ---------------------------------------------------------------------------
# Frozen bank (CPU negatives)
# ---------------------------------------------------------------------------


def test_unadmitted_frozen_bank_still_refuses_grad_requiring_hidden_and_routing() -> None:
    bank = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE)
    assert bank._frozen_dgrad_admitted is False
    hidden = torch.zeros(2, _HIDDEN, dtype=torch.bfloat16)
    routing = torch.zeros(2, 1, dtype=torch.float32)
    local_ids = torch.zeros(2, 1, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="scoring-only"):
        bank(hidden.clone().requires_grad_(True), routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(RuntimeError, match="scoring-only"):
        bank(hidden, routing.clone().requires_grad_(True), sglang_ep_native_local_ids=local_ids)


def test_trainable_bank_does_not_inherit_a_stale_admission_default() -> None:
    # The dgrad admission flag lives on the parent; the trainable subclass
    # must keep its own forward (its backward is the straight-through
    # surrogate, not the frozen dgrad) and the flag default must stay closed.
    bank = Glm52FullParamBlockFP8RoutedExperts(4, _HIDDEN, _INTERMEDIATE)
    assert bank._frozen_dgrad_admitted is False


# ---------------------------------------------------------------------------
# Frozen bank (Hopper CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_frozen_bank_value_bytes_identical_with_and_without_grad_engagement() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    bank = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    bank.load_prequantized(*_checkpoint_bytes(device))
    bank.enable_frozen_activation_dgrad()

    hidden, routing, local_ids = _grad_fixture(device)
    with torch.no_grad():
        scoring = bank(hidden, routing, sglang_ep_native_local_ids=local_ids, routed_scaling_factor=1.5)
    engaged = bank(
        hidden.clone().requires_grad_(True),
        routing.clone().requires_grad_(True),
        sglang_ep_native_local_ids=local_ids,
        routed_scaling_factor=1.5,
    )
    assert engaged.requires_grad
    assert torch.equal(engaged.detach().view(torch.uint8), scoring.view(torch.uint8))
    assert torch.count_nonzero(scoring) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_frozen_bank_activation_grads_match_trainable_bank_and_mutate_nothing(caplog) -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    checkpoint = _checkpoint_bytes(device)

    Glm52NativeBlockFP8Experts._frozen_dgrad_engagement_logged = False
    frozen = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    frozen.load_prequantized(*checkpoint)
    with caplog.at_level(logging.INFO, logger="xorl.models.transformers.glm5.native_fp8"):
        frozen.enable_frozen_activation_dgrad()
        frozen.enable_frozen_activation_dgrad()  # idempotent
    engagement = [record for record in caplog.records if "frozen-bank activation dgrad engaged" in record.message]
    assert len(engagement) == 1
    assert GLM52_NATIVE_EXPERTS_FROZEN_DGRAD_CONTRACT_VERSION in engagement[0].message

    packed_before = {
        name: getattr(frozen, name).detach().view(torch.uint8).clone()
        for name in (
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        )
    }

    hidden, routing, local_ids = _grad_fixture(device)
    frozen_hidden = hidden.clone().requires_grad_(True)
    frozen_routing = routing.clone().requires_grad_(True)
    output = frozen(
        frozen_hidden,
        frozen_routing,
        sglang_ep_native_local_ids=local_ids,
        routed_scaling_factor=1.5,
    )
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    assert frozen_hidden.grad is not None and frozen_routing.grad is not None
    assert bool(frozen_hidden.grad.abs().sum() > 0) and bool(frozen_routing.grad.abs().sum() > 0)
    # The sentinel row's hidden gradient is exactly zero (no expert touched it).
    assert torch.count_nonzero(frozen_hidden.grad[4]) == 0

    # Direct-vjp wiring identity (the autograd boundary passes exactly the
    # engaged tensors through).
    direct_hidden, direct_routing = frozen._frozen_activation_vjp(
        hidden,
        routing,
        local_ids,
        grad_output=grad_output,
        routed_scaling_factor=1.5,
        needs_input_grad=(True, True),
    )
    assert torch.equal(frozen_hidden.grad, direct_hidden)
    assert torch.equal(frozen_routing.grad, direct_routing)

    # Cross-implementation: the QUALIFIED trainable bank on identical bytes
    # produces bitwise-identical hidden/routing gradients (same checked
    # program, same bytes) — the frozen path is that treatment minus wgrad.
    trainable = Glm52FullParamBlockFP8RoutedExperts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    trainable.load_prequantized(*checkpoint)
    trainable_hidden = hidden.clone().requires_grad_(True)
    trainable_routing = routing.clone().requires_grad_(True)
    trainable(
        trainable_hidden,
        trainable_routing,
        sglang_ep_native_local_ids=local_ids,
        routed_scaling_factor=1.5,
    ).backward(grad_output)
    assert torch.equal(frozen_hidden.grad, trainable_hidden.grad)
    assert torch.equal(frozen_routing.grad, trainable_routing.grad)
    # ... and the trainable bank did produce master grads where the frozen
    # bank, by construction, has no master to grade.
    assert trainable.gate_up_weight_master.grad is not None

    # Frozen means frozen: no parameter gradients, no byte movement.
    for name, parameter in frozen.named_parameters():
        assert parameter.grad is None, f"frozen bank parameter {name} received a gradient"
        assert not parameter.requires_grad
    for name, before in packed_before.items():
        assert torch.equal(getattr(frozen, name).detach().view(torch.uint8), before), name


# ---------------------------------------------------------------------------
# Routing-weight surrogate
# ---------------------------------------------------------------------------


def test_routing_surrogate_returns_serving_values_verbatim_and_matches_reference_grads() -> None:
    torch.manual_seed(7)
    tokens, experts, topk = 6, 16, 4
    logits = torch.randn(tokens, experts, dtype=torch.float32)
    ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(tokens)]).to(torch.int32)

    for renormalize, scale, weights_dtype in (
        (True, 1.0, torch.float32),
        (True, 2.5, torch.bfloat16),
        (False, 1.0, torch.float32),
    ):
        serving_values = torch.rand(tokens, topk, dtype=torch.float32).to(weights_dtype)
        logits_leaf = logits.clone().requires_grad_(True)
        weights = glm52_fullparam_routing_weights_with_grad(
            logits_leaf,
            serving_values,
            ids,
            renormalize=renormalize,
            scale=scale,
        )
        # Values are the serving program's bytes, untouched.
        assert weights.dtype is serving_values.dtype
        assert torch.equal(weights.detach().view(torch.uint8), serving_values.view(torch.uint8))
        assert weights.requires_grad

        grad_weights = torch.rand(tokens, topk, dtype=torch.float32).to(weights_dtype)
        weights.backward(grad_weights)

        # Independent reference: autograd through the explicit regather program.
        reference_logits = logits.clone().requires_grad_(True)
        gathered = torch.sigmoid(reference_logits).gather(1, ids.long())
        if renormalize:
            reference = gathered / (gathered.sum(dim=-1, keepdim=True, dtype=torch.float32) + 1e-20)
        else:
            reference = gathered
        reference = reference * scale
        reference.backward(grad_weights.float())

        assert logits_leaf.grad is not None
        assert torch.equal(logits_leaf.grad, reference_logits.grad), (
            f"surrogate grad diverged from the regather reference (renormalize={renormalize}, scale={scale})"
        )


def test_routing_surrogate_validates_shapes_and_leaves_selection_gradless() -> None:
    logits = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    ids = torch.zeros(4, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="ids and weights of one shape"):
        glm52_fullparam_routing_weights_with_grad(logits, torch.zeros(4, 3), ids, renormalize=True, scale=1.0)
    with pytest.raises(ValueError, match="tokens, experts"):
        glm52_fullparam_routing_weights_with_grad(
            torch.zeros(4, 8, 1, requires_grad=True), torch.zeros(4, 2), ids, renormalize=True, scale=1.0
        )
    weights = glm52_fullparam_routing_weights_with_grad(logits, torch.rand(4, 2), ids, renormalize=True, scale=1.0)
    weights.sum().backward()
    assert logits.grad is not None
    assert GLM52_FULLPARAM_ROUTING_SURROGATE_CONTRACT_VERSION  # exported name stays pinned


def test_canonical_expert_slice_reconciles_declared_conventions() -> None:
    """Canonical dispatch reconciles both expert-count declarations.

    The dispatch compares local*ep_size against the bank's declared num_experts —
    GLOBAL for frozen/QLoRA banks (declare-global/store-local), but EP-LOCAL
    for the full-param bank.  The slice resolver must accept an EP-local
    full-param bank whose admission-assigned global range matches the
    dispatch-derived ownership, keep the frozen declared-global semantics,
    and fail closed on ownership mismatch or a missing range assignment."""

    from torch import nn

    from xorl.models.transformers.glm5.modeling_glm5 import Glm5MoEBlock

    def _block_with(bank: nn.Module) -> Glm5MoEBlock:
        block = Glm5MoEBlock.__new__(Glm5MoEBlock)  # slice resolver touches only .experts
        nn.Module.__init__(block)
        block.experts = bank
        return block

    ep_size, local, global_experts = 4, 4, 16

    # Full-param bank: declares LOCAL size, carries the admission-assigned range.
    for rank in range(ep_size):
        bank = Glm52FullParamBlockFP8RoutedExperts(local, _HIDDEN, _INTERMEDIATE)
        bank.assign_global_expert_range(rank * local, global_experts)
        assert _block_with(bank)._canonical_expert_slice(rank, ep_size) == (local, rank * local)

    # Ownership mismatch: admission assigned block 8.. but dispatch derives rank 0 -> 0.
    mismatched = Glm52FullParamBlockFP8RoutedExperts(local, _HIDDEN, _INTERMEDIATE)
    mismatched.assign_global_expert_range(2 * local, global_experts)
    with pytest.raises(RuntimeError, match="ownership must be identical"):
        _block_with(mismatched)._canonical_expert_slice(0, ep_size)

    # Unassigned range: the fail-closed property refuses to guess.
    unassigned = Glm52FullParamBlockFP8RoutedExperts(local, _HIDDEN, _INTERMEDIATE)
    with pytest.raises(RuntimeError, match="no assigned global expert range"):
        _block_with(unassigned)._canonical_expert_slice(0, ep_size)

    # Wrong world geometry: equal contiguous slices still enforced.
    wrong_world = Glm52FullParamBlockFP8RoutedExperts(local, _HIDDEN, _INTERMEDIATE)
    wrong_world.assign_global_expert_range(0, global_experts)
    with pytest.raises(RuntimeError, match="equal contiguous expert slices"):
        _block_with(wrong_world)._canonical_expert_slice(0, ep_size + 1)

    # Frozen native bank: declare-global/store-local (the production EP
    # pre-shrink leaves num_experts at the global count with local storage).
    frozen = Glm52NativeBlockFP8Experts(global_experts, _HIDDEN, _INTERMEDIATE)
    for name in (
        "gate_up_packed_weight_f32",
        "gate_up_weight_scale_inv",
        "down_packed_weight_f32",
        "down_weight_scale_inv",
    ):
        parameter = getattr(frozen, name)
        parameter.data = parameter.data[:local].clone()
    assert int(frozen.gate_up_proj.shape[0]) == local
    assert _block_with(frozen)._canonical_expert_slice(3, ep_size) == (local, 3 * local)
    with pytest.raises(RuntimeError, match="equal contiguous expert slices"):
        _block_with(frozen)._canonical_expert_slice(0, ep_size + 1)

    # Outside the expert-FSDP unshard window the packed parameters rest as
    # shards.  The resolver must be
    # declaration-based -- a storage-shape read would derive local_experts=1
    # and refuse a perfectly consistent bank.
    resting = Glm52NativeBlockFP8Experts(global_experts, _HIDDEN, _INTERMEDIATE)
    for name in (
        "gate_up_packed_weight_f32",
        "gate_up_weight_scale_inv",
        "down_packed_weight_f32",
        "down_weight_scale_inv",
    ):
        parameter = getattr(resting, name)
        parameter.data = parameter.data[:1].clone()  # sharded resting stand-in
    assert int(resting.gate_up_packed_weight_f32.shape[0]) == 1
    assert _block_with(resting)._canonical_expert_slice(3, ep_size) == (local, 3 * local)

    # A FULL frozen bank (reduced rigs) still resolves as world-of-one.
    full_frozen = Glm52NativeBlockFP8Experts(local, _HIDDEN, _INTERMEDIATE)
    assert _block_with(full_frozen)._canonical_expert_slice(0, 1) == (local, 0)


def test_route_detach_semantics_preserved_for_non_fullparam_lanes() -> None:
    """The generic MoE route still detaches routing weights when the router
    is not the full-param component — the frozen/QLoRA lanes' graph topology
    is untouched by the surrogate wiring."""

    import inspect

    from xorl.models.transformers.glm5.modeling_glm5 import Glm5MoEBlock

    source = inspect.getsource(Glm5MoEBlock.route)
    assert "_glm52_exact_fullparam_component" in source
    assert "routing_weights.detach()" in source
