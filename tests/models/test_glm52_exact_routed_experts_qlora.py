from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION,
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
    localize_glm52_ep16_expert_ids,
)
from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts


_HIDDEN = 128
_INTERMEDIATE = 128
_GLOBAL_EXPERTS = 256
_EP_SIZE = 16
_LOCAL_EXPERTS = 16


def _module(owner: int, device: torch.device | str = "cpu") -> Glm52ExactEP16BlockFP8QLoRARoutedExperts:
    return Glm52ExactEP16BlockFP8QLoRARoutedExperts(
        _HIDDEN,
        _INTERMEDIATE,
        ep_rank=owner,
        device=device,
    )


def _load_zero_base(module: Glm52ExactEP16BlockFP8QLoRARoutedExperts) -> None:
    device = module.gate_up_packed_weight_f32.device
    module.load_prequantized(
        torch.zeros(
            _LOCAL_EXPERTS,
            _HIDDEN,
            2 * _INTERMEDIATE,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.ones(_LOCAL_EXPERTS, 1, 2, dtype=torch.float32, device=device),
        torch.zeros(
            _LOCAL_EXPERTS,
            _INTERMEDIATE,
            _HIDDEN,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.ones(_LOCAL_EXPERTS, 1, 1, dtype=torch.float32, device=device),
    )


def _load_distinguishable_base(module: Glm52ExactEP16BlockFP8QLoRARoutedExperts) -> None:
    device = module.gate_up_packed_weight_f32.device
    gate_up = torch.empty(
        _LOCAL_EXPERTS,
        _HIDDEN,
        2 * _INTERMEDIATE,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    gate_up[..., :_INTERMEDIATE] = 0.015625
    gate_up[..., _INTERMEDIATE:] = 0.03125
    module.load_prequantized(
        gate_up,
        torch.ones(_LOCAL_EXPERTS, 1, 2, dtype=torch.float32, device=device),
        torch.full(
            (_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN),
            0.015625,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.ones(_LOCAL_EXPERTS, 1, 1, dtype=torch.float32, device=device),
    )


def _fill_distinguishable_factors(module: Glm52ExactEP16BlockFP8QLoRARoutedExperts) -> None:
    owner = module.ep_rank
    global_ids = torch.arange(_GLOBAL_EXPERTS, dtype=torch.float32, device=module.gate_proj_lora_A.device)
    with torch.no_grad():
        module.gate_proj_lora_A.fill_((owner + 1) / 512)
        module.up_proj_lora_A.fill_((owner + 17) / 1024)
        module.down_proj_lora_B.fill_((owner + 3) / 256)
        module.gate_proj_lora_B.copy_(((global_ids + 1) / 4096).view(_GLOBAL_EXPERTS, 1, 1))
        module.up_proj_lora_B.copy_(((global_ids + 257) / 4096).view(_GLOBAL_EXPERTS, 1, 1))
        module.down_proj_lora_A.copy_(((global_ids + 33) / 8192).view(_GLOBAL_EXPERTS, 1, 1))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint16)


def _standalone_hybrid_routed_vjp(
    module: Glm52ExactEP16BlockFP8QLoRARoutedExperts,
    hidden: torch.Tensor,
    routing: torch.Tensor,
    local_ids: torch.Tensor,
    effective_factors: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, ...]:
    """Independent staged-QloRA oracle; do not reuse the module surrogate."""

    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    gate_up_weight = block_quant_dequant(
        module.gate_up_proj.transpose(1, 2),
        module.gate_up_weight_scale_inv.transpose(1, 2),
        [128, 128],
        torch.bfloat16,
    )
    down_weight = block_quant_dequant(
        module.down_proj.transpose(1, 2),
        module.down_weight_scale_inv.transpose(1, 2),
        [128, 128],
        torch.bfloat16,
    )

    with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
        references = [
            hidden.float().detach().requires_grad_(True),
            routing.float().detach().requires_grad_(True),
            *(factor.float().detach().requires_grad_(True) for factor in effective_factors),
        ]
        reference_hidden, reference_routing = references[:2]
        gate_A, gate_B, up_A, up_B, down_A, down_B = references[2:]
        reference_output = reference_hidden * 0.0

        for local_expert in range(_LOCAL_EXPERTS):
            pair_rows, pair_topk = (local_ids == local_expert).nonzero(as_tuple=True)
            if pair_rows.numel() == 0:
                continue
            global_expert = module.expert_offset + local_expert
            expert_input = reference_hidden.index_select(0, pair_rows)
            base_gate_up = F.linear(expert_input.to(torch.bfloat16), gate_up_weight[local_expert])
            base_gate, base_up = base_gate_up.split(_INTERMEDIATE, dim=-1)
            gate_delta = (expert_input @ gate_A[0]) @ gate_B[global_expert]
            up_delta = (expert_input @ up_A[0]) @ up_B[global_expert]
            gate = (base_gate.float() + gate_delta).to(torch.bfloat16)
            up = (base_up.float() + up_delta).to(torch.bfloat16)
            activated = F.silu(gate.float()).to(torch.bfloat16) * up
            base_down = F.linear(activated, down_weight[local_expert])
            down_delta = (activated.float() @ down_A[global_expert]) @ down_B[0]
            down = (base_down.float() + down_delta).to(torch.bfloat16).float()
            scores = reference_routing[pair_rows, pair_topk].unsqueeze(1)
            reference_output = reference_output.index_add(
                0,
                pair_rows,
                down * scores * routed_scaling_factor,
            )

        return torch.autograd.grad(
            reference_output,
            references,
            grad_outputs=grad_output.float(),
            allow_unused=False,
        )


def test_routed_bank_contract_is_strict_rank_local_ep16_moe_tp1() -> None:
    module = _module(7)

    assert isinstance(module, Glm52NativeBlockFP8Experts)
    assert module.contract_version == GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION
    assert module._glm52_exact_active_lora_component is True
    assert module.fsdp_requires_full_precision is True
    assert module.adapter_gradient_producer_family == "module_managed"
    assert (module.num_experts, module.ep_size, module.num_local_experts, module.moe_tp_size) == (256, 16, 16, 1)
    assert (module.ep_rank, module.expert_offset, module.ep_dispatch) == (7, 112, "alltoall")
    assert module.r == module.active_r == module.lora_alpha == module.active_lora_alpha == 1
    assert module.scaling == 1.0
    assert {name for name, parameter in module.named_parameters() if parameter.requires_grad} == set(
        module.logical_factor_names
    )
    assert all(getattr(module, name).dtype is torch.float32 for name in module.logical_factor_names)
    assert tuple(module.gate_proj_lora_A.shape) == (1, _HIDDEN, 1)
    assert tuple(module.gate_proj_lora_B.shape) == (_GLOBAL_EXPERTS, 1, _INTERMEDIATE)
    assert tuple(module.up_proj_lora_A.shape) == (1, _HIDDEN, 1)
    assert tuple(module.up_proj_lora_B.shape) == (_GLOBAL_EXPERTS, 1, _INTERMEDIATE)
    assert tuple(module.down_proj_lora_A.shape) == (_GLOBAL_EXPERTS, _INTERMEDIATE, 1)
    assert tuple(module.down_proj_lora_B.shape) == (1, 1, _HIDDEN)

    rank_three = Glm52ExactEP16BlockFP8QLoRARoutedExperts(
        _HIDDEN, _INTERMEDIATE, ep_rank=0, r=3, lora_alpha=7, device="meta"
    )
    assert rank_three.gate_proj_lora_A.shape[-1] == 3
    assert rank_three.gate_proj_lora_B.shape[1] == 3
    with pytest.raises(ValueError, match="256 global experts"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            _HIDDEN,
            _INTERMEDIATE,
            ep_rank=0,
            num_experts=128,
        )
    with pytest.raises(ValueError, match="MoE-TP1"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(_HIDDEN, _INTERMEDIATE, ep_rank=0, moe_tp_size=2)
    with pytest.raises(ValueError, match="DeepEP is not admitted"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            _HIDDEN,
            _INTERMEDIATE,
            ep_rank=0,
            ep_dispatch="deepep",
        )
    with pytest.raises(ValueError, match=r"in \[0, 15\]"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(_HIDDEN, _INTERMEDIATE, ep_rank=16)
    module.set_runtime_lora_config(1, 1)
    with pytest.raises(ValueError, match="positive integer rank"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            _HIDDEN, _INTERMEDIATE, ep_rank=0, r=0, device="meta"
        )
    with pytest.raises(ValueError, match="positive integer alpha"):
        module.set_runtime_lora_config(1, 0)


def test_one_batched_global_grid_proves_all_16_owner_by_16_slot_remaps() -> None:
    global_grid = torch.arange(_GLOBAL_EXPERTS, dtype=torch.int64).reshape(_EP_SIZE, _LOCAL_EXPERTS)
    owner_maps = torch.stack(
        [localize_glm52_ep16_expert_ids(global_grid.contiguous(), owner) for owner in range(_EP_SIZE)]
    )
    expected = torch.full((_EP_SIZE, _EP_SIZE, _LOCAL_EXPERTS), -1, dtype=torch.int32)
    slots = torch.arange(_LOCAL_EXPERTS, dtype=torch.int32)
    for owner in range(_EP_SIZE):
        expected[owner, owner] = slots
    assert torch.equal(owner_maps, expected)

    with pytest.raises(TypeError, match="must be int32 or int64"):
        localize_glm52_ep16_expert_ids(global_grid.float().contiguous(), 0)
    with pytest.raises(ValueError, match="contiguous"):
        localize_glm52_ep16_expert_ids(global_grid.T, 0)
    invalid = global_grid.clone()
    invalid[0, 0] = _GLOBAL_EXPERTS
    with pytest.raises(ValueError, match=r"in \[0, 255\]"):
        localize_glm52_ep16_expert_ids(invalid, 0)


def test_all_owner_slot_physical_buffers_are_full_width_bf16_and_not_outer_tp_sliced() -> None:
    for owner in range(_EP_SIZE):
        module = _module(owner)
        _fill_distinguishable_factors(module)
        buffers = module.physical_factor_buffers()
        assert tuple(buffers["gate_up_lora_a_weights"].shape) == (8, 1, 2, _HIDDEN)
        assert tuple(buffers["gate_up_lora_b_weights"].shape) == (8, 16, 2 * _INTERMEDIATE, 1)
        assert tuple(buffers["down_lora_a_weights"].shape) == (8, 16, 1, _INTERMEDIATE)
        assert tuple(buffers["down_lora_b_weights"].shape) == (8, 1, _HIDDEN, 1)
        assert all(buffer.dtype is torch.bfloat16 for buffer in buffers.values())
        assert torch.equal(
            _bits(buffers["gate_up_lora_a_weights"][0, 0, 0]),
            _bits(module.gate_proj_lora_A[:, :, 0].to(torch.bfloat16)[0]),
        )
        assert torch.equal(
            _bits(buffers["gate_up_lora_a_weights"][0, 0, 1]),
            _bits(module.up_proj_lora_A[:, :, 0].to(torch.bfloat16)[0]),
        )
        for slot in range(_LOCAL_EXPERTS):
            global_expert = module.expert_offset + slot
            assert torch.equal(
                _bits(buffers["gate_up_lora_b_weights"][0, slot, :_INTERMEDIATE]),
                _bits(module.gate_proj_lora_B[global_expert].T.to(torch.bfloat16)),
            )
            assert torch.equal(
                _bits(buffers["gate_up_lora_b_weights"][0, slot, _INTERMEDIATE:]),
                _bits(module.up_proj_lora_B[global_expert].T.to(torch.bfloat16)),
            )
            assert torch.equal(
                _bits(buffers["down_lora_a_weights"][0, slot]),
                _bits(module.down_proj_lora_A[global_expert].T.to(torch.bfloat16)),
            )
        assert torch.equal(
            _bits(buffers["down_lora_b_weights"][0, 0]),
            _bits(module.down_proj_lora_B[0].T.to(torch.bfloat16)),
        )
        for buffer in buffers.values():
            assert torch.count_nonzero(buffer[1:]) == 0


def test_post_ep_owner_local_factor_banks_produce_the_same_physical_sampler_views() -> None:
    module = _module(7)
    _fill_distinguishable_factors(module)
    global_factors = tuple(
        getattr(module, name).detach().to(torch.bfloat16).contiguous() for name in module.logical_factor_names
    )
    local_factors = list(global_factors)
    for index in (1, 3, 4):
        local_factors[index] = global_factors[index][112:128].contiguous()

    global_buffers = module._physical_factor_buffers(*global_factors)
    local_buffers = module._physical_factor_buffers(*local_factors)

    assert global_buffers.keys() == local_buffers.keys()
    for name in global_buffers:
        assert torch.equal(global_buffers[name], local_buffers[name]), name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_all_256_owner_slots_run_literal_sampler_partials_routing_outputs_and_logical_vjps() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified GLM-5.2 routed component requires Hopper")
    device = torch.device("cuda")
    global_grid = torch.arange(_GLOBAL_EXPERTS, dtype=torch.int64, device=device).reshape(_EP_SIZE, _LOCAL_EXPERTS)

    for owner in range(_EP_SIZE):
        module = _module(owner, device)
        _load_distinguishable_base(module)
        _fill_distinguishable_factors(module)
        global_ids = global_grid[owner].reshape(_LOCAL_EXPERTS, 1).contiguous()
        hidden = (
            torch.arange(_LOCAL_EXPERTS * _HIDDEN, dtype=torch.float32, device=device)
            .reshape(_LOCAL_EXPERTS, _HIDDEN)
            .remainder_(17)
            .add_(1)
            .div_(32)
            .to(torch.bfloat16)
            .requires_grad_(True)
        )
        routing = ((global_ids.float() + 1) / 512).contiguous().requires_grad_(True)
        trace = module.sampler_value_trace(hidden.detach(), routing.detach(), global_ids)
        output = module(hidden, routing, selected_experts=global_ids)

        assert torch.equal(output.detach(), trace.owner_output)
        assert torch.count_nonzero(trace.gate_up_base) > 0
        assert torch.count_nonzero(trace.down_base_routed) > 0
        assert torch.count_nonzero(trace.gate_up_post_lora) > 0
        assert torch.count_nonzero(trace.activated) > 0
        assert torch.count_nonzero(trace.down_post_lora_routed) > 0
        assert not torch.equal(trace.gate_up_base, trace.gate_up_post_lora)
        assert not torch.equal(trace.down_base_routed, trace.down_post_lora_routed)
        assert torch.equal(output, trace.down_post_lora_routed[:, 0])

        # A positive logical cotangent keeps every intentionally routed slot
        # distinguishable; an alternating cotangent can legitimately cancel
        # one rank-one bank gradient and would not be a dispatch failure.
        grad_output = torch.ones_like(output)
        local_ids = module.localize_global_expert_ids(global_ids)
        effective = tuple(
            getattr(module, name).detach().to(torch.bfloat16).contiguous() for name in module.logical_factor_names
        )
        expected_gradients = module._surrogate_vjp(
            hidden.detach(),
            routing.detach(),
            local_ids,
            *effective,
            grad_output=grad_output,
            routed_scaling_factor=1.0,
            needs_input_grad=(True, True, False, True, True, True, True, True, True),
        )
        assert all(gradient is not None and gradient.dtype is torch.float32 for gradient in expected_gradients)
        output.backward(grad_output)
        assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
        assert routing.grad is not None and routing.grad.dtype is torch.float32
        assert torch.equal(hidden.grad, expected_gradients[0].to(torch.bfloat16))
        assert torch.equal(routing.grad, expected_gradients[1])
        for name, expected in zip(module.logical_factor_names, expected_gradients[2:], strict=True):
            assert torch.equal(getattr(module, name).grad, expected)
        for name in ("gate_proj_lora_B", "up_proj_lora_B", "down_proj_lora_A"):
            gradient = getattr(module, name).grad
            assert gradient is not None
            assert gradient.dtype is torch.float32
            counts = torch.count_nonzero(gradient.reshape(_GLOBAL_EXPERTS, -1), dim=1)
            owner_start = module.expert_offset
            owner_end = owner_start + _LOCAL_EXPERTS
            assert torch.all(counts[owner_start:owner_end] > 0)
            assert torch.count_nonzero(counts[:owner_start]) == 0
            assert torch.count_nonzero(counts[owner_end:]) == 0
        for name in ("gate_proj_lora_A", "up_proj_lora_A", "down_proj_lora_B"):
            gradient = getattr(module, name).grad
            assert gradient is not None and gradient.dtype is torch.float32
            assert torch.count_nonzero(gradient) > 0

        if owner == 0:
            half_trace = module.sampler_value_trace(
                hidden.detach(),
                (routing.detach() * 0.5).contiguous(),
                global_ids,
            )
            assert torch.equal(half_trace.gate_up_post_lora, trace.gate_up_post_lora)
            torch.testing.assert_close(
                half_trace.down_post_lora_routed.float(),
                trace.down_post_lora_routed.float() * 0.5,
                rtol=0,
                atol=2**-8,
            )
            torch.testing.assert_close(
                half_trace.owner_output.float(),
                trace.owner_output.float() * 0.5,
                rtol=0,
                atol=2**-8,
            )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_zero_token_routed_slots_receive_exact_zero_bank_gradients_and_stride_mismatches_fail() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified GLM-5.2 routed component requires Hopper")
    device = torch.device("cuda")
    module = _module(5, device)
    _load_zero_base(module)
    _fill_distinguishable_factors(module)
    hidden = torch.full((2, _HIDDEN), 0.25, dtype=torch.bfloat16, device=device, requires_grad=True)
    global_ids = torch.tensor([[80], [80]], dtype=torch.int64, device=device)
    routing = torch.tensor([[0.75], [0.5]], dtype=torch.float32, device=device, requires_grad=True)
    module(hidden, routing, selected_experts=global_ids).float().sum().backward()

    for name in ("gate_proj_lora_B", "up_proj_lora_B", "down_proj_lora_A"):
        gradient = getattr(module, name).grad
        assert gradient is not None
        assert torch.count_nonzero(gradient[80]) > 0
        assert torch.count_nonzero(gradient[:80]) == 0
        assert torch.count_nonzero(gradient[81:]) == 0

    hidden_backing = torch.zeros((2, 2 * _HIDDEN), dtype=torch.bfloat16, device=device)
    hidden_strided = hidden_backing[:, ::2]
    assert not hidden_strided.is_contiguous()
    with pytest.raises(ValueError, match="non-empty and contiguous"):
        module(hidden_strided, routing.detach(), selected_experts=global_ids)
    routing_backing = torch.ones((2, 2), dtype=torch.float32, device=device)
    routing_strided = routing_backing[:, ::2]
    assert not routing_strided.is_contiguous()
    with pytest.raises(ValueError, match="route-major"):
        module(hidden.detach(), routing_strided, selected_experts=global_ids)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_all_sentinel_owner_returns_exact_zero_and_zero_gradients() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified GLM-5.2 routed component requires Hopper")
    device = torch.device("cuda")
    module = _module(5, device)
    _load_zero_base(module)
    _fill_distinguishable_factors(module)
    hidden = torch.full((2, _HIDDEN), 0.25, dtype=torch.bfloat16, device=device, requires_grad=True)
    routing = torch.tensor([[0.75], [0.5]], dtype=torch.float32, device=device, requires_grad=True)
    unowned_global_ids = torch.tensor([[0], [1]], dtype=torch.int64, device=device)

    output = module(hidden, routing, selected_experts=unowned_global_ids)
    assert torch.count_nonzero(output) == 0
    output.float().sum().backward()

    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    assert routing.grad is not None and torch.count_nonzero(routing.grad) == 0
    for name in module.logical_factor_names:
        gradient = getattr(module, name).grad
        assert gradient is not None and gradient.dtype is torch.float32
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_topk8_mixed_owner_hybrid_vjps_match_standalone_reference() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified GLM-5.2 routed component requires Hopper")
    device = torch.device("cuda")
    module = _module(5, device)
    _load_distinguishable_base(module)
    with torch.no_grad():
        for offset, name in enumerate(module.logical_factor_names):
            parameter = getattr(module, name)
            values = torch.arange(parameter.numel(), dtype=torch.float32, device=device).reshape_as(parameter)
            parameter.copy_(values.remainder(31).sub(15).div(512).add((offset + 1) / 4096))

    hidden = (
        torch.arange(4 * _HIDDEN, dtype=torch.float32, device=device)
        .reshape(4, _HIDDEN)
        .remainder_(23)
        .sub_(11)
        .div_(16)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    global_ids = torch.tensor(
        [
            [80, 1, 81, 2, 82, 3, 83, 4],
            [84, 85, 86, 87, 88, 89, 90, 91],
            [200, 92, 201, 93, 202, 94, 203, 95],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.int64,
        device=device,
    )
    owned = (global_ids >= module.expert_offset) & (global_ids < module.expert_offset + _LOCAL_EXPERTS)
    expected_local_ids = torch.where(owned, global_ids - module.expert_offset, -1).to(torch.int32).contiguous()
    local_ids = module.localize_global_expert_ids(global_ids)
    assert global_ids.shape[1] == 8
    assert torch.equal(local_ids, expected_local_ids)
    assert torch.equal(
        torch.sort(local_ids[local_ids >= 0]).values,
        torch.arange(_LOCAL_EXPERTS, dtype=torch.int32, device=device),
    )
    assert torch.count_nonzero(local_ids == -1) == 16

    routing = (
        torch.arange(global_ids.numel(), dtype=torch.float32, device=device)
        .reshape_as(global_ids)
        .remainder_(13)
        .add_(1)
        .div_(17)
        .contiguous()
        .requires_grad_(True)
    )
    grad_output = (
        torch.arange(hidden.numel(), dtype=torch.float32, device=device)
        .reshape_as(hidden)
        .remainder_(19)
        .sub_(9)
        .div_(16)
        .to(torch.bfloat16)
    )
    routed_scaling_factor = 2.5
    effective_factors = tuple(
        getattr(module, name).detach().to(torch.bfloat16).contiguous() for name in module.logical_factor_names
    )
    expected_gradients = _standalone_hybrid_routed_vjp(
        module,
        hidden.detach(),
        routing.detach(),
        local_ids,
        effective_factors,
        grad_output,
        routed_scaling_factor=routed_scaling_factor,
    )

    trainables = (hidden, routing, *(getattr(module, name) for name in module.logical_factor_names))
    output = module(
        hidden,
        routing,
        selected_experts=global_ids,
        routed_scaling_factor=routed_scaling_factor,
    )
    actual_gradients = torch.autograd.grad(output, trainables, grad_outputs=grad_output)

    assert len(actual_gradients) == len(expected_gradients) == 8
    assert torch.equal(actual_gradients[0], expected_gradients[0].to(torch.bfloat16))
    for actual, expected in zip(actual_gradients[1:], expected_gradients[1:], strict=True):
        assert actual.dtype is torch.float32
        assert torch.equal(actual, expected)

    assert torch.count_nonzero(actual_gradients[0][3]) == 0
    assert torch.count_nonzero(actual_gradients[1][~owned]) == 0
    assert torch.count_nonzero(actual_gradients[1][owned]) == owned.sum()
    for index in (3, 5, 6):
        gradient = actual_gradients[index]
        per_expert_nonzero = torch.count_nonzero(gradient.reshape(_GLOBAL_EXPERTS, -1), dim=1)
        assert torch.all(per_expert_nonzero[module.expert_offset : module.expert_offset + _LOCAL_EXPERTS] > 0)
        assert torch.count_nonzero(per_expert_nonzero[: module.expert_offset]) == 0
        assert torch.count_nonzero(per_expert_nonzero[module.expert_offset + _LOCAL_EXPERTS :]) == 0
