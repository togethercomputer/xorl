from __future__ import annotations

import math

import pytest
import torch

from xorl.lora.utils import get_lora_state_dict
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    GLM52_EXACT_TP1_ABSORBED_KV_B_QLORA_CONTRACT_VERSION,
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.ops.block_fp8_native import NativeBlockFP8Linear


_NUM_HEADS = 64
_QK_NOPE_HEAD_DIM = 192
_V_HEAD_DIM = 256
_KV_LORA_RANK = 512
_OUT_FEATURES = _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM)
_GRAPH_LORA_SLOTS = 8


def _fill_factor(parameter: torch.Tensor, *, modulus: int, center: int, divisor: int) -> None:
    with torch.no_grad():
        parameter.copy_(
            torch.arange(parameter.numel(), dtype=torch.float32, device=parameter.device)
            .reshape_as(parameter)
            .remainder_(modulus)
            .sub_(center)
            .div_(divisor)
        )


def _module(device: torch.device | str = "cpu") -> Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA:
    module = Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(device=device)
    _fill_factor(module.lora_A, modulus=257, center=128, divisor=1021)
    _fill_factor(module.lora_B, modulus=251, center=125, divisor=2053)
    return module


def _pattern(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    modulus: int,
    center: int,
    divisor: int,
) -> torch.Tensor:
    return (
        torch.arange(math.prod(shape), dtype=torch.float32, device=device)
        .reshape(shape)
        .remainder_(modulus)
        .sub_(center)
        .div_(divisor)
        .to(torch.bfloat16)
    )


def test_absorbed_kv_b_contract_keeps_one_frozen_native_base_and_two_logical_masters() -> None:
    module = _module()

    assert isinstance(module, NativeBlockFP8Linear)
    assert module.contract_version == GLM52_EXACT_TP1_ABSORBED_KV_B_QLORA_CONTRACT_VERSION
    assert module._glm52_exact_active_lora_component is True
    assert module.adapter_gradient_producer_family == "module_managed"
    assert module.fsdp_requires_full_precision is True
    assert module.max_lora_rank == 1
    assert module.fixed_graph_lora_slots == _GRAPH_LORA_SLOTS
    assert (module.in_features, module.out_features) == (_KV_LORA_RANK, _OUT_FEATURES)
    assert (module.num_heads, module.qk_nope_head_dim, module.v_head_dim, module.kv_lora_rank) == (
        _NUM_HEADS,
        _QK_NOPE_HEAD_DIM,
        _V_HEAD_DIM,
        _KV_LORA_RANK,
    )
    assert module.r == module.active_r == module.lora_alpha == module.active_lora_alpha == 1
    assert module.scaling == 1.0
    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert trainable == set(module.logical_factor_names) == {"lora_A", "lora_B"}
    assert module.lora_A.dtype is torch.float32 and tuple(module.lora_A.shape) == (1, _KV_LORA_RANK)
    assert module.lora_B.dtype is torch.float32 and tuple(module.lora_B.shape) == (_OUT_FEATURES, 1)
    assert module.packed_weight_f32.requires_grad is False
    assert module.weight_scale_inv.requires_grad is False
    exported = get_lora_state_dict(module)
    assert tuple(exported) == module.logical_factor_names
    assert torch.equal(exported["lora_A"], module.lora_A)
    assert torch.equal(exported["lora_B"], module.lora_B)

    with pytest.raises(ValueError, match="only official"):
        Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(num_heads=32)
    rank_three = Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(r=3, lora_alpha=7, device="meta")
    assert rank_three.lora_A.shape == (3, 512)
    assert rank_three.lora_B.shape[-1] == 3
    with pytest.raises(ValueError, match="bias-free"):
        Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(bias=True)
    with pytest.raises(ValueError, match="only effective TP1"):
        Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(tp_size=2)
    module.set_runtime_lora_config(1, 1)
    with pytest.raises(ValueError, match="positive integer rank"):
        Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(r=0, device="meta")
    with pytest.raises(ValueError, match="positive integer alpha"):
        module.set_runtime_lora_config(1, 0)


def test_absorbed_kv_b_dtype_move_preserves_native_bytes_master_values_and_identity() -> None:
    module = _module()
    with torch.no_grad():
        module.packed_weight_f32.copy_(
            torch.arange(module.packed_weight_f32.numel(), dtype=torch.float32).reshape_as(module.packed_weight_f32)
        )
        module.weight_scale_inv.copy_(
            torch.arange(module.weight_scale_inv.numel(), dtype=torch.float32).reshape_as(module.weight_scale_inv)
            + 0.25
        )
    expected = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    identities = {name: id(parameter) for name, parameter in module.named_parameters()}

    module.to(dtype=torch.bfloat16)

    for name, parameter in module.named_parameters():
        assert parameter.dtype is torch.float32
        assert id(parameter) == identities[name]
        assert torch.equal(parameter, expected[name])


def test_absorbed_kv_b_rejects_direct_projection_and_castable_factor_state() -> None:
    module = _module()

    with pytest.raises(RuntimeError, match="cannot run as a direct projection"):
        module(torch.zeros(1, _KV_LORA_RANK, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="cannot run as a direct projection"):
        module.forward_partition(torch.zeros(1, _KV_LORA_RANK, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="requires an activation"):
        module(branch="q")
    with pytest.raises(ValueError, match="must be 'q' or 'v'"):
        module(torch.zeros(1, _NUM_HEADS, _KV_LORA_RANK), branch="ordinary")
    with pytest.raises(RuntimeError, match="materialization requires CUDA"):
        module(return_dequantized_weight=True)

    state = module.state_dict()
    state["lora_A"] = state["lora_A"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="lora_A must be FP32"):
        module.load_state_dict(state)


def _manual_lora_vjps(
    q_nope: torch.Tensor,
    attn_latent: torch.Tensor,
    effective_A: torch.Tensor,
    effective_B: torch.Tensor,
    grad_q: torch.Tensor,
    grad_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q_nope.float()
    attn = attn_latent.float()
    A = effective_A.float()
    B = effective_B.float().view(_NUM_HEADS, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM, 1)
    B_k = B[:, :_QK_NOPE_HEAD_DIM]
    B_v = B[:, _QK_NOPE_HEAD_DIM:]
    gq = grad_q.float()
    gv = grad_v.float()

    q_low = torch.einsum("shd,hdr->shr", q, B_k)
    d_q_low = torch.einsum("shc,rc->shr", gq, A)
    grad_q_nope = torch.einsum("shr,hdr->shd", d_q_low, B_k)
    grad_B_k = torch.einsum("shd,shr->hdr", q, d_q_low)
    grad_A_q = torch.einsum("shr,shc->rc", q_low, gq)
    grad_B_q = torch.zeros_like(B)
    grad_B_q[:, :_QK_NOPE_HEAD_DIM].copy_(grad_B_k)

    v_low = torch.einsum("shc,rc->shr", attn, A)
    d_v_low = torch.einsum("shd,hdr->shr", gv, B_v)
    grad_attn = torch.einsum("shr,rc->shc", d_v_low, A)
    grad_B_v_values = torch.einsum("shd,shr->hdr", gv, v_low)
    grad_A_v = torch.einsum("shr,shc->rc", d_v_low, attn)
    grad_B_v = torch.zeros_like(B)
    grad_B_v[:, _QK_NOPE_HEAD_DIM:].copy_(grad_B_v_values)
    return (
        grad_q_nope,
        grad_attn,
        grad_A_q,
        grad_B_q.reshape(_OUT_FEATURES, 1),
        grad_A_v,
        grad_B_v.reshape(_OUT_FEATURES, 1),
    )


def _assert_fp32_surrogate_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30)
    assert relative_l2 < 1e-6
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_official_absorbed_kv_b_full_s4_q_v_program_graph_metadata_and_summed_vjp() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 absorbed component requires Hopper")
    from sglang.kernels.ops.gemm.kv_b_lora_absorbed import (
        step_a_q_fwd,
        step_a_v_fwd,
        step_b_q_fwd,
        step_b_v_fwd,
    )
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend

    device = torch.device("cuda")
    batch_size, sequence_length = 2, 2
    rows = batch_size * sequence_length
    module = _module(device)
    weight = (
        torch.arange(_OUT_FEATURES * _KV_LORA_RANK, dtype=torch.float32, device=device)
        .remainder_(31)
        .sub_(15)
        .div_(16)
        .to(torch.float8_e4m3fn)
        .reshape(_OUT_FEATURES, _KV_LORA_RANK)
    )
    scales = (
        torch.arange(module.weight_scale_inv.numel(), dtype=torch.float32, device=device)
        .remainder_(13)
        .add_(1)
        .div_(64)
        .reshape_as(module.weight_scale_inv)
    )
    module.load_prequantized(weight, scales)
    frozen_base = {
        "packed_weight_f32": module.packed_weight_f32.detach().view(torch.uint8).clone(),
        "weight_scale_inv": module.weight_scale_inv.detach().clone(),
    }

    q_backing = _pattern(
        (batch_size, sequence_length, _NUM_HEADS, _QK_NOPE_HEAD_DIM + 64),
        device=device,
        modulus=127,
        center=63,
        divisor=67,
    )
    q_nope = q_backing[..., :_QK_NOPE_HEAD_DIM]
    assert not q_nope.is_contiguous()
    attn_latent = _pattern(
        (batch_size, sequence_length, _NUM_HEADS, _KV_LORA_RANK),
        device=device,
        modulus=113,
        center=56,
        divisor=79,
    )
    q_flat = q_nope.reshape(rows, _NUM_HEADS, _QK_NOPE_HEAD_DIM)
    attn_flat = attn_latent.reshape(rows, _NUM_HEADS, _KV_LORA_RANK)
    assert q_flat.stride() == (_NUM_HEADS * (_QK_NOPE_HEAD_DIM + 64), _QK_NOPE_HEAD_DIM + 64, 1)
    assert attn_latent.is_contiguous()

    # Build the real fixed decode-graph metadata: eight adapter segment slots,
    # one live slot0 rank-one adapter, and seven padded rank-zero slots.
    backend = TritonLoRABackend(max_loras_per_batch=_GRAPH_LORA_SLOTS, device=device)
    backend.init_cuda_graph_batch_info(max_bs_in_cuda_graph=rows, num_tokens_per_req=1)
    backend.batch_info = backend.cuda_graph_batch_info
    backend.batch_info.weight_indices[:rows].zero_()
    backend.batch_info.lora_ranks.zero_()
    backend.batch_info.lora_ranks[0] = 1
    backend.batch_info.scalings.zero_()
    backend.batch_info.scalings[0] = 1.0
    backend.compute_sgemm_routing(use_cuda_graph=True)
    batch_info = backend.sgemm_batch_info
    assert batch_info is backend.cuda_graph_sgemm_batch_info
    assert batch_info.use_cuda_graph is True
    assert batch_info.num_segments == batch_info.bs == _GRAPH_LORA_SLOTS
    assert torch.equal(
        batch_info.seg_lens,
        torch.tensor([rows] + [0] * 7, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        batch_info.seg_indptr,
        torch.tensor([0] + [rows] * 8, dtype=torch.int32, device=device),
    )
    assert torch.equal(batch_info.weight_indices, torch.arange(8, dtype=torch.int32, device=device))
    assert torch.equal(batch_info.lora_ranks, torch.tensor([1] + [0] * 7, dtype=torch.int32, device=device))
    assert torch.equal(batch_info.scalings, torch.tensor([1.0] + [0.0] * 7, device=device))
    assert torch.equal(batch_info.permutation, torch.arange(rows, dtype=torch.int32, device=device))

    # Independent full raw-S4 oracle, including post-load block dequantization,
    # the exact w_kc/w_vc stride construction, base BMM order, and correction.
    dequantized = block_quant_dequant(
        module.fp8_weight(),
        module.weight_scale_inv,
        [128, 128],
        torch.bfloat16,
    )
    w_kc, w_vc = dequantized.unflatten(
        0,
        (_NUM_HEADS, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM),
    ).split([_QK_NOPE_HEAD_DIM, _V_HEAD_DIM], dim=1)
    w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    w_vc = w_vc.contiguous().transpose(1, 2)
    assert tuple(w_kc.shape) == (_NUM_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK)
    assert w_kc.stride() == (_QK_NOPE_HEAD_DIM * _KV_LORA_RANK, 1, _QK_NOPE_HEAD_DIM)
    assert tuple(w_vc.shape) == (_NUM_HEADS, _KV_LORA_RANK, _V_HEAD_DIM)
    assert w_vc.stride() == (_KV_LORA_RANK * _V_HEAD_DIM, 1, _KV_LORA_RANK)

    effective_A = module.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_B = module.lora_B.detach().to(torch.bfloat16).contiguous()
    A_buffer = torch.zeros(
        (_GRAPH_LORA_SLOTS, 1, _KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=device,
    )
    B_buffer = torch.zeros(
        (_GRAPH_LORA_SLOTS, _OUT_FEATURES, 1),
        dtype=torch.bfloat16,
        device=device,
    )
    A_buffer[0].copy_(effective_A)
    B_buffer[0].copy_(effective_B)
    assert A_buffer.shape[1] == B_buffer.shape[2] == 1
    assert torch.count_nonzero(A_buffer[1:]) == torch.count_nonzero(B_buffer[1:]) == 0

    q_base = torch.bmm(q_flat.transpose(0, 1), w_kc).transpose(0, 1)
    expected_q = q_base.clone(memory_format=torch.preserve_format)
    q_low = step_a_q_fwd(q_flat, B_buffer, batch_info, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM)
    step_b_q_fwd(q_low, A_buffer, batch_info, expected_q)
    v_base_flat = torch.empty((rows, _NUM_HEADS * _V_HEAD_DIM), dtype=torch.bfloat16, device=device)
    v_base = v_base_flat.view(rows, _NUM_HEADS, _V_HEAD_DIM)
    torch.bmm(attn_flat.transpose(0, 1), w_vc, out=v_base.transpose(0, 1))
    expected_v = v_base.clone()
    v_low = step_a_v_fwd(attn_flat, A_buffer, batch_info)
    step_b_v_fwd(v_low, B_buffer, batch_info, expected_v, _QK_NOPE_HEAD_DIM, _V_HEAD_DIM)

    actual_q = module(q_nope, branch="q", batch_info=batch_info)
    actual_v = module(attn_latent, branch="v", batch_info=batch_info)
    torch.cuda.synchronize()
    assert tuple(actual_q.shape) == (batch_size, sequence_length, _NUM_HEADS, _KV_LORA_RANK)
    assert tuple(actual_v.shape) == (batch_size, sequence_length, _NUM_HEADS, _V_HEAD_DIM)
    assert torch.equal(actual_q.reshape_as(expected_q).view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_v.reshape_as(expected_v).view(torch.uint8), expected_v.view(torch.uint8))
    assert torch.count_nonzero(expected_q - q_base) > 0
    assert torch.count_nonzero(expected_v - v_base) > 0

    # Trainer calls need no model-level SGLang routing input. Its cached eager
    # slot0 metadata must produce the same admitted rank-one bytes.
    internal_q = module(q_nope, branch="q")
    internal_v = module(attn_latent, branch="v")
    assert torch.equal(internal_q.view(torch.uint8), actual_q.view(torch.uint8))
    assert torch.equal(internal_v.view(torch.uint8), actual_v.view(torch.uint8))
    materialized = module(return_dequantized_weight=True)
    assert torch.equal(materialized.view(torch.uint8), dequantized.view(torch.uint8))
    assert torch.equal(module.packed_weight_f32.detach().view(torch.uint8), frozen_base["packed_weight_f32"])
    assert torch.equal(module.weight_scale_inv, frozen_base["weight_scale_inv"])

    grad_q = _pattern(
        (rows, _NUM_HEADS, _KV_LORA_RANK),
        device=device,
        modulus=97,
        center=48,
        divisor=71,
    )
    grad_v = _pattern(
        (rows, _NUM_HEADS, _V_HEAD_DIM),
        device=device,
        modulus=89,
        center=44,
        divisor=73,
    )
    q_vjp = module._surrogate_q_vjp(
        q_flat,
        effective_A,
        effective_B,
        grad_q,
        needs_input_grad=(True, True, True),
    )
    v_vjp = module._surrogate_v_vjp(
        attn_flat,
        effective_A,
        effective_B,
        grad_v,
        needs_input_grad=(True, True, True),
    )
    lora_dq, lora_dv, q_dA, q_dB, v_dA, v_dB = _manual_lora_vjps(
        q_flat,
        attn_flat,
        effective_A,
        effective_B,
        grad_q,
        grad_v,
    )
    q_base_input = q_flat.detach().requires_grad_(True)
    q_base_reference = torch.bmm(q_base_input.transpose(0, 1), w_kc).transpose(0, 1)
    (q_base_dx,) = torch.autograd.grad(q_base_reference, q_base_input, grad_outputs=grad_q)
    v_base_input = attn_flat.detach().requires_grad_(True)
    v_base_reference = torch.bmm(v_base_input.transpose(0, 1), w_vc).transpose(0, 1)
    (v_base_dx,) = torch.autograd.grad(v_base_reference, v_base_input, grad_outputs=grad_v)
    _assert_fp32_surrogate_close(q_vjp[0], q_base_dx.float() + lora_dq)
    _assert_fp32_surrogate_close(v_vjp[0], v_base_dx.float() + lora_dv)
    _assert_fp32_surrogate_close(q_vjp[1], q_dA)
    _assert_fp32_surrogate_close(q_vjp[2], q_dB)
    _assert_fp32_surrogate_close(v_vjp[1], v_dA)
    _assert_fp32_surrogate_close(v_vjp[2], v_dB)
    assert torch.count_nonzero(q_vjp[1]) > 0 and torch.count_nonzero(v_vjp[1]) > 0
    assert torch.count_nonzero(q_vjp[2]) > 0 and torch.count_nonzero(v_vjp[2]) > 0

    # Both real call sites enter Module.__call__, creating two custom-function
    # nodes whose shared factor gradients must accumulate in ordinary autograd.
    q_leaf = q_nope.detach().requires_grad_(True)
    v_leaf = attn_latent.detach().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    output_q = module(q_leaf, branch="q", batch_info=batch_info)
    output_v = module(v_leaf, branch="v", batch_info=batch_info)
    torch.autograd.backward(
        (output_q, output_v),
        (
            grad_q.reshape_as(output_q),
            grad_v.reshape_as(output_v),
        ),
    )
    assert torch.equal(q_leaf.grad, q_vjp[0].reshape_as(q_leaf).to(torch.bfloat16))
    assert torch.equal(v_leaf.grad, v_vjp[0].reshape_as(v_leaf).to(torch.bfloat16))
    torch.testing.assert_close(module.lora_A.grad, q_vjp[1] + v_vjp[1], rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(module.lora_B.grad, q_vjp[2] + v_vjp[2], rtol=5e-5, atol=2e-6)

    # Wider-rank padded graph slots are outside the rank-one contract.
    batch_info.lora_ranks[1] = 1
    try:
        with pytest.raises(ValueError, match="padded slot rank0/scale0"):
            module(q_nope, branch="q", batch_info=batch_info)
    finally:
        batch_info.lora_ranks[1] = 0
    with pytest.raises(ValueError, match="official 192-wide slice"):
        module(q_nope.contiguous(), branch="q")
