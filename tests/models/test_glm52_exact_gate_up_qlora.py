from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xorl.models.transformers.glm5.exact_gate_up_qlora import (
    GLM52_EXACT_TP1_GATE_UP_QLORA_CONTRACT_VERSION,
    Glm52ExactTP1FusedGateUpBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.ops.block_fp8_native import NativeBlockFP8Linear
from xorl.ops.fused_silu_and_mul import exact_fp32_silu_and_mul


def _module() -> Glm52ExactTP1FusedGateUpBlockFP8QLoRA:
    module = Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, device=torch.device("cpu"))
    with torch.no_grad():
        module.gate_proj.lora_A.copy_(
            torch.tensor([[0.1001, -0.2002, 0.3003, -0.4004, 0.5005, -0.6006, 0.7007, -0.8008]])
        )
        module.up_proj.lora_A.copy_(
            torch.tensor([[-0.8108, 0.7107, -0.6106, 0.5105, -0.4104, 0.3103, -0.2102, 0.1101]])
        )
        module.gate_proj.lora_B.copy_(torch.arange(128, dtype=torch.float32).sub_(47).div_(311).unsqueeze(1))
        module.up_proj.lora_B.copy_(torch.arange(128, dtype=torch.float32).sub_(73).div_(277).neg_().unsqueeze(1))
    return module


def _literal_cpu_value(base_weight: torch.Tensor, captures: list):
    def run(input, gate_A, gate_B, up_A, up_B):
        captures.append(tuple(factor.detach().clone() for factor in (gate_A, gate_B, up_A, up_B)))
        base = F.linear(input.float(), base_weight.float()).to(torch.bfloat16)
        gate_a_output = F.linear(input.float(), gate_A.float()).to(torch.bfloat16)
        gate_delta = F.linear(gate_a_output.float(), gate_B.float()).to(torch.bfloat16)
        up_a_output = F.linear(input.float(), up_A.float()).to(torch.bfloat16)
        up_delta = F.linear(up_a_output.float(), up_B.float()).to(torch.bfloat16)
        return (base + torch.cat((gate_delta, up_delta), dim=-1)).to(torch.bfloat16)

    return run


def _literal_cpu_linear_value(base_weight: torch.Tensor):
    def run(input, effective_A, effective_B):
        base = F.linear(input.float(), base_weight.float()).to(torch.bfloat16)
        a_output = F.linear(input.float(), effective_A.float()).to(torch.bfloat16)
        delta = F.linear(a_output.float(), effective_B.float()).to(torch.bfloat16)
        return (base + delta).to(torch.bfloat16)

    return run


def _load_small_base(module: Glm52ExactTP1FusedGateUpBlockFP8QLoRA) -> tuple[torch.Tensor, ...]:
    gate_weight = torch.full((128, 8), 0.5, dtype=torch.float8_e4m3fn)
    up_weight = torch.full((128, 8), -0.25, dtype=torch.float8_e4m3fn)
    gate_scales = torch.tensor([[0.125]], dtype=torch.float32)
    up_scales = torch.tensor([[0.375]], dtype=torch.float32)
    module.load_gate_up_prequantized(gate_weight, gate_scales, up_weight, up_scales)
    return gate_weight, gate_scales, up_weight, up_scales


def test_fused_gate_up_contract_is_one_native_leaf_with_four_logical_fp32_factors() -> None:
    module = _module()

    assert isinstance(module, NativeBlockFP8Linear)
    assert module.contract_version == GLM52_EXACT_TP1_GATE_UP_QLORA_CONTRACT_VERSION
    assert module.fsdp_requires_full_precision is True
    assert module.out_features == 256
    assert module.r == module.active_r == module.lora_alpha == module.active_lora_alpha == 1
    assert module.scaling == 1.0
    assert not hasattr(module, "base")
    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert trainable == set(module.logical_factor_names)
    assert all(dict(module.named_parameters())[name].dtype is torch.float32 for name in trainable)
    assert module.gate_proj.adapter_gradient_producer_family == "module_managed"
    assert module.up_proj.adapter_gradient_producer_family == "module_managed"
    assert module.packed_weight_f32.requires_grad is False
    assert module.weight_scale_inv.requires_grad is False

    rank_three = Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, r=3, lora_alpha=7)
    assert rank_three.gate_proj.lora_A.shape == (3, 8)
    assert rank_three.gate_proj.lora_B.shape == (128, 3)
    with pytest.raises(ValueError, match="bias-free"):
        Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, bias=True)
    with pytest.raises(ValueError, match="rejects adaptive"):
        Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, enable_aqn=True)
    with pytest.raises(ValueError, match="only effective TP1"):
        Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, tp_size=2)
    with pytest.raises(ValueError, match="multiple of 128"):
        Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 127)
    with pytest.raises(RuntimeError, match="explicit gate/up pair"):
        Glm52ExactTP1FusedGateUpBlockFP8QLoRA.from_linear(nn.Linear(8, 128, bias=False))
    module.set_runtime_lora_config(1, 1)
    with pytest.raises(ValueError, match="positive integer rank"):
        Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, r=0)
    with pytest.raises(ValueError, match="positive integer alpha"):
        module.set_runtime_lora_config(1, 0)


def test_fused_gate_up_loader_makes_gate_then_up_order_explicit_and_strict() -> None:
    module = _module()
    gate_weight, gate_scales, up_weight, up_scales = _load_small_base(module)

    fused = module.fp8_weight()
    assert torch.equal(fused[:128].view(torch.uint8), gate_weight.view(torch.uint8))
    assert torch.equal(fused[128:].view(torch.uint8), up_weight.view(torch.uint8))
    assert torch.equal(module.weight_scale_inv, torch.cat((gate_scales, up_scales), dim=0))

    with pytest.raises(RuntimeError, match="row order is explicit"):
        module.load_prequantized(fused, module.weight_scale_inv)
    with pytest.raises(TypeError, match="gate_weight must remain"):
        module.load_gate_up_prequantized(gate_weight.float(), gate_scales, up_weight, up_scales)
    with pytest.raises(ValueError, match="up_weight shape"):
        module.load_gate_up_prequantized(gate_weight, gate_scales, up_weight[:127], up_scales)
    with pytest.raises(TypeError, match="up_weight_scale_inv must remain FP32"):
        module.load_gate_up_prequantized(gate_weight, gate_scales, up_weight, up_scales.to(torch.bfloat16))


def test_fused_gate_up_model_dtype_move_preserves_native_state_and_fp32_masters() -> None:
    module = _module()
    _load_small_base(module)
    packed_bytes = module.packed_weight_f32.detach().view(torch.uint8).clone()
    factor_values = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if name in module.logical_factor_names
    }
    factor_ids = {
        name: id(parameter) for name, parameter in module.named_parameters() if name in module.logical_factor_names
    }

    module.to(dtype=torch.bfloat16)

    assert module.packed_weight_f32.dtype is torch.float32
    assert module.weight_scale_inv.dtype is torch.float32
    assert torch.equal(module.packed_weight_f32.detach().view(torch.uint8), packed_bytes)
    for name, expected in factor_values.items():
        actual = dict(module.named_parameters())[name]
        assert id(actual) == factor_ids[name]
        assert actual.dtype is torch.float32
        assert torch.equal(actual, expected)


def test_fused_gate_up_rounds_each_master_once_and_preserves_logical_order(monkeypatch) -> None:
    module = _module()
    base_weight = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8).sub_(311).div_(977).to(torch.bfloat16)
    captures = []
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, captures))
    input = torch.arange(24, dtype=torch.float32).reshape(3, 8).sub_(7).div_(53).to(torch.bfloat16)

    actual = module(input)

    effective_factors = captures.pop()
    masters = (
        module.gate_proj.lora_A,
        module.gate_proj.lora_B,
        module.up_proj.lora_A,
        module.up_proj.lora_B,
    )
    for effective, master in zip(effective_factors, masters, strict=True):
        assert torch.equal(effective, master.detach().to(torch.bfloat16))
    expected = _literal_cpu_value(base_weight, [])(input, *effective_factors)
    assert torch.equal(actual, expected)


def test_fused_gate_up_surrogate_matches_two_logical_qlora_branches(monkeypatch) -> None:
    module = _module()
    base_weight = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8).sub_(617).div_(1237).to(torch.bfloat16)
    monkeypatch.setattr(module, "_dequantize_base_weight", lambda: base_weight)
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, []))
    input = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8).sub_(11).div_(37).to(torch.bfloat16)
    input.requires_grad_(True)
    grad_output = (
        torch.arange(2 * 2 * 256, dtype=torch.float32).reshape(2, 2, 256).remainder_(71).sub_(35).div_(73)
    ).to(torch.bfloat16)
    effective = tuple(
        factor.detach().to(torch.bfloat16)
        for factor in (
            module.gate_proj.lora_A,
            module.gate_proj.lora_B,
            module.up_proj.lora_A,
            module.up_proj.lora_B,
        )
    )

    logical = module._surrogate_vjp(
        input.detach(),
        *effective,
        grad_output,
        needs_input_grad=(True, True, True, True, True),
    )

    gate_base_input = input.detach().clone().requires_grad_(True)
    up_base_input = input.detach().clone().requires_grad_(True)
    gate_base_output = F.linear(gate_base_input, base_weight[:128])
    up_base_output = F.linear(up_base_input, base_weight[128:])
    gate_grad_output, up_grad_output = grad_output.split(128, dim=-1)
    torch.autograd.backward((gate_base_output, up_base_output), (gate_grad_output, up_grad_output))
    gate_lora_input = input.detach().float().requires_grad_(True)
    up_lora_input = input.detach().float().requires_grad_(True)
    reference_factors = tuple(factor.float().requires_grad_(True) for factor in effective)
    gate_output = F.linear(F.linear(gate_lora_input, reference_factors[0]), reference_factors[1])
    up_output = F.linear(F.linear(up_lora_input, reference_factors[2]), reference_factors[3])
    torch.cat((gate_output, up_output), dim=-1).backward(grad_output.float())
    expected_gate_dx = gate_base_input.grad.float() + gate_lora_input.grad
    expected_up_dx = up_base_input.grad.float() + up_lora_input.grad
    expected_dx = expected_gate_dx.to(torch.bfloat16) + expected_up_dx.to(torch.bfloat16)

    assert logical[0].dtype is torch.bfloat16
    assert torch.equal(logical[0], expected_dx)
    for actual_gradient, reference_factor in zip(logical[1:], reference_factors, strict=True):
        assert torch.equal(actual_gradient, reference_factor.grad)

    output = module(input)
    output.backward(grad_output)
    assert torch.equal(input.grad, expected_dx)
    for master, reference_factor in zip(
        (
            module.gate_proj.lora_A,
            module.gate_proj.lora_B,
            module.up_proj.lora_A,
            module.up_proj.lora_B,
        ),
        reference_factors,
        strict=True,
    ):
        assert torch.equal(master.grad, reference_factor.grad)


def test_fused_gate_up_input_gradient_matches_two_exact_logical_wrappers(monkeypatch) -> None:
    fused = _module()
    base_weight = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8).sub_(503).mul_(7).to(torch.bfloat16)
    monkeypatch.setattr(fused, "_dequantize_base_weight", lambda: base_weight)
    monkeypatch.setattr(fused, "_exact_forward_value", _literal_cpu_value(base_weight, []))
    gate = Glm52ExactTP1BlockFP8QLoRALinear(8, 128, device=torch.device("cpu"))
    up = Glm52ExactTP1BlockFP8QLoRALinear(8, 128, device=torch.device("cpu"))
    with torch.no_grad():
        gate.lora_A.copy_(fused.gate_proj.lora_A)
        gate.lora_B.copy_(fused.gate_proj.lora_B)
        up.lora_A.copy_(fused.up_proj.lora_A)
        up.lora_B.copy_(fused.up_proj.lora_B)
    monkeypatch.setattr(gate, "_dequantize_weight", lambda: base_weight[:128])
    monkeypatch.setattr(up, "_dequantize_weight", lambda: base_weight[128:])
    monkeypatch.setattr(gate, "_exact_forward_value", _literal_cpu_linear_value(base_weight[:128]))
    monkeypatch.setattr(up, "_exact_forward_value", _literal_cpu_linear_value(base_weight[128:]))

    values = torch.arange(24, dtype=torch.float32).reshape(3, 8).sub_(9).mul_(5).to(torch.bfloat16)
    fused_input = values.detach().clone().requires_grad_(True)
    reference_input = values.detach().clone().requires_grad_(True)
    grad_output = torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256).sub_(191).mul_(3).to(torch.bfloat16)

    fused(fused_input).backward(grad_output)
    torch.cat((gate(reference_input), up(reference_input)), dim=-1).backward(grad_output)

    assert torch.equal(fused_input.grad, reference_input.grad)
    assert torch.equal(fused.gate_proj.lora_A.grad, gate.lora_A.grad)
    assert torch.equal(fused.gate_proj.lora_B.grad, gate.lora_B.grad)
    assert torch.equal(fused.up_proj.lora_A.grad, up.lora_A.grad)
    assert torch.equal(fused.up_proj.lora_B.grad, up.lora_B.grad)


def test_fused_gate_up_factor_only_backward_does_not_materialize_base(monkeypatch) -> None:
    module = _module()
    base_weight = torch.zeros(256, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, []))
    monkeypatch.setattr(
        module,
        "_dequantize_base_weight",
        lambda: pytest.fail("factor-only VJP must not materialize the frozen fused base"),
    )

    module(torch.ones(2, 8, dtype=torch.bfloat16)).float().sum().backward()

    assert all(dict(module.named_parameters())[name].grad is not None for name in module.logical_factor_names)


def test_fused_gate_up_backward_rejects_any_master_mutation(monkeypatch) -> None:
    module = _module()
    base_weight = torch.zeros(256, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_dequantize_base_weight", lambda: base_weight)
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, []))
    output = module(torch.ones(2, 8, dtype=torch.bfloat16))

    with torch.no_grad():
        module.up_proj.lora_B.add_(1)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.float().sum().backward()


def test_fused_gate_up_contract_fails_closed_before_sglang_import() -> None:
    before = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    module = _module()

    with pytest.raises(TypeError, match="requires BF16 activations"):
        module(torch.zeros(1, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match="contiguous sampler-layout"):
        module(torch.zeros(8, 2, dtype=torch.bfloat16).transpose(0, 1))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module(torch.zeros(1, 8, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="cannot bypass active LoRA"):
        module.forward_partition(torch.zeros(1, 8, dtype=torch.bfloat16))
    module.gate_proj.lora_A = nn.Parameter(module.gate_proj.lora_A.to(torch.bfloat16))
    with pytest.raises(TypeError, match="gate_proj.lora_A must remain FP32"):
        module(torch.zeros(1, 8, dtype=torch.bfloat16))

    after = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    assert after == before


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_official_fused_gate_up_literal_bytes_graph_metadata_zero_and_gradients() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.srt.batch_invariant_ops.bi_silu_and_mul import fp32_silu_and_mul
    from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    from sglang.srt.lora.utils import LoRABatchInfo

    device = torch.device("cuda")
    rows, in_features, intermediate_size = 17, 6144, 12288
    gate_weight = torch.full(
        (intermediate_size, in_features),
        0.25,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    up_weight = torch.full(
        (intermediate_size, in_features),
        -0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    scale_shape = (intermediate_size // 128, in_features // 128)
    gate_scales = torch.full(scale_shape, 0.03125, dtype=torch.float32, device=device)
    up_scales = torch.full(scale_shape, 0.0625, dtype=torch.float32, device=device)
    module = Glm52ExactTP1FusedGateUpBlockFP8QLoRA(
        in_features,
        intermediate_size,
        device=device,
    )
    module.load_gate_up_prequantized(gate_weight, gate_scales, up_weight, up_scales)
    with torch.no_grad():
        module.gate_proj.lora_A.copy_(
            torch.arange(in_features, dtype=torch.float32, device=device)
            .remainder_(37)
            .sub_(18)
            .div_(1024)
            .unsqueeze(0)
        )
        module.up_proj.lora_A.copy_(
            torch.arange(in_features, dtype=torch.float32, device=device)
            .remainder_(43)
            .sub_(21)
            .div_(1536)
            .unsqueeze(0)
        )
        module.gate_proj.lora_B.copy_(
            torch.arange(intermediate_size, dtype=torch.float32, device=device)
            .remainder_(47)
            .sub_(23)
            .div_(2048)
            .unsqueeze(1)
        )
        module.up_proj.lora_B.copy_(
            torch.arange(intermediate_size, dtype=torch.float32, device=device)
            .remainder_(53)
            .sub_(26)
            .div_(1792)
            .unsqueeze(1)
        )
    input = (
        torch.arange(rows * in_features, dtype=torch.float32, device=device)
        .remainder_(127)
        .sub_(63)
        .div_(64)
        .reshape(rows, in_features)
        .to(torch.bfloat16)
    )
    input.requires_grad_(True)

    # The first module invocation is the cold component cell; the second is
    # the warm cell. Build the independent raw-S4 oracle only afterward.
    cold_actual = module(input)
    warm_actual = module(input)
    effective_gate_A = module.gate_proj.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_gate_B = module.gate_proj.lora_B.detach().to(torch.bfloat16).contiguous()
    effective_up_A = module.up_proj.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_up_B = module.up_proj.lora_B.detach().to(torch.bfloat16).contiguous()
    stacked_A = torch.cat((effective_gate_A, effective_up_A), dim=0).unsqueeze(0).contiguous()
    stacked_B = torch.cat((effective_gate_B, effective_up_B), dim=0).unsqueeze(0).contiguous()
    eager_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )
    direct_base = triton_w8a8_block_fp8_linear(
        input.detach(),
        module.fp8_weight().contiguous(),
        [128, 128],
        module.weight_scale_inv.contiguous(),
    )
    direct_A = sgemm_lora_a_fwd(input.detach(), stacked_A, eager_info, stack_num=2)
    expected = gate_up_lora_b_fwd(
        direct_A,
        stacked_B,
        eager_info,
        intermediate_size,
        base_output=direct_base.clone(),
    )
    assert torch.equal(cold_actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(warm_actual.view(torch.uint8), cold_actual.view(torch.uint8))
    # Serving's exact mode computes the one-round FP32 SwiGLU
    # (SiluAndMul.forward_exact, xorl-sglang f10b907d8); the trainer op must
    # match a one-round sampler oracle bitwise.
    trainer_activation = exact_fp32_silu_and_mul(cold_actual)
    sampler_activation = fp32_silu_and_mul(expected)
    assert torch.equal(trainer_activation.view(torch.uint8), sampler_activation.view(torch.uint8))

    # Build the exact adapter-merged metadata used after S4's production
    # decode routing. Sixteen live request rows all select adapter slot zero;
    # the backend merges them into the first of eight adapter segments while
    # retaining the graph's fixed segment arrays.
    graph_slots = 16
    max_loras_per_batch = 8
    graph_backend = TritonLoRABackend(max_loras_per_batch=max_loras_per_batch, device=device)
    graph_backend.init_cuda_graph_batch_info(max_bs_in_cuda_graph=graph_slots, num_tokens_per_req=1)
    graph_backend.batch_info = graph_backend.cuda_graph_batch_info
    graph_backend.batch_info.weight_indices[:graph_slots].zero_()
    graph_backend.batch_info.lora_ranks.zero_()
    graph_backend.batch_info.lora_ranks[0] = 1
    graph_backend.batch_info.scalings.zero_()
    graph_backend.batch_info.scalings[0] = 1.0
    graph_backend.compute_sgemm_routing(use_cuda_graph=True)
    graph_info = graph_backend.sgemm_batch_info
    assert graph_info is graph_backend.cuda_graph_sgemm_batch_info
    assert graph_info.bs == max_loras_per_batch
    assert torch.equal(
        graph_info.seg_lens,
        torch.tensor([graph_slots] + [0] * (max_loras_per_batch - 1), dtype=torch.int32, device=device),
    )
    assert torch.equal(
        graph_info.seg_indptr,
        torch.tensor([0] + [graph_slots] * max_loras_per_batch, dtype=torch.int32, device=device),
    )
    assert torch.equal(graph_info.weight_indices, torch.arange(max_loras_per_batch, dtype=torch.int32, device=device))
    assert torch.equal(graph_info.permutation, torch.arange(graph_slots, dtype=torch.int32, device=device))
    assert graph_info.max_len == graph_slots

    graph_input = input.detach()[:graph_slots].contiguous()
    graph_base = triton_w8a8_block_fp8_linear(
        graph_input,
        module.fp8_weight().contiguous(),
        [128, 128],
        module.weight_scale_inv.contiguous(),
    )
    graph_stacked_A = torch.zeros((max_loras_per_batch, *stacked_A.shape[1:]), dtype=stacked_A.dtype, device=device)
    graph_stacked_B = torch.zeros((max_loras_per_batch, *stacked_B.shape[1:]), dtype=stacked_B.dtype, device=device)
    graph_stacked_A[0].copy_(stacked_A[0])
    graph_stacked_B[0].copy_(stacked_B[0])
    graph_A = sgemm_lora_a_fwd(graph_input, graph_stacked_A, graph_info, stack_num=2)
    graph_output = graph_backend.run_gate_up_lora(
        graph_input,
        graph_stacked_A,
        graph_stacked_B,
        base_output=graph_base.clone(),
    )

    graph_eager_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, graph_slots], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float32, device=device),
        max_len=graph_slots,
        seg_lens=torch.tensor([graph_slots], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=graph_slots,
        has_active_lora=True,
    )
    graph_eager_A = sgemm_lora_a_fwd(graph_input, stacked_A, graph_eager_info, stack_num=2)
    graph_eager_output = gate_up_lora_b_fwd(
        graph_eager_A,
        stacked_B,
        graph_eager_info,
        intermediate_size,
        base_output=graph_base.clone(),
    )
    assert torch.equal(graph_A.view(torch.uint8), graph_eager_A.view(torch.uint8))
    assert torch.equal(graph_output.view(torch.uint8), graph_eager_output.view(torch.uint8))

    grad_output = (
        torch.arange(rows * 2 * intermediate_size, dtype=torch.float32, device=device)
        .remainder_(61)
        .sub_(30)
        .div_(31)
        .reshape(rows, 2 * intermediate_size)
        .to(torch.bfloat16)
    )
    base_weight = module._dequantize_base_weight().to(torch.bfloat16)
    gate_base_input = input.detach().clone().requires_grad_(True)
    up_base_input = input.detach().clone().requires_grad_(True)
    gate_base_output = F.linear(gate_base_input, base_weight[:intermediate_size])
    up_base_output = F.linear(up_base_input, base_weight[intermediate_size:])
    gate_grad_output, up_grad_output = grad_output.split(intermediate_size, dim=-1)
    torch.autograd.backward((gate_base_output, up_base_output), (gate_grad_output, up_grad_output))
    gate_lora_input = input.detach().float().requires_grad_(True)
    up_lora_input = input.detach().float().requires_grad_(True)
    reference_factors = tuple(
        factor.float().requires_grad_(True)
        for factor in (effective_gate_A, effective_gate_B, effective_up_A, effective_up_B)
    )
    gate_output = F.linear(F.linear(gate_lora_input, reference_factors[0]), reference_factors[1])
    up_output = F.linear(F.linear(up_lora_input, reference_factors[2]), reference_factors[3])
    torch.cat((gate_output, up_output), dim=-1).backward(grad_output.float())
    expected_gate_dx = gate_base_input.grad.float() + gate_lora_input.grad
    expected_up_dx = up_base_input.grad.float() + up_lora_input.grad
    expected_dx = expected_gate_dx.to(torch.bfloat16) + expected_up_dx.to(torch.bfloat16)

    cold_actual.backward(grad_output)

    assert torch.equal(input.grad, expected_dx)
    for master, reference_factor in zip(
        (
            module.gate_proj.lora_A,
            module.gate_proj.lora_B,
            module.up_proj.lora_A,
            module.up_proj.lora_B,
        ),
        reference_factors,
        strict=True,
    ):
        assert torch.equal(master.grad, reference_factor.grad)

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name in module.logical_factor_names:
            dict(module.named_parameters())[name].zero_()
    zero_output = module(input.detach())
    assert torch.equal(zero_output.view(torch.uint8), direct_base.view(torch.uint8))
