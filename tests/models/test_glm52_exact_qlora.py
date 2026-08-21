from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.glm5.exact_qlora import (
    GLM52_EXACT_TP1_QLORA_CONTRACT_VERSION,
    Glm52ExactTP1BlockFP8QLoRALinear,
)
from xorl.ops.exact.block_fp8_native import NativeBlockFP8Linear


def _module() -> Glm52ExactTP1BlockFP8QLoRALinear:
    module = Glm52ExactTP1BlockFP8QLoRALinear(8, 6, device=torch.device("cpu"))
    with torch.no_grad():
        module.lora_A.copy_(torch.tensor([[0.1001, -0.2002, 0.3003, -0.4004, 0.5005, -0.6006, 0.7007, -0.8008]]))
        module.lora_B.copy_(torch.tensor([[0.1101], [-0.2202], [0.3303], [-0.4404], [0.5505], [-0.6606]]))
    return module


def _literal_cpu_value(base_weight, captures):
    def run(input, effective_A, effective_B):
        captures.append((effective_A.detach().clone(), effective_B.detach().clone()))
        base = F.linear(input.float(), base_weight.float()).to(torch.bfloat16)
        a_output = F.linear(input.float(), effective_A.float()).to(torch.bfloat16)
        b_output = F.linear(a_output.float(), effective_B.float()).to(torch.bfloat16)
        return (base + b_output).to(torch.bfloat16)

    return run


def test_exact_tp1_wrapper_accepts_positive_rank_and_alpha_without_bias_or_aqn() -> None:
    module = _module()

    assert module.contract_version == GLM52_EXACT_TP1_QLORA_CONTRACT_VERSION
    assert module.r == module.active_r == 1
    assert module.lora_alpha == module.active_lora_alpha == 1
    assert module.scaling == 1.0
    assert module.fsdp_requires_full_precision is True
    assert module.lora_A.dtype is torch.float32
    assert module.lora_B.dtype is torch.float32

    rank_three = Glm52ExactTP1BlockFP8QLoRALinear(8, 6, r=3, lora_alpha=7)
    assert rank_three.lora_A.shape == (3, 8)
    assert rank_three.lora_B.shape == (6, 3)
    assert rank_three.scaling == 7 / 3
    with pytest.raises(ValueError, match="bias-free"):
        Glm52ExactTP1BlockFP8QLoRALinear(8, 6, bias=True)
    with pytest.raises(ValueError, match="rejects adaptive quantization noise"):
        Glm52ExactTP1BlockFP8QLoRALinear(8, 6, enable_aqn=True)
    module.set_runtime_lora_config(1, 1)
    with pytest.raises(ValueError, match="positive integer rank"):
        Glm52ExactTP1BlockFP8QLoRALinear(8, 6, r=0)
    with pytest.raises(ValueError, match="positive integer alpha"):
        module.set_runtime_lora_config(1, 0)


def test_exact_tp1_model_dtype_move_preserves_packed_state_master_dtype_and_identity() -> None:
    module = _module()
    with torch.no_grad():
        module.packed_weight_f32.copy_(
            torch.arange(module.packed_weight_f32.numel()).reshape_as(module.packed_weight_f32)
        )
        module.weight_block_scales.copy_(
            torch.arange(module.weight_block_scales.numel(), dtype=torch.uint8).reshape_as(module.weight_block_scales)
        )
    parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if name in {"lora_A", "lora_B", "packed_weight_f32"}
    }
    values = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    scale_bytes = module.weight_block_scales.clone()

    module.to(dtype=torch.bfloat16)

    for name, original in parameters.items():
        actual = dict(module.named_parameters())[name]
        assert actual is original
        assert actual.dtype is torch.float32
        assert torch.equal(actual, values[name])
    assert module.weight_block_scales.dtype is torch.uint8
    assert torch.equal(module.weight_block_scales, scale_bytes)


def test_exact_tp1_wrapper_rounds_master_factors_once_before_value_forward(monkeypatch) -> None:
    module = _module()
    base_weight = torch.arange(48, dtype=torch.float32).reshape(6, 8).div_(97).to(torch.bfloat16)
    captures = []
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, captures))

    input = torch.arange(24, dtype=torch.float32).reshape(3, 8).div_(53).to(torch.bfloat16)
    output = module(input)

    effective_A, effective_B = captures.pop()
    assert torch.equal(effective_A, module.lora_A.detach().to(torch.bfloat16))
    assert torch.equal(effective_B, module.lora_B.detach().to(torch.bfloat16))
    expected = _literal_cpu_value(base_weight, [])(input, effective_A, effective_B)
    assert torch.equal(output, expected)


def test_exact_tp1_surrogate_backward_matches_effective_factor_qlora_reference(monkeypatch) -> None:
    module = _module()
    base_weight = torch.arange(48, dtype=torch.float32).reshape(6, 8).sub_(17).div_(41).to(torch.bfloat16)
    monkeypatch.setattr(module, "_dequantize_weight", lambda: base_weight.float())
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, []))

    input = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8).sub_(9).div_(29).to(torch.bfloat16)
    input.requires_grad_(True)
    grad_output = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6).sub_(7).div_(31).to(torch.bfloat16)

    effective_A = module.lora_A.detach().to(torch.bfloat16)
    effective_B = module.lora_B.detach().to(torch.bfloat16)
    logical_dx, logical_dA, logical_dB = module._surrogate_vjp(
        input.detach(),
        effective_A,
        effective_B,
        grad_output,
        needs_input_grad=(True, True, True),
    )

    base_input = input.detach().clone().requires_grad_(True)
    F.linear(base_input, base_weight).backward(grad_output)
    lora_input = input.detach().float().requires_grad_(True)
    reference_A = module.lora_A.detach().to(torch.bfloat16).float().requires_grad_(True)
    reference_B = module.lora_B.detach().to(torch.bfloat16).float().requires_grad_(True)
    reference_lora = F.linear(F.linear(lora_input, reference_A), reference_B)
    reference_lora.backward(grad_output.float())

    expected_dx = base_input.grad.float() + lora_input.grad
    assert logical_dx.dtype is torch.float32
    assert torch.equal(logical_dx, expected_dx)
    assert torch.equal(logical_dA, reference_A.grad)
    assert torch.equal(logical_dB, reference_B.grad)

    output = module(input)
    output.backward(grad_output)

    assert torch.equal(input.grad, expected_dx.to(torch.bfloat16))
    assert torch.equal(module.lora_A.grad, reference_A.grad)
    assert torch.equal(module.lora_B.grad, reference_B.grad)


def test_exact_tp1_factor_only_backward_does_not_materialize_base(monkeypatch) -> None:
    module = _module()
    base_weight = torch.zeros(6, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, []))
    monkeypatch.setattr(
        module,
        "_dequantize_weight",
        lambda: pytest.fail("factor-only VJP must not dequantize the frozen base"),
    )

    module(torch.ones(2, 8, dtype=torch.bfloat16)).float().sum().backward()

    assert module.lora_A.grad is not None
    assert module.lora_B.grad is not None


def test_exact_tp1_backward_rejects_master_mutation(monkeypatch) -> None:
    module = _module()
    base_weight = torch.zeros(6, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_exact_forward_value", _literal_cpu_value(base_weight, []))
    monkeypatch.setattr(module, "_dequantize_weight", lambda: base_weight.float())
    output = module(torch.ones(2, 8, dtype=torch.bfloat16))

    with torch.no_grad():
        module.lora_A.add_(1)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.float().sum().backward()


def test_exact_tp1_contract_fails_before_any_sglang_import() -> None:
    before = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    module = _module()

    with pytest.raises(TypeError, match="requires BF16 activations"):
        module(torch.zeros(1, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match="contiguous sampler-layout"):
        module(torch.zeros(8, 2, dtype=torch.bfloat16).transpose(0, 1))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module(torch.zeros(1, 8, dtype=torch.bfloat16))

    after = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    assert after == before


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    ((6144, 576), (12288, 6144)),
    ids=("kv-a-partial-edge", "dense-down"),
)
def test_exact_tp1_literal_cuda_bytes_and_surrogate_gradients_match_direct_program(
    in_features: int,
    out_features: int,
) -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    from sglang.srt.lora.utils import LoRABatchInfo

    device = torch.device("cuda")
    # The two physical rows cover the official partial-edge kv_a projection
    # and the first-three-block dense down projection.
    rows = 17
    weight_values = torch.arange(out_features * in_features, device=device, dtype=torch.int32)
    weight = ((weight_values % 31) - 15).reshape(out_features, in_features).float().to(torch.float8_e4m3fn)
    scale_shape = ((out_features + 127) // 128, (in_features + 127) // 128)
    scales = (
        torch.arange(scale_shape[0] * scale_shape[1], device=device, dtype=torch.float32)
        .reshape(scale_shape)
        .remainder_(31)
        .add_(1)
        .div_(32)
        .contiguous()
    )

    native = NativeBlockFP8Linear(in_features, out_features, device=device)
    native.load_prequantized(weight, scales)
    exact = Glm52ExactTP1BlockFP8QLoRALinear(in_features, out_features, device=device)
    exact._source_fqn = "projection"
    exact._load_prequantized(
        lambda name: weight if name == "projection.weight" else scales,
    )
    with torch.no_grad():
        exact.lora_A.copy_(
            torch.arange(in_features, device=device, dtype=torch.float32).sub_(91).div_(257).unsqueeze(0)
        )
        exact.lora_B.copy_(
            torch.arange(out_features, device=device, dtype=torch.float32).sub_(211).div_(577).unsqueeze(1)
        )

    input = (
        torch.arange(rows * in_features, device=device, dtype=torch.float32)
        .remainder_(127)
        .sub_(63)
        .div_(64)
        .reshape(rows, in_features)
        .to(torch.bfloat16)
    )
    input.requires_grad_(True)
    effective_A = exact.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_B = exact.lora_B.detach().to(torch.bfloat16).contiguous()
    batch_info = LoRABatchInfo(
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

    direct_base = native(input.detach())
    direct_a = sgemm_lora_a_fwd(input.detach(), effective_A.unsqueeze(0), batch_info)
    expected = sgemm_lora_b_fwd(
        direct_a,
        effective_B.unsqueeze(0),
        batch_info,
        base_output=direct_base.clone(),
    )
    actual = exact(input)
    warm_actual = exact(input)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(warm_actual.view(torch.uint8), actual.view(torch.uint8))

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
    graph_batch_info = graph_backend.sgemm_batch_info
    assert graph_batch_info is graph_backend.cuda_graph_sgemm_batch_info
    assert graph_batch_info.bs == max_loras_per_batch
    assert torch.equal(
        graph_batch_info.seg_lens,
        torch.tensor([graph_slots] + [0] * (max_loras_per_batch - 1), dtype=torch.int32, device=device),
    )
    assert torch.equal(
        graph_batch_info.seg_indptr,
        torch.tensor([0] + [graph_slots] * max_loras_per_batch, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        graph_batch_info.weight_indices,
        torch.arange(max_loras_per_batch, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        graph_batch_info.permutation,
        torch.arange(graph_slots, dtype=torch.int32, device=device),
    )
    assert graph_batch_info.max_len == graph_slots

    graph_input = input.detach()[:graph_slots].contiguous()
    graph_base = native(graph_input)
    graph_A_weights = torch.zeros((max_loras_per_batch, 1, in_features), dtype=effective_A.dtype, device=device)
    graph_B_weights = torch.zeros((max_loras_per_batch, out_features, 1), dtype=effective_B.dtype, device=device)
    graph_A_weights[0].copy_(effective_A)
    graph_B_weights[0].copy_(effective_B)
    graph_a = graph_backend.run_lora_a_sgemm(graph_input, graph_A_weights)
    graph_output = graph_backend.run_lora_b_sgemm(
        graph_a,
        graph_B_weights,
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
    graph_eager_a = sgemm_lora_a_fwd(graph_input, effective_A.unsqueeze(0), graph_eager_info)
    graph_eager_output = sgemm_lora_b_fwd(
        graph_eager_a,
        effective_B.unsqueeze(0),
        graph_eager_info,
        base_output=graph_base.clone(),
    )
    assert torch.equal(graph_a.view(torch.uint8), graph_eager_a.view(torch.uint8))
    assert torch.equal(graph_output.view(torch.uint8), graph_eager_output.view(torch.uint8))

    grad_output = (
        torch.arange(rows * out_features, device=device, dtype=torch.float32)
        .remainder_(61)
        .sub_(30)
        .div_(31)
        .reshape(rows, out_features)
        .to(torch.bfloat16)
    )
    base_input = input.detach().clone().requires_grad_(True)
    base_weight = exact._dequantize_weight().to(torch.bfloat16)
    F.linear(base_input, base_weight).backward(grad_output)
    lora_input = input.detach().float().requires_grad_(True)
    reference_A = effective_A.float().requires_grad_(True)
    reference_B = effective_B.float().requires_grad_(True)
    F.linear(F.linear(lora_input, reference_A), reference_B).backward(grad_output.float())
    expected_dx = base_input.grad.float() + lora_input.grad

    actual.backward(grad_output)

    assert torch.equal(input.grad, expected_dx.to(torch.bfloat16))
    assert torch.equal(exact.lora_A.grad, reference_A.grad)
    assert torch.equal(exact.lora_B.grad, reference_B.grad)

    with torch.no_grad():
        exact.lora_A.zero_()
        exact.lora_B.zero_()
    zero_output = exact(input.detach())
    assert torch.equal(zero_output.view(torch.uint8), direct_base.view(torch.uint8))
