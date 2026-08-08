from __future__ import annotations

import pytest
import torch

from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear


def _fill_factor(parameter: torch.Tensor, *, modulus: int, center: int, divisor: int) -> None:
    with torch.no_grad():
        parameter.copy_(
            torch.arange(parameter.numel(), dtype=torch.float32, device=parameter.device)
            .reshape_as(parameter)
            .remainder_(modulus)
            .sub_(center)
            .div_(divisor)
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_official_attention_qkv_a_split_wrappers_match_sglang_fused_physical_program() -> None:
    """Prove the TP1 split trainer values equal S4's fused q-A/kv-A program."""

    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 attention component requires Hopper")
    from sglang.kernels.ops.gemm.qkv_lora_b import qkv_lora_b_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
    from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    from sglang.srt.lora.utils import LoRABatchInfo

    device = torch.device("cuda")
    rows = 17
    hidden_size = 6144
    q_lora_rank = 2048
    kv_output_size = 512 + 64
    total_output_size = q_lora_rank + kv_output_size

    q_weight = torch.full(
        (q_lora_rank, hidden_size),
        0.25,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kv_weight = torch.full(
        (kv_output_size, hidden_size),
        -0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    q_scales = torch.full(
        (q_lora_rank // 128, hidden_size // 128),
        0.03125,
        dtype=torch.float32,
        device=device,
    )
    kv_scales = torch.full(
        ((kv_output_size + 127) // 128, hidden_size // 128),
        0.0625,
        dtype=torch.float32,
        device=device,
    )

    q_projection = Glm52ExactTP1BlockFP8QLoRALinear(hidden_size, q_lora_rank, device=device)
    q_projection._source_fqn = "q_a_proj"
    q_projection._load_prequantized(
        lambda name: q_weight if name == "q_a_proj.weight" else q_scales,
    )
    kv_projection = Glm52ExactTP1BlockFP8QLoRALinear(hidden_size, kv_output_size, device=device)
    kv_projection._source_fqn = "kv_a_proj_with_mqa"
    kv_projection._load_prequantized(
        lambda name: kv_weight if name == "kv_a_proj_with_mqa.weight" else kv_scales,
    )
    _fill_factor(q_projection.lora_A, modulus=37, center=18, divisor=1024)
    _fill_factor(q_projection.lora_B, modulus=47, center=23, divisor=2048)
    _fill_factor(kv_projection.lora_A, modulus=43, center=21, divisor=1536)
    _fill_factor(kv_projection.lora_B, modulus=53, center=26, divisor=1792)

    input = (
        torch.arange(rows * hidden_size, dtype=torch.float32, device=device)
        .reshape(rows, hidden_size)
        .remainder_(127)
        .sub_(63)
        .div_(64)
        .to(torch.bfloat16)
    )
    input.requires_grad_(True)
    trainer_output = torch.cat((q_projection(input), kv_projection(input)), dim=-1)

    q_A = q_projection.lora_A.detach().to(torch.bfloat16).contiguous()
    q_B = q_projection.lora_B.detach().to(torch.bfloat16).contiguous()
    kv_A = kv_projection.lora_A.detach().to(torch.bfloat16).contiguous()
    kv_B = kv_projection.lora_B.detach().to(torch.bfloat16).contiguous()
    fused_A = torch.cat((q_A, kv_A), dim=0).unsqueeze(0).contiguous()
    fused_B = torch.cat((q_B, kv_B), dim=0).unsqueeze(0).contiguous()
    fused_weight = torch.cat((q_weight, kv_weight), dim=0).contiguous()
    fused_scales = torch.cat((q_scales, kv_scales), dim=0).contiguous()
    output_offset = torch.tensor([0, q_lora_rank, total_output_size], dtype=torch.int32, device=device)
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
    fused_base = triton_w8a8_block_fp8_linear(
        input.detach(),
        fused_weight,
        [128, 128],
        fused_scales,
    )
    q_base = triton_w8a8_block_fp8_linear(input.detach(), q_weight, [128, 128], q_scales)
    kv_base = triton_w8a8_block_fp8_linear(input.detach(), kv_weight, [128, 128], kv_scales)
    assert torch.equal(fused_base[:, :q_lora_rank].view(torch.uint8), q_base.view(torch.uint8))
    assert torch.equal(fused_base[:, q_lora_rank:].view(torch.uint8), kv_base.view(torch.uint8))

    fused_a_output = sgemm_lora_a_fwd(input.detach(), fused_A, eager_info, stack_num=2)
    q_a_output = sgemm_lora_a_fwd(input.detach(), q_A.unsqueeze(0), eager_info)
    kv_a_output = sgemm_lora_a_fwd(input.detach(), kv_A.unsqueeze(0), eager_info)
    assert torch.equal(fused_a_output[:, :1].view(torch.uint8), q_a_output.view(torch.uint8))
    assert torch.equal(fused_a_output[:, 1:].view(torch.uint8), kv_a_output.view(torch.uint8))

    fused_delta = qkv_lora_b_fwd(
        fused_a_output,
        fused_B,
        eager_info,
        output_offset,
        q_lora_rank,
        n_slices=2,
    )
    q_delta = sgemm_lora_b_fwd(q_a_output, q_B.unsqueeze(0), eager_info)
    kv_delta = sgemm_lora_b_fwd(kv_a_output, kv_B.unsqueeze(0), eager_info)
    assert torch.equal(fused_delta[:, :q_lora_rank].view(torch.uint8), q_delta.view(torch.uint8))
    assert torch.equal(fused_delta[:, q_lora_rank:].view(torch.uint8), kv_delta.view(torch.uint8))

    fused_output = qkv_lora_b_fwd(
        fused_a_output,
        fused_B,
        eager_info,
        output_offset,
        q_lora_rank,
        base_output=fused_base.clone(),
        n_slices=2,
    )
    assert torch.equal(trainer_output.view(torch.uint8), fused_output.view(torch.uint8))

    # Exercise the admitted fixed decode-graph routing, not only a hand-built
    # one-segment metadata object. The exact contract fixes max_lora_rank=1; using a wider
    # pool here would exercise a different, unadmitted physical layout.
    graph_slots = 16
    max_loras_per_batch = 8
    backend = TritonLoRABackend(max_loras_per_batch=max_loras_per_batch, device=device)
    backend.init_cuda_graph_batch_info(max_bs_in_cuda_graph=graph_slots, num_tokens_per_req=1)
    backend.batch_info = backend.cuda_graph_batch_info
    backend.batch_info.weight_indices[:graph_slots].zero_()
    backend.batch_info.lora_ranks.zero_()
    backend.batch_info.lora_ranks[0] = 1
    backend.batch_info.scalings.zero_()
    backend.batch_info.scalings[0] = 1.0
    backend.compute_sgemm_routing(use_cuda_graph=True)
    graph_info = backend.sgemm_batch_info
    assert graph_info is backend.cuda_graph_sgemm_batch_info
    assert torch.equal(
        graph_info.seg_lens,
        torch.tensor([graph_slots] + [0] * 7, dtype=torch.int32, device=device),
    )

    graph_input = input.detach()[:graph_slots].contiguous()
    max_lora_rank = 1
    graph_A = torch.zeros(
        (max_loras_per_batch, 2 * max_lora_rank, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    graph_B = torch.zeros(
        (max_loras_per_batch, total_output_size, max_lora_rank),
        dtype=torch.bfloat16,
        device=device,
    )
    graph_A[0, 0].copy_(q_A[0])
    graph_A[0, max_lora_rank].copy_(kv_A[0])
    graph_B[0, :q_lora_rank, 0].copy_(q_B[:, 0])
    graph_B[0, q_lora_rank:, 0].copy_(kv_B[:, 0])
    assert torch.count_nonzero(graph_A[1:]) == 0
    assert torch.count_nonzero(graph_B[1:]) == 0
    graph_base = triton_w8a8_block_fp8_linear(
        graph_input,
        fused_weight,
        [128, 128],
        fused_scales,
    )
    graph_output = backend.run_qkv_lora(
        graph_input,
        graph_A,
        graph_B,
        output_offset,
        q_lora_rank,
        base_output=graph_base.clone(),
        n_slices=2,
    )
    assert torch.equal(graph_output.view(torch.uint8), fused_output[:graph_slots].view(torch.uint8))
    split_graph_output = torch.cat((q_projection(graph_input), kv_projection(graph_input)), dim=-1)
    assert torch.equal(split_graph_output.view(torch.uint8), graph_output.view(torch.uint8))

    grad_output = (
        torch.arange(rows * total_output_size, dtype=torch.float32, device=device)
        .reshape(rows, total_output_size)
        .remainder_(61)
        .sub_(30)
        .div_(31)
        .to(torch.bfloat16)
    )
    q_grad_output, kv_grad_output = grad_output.split((q_lora_rank, kv_output_size), dim=-1)
    expected_q_vjp = q_projection._surrogate_vjp(
        input.detach(),
        q_A,
        q_B,
        q_grad_output,
        needs_input_grad=(True, True, True),
    )
    expected_kv_vjp = kv_projection._surrogate_vjp(
        input.detach(),
        kv_A,
        kv_B,
        kv_grad_output,
        needs_input_grad=(True, True, True),
    )
    trainer_output.backward(grad_output)
    assert input.grad is not None and input.grad.dtype is torch.bfloat16
    expected_input_grad = expected_q_vjp[0].to(input.dtype) + expected_kv_vjp[0].to(input.dtype)
    assert torch.equal(input.grad, expected_input_grad)
    assert q_projection.lora_A.grad is not None and q_projection.lora_A.grad.dtype is torch.float32
    assert q_projection.lora_B.grad is not None and q_projection.lora_B.grad.dtype is torch.float32
    assert kv_projection.lora_A.grad is not None and kv_projection.lora_A.grad.dtype is torch.float32
    assert kv_projection.lora_B.grad is not None and kv_projection.lora_B.grad.dtype is torch.float32
    assert torch.equal(q_projection.lora_A.grad, expected_q_vjp[1])
    assert torch.equal(q_projection.lora_B.grad, expected_q_vjp[2])
    assert torch.equal(kv_projection.lora_A.grad, expected_kv_vjp[1])
    assert torch.equal(kv_projection.lora_B.grad, expected_kv_vjp[2])
