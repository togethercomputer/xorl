from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.glm5.exact_gate_up_qlora import (
    Glm52ExactTP1FusedGateUpBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.ops.exact.fused_silu_and_mul import one_round_swiglu


def _pattern(
    shape: tuple[int, ...],
    *,
    modulus: int,
    center: int,
    divisor: int,
    device: torch.device,
) -> torch.Tensor:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return (
        torch.arange(numel, dtype=torch.float32, device=device)
        .remainder_(modulus)
        .sub_(center)
        .div_(divisor)
        .reshape(shape)
    )


def _logical_gate_up_surrogate_reference(
    module: Glm52ExactTP1FusedGateUpBlockFP8QLoRA,
    input: torch.Tensor,
    gate_A: torch.Tensor,
    gate_B: torch.Tensor,
    up_A: torch.Tensor,
    up_B: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the two pre-fusion QLoRA autograd boundaries independently."""

    gate_grad_output, up_grad_output = grad_output.split(module.intermediate_size, dim=-1)
    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        base_weight = module._dequantize_base_weight().to(torch.bfloat16)
        gate_weight, up_weight = base_weight.split(module.intermediate_size, dim=0)

        def branch_vjp(weight, factor_A, factor_B, branch_grad):
            base_input = input.detach().requires_grad_(True)
            base_output = F.linear(base_input, weight)
            (base_input_grad,) = torch.autograd.grad(
                base_output,
                base_input,
                grad_outputs=branch_grad.to(base_output.dtype),
            )
            lora_input = input.float().detach().requires_grad_(True)
            reference_A = factor_A.float().detach().requires_grad_(True)
            reference_B = factor_B.float().detach().requires_grad_(True)
            lora_output = F.linear(F.linear(lora_input, reference_A), reference_B)
            lora_input_grad, factor_A_grad, factor_B_grad = torch.autograd.grad(
                lora_output,
                (lora_input, reference_A, reference_B),
                grad_outputs=branch_grad.float(),
            )
            # Each old logical projection returned FP32 dX to the same BF16
            # activation. Autograd cast each contribution before accumulating
            # them in the shared BF16 input buffer.
            branch_input_grad = (base_input_grad.float() + lora_input_grad.float()).to(torch.bfloat16)
            return branch_input_grad, factor_A_grad, factor_B_grad

        gate_input_grad, gate_A_grad, gate_B_grad = branch_vjp(
            gate_weight,
            gate_A,
            gate_B,
            gate_grad_output,
        )
        up_input_grad, up_A_grad, up_B_grad = branch_vjp(
            up_weight,
            up_A,
            up_B,
            up_grad_output,
        )
    return gate_input_grad + up_input_grad, gate_A_grad, gate_B_grad, up_A_grad, up_B_grad


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_official_tp1_dense_vertical_composition_bytes_and_manual_vjp() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability(0)[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 dense vertical requires Hopper")
    from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
    from sglang.srt.batch_invariant_ops.bi_silu_and_mul import fp32_silu_and_mul
    from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
    from sglang.srt.lora.utils import LoRABatchInfo

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    rows, hidden_size, intermediate_size = 17, 6144, 12288

    gate_up = Glm52ExactTP1FusedGateUpBlockFP8QLoRA(hidden_size, intermediate_size, device=device)
    gate_weight = torch.full(
        (intermediate_size, hidden_size),
        0.25,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    up_weight = torch.full(
        (intermediate_size, hidden_size),
        -0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    gate_scale = torch.full(
        (intermediate_size // 128, hidden_size // 128),
        0.03125,
        dtype=torch.float32,
        device=device,
    )
    up_scale = torch.full_like(gate_scale, 0.0625)
    gate_up.load_gate_up_prequantized(gate_weight, gate_scale, up_weight, up_scale)

    down = Glm52ExactTP1BlockFP8QLoRALinear(intermediate_size, hidden_size, device=device)
    down_weight = torch.full(
        (hidden_size, intermediate_size),
        0.125,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    down_scale = torch.full(
        (hidden_size // 128, intermediate_size // 128),
        0.046875,
        dtype=torch.float32,
        device=device,
    )
    down._source_fqn = "dense.down_proj"
    down_state = {
        "dense.down_proj.weight": down_weight,
        "dense.down_proj.weight_scale_inv": down_scale,
    }
    down._load_prequantized(down_state.__getitem__)

    with torch.no_grad():
        gate_up.gate_proj.lora_A.copy_(_pattern((1, hidden_size), modulus=37, center=18, divisor=1024, device=device))
        gate_up.gate_proj.lora_B.copy_(
            _pattern((intermediate_size, 1), modulus=47, center=23, divisor=2048, device=device)
        )
        gate_up.up_proj.lora_A.copy_(_pattern((1, hidden_size), modulus=43, center=21, divisor=1536, device=device))
        gate_up.up_proj.lora_B.copy_(
            _pattern((intermediate_size, 1), modulus=53, center=26, divisor=1792, device=device)
        )
        down.lora_A.copy_(_pattern((1, intermediate_size), modulus=59, center=29, divisor=2304, device=device))
        down.lora_B.copy_(_pattern((hidden_size, 1), modulus=61, center=30, divisor=2560, device=device))

    input = _pattern(
        (rows, hidden_size),
        modulus=127,
        center=63,
        divisor=64,
        device=device,
    ).to(torch.bfloat16)
    input.requires_grad_(True)

    trainer_gate_up = gate_up(input)
    trainer_gate_up.retain_grad()
    trainer_activation = one_round_swiglu(trainer_gate_up)
    trainer_activation.retain_grad()
    trainer_output = down(trainer_activation)

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
    effective_gate_A = gate_up.gate_proj.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_gate_B = gate_up.gate_proj.lora_B.detach().to(torch.bfloat16).contiguous()
    effective_up_A = gate_up.up_proj.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_up_B = gate_up.up_proj.lora_B.detach().to(torch.bfloat16).contiguous()
    effective_down_A = down.lora_A.detach().to(torch.bfloat16).contiguous()
    effective_down_B = down.lora_B.detach().to(torch.bfloat16).contiguous()

    # Use only the loaded module state from here onward. This bounds the test's
    # live FP8 payload while the two surrogate VJPs materialize BF16 bases.
    del gate_weight, up_weight, gate_scale, up_scale, down_weight, down_scale, down_state
    raw_gate_up_weight = gate_up.fp8_weight().contiguous()
    raw_down_weight = (
        down._read_packed_weight_uint8().view(torch.float8_e4m3fn).reshape(hidden_size, intermediate_size).contiguous()
    )
    raw_down_scale = down._recover_tensor(
        down.weight_block_scales,
        down._scale_dtypes["weight_block_scales"],
    ).contiguous()

    with torch.no_grad():
        raw_gate_up_base = triton_w8a8_block_fp8_linear(
            input.detach(),
            raw_gate_up_weight,
            [128, 128],
            gate_up.weight_scale_inv.contiguous(),
        )
        raw_gate_up_A = sgemm_lora_a_fwd(
            input.detach(),
            torch.cat((effective_gate_A, effective_up_A), dim=0).unsqueeze(0).contiguous(),
            batch_info,
            stack_num=2,
        )
        raw_gate_up = gate_up_lora_b_fwd(
            raw_gate_up_A,
            torch.cat((effective_gate_B, effective_up_B), dim=0).unsqueeze(0).contiguous(),
            batch_info,
            intermediate_size,
            base_output=raw_gate_up_base.clone(),
        )
        # S4's exact mode resolves SiluAndMul.forward_exact to the one-round
        # FP32 SwiGLU (fp32_silu_and_mul, xorl-sglang f10b907d8); the raw
        # oracle uses serving's own op so the pair cannot self-confirm.
        raw_activation = fp32_silu_and_mul(raw_gate_up)
        raw_down_base = triton_w8a8_block_fp8_linear(
            raw_activation,
            raw_down_weight,
            [128, 128],
            raw_down_scale,
        )
        raw_down_A = sgemm_lora_a_fwd(
            raw_activation,
            effective_down_A.unsqueeze(0),
            batch_info,
        )
        raw_output = sgemm_lora_b_fwd(
            raw_down_A,
            effective_down_B.unsqueeze(0),
            batch_info,
            base_output=raw_down_base.clone(),
        )

    assert torch.equal(trainer_gate_up.view(torch.uint8), raw_gate_up.view(torch.uint8))
    assert torch.equal(trainer_activation.view(torch.uint8), raw_activation.view(torch.uint8))
    assert torch.equal(trainer_output.view(torch.uint8), raw_output.view(torch.uint8))

    final_grad = _pattern(
        (rows, hidden_size),
        modulus=67,
        center=33,
        divisor=71,
        device=device,
    ).to(torch.bfloat16)
    manual_activation_grad, manual_down_A_grad, manual_down_B_grad = down._surrogate_vjp(
        trainer_activation.detach(),
        effective_down_A,
        effective_down_B,
        final_grad,
        needs_input_grad=(True, True, True),
    )
    # Match the BF16 activation-storage boundary traversed by autograd, then
    # differentiate the exact one-round activation exactly as the trainer
    # does: through one_round_swiglu's own autograd definition.
    with torch.enable_grad():
        manual_gate_up_leaf = trainer_gate_up.detach().requires_grad_(True)
        (manual_gate_up_grad,) = torch.autograd.grad(
            one_round_swiglu(manual_gate_up_leaf),
            manual_gate_up_leaf,
            grad_outputs=manual_activation_grad.to(trainer_activation.dtype),
        )
    (
        manual_input_grad,
        manual_gate_A_grad,
        manual_gate_B_grad,
        manual_up_A_grad,
        manual_up_B_grad,
    ) = gate_up._surrogate_vjp(
        input.detach(),
        effective_gate_A,
        effective_gate_B,
        effective_up_A,
        effective_up_B,
        manual_gate_up_grad,
        needs_input_grad=(True, True, True, True, True),
    )
    (
        reference_input_grad,
        reference_gate_A_grad,
        reference_gate_B_grad,
        reference_up_A_grad,
        reference_up_B_grad,
    ) = _logical_gate_up_surrogate_reference(
        gate_up,
        input.detach(),
        effective_gate_A,
        effective_gate_B,
        effective_up_A,
        effective_up_B,
        manual_gate_up_grad,
    )
    assert torch.equal(manual_input_grad.to(torch.bfloat16), reference_input_grad)
    for actual, expected in zip(
        (manual_gate_A_grad, manual_gate_B_grad, manual_up_A_grad, manual_up_B_grad),
        (reference_gate_A_grad, reference_gate_B_grad, reference_up_A_grad, reference_up_B_grad),
        strict=True,
    ):
        assert torch.equal(actual, expected)

    trainer_output.backward(final_grad)

    assert torch.equal(trainer_activation.grad, manual_activation_grad.to(torch.bfloat16))
    assert torch.equal(trainer_gate_up.grad, manual_gate_up_grad)
    assert torch.equal(input.grad, reference_input_grad)
    expected_factor_gradients = {
        "gate_A": (gate_up.gate_proj.lora_A.grad, reference_gate_A_grad),
        "gate_B": (gate_up.gate_proj.lora_B.grad, reference_gate_B_grad),
        "up_A": (gate_up.up_proj.lora_A.grad, reference_up_A_grad),
        "up_B": (gate_up.up_proj.lora_B.grad, reference_up_B_grad),
        "down_A": (down.lora_A.grad, manual_down_A_grad),
        "down_B": (down.lora_B.grad, manual_down_B_grad),
    }
    for name, (actual, expected) in expected_factor_gradients.items():
        assert actual.dtype is torch.float32, name
        assert torch.equal(actual, expected), name
