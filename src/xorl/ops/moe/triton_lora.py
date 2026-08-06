"""Triton MoE LoRA implementations (EP + local).

Triton-specific instances of the backend-agnostic factory functions
in ``lora``, parameterised with Triton group GEMM kernels.
"""

import torch

from xorl.ops.group_gemm.kernel import group_gemm_same_mn, group_gemm_same_nk

from .lora import make_ep_lora_compute, make_local_lora_compute


# EP LoRA compute (Expert Parallelism)
_TritonEPGroupGemmWithLoRA = make_ep_lora_compute(group_gemm_same_nk, group_gemm_same_mn)


class TritonEPGroupGemmWithLoRA:
    """DTensor/FSDP-safe dtype adapter around the Triton autograd kernel.

    The grouped GEMM kernels accept BF16/FP16. Keep any FP32 FSDP parameter
    views outside the custom autograd boundary so normal cast backward writes
    the returned LoRA gradients into their authoritative FP32 parameters.
    """

    @staticmethod
    def apply(
        permute_tokens,
        cumsum,
        gate_proj,
        up_proj,
        down_proj,
        gate_proj_lora_A,
        gate_proj_lora_B,
        up_proj_lora_A,
        up_proj_lora_B,
        down_proj_lora_A,
        down_proj_lora_B,
        scaling,
        swiglu_limit=0.0,
    ):
        output_dtype = permute_tokens.dtype
        if permute_tokens.dtype not in {torch.bfloat16, torch.float16}:
            compute_dtype = torch.bfloat16
            permute_tokens = permute_tokens.to(compute_dtype)
            gate_proj = gate_proj.to(compute_dtype)
            up_proj = up_proj.to(compute_dtype)
            down_proj = down_proj.to(compute_dtype)
        output = _TritonEPGroupGemmWithLoRA.apply(
            permute_tokens,
            cumsum,
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_lora_A,
            gate_proj_lora_B,
            up_proj_lora_A,
            up_proj_lora_B,
            down_proj_lora_A,
            down_proj_lora_B,
            scaling,
            swiglu_limit,
        )
        return output.to(output_dtype)


# Local LoRA compute (single-GPU)
TritonMoeExpertsLoRAFunction = make_local_lora_compute(group_gemm_same_nk, group_gemm_same_mn)
TritonMoeExpertsLoRAFunction.__name__ = TritonMoeExpertsLoRAFunction.__qualname__ = "TritonMoeExpertsLoRAFunction"


def triton_moe_lora_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_proj_lora_A: torch.Tensor,
    gate_proj_lora_B: torch.Tensor,
    up_proj_lora_A: torch.Tensor,
    up_proj_lora_B: torch.Tensor,
    down_proj_lora_A: torch.Tensor,
    down_proj_lora_B: torch.Tensor,
    scaling: float,
    swiglu_limit: float = 0.0,
):
    """MoE + LoRA forward pass (local single-GPU path).

    EP is handled centrally by ``MoEExpertsLoRA._ep_forward()``.
    """
    return TritonMoeExpertsLoRAFunction.apply(
        num_experts,
        routing_weights.to(hidden_states.dtype),
        selected_experts,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_proj_lora_A,
        gate_proj_lora_B,
        up_proj_lora_A,
        up_proj_lora_B,
        down_proj_lora_A,
        down_proj_lora_B,
        scaling,
        swiglu_limit,
    )
