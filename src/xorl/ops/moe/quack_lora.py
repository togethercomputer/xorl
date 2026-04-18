"""Quack-backend MoE LoRA implementations (EP + local).

Created by the same factory functions used for triton, but parameterised
with quack group GEMM kernels.  The quack and triton GEMM APIs are
drop-in replacements — only the kernel function references differ.
"""

from xorl.ops.group_gemm.kernel.quack import quack_group_gemm_same_mn, quack_group_gemm_same_nk

from .lora import make_ep_lora_compute, make_local_lora_compute


# EP LoRA compute (Expert Parallelism)
QuackEPGroupGemmWithLoRA = make_ep_lora_compute(quack_group_gemm_same_nk, quack_group_gemm_same_mn)
QuackEPGroupGemmWithLoRA.__name__ = QuackEPGroupGemmWithLoRA.__qualname__ = "QuackEPGroupGemmWithLoRA"

# Local LoRA compute (single-GPU)
QuackMoeExpertsLoRAFunction = make_local_lora_compute(quack_group_gemm_same_nk, quack_group_gemm_same_mn)
QuackMoeExpertsLoRAFunction.__name__ = QuackMoeExpertsLoRAFunction.__qualname__ = "QuackMoeExpertsLoRAFunction"


def quack_moe_lora_forward(
    num_experts: int,
    routing_weights,
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
    scaling: float,
):
    """MoE + LoRA forward pass using quack group GEMM kernels (local single-GPU)."""
    return QuackMoeExpertsLoRAFunction.apply(
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
    )
