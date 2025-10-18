"""
MoE expert backend registry.

Maps implementation name -> expert forward callable.
Follows the same pattern as ``layers.attention.backend.ATTENTION_FUNCTIONS``.
"""

from typing import Callable, Dict

MOE_EXPERT_BACKENDS: Dict[str, Callable] = {}

# Eager is always available (no kernel deps)
from .eager import eager_expert_forward

MOE_EXPERT_BACKENDS["eager"] = eager_expert_forward

# Triton group GEMM kernels (custom Triton)
try:
    from .triton import triton_expert_forward

    MOE_EXPERT_BACKENDS["triton"] = triton_expert_forward
except ImportError:
    pass

# Native PyTorch grouped GEMM (torch._grouped_mm, backed by cuBLAS/CUTLASS)
try:
    from .native import native_expert_forward

    MOE_EXPERT_BACKENDS["native"] = native_expert_forward
except ImportError:
    pass

# Quack group GEMM kernels
try:
    from .quack import quack_expert_forward

    MOE_EXPERT_BACKENDS["quack"] = quack_expert_forward
except ImportError:
    pass

# ---------------------------------------------------------------------------
# EP expert compute registry
# Maps implementation name -> compute callable for Expert Parallelism.
# All compute functions share: (permute_tokens, cumsum, gate_proj, up_proj, down_proj) -> output
# ---------------------------------------------------------------------------

EP_EXPERT_COMPUTE: Dict[str, Callable] = {}

# Triton EP compute (custom Triton group GEMM with autograd)
try:
    from xorl.ops.moe.triton import TritonEPGroupGemm

    EP_EXPERT_COMPUTE["triton"] = TritonEPGroupGemm.apply
except ImportError:
    pass

# Quack EP compute (quack group GEMM with autograd)
try:
    from xorl.ops.moe.quack import QuackEPGroupGemm

    EP_EXPERT_COMPUTE["quack"] = QuackEPGroupGemm.apply
except ImportError:
    pass

# Native EP compute (torch._grouped_mm with alignment padding)
try:
    from .native import native_ep_compute

    EP_EXPERT_COMPUTE["native"] = native_ep_compute
except ImportError:
    pass


# ---------------------------------------------------------------------------
# EP dispatch/combine strategy registry
# Maps strategy name -> dispatch/combine callable.
# Dispatch: (hidden_states, routing_weights, selected_experts, num_experts, **kwargs) -> (permute_tokens, cumsum, ctx)
# Combine: (expert_output, ctx, **kwargs) -> output
# ---------------------------------------------------------------------------

EP_DISPATCH: Dict[str, Callable] = {}
EP_COMBINE: Dict[str, Callable] = {}

# AllToAll dispatch (always available when EP ops are available)
try:
    from xorl.distributed.moe import alltoall_pre_dispatch, alltoall_post_combine

    EP_DISPATCH["alltoall"] = alltoall_pre_dispatch
    EP_COMBINE["alltoall"] = alltoall_post_combine
except ImportError:
    pass

# DeepEP dispatch (optional — requires deep_ep package)
try:
    from xorl.distributed.moe.deepep import (
        token_pre_dispatch as deepep_pre_dispatch,
        tokens_post_combine as deepep_post_combine,
    )

    EP_DISPATCH["deepep"] = deepep_pre_dispatch
    EP_COMBINE["deepep"] = deepep_post_combine
except ImportError:
    pass

# ---------------------------------------------------------------------------
# EP expert compute with LoRA registry
# Maps implementation name -> compute callable for EP with LoRA adapters.
# All compute functions share: (permute_tokens, cumsum, gate_proj, up_proj, down_proj,
#     gate_proj_lora_A, gate_proj_lora_B, up_proj_lora_A, up_proj_lora_B,
#     down_proj_lora_A, down_proj_lora_B, scaling) -> output
# ---------------------------------------------------------------------------

EP_EXPERT_COMPUTE_LORA: Dict[str, Callable] = {}

# Triton EP LoRA compute (custom Triton group GEMM with autograd)
try:
    from xorl.ops.moe.triton_lora import TritonEPGroupGemmWithLoRA

    EP_EXPERT_COMPUTE_LORA["triton"] = TritonEPGroupGemmWithLoRA.apply
except ImportError:
    pass

# Quack EP LoRA compute (quack group GEMM with autograd)
try:
    from xorl.ops.moe.quack_lora import QuackEPGroupGemmWithLoRA

    EP_EXPERT_COMPUTE_LORA["quack"] = QuackEPGroupGemmWithLoRA.apply
except ImportError:
    pass

# Native EP LoRA compute (torch._grouped_mm with alignment padding)
try:
    from .native import native_ep_compute_lora

    EP_EXPERT_COMPUTE_LORA["native"] = native_ep_compute_lora
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Local expert compute with LoRA registry
# Maps implementation name -> local LoRA forward callable.
# All compute functions share: (num_experts, routing_weights, selected_experts,
#     hidden_states, gate_proj, up_proj, down_proj,
#     gate_proj_lora_A, gate_proj_lora_B, up_proj_lora_A, up_proj_lora_B,
#     down_proj_lora_A, down_proj_lora_B, scaling) -> output
# ---------------------------------------------------------------------------

MOE_EXPERT_BACKENDS_LORA: Dict[str, Callable] = {}

# Triton local LoRA compute
try:
    from xorl.ops.moe.triton_lora import triton_moe_lora_forward

    MOE_EXPERT_BACKENDS_LORA["triton"] = triton_moe_lora_forward
except ImportError:
    pass

# Quack local LoRA compute
try:
    from xorl.ops.moe.quack_lora import quack_moe_lora_forward

    MOE_EXPERT_BACKENDS_LORA["quack"] = quack_moe_lora_forward
except ImportError:
    pass

# Native local LoRA compute
try:
    from .native import native_expert_lora_forward

    MOE_EXPERT_BACKENDS_LORA["native"] = native_expert_lora_forward
except ImportError:
    pass


__all__ = [
    "MOE_EXPERT_BACKENDS",
    "EP_EXPERT_COMPUTE",
    "EP_DISPATCH",
    "EP_COMBINE",
    "EP_EXPERT_COMPUTE_LORA",
    "MOE_EXPERT_BACKENDS_LORA",
]
