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
# Fused signature: (permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size) -> output
# ---------------------------------------------------------------------------

EP_EXPERT_COMPUTE: Dict[str, Callable] = {}

# Triton EP compute (fused gate+up GEMM with autograd)
try:
    from xorl.ops.moe.triton import TritonEPGroupGemm

    EP_EXPERT_COMPUTE["triton"] = TritonEPGroupGemm.apply
except ImportError:
    pass

# Quack EP compute — adapt fused interface to old (gate_proj, up_proj) signature
try:
    from xorl.ops.moe.quack import QuackEPGroupGemm as _QuackEPGroupGemm

    def _quack_ep_fused(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size):
        gate_proj = gate_up_proj[..., :intermediate_size].contiguous()
        up_proj = gate_up_proj[..., intermediate_size:].contiguous()
        return _QuackEPGroupGemm.apply(permute_tokens, cumsum, gate_proj, up_proj, down_proj)

    EP_EXPERT_COMPUTE["quack"] = _quack_ep_fused
except ImportError:
    pass

# Native EP compute — adapt fused interface
try:
    from .native import native_ep_compute as _native_ep_compute

    def _native_ep_fused(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size):
        gate_proj = gate_up_proj[..., :intermediate_size].contiguous()
        up_proj = gate_up_proj[..., intermediate_size:].contiguous()
        return _native_ep_compute(permute_tokens, cumsum, gate_proj, up_proj, down_proj)

    EP_EXPERT_COMPUTE["native"] = _native_ep_fused
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
    from xorl.distributed.moe import alltoall_post_combine, alltoall_pre_dispatch

    EP_DISPATCH["alltoall"] = alltoall_pre_dispatch
    EP_COMBINE["alltoall"] = alltoall_post_combine
except ImportError:
    pass

# DeepEP dispatch (optional — requires deep_ep package)
try:
    from xorl.distributed.moe.deepep import (
        token_pre_dispatch as deepep_pre_dispatch,
    )
    from xorl.distributed.moe.deepep import (
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


# ---------------------------------------------------------------------------
# Moe_act registries: activation-recompute variants that drop gate_output/up_output
# from save_for_backward and recompute them in backward via local GEMMs.
# ---------------------------------------------------------------------------

EP_EXPERT_COMPUTE_MOE_ACT: Dict[str, Callable] = {}

# Triton EP moe_act compute
try:
    from xorl.ops.moe.triton import TritonEPGroupGemmMoeAct

    EP_EXPERT_COMPUTE_MOE_ACT["triton"] = TritonEPGroupGemmMoeAct.apply
except ImportError:
    pass

# Quack EP moe_act compute — adapt fused interface
try:
    from xorl.ops.moe.quack import QuackEPGroupGemmMoeAct as _QuackEPGroupGemmMoeAct

    def _quack_ep_moe_act_fused(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size):
        gate_proj = gate_up_proj[..., :intermediate_size].contiguous()
        up_proj = gate_up_proj[..., intermediate_size:].contiguous()
        return _QuackEPGroupGemmMoeAct.apply(permute_tokens, cumsum, gate_proj, up_proj, down_proj)

    EP_EXPERT_COMPUTE_MOE_ACT["quack"] = _quack_ep_moe_act_fused
except ImportError:
    pass

# Native EP moe_act compute — adapt fused interface
try:
    from .native import native_ep_compute_moe_act as _native_ep_compute_moe_act

    def _native_ep_moe_act_fused(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size):
        gate_proj = gate_up_proj[..., :intermediate_size].contiguous()
        up_proj = gate_up_proj[..., intermediate_size:].contiguous()
        return _native_ep_compute_moe_act(permute_tokens, cumsum, gate_proj, up_proj, down_proj)

    EP_EXPERT_COMPUTE_MOE_ACT["native"] = _native_ep_moe_act_fused
except ImportError:
    pass


MOE_EXPERT_BACKENDS_MOE_ACT: Dict[str, Callable] = {}

# Triton local moe_act compute
try:
    from .triton_moe_act import triton_expert_forward_moe_act

    MOE_EXPERT_BACKENDS_MOE_ACT["triton"] = triton_expert_forward_moe_act
except ImportError:
    pass

# Quack local moe_act compute
try:
    from .quack_moe_act import quack_expert_forward_moe_act

    MOE_EXPERT_BACKENDS_MOE_ACT["quack"] = quack_expert_forward_moe_act
except ImportError:
    pass

# Native local moe_act compute
try:
    from .native import native_expert_forward_moe_act

    MOE_EXPERT_BACKENDS_MOE_ACT["native"] = native_expert_forward_moe_act
except ImportError:
    pass


__all__ = [
    "MOE_EXPERT_BACKENDS",
    "EP_EXPERT_COMPUTE",
    "EP_DISPATCH",
    "EP_COMBINE",
    "EP_EXPERT_COMPUTE_LORA",
    "MOE_EXPERT_BACKENDS_LORA",
    "EP_EXPERT_COMPUTE_MOE_ACT",
    "MOE_EXPERT_BACKENDS_MOE_ACT",
]
