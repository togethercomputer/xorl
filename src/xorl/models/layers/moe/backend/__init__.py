"""
MoE expert backend registry.

Maps implementation name -> expert forward callable.
Follows the same pattern as ``layers.attention.backend.ATTENTION_FUNCTIONS``.
"""

from typing import Callable, Dict


MOE_EXPERT_BACKENDS: Dict[str, Callable] = {}

# Eager is always available (no kernel deps)
from .eager import eager_ep_compute, eager_ep_compute_lora, eager_expert_forward


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
EP_EXPERT_COMPUTE["eager"] = eager_ep_compute
EP_EXPERT_COMPUTE_MOE_ACT: Dict[str, Callable] = {}

# Triton EP compute (fused gate+up GEMM with autograd)
try:
    from xorl.ops.moe.triton import TritonEPGroupGemm, TritonEPGroupGemmMoeAct

    def _triton_ep_apply(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        swiglu_limit=0.0,
        gated=True,
        **_extras,  # GPT-OSS extras (gate_up_bias, down_bias) — triton doesn't support them
    ):
        del activation_native, fp8_grouped_backend, fp8_block_size
        if fp8_compute:
            raise NotImplementedError(
                "triton EP backend does not support FP8 expert compute. "
                "Use moe_implementation='quack' for FP8 grouped expert compute."
            )
        unsupported_extras = {key: value for key, value in _extras.items() if value is not None}
        if unsupported_extras or hidden_act == "clamped_swiglu":
            raise NotImplementedError(
                "triton EP backend does not support per-expert biases or clamped_swiglu "
                "activation (required by GPT-OSS). Use moe_implementation='native' instead."
            )
        return TritonEPGroupGemm.apply(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            intermediate_size,
            expert_scores,
            hidden_act,
            swiglu_limit,
            gated,
        )

    EP_EXPERT_COMPUTE["triton"] = _triton_ep_apply

    def _triton_ep_moe_act_apply(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        swiglu_limit=0.0,
        gated=True,
        **_extras,  # GPT-OSS extras (gate_up_bias, down_bias) — triton doesn't support them
    ):
        del activation_native, fp8_grouped_backend, fp8_block_size
        if fp8_compute:
            raise NotImplementedError(
                "triton moe_act EP backend does not support FP8 expert compute. "
                "Use moe_implementation='quack' for FP8 grouped expert compute."
            )
        unsupported_extras = {key: value for key, value in _extras.items() if value is not None}
        if unsupported_extras or hidden_act == "clamped_swiglu" or swiglu_limit:
            raise NotImplementedError(
                "triton moe_act EP backend does not support per-expert biases, clamped_swiglu, "
                "or swiglu_limit. Use a different checkpointing method or moe_implementation='native'."
            )
        if not gated:
            raise NotImplementedError("triton moe_act EP backend does not support non-gated experts")
        return TritonEPGroupGemmMoeAct.apply(
            permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores, hidden_act
        )

    EP_EXPERT_COMPUTE_MOE_ACT["triton"] = _triton_ep_moe_act_apply
except ImportError:
    pass

# Quack EP compute — fused gate_up_proj interface
try:
    from xorl.ops.moe.quack import QuackEPGroupGemm as _QuackEPGroupGemm

    def _quack_ep_fused(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        swiglu_limit=0.0,
        gated=True,
        **extras,  # GPT-OSS extras (gate_up_bias, down_bias)
    ):
        if not gated:
            raise NotImplementedError("quack backend does not support non-gated experts")
        unknown_extras = {
            key: value
            for key, value in extras.items()
            if key not in {"gate_up_bias", "down_bias"} and value is not None
        }
        if unknown_extras:
            raise NotImplementedError(f"quack EP backend received unsupported expert extras: {sorted(unknown_extras)}")
        return _QuackEPGroupGemm.apply(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            intermediate_size,
            expert_scores,
            hidden_act,
            activation_native,
            fp8_compute,
            fp8_grouped_backend,
            fp8_block_size,
            extras.get("gate_up_bias"),
            extras.get("down_bias"),
            swiglu_limit,
        )

    EP_EXPERT_COMPUTE["quack"] = _quack_ep_fused
except ImportError:
    pass

try:
    from xorl.ops.moe.quack import QuackEPGroupGemmMoeAct as _QuackEPGroupGemmMoeAct

    def _quack_ep_moe_act_fused(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        gated=True,
    ):
        if not gated:
            raise NotImplementedError("quack backend does not support non-gated experts")
        return _QuackEPGroupGemmMoeAct.apply(
            permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores, hidden_act
        )

    EP_EXPERT_COMPUTE_MOE_ACT["quack"] = _quack_ep_moe_act_fused
except ImportError:
    pass

# Native EP compute — fused gate_up_proj interface
try:
    from .native import native_ep_compute as _native_ep_compute

    def _native_ep_fused(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        swiglu_limit=0.0,
        gated=True,
        **extras,  # gate_up_bias, down_bias — optional, used for GPT-OSS
    ):
        del intermediate_size, activation_native, fp8_grouped_backend, fp8_block_size
        if fp8_compute:
            raise NotImplementedError(
                "native EP backend does not support FP8 expert compute. "
                "Use moe_implementation='quack' for FP8 grouped expert compute."
            )
        supported_extras = {"gate_up_bias", "down_bias"}
        unknown_extras = {
            key: value for key, value in extras.items() if key not in supported_extras and value is not None
        }
        if unknown_extras:
            raise NotImplementedError(f"native EP backend received unsupported expert extras: {sorted(unknown_extras)}")
        return _native_ep_compute(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            expert_scores,
            hidden_act=hidden_act,
            swiglu_limit=swiglu_limit,
            gate_up_bias=extras.get("gate_up_bias"),
            down_bias=extras.get("down_bias"),
            gated=gated,
        )

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
EP_EXPERT_COMPUTE_LORA["eager"] = eager_ep_compute_lora

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
    "EP_EXPERT_COMPUTE_MOE_ACT",
    "EP_DISPATCH",
    "EP_COMBINE",
    "EP_EXPERT_COMPUTE_LORA",
    "MOE_EXPERT_BACKENDS_LORA",
]
