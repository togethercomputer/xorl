"""Zero-copy XoRL GKN weight adapter for SGLang's fused-MoE runner.

XoRL stores expert weights in GKN layout while SGLang consumes the transposed
serving view. The paired xorl-sglang implementation accepts that view and uses
explicit tensor strides for both GEMMs; its TMA-only down path remains disabled
for non-contiguous weights.
"""

from __future__ import annotations

import functools
from typing import List, Optional

import torch


_STRIDED_FALLBACK_HINT = "set XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE=transient to use contiguous transpose copies"


@functools.lru_cache(maxsize=1)
def _load_fused_experts_impl():
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # noqa: PLC0415
            fused_experts_impl,
        )
    except ImportError as exc:
        raise ImportError(
            "XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE=strided requires the installed "
            "SGLang fused-MoE runner; "
            f"install the paired xorl-sglang package or {_STRIDED_FALLBACK_HINT}"
        ) from exc
    return fused_experts_impl


def serving_layout_or_gkn_view(weight: torch.Tensor) -> bool:
    """Return whether ``weight`` has serving layout or a zero-copy GKN view."""
    return weight.is_contiguous() or (weight.ndim == 3 and weight.transpose(1, 2).is_contiguous())


def fused_experts_impl_strided(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
    gate_up_interleaved: bool = False,
    a1_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Delegate to current SGLang while preserving XoRL's gate-first layout."""
    if not serving_layout_or_gkn_view(w1) or not serving_layout_or_gkn_view(w2):
        raise ValueError("Fused-MoE weights must be serving-contiguous or zero-copy GKN transpose views")
    if gate_up_interleaved:
        raise ValueError("XoRL GKN expert weights require gate_up_interleaved=False")

    return _load_fused_experts_impl()(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1=b1,
        b2=b2,
        inplace=inplace,
        activation=activation,
        is_gated=is_gated,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        swiglu_limit=swiglu_limit,
        gate_up_interleaved=gate_up_interleaved,
        a1_q=a1_q,
    )


__all__ = ["fused_experts_impl_strided", "serving_layout_or_gkn_view"]
