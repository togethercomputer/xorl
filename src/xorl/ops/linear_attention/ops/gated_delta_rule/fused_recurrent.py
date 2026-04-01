from __future__ import annotations

# Adapted from flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.
import torch
import triton
import triton.language as tl

from xorl.ops.linear_attention.ops.utils.op import exp
from xorl.ops.linear_attention.utils import input_guard


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_GV": lambda args: args["gv"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_h *= exp(tl.load(p_g).to(tl.float32))
        if USE_GK:
            b_h *= exp(tl.load(p_gk).to(tl.float32)[:, None])
        if USE_GV:
            b_h *= exp(tl.load(p_gv).to(tl.float32)[None, :])

        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV * K
        if USE_GV:
            p_gv += HV * V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        del ctx
        return fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )


def fused_recurrent_gated_delta_rule(*args, **kwargs):
    return FusedRecurrentFunction.apply(*args, **kwargs)
