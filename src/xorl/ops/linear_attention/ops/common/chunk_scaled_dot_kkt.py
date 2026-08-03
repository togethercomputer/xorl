# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.


import torch
import triton
import triton.language as tl

from xorl.ops.linear_attention.modules.bi_contract import is_gdn_contract_enabled
from xorl.ops.linear_attention.ops.utils import prepare_chunk_indices
from xorl.ops.linear_attention.ops.utils.op import exp
from xorl.ops.linear_attention.utils import autotune_cache_kwargs


@triton.jit(do_not_specialize=["T"])
def _chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    SAFE_EXP: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        if SAFE_EXP:
            # Serving masks the mathematically unused positive half before
            # exponentiation.  Besides avoiding overflow, this is part of the
            # compiled BI-prefill arithmetic contract.
            b_g_diff = tl.where(b_g_diff <= 0, b_g_diff, float("-inf"))
        b_A *= exp(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# Outside the bitwise-inference contract, retain the broad autotune space used
# by the native FLA path.  The contract path below launches the same kernel body
# directly with serving's fixed reduction geometry: KKT is a fp32 dot product,
# so allowing the autotuner to choose a different BK changes accumulation order
# and can perturb downstream bf16 activations even when every input bit agrees.
chunk_scaled_dot_kkt_fwd_kernel = triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "SAFE_EXP": lambda args: False,
    }
)(
    triton.autotune(
        configs=[
            triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
            for BK in [32, 64, 128]
            for num_warps in [2, 4, 8]
            for num_stages in [2, 3, 4]
        ],
        key=["H", "K", "BT", "IS_VARLEN"],
        **autotune_cache_kwargs,
    )(_chunk_scaled_dot_kkt_fwd_kernel)
)


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    launch_args = {
        "k": k,
        "g": g,
        "beta": beta,
        "A": A,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "T": T,
        "H": H,
        "K": K,
        "BT": BT,
    }
    if is_gdn_contract_enabled():
        _chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
            **launch_args,
            BK=64,
            IS_VARLEN=cu_seqlens is not None,
            USE_G=g is not None,
            SAFE_EXP=True,
            num_warps=8,
            num_stages=3,
        )
    else:
        chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](**launch_args)
    return A
