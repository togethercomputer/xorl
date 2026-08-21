"""Opt-in decode-shaped prep composition for FlashQLA recompute-decode.

The certified pinned recompute-decode path (with auto_cp disabled) spends more
time in unfused preparation than
in the fused TileLang kernel at the bs x 64 padded decode shape, dominated by
the hierarchical ``solve_tril`` merge. This module composes the same pipeline
with the decode-scheduled triangular solve (:mod:`.ops.utils.solve_tril_decode`,
bitwise-identical to the pinned kernel) and drops the ``zeros_like`` fill.

Everything else is byte-for-byte the pinned composition: the inductor l2norm,
the TileLang ``chunk_local_cumsum``, the FLA ``chunk_scaled_dot_kkt_fwd`` and
the fused ``fused_gdr_fwd`` kernel are called unchanged, so the output is
bitwise-identical to ``chunk_gated_delta_rule_fwd(..., auto_cp=False)`` with
l2norm applied outside. Nothing routes here by default; callers opt in.
"""

from __future__ import annotations

import torch

from xorl.ops.linear_attention.ops.utils.solve_tril_decode import solve_tril_decode


def chunk_gated_delta_rule_fwd_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute-decode forward, bitwise ≡ the pinned FlashQLA composition.

    Interface follows the cert harness' pinned call: raw (un-normalized) q/k in,
    l2norm applied in here, ``auto_cp`` never engaged, fp32 final state out.
    ``chunk_indices`` may be passed to keep the varlen index derivation out of
    the per-step path (it is static for the fixed padded decode shape).
    """
    from xorl.ops.linear_attention import tilelang_gemm_v1  # noqa: PLC0415

    tilelang_gemm_v1.patch()
    from xorl.ops._vendored.flashqla.ops.gated_delta_rule.chunk.hopper import fused_gdr_fwd  # noqa: PLC0415
    from xorl.ops._vendored.flashqla.ops.utils import chunk_local_cumsum  # noqa: PLC0415
    from xorl.ops._vendored.flashqla.utils import l2norm  # noqa: PLC0415
    from xorl.ops.linear_attention.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: PLC0415

    if scale is None:
        scale = k.shape[-1] ** -0.5

    q = l2norm(q)
    k = l2norm(k)
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=None, beta=beta, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    A = solve_tril_decode(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    o, _, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        output_h=False,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=None,
        raw_cu_seqlens=None,
    )
    return o.to(q.dtype), final_state
