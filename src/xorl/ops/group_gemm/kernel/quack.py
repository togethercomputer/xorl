# Copyright 2025 xorl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quack backend for Group GEMM operations.

This module provides wrapper functions that adapt quack-kernels' CuTe-based GEMM
kernels to the Xorl group GEMM API. Requires SM90+ (H100) or SM100+ (B200).
"""

import os
from typing import Optional

import torch

from xorl.ops.quack.gemm_interface import gemm as quack_gemm


def _quack_tuned_enabled() -> bool:
    v = os.getenv("XORL_QUACK_TUNED", "0").strip().lower()
    return v not in {"0", "false", "no", "off"}


_TUNED = _quack_tuned_enabled()


def cumsum_to_cu_seqlens(cumsum: torch.Tensor) -> torch.Tensor:
    """Convert cumsum to cu_seqlens by prepending a zero.

    Computes ``cu_seqlens = cat([0, cumsum])`` as a single int32 tensor.
    Call this ONCE per forward/backward and reuse across multiple GEMM calls.
    """
    return torch.cat([torch.zeros(1, device=cumsum.device, dtype=torch.int32), cumsum.to(torch.int32)])


def quack_group_gemm_same_nk(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    max_M: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_m: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped GEMM with same N, K dimensions using quack-kernels."""
    del max_M  # Kept for API compatibility with Triton backend.
    assert not transpose_a, "transpose_a not supported in quack backend"

    if transpose_b:
        _, N, _ = b.shape
    else:
        _, _, N = b.shape

    total_M = a.shape[0]

    if cu_seqlens_m is None:
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum_M)

    if transpose_b:
        # b: (G, N, K) -> (G, K, N) expected by quack input.
        b_quack = b.transpose(-2, -1).contiguous()
    else:
        b_quack = b

    if out is None:
        out = torch.empty(total_M, N, dtype=a.dtype, device=a.device)

    # tuned=False by default: MoE group GEMMs have variable total_M (token counts
    # change every step due to routing).  The quack autotuner keys on tensor shapes,
    # so each new total_M triggers a full ~60-config benchmark — making training
    # orders of magnitude slower.  Set XORL_QUACK_TUNED=1 to enable autotuning
    # (useful if shapes are stable or disk cache is warm).
    quack_gemm(
        A=a,
        B=b_quack,
        out=out,
        cu_seqlens_m=cu_seqlens_m,
        tuned=_TUNED,
    )

    return out


def quack_group_gemm_same_mn(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    cumsum_K: torch.Tensor,
    max_K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    cu_seqlens_k: Optional[torch.Tensor] = None,
) -> None:
    """Grouped GEMM with same M, N dimensions using quack-kernels."""
    del max_K  # Kept for API compatibility with Triton backend.
    assert not transpose_b, "transpose_b not supported in quack_group_gemm_same_mn"

    if cu_seqlens_k is None:
        cu_seqlens_k = cumsum_to_cu_seqlens(cumsum_K)

    if transpose_a:
        # a is (total_K, M), transpose to (M, total_K) with m-major layout.
        a_quack = a.T
    else:
        if a.stride(-2) != 1:
            a_quack = a.T.contiguous().T
        else:
            a_quack = a

    # tuned: same rationale as quack_group_gemm_same_nk.
    quack_gemm(
        A=a_quack,
        B=b,
        out=c,
        cu_seqlens_k=cu_seqlens_k,
        tuned=_TUNED,
    )
