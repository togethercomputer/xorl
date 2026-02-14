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

from typing import Optional

import os
import torch
from quack.gemm_interface import gemm as quack_gemm


def _quack_tuned_enabled() -> bool:
    v = os.getenv("XORL_QUACK_TUNED", "0").strip().lower()
    return v not in {"0", "false", "no", "off"}


def quack_group_gemm_same_nk(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    max_M: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped GEMM with same N, K dimensions using quack-kernels."""
    del max_M  # Kept for API compatibility with Triton backend.
    assert not transpose_a, "transpose_a not supported in quack backend"

    if transpose_b:
        _, N, _ = b.shape
    else:
        _, _, N = b.shape

    total_M = a.shape[0]

    cu_seqlens_m = torch.cat([torch.zeros(1, device=cumsum_M.device, dtype=torch.int32), cumsum_M.to(torch.int32)])

    if transpose_b:
        # b: (G, N, K) -> (G, K, N) expected by quack input.
        b_quack = b.transpose(-2, -1).contiguous()
    else:
        b_quack = b

    if out is None:
        out = torch.empty(total_M, N, dtype=a.dtype, device=a.device)

    quack_gemm(
        A=a,
        B=b_quack,
        out=out,
        cu_seqlens_m=cu_seqlens_m,
        tuned=_quack_tuned_enabled(),
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
) -> None:
    """Grouped GEMM with same M, N dimensions using quack-kernels."""
    del max_K  # Kept for API compatibility with Triton backend.
    assert not transpose_b, "transpose_b not supported in quack_group_gemm_same_mn"

    cu_seqlens_k = torch.cat([torch.zeros(1, device=cumsum_K.device, dtype=torch.int32), cumsum_K.to(torch.int32)])

    if transpose_a:
        # a is (total_K, M), transpose to (M, total_K) with m-major layout.
        a_quack = a.T
    else:
        if a.stride(-2) != 1:
            a_quack = a.T.contiguous().T
        else:
            a_quack = a

    quack_gemm(
        A=a_quack,
        B=b,
        out=c,
        cu_seqlens_k=cu_seqlens_k,
        tuned=_quack_tuned_enabled(),
    )
