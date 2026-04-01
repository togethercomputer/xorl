"""Block FP8 quantization/dequantization for 1D activations and FP8 GEMM.

1D block-based quantization: divides tensors into blocks along the last dimension,
computing per-block scale factors as max(abs(block)) / 448.0 (FP8 E4M3 max).

Also provides an autotuned FP8 GEMM kernel for block-quantized matmul.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


# ---------------------------------------------------------------------------
# 1D quantization kernel + wrapper
# ---------------------------------------------------------------------------


@triton.jit
def _block_fp8_quantize_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """1D block-based quantization kernel for FP8 conversion.

    Quantizes blocks of BLOCK_SIZE elements to FP8 (E4M3) format.
    For each block:
      - Load BLOCK_SIZE float32 values
      - Compute per-block scale: max(abs(x)) / 448.0
      - Scale values and convert to FP8
      - Store quantized values and scale factor

    Note: 448.0 is the max representable value in FP8 E4M3 format.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def block_fp8_quantize(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 using 1D block-based quantization.

    Divides the input tensor into blocks along the last dimension and quantizes
    each block independently to FP8 E4M3 format. Each block gets its own scale
    factor computed as max(abs(block)) / 448.0.

    Args:
        x: Input tensor to quantize (any shape, float32). Must be contiguous
           with last dim divisible by block_size.
        block_size: Number of elements per quantization block (default: 128).

    Returns:
        Tuple of (y, s):
            y: Quantized tensor in FP8 E4M3 format, same shape as x.
            s: Per-block scale factors, shape (*x.shape[:-1], x.shape[-1]//block_size).
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    _block_fp8_quantize_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# Backward-compat alias
block_fp8_quant = block_fp8_quantize


# ---------------------------------------------------------------------------
# 1D dequantization kernel + wrapper
# ---------------------------------------------------------------------------


@triton.jit
def _block_fp8_dequantize_kernel(y_ptr, s_ptr, x_ptr, BLOCK_SIZE: tl.constexpr):
    """1D block-based dequantization kernel.

    Each thread block processes BLOCK_SIZE consecutive elements, multiplying
    the quantized values by their per-block scale factor.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    s = tl.load(s_ptr + pid)
    y = tl.load(y_ptr + offs).to(tl.float32)
    x = y * s
    tl.store(x_ptr + offs, x)


def block_fp8_dequantize(y: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize a tensor using 1D block-based quantization along the last dimension.

    Args:
        y: Quantized tensor (any shape, FP8). Must be contiguous with last dim
           divisible by block_size.
        s: Per-block scale values, shape (*y.shape[:-1], y.shape[-1]//block_size).
        block_size: Number of elements per block (default: 128).

    Returns:
        Dequantized tensor in float32.
    """
    assert y.is_contiguous()
    assert y.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"
    x = torch.empty_like(y, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(y.numel(), meta["BLOCK_SIZE"]),)
    _block_fp8_dequantize_kernel[grid](y, s, x, BLOCK_SIZE=block_size)
    return x


# Backward-compat alias
block_fp8_dequant = block_fp8_dequantize


# ---------------------------------------------------------------------------
# Autotuned FP8 GEMM
# ---------------------------------------------------------------------------

_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=_gemm_configs, key=["N", "K"])
@triton.jit
def _block_fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Tiled matmul kernel for block-quantized FP8 tensors.

    Computes C[m, n] = sum_k(A[m, k] * scale_a[m, k//128] * B[n, k] * scale_b[n, k//128])
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def block_fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """Block-wise FP8 matrix multiplication with per-block scaling.

    Computes C = A @ B^T where A and B are block-quantized FP8 tensors.

    Args:
        a: Quantized activation tensor, shape (..., K).
        a_s: Per-block scales for a, shape (..., K//128).
        b: Quantized weight tensor, shape (N, K).
        b_s: Per-block scales for b, shape (N//128, K//128).

    Returns:
        Result in default dtype, shape (..., N).
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _block_fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
