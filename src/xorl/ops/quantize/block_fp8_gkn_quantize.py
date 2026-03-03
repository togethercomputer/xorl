"""Block FP8 quantization/dequantization for 2D weight matrices (GKN format).

Autotuned Triton kernels for 2D block-based FP8 quantization of weight tensors
with shape [K, N]. Each block_size x block_size tile gets its own scale factor.

Autotuning covers num_warps and num_stages only — BLOCK_SIZE is fixed to the
user's block_size (default 128) since it determines quantization granularity.

Targets >2000 GB/s throughput on H100 (memory-bound: 5 bytes/element for
read f32 + write fp8, or read fp8 + write f32).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotuning configurations (memory-bound kernels)
# BLOCK_SIZE is NOT autotuned — it must match the user's block_size to keep
# scale tensor shapes consistent.
# ---------------------------------------------------------------------------

_quant_configs = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in [4, 8]
    for ns in [3, 4, 5]
]

_dequant_configs = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in [4, 8]
    for ns in [3, 4, 5]
]


# ---------------------------------------------------------------------------
# 2D weight quantization kernel (GKN)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_quant_configs, key=["M", "N"])
@triton.jit
def _block_fp8_quantize_gkn_kernel(
    x_ptr, y_ptr, s_ptr,
    M, N, clip_ratio, seed,
    BLOCK_SIZE: tl.constexpr,
    STOCHASTIC: tl.constexpr,
):
    """2D block-based quantization kernel for weight matrices.

    Each program processes a BLOCK_SIZE x BLOCK_SIZE tile:
      - Compute per-tile scale: max(abs(tile)) / 448.0 * clip_ratio
      - Guard: s = max(s, 1e-12)
      - Quantize to FP8 E4M3 (with optional stochastic rounding)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    # Apply clip_ratio: reduce scale to clip outliers (values > clip_ratio * max get saturated)
    s = s * clip_ratio
    s = tl.maximum(s, 1e-12)
    y = x / s
    if STOCHASTIC:
        # Add noise proportional to ULP before rounding to fp8.
        # ULP of fp8 e4m3 at |y| is approximately |y| * 2^-3 = |y| * 0.125
        # Noise ~ Uniform(-0.5, 0.5) * ulp makes rounding unbiased.
        noise = tl.rand(seed, offs) - 0.5
        ulp = tl.abs(y) * 0.125 + 1e-6  # guard for zero
        y = y + noise * ulp
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n_blocks + pid_n, s)


def block_fp8_quantize_gkn(
    x: torch.Tensor, block_size: int = 128,
    clip_ratio: float = 1.0,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight matrix to FP8 using 2D block-based quantization.

    Each block_size x block_size tile gets its own scale factor computed as
    max(abs(tile)) / 448.0 * clip_ratio.

    Args:
        x: Weight matrix of shape (K, N), any float dtype. Must be contiguous.
        block_size: Tile size for both dimensions (default: 128).
        clip_ratio: Scale clipping ratio (0.0-1.0). Lower = more clipping.
        stochastic_rounding: If True, add noise before FP8 cast for unbiased rounding.

    Returns:
        Tuple of (y, s):
            y: Quantized weight in FP8 E4M3 format, shape (K, N).
            s: Per-tile scale factors, shape (ceil(K/block_size), ceil(N/block_size)) float32.
    """
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        (triton.cdiv(M, block_size), triton.cdiv(N, block_size)),
        dtype=torch.float32, device=x.device,
    )
    grid = lambda meta: (
        triton.cdiv(M, block_size),
        triton.cdiv(N, block_size),
    )
    seed = 0 if not stochastic_rounding else torch.randint(0, 2**31, (1,)).item()
    _block_fp8_quantize_gkn_kernel[grid](
        x, y, s, M, N, clip_ratio, seed,
        BLOCK_SIZE=block_size, STOCHASTIC=stochastic_rounding,
    )
    return y, s


# ---------------------------------------------------------------------------
# 2D weight dequantization kernel (GKN)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_dequant_configs, key=["M", "N"])
@triton.jit
def _block_fp8_dequantize_gkn_kernel(
    x_ptr, s_ptr, y_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """2D block-based dequantization kernel for weight matrices.

    Each program processes a BLOCK_SIZE x BLOCK_SIZE tile, multiplying
    quantized values by the tile's scale factor.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n_blocks + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def block_fp8_dequantize_gkn(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Dequantize a 2D weight matrix using 2D block-based quantization.

    Args:
        x: Quantized weight matrix of shape (M, N).
        s: 2D scale tensor of shape (ceil(M/block_size), ceil(N/block_size)).
        block_size: Tile size (default: 128).

    Returns:
        Dequantized weight matrix in default dtype.
    """
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, block_size),
        triton.cdiv(N, block_size),
    )
    _block_fp8_dequantize_gkn_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

block_fp8_weight_quant = block_fp8_quantize_gkn
block_fp8_weight_dequant = block_fp8_dequantize_gkn
block_fp8_weight_quant_gkn = block_fp8_quantize_gkn
