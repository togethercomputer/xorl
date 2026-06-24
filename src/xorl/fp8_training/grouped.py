"""Experimental FP8 grouped GEMM helpers for MoE training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


try:
    import deep_gemm as _DEEP_GEMM
    from deep_gemm.utils import per_block_cast_to_fp8 as _DEEP_GEMM_PER_BLOCK_CAST_TO_FP8
    from deep_gemm.utils import per_token_cast_to_fp8 as _DEEP_GEMM_PER_TOKEN_CAST_TO_FP8

    _DEEP_GEMM_IMPORT_ERROR: BaseException | None = None
except (ImportError, RuntimeError, AssertionError) as exc:  # pragma: no cover - optional dependency.
    _DEEP_GEMM = None
    _DEEP_GEMM_PER_BLOCK_CAST_TO_FP8 = None
    _DEEP_GEMM_PER_TOKEN_CAST_TO_FP8 = None
    _DEEP_GEMM_IMPORT_ERROR = exc

_FP8_E4M3_MAX = 448.0
_FP8_BLOCK_SIZE = 128
DEFAULT_FP8_GROUPED_BACKEND = "triton_grouped"
_FP8_GROUPED_BACKENDS = {"block_loop", "deep_gemm", "scalar_quack", "triton_grouped"}


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _output_dtype(*tensors: Tensor) -> torch.dtype:
    for tensor in tensors:
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            return tensor.dtype
    return torch.bfloat16


def _quantize_expert_scalar_fp8_same_nk(a: Tensor, b: Tensor, cumsum_m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize grouped same-NK operands with one scalar scale per expert."""

    if cumsum_m.device != a.device:
        cumsum_m = cumsum_m.to(a.device)
    starts = torch.cat([torch.zeros(1, dtype=cumsum_m.dtype, device=cumsum_m.device), cumsum_m[:-1]])
    a_fp8 = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    output_scales = torch.empty(b.shape[0], dtype=torch.float32, device=a.device)

    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum_m.tolist())):
        a_chunk = a[start:end]
        if a_chunk.numel() == 0:
            a_scale = a.new_tensor(1e-12, dtype=torch.float32)
        else:
            a_float = a_chunk.float()
            a_scale = a_float.abs().amax().clamp_min(1e-12) / _FP8_E4M3_MAX
            a_fp8[start:end].copy_(
                (a_float / a_scale).clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            )

        b_float = b[expert_idx].float()
        b_scale = b_float.abs().amax().clamp_min(1e-12) / _FP8_E4M3_MAX
        b_fp8[expert_idx].copy_(
            (b_float / b_scale).clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        )
        output_scales[expert_idx] = a_scale * b_scale

    return a_fp8.contiguous(), b_fp8.contiguous(), output_scales


def _rescale_expert_segments_(out: Tensor, scales: Tensor, cumsum_m: Tensor) -> Tensor:
    if cumsum_m.device != out.device:
        cumsum_m = cumsum_m.to(out.device)
    starts = torch.cat([torch.zeros(1, dtype=cumsum_m.dtype, device=cumsum_m.device), cumsum_m[:-1]])
    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum_m.tolist())):
        if end != start:
            out[start:end].mul_(scales[expert_idx].to(out.dtype))
    return out


def _pad_last_dim(x: Tensor, multiple: int) -> Tensor:
    pad = (-x.shape[-1]) % multiple
    if pad == 0:
        return x.contiguous()
    return F.pad(x, (0, pad)).contiguous()


def _block_fp8_matmul(a: Tensor, b_t: Tensor, block_size: int = _FP8_BLOCK_SIZE) -> Tensor:
    """Compute ``a @ b_t.T`` with block-FP8 operands."""

    from xorl.ops.quantize import block_fp8_gemm, block_fp8_quantize, block_fp8_quantize_gkn_rowwise  # noqa: PLC0415

    if a.shape[-1] != b_t.shape[-1]:
        raise RuntimeError(f"FP8 grouped matmul contraction mismatch: {a.shape[-1]} != {b_t.shape[-1]}")
    a_padded = _pad_last_dim(a.float(), block_size)
    b_padded = _pad_last_dim(b_t.float(), block_size)
    a_fp8, a_scales = block_fp8_quantize(a_padded, block_size=block_size)
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(b_padded, block_size=block_size)
    return block_fp8_gemm(
        a_fp8,
        a_scales,
        b_fp8,
        b_scales,
        block_size=block_size,
        weight_scale_layout="row",
        backend="auto",
    )


@triton.jit
def _batched_block_fp8_quantize_gkn_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_idx = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_base = expert_idx * N * K
    offsets = expert_base + offs_n[:, None] * K + offs_k[None, :]
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-12)
    y = (x / scale).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(s_ptr + expert_idx * N_BLOCKS * K_BLOCKS + pid_n * K_BLOCKS + pid_k, scale)


def _batched_block_fp8_quantize_gkn(x: Tensor, block_size: int = _FP8_BLOCK_SIZE) -> tuple[Tensor, Tensor]:
    """Quantize a contiguous ``[G, N, K]`` tensor with per-expert GKN block scales."""

    if not x.is_cuda:
        raise RuntimeError("batched block-FP8 GKN quantization requires a CUDA tensor")
    if x.dim() != 3 or not x.is_contiguous():
        raise RuntimeError(f"batched block-FP8 GKN quantization expects contiguous rank-3 input, got {x.shape=}")
    num_experts, n, k = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        num_experts,
        _ceil_div(n, block_size),
        _ceil_div(k, block_size),
        dtype=torch.float32,
        device=x.device,
    )
    if num_experts == 0 or n == 0 or k == 0:
        return y, scales

    grid = (
        triton.cdiv(n, block_size),
        triton.cdiv(k, block_size),
        num_experts,
    )
    _batched_block_fp8_quantize_gkn_kernel[grid](
        x,
        y,
        scales,
        N=n,
        K=k,
        N_BLOCKS=scales.shape[1],
        K_BLOCKS=scales.shape[2],
        BLOCK_SIZE=block_size,
    )
    return y, scales


@triton.jit
def _batched_block_fp8_quantize_gkn_rowwise_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_idx = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = expert_idx * N * K + pid_n * K + offs_k
    mask = (pid_n < N) & (offs_k < K)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-12)
    y = (x / scale).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(s_ptr + expert_idx * N * K_BLOCKS + pid_n * K_BLOCKS + pid_k, scale)


def _batched_block_fp8_quantize_gkn_rowwise(x: Tensor, block_size: int = _FP8_BLOCK_SIZE) -> tuple[Tensor, Tensor]:
    """Quantize a contiguous ``[G, N, K]`` tensor with per-row K-block scales."""

    if not x.is_cuda:
        raise RuntimeError("batched rowwise block-FP8 GKN quantization requires a CUDA tensor")
    if x.dim() != 3 or not x.is_contiguous():
        raise RuntimeError(f"batched rowwise block-FP8 GKN quantization expects contiguous rank-3 input, got {x.shape=}")
    num_experts, n, k = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        num_experts,
        n,
        _ceil_div(k, block_size),
        dtype=torch.float32,
        device=x.device,
    )
    if num_experts == 0 or n == 0 or k == 0:
        return y, scales

    grid = (
        n,
        triton.cdiv(k, block_size),
        num_experts,
    )
    _batched_block_fp8_quantize_gkn_rowwise_kernel[grid](
        x,
        y,
        scales,
        N=n,
        K=k,
        K_BLOCKS=scales.shape[2],
        BLOCK_SIZE=block_size,
    )
    return y, scales


@triton.jit
def _grouped_block_fp8_same_nk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    cu_seqlens_ptr,
    N: tl.constexpr,
    K_PADDED: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    expert_idx = tl.program_id(axis=2)

    expert_start = tl.load(cu_seqlens_ptr + expert_idx)
    expert_end = tl.load(cu_seqlens_ptr + expert_idx + 1)
    expert_m = expert_end - expert_start

    local_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_m = expert_start + local_m
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K_PADDED + offs_k[None, :]
    b_expert_base = b_ptr + expert_idx * N * K_PADDED
    b_ptrs = b_expert_base + offs_n[None, :] * K_PADDED + offs_k[:, None]

    a_s_ptrs = a_s_ptr + offs_m * K_BLOCKS
    b_s_expert_base = b_s_ptr + expert_idx * N * K_BLOCKS
    b_s_ptrs = b_s_expert_base + offs_n * K_BLOCKS

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block_idx in range(K_BLOCKS):
        a = tl.load(a_ptrs, mask=local_m[:, None] < expert_m, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        a_s = tl.load(a_s_ptrs, mask=local_m < expert_m, other=0.0)
        b_s = tl.load(b_s_ptrs, mask=offs_n < N, other=0.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (local_m[:, None] < expert_m) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_block_fp8_same_mn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    cu_seqlens_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    expert_idx = tl.program_id(axis=2)

    expert_start = tl.load(cu_seqlens_ptr + expert_idx)
    expert_end = tl.load(cu_seqlens_ptr + expert_idx + 1)
    expert_k = expert_end - expert_start

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block_idx in range(K_BLOCKS):
        local_k = k_block_idx * BLOCK_SIZE_K + offs_k
        global_k = expert_start + local_k

        a_ptrs = a_ptr + global_k[None, :] * M + offs_m[:, None]
        b_ptrs = b_ptr + global_k[:, None] * N + offs_n[None, :]
        a_mask = (offs_m[:, None] < M) & (local_k[None, :] < expert_k)
        b_mask = (local_k[:, None] < expert_k) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        a_scale = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-12)
        b_amax_per_n = tl.max(tl.abs(b), axis=0)
        b_scale = tl.maximum(tl.max(b_amax_per_n, axis=0) / 448.0, 1e-12)

        a_fp8 = (a / a_scale[:, None]).to(tl.float8e4nv)
        b_fp8 = (b / b_scale).to(tl.float8e4nv)
        accumulator += tl.dot(a_fp8, b_fp8) * a_scale[:, None] * b_scale

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_expert_base = c_ptr + expert_idx * M * N
    c_ptrs = c_expert_base + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def validate_fp8_grouped_backend(backend: str) -> str:
    if backend not in _FP8_GROUPED_BACKENDS:
        raise ValueError(
            f"Unsupported FP8 grouped backend: {backend!r}. "
            f"Expected one of {sorted(_FP8_GROUPED_BACKENDS)}"
        )
    return backend


def fp8_block_loop_group_gemm_same_nk(
    *,
    a: Tensor,
    b: Tensor,
    cumsum_M: Tensor,
    max_M: int,
    transpose_b: bool = False,
    out: Tensor | None = None,
    cu_seqlens_m: Tensor | None = None,
    block_size: int = _FP8_BLOCK_SIZE,
) -> Tensor:
    """Grouped same-NK GEMM using per-expert block-FP8 matmuls.

    This is the stable full-model training backend: activations and expert
    weights are quantized to E4M3 with block scales immediately before each
    GEMM, while optimizer-visible parameters remain BF16/FP32 masters.
    """

    del max_M, cu_seqlens_m
    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("block_loop FP8 grouped backend requires CUDA tensors")

    n = b.shape[1] if transpose_b else b.shape[2]
    if out is None:
        out = torch.empty(a.shape[0], n, dtype=_output_dtype(a, b), device=a.device)
    if a.shape[0] == 0:
        return out

    starts = torch.cat([torch.zeros(1, dtype=cumsum_M.dtype, device=cumsum_M.device), cumsum_M[:-1]])
    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum_M.tolist())):
        if end == start:
            continue
        lhs = a[start:end].contiguous()
        rhs_t = b[expert_idx].contiguous() if transpose_b else b[expert_idx].T.contiguous()
        out[start:end].copy_(_block_fp8_matmul(lhs, rhs_t, block_size=block_size).to(out.dtype))
    return out


def fp8_triton_grouped_group_gemm_same_nk(
    *,
    a: Tensor,
    b: Tensor,
    cumsum_M: Tensor,
    max_M: int,
    transpose_b: bool = False,
    out: Tensor | None = None,
    cu_seqlens_m: Tensor | None = None,
    block_size: int = _FP8_BLOCK_SIZE,
) -> Tensor:
    """Grouped same-NK GEMM using one Triton block-FP8 grouped kernel."""

    from xorl.ops.quantize import block_fp8_quantize  # noqa: PLC0415

    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("triton_grouped FP8 grouped backend requires CUDA tensors")

    if transpose_b:
        n = b.shape[1]
        b_t = b.contiguous()
    else:
        n = b.shape[2]
        b_t = b.transpose(-2, -1).contiguous()

    if out is None:
        out = torch.empty(a.shape[0], n, dtype=_output_dtype(a, b), device=a.device)
    if a.shape[0] == 0:
        return out
    if cumsum_M.numel() == 0:
        return out

    a_padded = _pad_last_dim(a.float(), block_size)
    b_padded = _pad_last_dim(b_t.float(), block_size)
    k_padded = a_padded.shape[-1]
    if b_padded.shape[-1] != k_padded:
        raise RuntimeError(f"FP8 grouped same_nk contraction mismatch: {a.shape[-1]} != {b_t.shape[-1]}")

    a_fp8, a_scales = block_fp8_quantize(a_padded, block_size=block_size)
    b_fp8, b_scales = _batched_block_fp8_quantize_gkn_rowwise(b_padded, block_size=block_size)

    block_m = 32
    block_n = 64 if n <= 64 else 128
    grid = (
        triton.cdiv(max(1, int(max_M)), block_m),
        triton.cdiv(n, block_n),
        b_padded.shape[0],
    )
    cu_seqlens = (
        cu_seqlens_m
        if cu_seqlens_m is not None
        else torch.cat([torch.zeros(1, dtype=cumsum_M.dtype, device=cumsum_M.device), cumsum_M])
    )
    _grouped_block_fp8_same_nk_kernel[grid](
        a_fp8,
        b_fp8,
        out,
        a_scales,
        b_scales,
        cu_seqlens,
        N=n,
        K_PADDED=k_padded,
        K_BLOCKS=k_padded // block_size,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_size,
    )
    return out


def fp8_scalar_quack_group_gemm_same_nk(
    *,
    a: Tensor,
    b: Tensor,
    cumsum_M: Tensor,
    max_M: int,
    transpose_b: bool = False,
    out: Tensor | None = None,
    cu_seqlens_m: Tensor | None = None,
    block_size: int = _FP8_BLOCK_SIZE,
) -> Tensor:
    """Quack grouped GEMM using per-expert scalar-scaled FP8 operands.

    This uses Quack's native FP8 grouped GEMM and rescales each expert segment
    after the matmul. It is less accurate than the block-scaled Triton path but
    avoids the very wide dynamic range of a single global scale across all
    routed tokens and experts.
    """

    from xorl.ops.group_gemm.kernel.quack import quack_group_gemm_same_nk  # noqa: PLC0415

    del block_size

    effective_cumsum_m = cu_seqlens_m[1:] if cu_seqlens_m is not None else cumsum_M
    a_fp8, b_fp8, output_scales = _quantize_expert_scalar_fp8_same_nk(a, b, effective_cumsum_m)
    if out is None:
        n = b.shape[1] if transpose_b else b.shape[2]
        out = torch.empty(a.shape[0], n, dtype=_output_dtype(a, b), device=a.device)

    quack_group_gemm_same_nk(
        a=a_fp8,
        b=b_fp8,
        cumsum_M=cumsum_M,
        max_M=max_M,
        transpose_b=transpose_b,
        out=out,
        cu_seqlens_m=cu_seqlens_m,
    )
    return _rescale_expert_segments_(out, output_scales, effective_cumsum_m)


def fp8_deep_gemm_group_gemm_same_nk(
    *,
    a: Tensor,
    b: Tensor,
    cumsum_M: Tensor,
    max_M: int,
    transpose_b: bool = False,
    out: Tensor | None = None,
    cu_seqlens_m: Tensor | None = None,
    block_size: int = _FP8_BLOCK_SIZE,
) -> Tensor:
    """DeepGEMM M-grouped FP8 GEMM with per-row K-block scales.

    DeepGEMM's contiguous M-grouped layout requires each expert segment to be
    padded to its M-block alignment. XoRL's grouped GEMM contract is compact, so
    this helper pads before the kernel and compacts back into the requested
    output layout.
    """

    del max_M, cu_seqlens_m
    if block_size != _FP8_BLOCK_SIZE:
        raise ValueError("DeepGEMM FP8 grouped backend currently requires block_size=128")
    deep_gemm = _DEEP_GEMM
    per_block_cast_to_fp8 = _DEEP_GEMM_PER_BLOCK_CAST_TO_FP8
    per_token_cast_to_fp8 = _DEEP_GEMM_PER_TOKEN_CAST_TO_FP8
    if (
        deep_gemm is None
        or per_block_cast_to_fp8 is None
        or per_token_cast_to_fp8 is None
    ):  # pragma: no cover - exercised without optional dependency.
        raise RuntimeError("DeepGEMM FP8 grouped backend requested but deep_gemm is not installed") from (
            _DEEP_GEMM_IMPORT_ERROR
        )

    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("DeepGEMM FP8 grouped backend requires CUDA tensors")

    if transpose_b:
        n = b.shape[1]
        b_nt = b.contiguous()
    else:
        n = b.shape[2]
        b_nt = b.transpose(-2, -1).contiguous()

    if out is None:
        out = torch.empty(a.shape[0], n, dtype=_output_dtype(a, b), device=a.device)
    if a.shape[0] == 0:
        return out

    alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    starts = torch.cat([torch.zeros(1, dtype=cumsum_M.dtype, device=cumsum_M.device), cumsum_M[:-1]])
    chunks: list[Tensor] = []
    m_indices: list[int] = []
    ranges: list[tuple[int, int, int, int]] = []
    padded_start = 0
    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum_M.tolist())):
        m = end - start
        if m == 0:
            continue
        aligned_m = _align(m, alignment)
        chunk = a[start:end]
        if aligned_m != m:
            chunk = F.pad(chunk, (0, 0, 0, aligned_m - m))
        chunks.append(chunk.contiguous())
        m_indices.extend([expert_idx] * m)
        m_indices.extend([-1] * (aligned_m - m))
        ranges.append((start, end, padded_start, padded_start + m))
        padded_start += aligned_m

    if not chunks:
        return out

    a_padded = torch.cat(chunks, dim=0).contiguous()
    a_fp8 = per_token_cast_to_fp8(a_padded, use_ue8m0=False)

    b_values = torch.empty_like(b_nt, dtype=torch.float8_e4m3fn)
    b_scales = torch.empty(
        b_nt.shape[0],
        _ceil_div(b_nt.shape[1], 128),
        _ceil_div(b_nt.shape[2], 128),
        dtype=torch.float32,
        device=b_nt.device,
    )
    for expert_idx in range(b_nt.shape[0]):
        b_values[expert_idx], b_scales[expert_idx] = per_block_cast_to_fp8(
            b_nt[expert_idx].contiguous(),
            use_ue8m0=False,
        )

    padded_out = torch.empty(a_padded.shape[0], n, dtype=out.dtype, device=out.device)
    m_indices_tensor = torch.tensor(m_indices, dtype=torch.int32, device=a.device)
    b_fp8 = (b_values, b_scales)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a_fp8,
        b_fp8,
        padded_out,
        m_indices_tensor,
        disable_ue8m0_cast=True,
    )
    for start, end, padded_valid_start, padded_valid_end in ranges:
        out[start:end].copy_(padded_out[padded_valid_start:padded_valid_end])
    return out


def fp8_group_gemm_same_nk(*, backend: str = DEFAULT_FP8_GROUPED_BACKEND, **kwargs) -> Tensor:
    backend = validate_fp8_grouped_backend(backend)
    if backend == "triton_grouped":
        return fp8_triton_grouped_group_gemm_same_nk(**kwargs)
    if backend == "block_loop":
        return fp8_block_loop_group_gemm_same_nk(**kwargs)
    if backend == "deep_gemm":
        return fp8_deep_gemm_group_gemm_same_nk(**kwargs)
    if backend == "scalar_quack":
        return fp8_scalar_quack_group_gemm_same_nk(**kwargs)
    raise AssertionError("unreachable")


def fp8_block_loop_group_gemm_same_mn(
    *,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    cumsum_K: Tensor,
    max_K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    cu_seqlens_k: Tensor | None = None,
    block_size: int = _FP8_BLOCK_SIZE,
) -> None:
    """Grouped wgrad GEMM using actual FP8 operands.

    Quack's SM90 varlen-K FP8 path currently requires incompatible major
    layouts for grouped wgrad, so use per-expert block-FP8 GEMMs here.
    """

    del max_K, cu_seqlens_k
    if transpose_b:
        raise NotImplementedError("FP8 grouped same_mn does not support transpose_b=True")

    starts = torch.cat([torch.zeros(1, dtype=cumsum_K.dtype, device=cumsum_K.device), cumsum_K[:-1]])
    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum_K.tolist())):
        if end == start:
            c[expert_idx].zero_()
            continue
        lhs = a[start:end].T.contiguous() if transpose_a else a[:, start:end].contiguous()
        rhs_t = b[start:end].T.contiguous()
        c[expert_idx].copy_(_block_fp8_matmul(lhs, rhs_t, block_size=block_size).to(c.dtype))


def fp8_triton_grouped_group_gemm_same_mn(
    *,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    cumsum_K: Tensor,
    max_K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    cu_seqlens_k: Tensor | None = None,
    block_size: int = _FP8_BLOCK_SIZE,
) -> None:
    """Grouped same-MN wgrad GEMM with in-kernel block-FP8 quantization."""

    if transpose_b:
        raise NotImplementedError("FP8 grouped same_mn does not support transpose_b=True")
    if not transpose_a:
        fp8_block_loop_group_gemm_same_mn(
            a=a,
            b=b,
            c=c,
            cumsum_K=cumsum_K,
            max_K=0,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            block_size=block_size,
        )
        return
    if not a.is_cuda or not b.is_cuda or not c.is_cuda:
        raise RuntimeError("triton_grouped FP8 grouped same_mn backend requires CUDA tensors")
    if c.dim() != 3:
        raise RuntimeError(f"FP8 grouped same_mn output must be rank 3, got shape={tuple(c.shape)}")

    num_experts, m, n = c.shape
    if num_experts == 0:
        return
    max_expert_k = int(max_K)
    if max_expert_k == 0:
        c.zero_()
        return

    c.zero_()
    block_m = 32
    block_n = 64 if n <= 64 else 128
    grid = (
        triton.cdiv(m, block_m),
        triton.cdiv(n, block_n),
        num_experts,
    )
    cu_seqlens = (
        cu_seqlens_k
        if cu_seqlens_k is not None
        else torch.cat([torch.zeros(1, dtype=cumsum_K.dtype, device=cumsum_K.device), cumsum_K])
    )
    _grouped_block_fp8_same_mn_kernel[grid](
        a.contiguous(),
        b.contiguous(),
        c,
        cu_seqlens,
        M=m,
        N=n,
        K_BLOCKS=_ceil_div(max_expert_k, block_size),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_size,
    )


def fp8_group_gemm_same_mn(*, backend: str = DEFAULT_FP8_GROUPED_BACKEND, **kwargs) -> None:
    backend = validate_fp8_grouped_backend(backend)
    if backend == "triton_grouped":
        fp8_triton_grouped_group_gemm_same_mn(**kwargs)
        return
    fp8_block_loop_group_gemm_same_mn(**kwargs)


fp8_quack_group_gemm_same_nk = fp8_scalar_quack_group_gemm_same_nk
fp8_quack_group_gemm_same_mn = fp8_block_loop_group_gemm_same_mn
