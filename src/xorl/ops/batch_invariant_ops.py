# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py
#
# Vendored into xorl from SGLang's
# python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py so that the
# xorl trainer/server forward can use the SAME batch-invariant Triton matmul
# kernels SGLang uses. This makes the linear layers, gate matmul and lm_head
# bit-for-bit match SGLang's reduction order, collapsing the cross-engine K3
# logprob tail.
#
# Two SGLang-internal dependencies are stubbed/inlined here so this module is
# self-contained:
#   - ENABLE_JIT_DEEPGEMM -> forced False (pure-Triton matmul path, no DeepGEMM)
#   - get_bool_env_var / calc_diff -> inlined below

import contextlib
import os
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl


# --- Stubs for SGLang-internal imports ---------------------------------------
# Force the pure-Triton matmul path (no DeepGEMM). DeepGEMM is an alternate
# bf16 GEMM backend that is NOT the batch-invariant Triton kernel, so disabling
# it guarantees every mm/addmm goes through matmul_kernel_persistent.
ENABLE_JIT_DEEPGEMM = False


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("true", "1")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# -----------------------------------------------------------------------------


_ENABLE_MM_DEEPGEMM = get_bool_env_var("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1")
# If true, allows to fallback to batch variant gemm when the shape cannot be run in DeepGEMM
_ENABLE_MM_FALLBACK_VARIANT = get_bool_env_var("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", "0")
_ENABLE_MM_COMPARISON_TEST = get_bool_env_var("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST")

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "is_batch_invariant_op_enabled",
    "get_batch_invariant_ops",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "rms_norm_batch_invariant",
    "fused_add_rms_norm_batch_invariant",
    "sglang_rms_norm_batch_invariant",
    "fused_rms_norm_backward",
]


_BATCH_INVARIANT_ALL_OPS = {
    "mm",
    "addmm",
    "mm_dtype",
    "log_softmax",
    "mean",
    "rms_norm",
    "bmm",
}
_BATCH_INVARIANT_ALIASES = {
    "matmul": "mm",
    "logsoftmax": "log_softmax",
    "log-softmax": "log_softmax",
    "rmsnorm": "rms_norm",
    "rms-norm": "rms_norm",
}


def _parse_batch_invariant_ops() -> set[str]:
    raw = os.environ.get("XORL_BATCH_INVARIANT_OPS", "all").strip().lower()
    if raw in ("", "1", "true", "yes", "all"):
        return set(_BATCH_INVARIANT_ALL_OPS)
    if raw in ("0", "false", "no", "none"):
        return set()

    ops = set()
    for part in raw.replace(";", ",").split(","):
        op = part.strip().lower().replace("-", "_")
        if not op:
            continue
        op = _BATCH_INVARIANT_ALIASES.get(op, op)
        if op == "mm":
            ops.add("mm_dtype")
        if op not in _BATCH_INVARIANT_ALL_OPS:
            raise ValueError(
                f"Unsupported XORL_BATCH_INVARIANT_OPS entry {part!r}; "
                f"supported values are: {sorted(_BATCH_INVARIANT_ALL_OPS)}"
            )
        ops.add(op)
    return ops


def _matmul_launch_metadata(grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def _matmul_persistent_triton(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Currently assuming bias is 1D, let Horace know if you run into this"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    # print(a.device, b.device, c.device)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


def _matmul_persistent_deepgemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out = torch.empty((M, N), device=a.device, dtype=dtype)

    try:
        import deep_gemm  # noqa: PLC0415

        deep_gemm.bf16_gemm_nn(a, b, out)
    except RuntimeError:
        return None

    # TODO can this be put in DeepGEMM's `c`?
    if bias is not None:
        out += bias

    return out


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    K, N = b.shape

    # DeepGEMM has minimum dimension requirements for TMA descriptors
    MIN_DEEPGEMM_DIM = 16

    if (
        _ENABLE_MM_DEEPGEMM
        and ENABLE_JIT_DEEPGEMM
        and (a.dtype == torch.bfloat16)
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= MIN_DEEPGEMM_DIM
    ):
        if _ENABLE_MM_COMPARISON_TEST:
            out_triton = _matmul_persistent_triton(a=a, b=b, bias=bias)
            out_deepgemm = _matmul_persistent_deepgemm(a=a, b=b, bias=bias)
            if out_deepgemm is not None:
                diff = calc_diff(out_triton, out_deepgemm)
                assert diff < 0.0001, f"{diff=} {out_triton=} {out_deepgemm=}"
                return out_deepgemm
            # DeepGEMM failed, use Triton result
            return out_triton

        result = _matmul_persistent_deepgemm(a=a, b=b, bias=bias)
        if result is not None:
            return result
        # DeepGEMM failed (e.g. dimensions too small for TMA descriptors),
        # fall through to batch-invariant Triton persistent kernel

    if _ENABLE_MM_FALLBACK_VARIANT:
        out = torch.einsum("ik,kj->ij", a, b)
        if bias is not None:
            out += bias
        return out

    return _matmul_persistent_triton(a=a, b=b, bias=bias)


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    # Load first block to infer dtype and initialize max_val with correct type
    col_idx_init = tl.arange(0, BLOCK_SIZE)
    mask_init = col_idx_init < n_cols
    vals_init = tl.load(row_start_ptr + col_idx_init, mask=mask_init, other=-float("inf"))
    max_val = tl.max(vals_init)

    # Continue with remaining blocks
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    # Initialize sum_exp with correct dtype by using tl.sum on a zero vector
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)

    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert -input.ndim <= dim < input.ndim, f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems


@triton.jit
def bmm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    B,
    M,
    N,
    K,  #
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """
    Batched matrix multiplication kernel that processes batches in parallel.
    Each tile processes a (BLOCK_SIZE_M, BLOCK_SIZE_N) output block for a specific batch.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_batch = num_pid_m * num_pid_n
    num_tiles_total = B * num_tiles_per_batch

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Process tiles in a deterministic order: batch-major ordering
    for tile_id in tl.range(start_pid, num_tiles_total, NUM_SMS, flatten=True):
        # Decompose tile_id into batch and within-batch tile
        batch_idx = tile_id // num_tiles_per_batch
        tile_in_batch = tile_id % num_tiles_per_batch

        pid_m, pid_n = _compute_pid(tile_in_batch, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Add batch offset
        if A_LARGE or B_LARGE:
            batch_idx_typed = batch_idx.to(tl.int64)
        else:
            batch_idx_typed = batch_idx

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + (batch_idx_typed * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (batch_idx_typed * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + batch_idx_typed * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process batches in parallel with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        # Check constraints
        assert a.shape[0] == b.shape[0], "Batch sizes must match"
        assert a.shape[2] == b.shape[1], "Incompatible dimensions"
        assert a.dtype == b.dtype, "Incompatible dtypes"

        B = a.shape[0]
        M = a.shape[1]
        K = a.shape[2]
        N = b.shape[2]
        dtype = a.dtype

        # Allocate output
        if out is None:
            c = torch.empty((B, M, N), device=a.device, dtype=dtype)
        else:
            c = out

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        # Use fixed kernel configuration for determinism
        configs = {
            torch.bfloat16: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
            torch.float16: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
            torch.float32: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
        }

        config = configs.get(dtype)
        if config is None:
            raise ValueError(
                f"Unsupported dtype {dtype} for bmm_batch_invariant. Supported dtypes are: {list(configs.keys())}"
            )

        # Grid: limit by NUM_SMS for persistent kernel approach
        num_tiles_per_batch = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"])
        num_tiles_total = B * num_tiles_per_batch
        grid = (min(NUM_SMS, num_tiles_total),)

        bmm_kernel_persistent[grid](
            a,
            b,
            c,  #
            B,
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),
            a.stride(2),  #
            b.stride(0),
            b.stride(1),
            b.stride(2),  #
            c.stride(0),
            c.stride(1),
            c.stride(2),  #
            NUM_SMS=NUM_SMS,  #
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=c.numel() > 2**31,
            **config,
        )

        return c
    else:
        raise ValueError(f"bmm_batch_invariant expects 3D tensors, got shapes {a.shape} and {b.shape}")


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.

    Adapted from https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/vllm/model_executor/layers/batch_invariant.py#L649

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)


# --------------------------------------------------------------------------- #
# Fused batch-invariant "sglang" RMSNorm (residual + no-residual)
#
# Vendored from SGLang's fused batch-invariant residual RMSNorm
# (python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py, branch
# feature/k3-train-serve-numerics). These reproduce, bit-for-bit, the eager
# ``normalization.sglang_residual_rms_norm`` path (fp32 upcast, ``mean_dim``
# variance, ``rsqrt``, fp32 weight multiply, cast last) while replacing its
# ~6-launch eager forward with three kernels. Bit-exactness keys:
#   - reuse the batch-invariant ``mean_dim`` for the variance (a hand-rolled
#     ``tl.sum`` reduction does NOT match it),
#   - ``tl.rsqrt`` matches ``torch.rsqrt(var + eps)`` (``1.0 / tl.sqrt`` does not),
#   - residual add is upcast -> add -> round back to the input dtype (matches
#     torch's fp32-accumulated bf16 elementwise add).
# The forward is order-identical to the eager path, so the static K3 forward is
# preserved exactly. These are forward-only; the trainer wraps them in an
# ``autograd.Function`` with a closed-form backward (see normalization.py).
# --------------------------------------------------------------------------- #
@triton.jit
def _add_residual_square_kernel(
    input_ptr,
    residual_ptr,
    residual_out_ptr,
    sq_ptr,
    input_row_stride: tl.constexpr,
    residual_row_stride: tl.constexpr,
    residual_out_row_stride: tl.constexpr,
    sq_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1 of the fused residual-add RMSNorm: residual add in the original
    dtype and the per-element square in float32.

        residual_out = (x + residual).to(orig_dtype)
        sq           = residual_out.float() ** 2     # float32

    ``sq`` is then reduced by the existing batch-invariant ``mean_dim`` kernel,
    so the variance reduction order is bit-identical to the eager
    ``x.pow(2).mean(-1)`` path this replaces.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row_idx * input_row_stride
    res_row = residual_ptr + row_idx * residual_row_stride
    res_out_row = residual_out_ptr + row_idx * residual_out_row_stride
    sq_row = sq_ptr + row_idx * sq_row_stride
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(in_row + col_idx, mask=mask, other=0.0)
        r = tl.load(res_row + col_idx, mask=mask, other=0.0)
        # Match torch's elementwise add for low-precision dtypes: upcast to
        # float32, add, round the result back to the original dtype. The
        # normalization then operates on this rounded value.
        s = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
        tl.store(res_out_row + col_idx, s, mask=mask)
        s_f32 = s.to(tl.float32)
        tl.store(sq_row + col_idx, s_f32 * s_f32, mask=mask)


@triton.jit
def _square_kernel(
    input_ptr,
    sq_ptr,
    input_row_stride: tl.constexpr,
    sq_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """No-residual analog of stage 1: per-element square in float32.

        sq = x.float() ** 2

    Reduced by ``mean_dim`` for a variance bit-identical to the eager
    ``x.float().pow(2).mean(-1)`` path (fp32 ``s * s`` == ``pow(x, 2)``).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row_idx * input_row_stride
    sq_row = sq_ptr + row_idx * sq_row_stride
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(in_row + col_idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        tl.store(sq_row + col_idx, x_f32 * x_f32, mask=mask)


@triton.jit
def _rms_normalize_with_var_kernel(
    input_ptr,
    var_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 2 of the fused RMSNorm: normalize by a precomputed per-row variance
    and multiply weight in float32, casting last.

        out = (x.float() * rsqrt(var + eps) * weight.float()).to(orig_dtype)

    ``tl.rsqrt`` bit-matches ``torch.rsqrt(var + eps)`` used by the eager path
    (``1.0 / tl.sqrt`` does not).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row_idx * input_row_stride
    out_row = output_ptr + row_idx * output_row_stride
    var = tl.load(var_ptr + row_idx)
    inv_rms = tl.rsqrt(var + eps)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(in_row + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        output_f32 = x.to(tl.float32) * inv_rms * weight.to(tl.float32)
        tl.store(out_row + col_idx, output_f32.to(x.dtype), mask=mask)


def fused_add_rms_norm_batch_invariant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch-invariant fused residual-add + RMS normalization.

    Returns ``(output, residual_out)`` bit-matching the eager
    ``residual_out = input + residual`` followed by
    ``normalization.sglang_residual_rms_norm(residual_out, weight, eps)`` for the
    closed dense Qwen3 recipe (fp32 upcast, ``mean_dim`` variance, ``rsqrt``,
    fp32 weight multiply, cast last).

    The eager path is ~6 small launches per call; here it is three: a fused
    residual-add+square, the batch-invariant ``mean_dim`` reduction (reused
    verbatim so the variance is bit-identical to ``x.pow(2).mean(-1)``), and a
    fused normalize. Forward-only; wrap in an autograd.Function for training.
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape == residual.shape, "Input and residual must share a shape"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    residual_2d = residual.reshape(-1, residual.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    residual_out = torch.empty_like(input_2d)
    sq = torch.empty((n_rows, n_cols), dtype=torch.float32, device=input.device)

    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _add_residual_square_kernel[grid](
        input_2d,
        residual_2d,
        residual_out,
        sq,
        input_2d.stride(0),
        residual_2d.stride(0),
        residual_out.stride(0),
        sq.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reuse the batch-invariant mean reduction verbatim: variance is then
    # bit-identical to the eager path's x.pow(2).mean(-1).
    var = mean_dim(sq, -1, keepdim=True).reshape(-1).contiguous()

    output = torch.empty_like(input_2d)
    _rms_normalize_with_var_kernel[grid](
        residual_out,
        var,
        weight,
        output,
        residual_out.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape), residual_out.reshape(original_shape)


@triton.jit
def _rms_norm_backward_kernel(
    normed_ptr,
    grad_out_ptr,
    grad_ro_ptr,
    weight_ptr,
    grad_in_ptr,
    gw_partial_ptr,
    normed_row_stride: tl.constexpr,
    grad_out_row_stride: tl.constexpr,
    grad_ro_row_stride: tl.constexpr,
    grad_in_row_stride: tl.constexpr,
    gw_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_RESIDUAL_GRAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm backward for the ``sglang``-style norm.

    Given ``normed_input`` (the tensor that was normalized), ``grad_output`` and
    optionally ``grad_residual_out``, computes per row::

        var  = mean(rf^2);  inv = rsqrt(var + eps)
        S    = sum_j grad_out_j * w_j * rf_j
        grad_normed_k = inv * grad_out_k * w_k - (inv^3 * S / D) * rf_k
        gw_partial_k  = grad_out_k * rf_k * inv       # summed over rows -> grad_weight

    When ``grad_residual_out`` is present it is added to ``grad_normed`` (the
    residual stream feeds both the norm and the next layer). This is the
    gradient path only; it does not enter the forward K3, so ``var`` is
    recomputed with a plain reduction (not ``mean_dim``).
    """
    row = tl.program_id(0).to(tl.int64)
    n_row = normed_ptr + row * normed_row_stride
    go_row = grad_out_ptr + row * grad_out_row_stride
    gi_row = grad_in_ptr + row * grad_in_row_stride
    gw_row = gw_partial_ptr + row * gw_row_stride

    sum_sq = tl.zeros([1], dtype=tl.float32)
    s_acc = tl.zeros([1], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_cols
        rf = tl.load(n_row + idx, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(go_row + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, rf * rf, 0.0))
        s_acc += tl.sum(tl.where(mask, go * w * rf, 0.0))
    var = sum_sq / n_cols
    inv = tl.rsqrt(var + eps)
    c = inv * inv * inv * s_acc / n_cols

    for off in range(0, n_cols, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_cols
        rf = tl.load(n_row + idx, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(go_row + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        grad_normed = inv * go * w - c * rf
        if HAS_RESIDUAL_GRAD:
            gro = tl.load(grad_ro_ptr + row * grad_ro_row_stride + idx, mask=mask, other=0.0).to(tl.float32)
            grad_normed = grad_normed + gro
        tl.store(gi_row + idx, grad_normed, mask=mask)
        tl.store(gw_row + idx, go * rf * inv, mask=mask)


def fused_rms_norm_backward(
    normed_input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    grad_output: torch.Tensor,
    grad_residual_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused backward for :func:`fused_add_rms_norm_batch_invariant` /
    :func:`sglang_rms_norm_batch_invariant`.

    Returns ``(grad_normed_input_fp32, grad_weight_fp32)``. ``grad_normed_input``
    already includes ``grad_residual_out`` when supplied. Not order-sensitive:
    gradients do not enter the forward K3.
    """
    original_shape = normed_input.shape
    ni = normed_input.reshape(-1, original_shape[-1]).contiguous()
    go = grad_output.reshape(-1, original_shape[-1]).contiguous()
    wf = weight.float().contiguous()

    n_rows, n_cols = ni.shape
    grad_in = torch.empty((n_rows, n_cols), dtype=torch.float32, device=ni.device)
    gw_partial = torch.empty((n_rows, n_cols), dtype=torch.float32, device=ni.device)

    has_residual = grad_residual_out is not None
    if has_residual:
        gro = grad_residual_out.reshape(-1, original_shape[-1]).contiguous()
    else:
        gro = ni  # unused; kernel does not read it when HAS_RESIDUAL_GRAD=False

    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_backward_kernel[grid](
        ni,
        go,
        gro,
        wf,
        grad_in,
        gw_partial,
        ni.stride(0),
        go.stride(0),
        gro.stride(0),
        grad_in.stride(0),
        gw_partial.stride(0),
        n_cols,
        eps,
        HAS_RESIDUAL_GRAD=has_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    grad_weight = gw_partial.sum(0)
    return grad_in.reshape(original_shape), grad_weight


def sglang_rms_norm_batch_invariant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Batch-invariant RMS normalization bit-matching the eager
    ``normalization.sglang_residual_rms_norm`` (no residual add).

    This is the no-residual analog of :func:`fused_add_rms_norm_batch_invariant`
    for the ``force_sglang_residual`` call sites (input layernorm at layer>0 and
    the final norm), which apply the residual-style fp32 normalization to a
    single input. Two kernels + ``mean_dim`` replace the eager ~6 launches.
    Forward-only; wrap in an autograd.Function for training.
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    sq = torch.empty((n_rows, n_cols), dtype=torch.float32, device=input.device)

    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _square_kernel[grid](
        input_2d,
        sq,
        input_2d.stride(0),
        sq.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    var = mean_dim(sq, -1, keepdim=True).reshape(-1).contiguous()

    output = torch.empty_like(input_2d)
    _rms_normalize_with_var_kernel[grid](
        input_2d,
        var,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


_ONES_CACHE: Dict[Tuple[str, int | None, torch.dtype, int], torch.Tensor] = {}


def _get_or_make_ones(input: torch.Tensor, normalized_shape: list[int]) -> torch.Tensor:
    assert len(normalized_shape) == 1, "Only last-dimension RMSNorm is supported"
    hidden_size = int(normalized_shape[0])
    key = (input.device.type, input.device.index, input.dtype, hidden_size)
    weight = _ONES_CACHE.get(key)
    if weight is None or weight.device != input.device:
        weight = torch.ones(hidden_size, device=input.device, dtype=input.dtype)
        _ONES_CACHE[key] = weight
    return weight


def _rms_norm_aten_compat(input, normalized_shape, weight=None, eps=None):
    normalized_shape = [int(dim) for dim in normalized_shape]
    if len(normalized_shape) != 1 or input.shape[-1] != normalized_shape[0]:
        raise NotImplementedError("Batch-invariant RMSNorm only supports last dimension")
    if weight is None:
        weight = _get_or_make_ones(input, normalized_shape)
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    return rms_norm_batch_invariant(input, weight, eps=eps)


def _mm_dtype_compat(a, b, out_dtype):
    out = mm_batch_invariant(a, b)
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_batch_invariant_OPS: set[str] = set()
_original_torch_bmm = None


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def get_batch_invariant_ops() -> tuple[str, ...]:
    return tuple(sorted(_batch_invariant_OPS)) if _batch_invariant_MODE else ()


def is_batch_invariant_op_enabled(op: str) -> bool:
    op = _BATCH_INVARIANT_ALIASES.get(op, op)
    return _batch_invariant_MODE and op in _batch_invariant_OPS


def enable_batch_invariant_mode(
    enable_bmm: bool = True,
):
    global _batch_invariant_MODE, _batch_invariant_LIB, _batch_invariant_OPS, _original_torch_bmm
    if _batch_invariant_MODE:
        return

    _batch_invariant_OPS = _parse_batch_invariant_ops()
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    if "mm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
    if "addmm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    if "log_softmax" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, "CUDA")
    if "mean" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")
    if "rms_norm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::rms_norm", _rms_norm_aten_compat, "CUDA")
    if "mm_dtype" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::mm.dtype", _mm_dtype_compat, "CUDA")

    if enable_bmm and "bmm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "CUDA")

        # Also monkeypatch torch.bmm directly as a fallback
        _original_torch_bmm = torch.bmm
        torch.bmm = bmm_batch_invariant


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _batch_invariant_OPS, _original_torch_bmm
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    if _original_torch_bmm is not None:
        torch.bmm = _original_torch_bmm
        _original_torch_bmm = None
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    _batch_invariant_OPS = set()


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    was_enabled = _batch_invariant_MODE
    if enabled == was_enabled:
        yield
        return

    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    try:
        yield
    finally:
        if was_enabled:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
