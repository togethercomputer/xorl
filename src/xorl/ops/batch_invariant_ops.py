# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py
#
# Vendored into xorl from SGLang's
# python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py so that the
# xorl trainer/server forward can use the SAME batch-invariant Triton matmul
# kernels SGLang uses. This makes the linear layers, gate matmul and lm_head
# bit-for-bit match SGLang's reduction order, collapsing the cross-engine K3
# logprob tail.
#
# SGLang-internal helpers are stubbed/inlined so this module is self-contained.
# DeepGEMM is discovered lazily and used only on a supported Hopper bf16 route;
# otherwise the vendored Triton path remains the fallback. Environment parsing
# and calc_diff are inlined below.

import contextlib
import os
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from xorl.ops.bi_gemm_configs import baseline_mm_config, lookup_mm_config


# --- Stubs for SGLang-internal imports ---------------------------------------
# DeepGEMM's bf16 NN GEMM is bitwise-identical to the batch-invariant Triton
# persistent kernel (verified per-shape by tune_bi_gemm.py and gated by
# tests/ops/test_bi_gemm_config_table.py). Serving's mm route already relies on
# that equality — SGLang sends every contiguous bf16 N>=16 mm to DeepGEMM while
# the trainer ran pure Triton — so routing the trainer the same way cannot move
# cross-engine bits, and it is markedly faster at trainer token counts.
# Requires Hopper+ and an importable deep_gemm; kill switch:
# XORL_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 (SGLANG_ prefix honored too).
# Detection is lazy (first mm) so importing this module never initializes CUDA.
_DEEPGEMM_READY: bool | None = None


def _deepgemm_ready() -> bool:
    global _DEEPGEMM_READY
    if _DEEPGEMM_READY is None:
        try:
            import deep_gemm  # noqa: F401, PLC0415

            _DEEPGEMM_READY = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9
                and hasattr(deep_gemm, "bf16_gemm_nn")
            )
        except Exception:  # noqa: BLE001
            _DEEPGEMM_READY = False
    return _DEEPGEMM_READY


ENABLE_JIT_DEEPGEMM = True  # gated per-call by _deepgemm_ready()

# Shapes whose table config exceeded shared memory at launch (Triton's epilogue
# staging for wide-output tiles is version-dependent): remembered so the hot
# path re-launches straight on the pinned baseline without re-raising.
_MM_CONFIG_OOM_SHAPES: set[tuple] = set()


def _launch_with_config_fallback(launch, dtype, M, N, K, out_itemsize=None):
    key = (str(dtype), M, N, K, out_itemsize)
    if key in _MM_CONFIG_OOM_SHAPES:
        launch(baseline_mm_config(dtype))
        return
    try:
        launch(lookup_mm_config(dtype, M, N, K, out_itemsize=out_itemsize))
    except OutOfResources:
        _MM_CONFIG_OOM_SHAPES.add(key)
        launch(baseline_mm_config(dtype))


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("true", "1")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# -----------------------------------------------------------------------------


_ENABLE_MM_DEEPGEMM = get_bool_env_var(
    "XORL_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM",
    os.getenv("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1"),
)
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
    "wrap_trunk_linears_batch_invariant",
    "is_trunk_linear_contract_enabled",
    "set_trunk_linear_contract",
    "RMSNormFamily",
    "RMS_NORM_FAMILY_NO_RESIDUAL",
    "RMS_NORM_FAMILY_RESIDUAL_TREE",
    "RMS_NORM_FAMILIES",
    "bi_rms_norm",
    "bi_fused_add_rms_norm",
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

    # Shape-keyed on the bit-neutral axes only; BLOCK_SIZE_K stays pinned per
    # dtype (bi_gemm_configs — the R1 config table, identical in both engines).
    def _launch(config):
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
            **config,
        )

    _launch_with_config_fallback(_launch, dtype, M, N, K)
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
        and (a.dtype == torch.bfloat16)
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= MIN_DEEPGEMM_DIM
        and _deepgemm_ready()
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


_INTERPOSE_GRAD_ERROR = (
    "Batch-invariant interposed op '{op}' received a grad-requiring input while grad is enabled. "
    "The global interpose (XORL_BATCH_INVARIANT_MATMUL / enable_batch_invariant_mode) is "
    "inference/verification-only: the aten::rms_norm override records no autograd graph (q/k-norm "
    "gradients silently vanish) and the torch.bmm monkeypatch detaches the graph. For a training "
    "forward on the batch-invariant contract use the module-scoped XORL_BI_TRUNK_LINEAR=1 lane "
    "instead."
)


def _guard_interpose_no_grad(op: str, *tensors) -> None:
    """Loud-fail: the global interpose must never see a training forward."""
    if torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors):
        raise RuntimeError(_INTERPOSE_GRAD_ERROR.format(op=op))


def mm_batch_invariant(a, b):
    _guard_interpose_no_grad("aten::mm", a, b)
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    _guard_interpose_no_grad("aten::addmm", bias, a, b)
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    _guard_interpose_no_grad("aten::_log_softmax", input)
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    _guard_interpose_no_grad("aten::mean.dim", input)
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if dim is None or len(dim) == 0:
        # aten::mean full-reduce dispatches here with dim=[]; the empty n_elems product returned the SUM
        dim = list(range(input.ndim))
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
    _guard_interpose_no_grad("aten::bmm/torch.bmm", a, b)
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


# --------------------------------------------------------------------------- #
# RMSNorm kernel-family contract
#
# Two batch-invariant RMSNorm kernel families coexist, and they disagree at
# 1 ulp on rare bf16 boundary values (~2/524288 at [4096, 128]), so silently
# swapping one for the other seeds K3 divergence that amplifies downstream
# (the 2026-07-04 norm-seed incident: 2.99e-5 K3 from five such seeds).
# Each family is pinned to the serving site-class that executes it:
#   - "serving_no_residual" (family-1): the looped ``tl.sum`` + ``1.0/tl.sqrt``
#     kernel (``rms_norm_batch_invariant``), what SGLang dispatches when
#     ``residual is None`` under batch-invariant mode and what the
#     ``aten::rms_norm`` interpose runs. Site-classes: qk-norm, layer-0 input
#     layernorm.
#   - "serving_residual_tree" (family-2): the ``mean_dim`` + ``tl.rsqrt``
#     fused residual-tree kernels (``fused_add_rms_norm_batch_invariant`` /
#     ``sglang_rms_norm_batch_invariant``), what SGLang dispatches for
#     residual calls under the rl-on-policy lane. Site-classes: input
#     layernorm at layer>0, post-attention layernorm, final norm.
# Every call site must name its family through ``bi_rms_norm`` /
# ``bi_fused_add_rms_norm``; never call the family kernels directly.
# --------------------------------------------------------------------------- #
RMS_NORM_FAMILY_NO_RESIDUAL = "serving_no_residual"
RMS_NORM_FAMILY_RESIDUAL_TREE = "serving_residual_tree"
RMS_NORM_FAMILIES = (RMS_NORM_FAMILY_NO_RESIDUAL, RMS_NORM_FAMILY_RESIDUAL_TREE)
RMSNormFamily = Literal["serving_no_residual", "serving_residual_tree"]


def bi_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    family: RMSNormFamily,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Single-tensor batch-invariant RMSNorm with an explicit kernel family.

    The single sanctioned entry point for both no-residual family kernels; the
    ``family`` keyword must name the serving counterpart of the call site.

    ``zero_centered`` is the Gemma-style Qwen3.5 form: an fp32 upcast with the
    ``1 + weight`` scale folded in fp32, cast back last, exactly as
    ``normalization.native_zero_centered_rms_norm`` composes it. It is an affine
    fold around the SAME family-1 reduction tree — not a third family — and only
    exists in no-residual form (Qwen3.5 residual-tree norms run the eager native
    path, never a batch-invariant kernel).
    """
    if zero_centered:
        if family != RMS_NORM_FAMILY_NO_RESIDUAL:
            raise ValueError(
                "zero-centered RMSNorm only exists in the 'serving_no_residual' family; "
                "Qwen3.5 residual-tree norms run the native path, not a batch-invariant kernel"
            )
        return rms_norm_batch_invariant(input.float(), 1.0 + weight.float(), eps=eps).type_as(input)
    if family == RMS_NORM_FAMILY_NO_RESIDUAL:
        return rms_norm_batch_invariant(input, weight, eps=eps)
    if family == RMS_NORM_FAMILY_RESIDUAL_TREE:
        return sglang_rms_norm_batch_invariant(input, weight, eps=eps)
    raise ValueError(f"Unknown RMSNorm family {family!r}; expected one of {RMS_NORM_FAMILIES}")


def bi_fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    family: RMSNormFamily,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add batch-invariant RMSNorm with an explicit kernel family.

    Only the serving residual tree has a fused-add kernel; requesting the
    no-residual family with a residual stream is a contract violation (serving
    never runs family-1 on a residual site) and raises.
    """
    if family == RMS_NORM_FAMILY_RESIDUAL_TREE:
        return fused_add_rms_norm_batch_invariant(input, residual, weight, eps=eps)
    if family == RMS_NORM_FAMILY_NO_RESIDUAL:
        raise ValueError(
            "RMSNorm family 'serving_no_residual' has no fused-add kernel: residual "
            "site-classes are 'serving_residual_tree' by the cross-engine contract"
        )
    raise ValueError(f"Unknown RMSNorm family {family!r}; expected one of {RMS_NORM_FAMILIES}")


# --------------------------------------------------------------------------- #
# Batch-invariant fused LM-head selected-token log-probability
#
# The K3 lm-head contract, vendored identically in xorl and SGLang
# (python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py). Both engines
# compute per-token logprobs of given token ids from bit-exact bf16 hidden
# states and the bf16 lm-head weight through the SAME reduction trees, so the
# results are bitwise identical cross-engine:
#   1. chunk GEMM — ``matmul_kernel_persistent`` with the family's fixed bf16
#      tile config and an fp32 output buffer. bf16xbf16 products are exact in
#      fp32, so reading the weight in bf16 with tensor-core fp32 accumulation
#      equals a GEMM over materialized fp32 upcasts with the same tree (and
#      deletes the fp32 weight copy the eager paths materialize).
#   2. chunk stats — per-row max and sum(exp(x - chunk_max)) in a fixed
#      sequential BLOCK loop (same discipline as ``_rms_norm_kernel``), plus
#      the selected-token logit gather.
#   3. merge — global max over chunk maxima (exact), then the rescaled sumexp
#      accumulated in pinned chunk order; lse = gmax + log(acc).
# All transcendentals stay inside these kernels: tl.exp/tl.log measured
# bit-identical across triton 3.5.1 (serving venv) and 3.7.1 (trainer venv),
# as is the fixed-tile tl.dot fp32 accumulator. VOCAB_CHUNK and STATS_BLOCK are
# contract constants — changing either changes the bits (the LSE reduction
# tree). The chunk GEMM's tile config is shape-keyed via bi_gemm_configs: only
# its BLOCK_SIZE_K (pinned there) is bit-relevant.
# Forward-only; the trainer wraps it in an autograd.Function (ops/loss).
# --------------------------------------------------------------------------- #

BI_LM_HEAD_VOCAB_CHUNK = 8192
_BI_LM_HEAD_STATS_BLOCK = 1024


@triton.jit
def _lm_head_chunk_stats_kernel(
    logits_ptr,
    token_ids_ptr,
    sel_ptr,
    m_ptr,
    s_ptr,
    temp_ptr,
    logits_row_stride,
    n_cols,
    col_offset,
    chunk_idx,
    n_chunks,
    BLOCK_SIZE: tl.constexpr,
    HAS_TEMP: tl.constexpr,
):
    """Per-row chunk statistics over an fp32 logits tile [N, n_cols]:
    chunk max, sum(exp(x - chunk_max)) in a fixed sequential block loop, and
    the selected-token logit when ``token_ids[row]`` falls in this chunk.
    With HAS_TEMP, logits are scaled by 1/temp[row] before the statistics and
    the selected logit; the fp32 divide runs in-kernel so every engine
    computes the identical scale (elementwise, so batch-invariance holds)."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_row_stride
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + row)
    else:
        inv_t = 1.0

    row_max = float("-inf")
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=float("-inf"))
        if HAS_TEMP:
            vals = vals * inv_t
        row_max = tl.maximum(row_max, tl.max(vals))

    sum_exp = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=float("-inf"))
        if HAS_TEMP:
            vals = vals * inv_t
        e = tl.exp(vals - row_max)
        sum_exp += tl.sum(tl.where(mask, e, 0.0))

    tl.store(m_ptr + row * n_chunks + chunk_idx, row_max)
    tl.store(s_ptr + row * n_chunks + chunk_idx, sum_exp)

    tok = tl.load(token_ids_ptr + row)
    local = tok - col_offset
    in_chunk = (local >= 0) & (local < n_cols)
    sel = tl.load(row_ptr + local, mask=in_chunk, other=0.0)
    if HAS_TEMP:
        sel = sel * inv_t
    tl.store(sel_ptr + row, sel, mask=in_chunk)


@triton.jit
def _lm_head_lse_merge_kernel(
    m_ptr,
    s_ptr,
    lse_ptr,
    n_chunks,
):
    """lse[row] = gmax + log(sum_c s_c * exp(m_c - gmax)), chunks in pinned order."""
    row = tl.program_id(0).to(tl.int64)
    base = row * n_chunks
    gmax = float("-inf")
    for c in range(n_chunks):
        gmax = tl.maximum(gmax, tl.load(m_ptr + base + c))
    acc = 0.0
    for c in range(n_chunks):
        acc += tl.load(s_ptr + base + c) * tl.exp(tl.load(m_ptr + base + c) - gmax)
    tl.store(lse_ptr + row, gmax + tl.log(acc))


def _bi_lm_head_chunk_gemm_fp32(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the family's persistent matmul with the shape-keyed bf16 config and
    an fp32 output buffer (the fp32 store path keeps the raw accumulator bits)."""
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    M, K = a.shape
    _, N = b.shape

    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)

    def _launch(config):
        matmul_kernel_persistent[grid](
            a,
            b,
            out,
            None,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_SMS=NUM_SMS,
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=out.numel() > 2**31,
            HAS_BIAS=False,
            **config,
        )

    _launch_with_config_fallback(_launch, a.dtype, M, N, K, out_itemsize=out.element_size())


def bi_lm_head_selected_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token ``log p(token_ids)`` for the LM head, batch-invariant and
    cross-engine bit-exact (the K3 lm-head contract).

    Args:
        hidden: ``[N, H]`` bf16 hidden states (pre lm-head).
        weight: ``[V, H]`` bf16 lm-head weight (kept resident in bf16; no fp32
            copy is materialized).
        token_ids: ``[N]`` integer token ids to score (callers must pre-clamp
            ignored positions to a valid id and mask outputs downstream).
        temperature: optional ``[N]`` fp32 per-row temperatures (> 0). Logits
            are scaled by ``1/temperature[row]`` inside the stats kernel (the
            divide runs in-kernel, so engines sharing the contract compute the
            identical scale). ``None`` is the exact temperature-1.0 path.

    Returns:
        ``(logprob, lse, selected)`` — all ``[N]`` fp32; ``logprob = selected - lse``
        (temperature-scaled when ``temperature`` is given).
    """
    assert hidden.ndim == 2 and weight.ndim == 2, "hidden and weight must be 2D"
    assert hidden.shape[1] == weight.shape[1], "hidden dim mismatch"
    assert hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
        "the lm-head contract takes bf16 hidden/weight (fp32 upcast is exact inside the GEMM)"
    )
    assert hidden.is_cuda, "CUDA only"

    hidden = hidden.contiguous()
    token_ids = token_ids.contiguous().to(device=hidden.device, dtype=torch.int64)
    n_tokens = hidden.shape[0]
    vocab = weight.shape[0]
    n_chunks = (vocab + vocab_chunk - 1) // vocab_chunk
    if temperature is not None:
        temperature = temperature.reshape(-1).to(device=hidden.device, dtype=torch.float32).contiguous()
        assert temperature.shape[0] == n_tokens, "temperature must be per-row [N]"
        assert bool((temperature > 0).all()), "temperature must be > 0"

    chunk_max = torch.empty((n_tokens, n_chunks), dtype=torch.float32, device=hidden.device)
    chunk_sumexp = torch.empty_like(chunk_max)
    selected = torch.zeros(n_tokens, dtype=torch.float32, device=hidden.device)
    lse = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
    logits_buf = torch.empty((n_tokens, vocab_chunk), dtype=torch.float32, device=hidden.device)

    for chunk_idx, col_start in enumerate(range(0, vocab, vocab_chunk)):
        col_end = min(col_start + vocab_chunk, vocab)
        n_cols = col_end - col_start
        logits_c = logits_buf[:, :n_cols]
        # [H, C] transposed view of the resident bf16 weight — the persistent
        # kernel takes explicit strides, so no copy is made.
        _bi_lm_head_chunk_gemm_fp32(hidden, weight[col_start:col_end].t(), logits_c)
        _lm_head_chunk_stats_kernel[(n_tokens,)](
            logits_c,
            token_ids,
            selected,
            chunk_max,
            chunk_sumexp,
            temperature,
            logits_c.stride(0),
            n_cols,
            col_start,
            chunk_idx,
            n_chunks,
            BLOCK_SIZE=_BI_LM_HEAD_STATS_BLOCK,
            HAS_TEMP=temperature is not None,
        )

    _lm_head_lse_merge_kernel[(n_tokens,)](chunk_max, chunk_sumexp, lse, n_chunks)
    # In exact math the selected logit never exceeds the LSE; clamp the one-ulp
    # fp boundary case (p~1 tokens) so contract logprobs are provably <= 0.
    return torch.clamp_max(selected - lse, 0.0), lse, selected


# --------------------------------------------------------------------------- #
# Batch-invariant MoE router GEMM (the K3 router contract)
#
# Vendored identically in xorl and SGLang so the MoE gate/router logits are
# computed through ONE reduction tree cross-engine. Unlike the capture/replay
# lane, live training routes independently of serving (no routing replay), so a
# ~1e-10..1e-4 router-logit reduction-order diff between the two engines' GEMMs
# can flip the top-k expert selection on razor-edge tokens and cause large,
# rare-token logprob divergence. This kernel removes that last term:
#   - bf16 hidden [N, H] @ bf16 gate weight [E, H]^T -> fp32 logits [N, E]
#   - ``matmul_kernel_persistent`` with a pinned tile config and an fp32 output
#     buffer (same discipline as the lm-head contract). bf16xbf16 products are
#     exact in fp32, so reading both operands in bf16 with tensor-core fp32
#     accumulation equals an fp32 GEMM over their (exact) fp32 upcasts, but with
#     the reduction order pinned identically in both engines — and without the
#     fp32 weight/activation copies the eager fp32-router paths materialize.
# num_experts is small (a single BLOCK_SIZE_N tile for the common E <= 128), so
# the whole GEMM is one persistent launch. The config below is part of the
# contract; changing any constant changes the bits.
# Enable via XORL_MOE_BI_ROUTER=1 (xorl) / SGLANG_BI_ROUTER=1 (sglang).
# Forward-only; the trainer wraps it in an autograd.Function with a closed-form
# (order-insensitive) backward — gradients do not enter the forward K3.
# --------------------------------------------------------------------------- #

_BI_ROUTER_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_stages": 3,
    "num_warps": 8,
}


def bi_router_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 MoE router logits ``[N, E]`` from bf16 hidden ``[N, H]`` and bf16 gate
    weight ``[E, H]`` (the K3 router contract).

    One persistent bf16-in / fp32-accumulate / fp32-out GEMM with a pinned launch
    config, vendored identically in xorl and SGLang so the router logits (and
    therefore the top-k expert selection) are bitwise identical cross-engine.
    bf16xbf16 products are exact in fp32, so this equals an fp32 GEMM over the
    upcast operands the eager fp32-router paths materialize — minus the fp32
    weight/activation copies and with a reduction order that no longer depends on
    the backend GEMM.

    Args:
        hidden: ``[N, H]`` bf16 hidden states (pre-gate).
        weight: ``[E, H]`` bf16 gate weight (kept resident in bf16; no fp32 copy).

    Returns:
        ``[N, E]`` fp32 router logits.
    """
    assert hidden.ndim == 2 and weight.ndim == 2, "hidden and weight must be 2D"
    assert hidden.shape[1] == weight.shape[1], "hidden dim mismatch"
    assert hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
        "the router contract takes bf16 hidden/weight (fp32 upcast is exact inside the GEMM)"
    )
    assert hidden.is_cuda, "CUDA only"

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    n_tokens = hidden.shape[0]
    num_experts = weight.shape[0]
    logits = torch.empty((n_tokens, num_experts), dtype=torch.float32, device=hidden.device)
    if n_tokens == 0:
        return logits

    # [H, E] transposed view of the resident bf16 gate weight — the persistent
    # kernel takes explicit strides, so no copy is made.
    b = weight.t()
    NUM_SMS = torch.cuda.get_device_properties(hidden.device).multi_processor_count
    M, K = hidden.shape
    N = num_experts

    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)

    matmul_kernel_persistent[grid](
        hidden,
        b,
        logits,
        None,
        M,
        N,
        K,
        hidden.stride(0),
        hidden.stride(1),
        b.stride(0),
        b.stride(1),
        logits.stride(0),
        logits.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=hidden.numel() > 2**31,
        B_LARGE=weight.numel() > 2**31,
        C_LARGE=logits.numel() > 2**31,
        HAS_BIAS=False,
        **_BI_ROUTER_GEMM_CONFIG,
    )
    return logits


def bi_router_topk_weights(
    topk_vals: torch.Tensor,
    norm_topk_prob: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Renormalize the gathered top-k router scores in a fixed reduction order,
    then cast — the second half of the K3 router contract.

    ``bi_router_gemm`` makes the router logits (and therefore ``torch.topk``
    selection and the gathered top-k softmax scores) bitwise identical
    cross-engine, but the stock renorm ``vals / vals.sum(dim=-1, keepdim=True)``
    is not: the small last-dim reduction ``sum(dim=-1)`` uses a build-dependent
    tree order, so the divisor can differ by ~1 fp32 ulp between the trainer's
    and the server's torch/triton, occasionally flipping the final bf16 weight
    on rare tokens. Elementwise ops (add, divide, round-to-bf16) are IEEE
    correctly-rounded and build-invariant, so accumulating the divisor with a
    pinned left-to-right order makes the top-k weights bit-identical too.

    Args:
        topk_vals: ``[N, top_k]`` fp32 gathered top-k router scores (softmax
            probabilities on the softmax path).
        norm_topk_prob: renormalize the top-k slice to sum to 1 (Qwen3 MoE
            default). When False the scores are only cast (already bit-identical
            cross-engine, since the softmax/top-k that produced them are).
        out_dtype: routing-weight dtype (the model activation dtype, bf16).

    Returns:
        ``[N, top_k]`` ``out_dtype`` routing weights.
    """
    assert topk_vals.dtype == torch.float32, "the router contract renorms fp32 top-k scores"
    if norm_topk_prob:
        denom = topk_vals[..., 0]
        for k in range(1, topk_vals.shape[-1]):
            denom = denom + topk_vals[..., k]
        topk_vals = topk_vals / denom.unsqueeze(-1)
    return topk_vals.to(out_dtype)


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
    _guard_interpose_no_grad("aten::rms_norm", input, weight)
    normalized_shape = [int(dim) for dim in normalized_shape]
    if len(normalized_shape) != 1 or input.shape[-1] != normalized_shape[0]:
        raise NotImplementedError("Batch-invariant RMSNorm only supports last dimension")
    if weight is None:
        weight = _get_or_make_ones(input, normalized_shape)
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    # The interpose IS the no-residual family: every F.rms_norm that reaches it
    # is a no-residual site (qk-norm, layer-0 input norm) by the family contract.
    return bi_rms_norm(input, weight, eps=eps, family=RMS_NORM_FAMILY_NO_RESIDUAL)


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


_TRUNK_LINEAR_NAMES = (
    "qkv_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_up_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen2/3.5-MoE shared-expert sigmoid gate: serving contracts it through the
    # global interpose (plain nn.Linear -> aten::mm), so the trunk lane must too.
    "shared_expert_gate",
)

_TRUNK_LINEAR_CONTRACT_ACTIVE = False


def is_trunk_linear_contract_enabled() -> bool:
    """True once :func:`wrap_trunk_linears_batch_invariant` armed the contract lane.

    RMSNorm dispatch keys off this to route no-residual (family-1) norms through
    the serving batch-invariant kernel (the qk-norm term of the K3 contract).
    """
    return _TRUNK_LINEAR_CONTRACT_ACTIVE


def set_trunk_linear_contract(enabled: bool) -> None:
    global _TRUNK_LINEAR_CONTRACT_ACTIVE
    _TRUNK_LINEAR_CONTRACT_ACTIVE = enabled


class _BatchInvariantTrunkLinearFn(torch.autograd.Function):
    """Forward through the batch-invariant persistent GEMM; backward stays cuBLAS."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"XORL_BI_TRUNK_LINEAR contract is bf16-only; got input={input.dtype}, weight={weight.dtype}."
            )
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        x2d = input.reshape(-1, input.shape[-1])
        out = matmul_persistent(x2d, weight.t(), bias=bias)
        return out.reshape(*input.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        go2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = (go2d @ weight).reshape_as(input)
        if ctx.needs_input_grad[1]:
            grad_weight = go2d.t() @ input.reshape(-1, input.shape[-1])
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = go2d.sum(dim=0)
        return grad_input, grad_weight, grad_bias


def wrap_trunk_linears_batch_invariant(
    model: torch.nn.Module, names: Tuple[str, ...] = _TRUNK_LINEAR_NAMES
) -> Dict[str, int]:
    """Route ONLY transformer-trunk nn.Linear forwards through the batch-invariant GEMM.

    Module-scoped alternative to enable_batch_invariant_mode(): forward bits match the
    serving-side matmul_persistent contract while backward/optimizer/loss stay on cuBLAS.

    Selection is explicit: a module is wrapped iff its leaf name is in ``names`` AND it is
    exactly ``torch.nn.Linear``. lm_head/embeddings never match the name set; routed MoE
    experts (FQN containing ``.experts.``) are skipped — they are contracted through the
    fused sglang expert path. LoRA/QLoRA-wrapped modules and FP8/TE/custom Linear
    subclasses RAISE: silently skipping them would void the bitwise contract the flag
    promises. Idempotent (already-wrapped modules are left alone).

    Returns ``{leaf_name: wrapped_count}`` and arms the contract lane
    (see :func:`is_trunk_linear_contract_enabled`).
    """
    import types  # noqa: PLC0415

    from xorl.lora.fold import lora_merged_forward_enabled  # noqa: PLC0415
    from xorl.lora.modules.base import LoraModule  # noqa: PLC0415
    from xorl.lora.modules.linear import LoraLinear  # noqa: PLC0415

    if is_batch_invariant_mode_enabled():
        raise RuntimeError(
            "XORL_BI_TRUNK_LINEAR cannot be combined with the global batch-invariant interpose "
            "(XORL_BATCH_INVARIANT_MATMUL): the wrapped backward would silently ride the interposed "
            "aten::mm instead of cuBLAS. Pick one lane."
        )

    def _forward(self, input):
        return _BatchInvariantTrunkLinearFn.apply(input, self.weight, self.bias)

    def _forward_lora_merged(self, input):
        # Merged-forward LoRA lane: the BI GEMM runs on the canonically folded
        # weight (identical bits to a serving engine that received W'); backward
        # flows dW' through the straight-through fold into the LoRA factors.
        return _BatchInvariantTrunkLinearFn.apply(input, self.merged_weight_for_forward(), self.bias)

    wrapped: Dict[str, int] = {}
    already_wrapped = 0
    for module_name, module in model.named_modules():
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in names:
            continue
        if ".experts." in f".{module_name}.":
            continue
        if type(module) is LoraLinear and lora_merged_forward_enabled():
            # Merged-forward contract lane: the adapted linear serves and trains
            # through the folded weight, so the trunk contract composes.
            if module.weight.dtype not in (torch.bfloat16, torch.float32):
                raise RuntimeError(
                    f"XORL_BI_TRUNK_LINEAR: {module_name} weight is {module.weight.dtype}; the trunk "
                    "contract is bf16-only."
                )
            if getattr(module, "_xorl_bi_trunk_wrapped", False):
                already_wrapped += 1
                continue
            module.forward = types.MethodType(_forward_lora_merged, module)
            module._xorl_bi_trunk_wrapped = True
            wrapped[leaf] = wrapped.get(leaf, 0) + 1
            continue
        if isinstance(module, LoraModule):
            raise NotImplementedError(
                f"XORL_BI_TRUNK_LINEAR: {module_name} is adapter-wrapped ({type(module).__qualname__}); "
                "the trunk contract composes with LoRA only under XORL_LORA_MERGED_FORWARD=1 "
                "(plain LoraLinear) — set the merged-forward flag or exclude the adapter."
            )
        if type(module) is not torch.nn.Linear:
            raise NotImplementedError(
                f"XORL_BI_TRUNK_LINEAR: {module_name} is {type(module).__qualname__}, not a plain "
                "nn.Linear; fp8/te/custom linears are outside the bf16 trunk contract."
            )
        if module.weight.dtype not in (torch.bfloat16, torch.float32):
            # fp32 is allowed here only as a mixed-precision master: FSDP2's mp_policy
            # casts it to bf16 before forward, and the runtime guard in
            # _BatchInvariantTrunkLinearFn enforces bf16 on the actual GEMM operands.
            raise RuntimeError(
                f"XORL_BI_TRUNK_LINEAR: {module_name} weight is {module.weight.dtype}; the trunk contract is bf16-only."
            )
        if getattr(module, "_xorl_bi_trunk_wrapped", False):
            already_wrapped += 1
            continue
        module.forward = types.MethodType(_forward, module)
        module._xorl_bi_trunk_wrapped = True
        wrapped[leaf] = wrapped.get(leaf, 0) + 1

    if not wrapped and not already_wrapped:
        raise RuntimeError(
            "XORL_BI_TRUNK_LINEAR=1 matched no trunk linears; expected leaf names "
            f"{sorted(names)} — wire the model's projections or drop the flag."
        )
    set_trunk_linear_contract(True)
    return wrapped


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
