import os

import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import get_parallel_state
from xorl.ops.group_gemm.kernel.moe import expert_histogram, moe_gather, moe_index_compute, moe_scatter
from xorl.ops.group_gemm.kernel.quack import (
    cumsum_to_cu_seqlens,
    quack_group_gemm_gated_same_nk,
    quack_group_gemm_same_mn,
    quack_group_gemm_same_nk,
)
from xorl.ops.moe.activations import CLAMPED_SWIGLU_LIMIT, check_hidden_act_supported
from xorl.ops.moe.triton import (
    _apply_swiglu_clamp_backward,
    _maybe_clamp_swiglu_gate,
    _moe_gate_activation,
    _moe_gate_activation_backward,
)


def _debug_ep_enabled() -> bool:
    v = os.environ.get("XORL_DEBUG_EP", "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


_DEBUG_EP = _debug_ep_enabled()


def _fp8_grouped_backend(backend: str | None = None) -> str:
    return os.environ.get("XORL_FP8_MOE_GROUPED_BACKEND", backend or "triton_grouped").strip()


def _group_gemm_same_nk(
    *,
    fp8_compute: bool = False,
    fp8_grouped_backend: str | None = None,
    fp8_block_size: int = 128,
    **kwargs,
):
    if fp8_compute:
        from xorl.fp8_training.grouped import fp8_group_gemm_same_nk  # noqa: PLC0415

        return fp8_group_gemm_same_nk(
            backend=_fp8_grouped_backend(fp8_grouped_backend),
            block_size=fp8_block_size,
            **kwargs,
        )
    else:
        fn = quack_group_gemm_same_nk
    return fn(**kwargs)


def _group_gemm_same_mn(
    *,
    fp8_compute: bool = False,
    fp8_grouped_backend: str | None = None,
    fp8_block_size: int = 128,
    **kwargs,
) -> None:
    if fp8_compute:
        from xorl.fp8_training.grouped import fp8_group_gemm_same_mn  # noqa: PLC0415

        fp8_group_gemm_same_mn(
            backend=_fp8_grouped_backend(fp8_grouped_backend),
            block_size=fp8_block_size,
            **kwargs,
        )
        return
    else:
        fn = quack_group_gemm_same_mn
    fn(**kwargs)


def _moe_gate_activation_mul(gate_output: torch.Tensor, up_output: torch.Tensor, hidden_act: str) -> torch.Tensor:
    if hidden_act == "clamped_swiglu":
        gate_activation = _moe_gate_activation(gate_output, hidden_act)
        up_clamped = up_output.clamp(min=-CLAMPED_SWIGLU_LIMIT, max=CLAMPED_SWIGLU_LIMIT)
        return gate_activation.mul_(up_clamped.add(1))

    if hidden_act == "silu":
        torch.ops.aten.silu_(gate_output)
        return gate_output.mul_(up_output)

    gate_activation = _moe_gate_activation(gate_output, hidden_act)
    gated_output = gate_activation * up_output
    del gate_activation
    return gated_output


def _moe_gate_activation_product(
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    hidden_act: str,
) -> torch.Tensor:
    gate_activation = _moe_gate_activation(gate_output, hidden_act)
    if hidden_act == "clamped_swiglu":
        up_clamped = up_output.clamp(min=-CLAMPED_SWIGLU_LIMIT, max=CLAMPED_SWIGLU_LIMIT)
        return gate_activation * (up_clamped + 1)
    return gate_activation * up_output


def _moe_gate_activation_forward(
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    hidden_act: str,
    activation_native: bool,
) -> torch.Tensor:
    if activation_native:
        return _moe_gate_activation_product(gate_output, up_output, hidden_act)
    return _moe_gate_activation_mul(gate_output, up_output, hidden_act)


def _moe_gate_activation_backward_pair(
    grad_gated_output: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    hidden_act: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_activation = _moe_gate_activation(gate_output, hidden_act)
    if hidden_act == "clamped_swiglu":
        up_clamped = up_output.clamp(min=-CLAMPED_SWIGLU_LIMIT, max=CLAMPED_SWIGLU_LIMIT)
        grad_gate_activation = grad_gated_output * (up_clamped + 1)
        grad_gate_output = _moe_gate_activation_backward(grad_gate_activation, gate_output, hidden_act)
        up_mask = (up_output >= -CLAMPED_SWIGLU_LIMIT) & (up_output <= CLAMPED_SWIGLU_LIMIT)
        grad_up_output = grad_gated_output * gate_activation * up_mask.to(grad_gated_output.dtype)
        return grad_gate_output, grad_up_output

    grad_up_output = gate_activation * grad_gated_output
    grad_gate_activation = grad_gated_output * up_output
    grad_gate_output = _moe_gate_activation_backward(grad_gate_activation, gate_output, hidden_act)
    return grad_gate_output, grad_up_output


def _counts_from_cumsum(cumsum: torch.Tensor) -> torch.Tensor:
    counts = cumsum.to(torch.long).clone()
    if counts.numel() > 1:
        counts[1:] = counts[1:] - counts[:-1].clone()
    return counts


def _expert_ids_from_cumsum(cumsum: torch.Tensor) -> torch.Tensor:
    counts = _counts_from_cumsum(cumsum)
    return torch.repeat_interleave(torch.arange(cumsum.numel(), device=cumsum.device, dtype=torch.long), counts)


def _add_expert_bias_by_ids_(
    output: torch.Tensor,
    bias: torch.Tensor | None,
    expert_ids: torch.Tensor,
    start: int | None = None,
    end: int | None = None,
) -> None:
    if bias is None or output.numel() == 0:
        return
    bias_slice = bias if start is None else bias[..., start:end]
    output.add_(bias_slice.to(output.dtype).index_select(0, expert_ids))


def _sum_expert_bias_grad_by_ids(
    grad: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    accum_dtype = torch.float32 if grad.dtype in {torch.bfloat16, torch.float16} else grad.dtype
    out = torch.zeros(num_experts, grad.shape[-1], dtype=accum_dtype, device=grad.device)
    if grad.numel() == 0:
        return out.to(out_dtype)

    chunk_tokens = _quack_deepep_scatter_chunk_tokens()
    for start in range(0, grad.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, grad.shape[0])
        grad_chunk = grad[start:end].to(accum_dtype)
        out.index_add_(0, expert_ids[start:end], grad_chunk)
        del grad_chunk
    return out.to(out_dtype)


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def _memory_trace_enabled() -> bool:
    return _env_flag("XORL_QUACK_EP_MEMORY_TRACE")


_BACKWARD_DEBUG_FINITE = _env_flag("XORL_QUACK_DEEPEP_BACKWARD_DEBUG_FINITE")
_BACKWARD_DEBUG_FINITE_REPORTS = 0


def _debug_assert_finite(name: str, tensor: torch.Tensor | None, context: str = "") -> None:
    """XORL_QUACK_DEEPEP_BACKWARD_DEBUG_FINITE=1: report the FIRST tensors that
    go non-finite inside the no-permute backward (syncs per check — debug only)."""
    global _BACKWARD_DEBUG_FINITE_REPORTS
    if not _BACKWARD_DEBUG_FINITE or _BACKWARD_DEBUG_FINITE_REPORTS >= 20:
        return
    if tensor is None or tensor.numel() == 0:
        return
    bad = int((~torch.isfinite(tensor.float())).sum())
    if bad:
        _BACKWARD_DEBUG_FINITE_REPORTS += 1
        print(
            f"[quack-deepep-bwd-finite][rank{_rank()}] {name}{f' ({context})' if context else ''}: "
            f"{bad}/{tensor.numel()} non-finite, shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
            flush=True,
        )


_MEMORY_TRACE = _memory_trace_enabled()
_MEMORY_TRACE_RANKS = os.environ.get("XORL_QUACK_EP_MEMORY_TRACE_RANKS", "all").strip().lower()
_MEMORY_TRACE_MIN_ALLOCATED_BYTES = int(
    float(os.environ.get("XORL_QUACK_EP_MEMORY_TRACE_MIN_ALLOCATED_GB", "0")) * (1024**3)
)
_MEMORY_TRACE_CALL_ID = 0


def _next_memory_trace_call_id() -> int:
    global _MEMORY_TRACE_CALL_ID
    _MEMORY_TRACE_CALL_ID += 1
    return _MEMORY_TRACE_CALL_ID


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", "0"))


def _memory_trace_rank_enabled(rank: int) -> bool:
    if _MEMORY_TRACE_RANKS in {"", "all", "*"}:
        return True
    return str(rank) in {r.strip() for r in _MEMORY_TRACE_RANKS.split(",") if r.strip()}


def _tensor_mib(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / (1024**2)


def _tensor_desc(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{name}=None"
    return f"{name}=shape{tuple(tensor.shape)} {tensor.dtype} {tensor.device} {_tensor_mib(tensor):.1f}MiB"


def _memory_trace_start(phase: str) -> tuple[int, bool]:
    call_id = _next_memory_trace_call_id()
    enabled = _MEMORY_TRACE and torch.cuda.is_available() and _memory_trace_rank_enabled(_rank())
    if enabled:
        torch.cuda.reset_peak_memory_stats()
        _memory_trace(phase, call_id, "start", force=True)
    return call_id, enabled


def _memory_trace(
    phase: str,
    call_id: int,
    stage: str,
    *named_tensors: tuple[str, torch.Tensor | None],
    force: bool = False,
) -> None:
    if not torch.cuda.is_available():
        return
    rank = _rank()
    enabled = (_MEMORY_TRACE and _memory_trace_rank_enabled(rank)) or force
    if not enabled:
        return
    allocated = torch.cuda.memory_allocated()
    if not force and allocated < _MEMORY_TRACE_MIN_ALLOCATED_BYTES:
        return
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    free, total = torch.cuda.mem_get_info()
    tensors = ", ".join(_tensor_desc(name, tensor) for name, tensor in named_tensors)
    print(
        f"[QuackEPMem r{rank} {phase}#{call_id}] {stage}: "
        f"alloc={allocated / (1024**3):.2f}GiB reserved={reserved / (1024**3):.2f}GiB "
        f"peak={peak / (1024**3):.2f}GiB free={free / (1024**3):.2f}GiB "
        f"total={total / (1024**3):.2f}GiB" + (f" | {tensors}" if tensors else ""),
        flush=True,
    )


def _quack_gated_activation(hidden_act: str) -> str:
    if hidden_act == "gelu_tanh":
        return "geglu"
    return "swiglu"


def _scatter_and_cumsum(hidden_states: torch.Tensor, expert_index: torch.Tensor, num_experts: int):
    splits = expert_histogram(expert_index, num_experts)
    cumsum_t = torch.cumsum(splits, dim=0)
    scatter_index = moe_index_compute(expert_index, cumsum_t)
    scatter_output = moe_scatter(hidden_states, scatter_index)
    return scatter_output, scatter_index, cumsum_t


_CHUNKED_DEEPEP_WARNED = False


def _warn_chunked_deepep_once() -> None:
    """Chunked mode multiplies DeepEP combines/cached-dispatches per dispatch
    handle (e.g. 16 each at chunk=256 on Qwen3.6-35B), far outside deep_ep's
    designed one-dispatch-one-combine handle cadence. Under
    recompute_full_layer this corrupted comm-stream-owned outputs (scattered
    non-finite elements -> nan grads); full-size chunks are also ~1.6x faster.
    See docs/notes/quack_moe_nan_root_cause_record_stream.md (2026-06-12).
    Chunking remains available as an explicit memory lever for huge shapes."""
    global _CHUNKED_DEEPEP_WARNED
    if not _CHUNKED_DEEPEP_WARNED:
        _CHUNKED_DEEPEP_WARNED = True
        print(
            "[xorl.ops.moe.quack] WARNING: chunked DeepEP no-permute mode enabled via "
            "XORL_QUACK_DEEPEP_*_CHUNK_SIZE — known to corrupt gradients under "
            "recompute_full_layer (see docs/notes/quack_moe_nan_root_cause_record_stream.md); "
            "prefer the full-size default.",
            flush=True,
        )


def _quack_deepep_hidden_chunk_size(hidden_dim: int) -> int:
    # Default 0 = full hidden dim (one DeepEP combine / cached dispatch per
    # handle, deep_ep's designed cadence). Non-full chunking is opt-in only.
    chunk = int(os.environ.get("XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE", "0"))
    if chunk <= 0:
        return hidden_dim
    if chunk < hidden_dim:
        _warn_chunked_deepep_once()
    return min(chunk, hidden_dim)


def _quack_deepep_intermediate_chunk_size(intermediate_dim: int) -> int:
    chunk = int(os.environ.get("XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE", "0"))
    if chunk <= 0:
        return intermediate_dim
    return min(chunk, intermediate_dim)


def _quack_deepep_scatter_chunk_tokens() -> int:
    return max(1, int(os.environ.get("XORL_QUACK_DEEPEP_SCATTER_CHUNK_TOKENS", "4096")))


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _quack_deepep_backward_dispatch_async_enabled() -> bool:
    return _env_flag("XORL_QUACK_DEEPEP_BACKWARD_DISPATCH_ASYNC", True)


def _quack_deepep_backward_dispatch_allocate_on_comm_stream_enabled(async_finish: bool) -> bool:
    return async_finish and _env_flag("XORL_QUACK_DEEPEP_BACKWARD_DISPATCH_ALLOCATE_ON_COMM_STREAM", True)


def _quack_deepep_backward_token_block_size(max_m: int) -> int:
    block = int(os.environ.get("XORL_QUACK_DEEPEP_BACKWARD_TOKEN_BLOCK_SIZE", "0"))
    if block <= 0 or max_m <= 0:
        return 0
    return min(block, max_m)


def _quack_deepep_backward_block_group_size(num_blocks: int) -> int:
    group_size = int(os.environ.get("XORL_QUACK_DEEPEP_BACKWARD_BLOCK_GROUP_SIZE", "1"))
    if group_size <= 0 or num_blocks <= 0:
        return 1
    return min(group_size, num_blocks)


def _quack_deepep_fused_gate_up_enabled() -> bool:
    return _env_flag("XORL_QUACK_DEEPEP_FUSED_GATE_UP", True)


def _quack_deepep_fused_gate_up_dgrad_enabled() -> bool:
    return _env_flag("XORL_QUACK_DEEPEP_FUSED_GATE_UP_DGRAD", True)


def _quack_deepep_fused_gate_up_wgrad_enabled() -> bool:
    return _env_flag("XORL_QUACK_DEEPEP_FUSED_GATE_UP_WGRAD", True)


def _quack_deepep_precompute_grad_gate_up_enabled() -> bool:
    return _env_flag("XORL_QUACK_DEEPEP_PRECOMPUTE_GRAD_GATE_UP", True)


def _quack_full_intermediate_chunk(intermediate_size: int, istart: int, iend: int) -> bool:
    return istart == 0 and iend == intermediate_size


def _expert_token_counts(cumsum: torch.Tensor) -> list[int]:
    counts = []
    prev = 0
    for end_value in cumsum.detach().cpu().tolist():
        expert_end = int(end_value)
        counts.append(expert_end - prev)
        prev = expert_end
    return counts


def _build_expert_token_blocks(cumsum: torch.Tensor, block_tokens: int, expert_counts: list[int] | None = None):
    blocks = []
    start = 0
    if expert_counts is None:
        expert_counts = _expert_token_counts(cumsum)
    for expert_idx, count in enumerate(expert_counts):
        expert_end = start + count
        while start < expert_end:
            end = min(start + block_tokens, expert_end)
            block_cumsum = cumsum.new_tensor([end - start])
            block_cu_seqlens = cumsum_to_cu_seqlens(block_cumsum)
            blocks.append((expert_idx, start, end, block_cumsum, block_cu_seqlens))
            start = end
        start = expert_end
    return blocks


def _group_expert_token_blocks_by_expert(
    expert_token_blocks,
    num_experts: int,
    expert_group_size: int,
):
    groups = [[] for _ in range(0, num_experts, expert_group_size)]
    for block in expert_token_blocks:
        expert_idx = block[0]
        groups[expert_idx // expert_group_size].append(block)
    return groups


def _empty_cache_for_retry(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.empty_cache()


def _empty_like_with_cache_retry(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return torch.empty_like(tensor)
    except torch.cuda.OutOfMemoryError:
        _empty_cache_for_retry(tensor)
        return torch.empty_like(tensor)


def _new_empty_with_cache_retry(tensor: torch.Tensor, *size) -> torch.Tensor:
    try:
        return tensor.new_empty(*size)
    except torch.cuda.OutOfMemoryError:
        _empty_cache_for_retry(tensor)
        return tensor.new_empty(*size)


def _iter_expert_token_blocks(cumsum: torch.Tensor, block_tokens: int):
    for expert_idx, start, end, block_cumsum, _ in _build_expert_token_blocks(cumsum, block_tokens):
        yield expert_idx, start, end, block_cumsum


def _empty_expert_grad_like(param: torch.Tensor, expert_counts: list[int]) -> torch.Tensor:
    grad = _empty_like_with_cache_retry(param)
    for expert_idx, count in enumerate(expert_counts):
        if count == 0:
            grad[expert_idx].zero_()
    return grad


def _index_select_rows_with_workspace(
    source: torch.Tensor,
    indices: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = indices.shape[0]
    cols = source.shape[1]
    if (
        workspace is None
        or workspace.shape[0] < rows
        or workspace.shape[1] != cols
        or workspace.dtype != source.dtype
        or workspace.device != source.device
    ):
        workspace = source.new_empty(rows, cols)
    output = workspace[:rows]
    torch.index_select(source, 0, indices, out=output)
    return output, workspace


def _copy_tensor_with_workspace(
    source: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_shape = tuple(source.shape)
    if (
        workspace is None
        or workspace.dim() != source.dim()
        or any(workspace.shape[dim] < size for dim, size in enumerate(source_shape))
        or workspace.dtype != source.dtype
        or workspace.device != source.device
    ):
        workspace = source.new_empty(source_shape)
    output = workspace[tuple(slice(0, size) for size in source_shape)]
    output.copy_(source)
    return output, workspace


def _maybe_precompute_grad_gate_up_chunk(
    grad_gate_chunk: torch.Tensor,
    grad_up_chunk: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    need_hidden_grad: bool,
    need_weight_grad: bool,
) -> torch.Tensor | None:
    if not _quack_deepep_precompute_grad_gate_up_enabled():
        return None
    if not _quack_full_intermediate_chunk(intermediate_size, istart, iend):
        return None
    if (need_hidden_grad and _quack_deepep_fused_gate_up_dgrad_enabled()) or (
        need_weight_grad and _quack_deepep_fused_gate_up_wgrad_enabled()
    ):
        return torch.cat((grad_gate_chunk, grad_up_chunk), dim=-1)
    return None


def _grad_up_chunk_from_gate_activation(
    gate_output: torch.Tensor,
    grad_gated_chunk: torch.Tensor,
    hidden_act: str,
) -> torch.Tensor:
    grad_up_chunk = _moe_gate_activation(gate_output, hidden_act)
    grad_up_chunk.mul_(grad_gated_chunk)
    return grad_up_chunk


def _gate_up_activation_backward_chunks(
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    grad_gated_chunk: torch.Tensor,
    hidden_act: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_act == "silu":
        grad_gate_activation = up_output
        grad_gate_activation.mul_(grad_gated_chunk)
        grad_gate_chunk = _moe_gate_activation_backward(grad_gate_activation, gate_output, hidden_act)
        grad_up_chunk = torch.ops.aten.silu_(gate_output)
        grad_up_chunk.mul_(grad_gated_chunk)
        return grad_gate_chunk, grad_up_chunk

    grad_up_chunk = _grad_up_chunk_from_gate_activation(gate_output, grad_gated_chunk, hidden_act)
    grad_gate_activation = grad_gated_chunk.mul_(up_output)
    grad_gate_chunk = _moe_gate_activation_backward(grad_gate_activation, gate_output, hidden_act)
    return grad_gate_chunk, grad_up_chunk


def _add_gate_up_proj_grad_block(
    grad_gate_up_proj: torch.Tensor,
    expert_idx: int,
    token_chunk: torch.Tensor,
    grad_gate_chunk: torch.Tensor,
    grad_up_chunk: torch.Tensor,
    cumsum: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_start: int,
    hidden_end: int,
    max_M: int,
    cu_seqlens: torch.Tensor,
    accumulate: bool = True,
) -> None:
    if _quack_full_intermediate_chunk(intermediate_size, istart, iend) and _quack_deepep_fused_gate_up_wgrad_enabled():
        grad_gate_up_chunk = torch.cat((grad_gate_chunk, grad_up_chunk), dim=-1)
        grad_gate_up_proj_chunk = torch.empty(
            1,
            hidden_end - hidden_start,
            2 * intermediate_size,
            dtype=grad_gate_up_proj.dtype,
            device=grad_gate_up_proj.device,
        )
        quack_group_gemm_same_mn(
            a=token_chunk,
            b=grad_gate_up_chunk,
            c=grad_gate_up_proj_chunk,
            cumsum_K=cumsum,
            max_K=max_M,
            transpose_a=True,
            transpose_b=False,
            cu_seqlens_k=cu_seqlens,
        )
        target = grad_gate_up_proj[expert_idx : expert_idx + 1, hidden_start:hidden_end, : 2 * intermediate_size]
        if accumulate:
            target.add_(grad_gate_up_proj_chunk)
        else:
            target.copy_(grad_gate_up_proj_chunk)
        del target
        del grad_gate_up_chunk, grad_gate_up_proj_chunk
        return

    grad_gate_proj_chunk = torch.empty(
        1,
        hidden_end - hidden_start,
        iend - istart,
        dtype=grad_gate_up_proj.dtype,
        device=grad_gate_up_proj.device,
    )
    quack_group_gemm_same_mn(
        a=token_chunk,
        b=grad_gate_chunk,
        c=grad_gate_proj_chunk,
        cumsum_K=cumsum,
        max_K=max_M,
        transpose_a=True,
        transpose_b=False,
        cu_seqlens_k=cu_seqlens,
    )
    target = grad_gate_up_proj[expert_idx : expert_idx + 1, hidden_start:hidden_end, istart:iend]
    if accumulate:
        target.add_(grad_gate_proj_chunk)
    else:
        target.copy_(grad_gate_proj_chunk)
    del target
    del grad_gate_proj_chunk

    grad_up_proj_chunk = torch.empty(
        1,
        hidden_end - hidden_start,
        iend - istart,
        dtype=grad_gate_up_proj.dtype,
        device=grad_gate_up_proj.device,
    )
    quack_group_gemm_same_mn(
        a=token_chunk,
        b=grad_up_chunk,
        c=grad_up_proj_chunk,
        cumsum_K=cumsum,
        max_K=max_M,
        transpose_a=True,
        transpose_b=False,
        cu_seqlens_k=cu_seqlens,
    )
    target = grad_gate_up_proj[
        expert_idx : expert_idx + 1, hidden_start:hidden_end, intermediate_size + istart : intermediate_size + iend
    ]
    if accumulate:
        target.add_(grad_up_proj_chunk)
    else:
        target.copy_(grad_up_proj_chunk)
    del target
    del grad_up_proj_chunk


def _scatter_expert_chunk_to_recv(
    expert_chunk: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_recv_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    from xorl.distributed.moe.deepep import (  # noqa: PLC0415
        _cast_if_needed,
        _deepep_combine_scatter_accum_dtype,
    )

    accum_dtype = _deepep_combine_scatter_accum_dtype(dtype)
    gather_chunk = torch.zeros(
        num_recv_tokens,
        expert_chunk.shape[1],
        dtype=accum_dtype,
        device=expert_chunk.device,
    )
    if expert_chunk.numel() == 0:
        return _cast_if_needed(gather_chunk, dtype)

    chunk_tokens = _quack_deepep_scatter_chunk_tokens()
    for start in range(0, expert_chunk.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, expert_chunk.shape[0])
        idx = permuted_indices[start:end].unsqueeze(1).expand(-1, expert_chunk.shape[1])
        if expert_chunk.dtype == gather_chunk.dtype:
            gather_chunk.scatter_add_(0, idx, expert_chunk[start:end])
        else:
            expert_slice = expert_chunk[start:end].to(gather_chunk.dtype)
            gather_chunk.scatter_add_(0, idx, expert_slice)
            del expert_slice

    return _cast_if_needed(gather_chunk, dtype)


def _quack_deepep_combine_chunk_async_enabled() -> bool:
    return _env_flag("XORL_QUACK_DEEPEP_COMBINE_CHUNK_ASYNC", True)


def _deepep_combine_chunk(buffer, gather_chunk: torch.Tensor, handle) -> torch.Tensor:
    from xorl.distributed.moe.deepep import EventHandle, EventOverlap  # noqa: PLC0415

    previous_event = EventOverlap(EventHandle())
    async_finish = _quack_deepep_combine_chunk_async_enabled()
    combined_chunk, _, event = buffer.buffer.combine(
        x=gather_chunk.contiguous(),
        handle=handle,
        config=buffer.combine_config,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=async_finish,
    )
    if getattr(event, "event", None) is not None:
        event.current_stream_wait()
    if async_finish:
        # combined_chunk is OWNED by the comm stream (allocate_on_comm_stream)
        # but consumed on the compute stream. Without record_stream, freeing it
        # returns the block to the comm-stream pool and the NEXT chunked
        # collective can overwrite it while the compute-stream read is still in
        # flight (the chunked path frees 16 such tensors per layer; bit the
        # Qwen3.6-35B recompute_full_layer runs as backward nan, 2026-06-12).
        combined_chunk.record_stream(torch.cuda.current_stream())
    return combined_chunk


def _write_combined_output_chunk(
    combined_output: torch.Tensor,
    combined_chunk: torch.Tensor,
    start: int,
    end: int,
    first_intermediate_chunk: bool,
) -> None:
    output_slice = combined_output[:, start:end]
    if first_intermediate_chunk:
        output_slice.copy_(combined_chunk)
    else:
        output_slice.add_(combined_chunk)
    del output_slice


def _deepep_dispatch_grad_chunk(buffer, grad_chunk: torch.Tensor, handle) -> torch.Tensor:
    from xorl.distributed.moe.deepep import EventHandle, EventOverlap  # noqa: PLC0415

    previous_event = EventOverlap(EventHandle())
    async_finish = _quack_deepep_backward_dispatch_async_enabled()
    allocate_on_comm_stream = _quack_deepep_backward_dispatch_allocate_on_comm_stream_enabled(async_finish)
    grad_gather, _, _, _, _, event = buffer.buffer.dispatch(
        x=grad_chunk.contiguous(),
        handle=handle,
        config=buffer.dispatch_config,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    if getattr(event, "event", None) is not None:
        event.current_stream_wait()
    if allocate_on_comm_stream:
        # Comm-stream-owned output consumed on the compute stream; see
        # _deepep_combine_chunk for the free/reuse race this prevents.
        grad_gather.record_stream(torch.cuda.current_stream())
    return grad_gather


def _gather_recv_hidden_chunk(
    recv_x: torch.Tensor,
    permuted_indices: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    return recv_x[:, start:end].index_select(0, permuted_indices).contiguous()


def _quack_gate_up_chunk_from_recv(
    recv_x: torch.Tensor,
    permuted_indices: torch.Tensor,
    cumsum: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_chunk_size: int,
    cu_seqlens: torch.Tensor,
    fp8_compute: bool = False,
    fp8_grouped_backend: str | None = None,
    fp8_block_size: int = 128,
    gate_up_bias: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    token_block_size: int = 0,
    token_blocks=None,
):
    max_M = permuted_indices.shape[0]
    if max_M == 0:
        return (
            recv_x.new_empty(0, iend - istart),
            recv_x.new_empty(0, iend - istart),
        )
    if token_block_size > 0:
        if token_blocks is None:
            token_blocks = _build_expert_token_blocks(cumsum, token_block_size)
        return _quack_gate_up_chunk_from_recv_token_blocks(
            recv_x,
            permuted_indices,
            cumsum,
            gate_up_proj,
            intermediate_size,
            istart,
            iend,
            hidden_chunk_size,
            token_blocks,
        )

    hidden_dim = recv_x.shape[1]
    if istart == 0 and iend == intermediate_size and _quack_deepep_fused_gate_up_enabled():
        gate_up_output = None
        for start in range(0, hidden_dim, hidden_chunk_size):
            end = min(start + hidden_chunk_size, hidden_dim)
            token_chunk = _gather_recv_hidden_chunk(recv_x, permuted_indices, start, end)
            gate_up_weight = gate_up_proj[..., start:end, : 2 * intermediate_size].contiguous()
            gate_up_part = quack_group_gemm_same_nk(
                a=token_chunk,
                b=gate_up_weight,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=False,
                cu_seqlens_m=cu_seqlens,
            )
            if gate_up_output is None:
                gate_up_output = gate_up_part
            else:
                gate_up_output.add_(gate_up_part)
                del gate_up_part
            del token_chunk, gate_up_weight

        assert gate_up_output is not None
        return gate_up_output[..., :intermediate_size], gate_up_output[..., intermediate_size:]

    gate_output = None
    up_output = None
    for start in range(0, hidden_dim, hidden_chunk_size):
        end = min(start + hidden_chunk_size, hidden_dim)
        token_chunk = _gather_recv_hidden_chunk(recv_x, permuted_indices, start, end)

        gate_weight = gate_up_proj[..., start:end, istart:iend].contiguous()
        gate_part = _group_gemm_same_nk(
            a=token_chunk,
            b=gate_weight,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        if gate_output is None:
            gate_output = gate_part
        else:
            gate_output.add_(gate_part)
            del gate_part
        del gate_weight

        up_weight = gate_up_proj[..., start:end, intermediate_size + istart : intermediate_size + iend].contiguous()
        up_part = _group_gemm_same_nk(
            a=token_chunk,
            b=up_weight,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        if up_output is None:
            up_output = up_part
        else:
            up_output.add_(up_part)
            del up_part
        del token_chunk, up_weight

    assert gate_output is not None and up_output is not None
    return gate_output, up_output


def _quack_gate_up_chunk_from_recv_token_blocks(
    recv_x: torch.Tensor,
    permuted_indices: torch.Tensor,
    cumsum: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_chunk_size: int,
    token_blocks,
):
    max_M = permuted_indices.shape[0]
    gate_output = recv_x.new_empty(max_M, iend - istart)
    up_output = recv_x.new_empty(max_M, iend - istart)
    hidden_dim = recv_x.shape[1]

    for expert_idx, block_start, block_end, block_cumsum, block_cu_seqlens in token_blocks:
        block_m = block_end - block_start
        block_indices = permuted_indices[block_start:block_end]
        gate_block = gate_output[block_start:block_end]
        up_block = up_output[block_start:block_end]
        first_hidden_chunk = True
        token_workspace = None
        gate_weight_workspace = None
        up_weight_workspace = None
        for start in range(0, hidden_dim, hidden_chunk_size):
            end = min(start + hidden_chunk_size, hidden_dim)
            token_chunk, token_workspace = _index_select_rows_with_workspace(
                recv_x[:, start:end],
                block_indices,
                token_workspace,
            )

            gate_weight, gate_weight_workspace = _copy_tensor_with_workspace(
                gate_up_proj[expert_idx : expert_idx + 1, start:end, istart:iend],
                gate_weight_workspace,
            )
            gate_part = quack_group_gemm_same_nk(
                a=token_chunk,
                b=gate_weight,
                cumsum_M=block_cumsum,
                max_M=block_m,
                transpose_b=False,
                out=gate_block if first_hidden_chunk else None,
                cu_seqlens_m=block_cu_seqlens,
            )
            if not first_hidden_chunk:
                gate_block.add_(gate_part)
                del gate_part
            del gate_weight

            up_weight, up_weight_workspace = _copy_tensor_with_workspace(
                gate_up_proj[
                    expert_idx : expert_idx + 1, start:end, intermediate_size + istart : intermediate_size + iend
                ],
                up_weight_workspace,
            )
            up_part = quack_group_gemm_same_nk(
                a=token_chunk,
                b=up_weight,
                cumsum_M=block_cumsum,
                max_M=block_m,
                transpose_b=False,
                out=up_block if first_hidden_chunk else None,
                cu_seqlens_m=block_cu_seqlens,
            )
            if not first_hidden_chunk:
                up_block.add_(up_part)
                del up_part
            del token_chunk, up_weight
            first_hidden_chunk = False
        del token_workspace
        del gate_weight_workspace, up_weight_workspace

    # NOTE: expert bias (GPT-OSS) is not applied on the token-blocks streaming path;
    # GPT-OSS bias routes through the non-streaming helper / explicit FP8 path.
    return gate_output, up_output


def _quack_gate_up_chunk_from_tokens(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    max_M: int,
    cu_seqlens: torch.Tensor,
    token_block_size: int = 0,
    token_blocks=None,
):
    if max_M == 0:
        return (
            permute_tokens.new_empty(0, iend - istart),
            permute_tokens.new_empty(0, iend - istart),
        )
    if token_block_size > 0:
        if token_blocks is None:
            token_blocks = _build_expert_token_blocks(cumsum, token_block_size)
        return _quack_gate_up_chunk_from_tokens_token_blocks(
            permute_tokens,
            cumsum,
            gate_up_proj,
            intermediate_size,
            istart,
            iend,
            token_blocks,
        )

    if istart == 0 and iend == intermediate_size and _quack_deepep_fused_gate_up_enabled():
        gate_up_weight = gate_up_proj[..., : 2 * intermediate_size].contiguous()
        gate_up_output = quack_group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_weight,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
        )
        del gate_up_weight
        return gate_up_output[..., :intermediate_size], gate_up_output[..., intermediate_size:]

    gate_proj = gate_up_proj[..., istart:iend].contiguous()
    gate_output = quack_group_gemm_same_nk(
        a=permute_tokens,
        b=gate_proj,
        cumsum_M=cumsum,
        max_M=max_M,
        transpose_b=False,
        cu_seqlens_m=cu_seqlens,
    )
    del gate_proj

    up_proj = gate_up_proj[..., intermediate_size + istart : intermediate_size + iend].contiguous()
    up_output = quack_group_gemm_same_nk(
        a=permute_tokens,
        b=up_proj,
        cumsum_M=cumsum,
        max_M=max_M,
        transpose_b=False,
        cu_seqlens_m=cu_seqlens,
    )
    del up_proj
    return gate_output, up_output


def _quack_gate_up_chunk_from_tokens_token_blocks(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    token_blocks,
):
    max_M = permute_tokens.shape[0]
    gate_output = permute_tokens.new_empty(max_M, iend - istart)
    up_output = permute_tokens.new_empty(max_M, iend - istart)

    for expert_idx, block_start, block_end, block_cumsum, block_cu_seqlens in token_blocks:
        block_m = block_end - block_start
        token_chunk = permute_tokens[block_start:block_end].contiguous()

        gate_weight = gate_up_proj[expert_idx : expert_idx + 1, :, istart:iend].contiguous()
        quack_group_gemm_same_nk(
            a=token_chunk,
            b=gate_weight,
            cumsum_M=block_cumsum,
            max_M=block_m,
            transpose_b=False,
            out=gate_output[block_start:block_end],
            cu_seqlens_m=block_cu_seqlens,
        )
        del gate_weight

        up_weight = gate_up_proj[
            expert_idx : expert_idx + 1, :, intermediate_size + istart : intermediate_size + iend
        ].contiguous()
        quack_group_gemm_same_nk(
            a=token_chunk,
            b=up_weight,
            cumsum_M=block_cumsum,
            max_M=block_m,
            transpose_b=False,
            out=up_output[block_start:block_end],
            cu_seqlens_m=block_cu_seqlens,
        )
        del token_chunk, up_weight

    return gate_output, up_output


def _quack_gate_up_block_from_tokens(
    permute_tokens: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    expert_idx: int,
    block_start: int,
    block_end: int,
    block_cumsum: torch.Tensor,
    block_cu_seqlens: torch.Tensor,
):
    block_m = block_end - block_start
    if block_m == 0:
        return (
            permute_tokens.new_empty(0, iend - istart),
            permute_tokens.new_empty(0, iend - istart),
        )

    token_chunk, token_workspace = _copy_tensor_with_workspace(permute_tokens[block_start:block_end])
    if _quack_full_intermediate_chunk(intermediate_size, istart, iend) and _quack_deepep_fused_gate_up_enabled():
        gate_up_weight, gate_up_weight_workspace = _copy_tensor_with_workspace(
            gate_up_proj[expert_idx : expert_idx + 1, :, : 2 * intermediate_size]
        )
        gate_up_output = quack_group_gemm_same_nk(
            a=token_chunk,
            b=gate_up_weight,
            cumsum_M=block_cumsum,
            max_M=block_m,
            transpose_b=False,
            cu_seqlens_m=block_cu_seqlens,
        )
        del token_chunk, token_workspace, gate_up_weight, gate_up_weight_workspace
        return gate_up_output[..., :intermediate_size], gate_up_output[..., intermediate_size:]

    gate_weight, gate_weight_workspace = _copy_tensor_with_workspace(
        gate_up_proj[expert_idx : expert_idx + 1, :, istart:iend]
    )
    gate_output = quack_group_gemm_same_nk(
        a=token_chunk,
        b=gate_weight,
        cumsum_M=block_cumsum,
        max_M=block_m,
        transpose_b=False,
        cu_seqlens_m=block_cu_seqlens,
    )
    del gate_weight

    up_weight, up_weight_workspace = _copy_tensor_with_workspace(
        gate_up_proj[expert_idx : expert_idx + 1, :, intermediate_size + istart : intermediate_size + iend],
        gate_weight_workspace,
    )
    up_output = quack_group_gemm_same_nk(
        a=token_chunk,
        b=up_weight,
        cumsum_M=block_cumsum,
        max_M=block_m,
        transpose_b=False,
        cu_seqlens_m=block_cu_seqlens,
    )
    del token_chunk, token_workspace, gate_weight_workspace, up_weight, up_weight_workspace
    return gate_output, up_output


def _quack_gate_up_block_from_recv(
    recv_x: torch.Tensor,
    permuted_indices: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_chunk_size: int,
    expert_idx: int,
    block_start: int,
    block_end: int,
    block_cumsum: torch.Tensor,
    block_cu_seqlens: torch.Tensor,
):
    block_m = block_end - block_start
    if block_m == 0:
        return (
            recv_x.new_empty(0, iend - istart),
            recv_x.new_empty(0, iend - istart),
        )

    block_indices = permuted_indices[block_start:block_end]
    hidden_dim = recv_x.shape[1]
    if _quack_full_intermediate_chunk(intermediate_size, istart, iend) and _quack_deepep_fused_gate_up_enabled():
        gate_up_output = None
        token_workspace = None
        gate_up_weight_workspace = None
        for start in range(0, hidden_dim, hidden_chunk_size):
            end = min(start + hidden_chunk_size, hidden_dim)
            token_chunk, token_workspace = _index_select_rows_with_workspace(
                recv_x[:, start:end],
                block_indices,
                token_workspace,
            )
            gate_up_weight, gate_up_weight_workspace = _copy_tensor_with_workspace(
                gate_up_proj[expert_idx : expert_idx + 1, start:end, : 2 * intermediate_size],
                gate_up_weight_workspace,
            )
            gate_up_part = quack_group_gemm_same_nk(
                a=token_chunk,
                b=gate_up_weight,
                cumsum_M=block_cumsum,
                max_M=block_m,
                transpose_b=False,
                cu_seqlens_m=block_cu_seqlens,
            )
            if gate_up_output is None:
                gate_up_output = gate_up_part
            else:
                gate_up_output.add_(gate_up_part)
                del gate_up_part
            del token_chunk, gate_up_weight

        assert gate_up_output is not None
        del token_workspace
        del gate_up_weight_workspace
        return gate_up_output[..., :intermediate_size], gate_up_output[..., intermediate_size:]

    gate_output = None
    up_output = None
    token_workspace = None
    gate_weight_workspace = None
    up_weight_workspace = None
    for start in range(0, hidden_dim, hidden_chunk_size):
        end = min(start + hidden_chunk_size, hidden_dim)
        token_chunk, token_workspace = _index_select_rows_with_workspace(
            recv_x[:, start:end],
            block_indices,
            token_workspace,
        )

        gate_weight, gate_weight_workspace = _copy_tensor_with_workspace(
            gate_up_proj[expert_idx : expert_idx + 1, start:end, istart:iend],
            gate_weight_workspace,
        )
        gate_part = quack_group_gemm_same_nk(
            a=token_chunk,
            b=gate_weight,
            cumsum_M=block_cumsum,
            max_M=block_m,
            transpose_b=False,
            cu_seqlens_m=block_cu_seqlens,
        )
        if gate_output is None:
            gate_output = gate_part
        else:
            gate_output.add_(gate_part)
            del gate_part
        del gate_weight

        up_weight, up_weight_workspace = _copy_tensor_with_workspace(
            gate_up_proj[expert_idx : expert_idx + 1, start:end, intermediate_size + istart : intermediate_size + iend],
            up_weight_workspace,
        )
        up_part = quack_group_gemm_same_nk(
            a=token_chunk,
            b=up_weight,
            cumsum_M=block_cumsum,
            max_M=block_m,
            transpose_b=False,
            cu_seqlens_m=block_cu_seqlens,
        )
        if up_output is None:
            up_output = up_part
        else:
            up_output.add_(up_part)
            del up_part
        del token_chunk, up_weight

    assert gate_output is not None and up_output is not None
    del token_workspace
    del gate_weight_workspace, up_weight_workspace
    return gate_output, up_output


def _quack_gated_chunk_from_tokens_token_blocks(
    permute_tokens: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    token_blocks,
    hidden_act: str,
) -> torch.Tensor:
    gated_chunk = _new_empty_with_cache_retry(permute_tokens, permute_tokens.shape[0], iend - istart)
    for expert_idx, block_start, block_end, block_cumsum, block_cu_seqlens in token_blocks:
        gate_block, up_block = _quack_gate_up_block_from_tokens(
            permute_tokens,
            gate_up_proj,
            intermediate_size,
            istart,
            iend,
            expert_idx,
            block_start,
            block_end,
            block_cumsum,
            block_cu_seqlens,
        )
        gated_block = _moe_gate_activation_mul(gate_block, up_block, hidden_act)
        gated_chunk[block_start:block_end].copy_(gated_block)
        del gate_block, up_block, gated_block
    return gated_chunk


def _quack_gated_chunk_from_recv_token_blocks(
    recv_x: torch.Tensor,
    permuted_indices: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_chunk_size: int,
    max_M: int,
    token_blocks,
    hidden_act: str,
) -> torch.Tensor:
    gated_chunk = _new_empty_with_cache_retry(recv_x, max_M, iend - istart)
    for expert_idx, block_start, block_end, block_cumsum, block_cu_seqlens in token_blocks:
        gate_block, up_block = _quack_gate_up_block_from_recv(
            recv_x,
            permuted_indices,
            gate_up_proj,
            intermediate_size,
            istart,
            iend,
            hidden_chunk_size,
            expert_idx,
            block_start,
            block_end,
            block_cumsum,
            block_cu_seqlens,
        )
        gated_block = _moe_gate_activation_mul(gate_block, up_block, hidden_act)
        gated_chunk[block_start:block_end].copy_(gated_block)
        del gate_block, up_block, gated_block
    return gated_chunk


def _quack_hidden_grad_chunk_from_gate_up(
    grad_gate_chunk: torch.Tensor,
    grad_up_chunk: torch.Tensor,
    cumsum: torch.Tensor,
    gate_up_proj: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_start: int,
    hidden_end: int,
    max_M: int,
    cu_seqlens: torch.Tensor,
    grad_gate_up_chunk: torch.Tensor | None = None,
) -> torch.Tensor:
    if _quack_full_intermediate_chunk(intermediate_size, istart, iend) and _quack_deepep_fused_gate_up_dgrad_enabled():
        owns_grad_gate_up_chunk = grad_gate_up_chunk is None
        if grad_gate_up_chunk is None:
            grad_gate_up_chunk = torch.cat((grad_gate_chunk, grad_up_chunk), dim=-1)
        gate_up_weight = gate_up_proj[..., hidden_start:hidden_end, : 2 * intermediate_size].contiguous()
        grad_hidden_chunk = quack_group_gemm_same_nk(
            a=grad_gate_up_chunk,
            b=gate_up_weight,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens,
        )
        if owns_grad_gate_up_chunk:
            del grad_gate_up_chunk
        del gate_up_weight
        return grad_hidden_chunk

    gate_weight = gate_up_proj[..., hidden_start:hidden_end, istart:iend].contiguous()
    grad_hidden_chunk = quack_group_gemm_same_nk(
        a=grad_gate_chunk,
        b=gate_weight,
        cumsum_M=cumsum,
        max_M=max_M,
        transpose_b=True,
        cu_seqlens_m=cu_seqlens,
    )
    del gate_weight

    up_weight = gate_up_proj[
        ..., hidden_start:hidden_end, intermediate_size + istart : intermediate_size + iend
    ].contiguous()
    grad_hidden_part = quack_group_gemm_same_nk(
        a=grad_up_chunk,
        b=up_weight,
        cumsum_M=cumsum,
        max_M=max_M,
        transpose_b=True,
        cu_seqlens_m=cu_seqlens,
    )
    del up_weight
    grad_hidden_chunk.add_(grad_hidden_part)
    del grad_hidden_part
    return grad_hidden_chunk


def _copy_gate_up_proj_grad_chunk(
    grad_gate_up_proj: torch.Tensor,
    token_chunk: torch.Tensor,
    grad_gate_chunk: torch.Tensor,
    grad_up_chunk: torch.Tensor,
    cumsum: torch.Tensor,
    intermediate_size: int,
    istart: int,
    iend: int,
    hidden_start: int,
    hidden_end: int,
    max_M: int,
    cu_seqlens: torch.Tensor,
    grad_gate_up_chunk: torch.Tensor | None = None,
) -> None:
    if _quack_full_intermediate_chunk(intermediate_size, istart, iend) and _quack_deepep_fused_gate_up_wgrad_enabled():
        owns_grad_gate_up_chunk = grad_gate_up_chunk is None
        if grad_gate_up_chunk is None:
            grad_gate_up_chunk = torch.cat((grad_gate_chunk, grad_up_chunk), dim=-1)
        grad_gate_up_proj_chunk = torch.empty(
            grad_gate_up_proj.shape[0],
            hidden_end - hidden_start,
            2 * intermediate_size,
            dtype=grad_gate_up_proj.dtype,
            device=grad_gate_up_proj.device,
        )
        quack_group_gemm_same_mn(
            a=token_chunk,
            b=grad_gate_up_chunk,
            c=grad_gate_up_proj_chunk,
            cumsum_K=cumsum,
            max_K=max_M,
            transpose_a=True,
            transpose_b=False,
            cu_seqlens_k=cu_seqlens,
        )
        grad_gate_up_proj[..., hidden_start:hidden_end, : 2 * intermediate_size].copy_(grad_gate_up_proj_chunk)
        if owns_grad_gate_up_chunk:
            del grad_gate_up_chunk
        del grad_gate_up_proj_chunk
        return

    grad_gate_proj_chunk = torch.empty(
        grad_gate_up_proj.shape[0],
        hidden_end - hidden_start,
        iend - istart,
        dtype=grad_gate_up_proj.dtype,
        device=grad_gate_up_proj.device,
    )
    quack_group_gemm_same_mn(
        a=token_chunk,
        b=grad_gate_chunk,
        c=grad_gate_proj_chunk,
        cumsum_K=cumsum,
        max_K=max_M,
        transpose_a=True,
        transpose_b=False,
        cu_seqlens_k=cu_seqlens,
    )
    grad_gate_up_proj[..., hidden_start:hidden_end, istart:iend].copy_(grad_gate_proj_chunk)
    del grad_gate_proj_chunk

    grad_up_proj_chunk = torch.empty(
        grad_gate_up_proj.shape[0],
        hidden_end - hidden_start,
        iend - istart,
        dtype=grad_gate_up_proj.dtype,
        device=grad_gate_up_proj.device,
    )
    quack_group_gemm_same_mn(
        a=token_chunk,
        b=grad_up_chunk,
        c=grad_up_proj_chunk,
        cumsum_K=cumsum,
        max_K=max_M,
        transpose_a=True,
        transpose_b=False,
        cu_seqlens_k=cu_seqlens,
    )
    grad_gate_up_proj[..., hidden_start:hidden_end, intermediate_size + istart : intermediate_size + iend].copy_(
        grad_up_proj_chunk
    )
    del grad_up_proj_chunk


class QuackMoeExpertsFunction(torch.autograd.Function):
    """Fused gate+up GEMM MoE compute. Mirrors ``TritonMoeExpertsFunction``."""

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh", "clamped_swiglu"})

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_proj=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
        swiglu_limit=0.0,
    ):
        check_hidden_act_supported(hidden_act, "quack", QuackMoeExpertsFunction.SUPPORTED_HIDDEN_ACTS)
        assert gate_up_proj is not None, "QuackMoeExpertsFunction requires a fused gate_up_proj"
        del activation_native
        ctx.hidden_act = hidden_act
        ctx.fp8_compute = fp8_compute
        ctx.fp8_grouped_backend = fp8_grouped_backend
        ctx.fp8_block_size = fp8_block_size
        ctx.has_gate_up_bias = gate_up_bias is not None
        ctx.has_down_bias = down_bias is not None
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        num_tokens = hidden_states.shape[0]
        top_k = expert_index.shape[1]

        scatter_output, scatter_index, cumsum_t = _scatter_and_cumsum(hidden_states, expert_index, num_experts)
        max_M = scatter_output.shape[0]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum_t)
        expert_ids = _expert_ids_from_cumsum(cumsum_t)

        gate_up_output = _group_gemm_same_nk(
            a=scatter_output,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        _add_expert_bias_by_ids_(gate_up_output, gate_up_bias, expert_ids)
        I = gate_up_output.shape[-1] // 2
        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]

        # Compose DSv4 gate-clamp (swiglu_limit) with the GPT-OSS-aware product
        # activation. For non-clamped_swiglu acts the product is silu(gate)*up.
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
        gated_activation = _moe_gate_activation_product(
            gate_for_activation, up_output, getattr(ctx, "hidden_act", "silu")
        )
        del gate_for_activation

        # Down projection (NO routing weights inside GEMM — apply after)
        down_output = _group_gemm_same_nk(
            a=gated_activation,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        del gated_activation
        _add_expert_bias_by_ids_(down_output, down_bias, expert_ids)

        # Unsort, apply routing weights, reshape+sum (deterministic accumulation)
        per_slot = down_output[scatter_index.flatten()].reshape(num_tokens, top_k, -1)
        output = (per_slot * gate_weights.unsqueeze(-1)).sum(dim=1)
        del down_output, per_slot

        ctx.save_for_backward(
            gate_weights,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            gate_up_proj,
            gate_up_bias if gate_up_bias is not None else gate_up_proj.new_empty(0),
            down_bias if down_bias is not None else down_proj.new_empty(0),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            gate_up_proj,
            gate_up_bias,
            down_bias,
        ) = ctx.saved_tensors
        # Recompute scattered routing weights for backward
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum_t)
        expert_ids = _expert_ids_from_cumsum(cumsum_t)

        # Recompute cheap intermediates
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, getattr(ctx, "swiglu_limit", 0.0))
        gated_activation = _moe_gate_activation_product(
            gate_for_activation, up_output, getattr(ctx, "hidden_act", "silu")
        )
        # gate_for_activation is kept alive for the activation backward below.
        gated_weighted = gated_activation * scattered_gate_weight

        grad_down_output = moe_scatter(grad_output, scatter_index)

        # FC2 dgrad
        fp8_compute = ctx.fp8_compute
        fp8_grouped_backend = ctx.fp8_grouped_backend
        fp8_block_size = ctx.fp8_block_size
        grad_gated_weighted = _group_gemm_same_nk(
            a=grad_down_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )

        # Routing weight gradient
        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        del gated_activation, grad_gated_weighted

        # Activation backward
        # GPT-OSS-aware activation backward on the (DSv4-clamped) gate, then chain
        # the swiglu_limit clamp gradient. _moe_gate_activation_backward_pair
        # recomputes the gate activation internally, so it is self-contained.
        grad_gate_for_activation, grad_up_output = _moe_gate_activation_backward_pair(
            grad_gated_activation,
            gate_for_activation,
            up_output,
            getattr(ctx, "hidden_act", "silu"),
        )
        grad_gate_output = _apply_swiglu_clamp_backward(
            grad_gate_for_activation, gate_output, getattr(ctx, "swiglu_limit", 0.0)
        )
        del grad_gated_activation, gate_output, up_output, gate_for_activation, grad_gate_for_activation

        # FC1 dgrad + wgrad — fused via gate_up_proj
        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output
        grad_gate_up_bias = None
        if ctx.has_gate_up_bias and gate_up_bias.requires_grad:
            grad_gate_up_bias = _sum_expert_bias_grad_by_ids(
                grad_gate_up_act,
                expert_ids,
                gate_up_proj.shape[0],
                gate_up_bias.dtype,
            )
        grad_scatter_output = _group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )

        # Run all same-NK dgrad calls before same-MN wgrad. The opt-in
        # DeepGEMM binding can reject calls made after Triton block-FP8 kernels
        # have run in this process.
        grad_down_scaled = grad_down_output * scattered_gate_weight
        grad_down_bias = None
        if ctx.has_down_bias and down_bias.requires_grad:
            grad_down_bias = _sum_expert_bias_grad_by_ids(
                grad_down_scaled,
                expert_ids,
                down_proj.shape[0],
                down_bias.dtype,
            )

        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            _group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del grad_down_output, grad_down_scaled, gated_weighted, scattered_gate_weight

        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            _group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del grad_gate_up_act, scatter_output

        # Unsort grad + reshape+sum (deterministic, matching forward)
        grad_hidden_states = (
            grad_scatter_output[scatter_index.flatten()]
            .reshape(hidden_states.shape[0], scatter_index.shape[1], -1)
            .sum(dim=1)
        )

        gradients = (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            None,  # gate_proj (unused — fused into gate_up_proj)
            None,  # up_proj   (unused — fused into gate_up_proj)
            grad_down_proj,  # down_proj
            grad_gate_up_proj,  # gate_up_proj
            None,  # hidden_act
            None,  # activation_native
            None,  # fp8_compute
            None,  # fp8_grouped_backend
            None,  # fp8_block_size
            grad_gate_up_bias,
            grad_down_bias,
            None,  # swiglu_limit
        )
        return gradients[: len(ctx.needs_input_grad)]


class QuackEPGroupGemm(torch.autograd.Function):
    """Memory-optimized EP expert GEMM. Recomputes cheap intermediates, explicit del."""

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh", "clamped_swiglu"})

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
        swiglu_limit=0.0,
    ):
        check_hidden_act_supported(hidden_act, "quack", QuackEPGroupGemm.SUPPORTED_HIDDEN_ACTS)
        ctx.hidden_act = hidden_act
        ctx.fp8_compute = fp8_compute
        ctx.fp8_grouped_backend = fp8_grouped_backend
        ctx.fp8_block_size = fp8_block_size
        ctx.has_gate_up_bias = gate_up_bias is not None
        ctx.has_down_bias = down_bias is not None
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        max_M = permute_tokens.shape[0]
        I = intermediate_size
        cu_seqlens = cumsum_to_cu_seqlens(cumsum)
        ctx.has_expert_scores = expert_scores is not None
        expert_ids = _expert_ids_from_cumsum(cumsum)
        trace_call_id, trace_enabled = _memory_trace_start("fwd")
        stage = "before_forward"

        # FP8 compute, GPT-OSS expert bias, and native activation require the
        # explicit separate-GEMM path; otherwise use the fused gated GEMM (fewer
        # kernels). swiglu_limit (DSv4 gate-clamp) composes with either.
        use_explicit_path = fp8_compute or activation_native or gate_up_bias is not None or down_bias is not None

        try:
            if _DEBUG_EP:
                return QuackEPGroupGemm._forward_debug(
                    ctx,
                    permute_tokens,
                    cumsum,
                    gate_up_proj,
                    down_proj,
                    I,
                    expert_scores,
                    max_M,
                    cu_seqlens,
                    fp8_compute,
                    fp8_grouped_backend,
                    fp8_block_size,
                    gate_up_bias,
                    down_bias,
                    activation_native,
                )

            stage = "gate_up_gated"
            if use_explicit_path:
                gate_up_output = _group_gemm_same_nk(
                    a=permute_tokens,
                    b=gate_up_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                _add_expert_bias_by_ids_(gate_up_output, gate_up_bias, expert_ids)
                gate_output = gate_up_output[..., :I]
                up_output = gate_up_output[..., I:]
                gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
                gated_output = _moe_gate_activation_forward(
                    gate_for_activation,
                    up_output,
                    getattr(ctx, "hidden_act", "silu"),
                    activation_native,
                )
                del gate_up_output, gate_output, up_output, gate_for_activation
            elif ctx.swiglu_limit > 0:
                gate_up_output = quack_group_gemm_same_nk(
                    a=permute_tokens,
                    b=gate_up_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens,
                )
                gate_output = gate_up_output[..., :I]
                up_output = gate_up_output[..., I:]
                gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
                gated_output = _moe_gate_activation(gate_for_activation, getattr(ctx, "hidden_act", "silu")) * up_output
                del gate_up_output, gate_output, up_output, gate_for_activation
            elif os.environ.get("XORL_QUACK_FUSED_GATED_INTERLEAVED", "0") == "1":
                # Quack's fused gated GEMM applies the activation over INTERLEAVED
                # (even=gate, odd=up) output columns. xorl MoE weights are
                # half-concatenated [gate; up], so this path silently produces
                # garbage (cos~0 vs reference; OPD loss 4.7 vs 0.54 — PTC-118R,
                # 2026-06-09) AND contradicts this op's own backward, which slices
                # half-concat. Opt-in only for genuinely interleaved-weight models.
                _, gated_output = quack_group_gemm_gated_same_nk(
                    a=permute_tokens,
                    b=gate_up_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    activation=_quack_gated_activation(getattr(ctx, "hidden_act", "silu")),
                    store_preact=False,
                    cu_seqlens_m=cu_seqlens,
                )
            else:
                # Explicit half-concat gating (pre-FP8-merge semantics): grouped
                # GEMM -> slice [gate; up] halves -> activation. Matches backward.
                gate_up_output = quack_group_gemm_same_nk(
                    a=permute_tokens,
                    b=gate_up_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens,
                )
                gate_output = gate_up_output[..., :I]
                up_output = gate_up_output[..., I:]
                gated_output = _moe_gate_activation_forward(
                    gate_output,
                    up_output,
                    getattr(ctx, "hidden_act", "silu"),
                    activation_native,
                )
                del gate_up_output, gate_output, up_output
            if trace_enabled:
                _memory_trace(
                    "fwd",
                    trace_call_id,
                    "after_gate_up_gated",
                    ("permute_tokens", permute_tokens),
                    ("gated_output", gated_output),
                    ("gate_up_proj", gate_up_proj),
                )

            # Down projection (NO expert_scores inside — apply after)
            stage = "down_gemm"
            down_output = _group_gemm_same_nk(
                a=gated_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=False,
                cu_seqlens_m=cu_seqlens,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
            if trace_enabled:
                _memory_trace(
                    "fwd",
                    trace_call_id,
                    "after_down_gemm",
                    ("gated_output", gated_output),
                    ("down_output", down_output),
                    ("down_proj", down_proj),
                )
            del gated_output
            _add_expert_bias_by_ids_(down_output, down_bias, expert_ids)

            if expert_scores is not None:
                stage = "score_weight"
                down_output.mul_(expert_scores.to(down_output.dtype).unsqueeze(-1))
                if trace_enabled:
                    _memory_trace(
                        "fwd",
                        trace_call_id,
                        "after_score_weight",
                        ("down_output", down_output),
                        ("expert_scores", expert_scores),
                    )

            if expert_scores is None:
                expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
            ctx.save_for_backward(
                permute_tokens,
                cumsum,
                gate_up_proj,
                down_proj,
                expert_scores,
                gate_up_bias if gate_up_bias is not None else gate_up_proj.new_empty(0),
                down_bias if down_bias is not None else down_proj.new_empty(0),
            )
            ctx.intermediate_size = I
            if trace_enabled:
                _memory_trace("fwd", trace_call_id, "saved_for_backward", ("down_output", down_output))
            return down_output
        except torch.OutOfMemoryError:
            _memory_trace("fwd", trace_call_id, f"OOM during {stage}", force=True)
            raise

    @staticmethod
    def _forward_debug(
        ctx,
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        I,
        expert_scores,
        max_M,
        cu_seqlens,
        fp8_compute,
        fp8_grouped_backend,
        fp8_block_size,
        gate_up_bias,
        down_bias,
        activation_native=False,
    ):
        """Instrumented forward with per-GEMM CUDA event timing."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        ctx.has_expert_scores = expert_scores is not None
        expert_ids = _expert_ids_from_cumsum(cumsum)

        ev[0].record()
        gate_up_output = _group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        _add_expert_bias_by_ids_(gate_up_output, gate_up_bias, expert_ids)
        ev[1].record()
        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, getattr(ctx, "swiglu_limit", 0.0))
        gated_output = _moe_gate_activation_forward(
            gate_for_activation,
            up_output,
            getattr(ctx, "hidden_act", "silu"),
            activation_native,
        )
        del gate_up_output, gate_output, up_output, gate_for_activation
        ev[2].record()

        # Down projection (NO expert_scores inside — apply after, matching normal path)
        down_output = _group_gemm_same_nk(
            a=gated_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        _add_expert_bias_by_ids_(down_output, down_bias, expert_ids)
        ev[3].record()
        del gated_output

        if expert_scores is not None:
            down_output.mul_(expert_scores.to(down_output.dtype).unsqueeze(-1))

        torch.cuda.synchronize()
        t_gate_up = ev[0].elapsed_time(ev[1])
        t_act = ev[1].elapsed_time(ev[2])
        t_down = ev[2].elapsed_time(ev[3])
        print(
            f"[QuackEP r{rank}] total_M={max_M} G={gate_up_proj.shape[0]} "
            f"K={gate_up_proj.shape[1]} N_gate_up={gate_up_proj.shape[2]} N_down={down_proj.shape[2]}\n"
            f"  cu_seqlens: dtype={cu_seqlens.dtype}, len={cu_seqlens.shape[0]}\n"
            f"  permute_tokens: stride={permute_tokens.stride()}, contiguous={permute_tokens.is_contiguous()}\n"
            f"  gate_up GEMM:  {t_gate_up:7.2f} ms\n"
            f"  silu+mul:      {t_act:7.2f} ms\n"
            f"  down GEMM:     {t_down:7.2f} ms\n"
            f"  total:         {t_gate_up + t_act + t_down:7.2f} ms",
            flush=True,
        )

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            expert_scores,
            gate_up_bias if gate_up_bias is not None else gate_up_proj.new_empty(0),
            down_bias if down_bias is not None else down_proj.new_empty(0),
        )
        ctx.intermediate_size = I
        return down_output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, cumsum, gate_up_proj, down_proj, expert_scores, gate_up_bias, down_bias = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = grad_output.shape[0]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum)
        fp8_compute = ctx.fp8_compute
        fp8_grouped_backend = ctx.fp8_grouped_backend
        fp8_block_size = ctx.fp8_block_size
        expert_ids = _expert_ids_from_cumsum(cumsum)

        # Recompute the gate/up activation instead of saving it from forward.
        # At 128K context this avoids keeping a large [tokens, 2I] tensor live
        # while allocating the down-projection output.
        gate_up_output = _group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        _add_expert_bias_by_ids_(gate_up_output, gate_up_bias if ctx.has_gate_up_bias else None, expert_ids)
        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]

        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, getattr(ctx, "swiglu_limit", 0.0))
        gated_output = _moe_gate_activation_product(gate_for_activation, up_output, getattr(ctx, "hidden_act", "silu"))
        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)

        # Forward was: out = down_GEMM(gated_output) * expert_scores
        # Skip the extra down-GEMM when expert_scores doesn't require a gradient
        # (e.g., train_router=False causes routing_weights to be detached upstream,
        # so ctx.needs_input_grad[5] is False and grad_expert_scores would be unused).
        grad_expert_scores = None
        if ctx.has_expert_scores and ctx.needs_input_grad[5]:
            down_output = _group_gemm_same_nk(
                a=gated_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=False,
                cu_seqlens_m=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
            _add_expert_bias_by_ids_(down_output, down_bias if ctx.has_down_bias else None, expert_ids)
            grad_expert_scores = (down_output * grad_output).sum(dim=-1).to(expert_scores_dtype)
            del down_output

        expert_scores_view = None
        if ctx.has_expert_scores:
            expert_scores_view = expert_scores.unsqueeze(-1)
            grad_output.mul_(expert_scores_view)
        grad_scaled = grad_output

        # dgrad FC2
        grad_gated_output = _group_gemm_same_nk(
            a=grad_scaled,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )

        # Activation backward
        grad_gate_for_activation, grad_up_output = _moe_gate_activation_backward_pair(
            grad_gated_output,
            gate_for_activation,
            up_output,
            getattr(ctx, "hidden_act", "silu"),
        )
        grad_gate_output = _apply_swiglu_clamp_backward(
            grad_gate_for_activation, gate_output, getattr(ctx, "swiglu_limit", 0.0)
        )
        del grad_gated_output, gate_output, up_output, gate_for_activation, grad_gate_for_activation

        # Fused dgrad FC1
        grad_gate_up_act = gate_up_output
        grad_gate_up_act[..., :I].copy_(grad_gate_output)
        grad_gate_up_act[..., I:].copy_(grad_up_output)
        del grad_gate_output, grad_up_output, gate_up_output
        grad_gate_up_bias = None
        if ctx.has_gate_up_bias and gate_up_bias.requires_grad:
            grad_gate_up_bias = _sum_expert_bias_grad_by_ids(
                grad_gate_up_act,
                expert_ids,
                gate_up_proj.shape[0],
                gate_up_bias.dtype,
            )
        grad_permute_tokens = _group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )

        # Run all same-NK dgrad calls before same-MN wgrad. The opt-in
        # DeepGEMM binding can reject calls made after Triton block-FP8 kernels
        # have run in this process.
        grad_down_bias = None
        if ctx.has_down_bias and down_bias.requires_grad:
            grad_down_bias = _sum_expert_bias_grad_by_ids(
                grad_scaled,
                expert_ids,
                down_proj.shape[0],
                down_bias.dtype,
            )

        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            _group_gemm_same_mn(
                a=gated_output,
                b=grad_scaled,
                c=grad_down_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del gated_output, grad_scaled
        if expert_scores_view is not None:
            grad_output.div_(expert_scores_view)
            del expert_scores_view

        # Fused wgrad FC1
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            _group_gemm_same_mn(
                a=permute_tokens,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del grad_gate_up_act

        gradients = (
            grad_permute_tokens,
            None,  # cumsum
            grad_gate_up_proj,
            grad_down_proj,
            None,  # intermediate_size
            grad_expert_scores,
            None,  # hidden_act
            None,  # activation_native
            None,  # fp8_compute
            None,  # fp8_grouped_backend
            None,  # fp8_block_size
            grad_gate_up_bias,
            grad_down_bias,
            None,  # swiglu_limit (non-differentiable float; was missing -> backward returned 13/14 grads)
        )
        return gradients[: len(ctx.needs_input_grad)]


class QuackEPDeepEPCombine(torch.autograd.Function):
    """Quack EP compute fused with chunked DeepEP combine.

    The regular EP path materializes ``expert_output`` as ``[num_valid, hidden]``
    before DeepEP scatter-add/combine. GLM-5 128K can route enough tokens to one
    rank that this allocation alone is several GiB. This function streams the
    down projection by hidden-dimension chunks and immediately combines each
    chunk, so the full expert-order hidden tensor never exists.
    """

    SUPPORTED_HIDDEN_ACTS = QuackEPGroupGemm.SUPPORTED_HIDDEN_ACTS

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores,
        buffer,
        dispatch_ctx,
        async_combine,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
    ):
        del async_combine  # Chunk copies require each combine result to be stream-visible.
        check_hidden_act_supported(hidden_act, "quack", QuackEPDeepEPCombine.SUPPORTED_HIDDEN_ACTS)
        ctx.hidden_act = hidden_act
        ctx.fp8_compute = fp8_compute
        ctx.fp8_grouped_backend = fp8_grouped_backend
        ctx.fp8_block_size = fp8_block_size
        ctx.has_gate_up_bias = gate_up_bias is not None
        ctx.has_down_bias = down_bias is not None

        max_M = permute_tokens.shape[0]
        I = intermediate_size
        hidden_dim = down_proj.shape[2]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum)
        expert_ids = _expert_ids_from_cumsum(cumsum)
        has_expert_scores = expert_scores is not None
        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(max_M)

        hidden_chunk_size = _quack_deepep_hidden_chunk_size(hidden_dim)
        intermediate_chunk_size = _quack_deepep_intermediate_chunk_size(I)
        combined_output = None
        for istart in range(0, I, intermediate_chunk_size):
            iend = min(istart + intermediate_chunk_size, I)
            if max_M == 0:
                gated_chunk = permute_tokens.new_empty(0, iend - istart)
            else:
                gate_proj = gate_up_proj[..., istart:iend].contiguous()
                gate_output = _group_gemm_same_nk(
                    a=permute_tokens,
                    b=gate_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del gate_proj
                _add_expert_bias_by_ids_(gate_output, gate_up_bias, expert_ids, istart, iend)
                up_proj = gate_up_proj[..., I + istart : I + iend].contiguous()
                up_output = _group_gemm_same_nk(
                    a=permute_tokens,
                    b=up_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del up_proj
                _add_expert_bias_by_ids_(up_output, gate_up_bias, expert_ids, I + istart, I + iend)
                gated_chunk = _moe_gate_activation_forward(gate_output, up_output, hidden_act, activation_native)
                del gate_output, up_output

            score_view = expert_scores.to(gated_chunk.dtype).unsqueeze(-1) if has_expert_scores else None
            for start in range(0, hidden_dim, hidden_chunk_size):
                end = min(start + hidden_chunk_size, hidden_dim)
                if max_M == 0:
                    down_chunk = permute_tokens.new_empty(0, end - start)
                else:
                    down_weight = down_proj[..., istart:iend, start:end].contiguous()
                    down_chunk = _group_gemm_same_nk(
                        a=gated_chunk,
                        b=down_weight,
                        cumsum_M=cumsum,
                        max_M=max_M,
                        transpose_b=False,
                        cu_seqlens_m=cu_seqlens,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    del down_weight
                    if istart == 0:
                        _add_expert_bias_by_ids_(down_chunk, down_bias, expert_ids, start, end)
                    if score_view is not None:
                        down_chunk.mul_(score_view)

                gather_chunk = _scatter_expert_chunk_to_recv(
                    down_chunk,
                    dispatch_ctx.permuted_indices,
                    dispatch_ctx.num_recv_tokens,
                    dispatch_ctx.dtype,
                )
                del down_chunk
                combined_chunk = _deepep_combine_chunk(buffer, gather_chunk, dispatch_ctx.handle)
                del gather_chunk
                if combined_output is None:
                    combined_output = permute_tokens.new_zeros(combined_chunk.shape[0], hidden_dim)
                combined_output[:, start:end].add_(combined_chunk)
                del combined_chunk
            del gated_chunk

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            expert_scores,
            dispatch_ctx.permuted_indices,
            gate_up_bias if gate_up_bias is not None else gate_up_proj.new_empty(0),
            down_bias if down_bias is not None else down_proj.new_empty(0),
        )
        ctx.buffer = buffer
        ctx.handle = dispatch_ctx.handle
        ctx.num_recv_tokens = dispatch_ctx.num_recv_tokens
        ctx.intermediate_size = I
        ctx.has_expert_scores = has_expert_scores
        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        (
            permute_tokens,
            cumsum,
            gate_up_proj,
            down_proj,
            expert_scores,
            permuted_indices,
            gate_up_bias,
            down_bias,
        ) = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = permute_tokens.shape[0]
        hidden_dim = down_proj.shape[2]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum)
        fp8_compute = ctx.fp8_compute
        fp8_grouped_backend = ctx.fp8_grouped_backend
        fp8_block_size = ctx.fp8_block_size
        expert_ids = _expert_ids_from_cumsum(cumsum)

        hidden_chunk_size = _quack_deepep_hidden_chunk_size(hidden_dim)
        intermediate_chunk_size = _quack_deepep_intermediate_chunk_size(I)

        grad_permute_tokens = None
        grad_gate_up_proj = torch.empty_like(gate_up_proj) if gate_up_proj.requires_grad else None
        grad_down_proj = torch.empty_like(down_proj) if down_proj.requires_grad else None
        grad_gate_up_bias = (
            torch.empty_like(gate_up_bias) if ctx.has_gate_up_bias and gate_up_bias.requires_grad else None
        )
        grad_down_bias = torch.empty_like(down_bias) if ctx.has_down_bias and down_bias.requires_grad else None
        grad_expert_scores = None
        if ctx.has_expert_scores and ctx.needs_input_grad[5]:
            grad_expert_scores = torch.zeros_like(expert_scores, dtype=torch.float32)
        score_view = expert_scores.to(grad_output.dtype).unsqueeze(-1) if ctx.has_expert_scores else None

        for istart in range(0, I, intermediate_chunk_size):
            iend = min(istart + intermediate_chunk_size, I)
            if max_M == 0:
                gate_output = permute_tokens.new_empty(0, iend - istart)
                up_output = permute_tokens.new_empty(0, iend - istart)
            else:
                gate_proj = gate_up_proj[..., istart:iend].contiguous()
                gate_output = _group_gemm_same_nk(
                    a=permute_tokens,
                    b=gate_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del gate_proj
                _add_expert_bias_by_ids_(
                    gate_output, gate_up_bias if ctx.has_gate_up_bias else None, expert_ids, istart, iend
                )
                up_proj = gate_up_proj[..., I + istart : I + iend].contiguous()
                up_output = _group_gemm_same_nk(
                    a=permute_tokens,
                    b=up_proj,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=False,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del up_proj
                _add_expert_bias_by_ids_(
                    up_output,
                    gate_up_bias if ctx.has_gate_up_bias else None,
                    expert_ids,
                    I + istart,
                    I + iend,
                )

            gated_chunk = _moe_gate_activation_product(gate_output, up_output, getattr(ctx, "hidden_act", "silu"))

            grad_gated_chunk = torch.zeros(
                max_M,
                iend - istart,
                dtype=permute_tokens.dtype,
                device=permute_tokens.device,
            )
            for start in range(0, hidden_dim, hidden_chunk_size):
                end = min(start + hidden_chunk_size, hidden_dim)
                grad_gather = _deepep_dispatch_grad_chunk(
                    ctx.buffer,
                    grad_output[:, start:end],
                    ctx.handle,
                )
                if grad_gather.dtype != grad_output.dtype:
                    grad_gather = grad_gather.to(grad_output.dtype)
                _debug_assert_finite("bwd/grad_gather_dispatch", grad_gather, f"istart={istart} start={start}")
                grad_expert_chunk = grad_gather.index_select(0, permuted_indices)
                del grad_gather

                down_weight = down_proj[..., istart:iend, start:end].contiguous()
                _debug_assert_finite("bwd/down_weight", down_weight, f"istart={istart} start={start}")
                if grad_expert_scores is not None:
                    down_chunk = _group_gemm_same_nk(
                        a=gated_chunk,
                        b=down_weight,
                        cumsum_M=cumsum,
                        max_M=max_M,
                        transpose_b=False,
                        cu_seqlens_m=cu_seqlens_m,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    if istart == 0:
                        _add_expert_bias_by_ids_(
                            down_chunk, down_bias if ctx.has_down_bias else None, expert_ids, start, end
                        )
                    grad_expert_scores.add_((down_chunk.float() * grad_expert_chunk.float()).sum(dim=-1))
                    del down_chunk

                if score_view is not None:
                    grad_expert_chunk.mul_(score_view)

                if grad_down_bias is not None and istart == 0:
                    grad_down_bias[:, start:end].copy_(
                        _sum_expert_bias_grad_by_ids(
                            grad_expert_chunk,
                            expert_ids,
                            down_proj.shape[0],
                            down_bias.dtype,
                        )
                    )

                if grad_down_proj is not None:
                    grad_down_chunk = torch.empty(
                        down_proj.shape[0],
                        iend - istart,
                        end - start,
                        dtype=down_proj.dtype,
                        device=down_proj.device,
                    )
                    _group_gemm_same_mn(
                        a=gated_chunk,
                        b=grad_expert_chunk,
                        c=grad_down_chunk,
                        cumsum_K=cumsum,
                        max_K=max_M,
                        transpose_a=True,
                        transpose_b=False,
                        cu_seqlens_k=cu_seqlens_m,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    _debug_assert_finite("bwd/grad_down_chunk_wgrad", grad_down_chunk, f"istart={istart} start={start}")
                    grad_down_proj[..., istart:iend, start:end].copy_(grad_down_chunk)
                    del grad_down_chunk

                grad_gated_part = _group_gemm_same_nk(
                    a=grad_expert_chunk,
                    b=down_weight,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=True,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                _debug_assert_finite("bwd/grad_gated_part", grad_gated_part, f"istart={istart} start={start}")
                grad_gated_chunk.add_(grad_gated_part)
                del grad_gated_part, grad_expert_chunk, down_weight
            del gated_chunk

            grad_gate_chunk, grad_up_chunk = _moe_gate_activation_backward_pair(
                grad_gated_chunk,
                gate_output,
                up_output,
                getattr(ctx, "hidden_act", "silu"),
            )
            _debug_assert_finite("bwd/grad_gate_chunk", grad_gate_chunk, f"istart={istart}")
            _debug_assert_finite("bwd/grad_up_chunk", grad_up_chunk, f"istart={istart}")
            del grad_gated_chunk, gate_output, up_output

            if grad_gate_up_bias is not None:
                grad_gate_up_bias[:, istart:iend].copy_(
                    _sum_expert_bias_grad_by_ids(
                        grad_gate_chunk,
                        expert_ids,
                        gate_up_proj.shape[0],
                        gate_up_bias.dtype,
                    )
                )
                grad_gate_up_bias[:, I + istart : I + iend].copy_(
                    _sum_expert_bias_grad_by_ids(
                        grad_up_chunk,
                        expert_ids,
                        gate_up_proj.shape[0],
                        gate_up_bias.dtype,
                    )
                )

            if grad_permute_tokens is None:
                grad_permute_tokens = torch.empty(
                    max_M,
                    hidden_dim,
                    dtype=permute_tokens.dtype,
                    device=permute_tokens.device,
                )
            for start in range(0, hidden_dim, hidden_chunk_size):
                end = min(start + hidden_chunk_size, hidden_dim)
                gate_weight = gate_up_proj[..., start:end, istart:iend].contiguous()
                grad_hidden_chunk = _group_gemm_same_nk(
                    a=grad_gate_chunk,
                    b=gate_weight,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=True,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del gate_weight
                if istart == 0:
                    grad_permute_tokens[:, start:end].copy_(grad_hidden_chunk)
                else:
                    grad_permute_tokens[:, start:end].add_(grad_hidden_chunk)
                del grad_hidden_chunk

                up_weight = gate_up_proj[..., start:end, I + istart : I + iend].contiguous()
                grad_hidden_chunk = _group_gemm_same_nk(
                    a=grad_up_chunk,
                    b=up_weight,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=True,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del up_weight
                grad_permute_tokens[:, start:end].add_(grad_hidden_chunk)
                del grad_hidden_chunk

            if grad_gate_up_proj is not None:
                grad_gate_proj_chunk = torch.empty(
                    gate_up_proj.shape[0],
                    hidden_dim,
                    iend - istart,
                    dtype=gate_up_proj.dtype,
                    device=gate_up_proj.device,
                )
                _group_gemm_same_mn(
                    a=permute_tokens,
                    b=grad_gate_chunk,
                    c=grad_gate_proj_chunk,
                    cumsum_K=cumsum,
                    max_K=max_M,
                    transpose_a=True,
                    transpose_b=False,
                    cu_seqlens_k=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                grad_gate_up_proj[..., istart:iend].copy_(grad_gate_proj_chunk)
                del grad_gate_proj_chunk

                grad_up_proj_chunk = torch.empty(
                    gate_up_proj.shape[0],
                    hidden_dim,
                    iend - istart,
                    dtype=gate_up_proj.dtype,
                    device=gate_up_proj.device,
                )
                _group_gemm_same_mn(
                    a=permute_tokens,
                    b=grad_up_chunk,
                    c=grad_up_proj_chunk,
                    cumsum_K=cumsum,
                    max_K=max_M,
                    transpose_a=True,
                    transpose_b=False,
                    cu_seqlens_k=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                grad_gate_up_proj[..., I + istart : I + iend].copy_(grad_up_proj_chunk)
                del grad_up_proj_chunk
            del grad_gate_chunk, grad_up_chunk

        if grad_expert_scores is not None:
            grad_expert_scores = grad_expert_scores.to(expert_scores.dtype)
        if grad_permute_tokens is None:
            grad_permute_tokens = torch.empty_like(permute_tokens)

        return (
            grad_permute_tokens,
            None,  # cumsum
            grad_gate_up_proj,
            grad_down_proj,
            None,  # intermediate_size
            grad_expert_scores,
            None,  # buffer
            None,  # dispatch_ctx
            None,  # async_combine
            None,  # hidden_act
            None,  # activation_native
            None,  # fp8_compute
            None,  # fp8_grouped_backend
            None,  # fp8_block_size
            grad_gate_up_bias,
            grad_down_bias,
        )


class QuackEPDeepEPNoPermute(torch.autograd.Function):
    """Quack EP compute for DeepEP recv-order tokens.

    DeepEP dispatch can return one recv-order token matrix plus routing
    metadata. This function gathers bounded hidden chunks from that matrix,
    runs Quack group GEMMs in expert order, and immediately scatters/combines
    down-projection chunks. It avoids the full ``recv_x.index_select`` tensor
    that otherwise dominates memory at GLM-5 128K.
    """

    SUPPORTED_HIDDEN_ACTS = QuackEPGroupGemm.SUPPORTED_HIDDEN_ACTS

    @staticmethod
    def forward(
        ctx,
        recv_x,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores,
        buffer,
        dispatch_ctx,
        async_combine,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
    ):
        del async_combine  # Chunk copies require each combine result to be stream-visible.
        check_hidden_act_supported(hidden_act, "quack", QuackEPDeepEPNoPermute.SUPPORTED_HIDDEN_ACTS)
        ctx.hidden_act = hidden_act
        ctx.fp8_compute = fp8_compute
        ctx.fp8_grouped_backend = fp8_grouped_backend
        ctx.fp8_block_size = fp8_block_size
        ctx.has_gate_up_bias = gate_up_bias is not None
        ctx.has_down_bias = down_bias is not None

        I = intermediate_size
        hidden_dim = down_proj.shape[2]
        max_M = dispatch_ctx.permuted_indices.shape[0]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum)
        expert_ids = _expert_ids_from_cumsum(cumsum)
        has_expert_scores = expert_scores is not None
        if expert_scores is None:
            expert_scores = recv_x.new_ones(max_M)

        hidden_chunk_size = _quack_deepep_hidden_chunk_size(hidden_dim)
        intermediate_chunk_size = _quack_deepep_intermediate_chunk_size(I)
        combined_output = None
        for istart in range(0, I, intermediate_chunk_size):
            iend = min(istart + intermediate_chunk_size, I)
            gate_output, up_output = _quack_gate_up_chunk_from_recv(
                recv_x,
                dispatch_ctx.permuted_indices,
                cumsum,
                gate_up_proj,
                I,
                istart,
                iend,
                hidden_chunk_size,
                cu_seqlens,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
                gate_up_bias=gate_up_bias,
                expert_ids=expert_ids,
            )
            gated_chunk = _moe_gate_activation_forward(gate_output, up_output, hidden_act, activation_native)
            del gate_output, up_output

            score_view = expert_scores.to(gated_chunk.dtype).unsqueeze(-1) if has_expert_scores else None
            for start in range(0, hidden_dim, hidden_chunk_size):
                end = min(start + hidden_chunk_size, hidden_dim)
                if max_M == 0:
                    down_chunk = recv_x.new_empty(0, end - start)
                else:
                    down_weight = down_proj[..., istart:iend, start:end].contiguous()
                    down_chunk = _group_gemm_same_nk(
                        a=gated_chunk,
                        b=down_weight,
                        cumsum_M=cumsum,
                        max_M=max_M,
                        transpose_b=False,
                        cu_seqlens_m=cu_seqlens,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    del down_weight
                    if istart == 0:
                        _add_expert_bias_by_ids_(down_chunk, down_bias, expert_ids, start, end)
                    if score_view is not None:
                        down_chunk.mul_(score_view)

                gather_chunk = _scatter_expert_chunk_to_recv(
                    down_chunk,
                    dispatch_ctx.permuted_indices,
                    dispatch_ctx.num_recv_tokens,
                    dispatch_ctx.dtype,
                )
                del down_chunk
                combined_chunk = _deepep_combine_chunk(buffer, gather_chunk, dispatch_ctx.handle)
                del gather_chunk
                if combined_output is None:
                    combined_output = recv_x.new_zeros(combined_chunk.shape[0], hidden_dim)
                combined_output[:, start:end].add_(combined_chunk)
                del combined_chunk
            del gated_chunk

        ctx.save_for_backward(
            recv_x,
            cumsum,
            gate_up_proj,
            down_proj,
            expert_scores,
            dispatch_ctx.permuted_indices,
            gate_up_bias if gate_up_bias is not None else gate_up_proj.new_empty(0),
            down_bias if down_bias is not None else down_proj.new_empty(0),
        )
        ctx.buffer = buffer
        ctx.handle = dispatch_ctx.handle
        ctx.intermediate_size = I
        ctx.has_expert_scores = has_expert_scores
        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        (
            recv_x,
            cumsum,
            gate_up_proj,
            down_proj,
            expert_scores,
            permuted_indices,
            gate_up_bias,
            down_bias,
        ) = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = permuted_indices.shape[0]
        hidden_dim = down_proj.shape[2]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum)
        fp8_compute = ctx.fp8_compute
        fp8_grouped_backend = ctx.fp8_grouped_backend
        fp8_block_size = ctx.fp8_block_size
        expert_ids = _expert_ids_from_cumsum(cumsum)

        hidden_chunk_size = _quack_deepep_hidden_chunk_size(hidden_dim)
        intermediate_chunk_size = _quack_deepep_intermediate_chunk_size(I)

        _debug_assert_finite("bwd-entry/grad_output", grad_output)
        _debug_assert_finite("bwd-entry/recv_x", recv_x)
        _debug_assert_finite("bwd-entry/gate_up_proj", gate_up_proj)
        _debug_assert_finite("bwd-entry/down_proj", down_proj)
        grad_recv_x = torch.zeros_like(recv_x)
        grad_gate_up_proj = torch.empty_like(gate_up_proj) if gate_up_proj.requires_grad else None
        grad_down_proj = torch.empty_like(down_proj) if down_proj.requires_grad else None
        grad_gate_up_bias = (
            torch.empty_like(gate_up_bias) if ctx.has_gate_up_bias and gate_up_bias.requires_grad else None
        )
        grad_down_bias = torch.empty_like(down_bias) if ctx.has_down_bias and down_bias.requires_grad else None
        grad_expert_scores = None
        if ctx.has_expert_scores and ctx.needs_input_grad[5]:
            grad_expert_scores = torch.zeros_like(expert_scores, dtype=torch.float32)
        score_view = expert_scores.to(grad_output.dtype).unsqueeze(-1) if ctx.has_expert_scores else None

        for istart in range(0, I, intermediate_chunk_size):
            iend = min(istart + intermediate_chunk_size, I)
            gate_output, up_output = _quack_gate_up_chunk_from_recv(
                recv_x,
                permuted_indices,
                cumsum,
                gate_up_proj,
                I,
                istart,
                iend,
                hidden_chunk_size,
                cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
                gate_up_bias=gate_up_bias if ctx.has_gate_up_bias else None,
                expert_ids=expert_ids,
            )
            _debug_assert_finite("bwd-recompute/gate_output", gate_output, f"istart={istart}")
            _debug_assert_finite("bwd-recompute/up_output", up_output, f"istart={istart}")

            gated_chunk = _moe_gate_activation_product(gate_output, up_output, getattr(ctx, "hidden_act", "silu"))
            _debug_assert_finite("bwd-recompute/gated_chunk", gated_chunk, f"istart={istart}")

            grad_gated_chunk = torch.zeros(
                max_M,
                iend - istart,
                dtype=recv_x.dtype,
                device=recv_x.device,
            )
            for start in range(0, hidden_dim, hidden_chunk_size):
                end = min(start + hidden_chunk_size, hidden_dim)
                grad_gather = _deepep_dispatch_grad_chunk(
                    ctx.buffer,
                    grad_output[:, start:end],
                    ctx.handle,
                )
                if grad_gather.dtype != grad_output.dtype:
                    grad_gather = grad_gather.to(grad_output.dtype)
                _debug_assert_finite("bwd/grad_gather_dispatch", grad_gather, f"istart={istart} start={start}")
                grad_expert_chunk = grad_gather.index_select(0, permuted_indices)
                del grad_gather

                down_weight = down_proj[..., istart:iend, start:end].contiguous()
                _debug_assert_finite("bwd/down_weight", down_weight, f"istart={istart} start={start}")
                if grad_expert_scores is not None:
                    down_chunk = _group_gemm_same_nk(
                        a=gated_chunk,
                        b=down_weight,
                        cumsum_M=cumsum,
                        max_M=max_M,
                        transpose_b=False,
                        cu_seqlens_m=cu_seqlens_m,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    if istart == 0:
                        _add_expert_bias_by_ids_(
                            down_chunk, down_bias if ctx.has_down_bias else None, expert_ids, start, end
                        )
                    grad_expert_scores.add_((down_chunk.float() * grad_expert_chunk.float()).sum(dim=-1))
                    del down_chunk

                if score_view is not None:
                    grad_expert_chunk.mul_(score_view)

                if grad_down_bias is not None and istart == 0:
                    grad_down_bias[:, start:end].copy_(
                        _sum_expert_bias_grad_by_ids(
                            grad_expert_chunk,
                            expert_ids,
                            down_proj.shape[0],
                            down_bias.dtype,
                        )
                    )

                if grad_down_proj is not None:
                    grad_down_chunk = torch.empty(
                        down_proj.shape[0],
                        iend - istart,
                        end - start,
                        dtype=down_proj.dtype,
                        device=down_proj.device,
                    )
                    _group_gemm_same_mn(
                        a=gated_chunk,
                        b=grad_expert_chunk,
                        c=grad_down_chunk,
                        cumsum_K=cumsum,
                        max_K=max_M,
                        transpose_a=True,
                        transpose_b=False,
                        cu_seqlens_k=cu_seqlens_m,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    _debug_assert_finite("bwd/grad_down_chunk_wgrad", grad_down_chunk, f"istart={istart} start={start}")
                    grad_down_proj[..., istart:iend, start:end].copy_(grad_down_chunk)
                    del grad_down_chunk

                grad_gated_part = _group_gemm_same_nk(
                    a=grad_expert_chunk,
                    b=down_weight,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=True,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                _debug_assert_finite("bwd/grad_gated_part", grad_gated_part, f"istart={istart} start={start}")
                grad_gated_chunk.add_(grad_gated_part)
                del grad_gated_part, grad_expert_chunk, down_weight
            del gated_chunk

            grad_gate_chunk, grad_up_chunk = _moe_gate_activation_backward_pair(
                grad_gated_chunk,
                gate_output,
                up_output,
                getattr(ctx, "hidden_act", "silu"),
            )
            _debug_assert_finite("bwd/grad_gate_chunk", grad_gate_chunk, f"istart={istart}")
            _debug_assert_finite("bwd/grad_up_chunk", grad_up_chunk, f"istart={istart}")
            del grad_gated_chunk, gate_output, up_output

            if grad_gate_up_bias is not None:
                grad_gate_up_bias[:, istart:iend].copy_(
                    _sum_expert_bias_grad_by_ids(
                        grad_gate_chunk,
                        expert_ids,
                        gate_up_proj.shape[0],
                        gate_up_bias.dtype,
                    )
                )
                grad_gate_up_bias[:, I + istart : I + iend].copy_(
                    _sum_expert_bias_grad_by_ids(
                        grad_up_chunk,
                        expert_ids,
                        gate_up_proj.shape[0],
                        gate_up_bias.dtype,
                    )
                )

            for start in range(0, hidden_dim, hidden_chunk_size):
                end = min(start + hidden_chunk_size, hidden_dim)
                gate_weight = gate_up_proj[..., start:end, istart:iend].contiguous()
                grad_hidden_chunk = _group_gemm_same_nk(
                    a=grad_gate_chunk,
                    b=gate_weight,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=True,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del gate_weight

                up_weight = gate_up_proj[..., start:end, I + istart : I + iend].contiguous()
                grad_hidden_part = _group_gemm_same_nk(
                    a=grad_up_chunk,
                    b=up_weight,
                    cumsum_M=cumsum,
                    max_M=max_M,
                    transpose_b=True,
                    cu_seqlens_m=cu_seqlens_m,
                    fp8_compute=fp8_compute,
                    fp8_grouped_backend=fp8_grouped_backend,
                    fp8_block_size=fp8_block_size,
                )
                del up_weight
                grad_hidden_chunk.add_(grad_hidden_part)
                del grad_hidden_part
                _debug_assert_finite("bwd/grad_hidden_chunk", grad_hidden_chunk, f"istart={istart} start={start}")
                grad_recv_x[:, start:end].index_add_(0, permuted_indices, grad_hidden_chunk)
                del grad_hidden_chunk

            if grad_gate_up_proj is not None:
                for start in range(0, hidden_dim, hidden_chunk_size):
                    end = min(start + hidden_chunk_size, hidden_dim)
                    token_chunk = _gather_recv_hidden_chunk(recv_x, permuted_indices, start, end)

                    grad_gate_proj_chunk = torch.empty(
                        gate_up_proj.shape[0],
                        end - start,
                        iend - istart,
                        dtype=gate_up_proj.dtype,
                        device=gate_up_proj.device,
                    )
                    _group_gemm_same_mn(
                        a=token_chunk,
                        b=grad_gate_chunk,
                        c=grad_gate_proj_chunk,
                        cumsum_K=cumsum,
                        max_K=max_M,
                        transpose_a=True,
                        transpose_b=False,
                        cu_seqlens_k=cu_seqlens_m,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    _debug_assert_finite(
                        "bwd/grad_gate_proj_chunk_wgrad", grad_gate_proj_chunk, f"istart={istart} start={start}"
                    )
                    grad_gate_up_proj[..., start:end, istart:iend].copy_(grad_gate_proj_chunk)
                    del grad_gate_proj_chunk

                    grad_up_proj_chunk = torch.empty(
                        gate_up_proj.shape[0],
                        end - start,
                        iend - istart,
                        dtype=gate_up_proj.dtype,
                        device=gate_up_proj.device,
                    )
                    _group_gemm_same_mn(
                        a=token_chunk,
                        b=grad_up_chunk,
                        c=grad_up_proj_chunk,
                        cumsum_K=cumsum,
                        max_K=max_M,
                        transpose_a=True,
                        transpose_b=False,
                        cu_seqlens_k=cu_seqlens_m,
                        fp8_compute=fp8_compute,
                        fp8_grouped_backend=fp8_grouped_backend,
                        fp8_block_size=fp8_block_size,
                    )
                    grad_gate_up_proj[..., start:end, I + istart : I + iend].copy_(grad_up_proj_chunk)
                    del token_chunk, grad_up_proj_chunk
            del grad_gate_chunk, grad_up_chunk

        if grad_expert_scores is not None:
            grad_expert_scores = grad_expert_scores.to(expert_scores.dtype)

        return (
            grad_recv_x,
            None,  # cumsum
            grad_gate_up_proj,
            grad_down_proj,
            None,  # intermediate_size
            grad_expert_scores,
            None,  # buffer
            None,  # dispatch_ctx
            None,  # async_combine
            None,  # hidden_act
            None,  # activation_native
            None,  # fp8_compute
            None,  # fp8_grouped_backend
            None,  # fp8_block_size
            grad_gate_up_bias,
            grad_down_bias,
        )


class QuackTPMoeExpertsFunction(torch.autograd.Function):
    """Memory-optimized TP expert function. Recomputes cheap intermediates, explicit del + all-reduce."""

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh", "clamped_swiglu"})

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        tp_group,
        hidden_act="silu",
        activation_native=False,
        fp8_compute=False,
        fp8_grouped_backend="triton_grouped",
        fp8_block_size=128,
        gate_up_bias=None,
        down_bias=None,
        swiglu_limit=0.0,
    ):
        check_hidden_act_supported(hidden_act, "quack", QuackTPMoeExpertsFunction.SUPPORTED_HIDDEN_ACTS)
        del activation_native
        ctx.hidden_act = hidden_act
        ctx.fp8_compute = fp8_compute
        ctx.fp8_grouped_backend = fp8_grouped_backend
        ctx.fp8_block_size = fp8_block_size
        ctx.has_gate_up_bias = gate_up_bias is not None
        ctx.has_down_bias = down_bias is not None
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        scatter_output, scatter_index, cumsum_t = _scatter_and_cumsum(hidden_states, expert_index, num_experts)
        max_M = scatter_output.shape[0]
        cu_seqlens = cumsum_to_cu_seqlens(cumsum_t)
        expert_ids = _expert_ids_from_cumsum(cumsum_t)

        gate_output = _group_gemm_same_nk(
            a=scatter_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        _add_expert_bias_by_ids_(gate_output, gate_up_bias, expert_ids, 0, gate_proj.shape[-1])
        up_output = _group_gemm_same_nk(
            a=scatter_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        _add_expert_bias_by_ids_(up_output, gate_up_bias, expert_ids, gate_proj.shape[-1], 2 * gate_proj.shape[-1])
        del scatter_output

        # Compose DSv4 gate-clamp (swiglu_limit) with the GPT-OSS-aware product
        # activation. For non-clamped_swiglu acts the product is silu(gate)*up.
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
        gated_activation = _moe_gate_activation_product(
            gate_for_activation, up_output, getattr(ctx, "hidden_act", "silu")
        )
        del gate_for_activation

        scattered_gate_weight = torch.empty_like(gate_weights.reshape(-1, 1))
        scattered_gate_weight[scatter_index.flatten()] = gate_weights.reshape(-1, 1)
        gated_weighted = gated_activation * scattered_gate_weight
        del gated_activation

        down_output = _group_gemm_same_nk(
            a=gated_weighted,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=False,
            cu_seqlens_m=cu_seqlens,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        del gated_weighted
        dist.all_reduce(down_output, group=tp_group)
        if down_bias is not None:
            down_output.add_(down_bias.to(down_output.dtype).index_select(0, expert_ids) * scattered_gate_weight)
        output = moe_gather(down_output, scatter_index).reshape(hidden_states.shape)
        del down_output

        ctx.tp_group = tp_group
        ctx.save_for_backward(
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
            gate_up_bias if gate_up_bias is not None else gate_proj.new_empty(0),
            down_bias if down_bias is not None else down_proj.new_empty(0),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
            gate_up_bias,
            down_bias,
        ) = ctx.saved_tensors
        tp_group = ctx.tp_group
        fp8_compute = ctx.fp8_compute
        fp8_grouped_backend = ctx.fp8_grouped_backend
        fp8_block_size = ctx.fp8_block_size
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]
        cu_seqlens_m = cumsum_to_cu_seqlens(cumsum_t)
        expert_ids = _expert_ids_from_cumsum(cumsum_t)

        # Recompute cheap intermediates (avoids saving them)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, getattr(ctx, "swiglu_limit", 0.0))
        gated_activation = _moe_gate_activation_product(
            gate_for_activation, up_output, getattr(ctx, "hidden_act", "silu")
        )
        # gate_for_activation is kept alive for the activation backward below.
        gated_weighted = gated_activation * scattered_gate_weight

        grad_down_output = moe_scatter(grad_output, scatter_index)
        grad_down_bias = None
        if ctx.has_down_bias and down_bias.requires_grad:
            grad_down_bias = _sum_expert_bias_grad_by_ids(
                grad_down_output * scattered_gate_weight,
                expert_ids,
                down_proj.shape[0],
                down_bias.dtype,
            )

        # dgrad FC2
        grad_gated_weighted = _group_gemm_same_nk(
            a=grad_down_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )

        # wgrad FC2
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            _group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del grad_down_output, gated_weighted

        # Routing weight gradient
        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        del gated_activation, grad_gated_weighted

        # Activation backward
        # GPT-OSS-aware activation backward on the (DSv4-clamped) gate, then chain
        # the swiglu_limit clamp gradient. _moe_gate_activation_backward_pair
        # recomputes the gate activation internally, so it is self-contained.
        grad_gate_for_activation, grad_up_output = _moe_gate_activation_backward_pair(
            grad_gated_activation,
            gate_for_activation,
            up_output,
            getattr(ctx, "hidden_act", "silu"),
        )
        grad_gate_output = _apply_swiglu_clamp_backward(
            grad_gate_for_activation, gate_output, getattr(ctx, "swiglu_limit", 0.0)
        )
        del grad_gated_activation, gate_output, up_output, gate_for_activation, grad_gate_for_activation

        # dgrad FC1: in-place add
        grad_scatter_output = _group_gemm_same_nk(
            a=grad_gate_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        grad_scatter_output += _group_gemm_same_nk(
            a=grad_up_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
            cu_seqlens_m=cu_seqlens_m,
            fp8_compute=fp8_compute,
            fp8_grouped_backend=fp8_grouped_backend,
            fp8_block_size=fp8_block_size,
        )
        handle = dist.all_reduce(grad_scatter_output, group=tp_group, async_op=True)

        grad_gate_up_bias = None
        if ctx.has_gate_up_bias and gate_up_bias.requires_grad:
            grad_gate_up_bias = torch.empty_like(gate_up_bias)
            grad_gate_up_bias[:, : gate_proj.shape[-1]].copy_(
                _sum_expert_bias_grad_by_ids(
                    grad_gate_output,
                    expert_ids,
                    gate_proj.shape[0],
                    gate_up_bias.dtype,
                )
            )
            grad_gate_up_bias[:, gate_proj.shape[-1] : 2 * gate_proj.shape[-1]].copy_(
                _sum_expert_bias_grad_by_ids(
                    grad_up_output,
                    expert_ids,
                    gate_proj.shape[0],
                    gate_up_bias.dtype,
                )
            )

        # wgrad FC1
        grad_gate_proj = None
        if gate_proj.requires_grad:
            grad_gate_proj = torch.empty_like(gate_proj)
            _group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_output,
                c=grad_gate_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del grad_gate_output
        grad_up_proj = None
        if up_proj.requires_grad:
            grad_up_proj = torch.empty_like(up_proj)
            _group_gemm_same_mn(
                a=scatter_output,
                b=grad_up_output,
                c=grad_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
                cu_seqlens_k=cu_seqlens_m,
                fp8_compute=fp8_compute,
                fp8_grouped_backend=fp8_grouped_backend,
                fp8_block_size=fp8_block_size,
            )
        del grad_up_output, scatter_output

        handle.wait()
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)
        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_gate_proj,  # gate_proj
            grad_up_proj,  # up_proj
            grad_down_proj,  # down_proj
            None,  # tp_group
            None,  # hidden_act
            None,  # activation_native
            None,  # fp8_compute
            None,  # fp8_grouped_backend
            None,  # fp8_block_size
            grad_gate_up_bias,  # gate_up_bias
            grad_down_bias,  # down_bias
            None,  # swiglu_limit
        )


def quack_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj: torch.Tensor = None,
    hidden_act: str = "silu",
    activation_native: bool = False,
    fp8_compute: bool = False,
    fp8_grouped_backend: str = "triton_grouped",
    fp8_block_size: int = 128,
    gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
):
    """Forward pass for MoE experts using quack group GEMM (local/TP only).

    EP is handled centrally by ``MoEExperts._ep_forward()``.
    """
    del module
    parallel_state = get_parallel_state()

    if parallel_state.tp_enabled:
        tp_group = parallel_state.tp_mesh.get_group()
        return QuackTPMoeExpertsFunction.apply(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            gate_proj,
            up_proj,
            down_proj,
            tp_group,
            hidden_act,
            activation_native,
            fp8_compute,
            fp8_grouped_backend,
            fp8_block_size,
            gate_up_bias,
            down_bias,
            swiglu_limit,
        )

    return QuackMoeExpertsFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_proj,
        hidden_act,
        activation_native,
        fp8_compute,
        fp8_grouped_backend,
        fp8_block_size,
        gate_up_bias,
        down_bias,
        swiglu_limit,
    )
