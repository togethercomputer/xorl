"""DeepEP dispatch/combine utilities for NVLink-optimized Expert Parallel MoE.

Key design choices:
- ``async_finish=True`` + ``allocate_on_comm_stream=True`` on every dispatch/combine
  call so NVLink communication runs on a dedicated COMM stream, not the COMPUTE stream.
- ``EventOverlap`` / ``EventHandle`` for proper stream coordination.
- Buffer auto-sizing via ``config.get_nvl_buffer_size_hint()`` when available.
- Fused autograd Functions: dispatch+permute and unpermute+combine in single boundaries.
- Deferred combine sync: async combine stores event for later synchronization.
- SM partitioning: ``Buffer.set_num_sms()`` controls communication SM allocation.
"""

import os as _os
import time as _time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


try:
    import deep_ep
    from deep_ep.utils import EventHandle, EventOverlap

    DEEPEP_AVAILABLE = True
except ImportError:
    DEEPEP_AVAILABLE = False
    deep_ep = None
    EventOverlap = None
    EventHandle = None


def check_deepep_available():
    if not DEEPEP_AVAILABLE:
        raise ImportError("DeepEP is not installed. Please install it from https://github.com/deepseek-ai/DeepEP")


_SKIP_PREFLIGHT_ENV = "XORL_SKIP_DEEPEP_INTERNODE_PREFLIGHT"


def _ep_group_spans_nodes(ep_group: Optional[dist.ProcessGroup]) -> Optional[bool]:
    """Best-effort node-span check from the torchrun rank layout; None if undeterminable."""
    if not dist.is_initialized():
        return None
    try:
        local_world = int(_os.environ.get("LOCAL_WORLD_SIZE", "0"))
    except ValueError:
        return None
    if local_world <= 0:
        return None
    world_size = dist.get_world_size()
    if world_size <= local_world:
        return False
    ranks = dist.get_process_group_ranks(ep_group) if ep_group is not None else range(world_size)
    return len({r // local_world for r in ranks}) > 1


def preflight_internode_transport(
    ep_group: Optional[dist.ProcessGroup],
    hidden_dim: int,
    buffer_size_gb: float = 2.0,
    num_sms: int = 20,
    buffer_hidden_bytes: Optional[int] = None,
) -> None:
    """Probe DeepEP's internode transport with a tiny dispatch+combine before training.

    A fabric or per-node IB fault that NCCL rides out (NCCL_IB_TIMEOUT/RETRY_CNT, HCA
    excludes) can leave NVSHMEM — DeepEP's internode transport, which honors none of the
    NCCL env — undeliverable. The first MoE dispatch then spins on
    moe_recv_expert_counter=-1 after minutes of model loading, holding the whole gang
    (observed on q30 EP16 2-node: pair/time-specific, independent of the MoE compute
    backend). Failing here instead costs seconds and names the participating nodes.

    Relies on DeepEP's own CPU timeout to unstick a dead dispatch; the buffer created
    here is the cached default buffer, sized with the same hidden_bytes training uses.
    """
    if _os.environ.get(_SKIP_PREFLIGHT_ENV, "0") == "1":
        return
    if not dist.is_initialized():
        return
    if not _ep_group_spans_nodes(ep_group):  # False or None (unknown topology)
        return

    import socket  # noqa: PLC0415

    from xorl.utils.logging import get_logger  # noqa: PLC0415

    world = dist.get_world_size(ep_group) if ep_group is not None else dist.get_world_size()
    hosts: List[Optional[str]] = [None] * world
    dist.all_gather_object(hosts, socket.gethostname(), group=ep_group)
    node_list = sorted({h for h in hosts if h})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hidden_dim = max(int(hidden_dim), 64)
    x = torch.randn(world, hidden_dim, device=device, dtype=torch.bfloat16)
    routing_weights = torch.ones(world, 1, device=device, dtype=torch.float32)
    selected_experts = torch.arange(world, device=device, dtype=torch.int64).unsqueeze(1)

    buffer = get_default_buffer(ep_group=ep_group, buffer_size_gb=buffer_size_gb, num_sms=num_sms)
    try:
        buffer.init_buffer(hidden_bytes=hidden_dim * 2 if buffer_hidden_bytes is None else int(buffer_hidden_bytes))
        recv_x, _, _, _, handle = dispatch_no_grad(buffer, x, routing_weights, selected_experts, num_experts=world)
        out = combine_no_grad(buffer, recv_x.contiguous(), handle)
    except Exception as exc:
        raise RuntimeError(
            f"DeepEP internode preflight failed on the EP group spanning nodes {node_list}: {exc}\n"
            "The internode transport (NVSHMEM) is not delivering between these nodes even though "
            "NCCL rendezvous worked — typically transient fabric or per-node IB health (NVSHMEM "
            "does not honor NCCL_IB_HCA excludes or NCCL retry/timeout settings). Reschedule onto "
            "different nodes or fall back to ep_dispatch='alltoall'. "
            f"Set {_SKIP_PREFLIGHT_ENV}=1 to skip this probe."
        ) from exc
    if not torch.allclose(out.float(), x.float(), atol=1e-2, rtol=1e-2):
        raise RuntimeError(
            f"DeepEP internode preflight round-trip corrupted data on EP group spanning {node_list}: "
            "dispatch+combine of an identity routing did not return the input. Suspect node/fabric "
            f"health; reschedule or use ep_dispatch='alltoall'. Set {_SKIP_PREFLIGHT_ENV}=1 to skip."
        )
    get_logger(__name__).info_rank0(f"DeepEP internode preflight OK across {len(node_list)} nodes: {node_list}")


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for one token.

    Uses at least 2 bytes (bf16 size) so buffer works for both fp8 and bf16
    without reallocation.
    """
    return x.size(1) * max(x.element_size(), 2)


# ---------------------------------------------------------------------------
# Deferred combine sync
# ---------------------------------------------------------------------------
_pending_combine_event: Optional["EventOverlap"] = None
_ALLOW_UNSAFE_ASYNC_COMBINE: Optional[bool] = None


def _allow_unsafe_async_combine() -> bool:
    if _ALLOW_UNSAFE_ASYNC_COMBINE is not None:
        return _ALLOW_UNSAFE_ASYNC_COMBINE
    return _os.environ.get("XORL_DEEPEP_UNSAFE_ASYNC_COMBINE", "").strip().lower() in {"1", "true", "yes", "on"}


def sync_pending_combine():
    """Wait for any pending async combine to complete.

    No-op if no async combine is pending.  Call this before reading the
    output tensor of a previous async combine on the default CUDA stream.
    """
    global _pending_combine_event
    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


def _store_pending_event(event: "EventOverlap"):
    """Store a combine event for deferred synchronization."""
    global _pending_combine_event
    # Sync any previously pending event first (safety — should not happen
    # if callers sync correctly, but prevents silent data corruption).
    sync_pending_combine()
    _pending_combine_event = event


# ---------------------------------------------------------------------------
# DeepEPBuffer — with SM partitioning
# ---------------------------------------------------------------------------
class DeepEPBuffer:
    def __init__(
        self,
        ep_group: Optional[dist.ProcessGroup] = None,
        buffer_size_gb: float = 2.0,
        low_latency_mode: bool = False,
        num_sms: int = 20,
    ):
        check_deepep_available()
        self.ep_group = ep_group
        self.buffer_size_gb = buffer_size_gb
        self.low_latency_mode = low_latency_mode
        self.num_sms = num_sms
        self._buffer: Optional["deep_ep.Buffer"] = None
        self._dispatch_config = None
        self._combine_config = None

    def init_buffer(self, hidden_bytes: int = 0):
        if self._buffer is not None:
            return

        # Set SM count BEFORE getting configs — configs embed Buffer.num_sms.
        deep_ep.Buffer.set_num_sms(self.num_sms)

        num_ranks = self.ep_group.size() if self.ep_group else 1

        # Compute optimal buffer sizes from DeepEP config hints when available,
        # falling back to user-specified fixed size.
        num_nvl_bytes = int(self.buffer_size_gb * 1e9)
        num_rdma_bytes = 0
        if num_ranks in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144, 160):
            self._dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            self._combine_config = deep_ep.Buffer.get_combine_config(num_ranks)
            if hidden_bytes > 0:
                try:
                    for config in (self._dispatch_config, self._combine_config):
                        num_nvl_bytes = max(
                            config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks),
                            num_nvl_bytes,
                        )
                        num_rdma_bytes = max(
                            config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks),
                            num_rdma_bytes,
                        )
                except (AttributeError, TypeError):
                    pass  # Fallback to fixed size for older DeepEP versions

        # DeepEP asserts 128-byte alignment, and caps NVL bytes at int32 max whenever an
        # RDMA (internode) buffer exists — otherwise it aborts with a cryptic C++ assert
        # (deep_ep.cpp 'num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and ...').
        num_nvl_bytes -= num_nvl_bytes % 128
        num_rdma_bytes -= num_rdma_bytes % 128
        int32_max = 2**31 - 1
        if num_rdma_bytes > 0 and num_nvl_bytes > int32_max:
            raise ValueError(
                f"deepep_buffer_size_gb={self.buffer_size_gb} gives num_nvl_bytes={num_nvl_bytes}, above "
                f"DeepEP's int32 limit ({int32_max}) for buffers with an internode (RDMA) component. "
                "Use deepep_buffer_size_gb <= 2.0 when the EP group spans nodes."
            )

        self._buffer = deep_ep.Buffer(
            group=self.ep_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=self.low_latency_mode,
            explicitly_destroy=True,
        )

    def destroy_buffer(self):
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None

    def __del__(self):
        try:
            self.destroy_buffer()
        except Exception:
            pass

    @property
    def buffer(self) -> "deep_ep.Buffer":
        if self._buffer is None:
            self.init_buffer()
        return self._buffer

    @property
    def dispatch_config(self):
        self.init_buffer()
        return self._dispatch_config

    @property
    def combine_config(self):
        self.init_buffer()
        return self._combine_config


# ---------------------------------------------------------------------------
# DispatchContext — shared state between dispatch and combine
# ---------------------------------------------------------------------------
@dataclass
class DispatchContext:
    handle: object
    permuted_scores: torch.Tensor
    permuted_indices: torch.Tensor
    num_recv_tokens: int
    num_valid: int
    dtype: torch.dtype
    device: torch.device
    hidden_dim: int
    # DeepEP normal mode returns rank-local expert ids (``-1`` marks a route
    # owned by another rank) and the transported FP32 routing weights.  The
    # generic grouped-GEMM path only needs their expert-sorted projections,
    # but DSV4's native Marlin runner consumes the receive layout directly so
    # its local top-k reduction is the same numerical program as serving.
    recv_topk_idx: Optional[torch.Tensor] = None
    recv_topk_weights: Optional[torch.Tensor] = None
    # Monotonic process-local identity for this original DeepEP handle.  Every
    # EP rank constructs handles in the same forward order; the optional trace
    # uses this value to diagnose rank-asymmetric backward scheduling without
    # adding a collective or touching the numerical program.
    call_id: int = -1


_deepep_call_counter = 0


def _next_deepep_call_id() -> int:
    global _deepep_call_counter
    call_id = _deepep_call_counter
    _deepep_call_counter += 1
    return call_id


def _trace_deepep_boundary(
    call_id: int,
    boundary: str,
    state: str,
    logical_rank: int = -1,
    trace_label: str | None = None,
) -> None:
    """Append one opt-in liveness record without synchronizing any device."""

    trace_dir = _os.environ.get("XORL_DEEPEP_BOUNDARY_TRACE_DIR", "").strip()
    if not trace_dir:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    _os.makedirs(trace_dir, exist_ok=True)
    label = "none" if trace_label is None else str(trace_label).replace(" ", "_")
    line = (
        f"{_time.monotonic_ns()} rank={rank} call_id={call_id} "
        f"boundary={boundary} state={state} logical_rank={logical_rank} label={label}\n"
    )
    with open(_os.path.join(trace_dir, f"rank{rank:05d}.log"), "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


# ---------------------------------------------------------------------------
# Permutation utility (used inside fused Functions)
# ---------------------------------------------------------------------------
def permute_for_experts(
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_valid: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Permute received tokens into expert-sorted order.

    Uses flat-view sorting: invalid entries (-1) get max sort key so they
    sort to the end, then we slice to ``[:num_valid]``.  This avoids boolean
    indexing (and its ``nonzero()`` + ``_index_put_impl_`` backward cost)
    and ``repeat_interleave``.

    Returns:
        permuted_x: Tokens sorted by expert ``[num_valid, hidden_dim]``.
        permuted_scores: Routing scores in same order ``[num_valid]``.
        permuted_indices: Original token indices for unpermute ``[num_valid]``.
        num_valid: Number of valid (token, expert) pairs.
    """
    permuted_scores, permuted_indices, num_valid = permutation_metadata_for_experts(
        recv_topk_idx,
        recv_topk_weights,
        num_valid=num_valid,
    )

    if num_valid == 0:
        return recv_x[:0], permuted_scores, permuted_indices, 0

    # Gather tokens and scores in expert-sorted order
    permuted_x = recv_x.index_select(0, permuted_indices)

    return permuted_x, permuted_scores, permuted_indices, num_valid


def permutation_metadata_for_experts(
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_valid: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return expert-order metadata without gathering ``recv_x``.

    This is used by memory-constrained fused expert paths that can gather
    hidden chunks directly from ``recv_x`` instead of materializing the full
    expert-order token matrix.
    """
    device = recv_topk_idx.device
    topk = recv_topk_idx.shape[1]

    if num_valid is not None and num_valid == 0:
        return (
            torch.empty(0, dtype=recv_topk_weights.dtype, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            0,
        )

    # Flat view of expert IDs and scores — no copy, no boolean indexing
    flat_expert_ids = recv_topk_idx.reshape(-1)  # [num_recv_tokens * topk]
    flat_scores = recv_topk_weights.reshape(-1)  # [num_recv_tokens * topk]

    # Invalid entries (-1) → max int64 so they sort to the end
    sort_keys = torch.where(
        flat_expert_ids >= 0,
        flat_expert_ids.long(),
        torch.iinfo(torch.int64).max,
    )
    sort_order = torch.argsort(sort_keys, stable=True)

    # Slice to valid entries only
    if num_valid is None:
        num_valid = (flat_expert_ids >= 0).sum().item()
    sort_order = sort_order[:num_valid]

    # Token indices via integer division (no repeat_interleave needed)
    permuted_indices = sort_order // topk
    permuted_scores = flat_scores[sort_order]

    return permuted_scores, permuted_indices, num_valid


def _deepep_grad_scatter_accum_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Return the accumulation dtype for DeepEP grad scatter.

    FP32 is the default to preserve existing numerics. Large bf16 full-weight
    runs can opt into input-dtype accumulation to avoid a short-lived fp32
    receive-gradient buffer in DeepEP backward.
    """
    mode = _os.environ.get("XORL_DEEPEP_GRAD_SCATTER_ACCUM_DTYPE", "fp32").strip().lower()
    if mode in {"fp32", "float32"}:
        return torch.float32
    if mode in {"input", "input_dtype", "same", "same_dtype"}:
        return input_dtype
    raise ValueError(
        "XORL_DEEPEP_GRAD_SCATTER_ACCUM_DTYPE must be one of "
        "'fp32', 'float32', 'input', 'input_dtype', 'same', or 'same_dtype'"
    )


def _deepep_combine_scatter_accum_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Return the accumulation dtype for DeepEP forward combine scatter.

    FP32 preserves existing numerics. Large full-weight runs can opt into
    input-dtype accumulation to avoid materializing both fp32 and bf16 combine
    buffers during the forward pass.
    """
    mode = _os.environ.get("XORL_DEEPEP_COMBINE_SCATTER_ACCUM_DTYPE", "fp32").strip().lower()
    if mode in {"fp32", "float32"}:
        return torch.float32
    if mode in {"input", "input_dtype", "same", "same_dtype"}:
        return input_dtype
    raise ValueError(
        "XORL_DEEPEP_COMBINE_SCATTER_ACCUM_DTYPE must be one of "
        "'fp32', 'float32', 'input', 'input_dtype', 'same', or 'same_dtype'"
    )


def _cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def _zeros_grad_recv_x(
    num_recv_tokens: int,
    hidden_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    try:
        return torch.zeros(num_recv_tokens, hidden_dim, dtype=dtype, device=device)
    except torch.OutOfMemoryError:
        retry = _os.environ.get("XORL_DEEPEP_GRAD_SCATTER_EMPTY_CACHE_ON_OOM", "0").strip().lower()
        if device.type != "cuda" or retry not in {"1", "true", "yes"}:
            raise
        torch.cuda.empty_cache()
        return torch.zeros(num_recv_tokens, hidden_dim, dtype=dtype, device=device)


def _scatter_expert_grad_to_recv(
    grad_expert_input: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_recv_tokens: int,
    hidden_dim: int,
    *,
    chunk_tokens: int | None = None,
) -> torch.Tensor:
    """Scatter expert-order gradients back to recv order with bounded cast memory."""
    accum_dtype = _deepep_grad_scatter_accum_dtype(grad_expert_input.dtype)
    grad_recv_x = _zeros_grad_recv_x(
        num_recv_tokens,
        hidden_dim,
        dtype=accum_dtype,
        device=grad_expert_input.device,
    )
    if grad_expert_input.numel() == 0:
        return _cast_if_needed(grad_recv_x, grad_expert_input.dtype)

    if chunk_tokens is None:
        chunk_tokens = int(_os.environ.get("XORL_DEEPEP_GRAD_INDEX_ADD_CHUNK_TOKENS", "2048"))
    chunk_tokens = max(1, chunk_tokens)

    if grad_expert_input.dtype == grad_recv_x.dtype:
        grad_recv_x.index_add_(0, permuted_indices, grad_expert_input)
    else:
        for start in range(0, grad_expert_input.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, grad_expert_input.shape[0])
            grad_chunk = grad_expert_input[start:end].float()
            grad_recv_x.index_add_(0, permuted_indices[start:end], grad_chunk)
            del grad_chunk

    return _cast_if_needed(grad_recv_x, grad_expert_input.dtype)


# ---------------------------------------------------------------------------
# Fused autograd Functions
# ---------------------------------------------------------------------------
class _FusedDispatchAndPermute(torch.autograd.Function):
    """Fused dispatch + expert permutation in a single autograd boundary.

    Forward:
        1. ``buffer.get_dispatch_layout()`` + ``buffer.dispatch(async)`` + sync
        2. ``permute_for_experts()`` — argsort + index_select
        3. Return ``(expert_input, cumsum, DispatchContext)``

    Backward:
        1. Scatter gradient from expert order to recv order (index_add)
        2. ``buffer.combine()`` to reverse dispatch → grad_hidden_states
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        buffer: "DeepEPBuffer",
        num_experts: int,
    ):
        call_id = _next_deepep_call_id()
        _trace_deepep_boundary(call_id, "original_dispatch_forward", "enter")
        buffer.init_buffer(hidden_bytes=get_hidden_bytes(x))
        topk_idx_deepep = topk_idx.to(deep_ep.topk_idx_t)
        topk_weights_f32 = topk_weights.to(torch.float32)

        # --- dispatch ---
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
            buffer.buffer.get_dispatch_layout(topk_idx_deepep, num_experts)
        )
        previous_event = EventOverlap(EventHandle())
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, event = buffer.buffer.dispatch(
            x=x.contiguous(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx_deepep,
            topk_weights=topk_weights_f32,
            config=buffer.dispatch_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        # Comm-stream-owned outputs consumed on the compute stream: without
        # record_stream, a freed block returns to the comm-stream pool and a
        # later collective can overwrite it while a compute-stream read is
        # still in flight (see _deepep_combine_chunk in ops/moe/quack.py).
        recv_x.record_stream(torch.cuda.current_stream())
        recv_topk_idx.record_stream(torch.cuda.current_stream())
        recv_topk_weights.record_stream(torch.cuda.current_stream())

        num_recv_tokens = recv_x.shape[0]
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.int32, device=x.device)
        cumsum = torch.cumsum(recv_counts_tensor, dim=0)
        # sum(recv_counts) is pure Python (list sum) — no GPU sync needed.
        total_valid_count = sum(recv_counts)

        # --- permute ---
        expert_input, permuted_scores, permuted_indices, num_valid = permute_for_experts(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_valid=total_valid_count,
        )

        # Save for backward
        ctx.save_for_backward(permuted_indices)
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        ctx.call_id = call_id
        ctx.num_recv_tokens = num_recv_tokens
        ctx.hidden_dim = x.shape[1]

        dispatch_ctx = DispatchContext(
            handle=handle,
            permuted_scores=permuted_scores,
            permuted_indices=permuted_indices,
            num_recv_tokens=num_recv_tokens,
            num_valid=num_valid,
            dtype=x.dtype,
            device=x.device,
            hidden_dim=x.shape[1],
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            call_id=call_id,
        )
        _trace_deepep_boundary(call_id, "original_dispatch_forward", "exit")
        return expert_input, cumsum, dispatch_ctx

    @staticmethod
    def backward(ctx, grad_expert_input, grad_cumsum, grad_dispatch_ctx):
        del grad_cumsum, grad_dispatch_ctx
        if grad_expert_input is None:
            return None, None, None, None, None

        (permuted_indices,) = ctx.saved_tensors
        buffer = ctx.buffer
        handle = ctx.handle

        # Step 1: Scatter gradient from expert order → recv order.
        grad_recv_x = _scatter_expert_grad_to_recv(
            grad_expert_input,
            permuted_indices,
            ctx.num_recv_tokens,
            ctx.hidden_dim,
        )

        # Step 2: Combine to reverse dispatch
        previous_event = EventOverlap(EventHandle())
        grad_x, _, event = buffer.buffer.combine(
            x=grad_recv_x.contiguous(),
            handle=handle,
            config=buffer.combine_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        grad_x.record_stream(torch.cuda.current_stream())

        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        return grad_x, None, None, None, None


class _FusedDispatchNoPermute(torch.autograd.Function):
    """DeepEP dispatch that returns recv-order tokens plus expert-order metadata.

    The regular path immediately gathers ``recv_x`` into expert order.  Large
    128K MoE runs can make that gather several GiB, so Quack's fused DeepEP
    path consumes ``recv_x`` directly and gathers hidden chunks as needed.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        buffer: "DeepEPBuffer",
        num_experts: int,
        complete_backward_device_boundary: bool,
        backward_trace_label: str | None,
        backward_layer_dependency: torch.Tensor | None,
        backward_shared_dependency: torch.Tensor | None,
    ):
        call_id = _next_deepep_call_id()
        _trace_deepep_boundary(
            call_id,
            "original_dispatch_forward",
            "enter",
            trace_label=backward_trace_label,
        )
        buffer.init_buffer(hidden_bytes=get_hidden_bytes(x))
        topk_idx_deepep = topk_idx.to(deep_ep.topk_idx_t)
        topk_weights_f32 = topk_weights.to(torch.float32)

        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
            buffer.buffer.get_dispatch_layout(topk_idx_deepep, num_experts)
        )
        previous_event = EventOverlap(EventHandle())
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, event = buffer.buffer.dispatch(
            x=x.contiguous(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx_deepep,
            topk_weights=topk_weights_f32,
            config=buffer.dispatch_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        # Comm-stream-owned outputs consumed on the compute stream: without
        # record_stream, a freed block returns to the comm-stream pool and a
        # later collective can overwrite it while a compute-stream read is
        # still in flight (see _deepep_combine_chunk in ops/moe/quack.py).
        recv_x.record_stream(torch.cuda.current_stream())
        recv_topk_idx.record_stream(torch.cuda.current_stream())
        recv_topk_weights.record_stream(torch.cuda.current_stream())

        num_recv_tokens = recv_x.shape[0]
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.int32, device=x.device)
        cumsum = torch.cumsum(recv_counts_tensor, dim=0)
        total_valid_count = sum(recv_counts)

        permuted_scores, permuted_indices, num_valid = permutation_metadata_for_experts(
            recv_topk_idx,
            recv_topk_weights,
            num_valid=total_valid_count,
        )

        ctx.buffer = buffer
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        ctx.call_id = call_id
        ctx.complete_backward_device_boundary = bool(complete_backward_device_boundary)
        ctx.backward_trace_label = backward_trace_label
        # These two ignored forward operands are ordering edges, not value
        # operands.  Returning explicit zero gradients for them only after the
        # terminal reverse-combine completes prevents their FSDP backward
        # branches from entering c10d while DeepEP still owns the device.
        ctx.backward_layer_dependency_meta = (
            None
            if backward_layer_dependency is None
            else (
                tuple(backward_layer_dependency.shape),
                backward_layer_dependency.dtype,
                backward_layer_dependency.device,
            )
        )
        ctx.backward_shared_dependency_meta = (
            None
            if backward_shared_dependency is None
            else (
                tuple(backward_shared_dependency.shape),
                backward_shared_dependency.dtype,
                backward_shared_dependency.device,
            )
        )

        dispatch_ctx = DispatchContext(
            handle=handle,
            permuted_scores=permuted_scores,
            permuted_indices=permuted_indices,
            num_recv_tokens=num_recv_tokens,
            num_valid=num_valid,
            dtype=x.dtype,
            device=x.device,
            hidden_dim=x.shape[1],
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            call_id=call_id,
        )
        _trace_deepep_boundary(
            call_id,
            "original_dispatch_forward",
            "exit",
            trace_label=backward_trace_label,
        )
        return recv_x, cumsum, dispatch_ctx

    @staticmethod
    def backward(ctx, grad_recv_x, grad_cumsum, grad_dispatch_ctx):
        del grad_cumsum, grad_dispatch_ctx
        if grad_recv_x is None:
            return None, None, None, None, None, None, None, None, None

        buffer = ctx.buffer
        handle = ctx.handle
        _trace_deepep_boundary(
            ctx.call_id,
            "input_reverse_combine",
            "enter",
            trace_label=ctx.backward_trace_label,
        )
        previous_event = EventOverlap(EventHandle())
        grad_x, _, event = buffer.buffer.combine(
            x=grad_recv_x.contiguous(),
            handle=handle,
            config=buffer.combine_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        grad_x.record_stream(torch.cuda.current_stream())
        _trace_deepep_boundary(
            ctx.call_id,
            "input_reverse_combine",
            "api_return",
            trace_label=ctx.backward_trace_label,
        )
        if ctx.complete_backward_device_boundary:
            # GLM's shared-expert FSDP root and transformer residual can use
            # c10d private streams.  Do not release either autograd edge while
            # this layer's final normal-mode DeepEP reverse-combine can still
            # be spinning on its own private stream: that cross-communicator
            # overlap forms a device cycle despite identical host order on all
            # ranks.  This boundary changes scheduling only; no value is cast,
            # communicated again, or added to a nonzero gradient.
            torch.cuda.current_stream(grad_x.device).synchronize()
            _trace_deepep_boundary(
                ctx.call_id,
                "input_reverse_combine",
                "device_complete",
                trace_label=ctx.backward_trace_label,
            )

        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        dependency_grads = []
        for metadata in (
            getattr(ctx, "backward_layer_dependency_meta", None),
            getattr(ctx, "backward_shared_dependency_meta", None),
        ):
            if metadata is None:
                dependency_grads.append(None)
                continue
            shape, dtype, device = metadata
            dependency_grads.append(torch.zeros(shape, dtype=dtype, device=device))
        return grad_x, None, None, None, None, None, None, *dependency_grads


class _FusedUnpermuteAndCombine(torch.autograd.Function):
    """Fused scatter-add (unpermute) + combine in a single autograd boundary.

    Forward:
        1. Scatter-add: ``output[idx[i]] += expert_output[i]``
        2. ``buffer.combine()`` to send results back to original ranks
        3. Optionally defer sync (async_combine) for overlap with next layer

    Backward:
        1. ``buffer.dispatch()`` to reverse combine → grad in recv order
        2. ``grad_expert[i] = grad[idx[i]]`` — reverse of scatter-add
    """

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        buffer: "DeepEPBuffer",
        dispatch_ctx: DispatchContext,
        async_combine: bool,
    ):
        dtype = dispatch_ctx.dtype
        hidden_dim = expert_output.shape[1] if expert_output.shape[0] > 0 else dispatch_ctx.hidden_dim
        device = expert_output.device

        # Step 1: Scatter-add (unpermute)
        if expert_output.shape[0] == 0:
            gather_output = torch.zeros(
                dispatch_ctx.num_recv_tokens,
                hidden_dim,
                dtype=dtype,
                device=device,
            )
        else:
            # Scores are already applied by the expert compute function
            # (triton/native backends multiply by expert_scores). Do NOT
            # re-apply here — that would double-count router weights.
            # re-apply here, because that would double-count router weights.
            accum_dtype = _deepep_combine_scatter_accum_dtype(dtype)
            gather_output = torch.zeros(
                dispatch_ctx.num_recv_tokens,
                hidden_dim,
                dtype=accum_dtype,
                device=device,
            )
            idx_2d = dispatch_ctx.permuted_indices.unsqueeze(1).expand(-1, hidden_dim)
            _CHUNK = 4096
            if expert_output.dtype == gather_output.dtype:
                for _i in range(0, expert_output.shape[0], _CHUNK):
                    _end = min(_i + _CHUNK, expert_output.shape[0])
                    gather_output.scatter_add_(0, idx_2d[_i:_end], expert_output[_i:_end])
            else:
                for _i in range(0, expert_output.shape[0], _CHUNK):
                    _end = min(_i + _CHUNK, expert_output.shape[0])
                    expert_chunk = expert_output[_i:_end].float()
                    gather_output.scatter_add_(0, idx_2d[_i:_end], expert_chunk)
                    del expert_chunk
            gather_output = _cast_if_needed(gather_output, dtype)

        # Step 2: Combine
        previous_event = EventOverlap(EventHandle())
        combined_x, _, event = buffer.buffer.combine(
            x=gather_output.contiguous(),
            handle=dispatch_ctx.handle,
            config=buffer.combine_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        if async_combine:
            _store_pending_event(event)
        else:
            event.current_stream_wait()
        # combined_x is comm-stream-owned (allocate_on_comm_stream): without
        # record_stream, its freed block returns to the comm-stream pool and a
        # later collective can reuse it while compute-stream reads are in flight.
        combined_x.record_stream(torch.cuda.current_stream())

        ctx.save_for_backward(dispatch_ctx.permuted_indices)
        ctx.buffer = buffer
        ctx.handle = dispatch_ctx.handle
        ctx.input_dtype = expert_output.dtype
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None

        (permuted_indices,) = ctx.saved_tensors
        buffer = ctx.buffer
        handle = ctx.handle

        # Step 1: Dispatch to reverse combine (sends grads back to expert-owning ranks)
        previous_event = EventOverlap(EventHandle())
        grad_gather, _, _, _, _, event = buffer.buffer.dispatch(
            x=grad_output.contiguous(),
            handle=handle,
            config=buffer.dispatch_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        # grad_gather is comm-stream-owned (allocate_on_comm_stream): guard its
        # block against comm-pool reuse while compute-stream reads are in flight.
        grad_gather.record_stream(torch.cuda.current_stream())

        if grad_gather.dtype != ctx.input_dtype:
            grad_gather = grad_gather.to(ctx.input_dtype)

        # Step 2: Reverse scatter-add
        grad_expert_output = grad_gather.index_select(0, permuted_indices)

        return grad_expert_output, None, None, None


class _FusedNativeReceiveCombine(torch.autograd.Function):
    """Handle-based combine for a runner that consumes DeepEP receive rows.

    Unlike :class:`_FusedUnpermuteAndCombine`, this boundary performs no
    trainer-owned scatter-add.  The native MoE runner has already reduced the
    local top-k slots for each received row, so DeepEP's original dispatch
    handle is the complete inverse movement required by serving.
    """

    @staticmethod
    def forward(
        ctx,
        recv_output: torch.Tensor,
        buffer: "DeepEPBuffer",
        dispatch_ctx: DispatchContext,
        async_combine: bool,
    ):
        previous_event = EventOverlap(EventHandle())
        combined_x, _, event = buffer.buffer.combine(
            x=recv_output.contiguous(),
            handle=dispatch_ctx.handle,
            config=buffer.combine_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        if async_combine:
            _store_pending_event(event)
        else:
            event.current_stream_wait()
        combined_x.record_stream(torch.cuda.current_stream())

        ctx.buffer = buffer
        ctx.handle = dispatch_ctx.handle
        ctx.input_dtype = recv_output.dtype
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None

        previous_event = EventOverlap(EventHandle())
        grad_recv, _, _, _, _, event = ctx.buffer.buffer.dispatch(
            x=grad_output.contiguous(),
            handle=ctx.handle,
            config=ctx.buffer.dispatch_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        grad_recv.record_stream(torch.cuda.current_stream())
        if grad_recv.dtype != ctx.input_dtype:
            grad_recv = grad_recv.to(ctx.input_dtype)
        return grad_recv, None, None, None


# ---------------------------------------------------------------------------
# No-grad dispatch/combine (for inference or profiling)
# ---------------------------------------------------------------------------
def dispatch_no_grad(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], object]:
    buffer.init_buffer(hidden_bytes=get_hidden_bytes(hidden_states))
    topk_idx = selected_experts.to(deep_ep.topk_idx_t)
    topk_weights = routing_weights.to(torch.float32)
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
        buffer.buffer.get_dispatch_layout(topk_idx, num_experts)
    )
    previous_event = EventOverlap(EventHandle())
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, event = buffer.buffer.dispatch(
        x=hidden_states,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=buffer.dispatch_config,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    event.current_stream_wait()
    return recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle


def combine_no_grad(
    buffer: DeepEPBuffer,
    gather_output: torch.Tensor,
    handle: object,
) -> torch.Tensor:
    previous_event = EventOverlap(EventHandle())
    combined_output, _, event = buffer.buffer.combine(
        x=gather_output,
        handle=handle,
        config=buffer.combine_config,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    event.current_stream_wait()
    return combined_output


# ---------------------------------------------------------------------------
# Profiling flag
# ---------------------------------------------------------------------------
_DEEPEP_PROFILE = _os.environ.get("XORL_DEEPEP_PROFILE", "0").strip().lower() not in {"0", "false", "no", "off", ""}


# ---------------------------------------------------------------------------
# Public API — registered in backend/__init__.py
# ---------------------------------------------------------------------------
def token_pre_dispatch(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    num_local_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchContext]:
    """Dispatch tokens to expert-owning ranks and permute into expert order.

    Calls ``sync_pending_combine()`` first to ensure any previous async
    combine has completed before starting a new dispatch.
    """
    sync_pending_combine()

    if _DEEPEP_PROFILE:
        return _token_pre_dispatch_profiled(
            buffer,
            hidden_states,
            routing_weights,
            selected_experts,
            num_experts,
        )

    expert_input, cumsum, ctx = _FusedDispatchAndPermute.apply(
        hidden_states,
        selected_experts,
        routing_weights,
        buffer,
        num_experts,
    )
    return expert_input, cumsum, ctx


def token_pre_dispatch_no_permute(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    num_local_experts: int = 0,
    *,
    complete_backward_device_boundary: bool = False,
    backward_trace_label: str | None = None,
    backward_layer_dependency: torch.Tensor | None = None,
    backward_shared_dependency: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchContext]:
    """Dispatch tokens with DeepEP but leave them in recv order.

    ``DispatchContext`` still carries expert-order indices and scores.  Fused
    expert implementations can use that metadata to gather bounded chunks.
    """
    del num_local_experts
    sync_pending_combine()
    recv_x, cumsum, ctx = _FusedDispatchNoPermute.apply(
        hidden_states,
        selected_experts,
        routing_weights,
        buffer,
        num_experts,
        complete_backward_device_boundary,
        backward_trace_label,
        backward_layer_dependency,
        backward_shared_dependency,
    )
    return recv_x, cumsum, ctx


def token_pre_dispatch_native(
    buffer: DeepEPBuffer,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    *,
    complete_backward_device_boundary: bool = False,
    backward_trace_label: str | None = None,
    backward_layer_dependency: torch.Tensor | None = None,
    backward_shared_dependency: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchContext]:
    """Dispatch without permutation and expose transported local top-k data.

    This is the native-runner boundary: expert ids and weights come from the
    actual DeepEP receive, never from replayed router state.  Routing weights
    are intentionally non-differentiable here; the admitted DSV4 exact program
    freezes its router while hidden-state and active-LoRA gradients cross the
    dispatch/combine autograd boundaries.
    """

    recv_x, _cumsum, ctx = token_pre_dispatch_no_permute(
        buffer=buffer,
        hidden_states=hidden_states,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        num_experts=num_experts,
        complete_backward_device_boundary=complete_backward_device_boundary,
        backward_trace_label=backward_trace_label,
        backward_layer_dependency=backward_layer_dependency,
        backward_shared_dependency=backward_shared_dependency,
    )
    if ctx.recv_topk_idx is None or ctx.recv_topk_weights is None:
        raise RuntimeError("DeepEP native dispatch did not preserve receive top-k metadata")
    return recv_x, ctx.recv_topk_idx, ctx.recv_topk_weights, ctx


def tokens_post_combine(
    buffer: DeepEPBuffer,
    expert_output: torch.Tensor,
    ctx: DispatchContext,
    async_combine: bool = False,
) -> torch.Tensor:
    """Unpermute expert outputs and combine back to original ranks.

    Args:
        async_combine: If True, combine runs asynchronously on the comm stream.
            The output tensor data is NOT valid on the default stream until
            ``sync_pending_combine()`` is called.  The next call to
            ``token_pre_dispatch()`` automatically syncs.
    """
    # The returned MoE output is consumed immediately by the transformer block
    # before the next DeepEP dispatch. Deferring the wait until the next dispatch
    # lets downstream compute read incomplete combine data. Keep the API flag for
    # provenance, but require an explicit unsafe opt-in before honoring it.
    if async_combine and not _allow_unsafe_async_combine():
        async_combine = False

    if _DEEPEP_PROFILE:
        return _tokens_post_combine_profiled(buffer, expert_output, ctx, async_combine)

    return _FusedUnpermuteAndCombine.apply(expert_output, buffer, ctx, async_combine)


def tokens_post_combine_native(
    buffer: DeepEPBuffer,
    recv_output: torch.Tensor,
    ctx: DispatchContext,
    async_combine: bool = False,
) -> torch.Tensor:
    """Combine native runner output in the original DeepEP receive layout."""

    if async_combine and not _allow_unsafe_async_combine():
        async_combine = False
    return _FusedNativeReceiveCombine.apply(recv_output, buffer, ctx, async_combine)


# ---------------------------------------------------------------------------
# Profiled versions
# ---------------------------------------------------------------------------
def _token_pre_dispatch_profiled(
    buffer,
    hidden_states,
    routing_weights,
    selected_experts,
    num_experts,
):
    """Profiled version of token_pre_dispatch — enabled by XORL_DEEPEP_PROFILE=1."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    ev[0].record()
    expert_input, cumsum, ctx = _FusedDispatchAndPermute.apply(
        hidden_states,
        selected_experts,
        routing_weights,
        buffer,
        num_experts,
    )
    ev[1].record()

    torch.cuda.synchronize()
    t_total = ev[0].elapsed_time(ev[1])
    if rank == 0:
        print(
            f"[DEEPEP PRE r{rank}] dispatch+permute={t_total:.1f}ms  "
            f"num_valid={ctx.num_valid}  hidden={hidden_states.shape}",
            flush=True,
        )
    return expert_input, cumsum, ctx


def _tokens_post_combine_profiled(buffer, expert_output, ctx, async_combine):
    """Profiled version of tokens_post_combine — enabled by XORL_DEEPEP_PROFILE=1."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    ev[0].record()
    # Force sync for profiling even if async_combine requested
    result = _FusedUnpermuteAndCombine.apply(expert_output, buffer, ctx, False)
    ev[1].record()

    torch.cuda.synchronize()
    t_total = ev[0].elapsed_time(ev[1])
    if rank == 0:
        print(
            f"[DEEPEP POST r{rank}] unpermute+combine={t_total:.1f}ms  expert_out={expert_output.shape}",
            flush=True,
        )
    return result


# ---------------------------------------------------------------------------
# Global buffer cache
# ---------------------------------------------------------------------------
_default_buffer: Optional[DeepEPBuffer] = None


def get_default_buffer(
    ep_group: Optional[dist.ProcessGroup] = None,
    buffer_size_gb: float = 2.0,
    num_sms: int = 20,
) -> DeepEPBuffer:
    global _default_buffer
    if _default_buffer is None:
        _default_buffer = DeepEPBuffer(
            ep_group=ep_group,
            buffer_size_gb=buffer_size_gb,
            num_sms=num_sms,
        )
    return _default_buffer


def destroy_default_buffer():
    global _default_buffer
    if _default_buffer is not None:
        _default_buffer.destroy_buffer()
        _default_buffer = None
