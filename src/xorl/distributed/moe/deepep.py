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
    device = recv_x.device
    topk = recv_topk_idx.shape[1]

    if num_valid is not None and num_valid == 0:
        return (
            recv_x[:0],
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

    # Gather tokens and scores in expert-sorted order
    permuted_x = recv_x.index_select(0, permuted_indices)
    permuted_scores = flat_scores[sort_order]

    return permuted_x, permuted_scores, permuted_indices, num_valid


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
        )
        return expert_input, cumsum, dispatch_ctx

    @staticmethod
    def backward(ctx, grad_expert_input, grad_cumsum, grad_dispatch_ctx):
        del grad_cumsum, grad_dispatch_ctx
        if grad_expert_input is None:
            return None, None, None, None, None

        (permuted_indices,) = ctx.saved_tensors
        buffer = ctx.buffer
        handle = ctx.handle

        # Step 1: Scatter gradient from expert order → recv order
        grad_recv_x = torch.zeros(
            ctx.num_recv_tokens,
            ctx.hidden_dim,
            dtype=grad_expert_input.dtype,
            device=grad_expert_input.device,
        )
        grad_recv_x.index_add_(0, permuted_indices, grad_expert_input)

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

        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        return grad_x, None, None, None, None


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
            gather_output = torch.zeros(
                dispatch_ctx.num_recv_tokens,
                hidden_dim,
                dtype=expert_output.dtype,
                device=device,
            )
            idx_2d = dispatch_ctx.permuted_indices.unsqueeze(1).expand(-1, hidden_dim)
            gather_output.scatter_add_(0, idx_2d, expert_output)

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

        if grad_gather.dtype != ctx.input_dtype:
            grad_gather = grad_gather.to(ctx.input_dtype)

        # Step 2: Reverse scatter-add
        grad_expert_output = grad_gather.index_select(0, permuted_indices)

        return grad_expert_output, None, None, None


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
    if _DEEPEP_PROFILE:
        return _tokens_post_combine_profiled(buffer, expert_output, ctx, async_combine)

    return _FusedUnpermuteAndCombine.apply(expert_output, buffer, ctx, async_combine)


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
