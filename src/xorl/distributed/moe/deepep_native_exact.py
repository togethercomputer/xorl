"""Native DeepEP transport with explicit BF16-leaf reduction contracts.

This is the reusable post-expert half of the native zero-K3 program.  The
original DeepEP dispatch handle is retained from the real top-k dispatch.  The
stock fused ``no_combine=False`` MoE program applies routing weights and sums
the owner-local top-k slots in its normal FP32 accumulator, then stores one
BF16 rank leaf.  Exact mode sends those leaves through DeepEP's one-call
deterministic hierarchical receiver tree.  Rank-serial implementations belong
only in tests and benchmarks as executable oracles.

The numerical boundaries are intentional:

* expert kernels may use their normal internal accumulators;
* the fused local combine stores one BF16 rank leaf;
* every value crossing DeepEP is BF16;
* the selected fixed fold promotes BF16 leaves for its FP64 reduction nodes
  and casts only at the specified BF16 leaf/consumer boundaries.

This is not the older synthetic post-expert rank-leaf dispatcher.  It uses the
handle produced by the actual pre-expert top-k dispatch and is consequently a
native DeepEP execution path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist


DEEPEP_DETERMINISTIC_PROTOCOL = "deepep_deterministic_hierarchical_bf16_v2"
DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL = DEEPEP_DETERMINISTIC_PROTOCOL
DEEPEP_DETERMINISTIC_SUPPORTED_EP_SIZES = frozenset({2, 4, 8, 16})
logger = logging.getLogger(__name__)
_engagement_logged = False


class DeepEPNativeExactError(RuntimeError):
    """The real DeepEP dispatch receipt violates the native exact contract."""


def validate_native_combine_geometry(ep_size: int) -> None:
    """Validate topology before dispatch can mutate DeepEP buffer state."""

    if ep_size not in DEEPEP_DETERMINISTIC_SUPPORTED_EP_SIZES:
        raise DeepEPNativeExactError(
            "DeepEP deterministic combine supports EP sizes "
            f"{sorted(DEEPEP_DETERMINISTIC_SUPPORTED_EP_SIZES)}, got EP{ep_size}"
        )


@dataclass(frozen=True)
class NativeDeepEPGeometry:
    ep_size: int
    ep_rank: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.ep_size <= 0:
            raise DeepEPNativeExactError("native DeepEP requires a positive EP size")
        if not 0 <= self.ep_rank < self.ep_size:
            raise DeepEPNativeExactError(f"native DeepEP EP rank {self.ep_rank} is outside [0, {self.ep_size})")
        if self.hidden_size <= 0:
            raise DeepEPNativeExactError("native DeepEP requires a positive hidden size")

    @property
    def wire_width(self) -> int:
        return self.hidden_size

    @property
    def wire_hidden_bytes(self) -> int:
        # The wire type is always BF16, independent of the model parameter type.
        return self.wire_width * torch.tensor([], dtype=torch.bfloat16).element_size()


def resolve_native_deepep_geometry(ep_group, hidden_size: int) -> NativeDeepEPGeometry:
    """Resolve physical group rank as the immutable logical leaf ordinal."""

    if not dist.is_initialized():
        raise DeepEPNativeExactError("native DeepEP requires initialized torch.distributed")
    return NativeDeepEPGeometry(
        ep_size=dist.get_world_size(ep_group),
        ep_rank=dist.get_rank(ep_group),
        hidden_size=int(hidden_size),
    )


def validate_native_receive_metadata(
    recv_output: torch.Tensor,
    dispatch_ctx,
    *,
    num_local_experts: int,
) -> None:
    """Fail closed on the actual normal-mode DeepEP receive receipt.

    A received row exists only because at least one route belongs to this rank,
    so every non-empty row must name a valid local expert.  ``-1`` remains the
    required marker for the other top-k slots.  Empty receive batches are a
    valid load-balancing outcome.
    """

    if recv_output.ndim != 2:
        raise DeepEPNativeExactError(
            f"native DeepEP runner output must be [recv_rows, hidden], got {tuple(recv_output.shape)}"
        )
    if recv_output.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            f"native DeepEP rank leaves must be BF16 before communication, got {recv_output.dtype}"
        )
    if not recv_output.is_contiguous():
        raise DeepEPNativeExactError("native DeepEP runner output must be contiguous")
    if int(dispatch_ctx.num_recv_tokens) != recv_output.shape[0]:
        raise DeepEPNativeExactError(
            "native DeepEP runner row count does not match its dispatch handle: "
            f"{recv_output.shape[0]} != {dispatch_ctx.num_recv_tokens}"
        )
    if int(dispatch_ctx.hidden_dim) != recv_output.shape[1]:
        raise DeepEPNativeExactError(
            "native DeepEP runner hidden width does not match its dispatch handle: "
            f"{recv_output.shape[1]} != {dispatch_ctx.hidden_dim}"
        )
    if num_local_experts <= 0:
        raise DeepEPNativeExactError("native DeepEP requires a positive local expert count")

    recv_ids = dispatch_ctx.recv_topk_idx
    recv_weights = dispatch_ctx.recv_topk_weights
    if recv_ids is None or recv_weights is None:
        raise DeepEPNativeExactError("native DeepEP dispatch did not retain receive top-k metadata")
    if recv_ids.ndim != 2 or recv_weights.shape != recv_ids.shape:
        raise DeepEPNativeExactError("native DeepEP receive ids and weights must have the same [recv_rows, topk] shape")
    if recv_ids.shape[0] != recv_output.shape[0]:
        raise DeepEPNativeExactError("native DeepEP receive metadata row count changed after dispatch")
    if recv_ids.dtype not in (torch.int32, torch.int64):
        raise DeepEPNativeExactError(f"native DeepEP receive ids must be integral, got {recv_ids.dtype}")
    if recv_weights.dtype is not torch.float32:
        raise DeepEPNativeExactError(
            f"native DeepEP receive routing weights must be FP32 metadata, got {recv_weights.dtype}"
        )

    valid = recv_ids >= 0
    if recv_ids.numel() and bool(torch.any(recv_ids < -1)):
        raise DeepEPNativeExactError("native DeepEP receive ids contain a marker below -1")
    if bool(torch.any(recv_ids[valid] >= num_local_experts)):
        raise DeepEPNativeExactError("native DeepEP delivered a route outside this rank's local expert slice")
    if recv_ids.shape[0] and bool(torch.any(~valid.any(dim=1))):
        raise DeepEPNativeExactError("native DeepEP delivered a receive row with no local route")
    if recv_weights.numel() and not bool(torch.isfinite(recv_weights).all()):
        raise DeepEPNativeExactError("native DeepEP receive routing weights are not finite")


def adapt_native_runner_metadata(
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt real DeepEP metadata to the shared serving-runner ABI."""

    if recv_topk_ids.ndim != 2 or recv_topk_weights.shape != recv_topk_ids.shape:
        raise DeepEPNativeExactError("native DeepEP runner metadata must share [recv_rows, topk] shape")
    if recv_topk_ids.dtype not in (torch.int32, torch.int64):
        raise DeepEPNativeExactError(f"native DeepEP runner expert ids must be integral, got {recv_topk_ids.dtype}")
    if recv_topk_weights.dtype is not torch.float32:
        raise DeepEPNativeExactError(f"native DeepEP runner weights must remain FP32, got {recv_topk_weights.dtype}")
    return (
        recv_topk_ids.to(torch.int32).contiguous(),
        recv_topk_weights.contiguous(),
    )


def reduce_native_runner_routes_to_bf16(
    route_output: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Reduce unweighted rank-local routes in FP32, then store a BF16 leaf.

    This compatibility helper is for a runner that explicitly returns one BF16
    expert result per receive-row/top-k slot. The selected native exact program
    instead uses the fused ``no_combine=False`` BF16 local leaf directly.
    """

    if route_output.ndim != 3:
        raise DeepEPNativeExactError("native DeepEP no-combine runner output must be [recv_rows, topk, hidden]")
    if route_output.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(f"native DeepEP runner routes must be BF16, got {route_output.dtype}")
    if recv_topk_ids.shape != route_output.shape[:2] or recv_topk_weights.shape != recv_topk_ids.shape:
        raise DeepEPNativeExactError(
            "native DeepEP runner routes and receive metadata have different row/top-k geometry"
        )
    if recv_topk_ids.dtype is not torch.int32 or recv_topk_weights.dtype is not torch.float32:
        raise DeepEPNativeExactError("native DeepEP runner reduction requires int32 ids and FP32 routing weights")
    valid = recv_topk_ids >= 0
    weighted_fp32 = torch.where(
        valid.unsqueeze(-1),
        route_output.to(torch.float32) * recv_topk_weights.unsqueeze(-1),
        torch.zeros((), dtype=torch.float32, device=route_output.device),
    )
    return weighted_fp32.sum(dim=1).to(torch.bfloat16).contiguous()


def native_zero_row_runner_routes(
    recv_hidden: torch.Tensor,
    recv_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Construct the no-combine runner result for an empty receive batch."""

    if recv_hidden.ndim != 2 or recv_hidden.shape[0] != 0 or recv_hidden.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError("native DeepEP zero-row bypass requires empty BF16 receive rows")
    if recv_topk_ids.ndim != 2 or recv_topk_ids.shape[0] != 0:
        raise DeepEPNativeExactError("native DeepEP zero-row metadata must be empty [0, topk]")
    return recv_hidden.new_empty((0, recv_topk_ids.shape[1], recv_hidden.shape[1]))


def native_exact_router_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build independent fixed-order FP32 routing metadata for DeepEP."""

    if router_logits.ndim != 2 or router_logits.dtype is not torch.float32:
        raise DeepEPNativeExactError("native DeepEP router logits must be FP32 [tokens, experts] metadata")
    if top_k <= 0 or top_k > router_logits.shape[1]:
        raise DeepEPNativeExactError("native DeepEP top-k is outside the expert geometry")

    from xorl.ops.batch_invariant_ops import bi_router_topk_weights  # noqa: PLC0415

    scores = torch.softmax(router_logits, dim=1, dtype=torch.float32)
    weights, expert_ids = torch.topk(scores, top_k, dim=-1)
    weights = bi_router_topk_weights(weights, renormalize, torch.bfloat16)
    return weights.to(torch.float32).contiguous(), expert_ids.contiguous()


def canonicalize_native_routing_metadata(routing_weights: torch.Tensor) -> torch.Tensor:
    """Preserve routing coefficients in DeepEP's required FP32 metadata ABI.

    Routing coefficients are kernel metadata, not expert-value wire payloads.
    Rounding an FP32 coefficient through BF16 here changes the fused
    ``no_combine=False`` rank leaf before its declared BF16 storage boundary
    and disagrees with serving, which supplies the original FP32 coefficient
    to the same fused kernel. Expert outputs and rank leaves remain BF16.
    """

    if routing_weights.dtype not in (torch.bfloat16, torch.float32):
        raise DeepEPNativeExactError(
            f"native DeepEP routing coefficients must be BF16 or FP32, got {routing_weights.dtype}"
        )
    if routing_weights.requires_grad:
        raise DeepEPNativeExactError("native DeepEP v1 requires a frozen router and frozen routing coefficients")
    return routing_weights.to(torch.float32).contiguous()


def _flatten_native_route_metadata(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    row_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten route metadata without an ambiguous ``reshape(0, -1)``.

    Idle/padded distributed ranks legitimately contribute zero token rows but
    must still enter DeepEP's collectives.  The top-k width is part of the
    metadata contract, so preserve that known width explicitly instead of
    asking PyTorch to infer it from zero elements.
    """

    if routing_weights.ndim < 2 or selected_experts.ndim < 2:
        raise DeepEPNativeExactError("native DeepEP route metadata must end in an explicit top-k dimension")
    top_k = int(routing_weights.shape[-1])
    if top_k <= 0 or int(selected_experts.shape[-1]) != top_k:
        raise DeepEPNativeExactError("native DeepEP selected experts and routing weights have different top-k geometry")
    expected = int(row_count) * top_k
    if routing_weights.numel() != expected or selected_experts.numel() != expected:
        raise DeepEPNativeExactError("native DeepEP route metadata does not cover every flattened token row")
    return (
        routing_weights.reshape(row_count, top_k).contiguous(),
        selected_experts.reshape(row_count, top_k).contiguous(),
    )


def native_dispatch_runner_combine(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    ep_group,
    num_experts: int,
    num_local_experts: int,
    buffer_size_gb: float,
    num_sms: int,
    runner: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    backward_layer_dependency: torch.Tensor | None = None,
    backward_shared_dependency: torch.Tensor | None = None,
    backward_trace_label: str | None = None,
    complete_backward_device_boundary: bool = False,
) -> torch.Tensor:
    """Own the complete reusable real-dispatch native DeepEP program.

    Model and LoRA adapters supply only their fused ``no_combine=False`` local
    expert runner.  This shared layer owns value/metadata validation, real
    top-k dispatch, runner ABI localization, BF16 rank leaves, handle-based
    combines, FP64 folding, and the reverse collectives installed by the
    combine autograd function. A route cube is rejected so the selected exact
    program cannot silently fall back to the superseded external reduction.
    """

    from xorl.distributed.moe.deepep import (  # noqa: PLC0415
        get_default_buffer,
        token_pre_dispatch_native,
    )

    if ep_group is None:
        raise DeepEPNativeExactError("native DeepEP requires a real EP process group")
    if hidden_states.ndim < 2:
        raise DeepEPNativeExactError("native DeepEP hidden states must end in a hidden dimension")
    if hidden_states.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(f"native DeepEP dispatch values must be BF16, got {hidden_states.dtype}")
    if routing_weights.dtype is not torch.float32:
        raise DeepEPNativeExactError(f"native DeepEP routing metadata must be FP32, got {routing_weights.dtype}")
    if routing_weights.requires_grad:
        raise DeepEPNativeExactError("native DeepEP v1 requires frozen FP32 routing metadata")
    original_shape = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    routing_flat, selected_flat = _flatten_native_route_metadata(
        routing_weights,
        selected_experts,
        row_count=hidden_flat.shape[0],
    )
    if selected_flat.shape != routing_flat.shape:
        raise DeepEPNativeExactError("native DeepEP selected experts and routing weights have different top-k geometry")

    geometry = resolve_native_deepep_geometry(ep_group, hidden_flat.shape[1])
    validate_native_combine_geometry(geometry.ep_size)
    if num_local_experts * geometry.ep_size != int(num_experts):
        raise DeepEPNativeExactError(
            "native DeepEP requires contiguous complete expert ownership: "
            f"{num_local_experts} local * {geometry.ep_size} ranks != {num_experts} experts"
        )
    buffer = get_default_buffer(
        ep_group=ep_group,
        buffer_size_gb=buffer_size_gb,
        num_sms=num_sms,
    )
    buffer.init_buffer(hidden_bytes=geometry.wire_hidden_bytes)
    recv_hidden, recv_local_ids, recv_weights, dispatch_ctx = token_pre_dispatch_native(
        buffer=buffer,
        hidden_states=hidden_flat,
        routing_weights=routing_flat,
        selected_experts=selected_flat,
        num_experts=num_experts,
        complete_backward_device_boundary=complete_backward_device_boundary,
        backward_trace_label=backward_trace_label,
        backward_layer_dependency=backward_layer_dependency,
        backward_shared_dependency=backward_shared_dependency,
    )
    if recv_hidden.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            f"native DeepEP dispatch returned {recv_hidden.dtype}; communication must stay BF16"
        )
    recv_local_ids, recv_weights = adapt_native_runner_metadata(
        recv_local_ids,
        recv_weights,
    )
    runner_output = runner(recv_hidden, recv_weights, recv_local_ids)
    recv_leaf = runner_output
    if recv_leaf.shape != recv_hidden.shape or recv_leaf.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            "native DeepEP fused no_combine=False runner must return one BF16 leaf per receive row"
        )
    folded = native_receive_combine_and_fold(
        recv_leaf.contiguous(),
        buffer=buffer,
        dispatch_ctx=dispatch_ctx,
        ep_group=ep_group,
        num_local_experts=num_local_experts,
        backward_layer_dependency=backward_layer_dependency,
        backward_trace_label=backward_trace_label,
    )
    return folded.reshape(original_shape)


def reduce_expert_rows_to_bf16_leaf(expert_output: torch.Tensor, dispatch_ctx) -> torch.Tensor:
    """Reduce expert-order route rows locally, then store one BF16 rank leaf.

    This helper is for grouped expert runners that emit one row per valid
    route.  Native receive-order runners should pass their already-reduced
    BF16 output directly to :func:`native_receive_combine_and_fold`.
    """

    if expert_output.ndim != 2:
        raise DeepEPNativeExactError("expert-order native output must be two-dimensional")
    if expert_output.shape[0] != dispatch_ctx.permuted_indices.numel():
        raise DeepEPNativeExactError("expert-order output does not cover every valid DeepEP route")
    if expert_output.shape[1] != int(dispatch_ctx.hidden_dim):
        raise DeepEPNativeExactError("expert-order output hidden width changed after DeepEP dispatch")
    if dispatch_ctx.permuted_indices.numel() and (
        int(dispatch_ctx.permuted_indices.min()) < 0
        or int(dispatch_ctx.permuted_indices.max()) >= int(dispatch_ctx.num_recv_tokens)
    ):
        raise DeepEPNativeExactError("DeepEP expert-order row index is outside the receive batch")

    # Local route arithmetic is deliberately wider than the BF16 wire/storage
    # boundary.  Out-of-place index_add keeps the operation differentiable.
    leaf_fp32 = torch.zeros(
        (int(dispatch_ctx.num_recv_tokens), int(dispatch_ctx.hidden_dim)),
        dtype=torch.float32,
        device=expert_output.device,
    )
    if expert_output.shape[0]:
        leaf_fp32 = leaf_fp32.index_add(
            0,
            dispatch_ctx.permuted_indices.to(torch.long),
            expert_output.to(torch.float32),
        )
    return leaf_fp32.to(torch.bfloat16).contiguous()


class _DeepEPDeterministicCombineBF16(torch.autograd.Function):
    """One topology-generic deterministic combine and one reverse dispatch."""

    @staticmethod
    def forward(
        ctx,
        local_leaf: torch.Tensor,
        buffer,
        dispatch_ctx,
        geometry: NativeDeepEPGeometry,
        backward_layer_dependency: torch.Tensor | None,
        backward_trace_label: str | None,
    ):
        del backward_layer_dependency
        if local_leaf.dtype is not torch.bfloat16 or not local_leaf.is_contiguous():
            raise DeepEPNativeExactError("DeepEP deterministic combine requires contiguous BF16 payload")
        if local_leaf.ndim != 2 or local_leaf.shape[1] != geometry.hidden_size:
            raise DeepEPNativeExactError("DeepEP deterministic combine received the wrong local-leaf geometry")
        try:
            from deep_ep import ReductionMode  # noqa: PLC0415
            from deep_ep.utils import EventHandle, EventOverlap  # noqa: PLC0415
        except (ImportError, AttributeError) as exc:
            raise DeepEPNativeExactError("installed DeepEP lacks ReductionMode.DETERMINISTIC") from exc
        from xorl.distributed.moe.deepep import _trace_deepep_boundary  # noqa: PLC0415

        reduction_mode = getattr(
            ReductionMode,
            "DETERMINISTIC",
            None,
        )
        if reduction_mode is None:
            raise DeepEPNativeExactError("installed DeepEP lacks ReductionMode.DETERMINISTIC")

        call_id = int(dispatch_ctx.call_id)
        _trace_deepep_boundary(
            call_id,
            "deterministic_combine_forward",
            "enter",
            trace_label=backward_trace_label,
        )
        previous_event = EventOverlap(EventHandle())
        combined, combined_weights, event = buffer.buffer.combine(
            x=local_leaf,
            handle=dispatch_ctx.handle,
            config=buffer.combine_config,
            reduction_mode=reduction_mode,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        combined.record_stream(torch.cuda.current_stream())
        if combined_weights is not None:
            raise DeepEPNativeExactError("DeepEP deterministic value combine unexpectedly returned routing metadata")
        if combined.dtype is not torch.bfloat16:
            raise DeepEPNativeExactError(f"DeepEP deterministic combine widened BF16 to {combined.dtype}")

        ctx.buffer = buffer
        ctx.handle = dispatch_ctx.handle
        ctx.geometry = geometry
        ctx.call_id = call_id
        ctx.backward_trace_label = backward_trace_label
        _trace_deepep_boundary(
            call_id,
            "deterministic_combine_forward",
            "exit",
            trace_label=backward_trace_label,
        )
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None
        from deep_ep.utils import EventHandle, EventOverlap  # noqa: PLC0415

        from xorl.distributed.moe.deepep import _trace_deepep_boundary  # noqa: PLC0415

        geometry = ctx.geometry
        if grad_output.ndim != 2 or grad_output.shape[1] != geometry.hidden_size:
            raise DeepEPNativeExactError("DeepEP deterministic backward received the wrong output geometry")
        _trace_deepep_boundary(
            ctx.call_id,
            "output_reverse_dispatch",
            "enter",
            trace_label=ctx.backward_trace_label,
        )
        grad_wire = grad_output.to(torch.bfloat16).contiguous()
        previous_event = EventOverlap(EventHandle())
        grad_recv, _, _, _, _, event = ctx.buffer.buffer.dispatch(
            x=grad_wire,
            handle=ctx.handle,
            config=ctx.buffer.dispatch_config,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        grad_recv.record_stream(torch.cuda.current_stream())
        if grad_recv.dtype is not torch.bfloat16:
            raise DeepEPNativeExactError(f"DeepEP deterministic backward widened BF16 to {grad_recv.dtype}")
        _trace_deepep_boundary(
            ctx.call_id,
            "output_reverse_dispatch",
            "exit",
            trace_label=ctx.backward_trace_label,
        )
        return grad_recv, None, None, None, None, None


def native_receive_combine_and_fold(
    recv_output: torch.Tensor,
    *,
    buffer,
    dispatch_ctx,
    ep_group,
    num_local_experts: int,
    backward_layer_dependency: torch.Tensor | None = None,
    backward_trace_label: str | None = None,
) -> torch.Tensor:
    """Transport native receive-order BF16 leaves with deterministic combine."""

    geometry = resolve_native_deepep_geometry(ep_group, recv_output.shape[1])
    validate_native_combine_geometry(geometry.ep_size)
    validate_native_receive_metadata(
        recv_output,
        dispatch_ctx,
        num_local_experts=num_local_experts,
    )
    combined = _DeepEPDeterministicCombineBF16.apply(
        recv_output,
        buffer,
        dispatch_ctx,
        geometry,
        backward_layer_dependency,
        backward_trace_label,
    )

    global _engagement_logged
    if not _engagement_logged:
        logger.info(
            "Native DeepEP exact combine ENGAGED: protocol=%s ep_size=%d "
            "wire_dtype=bf16 fold=%s wire_width=%d combine_calls=1 "
            "backward_schedule=single_reverse_dispatch_v1",
            DEEPEP_DETERMINISTIC_PROTOCOL,
            geometry.ep_size,
            "receiver_fp64_tree8_bf16_node_leaf_fp64_node_fold",
            geometry.wire_width,
        )
        _engagement_logged = True
    return combined


def native_expert_combine_and_fold(
    expert_output: torch.Tensor,
    *,
    buffer,
    dispatch_ctx,
    ep_group,
    num_local_experts: int,
) -> torch.Tensor:
    """Adapter for expert-order runners using deterministic combine."""

    local_leaf = reduce_expert_rows_to_bf16_leaf(expert_output, dispatch_ctx)
    return native_receive_combine_and_fold(
        local_leaf,
        buffer=buffer,
        dispatch_ctx=dispatch_ctx,
        ep_group=ep_group,
        num_local_experts=num_local_experts,
    )


__all__ = [
    "DEEPEP_DETERMINISTIC_PROTOCOL",
    "DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL",
    "DEEPEP_DETERMINISTIC_SUPPORTED_EP_SIZES",
    "DeepEPNativeExactError",
    "NativeDeepEPGeometry",
    "adapt_native_runner_metadata",
    "canonicalize_native_routing_metadata",
    "native_dispatch_runner_combine",
    "native_exact_router_topk",
    "native_expert_combine_and_fold",
    "native_receive_combine_and_fold",
    "native_zero_row_runner_routes",
    "reduce_expert_rows_to_bf16_leaf",
    "reduce_native_runner_routes_to_bf16",
    "resolve_native_deepep_geometry",
    "validate_native_receive_metadata",
    "validate_native_combine_geometry",
]
