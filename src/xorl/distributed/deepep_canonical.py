"""DeepEP-normal transport adapter for ``canonical_combine_v1``.

The adapter separates DeepEP transport from the canonical reduction:

1. PRIMARY dispatch — the ordinary rank-deduplicated DeepEP normal dispatch
   of hidden rows with top-k metadata. Route metadata is recorded here: the
   receipt inputs are what the transport delivered (``recv_topk_idx`` local
   ids with -1, ``recv_topk_weights`` FP32, ``recv_counts``), never
   re-derived from router state.
2. SLOT-LEVEL dispatch — a second, tiny dispatch whose logical tokens are
   the (token, slot) pairs (top-k = 1 each, payload = a 16-column BF16
   source fingerprint, DeepEP's minimum). Because every slot routes to
   exactly ONE rank, DeepEP's summing ``combine`` over this handle is a
   PURE per-slot transport (a sum with a single contributor), which is what
   makes DeepEP admissible under the canonical contract: the reduction
   stays outside the transport.
3. EXPERT-SIDE receive-layout validation — the slot-level receive order is
   VALIDATED against the primary receive (expert id equality, FP32 weight
   byte equality, fingerprint equality), then the mapping slot-row ->
   (primary row, k) is emitted. Fail closed on any mismatch.
4. RETURN combine — the expert rank packs [per-slot expert output row |
   4-column annotation block] and combines over the slot handle; owners
   receive rows in their own dispatch source order (slot-major).
5. OWNER receipt — built from the owner's emission order and the
   TRANSPORTED annotations (expert id, source token, k-slot encoded as
   exact small-integer BF16 values), then validated by
   ``validate_transport_receipt`` before ``canonical_combine`` runs.

The backward is trainer-owned: the return combine's transpose is a
handle-based reverse ``dispatch`` of the owner-side gradient rows, delivering
per-slot gradients to expert ranks in the validated slot receive order.

The slot-level dispatch adds route-metadata communication. A compatible
optimized implementation may instead derive the same counts and ordering
from the primary dispatch, provided it preserves all validation invariants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist

from xorl.distributed.canonical_combine import (
    CanonicalCombineError,
    CanonicalRouteMetadata,
    TransportReceipt,
)


logger = logging.getLogger(__name__)

DEEPEP_CANONICAL_TRANSPORT_ID = "deepep_normal_slot_v0"
_ANNOTATION_COLS = 16
_FINGERPRINT_COLS = 16

# DeepEP payload-alignment contract:
# - intranode TMA dispatch requires payload widths in multiples of 32 BYTES
#   (slime's documented minimum; 16 BF16 columns);
# - the INTERNODE kernels additionally assert
#   `EP_DEVICE_ASSERT(hidden_int4 % 32 == 0)` (internode.cu:1583, TMA reduce
#   path, DeepEP 1.2.1@567632d): payload widths must be multiples of
#   512 BYTES = 256 BF16 columns.
# All companion payloads are therefore padded to the internode alignment
# unconditionally (one code path; the pad is NaN-poisoned, never read, and
# excluded from receipts so the byte contract cannot silently widen), and
# admission fails closed on any unaligned width.
_INTRANODE_ALIGN_BYTES = 32
_INTERNODE_ALIGN_BYTES = 512
_BF16_BYTES = 2
_SLOT_META_COLS = _INTERNODE_ALIGN_BYTES // _BF16_BYTES  # 256 BF16 columns


class DeepEPReceiveAlignmentError(CanonicalCombineError):
    """Slot-level receive order disagrees with the primary dispatch."""


def validate_deepep_payload_alignment(width_elements: int, element_bytes: int = _BF16_BYTES) -> None:
    """Fail closed on payload widths the DeepEP kernels would reject or abort on.

    The internode constraint is enforced UNCONDITIONALLY: whether an EP group
    spans nodes is a launch-time property, so admission must not depend on
    whether the first execution happens to be intranode.
    """
    width_bytes = width_elements * element_bytes
    if width_bytes % _INTERNODE_ALIGN_BYTES:
        raise CanonicalCombineError(
            f"DeepEP payload width {width_elements} elems ({width_bytes} B) violates the "
            f"internode alignment (bytes % {_INTERNODE_ALIGN_BYTES} == 0; internode.cu:1583 "
            "hidden_int4 % 32). Pad with NaN-poisoned columns excluded from receipts."
        )


def _pad_cols_to_alignment(payload: torch.Tensor) -> torch.Tensor:
    """Right-pad columns with NaN to the internode alignment; never read back."""
    width = payload.shape[1]
    target = ((width * _BF16_BYTES + _INTERNODE_ALIGN_BYTES - 1) // _INTERNODE_ALIGN_BYTES) * (
        _INTERNODE_ALIGN_BYTES // _BF16_BYTES
    )
    if target == width:
        return payload.contiguous()
    pad = torch.full(
        (payload.shape[0], target - width), float("nan"), dtype=payload.dtype, device=payload.device
    )
    return torch.cat([payload, pad], dim=1).contiguous()


def pack_return_payload(rows: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
    """[rows H | annotations 16 | NaN pad to the internode alignment]."""
    packed = _pad_cols_to_alignment(torch.cat([rows, annotations], dim=1))
    validate_deepep_payload_alignment(packed.shape[1])
    return packed


def _require_deep_ep():
    from xorl.distributed.moe.deepep import (  # noqa: PLC0415
        DEEPEP_AVAILABLE,
        check_deepep_available,
    )

    check_deepep_available()
    return DEEPEP_AVAILABLE


@dataclass(frozen=True)
class SlotDispatchPlan:
    """Owner-side record of the slot-level dispatch source order."""

    slot_tokens: torch.Tensor  # [S] local token index per dispatched slot
    slot_ks: torch.Tensor  # [S] k column per dispatched slot
    slot_experts: torch.Tensor  # [S] GLOBAL expert id per dispatched slot
    handle: object
    num_slots: int


@dataclass(frozen=True)
class ExpertSlotView:
    """Expert-side validated mapping of slot-level rows to primary rows."""

    primary_row: torch.Tensor  # [Rs] index into recv_x
    primary_k: torch.Tensor  # [Rs] k column inside recv_topk_idx
    local_expert: torch.Tensor  # [Rs] local expert id per slot row
    num_slot_rows: int


def encode_annotations(
    slot_experts: torch.Tensor,
    slot_tokens: torch.Tensor,
    slot_ks: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """Pack (expert, token, k, source rank) as exact small-integer BF16 columns.

    BF16 represents integers exactly up to 256; ids are split into base-256
    digits so the transported annotation bytes decode exactly.
    """
    out = torch.zeros((slot_experts.numel(), _ANNOTATION_COLS), dtype=torch.bfloat16, device=slot_experts.device)
    experts = slot_experts.to(torch.int64)
    tokens = slot_tokens.to(torch.int64)
    values = (
        experts // 256,
        experts % 256,
        tokens // 256,
        tokens % 256,
        slot_ks.to(torch.int64) % 256,
        torch.full_like(experts, source_rank % 256),
    )
    for column, value in enumerate(values):
        out[:, column] = value.to(torch.bfloat16)
    return out


def decode_annotations(
    block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    digits = block.to(torch.float32).round().to(torch.int64)
    experts = digits[:, 0] * 256 + digits[:, 1]
    tokens = digits[:, 2] * 256 + digits[:, 3]
    ks = digits[:, 4]
    source_ranks = digits[:, 5]
    return experts, tokens, ks, source_ranks


def dispatch_primary(buffer, hidden: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, num_experts: int):
    """The ordinary rank-deduplicated DeepEP normal dispatch."""
    _require_deep_ep()
    from xorl.distributed.moe.deepep import dispatch_no_grad  # noqa: PLC0415

    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle = dispatch_no_grad(
        buffer, hidden.contiguous(), topk_weights, topk_ids, num_experts
    )
    return recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle


def dispatch_slot_level(
    buffer,
    hidden: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SlotDispatchPlan]:
    """Dispatch one 16-column fingerprint per ACTIVE (token, slot) pair.

    The REAL routing weight rides along per slot, so the expert-side
    alignment check compares transported bytes against transported bytes.
    """
    _require_deep_ep()
    from xorl.distributed.moe.deepep import dispatch_no_grad  # noqa: PLC0415

    if hidden.shape[1] < _FINGERPRINT_COLS:
        raise CanonicalCombineError(f"slot-level fingerprints require hidden >= {_FINGERPRINT_COLS}")
    active = topk_ids >= 0
    slot_tokens = (
        torch.arange(topk_ids.shape[0], device=topk_ids.device).unsqueeze(1).expand_as(topk_ids)[active]
    )
    slot_ks = (
        torch.arange(topk_ids.shape[1], device=topk_ids.device).unsqueeze(0).expand_as(topk_ids)[active]
    )
    slot_experts = topk_ids[active].to(torch.int64)

    fingerprints = hidden[:, :_FINGERPRINT_COLS].index_select(0, slot_tokens)
    annotations = encode_annotations(
        slot_experts, slot_tokens, slot_ks, dist.get_rank() if dist.is_initialized() else 0
    )
    payload = _pad_cols_to_alignment(torch.cat([fingerprints, annotations], dim=1))
    validate_deepep_payload_alignment(payload.shape[1])
    slot_idx = slot_experts.reshape(-1, 1)
    slot_weights = topk_weights[active].to(torch.float32).reshape(-1, 1).contiguous()
    recv_meta, recv_slot_idx, recv_slot_weights, recv_slot_counts, handle = dispatch_no_grad(
        buffer, payload, slot_weights, slot_idx, num_experts
    )
    plan = SlotDispatchPlan(
        slot_tokens=slot_tokens,
        slot_ks=slot_ks,
        slot_experts=slot_experts,
        handle=handle,
        num_slots=int(slot_tokens.numel()),
    )
    return recv_meta, recv_slot_idx, recv_slot_weights, plan


def validate_expert_receive_alignment(
    *,
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    recv_meta: torch.Tensor,
    recv_slot_idx: torch.Tensor,
    recv_slot_weights: torch.Tensor,
    num_local_experts: int,
    ep_rank: int,
) -> tuple[ExpertSlotView, torch.Tensor]:
    """slime's receive-layout validation, reduced to its invariants.

    Both dispatches enumerate this rank's routed slots. The primary side's
    slot list is built expert-major with stable arrival order; the
    slot-level side is expected in the SAME order (DeepEP produces the same
    source order for both dispatches — an invariant this function VALIDATES
    rather than assumes). Raise on any disagreement of expert id (local AND
    transported-global), FP32 weight bytes, or the 16-column source
    fingerprint. Returns the validated view plus the transported annotation
    block (passed through verbatim on the return combine so owners validate
    real delivery, not local bookkeeping).
    """
    valid = recv_topk_idx >= 0
    primary_positions = torch.nonzero(valid, as_tuple=False)
    primary_rows = primary_positions[:, 0]
    primary_ks = primary_positions[:, 1]
    primary_experts = recv_topk_idx[primary_rows, primary_ks].to(torch.int64)
    order = torch.argsort(primary_experts, stable=True)
    primary_rows = primary_rows.index_select(0, order)
    primary_ks = primary_ks.index_select(0, order)
    primary_experts = primary_experts.index_select(0, order)

    slot_valid = recv_slot_idx.reshape(-1) >= 0
    if int(slot_valid.sum()) != int(recv_slot_idx.numel()):
        raise DeepEPReceiveAlignmentError("slot-level dispatch delivered an invalid expert id")
    slot_experts = recv_slot_idx.reshape(-1).to(torch.int64)
    slot_order = torch.argsort(slot_experts, stable=True)
    slot_experts_sorted = slot_experts.index_select(0, slot_order)

    if primary_experts.numel() != slot_experts_sorted.numel():
        raise DeepEPReceiveAlignmentError(
            f"route count mismatch: primary={primary_experts.numel()} slot-level={slot_experts_sorted.numel()}"
        )
    if bool(torch.any(primary_experts != slot_experts_sorted)):
        raise DeepEPReceiveAlignmentError("expert-id order differs between primary and slot-level receives")

    expected_weights = recv_topk_weights[primary_rows, primary_ks]
    got_weights = recv_slot_weights.reshape(-1).index_select(0, slot_order)
    if not torch.equal(expected_weights.view(torch.int32), got_weights.view(torch.int32)):
        raise DeepEPReceiveAlignmentError(
            "FP32 routing-weight bytes differ between primary and slot-level receives"
        )

    fingerprints = recv_meta[:, :_FINGERPRINT_COLS]
    annotations = recv_meta[:, _FINGERPRINT_COLS : _FINGERPRINT_COLS + _ANNOTATION_COLS]
    expected_fp = recv_x[:, :_FINGERPRINT_COLS].index_select(0, primary_rows)
    got_fp = fingerprints.index_select(0, slot_order)
    if not torch.equal(expected_fp.view(torch.int16), got_fp.view(torch.int16)):
        raise DeepEPReceiveAlignmentError("source fingerprints differ between primary and slot-level receives")
    pad = recv_meta[:, _FINGERPRINT_COLS + _ANNOTATION_COLS :]
    if pad.numel() and not bool(torch.isnan(pad.float()).all()):
        raise DeepEPReceiveAlignmentError(
            "alignment padding arrived non-NaN: a writer touched bytes the contract says are dead"
        )

    ann_experts, _, _, _ = decode_annotations(annotations)
    expected_global = slot_experts + ep_rank * num_local_experts
    if bool(torch.any(ann_experts != expected_global)):
        raise DeepEPReceiveAlignmentError(
            "transported global expert ids disagree with the local receive layout"
        )

    inverse = torch.empty_like(slot_order)
    inverse[slot_order] = torch.arange(slot_order.numel(), device=slot_order.device)
    view = ExpertSlotView(
        primary_row=primary_rows.index_select(0, inverse),
        primary_k=primary_ks.index_select(0, inverse),
        local_expert=slot_experts,
        num_slot_rows=int(recv_slot_idx.numel()),
    )
    return view, annotations


class _SlotReturnCombine(torch.autograd.Function):
    """Per-slot return transport: combine forward, handle-based reverse dispatch backward."""

    @staticmethod
    def forward(ctx, payload: torch.Tensor, buffer, handle) -> torch.Tensor:
        from xorl.distributed.moe.deepep import combine_no_grad  # noqa: PLC0415

        ctx.buffer = buffer
        ctx.handle = handle
        return combine_no_grad(buffer, payload.contiguous(), handle)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from xorl.distributed.moe.deepep import EventHandle, EventOverlap  # noqa: PLC0415

        grad_recv, _, _, _, _, event = ctx.buffer.buffer.dispatch(
            x=grad_output.contiguous(),
            handle=ctx.handle,
            config=ctx.buffer.dispatch_config,
            previous_event=EventOverlap(EventHandle()),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        grad_recv.record_stream(torch.cuda.current_stream())
        return grad_recv, None, None


def return_slot_rows(payload: torch.Tensor, buffer, plan: SlotDispatchPlan) -> torch.Tensor:
    """Ship per-slot payload rows to owners; width must satisfy alignment."""
    validate_deepep_payload_alignment(payload.shape[1])
    return _SlotReturnCombine.apply(payload, buffer, plan.handle)


def build_owner_receipt(
    metadata: CanonicalRouteMetadata,
    plan: SlotDispatchPlan,
    returned: torch.Tensor,
    hidden_cols: int,
) -> tuple[torch.Tensor, TransportReceipt]:
    """Split returned rows and build the validated transport receipt.

    ``returned`` is [S, hidden_cols | 16 annotations | NaN alignment pad] in
    the slot-level dispatch's source order (slot-major); the layout is
    explicit via ``hidden_cols`` because the internode alignment pad makes
    width-based inference ambiguous. Annotations are the TRANSPORTED
    (expert, token, k) values — receipt validation checks the real
    delivery, not local bookkeeping. The pad is asserted NaN (dead bytes
    stayed dead across the return transport) and excluded from receipts.
    """
    if returned.shape[0] != plan.num_slots:
        raise CanonicalCombineError(
            f"return transport delivered {returned.shape[0]} rows, expected {plan.num_slots}"
        )
    if hidden_cols + _ANNOTATION_COLS > returned.shape[1]:
        raise CanonicalCombineError(
            f"returned width {returned.shape[1]} cannot hold hidden_cols={hidden_cols} + annotations"
        )
    rows = returned[:, :hidden_cols]
    ann_experts, ann_tokens, ann_ks, _ = decode_annotations(
        returned[:, hidden_cols : hidden_cols + _ANNOTATION_COLS].detach()
    )
    pad = returned[:, hidden_cols + _ANNOTATION_COLS :].detach()
    if pad.numel() and not bool(torch.isnan(pad.float()).all()):
        raise CanonicalCombineError(
            "return-path alignment padding arrived non-NaN: dead bytes were written in transit"
        )

    slot_to_row = torch.full_like(metadata.topk_ids, -1, dtype=torch.int64)
    slot_to_row[plan.slot_tokens, plan.slot_ks] = torch.arange(plan.num_slots, device=returned.device)
    receipt = TransportReceipt(
        slot_to_row=slot_to_row,
        num_rows=plan.num_slots,
        row_expert_ids=ann_experts,
        row_source_tokens=ann_tokens,
        transport_id=DEEPEP_CANONICAL_TRANSPORT_ID,
    )
    logger.info(
        "deepep canonical adapter engaged: slots=%d hidden=%d (validation deferred to canonical_combine)",
        plan.num_slots,
        hidden_cols,
    )
    return rows, receipt


def handle_is_intranode(handle) -> bool:
    """Classify a normal DeepEP handle by its tuple arity; fail closed otherwise.

    The intranode handle (six elements) carries ``recv_src_idx``, a true
    per-received-row source-token index. The internode handle (ten elements)
    carries ``recv_src_meta`` instead: an RDMA rank plus an NVLink-peer
    bitmask, not a source-token index. Derived receipts are therefore
    intranode-only; internode paths use transported annotations in the
    alignment-padded payload.
    """
    if not isinstance(handle, tuple):
        raise CanonicalCombineError(f"DeepEP handle must be a tuple, got {type(handle).__name__}")
    if len(handle) == 6:
        return True
    if len(handle) == 10:
        return False
    raise CanonicalCombineError(f"unsupported DeepEP handle length {len(handle)}")


def _handle_recv_src(handle) -> torch.Tensor:
    """Extract per-received-row source-token indices (intranode handles only)."""
    if not handle_is_intranode(handle):
        raise DeepEPReceiveAlignmentError(
            "internode DeepEP handles carry SourceMeta (src_rdma_rank + NVL bitmask), not "
            "per-row source-token indices — the derived receipt path is intranode-only; "
            "use transported annotations for internode execution"
        )
    src = handle[3]  # intranode recv_src_idx
    if not isinstance(src, torch.Tensor):
        raise CanonicalCombineError("DeepEP handle source metadata is not a tensor")
    if src.shape[0] == 0:  # ranks that received nothing (e.g. empty-expert skews)
        return src.reshape(0).to(torch.int64)
    return src.reshape(src.shape[0], -1)[:, 0].to(torch.int64)


def crosscheck_primary_source_indices(
    *,
    primary_handle,
    slot_handle,
    view: ExpertSlotView,
    topk: int,
) -> None:
    """Cross-check the source mapping derived from intranode handles.

    Instead of transporting annotations, derive source positions from the two
    handles' own source metadata: the slot handle's per-row source index is
    the source rank's FLAT slot index, and with a FULL valid top-k (the
    production router condition; slime's assume_all_routes_valid) it
    decodes as token = flat // K, k = flat % K. The decoded token must
    equal the PRIMARY handle's source-token index for the mapped primary
    row, and the decoded k must equal the mapped primary k column, for
    every slot row. Fail closed on any disagreement — the decode is
    undefined under dropping routers.
    """
    primary_src = _handle_recv_src(primary_handle)
    slot_src = _handle_recv_src(slot_handle)
    if slot_src.numel() != view.num_slot_rows:
        raise DeepEPReceiveAlignmentError(
            f"slot handle carries {slot_src.numel()} source rows, expected {view.num_slot_rows}"
        )
    decoded_tokens = slot_src // topk
    expected_tokens = primary_src.index_select(0, view.primary_row)
    if bool(torch.any(decoded_tokens != expected_tokens)):
        raise DeepEPReceiveAlignmentError(
            "primary-handle source tokens disagree with slot-handle flat indices "
            "(derived receipt path requires a full valid top-k and consistent source order)"
        )
    decoded_ks = slot_src % topk
    if bool(torch.any(decoded_ks != view.primary_k)):
        raise DeepEPReceiveAlignmentError("slot-handle k columns disagree with the primary receive layout")


def build_owner_receipt_derived(
    metadata: CanonicalRouteMetadata,
    plan: SlotDispatchPlan,
    returned: torch.Tensor,
) -> tuple[torch.Tensor, TransportReceipt]:
    """Build an owner receipt without annotation columns.

    ``returned`` is [S, H] exactly (the payload carries only expert output
    rows). The receipt's row annotations are DERIVED from the owner's own
    dispatch source order — admissible because (a) DeepEP's combine returns
    rows in the slot dispatch's source order by layout construction, and
    (b) the real cross-transport teeth moved to the expert side
    (fingerprint validation + crosscheck_primary_source_indices). The
    owner-side validation still runs as a self-consistency check rather than
    a transport check. Requires a full valid top-k (fail closed otherwise).
    """
    if bool(torch.any(~metadata.active_mask)):
        raise CanonicalCombineError(
            "derived receipts require a full valid top-k (inactive slots present); "
            "use the transported-annotation path for dropping routers"
        )
    if returned.shape[0] != plan.num_slots:
        raise CanonicalCombineError(
            f"return transport delivered {returned.shape[0]} rows, expected {plan.num_slots}"
        )
    slot_to_row = torch.full_like(metadata.topk_ids, -1, dtype=torch.int64)
    slot_to_row[plan.slot_tokens, plan.slot_ks] = torch.arange(plan.num_slots, device=returned.device)
    receipt = TransportReceipt(
        slot_to_row=slot_to_row,
        num_rows=plan.num_slots,
        row_expert_ids=plan.slot_experts,
        row_source_tokens=plan.slot_tokens,
        transport_id=DEEPEP_CANONICAL_TRANSPORT_ID + "+derived_receipt",
    )
    logger.info(
        "deepep canonical adapter engaged (derived receipt): slots=%d hidden=%d",
        plan.num_slots,
        returned.shape[1],
    )
    return returned, receipt


__all__ = [
    "DEEPEP_CANONICAL_TRANSPORT_ID",
    "DeepEPReceiveAlignmentError",
    "ExpertSlotView",
    "SlotDispatchPlan",
    "build_owner_receipt",
    "build_owner_receipt_derived",
    "crosscheck_primary_source_indices",
    "decode_annotations",
    "dispatch_primary",
    "dispatch_slot_level",
    "encode_annotations",
    "handle_is_intranode",
    "pack_return_payload",
    "return_slot_rows",
    "validate_deepep_payload_alignment",
    "validate_expert_receive_alignment",
]
