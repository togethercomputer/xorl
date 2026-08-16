"""Canonical per-token top-k MoE combine: transport-independent reduction.

This module separates TRANSPORT from REDUCTION for the MoE combine:

- The REDUCTION is one canonical program: for every token, its top-k expert
  contributions are accumulated in FP32 in ascending router-emitted slot order
  (k = 0..K-1), inactive slots are SKIPPED (never added, not added-as-zero),
  and the FP32 accumulator is rounded to BF16 exactly once at the end. Every
  output element has exactly one writer.

- A TRANSPORT is any mechanism (ordered EP all-to-all, DeepEP normal, DeepEP
  low-latency, single-node no-a2a, or an in-process simulation) that delivers
  the per-(token, slot) UNWEIGHTED expert contribution rows, in any physical
  row order and with any padding/grouping, together with a
  :class:`TransportReceipt` mapping every active canonical slot to its
  delivered row. A transport is admissible if and only if its receipt passes
  :func:`validate_transport_receipt`; combined bytes are then independent of
  the transport by construction.

The canonical reduction is shaped after the serving side's low-latency
combine (``sglang.kernels.ops.moe.ep_moe_kernels._fwd_kernel_ep_gather`` at
the pinned xorl-sglang submodule): per-token FP32 accumulator, slot-order
loop, ``acc += bf16(row).float() * fp32_weight``, one BF16 store. The Triton
backend here is a structural mirror of that kernel; the torch reference
backend is an independent implementation of the same tree (shared code must
not self-confirm a trainer/sampler pair). Divergence between them is a
contract violation: the executed serving program, not its API name, defines
the required arithmetic.

The backward is trainer-owned and deterministic without atomics:

- ``d contributions``: each delivered row has exactly one writing slot
  (receipt injectivity), so the route gradient is a pure one-writer scatter
  of ``bf16(grad_token.float() * weight)``.
- ``d topk_weights``: per active slot ``dot(grad_token, row)`` accumulated in
  FP32, materialized in canonical slot-major order so its bytes are also
  transport-independent.

Forward bytes are the contract; the backward carries run-to-run determinism
and cross-transport byte stability, not a cross-engine contract.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)

CANONICAL_COMBINE_VERSION = "canonical_combine_v1"

_INACTIVE = -1
_ENGAGEMENT_LOGGED: set[tuple[str, str, int, int, int]] = set()


class CanonicalCombineError(ValueError):
    """A canonical-combine contract violation. Always fail closed."""


class TransportReceiptError(CanonicalCombineError):
    """The transport's receive layout does not satisfy the admission criteria."""


# ---------------------------------------------------------------------------
# Route metadata (transport-independent canonical program inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalRouteMetadata:
    """The canonical route program for one MoE invocation.

    ``topk_ids``/``topk_weights`` are ``[T, K]`` in the ROUTER'S EMISSION
    ORDER. That slot order is part of the numerical program: the canonical
    reduction consumes slots in ascending k and never re-sorts. Both engines
    must therefore produce this tensor from the same router program
    (including its top-k tie-breaking); this module validates shape and
    self-consistency, not router equality — compare
    :func:`route_metadata_digest` across engines for that.

    ``topk_ids[t, k] == -1`` marks an inactive slot (dropped or padded);
    inactive slots are skipped by the reduction and carry no weight.
    """

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_experts: int
    version: str = CANONICAL_COMBINE_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_tokens(self) -> int:
        return int(self.topk_ids.shape[0])

    @property
    def topk(self) -> int:
        return int(self.topk_ids.shape[1])

    @property
    def active_mask(self) -> torch.Tensor:
        return self.topk_ids >= 0

    def validate(self) -> None:
        if self.version != CANONICAL_COMBINE_VERSION:
            raise CanonicalCombineError(f"Unsupported canonical combine version: {self.version!r}")
        ids, weights = self.topk_ids, self.topk_weights
        if ids.ndim != 2 or weights.ndim != 2 or ids.shape != weights.shape:
            raise CanonicalCombineError(
                f"topk_ids and topk_weights must be equal-shape [T, K]; got {tuple(ids.shape)} and {tuple(weights.shape)}"
            )
        if ids.dtype not in (torch.int32, torch.int64):
            raise CanonicalCombineError(f"topk_ids must be int32/int64, got {ids.dtype}")
        if weights.dtype is not torch.float32:
            raise CanonicalCombineError(
                f"topk_weights must be FP32 (weights are applied at combine), got {weights.dtype}"
            )
        if self.num_experts <= 0:
            raise CanonicalCombineError(f"num_experts must be positive, got {self.num_experts}")
        if bool(torch.any(ids >= self.num_experts)):
            raise CanonicalCombineError("topk_ids contains an expert id >= num_experts")
        if bool(torch.any(ids < _INACTIVE)):
            raise CanonicalCombineError("topk_ids below -1 are not a defined inactive encoding")
        # Duplicate active experts within one token would make the slot order
        # ambiguous under any expert-keyed transport; the router never emits
        # them, so their appearance is a corrupted-metadata signal.
        if ids.numel():
            sorted_ids, _ = torch.sort(ids, dim=1)
            dup = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)
            if bool(torch.any(dup)):
                raise CanonicalCombineError("a token repeats an active expert id in its top-k slots")
        if ids.numel() and bool(torch.any(~torch.isfinite(weights[self.active_mask]))):
            raise CanonicalCombineError("active topk_weights must be finite FP32 values")


def route_metadata_digest(metadata: CanonicalRouteMetadata) -> str:
    """Compact cross-engine digest of the canonical route program.

    Engines exchange this digest (not the tensors) to check that they hold
    the identical route program — same shapes, same slot order, same expert
    ids, same FP32 weight bytes — before trusting byte parity of the combine.
    """
    hasher = hashlib.sha256()
    hasher.update(CANONICAL_COMBINE_VERSION.encode())
    hasher.update(str((metadata.num_tokens, metadata.topk, metadata.num_experts)).encode())
    hasher.update(metadata.topk_ids.detach().to(torch.int64).cpu().contiguous().numpy().tobytes())
    hasher.update(metadata.topk_weights.detach().cpu().contiguous().view(torch.int32).numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Transport receipt (what any admissible transport must supply)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportReceipt:
    """The transport's claim about where each canonical slot's row landed.

    ``slot_to_row[t, k]`` is the row index of slot (t, k)'s unweighted BF16
    contribution inside the delivered ``[num_rows, H]`` tensor, or ``-1`` at
    inactive slots. ``row_expert_ids``/``row_source_tokens`` annotate every
    delivered row (``-1`` on pure padding rows); they exist so the mapping is
    VALIDATED against the transport's own delivery rather than trusted
    (slime PR 2262's receive-layout validation, reduced to its invariants).
    """

    slot_to_row: torch.Tensor
    num_rows: int
    row_expert_ids: torch.Tensor
    row_source_tokens: torch.Tensor
    transport_id: str


def validate_transport_receipt(
    metadata: CanonicalRouteMetadata,
    receipt: TransportReceipt,
) -> None:
    """Admission gate: raise unless the receipt satisfies every criterion.

    1. Alignment: ``slot_to_row`` is ``[T, K]`` like the metadata; row
       annotations are ``[num_rows]``.
    2. Coverage: every ACTIVE slot maps to a row in ``[0, num_rows)``;
       every INACTIVE slot maps to ``-1``.
    3. Injectivity: no delivered row is claimed by two slots (this is what
       makes the backward a one-writer scatter without atomics).
    4. Receive-layout validation: for every active slot,
       ``row_expert_ids[row] == topk_ids[t, k]`` and
       ``row_source_tokens[row] == t``.
    5. Two-sided accounting: every row annotated as real
       (``row_source_tokens >= 0``) is claimed by exactly one slot, and
       padding rows (``-1``) are claimed by none. Unclaimed rows are never
       read by the reduction.
    """
    if not receipt.transport_id:
        raise TransportReceiptError("transport_id must name the transport program")
    mapping = receipt.slot_to_row
    if mapping.shape != metadata.topk_ids.shape:
        raise TransportReceiptError(
            f"slot_to_row shape {tuple(mapping.shape)} does not match route metadata {tuple(metadata.topk_ids.shape)}"
        )
    if mapping.dtype not in (torch.int32, torch.int64):
        raise TransportReceiptError(f"slot_to_row must be int32/int64, got {mapping.dtype}")
    if receipt.num_rows < 0:
        raise TransportReceiptError(f"num_rows must be nonnegative, got {receipt.num_rows}")
    for name, annotation in (
        ("row_expert_ids", receipt.row_expert_ids),
        ("row_source_tokens", receipt.row_source_tokens),
    ):
        if annotation.ndim != 1 or annotation.numel() != receipt.num_rows:
            raise TransportReceiptError(f"{name} must be a rank-1 tensor of num_rows={receipt.num_rows} entries")
        if annotation.dtype not in (torch.int32, torch.int64):
            raise TransportReceiptError(f"{name} must be int32/int64, got {annotation.dtype}")

    active = metadata.active_mask
    if bool(torch.any(mapping[~active] != _INACTIVE)):
        raise TransportReceiptError("inactive slots must map to -1")
    active_rows = mapping[active].to(torch.int64)
    if bool(torch.any(active_rows < 0)) or bool(torch.any(active_rows >= receipt.num_rows)):
        raise TransportReceiptError("an active slot maps outside the delivered rows")

    claims = torch.bincount(active_rows, minlength=receipt.num_rows)
    if bool(torch.any(claims > 1)):
        raise TransportReceiptError("two slots claim one delivered row (injectivity violated)")

    token_index = torch.arange(metadata.num_tokens, device=mapping.device).unsqueeze(1).expand_as(mapping)
    claimed_experts = receipt.row_expert_ids.to(torch.int64).index_select(0, active_rows)
    claimed_tokens = receipt.row_source_tokens.to(torch.int64).index_select(0, active_rows)
    if bool(torch.any(claimed_experts != metadata.topk_ids[active].to(torch.int64))):
        raise TransportReceiptError(
            "receive-layout validation failed: a delivered row's expert id disagrees with its slot"
        )
    if bool(torch.any(claimed_tokens != token_index[active].to(torch.int64))):
        raise TransportReceiptError(
            "receive-layout validation failed: a delivered row's source token disagrees with its slot"
        )

    real_rows = receipt.row_source_tokens.to(torch.int64) >= 0
    if bool(torch.any(claims[real_rows] != 1)):
        raise TransportReceiptError(
            "a delivered real row is claimed by no slot (a contribution would be dropped silently)"
        )
    if bool(torch.any(claims[~real_rows] != 0)):
        raise TransportReceiptError("a slot claims a padding row")


def _log_engagement(metadata: CanonicalRouteMetadata, receipt: TransportReceipt, backend: str) -> None:
    key = (receipt.transport_id, backend, metadata.num_tokens, metadata.topk, receipt.num_rows)
    if key in _ENGAGEMENT_LOGGED:
        return
    _ENGAGEMENT_LOGGED.add(key)
    logger.info(
        "canonical_combine_v1 engaged: transport=%s backend=%s tokens=%d topk=%d rows=%d digest=%s",
        receipt.transport_id,
        backend,
        metadata.num_tokens,
        metadata.topk,
        receipt.num_rows,
        route_metadata_digest(metadata)[:16],
    )


# ---------------------------------------------------------------------------
# Canonical reduction backends
# ---------------------------------------------------------------------------


def _combine_forward_reference(
    contributions: torch.Tensor,
    topk_weights: torch.Tensor,
    slot_to_row: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Torch reference of the canonical tree: slot-order FP32 FMA chain.

    The serving program is an FMA chain: Triton contracts ``acc += v * w``
    into fused multiply-add, so the fp32 product is not rounded before the
    add. A mul-then-add transcription can agree at one hidden size and diverge
    at another; the executed kernel is the contract.

    Each step emulates one fp32 FMA in fp64: with 24-bit significands the
    product is exact in fp64, and one fp64 add followed by the fp32 round is
    the fused operation up to a theoretical fp64->fp32 double-rounding
    corner that the GPU identity gates continuously police. Inactive slots
    are SKIPPED (their would-be rows never reach the accumulator).
    """
    num_tokens, topk = slot_to_row.shape
    hidden = contributions.shape[1]
    acc = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=contributions.device)
    safe_rows = slot_to_row.to(torch.int64).clamp_min(0)
    for k in range(topk):
        rows = contributions.index_select(0, safe_rows[:, k]).to(torch.float64)
        update = (acc.to(torch.float64) + rows * topk_weights[:, k].to(torch.float64).unsqueeze(1)).to(torch.float32)
        acc = torch.where(active[:, k].unsqueeze(1), update, acc)
    return acc.to(contributions.dtype)


try:  # Triton is optional on CPU-only environments.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _canonical_combine_fwd_kernel(
        total_token_num,
        input_tensor,
        input_stride0,
        topk_ids,
        topk_ids_stride0,
        topk_weights,
        topk_weights_stride0,
        slot_to_row,
        slot_to_row_stride0,
        output_tensor,
        output_stride0,
        topk_num: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # Structural mirror of sglang _fwd_kernel_ep_gather (bitwise identity
        # verified at submodule 1d84365de; the pin has since moved — see the
        # contract doc's verification status): one program per (hidden block,
        # token stripe); per token
        # an FP32 accumulator over slots in ascending k; inactive slots are
        # skipped inside the loop; one store per output element.
        cur_block = tl.program_id(0).to(tl.int64)
        start_token = tl.program_id(1)
        grid_num = tl.num_programs(1)

        for token_i32 in range(start_token, total_token_num, grid_num):
            token = token_i32.to(tl.int64)
            off_d = tl.arange(0, BLOCK_D)
            accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
            for k_i32 in range(0, topk_num):
                k = k_i32.to(tl.int64)
                expert_id = tl.load(topk_ids + token * topk_ids_stride0 + k)
                if expert_id >= 0:
                    row_i32 = tl.load(slot_to_row + token * slot_to_row_stride0 + k)
                    row = row_i32.to(tl.int64)
                    weight = tl.load(topk_weights + token * topk_weights_stride0 + k)
                    values = tl.load(input_tensor + row * input_stride0 + cur_block * BLOCK_D + off_d)
                    accumulator += values.to(tl.float32) * weight
            tl.store(
                output_tensor + token * output_stride0 + cur_block * BLOCK_D + off_d,
                accumulator.to(output_tensor.dtype.element_ty),
            )

    def _combine_forward_triton(
        contributions: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        slot_to_row: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, topk = slot_to_row.shape
        hidden = contributions.shape[1]
        output = torch.empty((num_tokens, hidden), dtype=contributions.dtype, device=contributions.device)
        if num_tokens == 0:
            return output
        block_d = 128 if hidden % 1024 != 0 else 1024
        if hidden % block_d:
            raise CanonicalCombineError(
                f"triton canonical combine requires hidden divisible by {block_d}, got {hidden}"
            )
        grid = (hidden // block_d, min(num_tokens, 1024))
        _canonical_combine_fwd_kernel[grid](
            num_tokens,
            contributions,
            contributions.stride(0),
            topk_ids,
            topk_ids.stride(0),
            topk_weights,
            topk_weights.stride(0),
            slot_to_row,
            slot_to_row.stride(0),
            output,
            output.stride(0),
            topk_num=topk,
            BLOCK_D=block_d,
            num_warps=2,
        )
        return output


def _resolve_backend(backend: str, device: torch.device) -> str:
    if backend not in ("auto", "reference", "triton"):
        raise CanonicalCombineError(f"Unknown canonical combine backend: {backend!r}")
    if backend == "auto":
        return "triton" if (device.type == "cuda" and _TRITON_AVAILABLE) else "reference"
    if backend == "triton":
        if not _TRITON_AVAILABLE:
            raise CanonicalCombineError("triton backend requested but triton is not importable")
        if device.type != "cuda":
            raise CanonicalCombineError("triton backend requires CUDA tensors")
    return backend


# ---------------------------------------------------------------------------
# Autograd operator
# ---------------------------------------------------------------------------


class _CanonicalCombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        contributions: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        slot_to_row: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        active = topk_ids >= 0
        if backend == "triton":
            output = _combine_forward_triton(
                contributions.contiguous(),
                topk_ids.contiguous(),
                topk_weights.contiguous(),
                slot_to_row.contiguous(),
            )
        else:
            output = _combine_forward_reference(contributions, topk_weights, slot_to_row, active)
        ctx.save_for_backward(contributions, topk_weights, slot_to_row, active)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        contributions, topk_weights, slot_to_row, active = ctx.saved_tensors
        needs_contrib = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[1]
        grad_contributions = None
        grad_weights = None

        # Canonical slot-major flattening (t-major, then k): every derived
        # tensor below is built in this order, so backward bytes cannot
        # depend on the transport's physical row order.
        active_flat = active.reshape(-1)
        slot_rows = slot_to_row.to(torch.int64).reshape(-1)[active_flat]
        slot_tokens = (
            torch.arange(active.shape[0], device=active.device).unsqueeze(1).expand_as(active).reshape(-1)[active_flat]
        )
        grad_tokens_fp32 = grad_output.to(torch.float32).index_select(0, slot_tokens)

        if needs_contrib:
            slot_weights = topk_weights.reshape(-1)[active_flat]
            per_slot = (grad_tokens_fp32 * slot_weights.unsqueeze(1)).to(contributions.dtype)
            grad_contributions = torch.zeros_like(contributions)
            # One writer per row: receipt injectivity makes slot_rows unique.
            grad_contributions.index_copy_(0, slot_rows, per_slot)

        if needs_weights:
            rows_fp32 = contributions.to(torch.float32).index_select(0, slot_rows)
            per_slot_dot = (grad_tokens_fp32 * rows_fp32).sum(dim=1)
            flat = torch.zeros(topk_weights.numel(), dtype=topk_weights.dtype, device=topk_weights.device)
            flat.index_copy_(
                0,
                torch.nonzero(active_flat, as_tuple=False).reshape(-1),
                per_slot_dot,
            )
            grad_weights = flat.reshape(topk_weights.shape)

        return grad_contributions, grad_weights, None, None, None


def canonical_combine(
    contributions: torch.Tensor,
    metadata: CanonicalRouteMetadata,
    receipt: TransportReceipt,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Reduce delivered per-slot rows under the canonical_combine_v1 program.

    ``contributions`` is the transport-delivered ``[num_rows, H]`` BF16
    tensor of UNWEIGHTED per-(token, slot) expert outputs. Returns the
    combined ``[T, H]`` BF16 tokens. Byte-equal across every admissible
    transport by construction.

    The receipt is ALWAYS validated: admission is what makes the backward a
    one-writer scatter, so an unvalidated receipt (e.g. duplicate row
    claims) would silently turn gradients overwrite/order-dependent. There
    is deliberately no bypass switch — a fail-open flag on a byte contract
    is a contract violation, not an optimization knob.
    """
    if contributions.ndim != 2:
        raise CanonicalCombineError(f"contributions must be [rows, hidden], got {tuple(contributions.shape)}")
    if contributions.dtype is not torch.bfloat16:
        raise CanonicalCombineError(f"canonical_combine_v1 admits BF16 contributions only, got {contributions.dtype}")
    if contributions.shape[0] != receipt.num_rows:
        raise CanonicalCombineError(
            f"contributions rows {contributions.shape[0]} disagree with receipt num_rows {receipt.num_rows}"
        )
    validate_transport_receipt(metadata, receipt)
    resolved = _resolve_backend(backend, contributions.device)
    _log_engagement(metadata, receipt, resolved)
    return _CanonicalCombineFunction.apply(
        contributions,
        metadata.topk_weights,
        metadata.topk_ids,
        receipt.slot_to_row,
        resolved,
    )


__all__ = [
    "CANONICAL_COMBINE_VERSION",
    "CanonicalCombineError",
    "CanonicalRouteMetadata",
    "TransportReceipt",
    "TransportReceiptError",
    "canonical_combine",
    "route_metadata_digest",
    "validate_transport_receipt",
]
