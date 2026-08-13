"""Canonical scan chain for exact GDN under Ulysses context parallelism.

Cross-rank recurrent state moves as a sequential fp32 final-state handoff in
declared canonical order (ascending rank equals document order). Each hop
transports operands, never results: the fp32 boundary state is captured
bit-exactly by the unmodified U1 scan kernel and re-injected bit-exactly
downstream. No arithmetic happens on the wire and no summary or fold
arithmetic exists outside the canonical U1 composition.

A chain hop is admitted only if its receipt validates before any dependent
compute; validation failures raise (fail closed), and engagement is logged
once per (layer, direction) so a silent no-op is impossible.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist


logger = logging.getLogger("xorl.gdn_cp_chain")

_DIGEST_BYTES = 32
_CHUNK = 64

# Engagement bookkeeping: {tag: hop_count}. Logged once per tag; counters
# always increment so gates can assert coverage.
_ENGAGEMENTS: dict[str, int] = {}


class ChainReceiptError(RuntimeError):
    """A chain hop failed admission. Fail closed: never fall back."""


@dataclass(frozen=True)
class ChainReceipt:
    """Byte-relevant facts about one state handoff, derived independently on
    both sides from shared collator/cp metadata (everything except
    `state_digest`, which travels with the payload and certifies transport).

    in_doc_offset is the receiving shard's start measured from the crossing
    document's start (== cp_context.pre_num_conv_tokens for the first local
    sequence): the canonical grid demands it be a multiple of 64.
    """

    sender_rank: int
    receiver_rank: int
    in_doc_offset: int
    state_shape: tuple[int, ...]
    state_dtype: torch.dtype
    state_digest: bytes | None = None


def state_digest(state: torch.Tensor) -> bytes:
    return hashlib.sha256(state.detach().contiguous().cpu().numpy().tobytes()).digest()


def _record_engagement(tag: str, message: str) -> None:
    if tag not in _ENGAGEMENTS:
        _ENGAGEMENTS[tag] = 0
        logger.info("gdn-cp chain engaged: %s", message)
    _ENGAGEMENTS[tag] += 1


def engagement_count(tag: str) -> int:
    return _ENGAGEMENTS.get(tag, 0)


def validate_chain_receipt(receipt: ChainReceipt, *, state: torch.Tensor) -> None:
    """Admission check, run BEFORE any compute consumes the received state."""
    if receipt.sender_rank != receipt.receiver_rank - 1:
        raise ChainReceiptError(
            f"non-canonical hop order: sender {receipt.sender_rank} -> "
            f"receiver {receipt.receiver_rank}; the declared order is ascending rank",
        )
    if receipt.in_doc_offset <= 0 or receipt.in_doc_offset % _CHUNK != 0:
        raise ChainReceiptError(
            f"shard boundary at in-document offset {receipt.in_doc_offset} is not on the "
            f"document's {_CHUNK}-token chunk grid; the collator alignment contract "
            "is violated",
        )
    if receipt.state_dtype != torch.float32 or state.dtype != torch.float32:
        raise ChainReceiptError(
            f"chain state must be fp32 end-to-end, got receipt={receipt.state_dtype}, "
            f"tensor={state.dtype}",
        )
    if tuple(state.shape) != tuple(receipt.state_shape):
        raise ChainReceiptError(
            f"state shape {tuple(state.shape)} != receipt shape {tuple(receipt.state_shape)}",
        )
    if receipt.state_digest is None:
        raise ChainReceiptError("receipt carries no transport digest")
    actual = state_digest(state)
    if actual != receipt.state_digest:
        raise ChainReceiptError(
            "state digest mismatch after transport: the wire did not preserve the "
            f"fp32 bytes (expected {receipt.state_digest.hex()[:16]}..., "
            f"got {actual.hex()[:16]}...)",
        )


def _wire_device(group: dist.ProcessGroup | None) -> str:
    return "cpu" if dist.get_backend(group) == "gloo" else "cuda"


def chain_send_state(
    state: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup | None = None,
    *,
    tag_prefix: str = "gdn",
) -> None:
    """Send one fp32 boundary state + its digest to the next rank (P2P).

    The payload is the state's raw bytes (contiguous fp32) followed by a
    32-byte SHA-256 digest tensor. Cast-free: fp32 in, fp32 on the wire.
    """
    if state.dtype != torch.float32:
        raise ChainReceiptError(f"refusing to send non-fp32 chain state ({state.dtype})")
    wire = _wire_device(group)
    payload = state.detach().contiguous().to(wire)
    digest = torch.frombuffer(bytearray(state_digest(state)), dtype=torch.uint8).to(wire)
    dist.send(payload, dst=dst, group=group)
    dist.send(digest, dst=dst, group=group)
    _record_engagement(
        f"{tag_prefix}:send",
        f"rank {dist.get_rank(group)} -> {dst}, state {tuple(state.shape)} fp32",
    )


def chain_pre_scan_initial_state(
    *,
    num_seqs: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    context,
    device: torch.device | str,
    tag_prefix: str = "gdn",
) -> torch.Tensor:
    """Receive-side of the canonical scan chain for one GDN forward.

    Returns the `[N, H, K, V]` fp32 initial-state tensor for the local
    fragment scan: row 0 carries the received boundary state when the first
    local sequence continues a document from the previous rank; all other
    rows (locally-starting sequences) stay zero. Receipt validation happens
    inside chain_recv_state BEFORE the state is returned (fail closed).
    """
    initial_state = torch.zeros(num_seqs, heads, key_dim, value_dim,
                                dtype=torch.float32, device=device)
    if not context.is_first_rank:
        in_doc_offset = int(context.pre_num_conv_tokens or 0)
        rank = dist.get_rank(context.group)
        initial_state[0] = chain_recv_state(
            (heads, key_dim, value_dim),
            src=rank - 1,
            group=context.group,
            in_doc_offset=in_doc_offset,
            device=device,
            tag_prefix=tag_prefix,
        )
    return initial_state


def chain_post_scan_send(
    final_state: torch.Tensor,
    *,
    context,
    tag_prefix: str = "gdn",
) -> None:
    """Send-side: ship the LAST local sequence's fp32 final state onward when
    that document continues into the next rank."""
    if context.is_last_rank:
        return
    rank = dist.get_rank(context.group)
    chain_send_state(final_state[-1], dst=rank + 1, group=context.group, tag_prefix=tag_prefix)


def chain_recv_state(
    shape: tuple[int, ...],
    src: int,
    group: dist.ProcessGroup | None = None,
    *,
    in_doc_offset: int,
    device: torch.device | str = "cuda",
    tag_prefix: str = "gdn",
) -> torch.Tensor:
    """Receive one fp32 boundary state, VALIDATE its receipt, return it on
    `device`. Raises ChainReceiptError before returning anything on failure."""
    wire = _wire_device(group)
    payload = torch.empty(shape, dtype=torch.float32, device=wire)
    digest = torch.empty(_DIGEST_BYTES, dtype=torch.uint8, device=wire)
    dist.recv(payload, src=src, group=group)
    dist.recv(digest, src=src, group=group)
    receipt = ChainReceipt(
        sender_rank=src,
        receiver_rank=dist.get_rank(group),
        in_doc_offset=in_doc_offset,
        state_shape=tuple(shape),
        state_dtype=torch.float32,
        state_digest=bytes(digest.cpu().numpy().tobytes()),
    )
    validate_chain_receipt(receipt, state=payload)
    _record_engagement(
        f"{tag_prefix}:recv",
        f"rank {receipt.receiver_rank} <- {src}, state {tuple(shape)} fp32, "
        f"in-doc offset {in_doc_offset}",
    )
    return payload.to(device)
