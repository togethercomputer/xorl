"""Detect shared-prefix groups and repack a packed micro-batch.

A packed RL micro-batch is a single concatenated sequence ``[1, T]`` whose
per-sequence boundaries are given by ``cu_seqlens``. Prompt tokens carry an
ignored label (``target_tokens == -100``); the first non-ignored label of a
sequence sits at the *last* prompt position ``P-1`` (it predicts the first
response token).

``shared_prefix_repack_batch`` groups sequences that share an identical prompt
prefix and lays out, per group, a shared prefix block ``[p_0 .. p_{P-2}]``
(KV only — its query outputs are never trained) followed by one decoded block
per member ``[p_{P-1}, r_0, ..., r_{R-1}]``. The last prompt token ``p_{P-1}`` is
**duplicated** into each member's decoded block; this makes every *trained*
position live in a per-member block, so the repacked<->original mapping is 1:1
(no position is shared across members). That 1:1 property is what lets the loss
run directly on the repacked layout and compose with sequence-parallel sharding.

Attention over a decoded query then decomposes as:
    full causal = (cross-attn to the shared prefix [p_0..p_{P-2}], non-causal)
                  merged with (decoded self-attn within its own block, causal)
which the :mod:`shared_prefix_attention` backend computes. Groups with ``P == 1``
have an empty shared block and use decoded self-attn only.

All grouping is auto-detected from token ids + labels; nothing in the training
request or wire protocol changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from xorl.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


IGNORE_INDEX = -100

# Sequence-aligned fields repacked alongside input_ids (1-D, length T).
_LOSS_FIELDS = ("target_tokens", "labels", "advantages", "logprobs", "rollout_logprobs")
# Repacked fill values for the shared-prefix positions (never trained).
_FILL = {"target_tokens": IGNORE_INDEX, "labels": IGNORE_INDEX}


@dataclass
class SharedPrefixContext:
    """Indices / cu_seqlens driving shared-prefix attention + output remap.

    Attention indices/cu_seqlens are in *repacked* coordinates and remain valid
    after Ulysses all-to-all re-gathers the full sequence inside attention.

        shared_idx        prefix tokens [p_0..p_{P-2}] of groups with P>=2
        dec_idx           all decoded tokens (queries) in member order
        cu_shared         prefix self-attn / cross-attn key segments (per such group)
        cross_local_idx   positions within ``dec_idx`` whose group has a prefix
        cu_cross_q        cross-attn query segments (per such group)
        cu_dec            decoded self-attn segments (per member; includes the
                          duplicated boundary token)

    Output remap (repacked per-token outputs -> original layout):
        dec_idx -> dec_orig_idx (1:1)
    """

    shared_idx: torch.Tensor
    dec_idx: torch.Tensor
    cu_shared: torch.Tensor
    max_shared: int
    cross_local_idx: torch.Tensor
    cu_cross_q: torch.Tensor
    max_cross_q: int
    cu_dec: torch.Tensor
    max_dec: int
    dec_orig_idx: torch.Tensor
    repacked_len: int
    orig_len: int
    num_groups: int
    has_shared_prefix: bool
    # Original packed position ids ([1, T_orig]); used to emit per-token outputs
    # in the layout the client sent.
    orig_position_ids: torch.Tensor

    def to(self, device) -> "SharedPrefixContext":
        """Move all index/cu tensors to ``device`` (ints/bools untouched)."""
        return SharedPrefixContext(
            shared_idx=self.shared_idx.to(device),
            dec_idx=self.dec_idx.to(device),
            cu_shared=self.cu_shared.to(device),
            max_shared=self.max_shared,
            cross_local_idx=self.cross_local_idx.to(device),
            cu_cross_q=self.cu_cross_q.to(device),
            max_cross_q=self.max_cross_q,
            cu_dec=self.cu_dec.to(device),
            max_dec=self.max_dec,
            dec_orig_idx=self.dec_orig_idx.to(device),
            repacked_len=self.repacked_len,
            orig_len=self.orig_len,
            num_groups=self.num_groups,
            has_shared_prefix=self.has_shared_prefix,
            orig_position_ids=self.orig_position_ids.to(device),
        )


def _to_flat(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        assert x.size(0) == 1, f"expected batch dim 1, got {tuple(x.shape)}"
        x = x.squeeze(0)
    return x.reshape(-1)


def detect_prompt_groups(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[List[List[int]], List[int]]:
    """Group packed sequences that share an identical prompt prefix.

    Returns ``(groups, prompt_lens)``: groups in first-appearance order, members
    in original order; ``prompt_lens[i]`` is ``P_i`` (``= seq_len`` for a
    prefix-only / pad sequence with no valid labels).
    """
    ids = _to_flat(input_ids).tolist()
    lab = _to_flat(labels).tolist()
    bounds = cu_seqlens.tolist()
    num_seqs = len(bounds) - 1

    prompt_lens: List[int] = []
    keys: List[Tuple] = []
    for i in range(num_seqs):
        start, end = bounds[i], bounds[i + 1]
        seq_len = end - start
        first_valid = next((j for j in range(seq_len) if lab[start + j] != ignore_index), None)
        prompt_len = seq_len if first_valid is None else first_valid + 1
        prompt_lens.append(prompt_len)
        keys.append((prompt_len, tuple(ids[start : start + prompt_len])))

    order: List[Tuple] = []
    groups_by_key: Dict[Tuple, List[int]] = {}
    for i, key in enumerate(keys):
        if key not in groups_by_key:
            groups_by_key[key] = []
            order.append(key)
        groups_by_key[key].append(i)
    return [groups_by_key[k] for k in order], prompt_lens


def _build_plan(input_ids: torch.Tensor, labels: torch.Tensor, cu_seqlens: torch.Tensor, ignore_index: int):
    """Build the repacked token layout + index plan (python lists)."""
    ids = _to_flat(input_ids).tolist()
    bounds = cu_seqlens.tolist()
    groups, prompt_lens = detect_prompt_groups(input_ids, labels, cu_seqlens, ignore_index)
    has_shared_prefix = any(len(g) > 1 for g in groups)

    rep_ids: List[int] = []
    rep_pos: List[int] = []
    shared_idx: List[int] = []
    dec_idx: List[int] = []
    dec_orig_idx: List[int] = []
    cross_local_idx: List[int] = []
    cu_shared = [0]
    cu_cross_q = [0]
    cu_dec = [0]
    max_shared = max_cross_q = max_dec = 0

    rep = 0
    for group in groups:
        first = group[0]
        g_start = bounds[first]
        P = prompt_lens[first]
        shared_len = P - 1  # prefix tokens excluding the duplicated boundary

        if shared_len > 0:
            rep_ids.extend(ids[g_start : g_start + shared_len])
            rep_pos.extend(range(shared_len))
            shared_idx.extend(range(rep, rep + shared_len))
            cu_shared.append(cu_shared[-1] + shared_len)
            max_shared = max(max_shared, shared_len)
            rep += shared_len

        group_cross_dec = 0
        for i in group:
            s, e = bounds[i], bounds[i + 1]
            resp_len = (e - s) - P
            block_len = resp_len + 1  # duplicated boundary p_{P-1} + responses
            # tokens: [p_{P-1}, r_0, ..., r_{R-1}] ; orig positions [P-1, P, ..., P+R-1]
            rep_ids.extend(ids[s + P - 1 : e])
            rep_pos.extend(range(P - 1, P + resp_len))
            dec_idx.extend(range(rep, rep + block_len))
            dec_orig_idx.extend(range(s + P - 1, e))
            cu_dec.append(cu_dec[-1] + block_len)
            max_dec = max(max_dec, block_len)
            if shared_len > 0:
                cross_local_idx.extend(range(len(dec_idx) - block_len, len(dec_idx)))
                group_cross_dec += block_len
            rep += block_len
        if shared_len > 0:
            cu_cross_q.append(cu_cross_q[-1] + group_cross_dec)
            max_cross_q = max(max_cross_q, group_cross_dec)

    plan = dict(
        rep_ids=rep_ids,
        rep_pos=rep_pos,
        shared_idx=shared_idx,
        dec_idx=dec_idx,
        dec_orig_idx=dec_orig_idx,
        cross_local_idx=cross_local_idx,
        cu_shared=cu_shared,
        cu_cross_q=cu_cross_q,
        cu_dec=cu_dec,
        max_shared=max_shared,
        max_cross_q=max_cross_q,
        max_dec=max_dec,
        repacked_len=rep,
        orig_len=int(bounds[-1]),
        num_groups=len(groups),
        has_shared_prefix=has_shared_prefix,
    )
    return plan


def shared_prefix_repack_batch(
    batch: Dict[str, Any],
    ignore_index: int = IGNORE_INDEX,
) -> Optional[Dict[str, Any]]:
    """Repack a packed batch into shared-prefix layout (or ``None`` if no sharing).

    Requires ``input_ids`` and a label field (``target_tokens`` or ``labels``).
    ``cu_seqlens`` is taken from ``cu_seq_lens_q`` if present, else derived from
    ``position_ids``. Returns a new batch dict with repacked ``input_ids`` /
    ``position_ids`` / sequence-aligned loss fields and a ``shared_prefix_context``
    entry; ``cu_seq_lens_*`` / ``max_length_*`` are dropped (the shared-prefix
    backend drives attention from the context). The original batch is unchanged.
    """
    input_ids = batch.get("input_ids")
    labels = batch.get("target_tokens", batch.get("labels"))
    if input_ids is None or labels is None:
        return None

    cu = batch.get("cu_seq_lens_q")
    if cu is None:
        position_ids = batch.get("position_ids")
        if position_ids is None:
            return None
        (cu, _), (_, _) = prepare_fa_kwargs_from_position_ids(position_ids)

    plan = _build_plan(input_ids, labels, cu, ignore_index)
    if not plan["has_shared_prefix"]:
        return None

    device = input_ids.device
    long = lambda lst: torch.tensor(lst, device=device, dtype=torch.long)  # noqa: E731
    i32 = lambda lst: torch.tensor(lst, device=device, dtype=torch.int32)  # noqa: E731

    # Original position ids for emitting per-token outputs in the client's layout.
    orig_pos = batch.get("position_ids")
    if orig_pos is not None:
        orig_position_ids = orig_pos if orig_pos.dim() == 2 else orig_pos.unsqueeze(0)
    else:
        bounds = cu.tolist()
        orig_position_ids = torch.cat(
            [torch.arange(bounds[i + 1] - bounds[i], device=device, dtype=torch.long) for i in range(len(bounds) - 1)]
        ).unsqueeze(0)

    dec_idx = long(plan["dec_idx"])
    dec_orig_idx = long(plan["dec_orig_idx"])
    ctx = SharedPrefixContext(
        shared_idx=long(plan["shared_idx"]),
        dec_idx=dec_idx,
        cu_shared=i32(plan["cu_shared"]),
        max_shared=plan["max_shared"],
        cross_local_idx=long(plan["cross_local_idx"]),
        cu_cross_q=i32(plan["cu_cross_q"]),
        max_cross_q=plan["max_cross_q"],
        cu_dec=i32(plan["cu_dec"]),
        max_dec=plan["max_dec"],
        dec_orig_idx=dec_orig_idx,
        repacked_len=plan["repacked_len"],
        orig_len=plan["orig_len"],
        num_groups=plan["num_groups"],
        has_shared_prefix=True,
        orig_position_ids=orig_position_ids,
    )

    out: Dict[str, Any] = dict(batch)
    out["input_ids"] = long(plan["rep_ids"]).unsqueeze(0)
    out["position_ids"] = long(plan["rep_pos"]).unsqueeze(0)
    out["attention_mask"] = torch.ones((1, ctx.repacked_len), dtype=torch.long, device=device)

    # Repack sequence-aligned loss fields by the same 1:1 decoded mapping;
    # shared-prefix positions are filled with ignore/0 (never trained).
    for field in _LOSS_FIELDS:
        src = batch.get(field)
        if src is None:
            continue
        src_flat = _to_flat(src)
        fill = _FILL.get(field, 0)
        repacked = src_flat.new_full((ctx.repacked_len,), fill)
        repacked[dec_idx] = src_flat[dec_orig_idx]
        out[field] = repacked.unsqueeze(0)

    out["shared_prefix_context"] = ctx
    for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        out.pop(key, None)
    return out


def shared_prefix_remap_to_original(
    per_token: torch.Tensor,
    ctx: SharedPrefixContext,
    fill: float = 0.0,
) -> torch.Tensor:
    """Map a repacked per-token tensor back to the original packed layout.

    Args:
        per_token: ``[..., T_rep]`` per-token values in repacked order.
        ctx: context from :func:`shared_prefix_repack_batch`.
        fill: value for original positions with no repacked counterpart
            (prompt-interior positions, which are never trained).

    Returns:
        ``[..., T_orig]`` in the original packed layout.
    """
    out_shape = (*per_token.shape[:-1], ctx.orig_len)
    out = per_token.new_full(out_shape, fill)
    out.index_copy_(
        -1, ctx.dec_orig_idx.to(per_token.device), per_token.index_select(-1, ctx.dec_idx.to(per_token.device))
    )
    return out
