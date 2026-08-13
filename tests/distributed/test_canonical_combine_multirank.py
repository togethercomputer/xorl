"""Multi-rank canonical-combine gate: two REAL collective transports, one program.

Four gloo ranks act as an EP4 group. Experts are partitioned contiguously
across ranks; every token has an owner rank (t % world). The SAME per-slot
contribution bytes travel to token owners through two genuinely different
collective transports:

- ``allgather_expert_major``: each expert rank packs its rows expert-major
  with per-rank padding; owners receive the full gathered set and filter.
- ``alltoall_owner_slot_major``: each expert rank packs per-destination
  buckets in owner-token slot-major order; owners receive rank-major padded
  segments via all_to_all_single.

Each owner validates the transport receipt, runs canonical_combine over its
token slice, and compares BYTES (forward, contribution grads mapped back to
canonical slot space, and routing-weight grads) across both transports and a
single-process oracle. Any mismatch exits nonzero and fails the launcher.

Follows the self-launching pattern of test_canonical_moe_contract.py; gloo
keeps it runnable on CPU-only boxes and off busy GPUs.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


if __name__ != "__main__":
    import pytest
    from distributed_utils import run_distributed_script

    pytestmark = [pytest.mark.distributed]

    @pytest.mark.cpu
    def test_two_collective_transports_are_byte_equal():
        result = run_distributed_script(__file__, num_gpus=4, timeout=180)
        result.assert_success("canonical combine over allgather vs all-to-all transports")


_T, _K, _E, _H = 41, 8, 32, 64
_WORLD = 4


def _build_shared_problem():
    """Every rank deterministically rebuilds the same route program and rows."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from xorl.distributed.canonical_combine import CanonicalRouteMetadata

    generator = torch.Generator(device="cpu").manual_seed(424242)
    scores = torch.randn((_T, _E), generator=generator, dtype=torch.float32)
    weights, ids = torch.topk(scores.softmax(dim=-1), k=_K, dim=-1)
    ids = ids.to(torch.int32)
    dropped = torch.rand((_T, _K), generator=generator) < 0.05
    dropped[:, 0] = False
    ids = ids.masked_fill(dropped, -1)
    weights = weights.masked_fill(dropped, 0.0)
    metadata = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=_E)

    active = metadata.active_mask
    slot_tokens = torch.arange(_T).unsqueeze(1).expand_as(active)[active]
    slot_experts = ids[active].to(torch.int64)
    slot_gids = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    table = (
        torch.randn((slot_tokens.numel(), _H), generator=generator, dtype=torch.float32)
    ).to(torch.bfloat16)
    return metadata, table, slot_tokens, slot_experts, slot_gids


def _expert_rank(slot_experts: torch.Tensor) -> torch.Tensor:
    return slot_experts // (_E // _WORLD)


def _token_owner(slot_tokens: torch.Tensor) -> torch.Tensor:
    return slot_tokens % _WORLD


def _pad_rows(rows: torch.Tensor, gids: torch.Tensor, count: int):
    pad = count - rows.shape[0]
    if pad:
        rows = torch.cat([rows, torch.full((pad, _H), float("nan"), dtype=rows.dtype)])
        gids = torch.cat([gids, torch.full((pad,), -1, dtype=gids.dtype)])
    return rows, gids


def _transport_allgather_expert_major(rank, table, slot_tokens, slot_experts, slot_gids):
    """All ranks receive everything; owners filter their tokens' rows."""
    mine = _expert_rank(slot_experts) == rank
    order = torch.argsort(slot_experts[mine], stable=True)  # expert-major, slot-major within
    local_rows = table[mine][order].contiguous()
    local_gids = slot_gids[mine][order].contiguous()

    counts = torch.zeros(_WORLD, dtype=torch.int64)
    counts[rank] = local_rows.shape[0]
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    max_count = int(counts.max().item())

    padded_rows, padded_gids = _pad_rows(local_rows, local_gids, max_count)
    gathered_rows = [torch.empty_like(padded_rows) for _ in range(_WORLD)]
    gathered_gids = [torch.empty_like(padded_gids) for _ in range(_WORLD)]
    dist.all_gather(gathered_rows, padded_rows)
    dist.all_gather(gathered_gids, padded_gids)

    all_rows = torch.cat(gathered_rows)
    all_gids = torch.cat(gathered_gids)
    owned_tokens = _token_owner(slot_tokens)
    owned_gids = set(slot_gids[owned_tokens == rank].tolist())
    keep = torch.tensor([int(g.item()) in owned_gids for g in all_gids], dtype=torch.bool)
    return all_rows[keep].contiguous(), all_gids[keep].contiguous()


def _transport_alltoall_owner_slot_major(rank, table, slot_tokens, slot_experts, slot_gids):
    """Expert ranks bucket rows per destination owner in slot-major order."""
    mine = _expert_rank(slot_experts) == rank
    owners = _token_owner(slot_tokens)

    send_rows, send_gids, send_counts = [], [], []
    for destination in range(_WORLD):
        bucket = mine & (owners == destination)
        order = torch.argsort(slot_gids[bucket], stable=True)  # owner-token slot-major
        send_rows.append(table[bucket][order])
        send_gids.append(slot_gids[bucket][order])
        send_counts.append(int(bucket.sum().item()))

    count_matrix = torch.zeros((_WORLD, _WORLD), dtype=torch.int64)
    count_matrix[rank] = torch.tensor(send_counts)
    dist.all_reduce(count_matrix, op=dist.ReduceOp.SUM)
    max_count = int(count_matrix.max().item())

    send_row_buf = torch.empty((_WORLD * max_count, _H), dtype=table.dtype)
    send_gid_buf = torch.empty((_WORLD * max_count,), dtype=torch.int64)
    for destination in range(_WORLD):
        rows, gids = _pad_rows(send_rows[destination], send_gids[destination], max_count)
        send_row_buf[destination * max_count : (destination + 1) * max_count] = rows
        send_gid_buf[destination * max_count : (destination + 1) * max_count] = gids

    recv_row_buf = torch.empty_like(send_row_buf)
    recv_gid_buf = torch.empty_like(send_gid_buf)
    dist.all_to_all_single(recv_row_buf, send_row_buf)
    dist.all_to_all_single(recv_gid_buf, send_gid_buf)

    keep = recv_gid_buf >= 0
    return recv_row_buf[keep].contiguous(), recv_gid_buf[keep].contiguous()


def _combine_owned_slice(rank, metadata, delivered_rows, delivered_gids, transport_id):
    """Build the owner's receipt from delivered annotations, validate, combine."""
    from xorl.distributed.canonical_combine import (
        CanonicalRouteMetadata,
        TransportReceipt,
        canonical_combine,
    )

    owned = torch.arange(_T) % _WORLD == rank
    local_metadata = CanonicalRouteMetadata(
        topk_ids=metadata.topk_ids[owned].contiguous(),
        topk_weights=metadata.topk_weights[owned].clone().requires_grad_(True),
        num_experts=_E,
    )
    owned_token_of_global = torch.full((_T,), -1, dtype=torch.int64)
    owned_token_of_global[owned] = torch.arange(int(owned.sum().item()))

    gid_to_row = torch.full((_T * _K,), -1, dtype=torch.int64)
    gid_to_row[delivered_gids] = torch.arange(delivered_gids.numel(), dtype=torch.int64)

    local_active = local_metadata.active_mask
    local_gids = (
        torch.nonzero(owned, as_tuple=False).reshape(-1).unsqueeze(1) * _K
        + torch.arange(_K).unsqueeze(0)
    )
    slot_to_row = torch.full_like(local_metadata.topk_ids, -1, dtype=torch.int64)
    slot_to_row[local_active] = gid_to_row[local_gids[local_active]]

    row_expert_ids = metadata.topk_ids.reshape(-1)[delivered_gids].to(torch.int64)
    row_source_tokens = owned_token_of_global[delivered_gids // _K]
    receipt = TransportReceipt(
        slot_to_row=slot_to_row,
        num_rows=delivered_rows.shape[0],
        row_expert_ids=row_expert_ids,
        row_source_tokens=row_source_tokens,
        transport_id=transport_id,
    )

    rows = delivered_rows.clone().requires_grad_(True)
    output = canonical_combine(rows, local_metadata, receipt, backend="reference")

    upstream_generator = torch.Generator(device="cpu").manual_seed(31337 + rank)
    upstream = torch.randn(output.shape, generator=upstream_generator, dtype=torch.float32)
    upstream_bf16 = upstream.to(output.dtype)
    output.backward(upstream_bf16)

    # Independent GRADIENT oracle: stock differentiable torch ops, no custom
    # Function. Labels: d(topk_weights) BYTE-exact; d(contributions)
    # BYTE-exact after signed-zero normalization (+0.0) (autograd's scatter
    # accumulation rewrites -0.0 grads to +0.0; the one-writer custom
    # backward preserves them); forward NUMERICAL sanity only — the
    # canonical forward is an FMA chain while the
    # oracle's mul-then-add forward exists to reproduce the trainer-owned
    # rounded-product BACKWARD; bitwise forward parity belongs to the CUDA
    # backend/identity gates.
    oracle_out, oracle_grad_rows, oracle_grad_weights = _dense_autograd_oracle(
        delivered_rows, local_metadata, receipt, upstream_bf16
    )
    out_f32 = output.detach().float()
    rel = (out_f32 - oracle_out.float()).norm(dim=1) / out_f32.norm(dim=1).clamp_min(1.0)
    assert float(rel.max()) < 1e-2, (
        f"rank {rank} {transport_id}: forward not numerically close to the oracle"
    )
    assert torch.equal(
        local_metadata.topk_weights.grad.detach().view(torch.int32), oracle_grad_weights.view(torch.int32)
    ), f"rank {rank} {transport_id}: d(topk_weights) FP32 bytes differ from the dense autograd oracle"
    assert torch.equal(_bits(rows.grad.detach() + 0.0), _bits(oracle_grad_rows + 0.0)), (
        f"rank {rank} {transport_id}: d(contributions) bytes differ from the oracle (signed-zero normalized)"
    )

    slot_rows = receipt.slot_to_row[local_active]
    return {
        "output": output.detach(),
        "grad_table_slot_major": rows.grad.detach().index_select(0, slot_rows),
        "grad_weights": local_metadata.topk_weights.grad.detach(),
    }


def _dense_autograd_oracle(rows_bf16, metadata, receipt, upstream):
    """Stock-torch gradient oracle for the canonical program (see caller)."""
    rows = rows_bf16.detach().clone().requires_grad_(True)
    weights = metadata.topk_weights.detach().clone().requires_grad_(True)
    active = metadata.active_mask
    safe = receipt.slot_to_row.to(torch.int64).clamp_min(0)
    acc = torch.zeros((metadata.num_tokens, rows.shape[1]), dtype=torch.float32, device=rows.device)
    for k in range(metadata.topk):
        values = rows.index_select(0, safe[:, k]).to(torch.float32) * weights[:, k].unsqueeze(1)
        acc = acc + torch.where(active[:, k].unsqueeze(1), values, torch.zeros_like(values))
    out = acc.to(torch.bfloat16)
    out.backward(upstream)
    return out.detach(), rows.grad.detach(), weights.grad.detach()


def _oracle(rank, metadata, table, slot_tokens, slot_gids):
    """Single-process oracle: the owner's slice combined from the shared table."""
    owned_slots = _token_owner(slot_tokens) == rank
    return _combine_owned_slice(
        rank,
        metadata,
        table[owned_slots].contiguous(),
        slot_gids[owned_slots].contiguous(),
        "oracle_local_slot_major",
    )


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16 if x.dtype is torch.bfloat16 else torch.int32)


def _run_multirank_case() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == _WORLD, "this gate is defined for exactly 4 ranks"
    torch.manual_seed(0)

    metadata, table, slot_tokens, slot_experts, slot_gids = _build_shared_problem()

    delivered_a = _transport_allgather_expert_major(rank, table, slot_tokens, slot_experts, slot_gids)
    delivered_b = _transport_alltoall_owner_slot_major(rank, table, slot_tokens, slot_experts, slot_gids)
    result_a = _combine_owned_slice(rank, metadata, *delivered_a, "mr_allgather_expert_major")
    result_b = _combine_owned_slice(rank, metadata, *delivered_b, "mr_alltoall_owner_slot_major")
    result_o = _oracle(rank, metadata, table, slot_tokens, slot_gids)

    # The two transports really did deliver different physical layouts.
    assert delivered_a[1].shape == delivered_b[1].shape
    assert not torch.equal(delivered_a[1], delivered_b[1]), (
        "transports delivered identical receive orders; the gate would prove nothing"
    )

    for key in ("output", "grad_table_slot_major", "grad_weights"):
        assert torch.equal(_bits(result_a[key]), _bits(result_b[key])), (
            f"rank {rank}: {key} bytes differ between collective transports"
        )
        assert torch.equal(_bits(result_a[key]), _bits(result_o[key])), (
            f"rank {rank}: {key} bytes differ from the single-process oracle"
        )

    dist.barrier()
    if rank == 0:
        print("multirank canonical combine: byte equality across transports and oracle")
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_multirank_case()
