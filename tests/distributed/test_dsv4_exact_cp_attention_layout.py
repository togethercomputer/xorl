"""CPU/gloo contracts for variable-row DSV4 exact CP transport.

The production path compacts only live collator rows, pads compute to the
largest stage-local owner count, gathers those padded tensors with a VJP, and
restores packed logical order. These cases cover the real 100-token/CP8 seam,
two independent DP-local CP groups, and ring2 x Ulysses2 packed documents.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.data.collators.sequence_shard_collator import zigzag_reorder_packed_sequence  # noqa: E402
from xorl.ops.dsv4.cp_utils import (  # noqa: E402
    build_dsv4_exact_cp_layout,
    gather_dsv4_exact_cp_rows,
)


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _packed_metadata(document_lengths: list[int], *, cp_size: int, ringattn_size: int):
    live_rows = sum(document_lengths)
    pad_multiple = 2 * cp_size if ringattn_size > 1 else cp_size
    storage_rows = ((live_rows + pad_multiple - 1) // pad_multiple) * pad_multiple
    pad_rows = storage_rows - live_rows

    logical = torch.full((1, storage_rows), -1, dtype=torch.int64)
    logical[:, :live_rows] = torch.arange(live_rows)
    request_ids = torch.full_like(logical, -1)
    request_positions = torch.zeros_like(logical)
    live_mask = torch.zeros_like(logical, dtype=torch.bool)
    positions = []
    cursor = 0
    for request_id, length in enumerate(document_lengths):
        request_ids[:, cursor : cursor + length] = request_id
        request_positions[:, cursor : cursor + length] = torch.arange(length)
        live_mask[:, cursor : cursor + length] = True
        positions.append(torch.arange(length))
        cursor += length
    if pad_rows:
        # This is the collator's tail-pad document: it participates in ring
        # storage balancing but is never a live model/request row.
        positions.append(torch.arange(pad_rows))
    position_ids = torch.cat(positions).view(1, -1)

    if ringattn_size > 1:
        logical = zigzag_reorder_packed_sequence(logical, position_ids, ringattn_size)
        request_ids = zigzag_reorder_packed_sequence(request_ids, position_ids, ringattn_size)
        request_positions = zigzag_reorder_packed_sequence(request_positions, position_ids, ringattn_size)
        live_mask = zigzag_reorder_packed_sequence(live_mask, position_ids, ringattn_size)
    return logical, request_ids, request_positions, live_mask


def _run_case() -> None:
    dist.init_process_group("gloo")
    try:
        topology = os.environ["XORL_DSV4_CP_TOPOLOGY"]
        dp_size, cp_size, ringattn_size = {
            "dp1_cp8_100": (1, 8, 1),
            "dp2_cp4": (2, 4, 1),
            "dp2_ring2_ulysses2": (2, 4, 2),
        }[topology]
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        assert world_size == dp_size * cp_size == 8

        cp_groups = []
        for group_dp_rank in range(dp_size):
            ranks = list(range(group_dp_rank * cp_size, (group_dp_rank + 1) * cp_size))
            cp_groups.append(dist.new_group(ranks=ranks, backend="gloo"))
        dp_rank, cp_rank = divmod(rank, cp_size)
        cp_group = cp_groups[dp_rank]

        if topology == "dp1_cp8_100":
            document_lengths = [100]
        elif topology == "dp2_cp4":
            document_lengths = [5, 10] if dp_rank == 0 else [7, 10]
        else:
            # Every real and tail-pad document is divisible by 2*ring=4,
            # matching the production collator. The two DP planes deliberately
            # have different packed lengths and local live-row distributions.
            document_lengths = [8, 12] if dp_rank == 0 else [4, 12]

        logical, request_ids, request_positions, live_mask = _packed_metadata(
            document_lengths,
            cp_size=cp_size,
            ringattn_size=ringattn_size,
        )
        local_storage_rows = logical.shape[1] // cp_size
        start = cp_rank * local_storage_rows
        stop = start + local_storage_rows
        local_logical = logical[:, start:stop]
        local_request_ids = request_ids[:, start:stop]
        local_request_positions = request_positions[:, start:stop]
        local_live_mask = live_mask[:, start:stop]

        local_live_count = int(local_live_mask.sum())
        compute_rows_tensor = torch.tensor([local_live_count], dtype=torch.int64)
        dist.all_reduce(compute_rows_tensor, op=dist.ReduceOp.MAX, group=cp_group)
        compute_rows = max(1, int(compute_rows_tensor.item()))
        layout = build_dsv4_exact_cp_layout(
            local_logical,
            local_request_ids,
            local_request_positions,
            local_live_mask,
            compute_rows=compute_rows,
            cp_group=cp_group,
        )

        sequence_length = sum(document_lengths)
        assert torch.equal(layout.global_logical_rows, torch.arange(sequence_length))
        assert layout.local_live_count == local_live_count
        if topology == "dp1_cp8_100":
            assert compute_rows == 13
            assert sequence_length == 100

        base = 1000.0 * dp_rank + 0.125
        full_kv = torch.arange(sequence_length, dtype=torch.float64).view(1, -1, 1) + base
        full_source = full_kv * 0.5 + 3.0
        compact_logical = layout.local_logical_rows[:local_live_count]

        for compressed in (False, True):
            compact_kv = full_kv.index_select(1, compact_logical)
            local_kv = F.pad(compact_kv, (0, 0, 0, compute_rows - local_live_count)).requires_grad_(True)
            gathered_kv = gather_dsv4_exact_cp_rows(
                local_kv,
                dim=1,
                layout=layout,
                cp_group=cp_group,
            )
            torch.testing.assert_close(gathered_kv, full_kv, rtol=0, atol=0)

            local_source = None
            gathered_source = None
            if compressed:
                compact_source = full_source.index_select(1, compact_logical)
                local_source = F.pad(
                    compact_source,
                    (0, 0, 0, compute_rows - local_live_count),
                ).requires_grad_(True)
                gathered_source = gather_dsv4_exact_cp_rows(
                    local_source,
                    dim=1,
                    layout=layout,
                    cp_group=cp_group,
                )
                torch.testing.assert_close(gathered_source, full_source, rtol=0, atol=0)

            gathered_output = 0.25 * gathered_kv + 0.01 * gathered_kv.cumsum(dim=1).square()
            if gathered_source is not None:
                gathered_output = gathered_output + 0.02 * gathered_source.cumsum(dim=1).sin()

            # Restore local queries to collator storage order before the loss.
            compact_output = gathered_output.index_select(1, compact_logical)
            storage_output = gathered_output.new_zeros((1, local_storage_rows, 1)).index_copy(
                1,
                layout.local_storage_indices,
                compact_output,
            )
            storage_weights = storage_output.new_zeros((1, local_storage_rows, 1))
            storage_weights[:, layout.local_storage_indices, 0] = (compact_logical + 1).to(storage_weights.dtype)
            (storage_output * storage_weights).sum().backward()
            assert torch.equal(
                storage_output[:, ~local_live_mask.reshape(-1), :],
                torch.zeros_like(storage_output[:, ~local_live_mask.reshape(-1), :]),
            )

            reference_kv = full_kv.clone().requires_grad_(True)
            reference_source = full_source.clone().requires_grad_(compressed)
            reference_output = 0.25 * reference_kv + 0.01 * reference_kv.cumsum(dim=1).square()
            if compressed:
                reference_output = reference_output + 0.02 * reference_source.cumsum(dim=1).sin()
            reference_weights = torch.arange(1, sequence_length + 1, dtype=torch.float64).view(1, -1, 1)
            (reference_output * reference_weights).sum().backward()

            expected_kv_grad = F.pad(
                reference_kv.grad.index_select(1, compact_logical),
                (0, 0, 0, compute_rows - local_live_count),
            )
            # Gloo's reduce-scatter tree reassociates FP64 additions relative
            # to the serial oracle; the accepted envelope is below 2 ulp at
            # the largest 100-row gradient in this test.
            torch.testing.assert_close(local_kv.grad, expected_kv_grad, rtol=2e-15, atol=5e-10)
            if compressed:
                expected_source_grad = F.pad(
                    reference_source.grad.index_select(1, compact_logical),
                    (0, 0, 0, compute_rows - local_live_count),
                )
                torch.testing.assert_close(local_source.grad, expected_source_grad, rtol=2e-15, atol=5e-10)
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    @pytest.mark.parametrize("topology", ["dp1_cp8_100", "dp2_cp4", "dp2_ring2_ulysses2"])
    def test_dsv4_exact_cp_variable_transport_scatter_and_backward(topology: str) -> None:
        result = run_distributed_script(
            os.path.abspath(__file__),
            num_gpus=8,
            timeout=120,
            extra_env={
                "XORL_DSV4_CP_TOPOLOGY": topology,
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        result.assert_success(f"DSV4 exact {topology} variable-row CP transport")


if __name__ == "__main__":
    _run_case()
