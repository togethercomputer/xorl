"""CPU/gloo contract for DSV4 exact context-parallel attention transport.

The production exact path keeps queries on their trainer CP shard, restores
gathered KV/compressor sources to logical sequence order, and supplies logical
query positions to the literal attention kernel.  This test checks both
contiguous/Ulysses storage and ring/Ulysses zigzag storage, including the
differentiable transport on two separate DP-local CP groups.  Pipeline rank is
deliberately absent: CP groups are stage-local, so the mechanism is identical
on every PP stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.ops.dsv4.cp_utils import (  # noqa: E402
    all_gather_cp,
    get_q_positions_for_cp,
    get_zigzag_cp_layout,
)


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_case() -> None:
    dist.init_process_group("gloo")
    try:
        topology = os.environ["XORL_DSV4_CP_TOPOLOGY"]
        dp_size, cp_size, ringattn_size = {
            "dp1_cp8": (1, 8, 1),
            "dp2_cp4": (2, 4, 1),
            "dp2_ring2_ulysses2": (2, 4, 2),
        }[topology]
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        assert world_size == dp_size * cp_size == 8

        cp_groups = []
        for dp_rank in range(dp_size):
            ranks = list(range(dp_rank * cp_size, (dp_rank + 1) * cp_size))
            cp_groups.append(dist.new_group(ranks=ranks, backend="gloo"))
        dp_rank, cp_rank = divmod(rank, cp_size)
        cp_group = cp_groups[dp_rank]

        local_length = 4 if ringattn_size > 1 else 3
        sequence_length = local_length * cp_size
        full_kv = (
            torch.arange(sequence_length, dtype=torch.float64).view(1, sequence_length, 1) + 1000.0 * dp_rank + 0.125
        )
        full_compressor_source = full_kv * 0.5 + 3.0
        if ringattn_size > 1:
            # Two packed documents deliberately reuse RoPE positions. Unique
            # logical rows, not resetting position IDs, restore gather order.
            document_length = sequence_length // 2
            logical_documents = [
                torch.arange(start, start + document_length).chunk(2 * ringattn_size) for start in (0, document_length)
            ]
            rope_documents = [torch.arange(document_length).chunk(2 * ringattn_size) for _ in range(2)]
            storage_logical_rows = torch.cat(
                [
                    part
                    for ring_rank in range(ringattn_size)
                    for chunks in logical_documents
                    for part in (chunks[ring_rank], chunks[2 * ringattn_size - 1 - ring_rank])
                ]
            )
            storage_rope_positions = torch.cat(
                [
                    part
                    for ring_rank in range(ringattn_size)
                    for chunks in rope_documents
                    for part in (chunks[ring_rank], chunks[2 * ringattn_size - 1 - ring_rank])
                ]
            )
            assert torch.unique(storage_rope_positions).numel() < sequence_length
        else:
            storage_logical_rows = torch.arange(sequence_length)
        storage_start = cp_rank * local_length
        storage_stop = storage_start + local_length
        local_logical_rows = storage_logical_rows[storage_start:storage_stop]
        if ringattn_size > 1:
            query_positions, restore_order = get_zigzag_cp_layout(
                local_logical_rows,
                cp_group=cp_group,
            )
        else:
            query_positions = get_q_positions_for_cp(
                local_length,
                cp_size=cp_size,
                cp_group=cp_group,
                device=full_kv.device,
            )
            restore_order = torch.arange(sequence_length)
        assert torch.equal(query_positions, local_logical_rows)

        for compressed in (False, True):
            local_kv = full_kv.index_select(1, local_logical_rows).clone().requires_grad_(True)
            gathered_kv = all_gather_cp(local_kv, dim=1, cp_group=cp_group).index_select(1, restore_order)
            assert torch.equal(gathered_kv, full_kv)

            local_compressor = None
            gathered_compressor = None
            if compressed:
                local_compressor = (
                    full_compressor_source.index_select(1, local_logical_rows).clone().requires_grad_(True)
                )
                gathered_compressor = all_gather_cp(local_compressor, dim=1, cp_group=cp_group).index_select(
                    1,
                    restore_order,
                )
                assert torch.equal(gathered_compressor, full_compressor_source)

            # A causal, nonlinear stand-in makes every later query depend on
            # remote earlier KV rows and, for C4/C128, remote compressor rows.
            # Each rank owns only its local query loss; the gather VJP must
            # return all remote query uses to each source-row owner.
            gathered_output = 0.25 * gathered_kv + 0.01 * gathered_kv.cumsum(dim=1).square()
            if gathered_compressor is not None:
                gathered_output = gathered_output + 0.02 * gathered_compressor.cumsum(dim=1).sin()
            local_weights = (query_positions + 1).to(torch.float64).view(1, local_length, 1)
            local_output = gathered_output.index_select(1, query_positions)
            (local_output * local_weights).sum().backward()

            reference_kv = full_kv.clone().requires_grad_(True)
            reference_compressor = full_compressor_source.clone().requires_grad_(compressed)
            reference_output = 0.25 * reference_kv + 0.01 * reference_kv.cumsum(dim=1).square()
            if compressed:
                reference_output = reference_output + 0.02 * reference_compressor.cumsum(dim=1).sin()
            reference_weights = torch.arange(1, sequence_length + 1, dtype=torch.float64).view(1, sequence_length, 1)
            (reference_output * reference_weights).sum().backward()

            torch.testing.assert_close(
                local_output.detach(),
                reference_output.index_select(1, query_positions).detach(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                local_kv.grad,
                reference_kv.grad.index_select(1, local_logical_rows),
                # The VJP sums rank contributions in a collective tree; its
                # FP64 association can differ from the serial oracle by one ulp.
                rtol=1e-15,
                atol=1e-12,
            )
            if compressed:
                torch.testing.assert_close(
                    local_compressor.grad,
                    reference_compressor.grad.index_select(1, local_logical_rows),
                    rtol=1e-15,
                    atol=1e-12,
                )
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    @pytest.mark.parametrize("topology", ["dp1_cp8", "dp2_cp4", "dp2_ring2_ulysses2"])
    def test_dsv4_exact_cp_transport_and_backward(topology: str) -> None:
        result = run_distributed_script(
            os.path.abspath(__file__),
            num_gpus=8,
            timeout=120,
            extra_env={
                "XORL_DSV4_CP_TOPOLOGY": topology,
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        result.assert_success(f"DSV4 exact {topology} CP transport")


if __name__ == "__main__":
    _run_case()
