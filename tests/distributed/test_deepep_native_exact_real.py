"""Real-GPU component gate for the original-handle native exact transport."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


_WORKER_ENV = "XORL_TEST_DEEPEP_NATIVE_EXACT_WORKER"
_WORLD_ENV = "XORL_TEST_DEEPEP_NATIVE_EXACT_WORLD"


def _hierarchical_fold(leaves: torch.Tensor) -> torch.Tensor:
    """Independent Tree8/BF16-node/ascending-FP64-node reference."""
    node_leaves = []
    for begin in range(0, leaves.shape[0], 8):
        node = leaves[begin : begin + 8]
        if node.shape[0] < 8:
            node = torch.cat(
                (
                    node,
                    torch.zeros(
                        (8 - node.shape[0], *node.shape[1:]),
                        dtype=node.dtype,
                        device=node.device,
                    ),
                ),
                dim=0,
            )
        p01 = node[0].double() + node[1].double()
        p23 = node[2].double() + node[3].double()
        p45 = node[4].double() + node[5].double()
        p67 = node[6].double() + node[7].double()
        node_leaves.append(((p01 + p23) + (p45 + p67)).to(torch.bfloat16))
    value = node_leaves[0].double()
    for node_leaf in node_leaves[1:]:
        value = value + node_leaf.double()
    return value.to(torch.bfloat16)


def _worker_main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    assert world in (2, 4, 8, 16)

    from xorl.distributed.moe import deepep as deepep_module
    from xorl.distributed.moe.deepep import (
        get_default_buffer,
        token_pre_dispatch_native,
    )
    from xorl.distributed.moe.deepep_native_exact import (
        NativeDeepEPGeometry,
        native_receive_combine_and_fold,
    )

    rows, hidden = 3, 2048
    generator = torch.Generator(device="cpu").manual_seed(271828 + rank)
    hidden_states = torch.randn((rows, hidden), generator=generator).to(torch.bfloat16).to(device).requires_grad_(True)
    # Every source token has exactly one route to every physical expert rank.
    # DeepEP therefore performs its real top-k dispatch, while the expected
    # return leaf from rank r is unambiguously x + (r+1).
    selected = torch.arange(world, dtype=torch.int64, device=device).expand(rows, world).contiguous()
    weights = torch.ones((rows, world), dtype=torch.float32, device=device)

    geometry = NativeDeepEPGeometry(ep_size=world, ep_rank=rank, hidden_size=hidden)
    buffer = get_default_buffer(ep_group=dist.group.WORLD, buffer_size_gb=2.0, num_sms=20)
    buffer.init_buffer(hidden_bytes=geometry.wire_hidden_bytes)
    recv_hidden, recv_ids, recv_weights, dispatch_ctx = token_pre_dispatch_native(
        buffer=buffer,
        hidden_states=hidden_states,
        routing_weights=weights,
        selected_experts=selected,
        num_experts=world,
    )
    assert recv_hidden.dtype is torch.bfloat16
    assert recv_ids.dtype in (torch.int32, torch.int64)
    assert recv_weights.dtype is torch.float32
    local_leaf = (recv_hidden + float(rank + 1)).to(torch.bfloat16).contiguous()
    output = native_receive_combine_and_fold(
        local_leaf,
        buffer=buffer,
        dispatch_ctx=dispatch_ctx,
        ep_group=dist.group.WORLD,
        num_local_experts=1,
    )

    expected_leaves = torch.stack(
        [(hidden_states.detach() + float(source + 1)).to(torch.bfloat16) for source in range(world)]
    )
    expected = _hierarchical_fold(expected_leaves)
    assert output.dtype is torch.bfloat16
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16)), (
        f"rank {rank}: original-handle BF16 segmented combine differs from the explicit FP64 fold"
    )

    # The complete logical-rank loop must be one autograd node.  Independent
    # sibling nodes have no cross-rank execution-order guarantee and can enter
    # DeepEP's reverse-dispatch barriers in different logical-rank orders.
    pending = [output.grad_fn]
    seen = set()
    ordered_combine_nodes = 0
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        # Retain each Python wrapper.  Saving only id(node) permits an
        # already-released wrapper's address to be reused while traversing.
        seen.add(node)
        expected_node = "_DeepEPDeterministicCombineBF16Backward"
        if type(node).__name__ == expected_node:
            ordered_combine_nodes += 1
        pending.extend(next_node for next_node, _index in node.next_functions)
    assert ordered_combine_nodes == 1, (
        f"rank {rank}: expected one deterministic DeepEP autograd boundary, found {ordered_combine_nodes}"
    )

    output.float().sum().backward()
    torch.cuda.synchronize()
    assert hidden_states.grad is not None
    assert torch.equal(
        hidden_states.grad,
        torch.full_like(hidden_states, float(world)),
    ), f"rank {rank}: original-handle native exact backward did not reverse dispatch/combine"

    trace_dir = os.environ.get("XORL_DEEPEP_BOUNDARY_TRACE_DIR", "").strip()
    if trace_dir:
        trace_lines = Path(trace_dir, f"rank{rank:05d}.log").read_text().splitlines()
        reverse_dispatch_enters = [
            line for line in trace_lines if "boundary=output_reverse_dispatch state=enter" in line
        ]
        expected_reverse_dispatches = 1
        assert len(reverse_dispatch_enters) == expected_reverse_dispatches, (
            f"rank {rank}: expected {expected_reverse_dispatches} reverse dispatches, "
            f"observed {len(reverse_dispatch_enters)}"
        )

    widths = [None] * world
    dist.all_gather_object(widths, geometry.wire_width)
    assert widths == [hidden] * world
    if rank == 0:
        print(
            "deepep_native_exact_real_gate_ok "
            f"world={world} dispatch_width={hidden} wire_width={geometry.wire_width} "
            "combine_mode=deterministic combine_calls=1 wire_dtype=bf16 "
            "fold=hierarchical_tree8_bf16_node_fp64_node "
            "reverse_dispatches=1 backward=ok",
            flush=True,
        )
    if deepep_module._default_buffer is not None:
        deepep_module._default_buffer.destroy_buffer()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__" and os.environ.get(_WORKER_ENV) == "1":
    sys.exit(_worker_main())


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def test_original_dispatch_handle_accepts_native_exact_combine_program():
    pytest.importorskip("deep_ep")
    import deep_ep

    if not hasattr(deep_ep, "ReductionMode"):
        pytest.skip("installed DeepEP lacks the deterministic reduction program")
    from distributed_utils import gpu_count, run_distributed_script

    world = int(os.environ.get(_WORLD_ENV, "8"))
    if gpu_count() < world:
        pytest.skip(f"requires {world} GPUs, found {gpu_count()}")
    result = run_distributed_script(
        __file__,
        num_gpus=world,
        timeout=300,
        extra_env={_WORKER_ENV: "1"},
    )
    if not result.success:
        print("--- distributed worker stdout ---", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("--- distributed worker stderr ---", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    result.assert_success(f"native exact original-handle gate (world {world})")
    assert "deepep_native_exact_real_gate_ok" in result.stdout
