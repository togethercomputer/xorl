"""Distributed correctness tests for Muon ``distributed_mode='full_gradient'``.

Verifies that the full-gradient path produces the same parameter update as
running Muon on a single rank with the unsharded gradient (i.e., recovers
exact Muon math under FSDP2/DTensor sharding), and that the existing
``shard_local`` mode does NOT match — sanity check that we are exercising
the new code path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.optim.muon import Muon
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than


pytestmark = [pytest.mark.distributed]


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _setup_dist():
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=_world_size(),
        dp_replicate_size=1,
        dp_shard_size=_world_size(),
        tp_size=1,
        ep_size=1,
        pp_size=1,
        ulysses_size=1,
        ringattn_size=1,
        dp_mode="fsdp2",
    )
    return torch.device("cuda", local_rank)


def _single_rank_oracle(weight_full: torch.Tensor, grad_full: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Run Muon on a single rank with the full unsharded gradient and return updated weight."""
    p = torch.nn.Parameter(weight_full.clone())
    p.grad = grad_full.clone()
    opt = Muon(
        [{"params": [p], "lr": 0.1, "use_muon": True, "weight_decay": 0.0}],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_steps=5,
        weight_decay=0.0,
        distributed_mode=mode,
    )
    opt.step()
    return p.detach()


def _full_tensor(d):
    if hasattr(d, "full_tensor"):
        return d.full_tensor()
    return d


def _layout_shape_and_placements(layout: str):
    """Return (global_shape, placements) for a named test layout."""
    if layout == "linear_2d":
        # Plain Linear weight, row-sharded on dim 0 — the dense FSDP2 case.
        return (16, 12), [Shard(0)]
    if layout == "moe_experts_3d":
        # 3D MoE expert weight [E, H, I] sharded on dim 1 — mirrors how
        # ``parallelize_model_fsdp2`` shards EP-experts (Shard(1) on ep_fsdp).
        # Tests the deferred 3D-reshape branch in ``_muon_step``.
        return (4, 32, 64), [Shard(1)]
    raise ValueError(f"unknown layout: {layout!r}")


def _run(distributed_mode: str, layout: str) -> None:
    device = _setup_dist()
    mesh = get_parallel_state().fsdp_mesh

    torch.manual_seed(42)
    global_shape, placements = _layout_shape_and_placements(layout)
    weight_full = torch.randn(*global_shape, device=device, dtype=torch.float32)
    grad_full = torch.randn(*global_shape, device=device, dtype=torch.float32)
    # Make every rank see the same global init by broadcasting from rank 0.
    dist.broadcast(weight_full, src=0)
    dist.broadcast(grad_full, src=0)

    weight_d = distribute_tensor(weight_full.clone(), mesh, placements)
    p = torch.nn.Parameter(weight_d)

    grad_d = distribute_tensor(grad_full.clone(), mesh, placements)
    p.grad = grad_d

    opt = Muon(
        [{"params": [p], "lr": 0.1, "use_muon": True, "weight_decay": 0.0}],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_steps=5,
        weight_decay=0.0,
        distributed_mode=distributed_mode,
    )
    opt.step()

    full_after = _full_tensor(p.data)

    # Build a single-process oracle running full-gradient mode and compare.
    expected_full_grad = _single_rank_oracle(weight_full, grad_full, mode="full_gradient")

    if dist.get_rank() == 0:
        if distributed_mode == "full_gradient":
            err = (full_after - expected_full_grad).abs().max().item()
            assert err < 1e-4, (
                f"[layout={layout}] full_gradient mode does not match single-rank oracle: max abs err = {err}"
            )
            print(f"[rank 0] [{layout}] full_gradient max err vs oracle: {err:.6e}")
        else:
            # shard_local mode: NS runs per local-shard. Should NOT match the
            # full-gradient oracle on multi-rank — the orthogonalization is
            # genuinely different between shard-local row-slabs (or H-strips
            # for 3D) and the full matrix.
            err = (full_after - expected_full_grad).abs().max().item()
            print(f"[rank 0] [{layout}] shard_local diff vs full-gradient oracle: {err:.6e}")
            if dist.get_world_size() > 1:
                assert err > 1e-4, (
                    f"[layout={layout}] shard_local should differ from full-gradient oracle "
                    f"on multi-rank; got max abs err = {err} (test premise broken if too small)"
                )

    dist.barrier()
    dist.destroy_process_group()


def _main() -> None:
    mode = os.environ.get("XORL_TEST_MUON_MODE", "full_gradient")
    layout = os.environ.get("XORL_TEST_MUON_LAYOUT", "linear_2d")
    _run(mode, layout)


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_full_gradient_matches_single_rank_oracle_2d():
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"XORL_TEST_MUON_MODE": "full_gradient", "XORL_TEST_MUON_LAYOUT": "linear_2d"},
        )
        result.assert_success("2D Shard(0) full_gradient Muon should match single-rank oracle")

    @skip_if_gpu_count_less_than(2)
    def test_shard_local_differs_from_full_gradient_oracle_2d():
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"XORL_TEST_MUON_MODE": "shard_local", "XORL_TEST_MUON_LAYOUT": "linear_2d"},
        )
        result.assert_success("2D shard_local should differ from full-gradient oracle on >1 rank")

    @skip_if_gpu_count_less_than(2)
    def test_full_gradient_matches_single_rank_oracle_3d_moe():
        # Exercises the deferred-reshape path in ``_muon_step`` for an EP-experts-style
        # 3D weight ``[E, H, I]`` sharded on ``H`` (Shard(1)).
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"XORL_TEST_MUON_MODE": "full_gradient", "XORL_TEST_MUON_LAYOUT": "moe_experts_3d"},
        )
        result.assert_success("3D Shard(1) full_gradient Muon should match single-rank oracle")

    @skip_if_gpu_count_less_than(2)
    def test_shard_local_differs_from_full_gradient_oracle_3d_moe():
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"XORL_TEST_MUON_MODE": "shard_local", "XORL_TEST_MUON_LAYOUT": "moe_experts_3d"},
        )
        result.assert_success("3D shard_local should differ from full-gradient oracle on >1 rank")


if __name__ == "__main__":
    _main()
