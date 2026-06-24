"""Regression tests for vocab-parallel CE with uneven vocab shards."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _local_shard_bounds(vocab_size: int, world_size: int, rank: int) -> tuple[int, int]:
    base = vocab_size // world_size
    remainder = vocab_size % world_size
    size = base + int(rank < remainder)
    start = rank * base + min(rank, remainder)
    return start, start + size


def _run_uneven_vocab_shard_case() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        torch.manual_seed(1234)
        batch_tokens = 9
        hidden_size = 5
        vocab_size = 11

        assert vocab_size % world_size != 0

        hidden_ref = torch.randn(batch_tokens, hidden_size, requires_grad=True)
        weight_ref = torch.randn(vocab_size, hidden_size, requires_grad=True)
        labels = torch.tensor([0, 5, 6, 10, -100, 4, 8, 1, 9])

        ref_logits = hidden_ref @ weight_ref.t()
        ref_ce = F.cross_entropy(ref_logits, labels, reduction="none", ignore_index=-100)
        valid = (labels != -100).sum().clamp(min=1).float()
        (ref_ce.sum() / valid).backward()

        shard_start, shard_end = _local_shard_bounds(vocab_size, world_size, rank)
        hidden_par = hidden_ref.detach().clone().requires_grad_(True)
        weight_par = weight_ref.detach()[shard_start:shard_end].contiguous().requires_grad_(True)

        par_ce = vocab_parallel_cross_entropy(
            hidden_par,
            weight_par,
            labels,
            dist.group.WORLD,
            ignore_index=-100,
            use_compile=False,
        )
        (par_ce.sum() / valid).backward()

        torch.testing.assert_close(par_ce, ref_ce, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(hidden_par.grad, hidden_ref.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(weight_par.grad, weight_ref.grad[shard_start:shard_end], rtol=1e-5, atol=1e-6)
        if rank == 0:
            print("Uneven vocab shard case passed")
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_vocab_parallel_ce_uneven_vocab_shards_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=2, timeout=120)
        result.assert_success()


if __name__ == "__main__":
    _run_uneven_vocab_shard_case()
