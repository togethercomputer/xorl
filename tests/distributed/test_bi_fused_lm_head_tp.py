"""Distributed gates for the exact bi-fused LM-head TP path.

Run through pytest (which self-launches torchrun) or directly with two ranks::

    torchrun --nproc_per_node=2 tests/distributed/test_bi_fused_lm_head_tp.py
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import xorl.distributed.parallel_state as parallel_state_impl
from xorl.objectives.causallm_loss import causallm_loss_function
from xorl.objectives.reducers import TokenPartial
from xorl.ops import bi_families_v2
from xorl.ops.loss.bi_fused_lm_head import (
    bi_fused_per_token_ce,
    bi_fused_vocab_parallel_per_token_ce,
)


_HIDDEN = 128
_VOCAB = 512


def _setup() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def _full_weight() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(731)
    return torch.randn((_VOCAB, _HIDDEN), generator=generator).to(torch.bfloat16).cuda()


def _local_rows(rank: int, row_counts: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = row_counts[rank]
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    hidden = torch.randn((rows, _HIDDEN), generator=generator).to(torch.bfloat16).cuda()
    labels = (torch.arange(rows, device="cuda", dtype=torch.int64) * 67 + rank * 101) % _VOCAB
    if rows:
        labels[-1] = -100
    return hidden, labels


def _assert_forward_bytes_and_backward(
    rank: int,
    row_counts: tuple[int, int],
    *,
    trainable_weight: bool,
    mixed_metadata: bool,
) -> None:
    full_weight = _full_weight()
    local_vocab = _VOCAB // 2
    shard_start = rank * local_vocab
    local_weight = full_weight[shard_start : shard_start + local_vocab].contiguous().requires_grad_(trainable_weight)
    hidden, labels = _local_rows(rank, row_counts, seed=1901 + sum(row_counts))
    hidden = hidden.requires_grad_(True)

    temperature: float | torch.Tensor = 1.0
    top_ks = top_ps = min_ps = None
    if mixed_metadata and rank == 0:
        temperature = torch.linspace(0.7, 1.3, hidden.shape[0], dtype=torch.float32, device="cuda")
        top_ks = torch.full((hidden.shape[0],), _VOCAB, dtype=torch.int64, device="cuda")
        top_ps = torch.ones(hidden.shape[0], dtype=torch.float32, device="cuda")
        min_ps = torch.zeros(hidden.shape[0], dtype=torch.float32, device="cuda")

    actual = bi_fused_vocab_parallel_per_token_ce(
        hidden,
        local_weight,
        labels,
        dist.group.WORLD,
        temperature=temperature,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
    )
    reference_weight = full_weight.detach().clone().requires_grad_(trainable_weight)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference = bi_fused_per_token_ce(
        reference_hidden,
        reference_weight,
        labels,
        temperature=temperature,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
    )
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))

    global_valid = (labels != -100).sum().to(torch.float32)
    dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
    loss_output = causallm_loss_function(
        hidden.detach(),
        local_weight.detach(),
        labels,
        return_per_token=True,
        ce_mode="bi_fused",
        tp_group=dist.group.WORLD,
        lm_head_fp32=True,
        lm_head=SimpleNamespace(_xorl_fsdp_sharded_lm_head_loss=True),
        loss_reducer=TokenPartial(scale=global_valid),
        logprob_temperature=temperature,
        logprob_top_k=top_ks if top_ks is not None else _VOCAB,
        logprob_top_p=top_ps if top_ps is not None else 1.0,
        logprob_min_p=min_ps if min_ps is not None else 0.0,
    )
    expected_loss = reference.detach().sum() / global_valid
    assert torch.equal(
        loss_output.loss.reshape(1).view(torch.uint8),
        expected_loss.reshape(1).view(torch.uint8),
    )
    assert torch.equal(loss_output.per_token_loss.view(torch.uint8), reference.view(torch.uint8))

    grad = torch.linspace(0.25, 1.25, actual.shape[0], dtype=torch.float32, device="cuda")
    (actual * grad).sum().backward()
    if actual.numel():
        (reference * grad).sum().backward()
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=3e-2, atol=3e-2)
    else:
        assert hidden.grad is not None and hidden.grad.numel() == 0

    if trainable_weight:
        expected_weight_grad = reference_weight.grad.float()
        dist.all_reduce(expected_weight_grad, op=dist.ReduceOp.SUM)
        expected_shard = expected_weight_grad[shard_start : shard_start + local_vocab]
        torch.testing.assert_close(local_weight.grad.float(), expected_shard, rtol=3e-2, atol=3e-2)
    else:
        assert local_weight.grad is None


def _run_cases(rank: int, world_size: int) -> None:
    assert world_size == 2
    bi_families_v2._select_glm52_families_v2()
    try:
        # More than one local scheduling chunk on rank 0 and a smaller owner on rank 1.
        _assert_forward_bytes_and_backward(
            rank,
            (11, 3),
            trainable_weight=True,
            mixed_metadata=True,
        )
        dist.barrier()

        # Every rank follows the same collective schedule when one owner has no rows.
        _assert_forward_bytes_and_backward(
            rank,
            (11, 0),
            trainable_weight=False,
            mixed_metadata=False,
        )
        dist.barrier()

        # An entirely empty microbatch remains differentiable and collective-safe.
        full_weight = _full_weight()
        local_weight = full_weight[rank * 256 : (rank + 1) * 256].contiguous()
        hidden = torch.empty((0, _HIDDEN), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        labels = torch.empty((0,), dtype=torch.int64, device="cuda")
        empty_ce = bi_fused_vocab_parallel_per_token_ce(
            hidden,
            local_weight,
            labels,
            dist.group.WORLD,
        )
        assert empty_ce.shape == (0,)
        empty_ce.sum().backward()
        assert hidden.grad is not None and hidden.grad.shape == hidden.shape
    finally:
        bi_families_v2._select_nonexact_families()


def main() -> None:
    rank, world_size = _setup()
    previous_parallel_state = parallel_state_impl._PARALLEL_STATE
    parallel_state_impl._PARALLEL_STATE = SimpleNamespace(
        lm_head_tp_size=world_size,
        lm_head_tp_group=dist.group.WORLD,
        lm_head_tp_replica_group=None,
        tp_enabled=False,
    )
    try:
        _run_cases(rank, world_size)
        dist.barrier()
        if rank == 0:
            print("bi_fused LM-head TP ragged/empty forward+backward passed")
    finally:
        parallel_state_impl._PARALLEL_STATE = previous_parallel_state
        dist.destroy_process_group()


if __name__ != "__main__":
    import pytest

    from tests.distributed.distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

    @pytest.mark.gpu
    @pytest.mark.distributed
    @skip_if_gpu_count_less_than(2)
    def test_bi_fused_lm_head_tp_2gpu() -> None:
        result = run_distributed_script(os.path.abspath(__file__), num_gpus=2, timeout=240)
        result.assert_success()


if __name__ == "__main__":
    main()
