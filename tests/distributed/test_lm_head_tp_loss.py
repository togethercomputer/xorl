"""Regression test for lm-head-only vocab TP over sequence-sharded inputs."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xorl.ops.loss.causallm_loss import fsdp_sharded_causallm_loss_function  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_lm_head_tp_loss_case() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        assert world_size == 4
        lm_tp = 2
        tp_rank = rank % lm_tp
        lm_tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        replica_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        lm_tp_group = lm_tp_groups[rank // lm_tp]
        replica_group = replica_groups[tp_rank]

        torch.manual_seed(2026)
        batch_size = 1
        local_seq = 2
        hidden_size = 5
        vocab_size = 8
        local_vocab = vocab_size // lm_tp

        hidden_chunks = [
            torch.randn(batch_size, local_seq, hidden_size, dtype=torch.float32) for _ in range(world_size)
        ]
        label_chunks = [
            torch.tensor([[0, 1]]),
            torch.tensor([[4, -100]]),
            torch.tensor([[2, 5]]),
            torch.tensor([[7, 3]]),
        ]
        full_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)

        hidden_ref = torch.cat(hidden_chunks, dim=1).detach().clone().requires_grad_(True)
        weight_ref = full_weight.detach().clone().requires_grad_(True)
        labels_ref = torch.cat(label_chunks, dim=1)
        ref_logits = hidden_ref.reshape(-1, hidden_size) @ weight_ref.t()
        ref_labels = labels_ref.reshape(-1)
        valid = (ref_labels != -100).sum().clamp(min=1).float()
        ref_loss = F.cross_entropy(ref_logits, ref_labels, reduction="sum", ignore_index=-100) / valid
        ref_loss.backward()

        local_hidden = hidden_chunks[rank].detach().clone().requires_grad_(True)
        start = tp_rank * local_vocab
        local_weight = full_weight[start : start + local_vocab].detach().clone().requires_grad_(True)
        local_labels = label_chunks[rank]
        global_valid_tokens = (local_labels != -100).sum().to(dtype=torch.float32)
        dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM)

        loss = fsdp_sharded_causallm_loss_function(
            hidden_states=local_hidden,
            weight=local_weight,
            labels=local_labels,
            sp_group=lm_tp_group,
            fsdp_group=lm_tp_group,
            sequence_group=lm_tp_group,
            vocab_group=lm_tp_group,
            loss_reduce_group=dist.group.WORLD,
            loss_reduce_divisor=lm_tp,
            num_chunks=2,
            global_valid_tokens=global_valid_tokens,
        ).loss
        loss.backward()
        dist.all_reduce(local_weight.grad, op=dist.ReduceOp.SUM, group=replica_group)

        ref_hidden_grad = hidden_ref.grad[:, rank * local_seq : (rank + 1) * local_seq, :]
        ref_weight_grad = weight_ref.grad[start : start + local_vocab]
        torch.testing.assert_close(loss.detach(), ref_loss.detach(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(local_hidden.grad, ref_hidden_grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(local_weight.grad, ref_weight_grad, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_lm_head_tp_loss_cpu():
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=120)
        result.assert_success()


if __name__ == "__main__":
    _run_lm_head_tp_loss_case()
