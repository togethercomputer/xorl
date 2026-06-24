"""End-to-end lm-head-only TP: a *real* FSDP-sharded lm_head over the dedicated
lm_head_mesh + the vocab-parallel CE, checked against an eager reference for both
loss and gradients. Unlike test_lm_head_tp_loss (which hand-slices the weight),
this exercises the production path: fully_shard(lm_head, mesh=lm_head_mesh) makes
lm_head.weight a DTensor and the loss to_local()s it. FSDP's reduce hook does NOT
fire (the vocab-parallel CE reads the weight directly), so the cp_replica x DP
gradients are summed explicitly by sync_lm_head_tp_gradient -- which is exactly
what this test validates, for dp=1 (cp_replica only) and dp=2 (DP replica).
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch.distributed as dist  # noqa: E402

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state  # noqa: E402
from xorl.ops.loss.causallm_loss import fsdp_sharded_causallm_loss_function  # noqa: E402
from xorl.trainers.training_utils import sync_lm_head_tp_gradient  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_case(dp_shard: int, ulysses: int, lm_head_tp: int) -> None:
    dist.init_process_group(backend="gloo")
    try:
        init_parallel_state(
            dp_size=dp_shard,
            dp_shard_size=dp_shard,
            ulysses_size=ulysses,
            lm_head_tp_size=lm_head_tp,
            device_type="cpu",
        )
        ps = get_parallel_state()
        rank = dist.get_rank()
        # rank layout with pp=dp_replicate=ringattn=tp=1: rank = dp_idx*ulysses + u_idx
        dp_idx, u_idx = divmod(rank, ulysses)

        hidden_size, vocab_size = 5, 8
        local_seq = 2
        full_seq = ulysses * local_seq
        torch.manual_seed(2026)

        # One distinct batch per DP cell; the weight is shared. Built identically on
        # every rank so each can compute the global reference + global valid tokens.
        full_hidden = torch.randn(dp_shard, 1, full_seq, hidden_size, dtype=torch.float32)
        full_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)
        full_labels = torch.randint(0, vocab_size, (dp_shard, 1, full_seq))
        full_labels[0, 0, 0] = -100  # one masked token

        # Eager reference: CE summed over every DP cell + the whole sequence, divided
        # by the global valid-token count. ref_weight.grad sums all cells.
        ref_weight = full_weight.detach().clone().requires_grad_(True)
        ref_hiddens = [full_hidden[d].detach().clone().requires_grad_(True) for d in range(dp_shard)]
        ce_sum = torch.zeros((), dtype=torch.float32)
        total_valid = 0
        for d in range(dp_shard):
            logits = ref_hiddens[d].reshape(-1, hidden_size) @ ref_weight.t()
            labs = full_labels[d].reshape(-1)
            ce_sum = ce_sum + F.cross_entropy(logits, labs, reduction="sum", ignore_index=-100)
            total_valid += int((labs != -100).sum().item())
        gv = float(max(total_valid, 1))
        ref_loss = ce_sum / gv
        ref_loss.backward()

        # lm-head-TP path: a real FSDP-sharded lm_head over lm_head_mesh.
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(full_weight)
        setattr(lm_head, "_xorl_fsdp_sharded_lm_head_loss", True)
        fully_shard(lm_head, mesh=ps.lm_head_mesh)
        lm_head.set_gradient_divide_factor(1.0)  # sum (global-valid normalization)

        local_hidden = (
            full_hidden[dp_idx][:, u_idx * local_seq : (u_idx + 1) * local_seq, :].detach().clone().requires_grad_(True)
        )
        local_labels = full_labels[dp_idx][:, u_idx * local_seq : (u_idx + 1) * local_seq]
        global_valid_tokens = torch.tensor(gv, dtype=torch.float32)

        out = fsdp_sharded_causallm_loss_function(
            hidden_states=local_hidden,
            weight=lm_head.weight,  # DTensor; loss to_local()s it
            labels=local_labels,
            sp_group=ps.lm_head_tp_group,
            fsdp_group=ps.lm_head_tp_group,
            num_chunks=2,
            global_valid_tokens=global_valid_tokens,
            sequence_group=ps.lm_head_tp_group,
            vocab_group=ps.lm_head_tp_group,
            # Sum per-cp_replica/DP losses (distinct sequence shards / batches of the
            # same vocab slice). Non-differentiable; the matching weight-grad sum is
            # sync_lm_head_tp_gradient below. divisor=1 (no within-replica duplication).
            loss_reduce_group=ps.lm_head_tp_replica_group,
            loss_reduce_divisor=1.0,
        )
        loss = out.loss
        loss.backward()

        # The framework's lm-head-TP grad sync: sum the weight grad over the replica
        # dim (cp_replica x DP), which FSDP's hook never did (weight used directly).
        sync_lm_head_tp_gradient(lm_head, ps.lm_head_tp_replica_group)

        # Loss matches the global reference on every rank.
        torch.testing.assert_close(loss.detach(), ref_loss.detach(), rtol=1e-5, atol=1e-5)

        # Gather the full weight grad over the lm_head_tp dim and compare.
        local_wgrad = lm_head.weight.grad.to_local()
        tp_world = dist.get_world_size(ps.lm_head_tp_group)
        gathered = [torch.empty_like(local_wgrad) for _ in range(tp_world)]
        dist.all_gather(gathered, local_wgrad.contiguous(), group=ps.lm_head_tp_group)
        full_wgrad = torch.cat(gathered, dim=0)
        torch.testing.assert_close(full_wgrad, ref_weight.grad, rtol=1e-4, atol=1e-5)

        # local hidden grad matches the reference slice for this (dp, seq) shard.
        ref_hidden_grad = ref_hiddens[dp_idx].grad[:, u_idx * local_seq : (u_idx + 1) * local_seq, :]
        torch.testing.assert_close(local_hidden.grad, ref_hidden_grad, rtol=1e-4, atol=1e-5)
        print(f"rank{rank} dp_shard={dp_shard} OK loss={loss.item():.6f}")
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_lm_head_tp_fsdp_e2e_dp1_cpu():
        # dp=1, cp=4, lm_head_tp=2 -> cp_replica=2 (replica dim is purely sequence).
        result = run_distributed_script(
            SCRIPT_PATH, num_gpus=4, timeout=180, extra_env={"XORL_LMHEAD_E2E_CFG": "1,4,2"}
        )
        result.assert_success()

    def test_lm_head_tp_fsdp_e2e_dp2_cpu():
        # dp=2, cp=2, lm_head_tp=2 -> cp_replica=1; the replica dim is the DP dim, so
        # this validates that distinct-batch DP gradients are summed for lm-head TP.
        result = run_distributed_script(
            SCRIPT_PATH, num_gpus=4, timeout=180, extra_env={"XORL_LMHEAD_E2E_CFG": "2,2,2"}
        )
        result.assert_success()


if __name__ == "__main__":
    cfg = os.environ.get("XORL_LMHEAD_E2E_CFG", "1,4,2")
    _dp, _u, _tp = (int(x) for x in cfg.split(","))
    _run_case(_dp, _u, _tp)
