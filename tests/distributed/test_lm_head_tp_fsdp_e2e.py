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
from xorl.ops.loss.opd_loss import opd_vocab_parallel_loss_function  # noqa: E402
from xorl.ops.loss.reducers import TokenPartial  # noqa: E402
from xorl.server.runner.model_runner import ModelRunner  # noqa: E402
from xorl.trainers.training_utils import sync_lm_head_tp_gradient, sync_lm_head_tp_parameters  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_case(dp_replicate: int, dp_shard: int, ulysses: int, lm_head_tp: int, ep_size: int = 1) -> None:
    dist.init_process_group(backend="gloo")
    try:
        dp_size = dp_replicate * dp_shard
        init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=dp_replicate,
            dp_shard_size=dp_shard,
            ulysses_size=ulysses,
            ep_size=ep_size,
            lm_head_tp_size=lm_head_tp,
            device_type="cpu",
        )
        ps = get_parallel_state()
        rank = dist.get_rank()
        if ep_size > 1:
            assert ps.ep_size == ep_size
            assert ps.ep_enabled
            assert ps.ep_fsdp_device_mesh is not None
            expected_tp = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}[rank]
            expected_replica = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}[rank]
            assert dist.get_process_group_ranks(ps.lm_head_tp_group) == expected_tp
            assert dist.get_process_group_ranks(ps.lm_head_tp_replica_group) == expected_replica
        # rank layout with pp=dp_replicate=ringattn=tp=1:
        # - CP-sourced lm-head TP: rank = dp_idx * ulysses + cp_idx.
        # - no-CP DP-sourced lm-head TP: ulysses=1, so dp_idx=rank and cp_idx=0.
        # For HSDP, dp_idx spans both dp_replicate and dp_shard cells; lm-head TP
        # groups still stay inside each dp_shard row.
        cp_size = max(1, ulysses)
        dp_idx, cp_idx = divmod(rank, cp_size)

        hidden_size, vocab_size = 5, 8
        local_seq = 2
        full_seq = cp_size * local_seq
        torch.manual_seed(2026)

        # One distinct batch per DP cell; the weight is shared. Built identically on
        # every rank so each can compute the global reference + global valid tokens.
        full_hidden = torch.randn(dp_size, 1, full_seq, hidden_size, dtype=torch.float32)
        full_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)
        full_labels = torch.randint(0, vocab_size, (dp_size, 1, full_seq))
        full_labels[0, 0, 0] = -100  # one masked token

        # Eager reference: CE summed over every DP cell + the whole sequence, divided
        # by the global valid-token count. ref_weight.grad sums all cells.
        ref_weight = full_weight.detach().clone().requires_grad_(True)
        ref_hiddens = [full_hidden[d].detach().clone().requires_grad_(True) for d in range(dp_size)]
        ce_sum = torch.zeros((), dtype=torch.float32)
        total_valid = 0
        for d in range(dp_size):
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

        local_weight, vocab_start, vocab_end, vocab_group = ModelRunner._opd_student_vocab_shard(lm_head.weight)
        assert vocab_group == ps.lm_head_tp_group
        assert local_weight.shape[0] == vocab_end - vocab_start
        expected_rows = vocab_size // lm_head_tp
        vocab_rank = dist.get_rank(ps.lm_head_tp_group)
        assert (vocab_start, vocab_end) == (
            vocab_rank * expected_rows,
            (vocab_rank + 1) * expected_rows,
        )
        with torch.no_grad():
            if dist.get_rank(ps.lm_head_tp_replica_group) != 0:
                local_weight.add_(float(rank + 1))
        sync_lm_head_tp_parameters(lm_head, ps.lm_head_tp_replica_group)
        synced_weight, vocab_start, vocab_end, vocab_group = ModelRunner._opd_student_vocab_shard(lm_head.weight)
        assert vocab_group == ps.lm_head_tp_group
        torch.testing.assert_close(synced_weight.detach().clone(), full_weight[vocab_start:vocab_end])

        local_hidden = (
            full_hidden[dp_idx][:, cp_idx * local_seq : (cp_idx + 1) * local_seq, :]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        local_labels = full_labels[dp_idx][:, cp_idx * local_seq : (cp_idx + 1) * local_seq]
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
        ref_hidden_grad = ref_hiddens[dp_idx].grad[:, cp_idx * local_seq : (cp_idx + 1) * local_seq, :]
        torch.testing.assert_close(local_hidden.grad, ref_hidden_grad, rtol=1e-4, atol=1e-5)
        print(f"rank{rank} dp_replicate={dp_replicate} dp_shard={dp_shard} ep={ep_size} OK loss={loss.item():.6f}")
    finally:
        dist.destroy_process_group()


def _run_opd_case(dp_replicate: int, dp_shard: int, ulysses: int, lm_head_tp: int) -> None:
    dist.init_process_group(backend="gloo")
    try:
        dp_size = dp_replicate * dp_shard
        init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=dp_replicate,
            dp_shard_size=dp_shard,
            ulysses_size=ulysses,
            lm_head_tp_size=lm_head_tp,
            device_type="cpu",
        )
        ps = get_parallel_state()
        rank = dist.get_rank()
        cp_size = max(1, ulysses)
        dp_idx, cp_idx = divmod(rank, cp_size)

        hidden_size, vocab_size = 6, 10
        local_seq = 3
        full_seq = cp_size * local_seq
        torch.manual_seed(2027)

        full_student_hidden = torch.randn(dp_size, 1, full_seq, hidden_size, dtype=torch.float32) * 0.25
        full_teacher_hidden = torch.randn(dp_size, 1, full_seq, hidden_size, dtype=torch.float32) * 0.25
        full_student_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32) * 0.05
        full_teacher_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32) * 0.05
        full_labels = torch.randint(0, vocab_size, (dp_size, 1, full_seq))
        full_labels[0, 0, 0] = -100
        full_token_weights = torch.linspace(0.75, 1.35, dp_size * full_seq, dtype=torch.float32).reshape(
            dp_size, 1, full_seq
        )

        ref_weight = full_student_weight.detach().clone().requires_grad_(True)
        ref_hiddens = [full_student_hidden[d].detach().clone().requires_grad_(True) for d in range(dp_size)]
        ref_loss = torch.zeros((), dtype=torch.float32)
        for d in range(dp_size):
            student_logits = ref_hiddens[d].reshape(-1, hidden_size) @ ref_weight.t()
            teacher_logits = full_teacher_hidden[d].reshape(-1, hidden_size) @ full_teacher_weight.t()
            labels = full_labels[d].reshape(-1)
            token_weights = full_token_weights[d].reshape(-1)
            student_logprobs = F.log_softmax(student_logits, dim=-1)
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
            token_kl = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(dim=-1)
            token_kl = token_kl * (labels != -100).float()
            ref_loss = ref_loss + (token_kl * token_weights).sum()
        ref_loss.backward()

        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(full_student_weight)
        setattr(lm_head, "_xorl_fsdp_sharded_lm_head_loss", True)
        fully_shard(lm_head, mesh=ps.lm_head_mesh)
        lm_head.set_gradient_divide_factor(1.0)

        local_weight, vocab_start, vocab_end, vocab_group = ModelRunner._opd_student_vocab_shard(lm_head.weight)
        assert vocab_group == ps.lm_head_tp_group
        with torch.no_grad():
            if dist.get_rank(ps.lm_head_tp_replica_group) != 0:
                local_weight.add_(float(rank + 1))
        sync_lm_head_tp_parameters(lm_head, ps.lm_head_tp_replica_group)
        local_weight, vocab_start, vocab_end, vocab_group = ModelRunner._opd_student_vocab_shard(lm_head.weight)
        assert vocab_group == ps.lm_head_tp_group
        torch.testing.assert_close(local_weight.detach().clone(), full_student_weight[vocab_start:vocab_end])
        teacher_weight_local = full_teacher_weight[vocab_start:vocab_end].detach().clone()

        local_student_hidden = (
            full_student_hidden[dp_idx][:, cp_idx * local_seq : (cp_idx + 1) * local_seq, :]
            .detach()
            .clone()
            .reshape(-1, hidden_size)
            .requires_grad_(True)
        )
        local_teacher_hidden = (
            full_teacher_hidden[dp_idx][:, cp_idx * local_seq : (cp_idx + 1) * local_seq, :]
            .detach()
            .clone()
            .reshape(-1, hidden_size)
        )
        local_labels = full_labels[dp_idx][:, cp_idx * local_seq : (cp_idx + 1) * local_seq].reshape(-1)
        local_token_weights = full_token_weights[dp_idx][:, cp_idx * local_seq : (cp_idx + 1) * local_seq].reshape(-1)

        out = opd_vocab_parallel_loss_function(
            student_hidden_flat=local_student_hidden,
            student_weight_local=local_weight,
            labels=local_labels,
            teacher_hidden_flat=local_teacher_hidden,
            teacher_weight_local=teacher_weight_local,
            teacher_weights=local_token_weights,
            loss_reducer=TokenPartial(scale=torch.tensor(1.0)),
            hidden_match_coef=0.0,
            teacher_lm_head_fp32=True,
            group=vocab_group,
        )
        out.loss.backward()
        sync_lm_head_tp_gradient(lm_head, ps.lm_head_tp_replica_group)

        reported_loss_sum = out.loss.detach().clone()
        dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(reported_loss_sum, ref_loss.detach(), rtol=1e-5, atol=1e-5)

        local_wgrad = lm_head.weight.grad.to_local()
        tp_world = dist.get_world_size(ps.lm_head_tp_group)
        gathered = [torch.empty_like(local_wgrad) for _ in range(tp_world)]
        dist.all_gather(gathered, local_wgrad.contiguous(), group=ps.lm_head_tp_group)
        full_wgrad = torch.cat(gathered, dim=0)
        torch.testing.assert_close(full_wgrad, ref_weight.grad, rtol=1e-4, atol=1e-5)

        ref_hidden_grad = (
            ref_hiddens[dp_idx].grad[:, cp_idx * local_seq : (cp_idx + 1) * local_seq, :].reshape(-1, hidden_size)
        )
        torch.testing.assert_close(local_student_hidden.grad, ref_hidden_grad, rtol=1e-4, atol=1e-5)
        print(
            f"rank{rank} OPD dp_replicate={dp_replicate} dp_shard={dp_shard} OK raw_loss={reported_loss_sum.item():.6f}"
        )
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    def test_lm_head_tp_fsdp_topology_and_loss_mode_policy():
        cases = (
            ("cp-replica-dp1", "1,1,4,2", None),
            ("cp-dp2-ep2", "1,2,2,2,2", None),
            ("no-cp-dp", "1,4,1,2", None),
            ("no-cp-hsdp", "2,2,1,2", None),
            ("opd-no-cp-dp", "1,4,1,2", "opd"),
            ("opd-no-cp-hsdp", "2,2,1,2", "opd"),
        )
        for case_id, config, mode in cases:
            extra_env = {"XORL_LMHEAD_E2E_CFG": config}
            if mode is not None:
                extra_env["XORL_LMHEAD_E2E_MODE"] = mode
            result = run_distributed_script(
                SCRIPT_PATH,
                num_gpus=4,
                timeout=180,
                extra_env=extra_env,
            )
            try:
                result.assert_success()
            except AssertionError as error:
                raise AssertionError(f"{case_id}: {error}") from error


if __name__ == "__main__":
    cfg = os.environ.get("XORL_LMHEAD_E2E_CFG", "1,1,4,2")
    parts = [int(x) for x in cfg.split(",")]
    if len(parts) == 3:
        _rep, _dp, _u, _tp, _ep = 1, *parts, 1
    elif len(parts) == 4:
        _rep, _dp, _u, _tp = parts
        _ep = 1
    elif len(parts) == 5:
        _rep, _dp, _u, _tp, _ep = parts
    else:
        raise ValueError(f"XORL_LMHEAD_E2E_CFG must have 3, 4, or 5 comma-separated ints, got {cfg!r}")
    if os.environ.get("XORL_LMHEAD_E2E_MODE") == "opd":
        _run_opd_case(_rep, _dp, _u, _tp)
    else:
        _run_case(_rep, _dp, _u, _tp, _ep)
