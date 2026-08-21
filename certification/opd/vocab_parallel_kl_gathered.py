#!/usr/bin/env python3
"""Multi-process test for vocab_parallel_reverse_kl_gathered — the FSDP integration
glue (gather activations, shard weights, local-slice grad).

Each rank holds its OWN token slice [n_local,H] + its vocab shard [V/world,H] (the
real FSDP layout: the lm-head shard group also data-shards tokens). We check that
after gather→VP-KL→loss.sum()→backward:
  - this rank's local hidden grad == single-process full-vocab reference grad for
    its token slice (NO cross-rank double-count), and
  - this rank's weight-shard grad == reference grad for its vocab shard.

Run (no GPU needed):
  PYTHONPATH=src .venv/bin/python certification/opd/vocab_parallel_kl_gathered.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from xorl.objectives.opd_loss import opd_vocab_parallel_loss_function
from xorl.objectives.opd_streaming_kl import streaming_reverse_kl_function
from xorl.objectives.reducers import TokenPartial
from xorl.ops.loss.vocab_parallel_reverse_kl import vocab_parallel_reverse_kl_gathered
from xorl.server.runner.model_runner import ModelRunner


WORLD = 4
NLOCAL = 12  # tokens per rank
H = 96
VLOCAL = 130  # vocab rows per rank
N = WORLD * NLOCAL
V = WORLD * VLOCAL
UNEVEN_COUNTS = (7, 0, 13, 3)
IGNORE = -100


def _full_inputs(num_tokens=N):
    torch.manual_seed(7)
    sh = torch.randn(num_tokens, H, dtype=torch.float32) * 0.3
    th = torch.randn(num_tokens, H, dtype=torch.float32) * 0.3
    sw = torch.randn(V, H, dtype=torch.float32) * 0.05
    tw = torch.randn(V, H, dtype=torch.float32) * 0.05
    labels = torch.randint(0, V, (num_tokens,))
    labels[: num_tokens // 6] = IGNORE
    return sh, th, sw, tw, labels


def reference(sh, sw, th, tw, labels):
    sh = sh.clone().requires_grad_(True)
    sw = sw.clone().requires_grad_(True)
    s = sh @ sw.t()
    t = th @ tw.t()
    slp = F.log_softmax(s, -1)
    tlp = F.log_softmax(t, -1)
    kl = (slp.exp() * (slp - tlp)).sum(-1) * (labels != IGNORE).float()
    kl.sum().backward()
    return sh.grad.detach(), sw.grad.detach()


def reference_opd(sh, sw, th, tw, labels, token_weights, hidden_weights, hidden_coef):
    sh = sh.clone().requires_grad_(True)
    sw = sw.clone().requires_grad_(True)
    s = sh @ sw.t()
    t = th @ tw.t()
    slp = F.log_softmax(s, -1)
    tlp = F.log_softmax(t, -1)
    kl = (slp.exp() * (slp - tlp)).sum(-1) * (labels != IGNORE).float()
    hidden = ((sh - th) ** 2).mean(dim=-1)
    loss = (kl * token_weights).sum() + hidden_coef * (hidden * hidden_weights).sum()
    loss.backward()
    return sh.grad.detach(), sw.grad.detach(), loss.detach(), kl.detach(), (kl * token_weights).detach()


def _worker(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    sh, th, sw, tw, labels = _full_inputs()
    ts, te = rank * NLOCAL, (rank + 1) * NLOCAL
    vs, ve = rank * VLOCAL, (rank + 1) * VLOCAL

    local_sh = sh[ts:te].clone().requires_grad_(True)
    local_th = th[ts:te].clone()
    sw_local = sw[vs:ve].clone().requires_grad_(True)
    tw_local = tw[vs:ve].clone()

    kl = vocab_parallel_reverse_kl_gathered(
        local_student_hidden=local_sh,
        student_weight_local=sw_local,
        local_teacher_hidden=local_th,
        teacher_weight_local=tw_local,
        labels_full=labels,
        ignore_index=IGNORE,
        group=None,
    )
    kl.sum().backward()

    if rank == 0:
        gh_ref, gw_ref = reference(sh, sw, th, tw, labels)
        ret["gh"] = (local_sh.grad - gh_ref[ts:te]).abs().max().item()
        ret["gh_scale"] = gh_ref[ts:te].abs().max().item()
        ret["gw"] = (sw_local.grad - gw_ref[vs:ve]).abs().max().item()
        ret["gw_scale"] = gw_ref[vs:ve].abs().max().item()
        ret["kl0"] = float(kl[0].item())
    dist.barrier()
    dist.destroy_process_group()


def _worker_opd(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29603")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    hidden_coef = 0.35
    num_tokens = sum(UNEVEN_COUNTS)
    sh, th, sw, tw, labels = _full_inputs(num_tokens)
    labels = labels.masked_fill(labels == IGNORE, 0)
    token_weights = torch.linspace(0.5, 1.5, num_tokens)
    hidden_weights = torch.linspace(1.2, 0.4, num_tokens)

    token_start = sum(UNEVEN_COUNTS[:rank])
    token_end = token_start + UNEVEN_COUNTS[rank]
    vocab_start = rank * VLOCAL
    vocab_end = (rank + 1) * VLOCAL

    local_sh = sh[token_start:token_end].clone().requires_grad_(True)
    local_th = th[token_start:token_end].clone()
    sw_local = sw[vocab_start:vocab_end].clone().requires_grad_(True)
    tw_local = tw[vocab_start:vocab_end].clone()
    local_labels = labels[token_start:token_end].clone()
    local_token_weights = token_weights[token_start:token_end].clone()
    local_hidden_weights = hidden_weights[token_start:token_end].clone()

    out = opd_vocab_parallel_loss_function(
        student_hidden_flat=local_sh,
        student_weight_local=sw_local,
        labels=local_labels,
        teacher_hidden_flat=local_th,
        teacher_weight_local=tw_local,
        teacher_weights=local_token_weights,
        hidden_match_weights=local_hidden_weights,
        loss_reducer=TokenPartial(scale=torch.tensor(1.0)),
        hidden_match_coef=hidden_coef,
        hidden_match_mode="mse",
        teacher_lm_head_fp32=True,
        group=None,
        debug_token_outputs=True,
    )
    out.loss.backward()
    reported_loss_sum = out.loss.detach().clone()
    dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)

    gh_ref, gw_ref, ref_loss, ref_kl, ref_weighted_kl = reference_opd(
        sh, sw, th, tw, labels, token_weights, hidden_weights, hidden_coef
    )
    debug_kl = out.metrics["_opd_debug_local_token_kl"].detach().cpu()
    debug_weighted_kl = out.metrics["_opd_debug_local_weighted_token_kl"].detach().cpu()
    debug_token_weight = out.metrics["_opd_debug_local_token_weight"].detach().cpu()
    ref_kl_local = ref_kl[token_start:token_end].detach().cpu()
    ref_weighted_local = ref_weighted_kl[token_start:token_end].detach().cpu()
    ref_weight_local = token_weights[token_start:token_end].detach().cpu()
    if debug_kl.numel():
        debug_kl_err = (debug_kl - ref_kl_local).abs().max().item()
        debug_weighted_kl_err = (debug_weighted_kl - ref_weighted_local).abs().max().item()
        debug_weight_err = (debug_token_weight - ref_weight_local).abs().max().item()
    else:
        debug_kl_err = 0.0
        debug_weighted_kl_err = 0.0
        debug_weight_err = 0.0
    if local_sh.numel():
        gh_err = (local_sh.grad - gh_ref[token_start:token_end]).abs().max().item()
        gh_scale = gh_ref[token_start:token_end].abs().max().item()
    else:
        gh_err = 0.0
        gh_scale = 1.0
    gw_err = (sw_local.grad - gw_ref[vocab_start:vocab_end]).abs().max().item()
    gw_scale = gw_ref[vocab_start:vocab_end].abs().max().item()
    ret[f"opd_gh_{rank}"] = gh_err
    ret[f"opd_gh_scale_{rank}"] = gh_scale
    ret[f"opd_gw_{rank}"] = gw_err
    ret[f"opd_gw_scale_{rank}"] = gw_scale
    ret[f"opd_report_loss_{rank}"] = float(reported_loss_sum.item())
    ret[f"opd_ref_loss_{rank}"] = float(ref_loss.item())
    ret[f"opd_kl_mean_{rank}"] = float(out.metrics["opd_kl"])
    ret[f"opd_ref_kl_mean_{rank}"] = float(ref_kl.mean().item())
    ret[f"opd_weighted_kl_mean_{rank}"] = float(out.metrics["opd_weighted_kl"])
    ret[f"opd_ref_weighted_kl_mean_{rank}"] = float(ref_weighted_kl.mean().item())
    ret[f"opd_vp_group_tokens_{rank}"] = int(out.metrics["opd_vocab_parallel_group_tokens"])
    ret[f"opd_vp_kl_sum_{rank}"] = float(out.metrics["opd_vocab_parallel_kl_sum"])
    ret[f"opd_ref_kl_sum_{rank}"] = float(ref_kl.sum().item())
    ret[f"opd_vp_weighted_kl_sum_{rank}"] = float(out.metrics["opd_vocab_parallel_weighted_kl_sum"])
    ret[f"opd_ref_weighted_kl_sum_{rank}"] = float(ref_weighted_kl.sum().item())
    ret[f"opd_debug_count_{rank}"] = int(debug_kl.numel())
    ret[f"opd_debug_kl_err_{rank}"] = float(debug_kl_err)
    ret[f"opd_debug_weighted_kl_err_{rank}"] = float(debug_weighted_kl_err)
    ret[f"opd_debug_weight_err_{rank}"] = float(debug_weight_err)
    dist.barrier()
    dist.destroy_process_group()


def _worker_opd_two_lm_tp_groups(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29605")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    lm_tp = 2
    assert world == 4
    tp_rank = rank % lm_tp
    group_idx = rank // lm_tp
    lm_tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    replica_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    lm_tp_group = lm_tp_groups[group_idx]
    replica_group = replica_groups[tp_rank]

    counts = (5, 0, 7, 3)
    num_tokens = sum(counts)
    vocab = lm_tp * VLOCAL
    torch.manual_seed(19)
    sh = torch.randn(num_tokens, H, dtype=torch.float32) * 0.25
    th = torch.randn(num_tokens, H, dtype=torch.float32) * 0.25
    sw = torch.randn(vocab, H, dtype=torch.float32) * 0.04
    tw = torch.randn(vocab, H, dtype=torch.float32) * 0.04
    labels = torch.randint(0, vocab, (num_tokens,))
    labels[::5] = IGNORE
    token_weights = torch.linspace(0.7, 1.3, num_tokens)

    token_start = sum(counts[:rank])
    token_end = token_start + counts[rank]
    vocab_start = tp_rank * VLOCAL
    vocab_end = (tp_rank + 1) * VLOCAL

    local_sh = sh[token_start:token_end].clone().requires_grad_(True)
    local_th = th[token_start:token_end].clone()
    sw_local = sw[vocab_start:vocab_end].clone().requires_grad_(True)
    tw_local = tw[vocab_start:vocab_end].clone()
    local_labels = labels[token_start:token_end].clone()
    local_token_weights = token_weights[token_start:token_end].clone()

    out = opd_vocab_parallel_loss_function(
        student_hidden_flat=local_sh,
        student_weight_local=sw_local,
        labels=local_labels,
        teacher_hidden_flat=local_th,
        teacher_weight_local=tw_local,
        teacher_weights=local_token_weights,
        loss_reducer=TokenPartial(scale=torch.tensor(1.0)),
        teacher_lm_head_fp32=True,
        group=lm_tp_group,
    )
    out.loss.backward()

    # ModelRunner reports the detached raw loss by summing over all ranks, while
    # each TP group rank contributes group_loss / lm_tp. The result should be the
    # one-copy global token loss, not lm_tp duplicates.
    reported_loss_sum = out.loss.detach().clone()
    dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)

    # Production sync_lm_head_tp_gradient sums the same vocab shard across replica
    # groups after backward. Mirror that here before comparing to the global ref.
    dist.all_reduce(sw_local.grad, op=dist.ReduceOp.SUM, group=replica_group)

    sh_ref = sh.clone().requires_grad_(True)
    sw_ref = sw.clone().requires_grad_(True)
    s = sh_ref @ sw_ref.t()
    t = th @ tw.t()
    slp = F.log_softmax(s, -1)
    tlp = F.log_softmax(t, -1)
    kl = (slp.exp() * (slp - tlp)).sum(-1) * (labels != IGNORE).float()
    ref_loss = (kl * token_weights).sum()
    ref_loss.backward()

    if local_sh.numel():
        gh_err = (local_sh.grad - sh_ref.grad[token_start:token_end]).abs().max().item()
        gh_scale = sh_ref.grad[token_start:token_end].abs().max().item()
    else:
        gh_err = 0.0
        gh_scale = 1.0
    gw_err = (sw_local.grad - sw_ref.grad[vocab_start:vocab_end]).abs().max().item()
    gw_scale = sw_ref.grad[vocab_start:vocab_end].abs().max().item()

    ret[f"opd2_gh_{rank}"] = gh_err
    ret[f"opd2_gh_scale_{rank}"] = gh_scale
    ret[f"opd2_gw_{rank}"] = gw_err
    ret[f"opd2_gw_scale_{rank}"] = gw_scale
    ret[f"opd2_report_loss_{rank}"] = float(reported_loss_sum.item())
    ret[f"opd2_ref_loss_{rank}"] = float(ref_loss.item())
    dist.barrier()
    dist.destroy_process_group()


def _worker_opd_streaming_vs_vp(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29606")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    counts = (7, 0, 13, 3)
    num_tokens = sum(counts)
    sh, th, sw, tw, labels = _full_inputs(num_tokens)
    token_weights = torch.linspace(0.7, 1.3, num_tokens)
    loss_clamp = 10.0

    token_start = sum(counts[:rank])
    token_end = token_start + counts[rank]
    vocab_start = rank * VLOCAL
    vocab_end = (rank + 1) * VLOCAL

    local_sh = sh[token_start:token_end].clone().requires_grad_(True)
    local_th = th[token_start:token_end].clone()
    sw_local = sw[vocab_start:vocab_end].clone().requires_grad_(True)
    tw_local = tw[vocab_start:vocab_end].clone()
    local_labels = labels[token_start:token_end].clone()
    local_token_weights = token_weights[token_start:token_end].clone()

    out = opd_vocab_parallel_loss_function(
        student_hidden_flat=local_sh,
        student_weight_local=sw_local,
        labels=local_labels,
        teacher_hidden_flat=local_th,
        teacher_weight_local=tw_local,
        teacher_weights=local_token_weights,
        loss_reducer=TokenPartial(scale=torch.tensor(1.0)),
        hidden_match_coef=0.0,
        loss_max_clamp=loss_clamp,
        teacher_lm_head_fp32=True,
        group=None,
    )
    out.loss.backward()
    reported_loss_sum = out.loss.detach().clone()
    dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)

    sh_ref = sh.clone().requires_grad_(True)
    sw_ref = sw.clone().requires_grad_(True)
    streaming_kl = streaming_reverse_kl_function(
        sh_ref,
        sw_ref,
        th,
        tw,
        labels,
        ignore_index=IGNORE,
        vocab_chunk_size=113,
    ).clamp(min=-loss_clamp, max=loss_clamp)
    ref_loss = (streaming_kl * token_weights).sum()
    ref_loss.backward()

    if local_sh.numel():
        gh_err = (local_sh.grad - sh_ref.grad[token_start:token_end]).abs().max().item()
        gh_scale = sh_ref.grad[token_start:token_end].abs().max().item()
    else:
        gh_err = 0.0
        gh_scale = 1.0
    gw_err = (sw_local.grad - sw_ref.grad[vocab_start:vocab_end]).abs().max().item()
    gw_scale = sw_ref.grad[vocab_start:vocab_end].abs().max().item()

    ret[f"stream_vp_gh_{rank}"] = gh_err
    ret[f"stream_vp_gh_scale_{rank}"] = gh_scale
    ret[f"stream_vp_gw_{rank}"] = gw_err
    ret[f"stream_vp_gw_scale_{rank}"] = gw_scale
    ret[f"stream_vp_report_loss_{rank}"] = float(reported_loss_sum.item())
    ret[f"stream_vp_ref_loss_{rank}"] = float(ref_loss.item())
    ret[f"stream_vp_kl_mean_{rank}"] = float(out.metrics["opd_kl"])
    ret[f"stream_vp_ref_kl_mean_{rank}"] = float(streaming_kl.mean().item())
    dist.barrier()
    dist.destroy_process_group()


def _worker_dtensor_shard(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29604")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = DeviceMesh("cpu", torch.arange(world))
    full_weight = torch.arange(23 * 5, dtype=torch.float32).reshape(23, 5)
    weight = distribute_tensor(full_weight, mesh, [Shard(0)])

    local_weight, start, end, group = ModelRunner._opd_student_vocab_shard(weight)

    torch.testing.assert_close(local_weight, full_weight[start:end])
    ret[f"dtensor_rows_{rank}"] = (int(start), int(end), int(local_weight.shape[0]), dist.get_world_size(group))
    dist.barrier()
    dist.destroy_process_group()


def _worker_uneven(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29602")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    sh, th, sw, tw, labels = _full_inputs(sum(UNEVEN_COUNTS))
    token_start = sum(UNEVEN_COUNTS[:rank])
    token_end = token_start + UNEVEN_COUNTS[rank]
    vocab_start = rank * VLOCAL
    vocab_end = (rank + 1) * VLOCAL

    local_sh = sh[token_start:token_end].clone().requires_grad_(True)
    local_th = th[token_start:token_end].clone()
    sw_local = sw[vocab_start:vocab_end].clone().requires_grad_(True)
    tw_local = tw[vocab_start:vocab_end].clone()
    local_labels = labels[token_start:token_end].clone()

    kl = vocab_parallel_reverse_kl_gathered(
        local_student_hidden=local_sh,
        student_weight_local=sw_local,
        local_teacher_hidden=local_th,
        teacher_weight_local=tw_local,
        local_labels=local_labels,
        ignore_index=IGNORE,
        group=None,
    )
    kl.sum().backward()

    gh_ref, gw_ref = reference(sh, sw, th, tw, labels)
    if local_sh.numel():
        gh_err = (local_sh.grad - gh_ref[token_start:token_end]).abs().max().item()
        gh_scale = gh_ref[token_start:token_end].abs().max().item()
    else:
        gh_err = 0.0
        gh_scale = 1.0
    gw_err = (sw_local.grad - gw_ref[vocab_start:vocab_end]).abs().max().item()
    gw_scale = gw_ref[vocab_start:vocab_end].abs().max().item()
    ret[f"gh_{rank}"] = gh_err
    ret[f"gh_scale_{rank}"] = gh_scale
    ret[f"gw_{rank}"] = gw_err
    ret[f"gw_scale_{rank}"] = gw_scale
    ret[f"kl_rows_{rank}"] = int(kl.shape[0])
    dist.barrier()
    dist.destroy_process_group()


def main():
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(WORLD, ret), nprocs=WORLD, join=True)
    gh_rel = ret["gh"] / max(ret["gh_scale"], 1e-30)
    gw_rel = ret["gw"] / max(ret["gw_scale"], 1e-30)
    print(f"gather-activations VP-KL integration (world={WORLD}, N={N}, V={V}, H={H}):")
    print(f"  local hidden grad  max|abs|={ret['gh']:.3e}  rel={gh_rel:.3e}  (slice of full ref)")
    print(f"  weight shard grad  max|abs|={ret['gw']:.3e}  rel={gw_rel:.3e}")
    ok = gh_rel < 1e-4 and gw_rel < 1e-4
    print(f"  => {'PASS — local-slice grad matches ref, no double-count' if ok else 'FAIL'}")

    ret_uneven = mgr.dict()
    mp.spawn(_worker_uneven, args=(WORLD, ret_uneven), nprocs=WORLD, join=True)
    gh_rel_uneven = max(ret_uneven[f"gh_{rank}"] / max(ret_uneven[f"gh_scale_{rank}"], 1e-30) for rank in range(WORLD))
    gw_rel_uneven = max(ret_uneven[f"gw_{rank}"] / max(ret_uneven[f"gw_scale_{rank}"], 1e-30) for rank in range(WORLD))
    kl_rows = {ret_uneven[f"kl_rows_{rank}"] for rank in range(WORLD)}
    expected_rows = sum(UNEVEN_COUNTS)
    uneven_ok = gh_rel_uneven < 1e-4 and gw_rel_uneven < 1e-4 and kl_rows == {expected_rows}
    print(f"uneven-token gathered VP-KL (counts={UNEVEN_COUNTS}, total={expected_rows}):")
    print(f"  local hidden grad max rel={gh_rel_uneven:.3e}")
    print(f"  weight shard grad max rel={gw_rel_uneven:.3e}")
    print(f"  gathered KL rows per rank={sorted(kl_rows)}")
    print(f"  => {'PASS — padded gather handles uneven/zero-token ranks' if uneven_ok else 'FAIL'}")

    ret_opd = mgr.dict()
    mp.spawn(_worker_opd, args=(WORLD, ret_opd), nprocs=WORLD, join=True)
    opd_gh_rel = max(ret_opd[f"opd_gh_{rank}"] / max(ret_opd[f"opd_gh_scale_{rank}"], 1e-30) for rank in range(WORLD))
    opd_gw_rel = max(ret_opd[f"opd_gw_{rank}"] / max(ret_opd[f"opd_gw_scale_{rank}"], 1e-30) for rank in range(WORLD))
    report_loss_err = max(
        abs(ret_opd[f"opd_report_loss_{rank}"] - ret_opd[f"opd_ref_loss_{rank}"]) for rank in range(WORLD)
    )
    kl_metric_err = max(
        abs(ret_opd[f"opd_kl_mean_{rank}"] - ret_opd[f"opd_ref_kl_mean_{rank}"]) for rank in range(WORLD)
    )
    weighted_metric_err = max(
        abs(ret_opd[f"opd_weighted_kl_mean_{rank}"] - ret_opd[f"opd_ref_weighted_kl_mean_{rank}"])
        for rank in range(WORLD)
    )
    vp_diag_rows = {ret_opd[f"opd_vp_group_tokens_{rank}"] for rank in range(WORLD)}
    vp_diag_kl_sum_err = max(
        abs(ret_opd[f"opd_vp_kl_sum_{rank}"] - ret_opd[f"opd_ref_kl_sum_{rank}"]) for rank in range(WORLD)
    )
    vp_diag_weighted_kl_sum_err = max(
        abs(ret_opd[f"opd_vp_weighted_kl_sum_{rank}"] - ret_opd[f"opd_ref_weighted_kl_sum_{rank}"])
        for rank in range(WORLD)
    )
    debug_counts = [ret_opd[f"opd_debug_count_{rank}"] for rank in range(WORLD)]
    debug_kl_err = max(ret_opd[f"opd_debug_kl_err_{rank}"] for rank in range(WORLD))
    debug_weighted_kl_err = max(ret_opd[f"opd_debug_weighted_kl_err_{rank}"] for rank in range(WORLD))
    debug_weight_err = max(ret_opd[f"opd_debug_weight_err_{rank}"] for rank in range(WORLD))
    opd_ok = (
        opd_gh_rel < 1e-4
        and opd_gw_rel < 1e-4
        and report_loss_err < 1e-5
        and kl_metric_err < 1e-6
        and weighted_metric_err < 1e-6
        and vp_diag_rows == {sum(UNEVEN_COUNTS)}
        and vp_diag_kl_sum_err < 1e-5
        and vp_diag_weighted_kl_sum_err < 1e-5
        and tuple(debug_counts) == UNEVEN_COUNTS
        and debug_kl_err < 1e-6
        and debug_weighted_kl_err < 1e-6
        and debug_weight_err < 1e-6
    )
    print("vocab-parallel OPD loss helper (weighted KL + local hidden MSE):")
    print(f"  local hidden grad max rel={opd_gh_rel:.3e}")
    print(f"  weight shard grad max rel={opd_gw_rel:.3e}")
    print(f"  all-reduced reported loss abs err={report_loss_err:.3e}")
    print(f"  KL metric abs err={kl_metric_err:.3e}; weighted KL metric abs err={weighted_metric_err:.3e}")
    print(
        "  debug numerator rows="
        f"{sorted(vp_diag_rows)} KL sum err={vp_diag_kl_sum_err:.3e} "
        f"weighted KL sum err={vp_diag_weighted_kl_sum_err:.3e}"
    )
    print(
        "  local debug vectors "
        f"counts={debug_counts} KL err={debug_kl_err:.3e} "
        f"weighted KL err={debug_weighted_kl_err:.3e} weight err={debug_weight_err:.3e}"
    )
    print(f"  => {'PASS' if opd_ok else 'FAIL'}")

    ret_opd2 = mgr.dict()
    mp.spawn(_worker_opd_two_lm_tp_groups, args=(WORLD, ret_opd2), nprocs=WORLD, join=True)
    opd2_gh_rel = max(
        ret_opd2[f"opd2_gh_{rank}"] / max(ret_opd2[f"opd2_gh_scale_{rank}"], 1e-30) for rank in range(WORLD)
    )
    opd2_gw_rel = max(
        ret_opd2[f"opd2_gw_{rank}"] / max(ret_opd2[f"opd2_gw_scale_{rank}"], 1e-30) for rank in range(WORLD)
    )
    opd2_report_loss_err = max(
        abs(ret_opd2[f"opd2_report_loss_{rank}"] - ret_opd2[f"opd2_ref_loss_{rank}"]) for rank in range(WORLD)
    )
    opd2_ok = opd2_gh_rel < 1e-4 and opd2_gw_rel < 1e-4 and opd2_report_loss_err < 1e-5
    print("vocab-parallel OPD loss helper (two DP-sourced lm-head TP groups):")
    print(f"  local hidden grad max rel={opd2_gh_rel:.3e}")
    print(f"  replica-summed weight shard grad max rel={opd2_gw_rel:.3e}")
    print(f"  world-reduced reported loss abs err={opd2_report_loss_err:.3e}")
    print(f"  => {'PASS' if opd2_ok else 'FAIL'}")

    ret_stream_vp = mgr.dict()
    mp.spawn(_worker_opd_streaming_vs_vp, args=(WORLD, ret_stream_vp), nprocs=WORLD, join=True)
    stream_vp_gh_rel = max(
        ret_stream_vp[f"stream_vp_gh_{rank}"] / max(ret_stream_vp[f"stream_vp_gh_scale_{rank}"], 1e-30)
        for rank in range(WORLD)
    )
    stream_vp_gw_rel = max(
        ret_stream_vp[f"stream_vp_gw_{rank}"] / max(ret_stream_vp[f"stream_vp_gw_scale_{rank}"], 1e-30)
        for rank in range(WORLD)
    )
    stream_vp_report_loss_err = max(
        abs(ret_stream_vp[f"stream_vp_report_loss_{rank}"] - ret_stream_vp[f"stream_vp_ref_loss_{rank}"])
        for rank in range(WORLD)
    )
    stream_vp_kl_err = max(
        abs(ret_stream_vp[f"stream_vp_kl_mean_{rank}"] - ret_stream_vp[f"stream_vp_ref_kl_mean_{rank}"])
        for rank in range(WORLD)
    )
    stream_vp_ok = (
        stream_vp_gh_rel < 1e-4
        and stream_vp_gw_rel < 1e-4
        and stream_vp_report_loss_err < 1e-5
        and stream_vp_kl_err < 1e-6
    )
    print("vocab-parallel OPD vs streaming full-vocab OPD:")
    print(f"  local hidden grad max rel={stream_vp_gh_rel:.3e}")
    print(f"  weight shard grad max rel={stream_vp_gw_rel:.3e}")
    print(f"  world-reduced reported loss abs err={stream_vp_report_loss_err:.3e}")
    print(f"  KL metric abs err={stream_vp_kl_err:.3e}")
    print(f"  => {'PASS' if stream_vp_ok else 'FAIL'}")

    ret_dtensor = mgr.dict()
    mp.spawn(_worker_dtensor_shard, args=(WORLD, ret_dtensor), nprocs=WORLD, join=True)
    rows = [ret_dtensor[f"dtensor_rows_{rank}"] for rank in range(WORLD)]
    dtensor_ok = rows == [(0, 6, 6, WORLD), (6, 12, 6, WORLD), (12, 18, 6, WORLD), (18, 23, 5, WORLD)]
    print("ModelRunner DTensor vocab-shard helper:")
    print(f"  row ranges={rows}")
    print(f"  => {'PASS' if dtensor_ok else 'FAIL'}")
    return 0 if ok and uneven_ok and opd_ok and opd2_ok and stream_vp_ok and dtensor_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
