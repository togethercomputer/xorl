#!/usr/bin/env python3
"""Multi-process correctness test for vocab_parallel_reverse_kl_function.

Shards the lm_head along the vocab dim across `world` gloo/CPU ranks and checks
the vocab-parallel reverse-KL (kl, grad_hidden, grad_weight) against a
single-process full-vocab brute-force reference. Gradient-identity is the bar.

Run (no GPU needed):
  python -m pytest tests/ops/loss/test_vocab_parallel_reverse_kl.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


WORLD = 4
N = 48  # tokens
H = 96  # hidden
V = 4 * 130  # vocab (divisible by WORLD; 520)
IGNORE = -100


# The kernel computes in float32 (the real lm_head_fp32 path), so compare
# against a float32 reference; residual is float32 summation-order noise.
def _full_inputs():
    torch.manual_seed(1234)
    sh = torch.randn(N, H, dtype=torch.float32) * 0.3
    th = torch.randn(N, H, dtype=torch.float32) * 0.3
    sw = torch.randn(V, H, dtype=torch.float32) * 0.05
    tw = torch.randn(V, H, dtype=torch.float32) * 0.05
    labels = torch.randint(0, V, (N,))
    labels[: N // 6] = IGNORE
    return sh, th, sw, tw, labels


def reference(sh, sw, th, tw, labels):
    sh = sh.clone().requires_grad_(True)
    sw = sw.clone().requires_grad_(True)
    s_logits = sh @ sw.t()
    t_logits = th @ tw.t()
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    valid = (labels != IGNORE).to(kl.dtype)
    kl = kl * valid
    kl.sum().backward()
    return kl.detach(), sh.grad.detach(), sw.grad.detach()


def _worker(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from xorl.ops.loss.vocab_parallel_reverse_kl import vocab_parallel_reverse_kl_function  # noqa: PLC0415

    sh, th, sw, tw, labels = _full_inputs()
    shard = V // world
    lo, hi = rank * shard, (rank + 1) * shard

    sh_v = sh.clone().requires_grad_(True)
    sw_local = sw[lo:hi].clone().requires_grad_(True)
    tw_local = tw[lo:hi].clone()

    kl = vocab_parallel_reverse_kl_function(
        student_hidden_states=sh_v,
        student_weight_local=sw_local,
        teacher_hidden_states=th,
        teacher_weight_local=tw_local,
        labels=labels,
        ignore_index=IGNORE,
        group=None,
    )
    kl.sum().backward()

    if rank == 0:
        kl_ref, gh_ref, gw_ref = reference(sh, sw, th, tw, labels)
        # gather all ranks' local grad_weight shards
        gws = [torch.zeros_like(sw_local) for _ in range(world)]
        dist.all_gather(gws, sw_local.grad.contiguous())
        gw_full = torch.cat(gws, dim=0)
        ret["kl"] = (kl.detach() - kl_ref).abs().max().item()
        ret["gh"] = (sh_v.grad.detach() - gh_ref).abs().max().item()
        ret["gw"] = (gw_full - gw_ref).abs().max().item()
        ret["kl_scale"] = kl_ref.abs().max().item()
        ret["gh_scale"] = gh_ref.abs().max().item()
        ret["gw_scale"] = gw_ref.abs().max().item()
    else:
        dist.all_gather([torch.zeros_like(sw_local) for _ in range(world)], sw_local.grad.contiguous())
    dist.barrier()
    dist.destroy_process_group()


def main():
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(WORLD, ret), nprocs=WORLD, join=True)
    print(f"vocab-parallel reverse-KL vs full-vocab reference (world={WORLD}, V={V}, N={N}, H={H}):")
    kl_rel = ret["kl"] / max(ret["kl_scale"], 1e-30)
    gh_rel = ret["gh"] / max(ret["gh_scale"], 1e-30)
    gw_rel = ret["gw"] / max(ret["gw_scale"], 1e-30)
    print(f"  kl          max|abs|={ret['kl']:.3e}  rel={kl_rel:.3e}  (scale {ret['kl_scale']:.3e})")
    print(f"  grad_hidden max|abs|={ret['gh']:.3e}  rel={gh_rel:.3e}  (scale {ret['gh_scale']:.3e})")
    print(f"  grad_weight max|abs|={ret['gw']:.3e}  rel={gw_rel:.3e}  (scale {ret['gw_scale']:.3e})")
    # float32 summation-order tolerance.
    ok = kl_rel < 1e-3 and gh_rel < 1e-4 and gw_rel < 1e-4
    print(f"  => {'PASS (matches full-vocab reference to float32 precision)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
