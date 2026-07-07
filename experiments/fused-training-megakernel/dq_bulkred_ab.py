"""DQ C>1 bulk-reduce drain A/B (session b1d36305, MK_ATTN_DQ_BULK_RED lane).

Copy of the results/env_ab_main.py pattern, defaulting MKAB_TREE to this
worktree. argv: shape order env1=val1[,env2=val2...] [reps]
Model A = current defaults, model B = defaults + env overrides. Route summary
(instr count), loss/grad parity, paired alternating timing.

MKAB_FORCE_MODE=<df|pdf|...> forces the same executor mode for BOTH models
(needed when the ptxas UBLKRED audit only clears one image; see
mk._audit_bulkred_sass).
"""
import os, statistics, sys
import torch

sys.path.insert(0, os.environ.get("MKAB_TREE", os.path.dirname(os.path.abspath(__file__))))

SHAPES = {
    "small": dict(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024),
    "nano": dict(),
    "deep": dict(L=12),
    "s128": dict(S=128),
    "s256": dict(S=256),
    "s1024": dict(S=1024),
    "s2048": dict(S=2048),
    "s3072": dict(S=3072),
    "s4096": dict(S=4096),
    "s8192": dict(S=8192),
}


def build(shape, envs):
    for k, v in envs:
        os.environ[k] = v
    from model import Cfg, MKQwen3
    m = MKQwen3(Cfg(**SHAPES[shape]), seed=0)
    for k, _ in envs:
        os.environ.pop(k, None)
    return m


def main():
    shape = sys.argv[1]
    order = sys.argv[2]
    envs = [tuple(kv.split("=")) for kv in sys.argv[3].split(",")]
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    S = SHAPES[shape].get("S", 512)
    V = SHAPES[shape].get("V", 8192)
    tokens = torch.randint(0, V, (S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100
    if order == "default_first":
        ma = build(shape, [])
        mb = build(shape, envs)
    else:
        mb = build(shape, envs)
        ma = build(shape, [])
    force_mode = os.environ.get("MKAB_FORCE_MODE")
    if force_mode:
        ma.default_mode = force_mode
        mb.default_mode = force_mode
        print(f"FORCED MODE {force_mode}", flush=True)
    na = sum(len(w) for w in ma.prog.waves)
    nb = sum(len(w) for w in mb.prog.waves)
    print(f"ROUTE {shape} n_instr default={na} variant={nb}", flush=True)
    ma.step(tokens, labels); mb.step(tokens, labels); torch.cuda.synchronize()
    la, lb = float(ma.loss.item()), float(mb.loss.item())
    worst, wn = 0.0, ""
    for n in ma.grads:
        ga, gb = ma.grads[n].float(), mb.grads[n].float()
        d = ga.abs().max().item()
        if d < 1e-8:
            continue
        r = (ga - gb).abs().max().item() / d
        if r > worst:
            worst, wn = r, n
    print(f"PARITY {shape} {sys.argv[3]} loss {la:.5f} vs {lb:.5f} worst_grad_rel {worst:.6f} ({wn})", flush=True)
    assert abs(la - lb) < 5e-3 and worst < 0.03

    def t1(m):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); m.step(tokens, labels); e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3

    for _ in range(8):
        ma.step(tokens, labels); mb.step(tokens, labels)
    torch.cuda.synchronize()
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else (40 if S <= 4096 else 16)
    ta, tb, w = [], [], 0
    for _ in range(reps):
        a = t1(ma); b = t1(mb); ta.append(a); tb.append(b); w += b < a
    print(f"TIMING {shape} {sys.argv[3]} order={order} default {statistics.median(ta):.2f}us "
          f"variant {statistics.median(tb):.2f}us delta {statistics.median(tb)-statistics.median(ta):+.2f}us "
          f"wins {w}/{reps}", flush=True)


if __name__ == "__main__":
    main()
