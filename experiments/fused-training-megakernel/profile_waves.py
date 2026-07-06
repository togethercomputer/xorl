"""Per-wave clock64 attribution of megakernel step time, grouped by op type."""

import collections

import mk
import torch
from model import Cfg, MKQwen3


OPNAMES = {v: k for k, v in vars(mk).items() if k.startswith("OP_")}


def profile(cfg):
    m = MKQwen3(cfg, seed=0)
    tokens = torch.randint(0, cfg.V, (cfg.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100
    m.tokens.copy_(tokens)
    m.labels.copy_(labels)
    m.inv_valid.fill_(1.0 / (cfg.S - 1))
    clk = torch.zeros(m.n_waves + 1, device="cuda", dtype=torch.int64)
    for _ in range(3):
        m.prog.run(m.ext, smem_bytes=getattr(m, "_smem_bytes", None), wave_clk=clk)
    torch.cuda.synchronize()
    d = clk.diff().cpu()
    per_op = collections.Counter()
    for w, wave in enumerate(m.prog.waves):
        key = "+".join(sorted({OPNAMES[op] for op, _, _ in wave}))
        per_op[key] += d[w].item()
    total = sum(per_op.values())
    ghz = 1.98e9
    print(f"cfg {cfg}\ntotal ~{total / ghz * 1e3:.2f} ms across {m.n_waves} waves")
    for k, v in per_op.most_common():
        print(f"  {k:40s} {v / ghz * 1e6:9.0f} us  {100 * v / total:5.1f}%")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    profile(Cfg())
    profile(Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024))
