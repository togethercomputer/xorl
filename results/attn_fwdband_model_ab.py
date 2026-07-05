"""In-model banded-chunk A/B (session 2853e0de).

argv: S T order(default_first|band_first)
Builds two MKQwen3 models at Cfg(S=S) — one with MK_ATTN_FWD_BAND unset (current
default) and one with MK_ATTN_FWD_BAND=T — in the given construction order, checks
loss/gradient parity between them, then times alternating paired steps.
Both models share one extension build (band decode is arg-driven, not a compile
flag), so there is no build-cache race and no cross-build noise.
"""

import os
import statistics
import sys
from pathlib import Path

import torch


EXPDIR = str(Path(__file__).resolve().parents[1] / "experiments/fused-training-megakernel")
sys.path.insert(0, EXPDIR)


def build_model(S, band):
    if band:
        os.environ["MK_ATTN_FWD_BAND"] = band
    else:
        os.environ.pop("MK_ATTN_FWD_BAND", None)
    from model import Cfg, MKQwen3
    return MKQwen3(Cfg(S=S), seed=0)


def main():
    S = int(sys.argv[1])
    T = sys.argv[2]
    order = sys.argv[3] if len(sys.argv) > 3 else "default_first"
    torch.cuda.set_device(0)

    torch.manual_seed(1)
    V = 8192
    tokens = torch.randint(0, V, (S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100

    if order == "default_first":
        m_def = build_model(S, None)
        m_band = build_model(S, T)
    else:
        m_band = build_model(S, T)
        m_def = build_model(S, None)

    # route check: instruction counts must differ (bands emit >2 attn-bwd instrs/layer)
    n_def = len(m_def.prog._instrs) if hasattr(m_def.prog, "_instrs") else -1
    n_band = len(m_band.prog._instrs) if hasattr(m_band.prog, "_instrs") else -1
    print(f"ROUTE S={S} T={T} instrs default={n_def} banded={n_band}", flush=True)

    # parity: one step each, same inputs
    m_def.step(tokens, labels)
    m_band.step(tokens, labels)
    torch.cuda.synchronize()
    l_def = float(m_def.loss.item())
    l_band = float(m_band.loss.item())
    worst = 0.0
    worst_name = ""
    for name in m_def.grads:
        ga, gb = m_def.grads[name].float(), m_band.grads[name].float()
        denom = ga.abs().max().item()
        if denom < 1e-8:
            continue
        rel = (ga - gb).abs().max().item() / denom
        if rel > worst:
            worst, worst_name = rel, name
    print(f"PARITY S={S} T={T} loss {l_def:.5f} vs {l_band:.5f} "
          f"worst_grad_rel {worst:.6f} ({worst_name})", flush=True)
    assert abs(l_def - l_band) < 5e-3 and worst < 0.03, "band parity failed"

    # paired alternating timing
    def time_one(m):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        m.step(tokens, labels)
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3

    for _ in range(8):
        m_def.step(tokens, labels)
        m_band.step(tokens, labels)
    torch.cuda.synchronize()
    reps = 40 if S <= 4096 else 16
    td, tb, wins = [], [], 0
    for _ in range(reps):
        a = time_one(m_def)
        b = time_one(m_band)
        td.append(a)
        tb.append(b)
        wins += b < a
    md, mb = statistics.median(td), statistics.median(tb)
    print(f"TIMING S={S} T={T} order={order} default {md:.2f}us banded {mb:.2f}us "
          f"delta {mb - md:+.2f}us wins {wins}/{reps}", flush=True)
    print("MODEL-AB DONE", flush=True)


if __name__ == "__main__":
    main()
