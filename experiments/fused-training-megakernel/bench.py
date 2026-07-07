"""Benchmark: fused megakernel vs eager / torch.compile / compile+CUDAGraph fwd+bwd.

All variants run the same bf16 Qwen3-architecture model, same shapes, and time one
training fwd+bwd (including gradient zeroing). CUDA-event timing, median of 50.

Run: CUDA_VISIBLE_DEVICES=<idle> <fa4-venv>/bin/python bench.py
"""

import torch
import torch.nn.functional as F
from model import Cfg, MKQwen3, rope_tables


class TorchQwen3(torch.nn.Module):
    """bf16 eager twin of the megakernel model (same math, torch ops)."""

    def __init__(self, cfg: Cfg, params):
        super().__init__()
        self.cfg = cfg
        self.p = torch.nn.ParameterDict({k.replace(".", "_"): torch.nn.Parameter(v.clone()) for k, v in params.items()})
        cos, sin = rope_tables(cfg, "cuda")
        self.register_buffer("cos", cos.to(torch.bfloat16))
        self.register_buffer("sin", sin.to(torch.bfloat16))

    def forward(self, tokens, labels):
        c = self.cfg
        P = self.p

        def rms(x, w):
            return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + c.eps).to(x.dtype)) * w

        def rope(t):
            a, b = t[..., : c.D // 2], t[..., c.D // 2 :]
            cc, ss = self.cos[:, None, :], self.sin[:, None, :]
            return torch.cat([a * cc - b * ss, b * cc + a * ss], dim=-1)

        x = P["emb"][tokens.long()]
        for l in range(c.L):
            xn = rms(x, P[f"w1_{l}"])
            qkv = (xn @ P[f"wqkv_{l}"].T).view(c.S, c.nq + 2 * c.nkv, c.D)
            q, k, v = qkv[:, : c.nq], qkv[:, c.nq : c.nq + c.nkv], qkv[:, c.nq + c.nkv :]
            # 4-D [1,H,S,D] + enable_gqa: flash-eligible SDPA. The original 3-D call
            # SILENTLY math-decomposed (materialized S x S softmax + tf32 gemms) at
            # every S — flash requires 4-D inputs — making every baseline number in
            # the v3 program soft (nano 711 -> 633, small 2733 -> 1910 when fixed)
            # and manufacturing a fake long-S crossover (see NOTES P4b retraction).
            q = rope(rms(q, P[f"qn_{l}"])).permute(1, 0, 2).unsqueeze(0)
            k = rope(rms(k, P[f"kn_{l}"])).permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + o.squeeze(0).permute(1, 0, 2).reshape(c.S, c.nq * c.D) @ P[f"wo_{l}"].T
            g, u = (rms(x, P[f"w2_{l}"]) @ P[f"wgu_{l}"].T).chunk(2, dim=-1)
            x = x + (F.silu(g) * u) @ P[f"wd_{l}"].T
        logits = rms(x, P["wf"]) @ P["wlm"].T
        return F.cross_entropy(logits.float(), labels.long(), ignore_index=-100)


def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_cfg(cfg: Cfg):
    print(f"=== {cfg} ===")
    mk_model = MKQwen3(cfg, seed=0)
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.V, (cfg.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100

    results = {}
    results["megakernel"] = time_fn(lambda: mk_model.step(tokens, labels))

    # additive row: same kernel launched by CUDA-graph replay, symmetric with the
    # graphed baselines below (host launch + pybind + Python step() off the meter).
    # The primary "megakernel" row is unchanged; scoreboard adoption is a separate
    # explicit decision (measurement-honesty rule 7).
    try:
        results["megakernel+graph"] = time_fn(mk_model.make_graphed_step(tokens, labels))
    except Exception as e:  # older drivers may refuse cooperative-launch capture
        print(f"  megakernel+graph skipped: {e}")

    tm = TorchQwen3(cfg, mk_model.params).cuda()

    def eager_step():
        for p in tm.parameters():
            if p.grad is not None:
                p.grad = None
        tm(tokens, labels).backward()

    results["eager"] = time_fn(eager_step)

    tmc = torch.compile(TorchQwen3(cfg, mk_model.params).cuda())

    def compiled_step():
        for p in tmc.parameters():
            if p.grad is not None:
                p.grad = None
        tmc(tokens, labels).backward()

    results["compile"] = time_fn(compiled_step, warmup=15)

    # compile + manual whole-step CUDA graph (the enable_cudagraph_step analogue)
    tmg = torch.compile(TorchQwen3(cfg, mk_model.params).cuda())
    for _ in range(5):  # materialize grads + settle compile
        for p in tmg.parameters():
            p.grad = None
        tmg(tokens, labels).backward()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for p in tmg.parameters():
                p.grad.zero_()
            tmg(tokens, labels).backward()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for p in tmg.parameters():
            p.grad.zero_()
        tmg(tokens, labels).backward()
    results["compile+cudagraph"] = time_fn(graph.replay)

    # hardened baseline (the honest goalpost): foreach grad zeroing instead of one
    # zero_() node per param, max-autotune inductor kernels, same manual whole-step graph
    tmh = torch.compile(TorchQwen3(cfg, mk_model.params).cuda(), mode="max-autotune-no-cudagraphs")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # ALL warmup off the default stream: a default-stream
        for _ in range(5):  # backward leaves AccumulateGrad stream refs that break capture
            for p in tmh.parameters():
                p.grad = None
            tmh(tokens, labels).backward()
        hgrads = [p.grad for p in tmh.parameters()]
        for _ in range(3):
            torch._foreach_zero_(hgrads)
            tmh(tokens, labels).backward()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    hgraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(hgraph):
        torch._foreach_zero_(hgrads)
        tmh(tokens, labels).backward()
    results["compile+cudagraph+"] = time_fn(hgraph.replay)

    base = results["eager"]
    for name, ms in results.items():
        print(f"  {name:20s} {ms * 1e3:9.1f} us/step   {base / ms:5.2f}x vs eager")
    return results


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    torch.cuda.set_device(0)
    if which in ("nano", "both"):
        bench_cfg(Cfg())  # nano: H256 L4 S512
    if which in ("small", "both"):
        bench_cfg(Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024))  # small
