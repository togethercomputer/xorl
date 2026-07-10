"""Scheduler hop-cost microbenchmarks for the dataflow executor.

Builds strictly serial instruction chains (each link uses DISTINCT buffers so the
dependency builder emits exactly one RAW edge per hop — no transitive-history fan-out)
and measures the per-hop cost:

  axpy chain     -> c_hop        pure scheduler hop (publish -> discover -> claim) + tiny op
  wmma gemm      -> c_gemm_hop   + WMMA prologue/epilogue/math (64x128x128 NT)
  wgmma gemm     -> c_wgmma_hop  + wgmma pipeline prologue/epilogue (128x64x64 NT)
  rmsnorm        -> c_row_hop    row-op with block reduction (128 rows x H64)
  gemm+rmsnorm   -> alternating wgmma/rowop, the real per-layer chain texture

Decision rules (plan Phase 0): c_hop > 5us -> fix ring signaling before any rewrite;
c_gemm_hop - c_hop = what warp-spec prologue/epilogue overlap can reclaim per hop.

Run: CUDA_VISIBLE_DEVICES=<idle> .venv-fa4/bin/python hop_bench.py
"""

import mk
import torch


def time_prog(prog, ext, iters=20, warmup=5):
    for _ in range(warmup):
        prog.run(ext)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        prog.run(ext)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]  # ms


def gap_span(prog, ext, n):
    """Median scheduling gap and span per hop from the iclk stamps."""
    iclk = torch.zeros(2 * n, dtype=torch.int64, device="cuda")
    prog.run(ext, wave_clk=iclk)
    torch.cuda.synchronize()
    clk = iclk.cpu()
    starts, ends = clk[0::2], clk[1::2]
    gaps = (starts[1:] - ends[:-1]).float()  # chain order == instruction order here
    spans = (ends - starts).float()
    return gaps.median().item() / 1e3, spans.median().item() / 1e3


def report(name, prog, ext, n):
    ms = time_prog(prog, ext)
    hop_us = ms * 1e3 / (n - 1)
    gap, span = gap_span(prog, ext, n)
    print(
        f"{name:16s} {n:4d} links: {ms * 1e3:9.1f} us total  "
        f"{hop_us:6.2f} us/hop  (median gap {gap:5.2f} + span {span:5.2f})"
    )
    return hop_us


def axpy_chain(n=400, elems=256):
    p = mk.Program()
    bufs = [torch.zeros(elems, device="cuda", dtype=torch.float32) for _ in range(n)]
    ids = [p.buf(t) for t in bufs]
    p.instr(mk.OP_FILL_F32, 1, [ids[0], elems, mk.f2i(1.0)])
    for i in range(1, n):
        p.instr(mk.OP_AXPY_F32, 1, [ids[i], ids[i - 1], elems, mk.f2i(1.0)])
    return p.finalize(), n


def gemm_chain(n=256, wgmma=False):
    p = mk.Program()
    if wgmma:
        M, N, K, flags = 128, 64, 64, 2 | 128
        tiles = mk.gemm_tiles_wgmma(M, N)
    else:
        M, N, K, flags = 64, 128, 128, 2
        tiles = mk.gemm_tiles(M, N)
    assert N == K  # output feeds the next link's A
    a0 = torch.randn(M, K, device="cuda").to(torch.bfloat16) * 0.05
    b = torch.randn(N, K, device="cuda").to(torch.bfloat16) * 0.05
    cs = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    ia, ib = p.buf(a0), p.buf(b)
    ics = [p.buf(t) for t in cs]
    prev = ia
    for i in range(n):
        p.instr(mk.OP_GEMM, tiles, [prev, ib, ics[i], M, N, K, flags, 0])
        prev = ics[i]
    return p.finalize(), n


def rmsnorm_chain(n=256, S=128, H=64):
    p = mk.Program()
    x0 = torch.randn(S, H, device="cuda").to(torch.bfloat16)
    w = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    xs = [torch.empty(S, H, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    rs = [torch.empty(S, device="cuda", dtype=torch.float32) for _ in range(n)]
    iw = p.buf(w)
    prev = p.buf(x0)
    eps = mk.f2i(1e-6)
    for i in range(n):
        cur, ir = p.buf(xs[i]), p.buf(rs[i])
        p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(S), [prev, iw, cur, ir, H, eps, S])
        prev = cur
    return p.finalize(), n


def layer_chain(n=128, S=128, H=64):
    """gemm(wgmma NT SxHxH) -> rmsnorm(S rows) alternating: the per-layer chain texture."""
    p = mk.Program()
    w = torch.randn(H, H, device="cuda").to(torch.bfloat16) * 0.05
    wn = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    x0 = torch.randn(S, H, device="cuda").to(torch.bfloat16)
    iw, iwn = p.buf(w), p.buf(wn)
    prev = p.buf(x0)
    eps = mk.f2i(1e-6)
    tiles = mk.gemm_tiles_wgmma(S, H)
    cnt = 0
    for i in range(n):
        y = p.buf(torch.empty(S, H, device="cuda", dtype=torch.bfloat16))
        p.instr(mk.OP_GEMM, tiles, [prev, iw, y, S, H, H, 2 | 128, 0])
        cnt += 1
        z = p.buf(torch.empty(S, H, device="cuda", dtype=torch.bfloat16))
        ir = p.buf(torch.empty(S, device="cuda", dtype=torch.float32))
        p.instr(mk.OP_RMSNORM_FWD, mk.rowop_tiles(S), [y, iwn, z, ir, H, eps, S])
        cnt += 1
        prev = z
    return p.finalize(), cnt


if __name__ == "__main__":
    torch.cuda.set_device(0)
    ext = mk.load_ext()
    print(f"nblocks={ext.nblocks()}")
    prog, n = axpy_chain()
    c_hop = report("axpy (c_hop)", prog, ext, n)
    prog, n = gemm_chain(wgmma=False)
    c_gemm = report("wmma gemm", prog, ext, n)
    prog, n = gemm_chain(wgmma=True)
    c_wg = report("wgmma gemm", prog, ext, n)
    prog, n = rmsnorm_chain()
    report("rmsnorm row-op", prog, ext, n)
    prog, n = layer_chain()
    report("gemm+rmsnorm", prog, ext, n)
    print(f"\nc_hop={c_hop:.2f} us  (>5 -> fix ring signaling first)")
    print(f"warp-spec reclaimable per gemm hop: wmma {c_gemm - c_hop:.2f} us, wgmma {c_wg - c_hop:.2f} us")
