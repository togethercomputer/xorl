"""attn-pdf-feed Phase 1 layout/handshake proof (session b1d36305).

Runs OP_ATTN_DQ_WG / OP_ATTN_DKV_WG standalone on the pdf executor with the
WG2 cp.async producer feed ON vs OFF and compares the fp32 workspace:

  - dq C=1 (direct-store epilogue, single writer per q row): BITWISE required.
  - dkv G=1 C=1 (one atomicAdd per address on a zeroed ws): BITWISE required.
  - dq C=4 and dkv G=2 (s8192 production tiling; fp32 atomic order varies):
    feed-on vs feed-off allclose at 1e-4 rel, plus feed-on vs the C=1 golden.
  - df-mode run of the feed-on extension (active=0 fallback): BITWISE vs
    the feed-off pdf run (consumer path must be untouched when dormant).

Extensions are built with the s8192 production attention body flags
(exp2 approx+prebias, dq rs-feed, dq fp32-P, dq float2 store, dkv float2
atomic) so the feed is proven against the shipped dq/dkv bodies.

Usage: CUDA_VISIBLE_DEVICES=<idle> TORCH_EXTENSIONS_DIR=<dir> python
       results/attn_pdf_feed_probe_b1d36305.py <worktree mk dir>
"""

import os
import sys


MKDIR = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(__file__), "..", "experiments", "fused-training-megakernel")
)
sys.path.insert(0, os.path.abspath(MKDIR))

import mk  # noqa: E402
import torch  # noqa: E402


torch.manual_seed(0)
DEV = "cuda"

BODY = dict(
    attn_exp2_approx=1,
    attn_exp2_prebias=1,
    attn_dq_rs_feed=1,
    attn_dq_fp32_p=1,
    attn_dq_float2_store=1,
    attn_dkv_float2_atomic=1,
)

print("building feed-off pdf ext ...", flush=True)
EXT_OFF = mk.load_ext(pdf_producer=1, **BODY)
print("building feed-on (=2) ext ...", flush=True)
EXT_ON = mk.load_ext(pdf_producer=1, attn_pdf_feed=2, **BODY)


def gen_inputs(S, nq, nkv, D=64):
    stride = (nq + 2 * nkv) * D
    scale = mk.f2i(D**-0.5)
    qkv = (torch.randn(S, stride, device=DEV) * 0.5).to(torch.bfloat16)
    O = torch.empty(S, nq * D, device=DEV, dtype=torch.bfloat16)
    LSE = torch.empty(nq, S, device=DEV, dtype=torch.float32)
    dO = (torch.randn(S, nq * D, device=DEV) * 0.5).to(torch.bfloat16)
    Drow = torch.empty(nq, S, device=DEV, dtype=torch.float32)
    p = mk.Program()
    p.instr(mk.OP_ATTN_FWD_WG, nq * (S // 128), [p.buf(qkv), p.buf(O), p.buf(LSE), S, nq, nkv, D, scale])
    p.finalize().run(EXT_OFF)
    torch.cuda.synchronize()
    p = mk.Program()
    p.instr(mk.OP_ATTN_DPRE, S, [p.buf(dO), p.buf(O), p.buf(Drow), S, nq, D])
    p.finalize().run(EXT_OFF)
    torch.cuda.synchronize()
    return qkv, dO, LSE, Drow, stride, scale


def run_op(ext, mode, op, ntiles, qkv, dO, LSE, Drow, S, nq, nkv, scale, C, stride):
    ws = torch.zeros(S, stride, device=DEV, dtype=torch.float32)
    p = mk.Program()
    p.instr(op, ntiles, [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(ws), S, nq, nkv, 64, scale, C])
    p.finalize().run(ext, mode=mode)
    torch.cuda.synchronize()
    return ws


def bitwise(name, a, b):
    same = torch.equal(a.view(torch.int32), b.view(torch.int32))
    print(f"  {name:44s} {'BITWISE-OK' if same else 'MISMATCH'}", flush=True)
    if not same:
        d = (a - b).abs()
        nz = (a.view(torch.int32) != b.view(torch.int32)).sum().item()
        print(f"    diff words={nz} max_abs={d.max().item():.3e}")
    return same


def close(name, a, b, tol):
    ref = b.abs().max().item() + 1e-8
    err = (a - b).abs().max().item() / ref
    ok = err <= tol
    print(f"  {name:44s} rel_err={err:.2e} {'OK' if ok else 'FAIL'}", flush=True)
    return ok


fails = 0

# ---- s8192 shape: dq --------------------------------------------------------
S, nq, nkv = 8192, 4, 2
qkv, dO, LSE, Drow, stride, scale = gen_inputs(S, nq, nkv)
nqt = S // 128
print(f"dq S={S} nq={nq} nkv={nkv}:", flush=True)

dq_off_c1 = run_op(EXT_OFF, "pdf", mk.OP_ATTN_DQ_WG, nq * nqt * 1, qkv, dO, LSE, Drow, S, nq, nkv, scale, 1, stride)
dq_on_c1 = run_op(EXT_ON, "pdf", mk.OP_ATTN_DQ_WG, nq * nqt * 1, qkv, dO, LSE, Drow, S, nq, nkv, scale, 1, stride)
fails += not bitwise("dq C=1 pdf feed-on vs feed-off", dq_on_c1, dq_off_c1)

dq_on_df = run_op(EXT_ON, "df", mk.OP_ATTN_DQ_WG, nq * nqt * 1, qkv, dO, LSE, Drow, S, nq, nkv, scale, 1, stride)
fails += not bitwise("dq C=1 df feed-on(dormant) vs pdf feed-off", dq_on_df, dq_off_c1)

dq_off_c4 = run_op(EXT_OFF, "pdf", mk.OP_ATTN_DQ_WG, nq * nqt * 4, qkv, dO, LSE, Drow, S, nq, nkv, scale, 4, stride)
dq_on_c4 = run_op(EXT_ON, "pdf", mk.OP_ATTN_DQ_WG, nq * nqt * 4, qkv, dO, LSE, Drow, S, nq, nkv, scale, 4, stride)
fails += not close("dq C=4 pdf feed-on vs feed-off", dq_on_c4, dq_off_c4, 1e-4)
fails += not close("dq C=4 feed-on vs C=1 golden", dq_on_c4, dq_off_c1, 1e-4)

# ---- s8192 shape: dkv (G=2, atomic epilogue -> tolerance) -------------------
G = nq // nkv
print(f"dkv S={S} nq={nq} nkv={nkv} (G={G}):", flush=True)
dkv_off = run_op(EXT_OFF, "pdf", mk.OP_ATTN_DKV_WG, nkv * nqt * G * 1, qkv, dO, LSE, Drow, S, nq, nkv, scale, 1, stride)
dkv_on = run_op(EXT_ON, "pdf", mk.OP_ATTN_DKV_WG, nkv * nqt * G * 1, qkv, dO, LSE, Drow, S, nq, nkv, scale, 1, stride)
fails += not close("dkv G=2 C=1 pdf feed-on vs feed-off", dkv_on, dkv_off, 1e-4)

# ---- G=1 shape: dkv bitwise (single atomic per address) ---------------------
S1, nq1, nkv1 = 2048, 2, 2
qkv1, dO1, LSE1, Drow1, stride1, scale1 = gen_inputs(S1, nq1, nkv1)
nqt1 = S1 // 128
print(f"dkv S={S1} nq={nq1} nkv={nkv1} (G=1):", flush=True)
dkv1_off = run_op(
    EXT_OFF, "pdf", mk.OP_ATTN_DKV_WG, nkv1 * nqt1 * 1 * 1, qkv1, dO1, LSE1, Drow1, S1, nq1, nkv1, scale1, 1, stride1
)
dkv1_on = run_op(
    EXT_ON, "pdf", mk.OP_ATTN_DKV_WG, nkv1 * nqt1 * 1 * 1, qkv1, dO1, LSE1, Drow1, S1, nq1, nkv1, scale1, 1, stride1
)
fails += not bitwise("dkv G=1 C=1 pdf feed-on vs feed-off", dkv1_on, dkv1_off)

# ---- combined dq+dkv in one pdf launch (mailbox request interleaving) -------
print("combined dq+dkv one launch:", flush=True)


def run_both(ext, mode):
    wsq = torch.zeros(S, stride, device=DEV, dtype=torch.float32)
    p = mk.Program()
    p.instr(
        mk.OP_ATTN_DKV_WG,
        nkv * nqt * G,
        [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(wsq), S, nq, nkv, 64, scale, 1],
    )
    p.instr(
        mk.OP_ATTN_DQ_WG,
        nq * nqt,
        [p.buf(qkv), p.buf(dO), p.buf(LSE), p.buf(Drow), p.buf(wsq), S, nq, nkv, 64, scale, 1],
    )
    p.finalize().run(ext, mode=mode)
    torch.cuda.synchronize()
    return wsq


both_off = run_both(EXT_OFF, "pdf")
both_on = run_both(EXT_ON, "pdf")
fails += not close("dq+dkv combined pdf feed-on vs feed-off", both_on, both_off, 1e-4)

print("PROBE_RESULT:", "PASS" if fails == 0 else f"FAIL({fails})", flush=True)
sys.exit(1 if fails else 0)
