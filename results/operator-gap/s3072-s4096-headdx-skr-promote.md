# s3072/s4096 head-dX SKR promotion (+ s1024 order-flip no-go, shape-sweep close-out)

Session b603d819, 2026-07-06/07. Closes the head-dX SKR shape-extension sweep
(ranked item 2 remainder in NEXT-AGENT-PROMPT.md); with this the SKR gate map
covers every shape in the matrix: promoted {small 2, nano 4, deep 4, s3072 2,
s4096 2}, no-go {s128, s256, s1024, s2048, s8192(K/CTAs=64 < 90, never probed
— gate law), qwen (n256+TMA route owns the head)}.

## What changed

`_HEAD_DX_SKR` gains exact tuples (256,3072,768,8192,4,2,64,4): 2 and
(256,4096,768,8192,4,2,64,4): 2. Routing-only — the SKR mechanism (GEMM flag
bit15 + OP_SKR_REDUCE) is the one promoted at 193a108 for small.

## Why it wins here (mechanism)

The prior s3072/s4096 head-dX default was the exact-gated **n256 direct** route
(flags=16520) at **24/32 CTAs** — parallelism-starved on 132 SMs with the full
K=8192 serial chain per tile. SKR-2 routes n128 (48/64 base tiles) x 2 K-slices
= 96/128 CTAs, still sub-wave, halving each tile's serial K chain, for one
~3-4us reduce hop. K/CTAs = 170 (s3072) / 128 (s4096) — both above nvjet's ~90
split gate; in-model reproduces nvjet's own policy. skr=4 measured inferior
(-100.4 vs -118.6 at s3072): 192 CTAs crosses one wave and adds slab traffic —
the wave-quantization law again.

## Evidence (GPU 6, parked-tenant idle, absolutes in-band, paired alternating
CUDA events, fresh process per run; driver results/headdx_skr_shape_ab_b603d819.py)

Head-anchored at 0abc08f (logs mkv3-p4b-hdskr-shape-*-2026070*):
- s3072 skr=2: default-first **-118.6us 40/40**; variant-first **-120.6us 40/40**
  (defaults 2562.5/2540.4us). skr=4 check: -100.4 40/40 (inferior — curve bends).
- s4096 skr=2: default-first **-90.0us 40/40**; variant-first **-75.6us 39/40**
  (defaults 3150.4/3134.4us).
- Base-anchor 5c6e234 pre-checks agreed: s3072 -119.4/-118.5 40/40; s4096
  -75.3/-72.7 40/40.
- Parity everywhere: loss diff <= 1e-05, worst selected grad rel <= 7.1e-03 (kn/qn).

Promoted-vs-forced-old (MK_HEAD_DX_SKR=0), first pass at 0abc08f: s3072 old
+120.0/+115.2 0/40 both orders (default-first re-run after a co-tenant blip);
s4096 old +79.6/+80.8 0/40 both orders. Re-anchored confirmations at the final
promotion surface on 30473f1 (s3072 old arm = TMA-fed n256 per 0b7ed2a):
s3072 old +95.2/+89.4us 0/40 both orders (defaults 2435.2/2426.9us);
s4096 old +77.4/+80.9us 0/40 both orders (defaults 3059.5/3062.1us).
Clean GPU-6 windows (util 0-4%), parity clean (worst kn/qn <= 6.1e-03).

## Supersession note (0b7ed2a interaction)

While this promotion was in flight, 0b7ed2a promoted the n256 TMA feed at
s3072 — whose ONLY eligible row is this same head-dX 3072x256x8192 (-7.1/-10.2
vs the un-fed n256). SKR-2 reroutes that row to n128 slabs, so the TMA row
leaves the s3072 default route: this promotion SUPERSEDES the 0b7ed2a default
(the gate remains reachable via MK_HEAD_DX_SKR=0, and the ledger's long-K TMA
principle is untouched — it simply now applies to no default H256 row below
qwen). The re-anchored forced-old arm below measures against the STRONGER
TMA-fed old route, so it is the honest marginal value of SKR at s3072.

## ncu local-LD sector gate (l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum)

- s4096: promoted == old == 2022 sectors/step, bit-identical. PASS.
- s3072: promoted 2.676M vs old 2.715M — promoted LOWER, PASS. NB both arms
  carry a large pre-existing s3072 local-LD baseline (~2.7M sectors vs
  s4096's 2K) that predates this change — flagged to the LD-sweep owner on
  the board; attribute if it grows.

## s1024 NO-GO (order-flip, resweep-law casualty)

s1024 skr=4 won both orders at 5c6e234 on GPU 7 (-10.8 39/40, -5.9 35/40) but
at 0abc08f (post per-warp qknorm-rope-bwd dw partials 61e37bd) it order-flips:
-6.3 69/80 default-first, **+10.0 4/80 variant-first**. Peer GPU-3 read at an
intermediate head agreed (-3.5, 29/40 "noise"). Stays on its atomic sk route;
recheck only after the next structural change to the s1024 step (R4).

## Sweep no-gos (this session, GPU 7 clean-prefix window + peer confirmation)

- s128 skr=16: +19.6 0/40 (peer: +36.1 0/40 at skr=4). s256 skr=16: +9.3 2/40
  (peer: +11.9 2/40). Mechanism (peer): n128 tile shape collapses at M<=256;
  mine adds: at tiny S*H the reduce hop exceeds the atomic tax.
- s1024 skr=8: +16..+20 0/40 both orders (past-parallelism slab traffic);
  skr=2 -6.0 29/40 (weaker than skr=4).
- GPU 7 went active-co-tenant mid-batch (util 45->84%): the s3072 -316us and
  s4096 -78us rows in mkv3-p4b-hdskr-shape-s{3072,4096}-skr2-default_first-
  20260706T22*Z.log are contaminated artifacts; all promotion evidence is from
  the clean GPU 6 windows above.

## Gates

- R0: res-usage at 0abc08f flavor pair (idle32 gmbar class): megakernel_df
  255/48 and megakernel_ws 168/96 BIT-IDENTICAL; megakernel stack 48->64
  (+16B, REG/LOCAL unchanged) — same benign delta as the small promotion.
  No smem/carveout change; slab (2*S*H*4 = 6/8MB) is global-memory only.
- R1: head-dX is the exposed lm-head dX hop at long-S (on-path in every
  profile); not banded.
- R2: pure work/boundary removal (serial-K halving + no added sync class;
  reduce hop = existing rowop pattern).
- R3: exposure ~1 at S>=3072 — realized -75..-121us is ~2-4% of step,
  consistent with the span sizes.
- R4: post-landing, cheap env knobs at s3072/s4096 should be resworn by the
  next session (band budgets T, band order, fwd-band T22 at s4096 in
  particular — all were tuned against the old head region).
