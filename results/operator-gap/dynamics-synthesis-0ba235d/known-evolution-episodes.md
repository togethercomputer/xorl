# Known law-revision episodes (from direct reads; wave 2 must recover + date these)

E1 one-pass attention bwd: Jul-5 "REFUTED, store-amplification law (+119us s4096)" ->
Jul-7 FA4 map: "reassess under pdf budget; 1.4x recompute handicap = biggest lever" ->
Jul-7 standalone GO (208/224 regs, 1.41x/1.31x; old refutation was the atomic drain) ->
Jul-7 in-model monolithic PDR NO-GO (+1.9ms; +1.25ms best retune) ->
post-PDR law: overlap/banded exposure dominates GEMM-unit count; "banded one-pass"
is the surviving direction. LESSON: a law can be right about the mechanism (drain)
and wrong about the binding constraint (overlap exposure).

E2 lm-head NT fwd floor: Jul-6 "FLOOR 146us/1.66x - DRAM writeback behavior, not
portable" -> Jul-7 overnight: floor BROKEN (SW128 64-col-slab TMA C store +
store-side EVICT_FIRST = 110.7us/1.21TB/s; load-side EVICT_FIRST harmful) ->
landed default-on long-S (s8192 -101/-66us). LESSON: "floor" verdicts are
floor-under-mechanisms-tried; the store-policy axis was unexplored.

E3 split-K reduce: Jul-5 "rejected - atomic sinks already win" -> Jul-7 head-dX SKR
promoted with a per-shape gate map (small/nano/deep/s3072/s4096; s4096 old-route
penalty grew to +116us under pdf composition). LESSON: global verdicts hide
per-shape gates; promotions compose (SKR x dq-feed x pdf grew the win).

E4 cooperative launch: early "CGA/cluster B-multicast needs cooperative launch ->
rejected" -> "cluster+cooperative launch COMPOSES (memos corrected)" -> Jul-7 eve
"coop launch IS graph-capturable, +8-17us/step all shapes". LESSON: structural
impossibility claims rot fastest.

E5 qwen arc: Jul-5 4.05x (WMMA D128 fallback ~2.9ms, lm-head spans 3.4-4.3x nvjet,
EMBED_BWD waits 14.2ms on dW sinks) -> D128 WGMMA trio -> n256 routing + TMA feeds
-> pdf producer executor (lm-head dX 2453->1680us) -> exact-qwen NT supertile
PDF-only route -> 1.028x. LESSON: the biggest ratio closed via per-family routing +
producer feed, zero exotic new math.

E6 dW TN rows: standalone-neutral -> -300us in-model. Twin of E1's inverse
(standalone GO -> in-model NO-GO). Formed measurement law: standalone evidence
gates trying in-model, in BOTH directions.

E7 wgmma-inline/frame war (Jul-8 wgi round-3): killed 60% of spill tax, pipelining
held, STILL 0 wins - attention-bwd working set at 240 regs is real; in-chain HGMMA
operands fed from local. LESSON: removing a visible tax is worthless if the
binding constraint (working set) stands.

E8 qwen NT singleton drain (Jul-8/9): every plain C++ respelling (helper call, raw,
direct-issue, descriptor forms, inlining) failed to change HGMMA=DEPBAR=1 cadence;
standalone probes group {16:4}; collapse trigger isolated to translation-unit
body-context (forceinline wrapper call). LESSON: ptxas cadence is body-context
dependent; SASS-visibility gate before timing is mandatory; source-form lanes
against a compiler behavior are dead without a lowering reproducer.

E9 fusion-for-hop-deletion: bit8 qk-rope fwd epilogue fold PROMOTED (tile boundary =
op boundary, no cross-tile data) vs qkrope BWD fold NO-GO +811us (needed
last-arrival counters + hot-path atomics) vs rmsnorm-fwd true-deletion sized
-100..-180us deep (commutes through GEMM; epilogue row-scale only). LESSON: a
fusion pays iff it deletes cp hops WITHOUT adding on-path serialization; the
dependency algebra (does the op commute through the contraction?) decides.

E10 producer executor residency: pdf wins long-S/qwen (WG2 TMA producer) but parked
producer WG costs +34us nano / +145us small -> per-shape image routing (df short,
pdf long). LESSON: warp-slot dilution is a real tax; every image is a point on a
residency curve, not a universal upgrade.
