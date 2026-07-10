# Era slice: ARCHIVE-20260707 part 3 (lines 11100-end; Jul 7: pdf executor, dq feeds, one-pass arc, FA4 ground truth)

## KEY PROMOTIONS
- megakernel_pdf producer executor (589ee3d): qwen-l1 -1056..-1133 12/12, l2 -1370..-1511, s3072 -21, s8192 -165; small +145 stays df (residency tax). Per-shape _PDF_MODE gate; bit-identical off-gate.
- D64 pdf-feed: S3072 (-18/-23) PROMOTED; S8192 NO-GO (+112/+68); S4096 pdf+d64feed PROMOTED (-8.5/-9.5; plain pdf alone was +41/+27 NO-GO); S2048 pdf+d64feed PROMOTED (forced-old +22/+29, 0-1/500).
- Attention PDF-feed (WG2 cp.async): exact S8192 dq-feed PROMOTED (-60/-61 cert); =2 dq+dkv no extra; S4096/S2048 widen NO-GO.
- D64 dQ RS-feed: S4096 (-34/-29, 160/160) + S8192 (-57/-66, 80/80) PROMOTED; S3072/S2048 NO-GO. dQ fp32-P: S8192 (-63/-67, 80/80) + S3072 (-10/-15) PROMOTED; S4096 neutral.
- dQ bulk-reduce drain (cp.reduce.async.bulk.add.f32): exact-S4096 PROMOTED (-41/-33, 40/40); S8192 REFUTED (+55/+42 — C=4 tail bands critical-path, mandatory wait_group 0). Exposed ptxas 13.1 ADD.U64 clone bug -> forceinline + per-clone SASS audit repo-wide.
- exp2 prebias: S4096 (-3/-14); S8192 FLIPPED post-dq-feed (+58 -> -50/-64) PROMOTED; S2048/S3072 NO-GO.
- fwd defer-rsum: PROMOTED all D64 wg-fwd (s8192 -37/-47, s4096 -10/-7; off costs +54).
- dkv LSE/Drow smem prefetch: S8192 -55..-64 PROMOTED.
- S4096 dQ scalar store (float2 OFF): PROMOTED (-11/-3).
- qwen NT lm-head n256 NT TMA feed: L1 PROMOTED (forced-old +66/+79); L2 default NO-GO at this time.
- Scheduling: s8192 band LPT order post-pdf (-16/-20); qwen L2 hot-embed (forced-old +293/+134); qwen L2 WGU dW hot leaf (forced-cold +133/+158); qkv-v slot deps S3072+S8192 (-27/-37, -13/-16); s3072 bwd-band T20 post-p-pack flip; nano idle32.
- GEMM dX TMA-reduce split-K short shapes (abf854b): s128 -19/-6, s256 -14/-17, nano -7/-8, deep -23/-31.
- D64 GEMM TMA widening S2048 (-16/-8); s2048 qkrope-n128 post-wave flip (-14/-11).
- dW sk1 no-atomic short shapes (s128/s256/nano) PROMOTED; s1024 NO-GO order-flipped.
- graphstep (make_graphed_step): GO, +8..+37us/step recoverable, meter-policy decision.
- microbatch-interleave (mbi): GO/de-risked — merged/2 ratio s128 0.599, deep 0.654, nano 0.664, small 0.820; native ratio 0.694 s128; beats compile+CUDAGraph per-sequence at worst shapes.

## ONE-PASS ARC (critical sequence)
- Standalone feasibility GO: dK/dV persistent + dQ transient bulk drain, REG 202-209/224, parity clean, 1.41x s4096 / 1.31x s8192; bulk drain beats atomics ~8%; REFUTES old +119us claim as drain+context artifact.
- Model integration NO-GO: S4096 +413/+421, S8192 +1312/+1327 (stack ws224/mk336).
- Chunk sweep NO-GO (C changes tile count not critical path). Bulk-drain variant NO-GO (S4096 +1087, S8192 +4476).
- (Part of same arc, from other slice: monolithic PDR +1915/+1886, best retune +1249; banded PDR +428/+383.)
- LAW FORMED: standalone ratio is not a model promise; band/LPT overlap deletion dominates GEMM-unit savings.

## KEY NO-GOs
- dS quad-store (S8192 +924!); DKV P RS-feed (compile: RS A must be K-major); DKV S^T RS-feed (+28/+38); DKV split-wait (wash); DQ split-wait D64 (order-collapsed at 120 reps); DKV_ROW_BCAST dead knob (closed-by-analysis).
- qwen head-dX SKR2 (+1160/+1142!); head-dX n256-SKR (+33..+113); qwen SKR law: sub-wave SMs already recycled into off-path dW sink pool; SKR only pays when fat GEMM is exclusive.
- qwen CE lanes all NO-GO: CE-partial off (+589/+574 — bit11 load-bearing), fused epilogue, chunked epilogue, CE256 partials (x3 independent), full-row CE split (+558/+604).
- qwen NT supertile port NO-GO (+97..+524 despite standalone 470TF vs in-model 326TF); NT fwd ladder closed-by-analysis: residual is tile geometry, ~1.6x NT DRAM-writeback floor (LATER BROKEN by EVICT_FIRST store — see Jul-7 overnight).
- lm dW body lanes: m128-pair REFUTED (+8%); m128n128/4stage (+744); m128n256/4stage standalone -7% -> model port NO-GO (198KB page tax); TN-TMA off NO-GO (+671..+1856 — feed load-bearing). lm dW = ~575us interpreter tax + 2.0x steady-state mainloop vs nvjet.
- GEMM stage-merge NO-GO all shapes (+16..+73); long-K deep-stage pipe NO-GO (smem208 tax).
- df2 at qwen (+772/+1260); pdf+gates mixed wash; pdfg executor wash ("executor-regime wins don't compose across executors"); pdf on chain-bound shapes NO-GO (deep +76/+92, nano +37/+34 — pdf wins are feed/exposure, not generic shell).
- Dual-feed-drain family (5 slices) all NO-GO (delayed +24/+59; one-pending +42/+86; ring DEADLOCK; ringflush +180/+154).
- Phase-adaptive cold-release NO-GO (cuts wait 1046->543 but +285 span contention); nanosleep backoff NO-GO; S8192 claim shrink CATASTROPHIC (+1200..+4754); MK_ACCT_RELEASE_ATOM PARITY-FAIL (fence is data-visibility load-bearing).
- EXPFOLD NO-GO (+47/+43 — deleted FMULs were free co-issue with MUFU; MUFU-bound).
- D128 rowsplit tensor-map bulk-reduce: C2-vs-C2 wins but promotion-relevant C1-vs-C2 order-mixed NO-GO. Fence fix makes _br parity-clean (closed-by-analysis).
- baseline CE hardening: ce-bf16 NOT loss-equivalent (rejected as baseline change); ce-chunk capture-unsafe. fp32-CE denominators stand.

## MEASUREMENT
- FA4 at-shape ground truth: bwd-main 389 TF/s s8192 / 285 s4096 / 212 s3072 / 137 s2048; "450-550" folklore = machine-filling artifact; s8192 op headroom ~1.9x of which ~1.4x = two-pass recompute.
- Attention TMA layout feasibility: GEMM SW128 TMA CANNOT feed wga_off64 (504/512 vector mismatch); producer feed must be WG2 cp.async.
- Certified trajectory Jul-7: qwen ~4x -> 1.11/1.23; s8192 ~2.02x; deep 1.33.

## PATTERNS (this slice)
FAILURES: register/stack/smem cliff (~8); standalone-win-dies-in-model (~6); mandatory completion wait on critical tail (~4); order-mixed phantoms at 40 reps -> died at 120-320 (~10); scheduling can't reach dependency-structural waits (4 independent mechanisms failed on l2 EMBED_BWD join); toolchain traps (ptxas U64 clone, tensor-map no row-stride, missing proxy fence, CUTE RS-A K-major); free-co-issue deletions worthless (EXPFOLD).
WINS: register-feed/smem-store elimination (4); producer/TMA feed to the RIGHT shapes (5; same feed S3072 yes / S8192 no); bulk-reduce drains off-tail only (2); per-shape hot-leaf classification (2); exact-shape gating rescued everything.
PROCESS: 4+ concurrent agent identities, few collisions; 3 explicit duplicate-work events; resweep law load-bearing (4+ verdict flips post-structural-change); certification catches uniform taxes; standalone-probe-before-port necessary but not sufficient; heavy docs overhead notable.
