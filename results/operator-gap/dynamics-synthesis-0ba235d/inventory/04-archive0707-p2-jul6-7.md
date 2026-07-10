# Era slice: ARCHIVE-20260707 part 2 (lines 5500-11100; Jul 6-7: P4b S-sweep, opgap probes, spill saga, SKR/TMA ladder, qwen L2, producer-df)

## PROMOTED highlights
- P4b S-sweep gates: coldcap-S1024 cap48->64; nano/S1024/S2048 attn chunk retunes; dw-split-target (nano -27, S4096 -75); S1024 WGMMA-NN thresh (-108!); short WGMMA-NN (S128 -6, S256 -7); head-dx targets per shape (S256 -17, S128 -20, S1024 -12); S128 NN_MIN=4 (-12); small RMS R2 (-10); short coldcap0 (S128 -6).
- OP_INV_VALID in-kernel (-75 nano, -93 small!); launcher input binding (-13/-23); swiglu FMA (-8/-17); fastlog-restore (-20 small); nano chunk c22.
- exp2 family gated: attn-exp2 (small -38, S2048 -20); lmhead-exp2 (small -57, S2048 -66); ce-bwd-exp2 (small -36); TN-WGMMA-dW broadened (small -77); head-dx sk1 no-atomic S2048 (-8); head-dx n128 f32 small (-11); swiglu cache-sig (small -8); drow zero-fill skip narrow; RMS dx H256 nano/S1024.
- opgap probe program: fat 128x128 tiles CONFIRMED gated (wave-gate law, s8192 lm_head -15-25%, gu -38%; REFUTED sub-wave); fa4b-w128 fwd CONFIRMED standalone (+11-19%) [later absorbed in-model]; fa4c D128 trio CONFIRMED (~375us/layer vs 2.9ms WMMA); barrier-free mbarrier ring CONFIRMED standalone (-20-27% every BW-bound dX; nvjet gap 3.0->1.3x); splitK separate-reduce CONFIRMED (dX family CLOSED 1.03-1.29x everywhere).
- D128 in-model: trio default-on (+2081 qwen); dq row-split (+33); fwd mbar ring (+38); dW fill elision (+524); stage3 ring (+424); nmajor (+35); claim1 (+37); sparse embed (+259); swiglu 2W-only (+399 — win was 2W not cache).
- noinline saga: WG128 noinline (small -103, STACK 176->128); D64 trio noinline (local-ld 5.3M->0, S8192 -502) — "fat op bodies must be __noinline__" law formed.
- gemm mbar-ring gated long-D64 (small -76, S8192 -150); S8192 row-bcast rollback (+32 — didn't compose with ring); head-block n256 long-S (S8192 -245); sdp-batch x2 (S8192 -113; 2.16x cert); qknorm per-warp partials (small -21, S8192 -43); combine unroll S8192/S4096.
- small SKR skr=2 (-115!); small/s1024/s2048 lmhead n256 resweep flips; nano/deep SKR-4; s3072/s4096 SKR-2 (-119/-90); small idle64.
- qwen n256 NN TMA-feed (-340); TN TMA (-294 in-model, standalone-neutral!); l2 GEMM-cluster (-1530); l2 head-dX n256 (-2152); l2 support batch; d64 ring TMA S3072-8192 (-24..-31); producer-df pdf LANDED (qwen l1 -1056/-1127, l2 -1511; small +142 stays df); pdf phase-1 tax curve (qwen INVERTS -4.5% — 240-image beats 255 there).

## NO-GO highlights
- rms-dx-R4 repeatedly (+9..+33); allhot (+34/+65); early fills; claim resweeps (132 always; claim64 +1202 S8192!); executor ws/df2 rechecks (0 wins ever); ce-skip-ignore x3; masked-exp skip (+223 small); dS FMA; drow n128 (+44/+61); qkrope n128 broad (+34); swiglu rcp/exp2/DSSG/3W/4W_V4; dx-n128-split nano (+110); n128-nn-min64 (+54/+66).
- SW128-deep-K-ring REFUTED ("4th strike overlap-depth" — nvjet edge not feed depth); fa4b-STSM REFUTED (register-lifetime 4th strike); fa4b-w128-dq REFUTED (REG:224 ceiling); mbar-ring generic in-model port (-15/-51 = NO-GO both) — "standalone wins, in-model absorbed".
- DSMEM GMMA (descriptor can't carry rank); paired-M TMA multicast (44.1 vs 46.7 — L2 already free multicast); cluster-B-multicast round 6 (B L2-absorbed).
- Dispatch-spill saga: STACK-predicts-runtime REFUTED/WITHDRAWN (cleanest stack = slowest!); trampoline neutral; dead-code gating neutral; splitK-direct-epilogue (+27); ncu proved executed-local-ld is the metric.
- epifuse phase1 (hop replacement, +10-19 long-S); phase2 xn2-deletion (+16; B-rms micro correctness fail); partials-fed RMS (bit16, cache-state law + flag-gate spill law).
- reg168 blanket cap (+158/+307 — needs GEMM-only split); producer-warp standalone CONFIRMED 1.03x nvjet but uniform rewrite REFUTED (register-point map: {168,224,255} per-family routing).
- lm dW body: m128-pair REFUTED (m256 phys impossible +378% spill); m128n128 (+744); m128n256 standalone -7% -> port NO-GO (198KB page).
- attn-bwd-pipe-cheap HUNG x2 (needs full ring ownership); dq ping-pong (+3/+13/+32 scales with S); dkv/dq elected-TMA long-D64 generic (+34..+75 all K_MIN); NT-supertile port (+97..+524); S8192 TMA (12/13 rows short-K).
- OP_SKR_REDUCE fatter chunks ABSORBED (±3us in 536us DQ wait slack) — WAIT-column principle formed.
- s3072/s4096 dq-wait sink-steal hypothesis REFUTED (wait = legitimate band drain; only lever = op throughput).
- 4 ready-order/fair-quantum S8192 scheduler probes: wait relocates, span grows (from p2 tail: also in live board).
- baseline CE hardening rejected (ce-bf16 not loss-equivalent; ce-chunk capture-unsafe).

## MEASUREMENT/DIAGNOSTIC
- operator-gap scoreboard Jul-5: s128 1.72x..s8192 2.30x; gemm_dx worst bucket everywhere (3.8-8.3x); attention crosses at s2048.
- dkv stage anatomy: ALU 41%/drain 21%/score gemm 6-10%/syncs 28%; span cuts compound but step doesn't follow.
- fwd anatomy: softmax ALU 42.6% (exp 72% of that), PV drain 28% exposed, sub-translates 0.7:1.
- S8192 bwd-attn iclk: 83% packed — scheduling exonerated; lever is op body (200 vs [then-believed] 450-550 TF).
- NT floor 1.66x declared (C-write 0.95 vs 1.54 TB/s) [BROKEN Jul-7 overnight by EVICT_FIRST store].
- FA4 at-shape ground truth: 389/285/212/137 TF/s (the 450-550 was machine-filling folklore).

## PATTERNS (this slice)
FAILURES: order-mixed ~18; register-lifetime/ceiling >=5 strikes; absorption >=6; in-model != standalone >=5; dispatch/codegen spill (global reg allocation); overlap-depth without decoupled issue >=4 strikes; extra-hop multiplication (SKR per-layer, splits); short-M/K non-amortization.
WINS: boundary/barrier removal (biggest, portable); gated exp2/fastlog/FMA; right-sizing routes per shape (nvjet-style register-point routing); removing serialized host/support work; TMA feed + producer WG for giant long-K rows; noinline isolation; resweep flips.
PROCESS: co-tenancy poisoning recurrent; STACK->runtime WITHDRAW was a process win; IMPROVEMENTS.md ledger created explicitly to stop re-deriving ~40 dead ends; convergent duplicate work reconciled via MERGE-PROPOSAL.
