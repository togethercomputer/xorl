# Era slice: NOTES.md part 2 (P4b qwen program -> Jul-8 SASS-gate frontier)

## KEY PROMOTIONS (chronological clusters)
- qwen dW consolidation: sk1 no-atomic (+3120us; split-K was a scheduler artifact); dW n256 TN fp32 direct (+3178); head-dX/lm-head n256 confirmations (+504/+862 forced-old).
- D128 WGMMA trio default-on (+1885/+1627 vs WMMA, ~9% of step); needed carveout-refresh fix + 120KB carveout.
- qwen no-residual + residual NT bf16 n256 direct (+228, +276; preserves fused SSQ/variance elision).
- D128 dQ row-split (dQ 317->186us; +32/+35); D128 fwd mbarrier ring (fwd 225->131us; +38/+31); direct-store dW fill elision (+523/+514, n_instr 51->46); n256 3-stage operand ring (+424/+387; lm-head 2756->2574).
- Cooperative cluster launch: opt-in PROMOTED neutral (-2.7us) — enables B-multicast, no tax. N-major tile order (+39/+29).
- Sparse embedding-grad clear (OP_EMBED_ZERO_ROWS, +259/+234); D128 claim1 (+37/+35, overturned earlier wash after sparse-embed); SwiGLU_BWD_2W qwen (+399/+386 — decomposition showed win is 2W, NOT cache; cache-only -251).
- __noinline__ D128 trio (small 3660->3557, STACK 176->128) + D64 trio (local-LD 4.09M->0; S8192 -502) — dispatch-spill crisis resolution. LAW: fat op bodies noinline; certify by executed local-load sectors, NOT STACK.
- bwd S/dP one-batch x2 + fused ALU (S8192 -113/-93); D128 dQ rowsplit register-A dS feed (103.6->99.6us standalone; avoids dispatch tax that killed dKV S^T port).
- ATTN_FWD_WG cross-stage PV pipeline (S8192 -53/-41, small -18/-15 — far below -270 hope, PV drain absorbed = 8th absorption strike; kept).
- Long-S head block n256 direct (S8192 -245); combine unroll S8192+S4096 (kills dynamic w[8] local loads).
- Fused-qkrope n128 S8192 (+56/+52 forced-old; span 211->173), S4096 (-41/-31), S3072 (-43/-39); S2048 NO-GO then FLIPPED post-d64-feed.
- Small head-dX SKR skr=2 (-115/-107, 40/40; wave-quantization law: half-wave 32-tile -> one wave); nano/deep skr=4 (-10/-14, -32/-31); s3072/s4096 SKR-2 (-118/-120, -90/-75; SUPERSEDES s3072 n256-TMA).
- Small lm_head n256 post-SKR resweep FLIP (-33/-22; prior regress verdict stale); s1024/s2048 lm_head n256 (-25/-20).
- QKNORM per-warp dw partials (small -21, S4096 -16.5, S8192 -42/-65; was 30x above bandwidth floor from smem-atomic serialization on 64 addresses).
- qwen n256 NN TMA-feed (-340/-335; elected-thread cp.async.bulk.tensor + expect_tx; "the different ring protocol the multicast memo needed"); TN TMA-feed in-model (-294/-333 despite standalone order-mixed — R1-exception: off-path dW sinks stop burning issue slots).
- qwen4b-L2 cluster extension (-1530/-1509); L2 head-dX n256 (-2152/-2149; L2 16.21->12.06ms -26%); L2 peel-resweep: dq-RS DECOUPLED to L1-only; NN-only TMA INVERTS at L2 (+1.5ms — doubled TN dW contends with elected-thread issue; full TMA removes cp.async pressure).
- D64 ring TMA feed S3072/4096/8192 (-23..-31); S2048 widened later. TWO PORT LESSONS: merged use_tma branch taxes never-taken path ~30us (TMA path must be FULLY SEPARATE loop); m64n64 was last non-noinline fat frame. LOSS-PARITY FACT: df loss not bit-stable within one arm — gate = cross-arm delta within within-arm replay spread.
- Producer-df mode=pdf LANDED: 384 threads/__maxnreg__(168), consumers setmaxnreg.inc 240, WG2 dec 24 TMA issuer + MkPdfFeed mailbox (qwen-l1 -1056/-1127, lm-head dX 2453->1680 -32%; l2 -1511/-1370; s3072 -20; s8192 -171/-162; small +142 stays df). MK_DF_MAXNREG=240 flat cap WORSE (+30/+44): the win is region-compiled image + producer, not lower cap.
- small idle64; nano/deep idle64->32 gates; mbi/graphstep (see other slices).

## KEY NO-GOs
- mbarrier feed-ring in-model generic port (standalone -20-27% -> in-model +16/+51 — interpreter helpers still wait<0> per batch); D128 dQ mbarrier/split-wait (absorbed after row-split); D128 dQ register-A feed in-model (absorbed post-row-split).
- DSMEM-fed GMMA (descriptor can't carry cluster rank; err 32.5); production-shaped n256 pair TMA multicast (tma-nosync 46.7 vs cpasync 44.1 — needs decoupled producer).
- MK_CLAIM 64/32/16 multi-tile (claim64 +3359, claim16 +17797!); selective head-dX M-major mask.
- qwen H2560 RMS R4 (-22/-25 forced); combined RMS bwd; CE fwd/bwd fusion (order-mixed; local hop not bottleneck); head-dX n256 split-K probe (extra fill/wave/atomics).
- fwd KV-widening w128 in-model: NO-GO everywhere (5th absorption strike; banded shapes expose only short tiles; numerics caveat 0.021).
- Epilogue-fusion phase 1 rmsnorm dissolution (MK_EPIFUSE_MLP): NO-GO — critical_path 144->144 (hop REPLACEMENT != deletion; 7th absorption strike). Phase 2 xn2-deletion: NO-GO (+16; bwd decisive loser; B-rms micro-opt correctness fail).
- Partials-fed RMS dX (bit16): NO-GO — rowop bucket shrinks but dX gemm grows +40 (cache-state law: xn loads cold vs dy L2-hot); flag-gated accumulator in hot epilogue spills even flag-off -> separate loops law.
- dq cross-stage ping-pong (MK_ATTN_DQ_PIPE): +3/+13/+30 — dq critical path is score->ALU; two in-flight batches contend on tensor pipe with sibling WG; in-place ping-pong DEAD.
- gate_up dX per-layer SKR (+83 — 8 reduce hops invert economics; SKR pays iff ONE giant sub-wave long-K gemm amortizes ONE reduce).
- S8192 ready-order family (4 probes): DQ-tail/tail-pair/DQ-family/hot-fair — ALL relocate wait to siblings, span grows (+67..+201). Multi-sibling fanout = executor-structural.
- S1024 stale-check battery (0ba235d): qkrope n128, SSQ-off, RSSQ direct epilogue, FWD w128 refresh, forced pdf composition, FWD GQA64, DKV PDS ring (stack 96->368), MLP dX static — ALL NO-GO (mostly order-split).
- qwen NT SASS-gate chain: wait-distance, CE-store single-pass, descriptor-precompute, x2-commit — NO-GO at SASS gate (ptxas preserves DEPBAR cadence); singleton-drain trigger isolated = call_inline_wrapper (forceinline wrapper CALL) -> direct in-body _ntdi STILL byte-identical NO-GO; small-attn inline NO-GO (cadence unmoved, stack grows).
- dkv stage-anatomy: span 470.9us = ALU 40.8% + drain 21% + score gemm 10.2% + syncs 28.1%; span cuts COMPOUND but step doesn't follow (no-alu cut 192us span but 485us step ???); fwd anatomy: softmax ALU 42.6% (exp 30.7%), PV drain 27.8%; fwd span SUB-translates 0.7:1 vs dkv 2.5x. DCE LAW: accumulator-dead wgmma NOT protected by asm volatile.
- Producer-df 240/24 first attempt CLOSED NO-GO pre-pdf (parked-WG2 residency +34us nano ~5% — even nanosleep-parked WG dilutes issue) — later SUPERSEDED by real pdf executor with feed.

## PATTERNS (this slice)
- Absorption ledger ~8 strikes; resweep law ~9 flips (+1 inverse revert); order-split ~25+; noinline/separate-loop codegen law 4 events; SASS-gate rejection ~9 (efficient pre-timing kills); wait-relocation 4; long-K/wave-quantization economics ~7 (SKR gate: K/CTAs >= nvjet ~90; one sub-wave amortizes one reduce); exact-gating universal; certification catches uniform taxes invisible to A/Bs; R1-exception (TN-TMA: in-model-only composition WIN).
