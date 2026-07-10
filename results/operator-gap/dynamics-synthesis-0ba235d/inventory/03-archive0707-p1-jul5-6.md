# Era slice: ARCHIVE-20260707 part 1 (lines 1-5500; Jul 5-6 era: matrix + probe rounds + qwen n256 cascade + banding)

## KEY PROMOTIONS
- qkrope n128 routing S3072/S4096 (-42/-32, -43/-39), s8192 (+56 forced-old); s8192 exact n256 NN BF16 (-29/-18); small NT n128 retune (+56 forced-old, 1.78x); small MLP-dX n128 gates (+39/+33); small direct-BF16 epilogue (+28, 80/80); head-dX target retunes global 512->256->192 + per-shape target-96 gates (S2048 -39/-14, S3072 -20/-16, small -16/-12, nano -15); n128 short-row auto-gate (s128 -50, s256 -19, nano -9, deep -38); nano NN_MIN=16 M=512-only (-19.8, 479/480).
- dW: cold split-K target retune (nano -57, small -232, S4096 -170); long-S TN WGMMA dW gate K>=3072 (S3072 -54, S4096 -107, S8192 -333); resweeps confirm load-bearing.
- QWEN n256 cascade (all PROMOTED, exact-gated): lm-head direct n256 (+909); head-dX n256 fp32 (+597); head-dX n128 no-atomic (+1519); dW sk1 no-atomic (+3120); vocab-dW TN n256 (+3178); MLP-dX n256 (+99); qkv-dX n256 (+26/+32); D128 fused dOatt/Drow n256 (+120/+140); n256 mbar ring (-169/-191); NT mbar split (-112/-141); D128 dQ Cq=1 (+85/+95); cold-cap0 giant-L1 (29433->22094); SWIGLU_BWD_4W (+356/+343); D128 QKBWD cached-pair (+29); CE_BWD label-fixup (-26).
- D128 WGMMA attention trio (redundant-S + split-D): PROMOTED default-on (+1885/+1628 qwen; fwd -903/-935).
- Banded attention-bwd chunking (S8192 -532/-525); fwd-band (S2048 -20, S3072 -46, S8192 -452); combine row-batching R=8 (S3072 -33, S4096 -29, S8192 -70); dq_first band order S8192-only (-103/-74; regresses elsewhere); QKBWD split-V S8192 (-35/-63) widened S4096/S3072.
- DKV float2-atomic (nano -5.7..S8192 -47); DKV direct-atomic (nano -18, small -49, S2048 -41) — earlier attempt NO-GO, FLIPPED after route changes; DQ C=1 direct-store + register-direct (nano -11, small -24); DQ float2 S3072/S4096(+S8192 later).
- Rowops: RMS dx/dw split (structural); RMS_BWD_DX R4 narrow (S2048, small-H512); RMS dx H256 exact-S8192 (-25/-43); SWIGLU_BWD_2W + cached variants per-shape; SWIGLU 1W sigmoid-cache H256/S1024 only; QKBWD D64 cache (nano -7..S4096 -31); Drow/dOatt WGMMA long-S gate then broadened (small -21).
- MK_ATTN_FAST_LOG (small -14, S2048 -12, S4096 -11); cold-cap shape-gates (uncap S>=2048; mid-S cap33/48); idle32 gates incl S8192 post-band flip; MK_CLAIM=132 confirmed every sweep.
- knob consolidation ~12 gates -> one table, route-identical verified, MERGED.

## KEY NO-GOs / REFUTED
- Broad n256 NT (+150/+162); m64n256 small MLP-dX (+210/+251); small dX n128 split-K (+65/+189); n128 Drow epilogue (+80).
- SW128 deep-K stages: standalone win, 208KB page -> global negative (nano +6, S8192 +95). Fat-tile 128x256: 160KB coop launch FAILS.
- Fused one-pass bwd: REFUTED S4096 +119, 0/40 — store-amplification vs L2-resident re-reads; "scales worse with S, do not revisit."
- MK_ATTN_PIPE rechecks decisive NO-GO (+40..+303); FA4-B KV-widening standalone-win -> in-model NO-GO by absorption (banded shapes expose only short tiles); uniform-C fwd split + global combine (+9..+222).
- DKV G2 GQA-fuse (+221..+455, lost G-parallelism); D128 DKV S^T port (+10, STACK 32->160); DKV/DQ X2_SD batching; DKV C>=2 sweeps (keep C=1); DKV cold-ring demotion (+41, must stay hot); tail-barrier elision (small +186, barriers load-bearing); DQ split-mask (+42/+46); short-shape DQ cold demotion catastrophic.
- QKNORM split-DW REFUTED (unlike RMS: too much cold work); SWIGLU cache-sig qwen NO-GO; SWIGLU_BWD_3W no change; DSSG cache (+53); rcp/exp2/fast-inv precision NO-GOs (fast-inv failed SGD sanity).
- CE fast-log, CE_BWD R2, CE label-fixup broadening: NO-GO (can't split cheap-vs-expensive shapes with one knob).
- Executor rechecks: df2 +207..+482, ws +70..+314, 0 wins — df best every time; df2 hot/cold rings CORRECTNESS NO-GO (grad drift 0.112); MPK probe 5.09us/hop vs df 3.0; OCC2 REFUTED (+32-40%).
- per-layer scratch: deps 2030->1190 but NO-GO (false deps already absorbed); D128 DQ claim=1: standalone -48/-85 -> in-model wash ("6th absorption strike"); rowop long-S decomposition: bodies at baseline parity, bucket is co-scheduling span-stretch.
- drow direct-store in n256 weak REFUTED; dispatch-spill ALERT corrected (stack diagnostic-only, not promotion gate).

## PATTERNS (this slice)
FAILURES: order-mixed bias ~18+; ABSORPTION ~6 strikes ("only dependent-latency-chain shortening pays at 8 warps/SM"); store-amplification; stack blowup from fat bodies; wave-quantization cliffs (banded T budgets sharp optima); one knob can't split cheap/expensive shapes; broad-vs-narrow (broad always loses).
WINS: dependent-latency-chain shortening; narrow exact-shape/K-threshold gates; direct-atomic/direct-store epilogues once route makes smem drain the bottleneck; regime-shift resweeps (11x avg promotion value); qwen giant-vocab n256 cascade.
PROCESS: measurement discipline backbone (two orders, paired medians, fresh proc, guards, parity, forced-old confirm); SDPA 3-D soft-baseline bug = biggest process failure (invalidated all prior ratios); negative-result ledger prevented thrash; co-tenant contamination quarantines; claim protocol worked with rare dirty-tree mistake (retracted+rerun).
