# Era slice: ARCHIVE-20260708 (both halves; Jul 8: gate batch, mbi, post-burst cert, s1024 grind, wgmma-inline war, qwen ST-S3)

## PROMOTED
- dX TMA-reduce split-K short shapes (abf854b, -6..-30); dQ bulk-red gate map deep/nano/s1024/s2048/s3072 (-3..-23); s8192 REFUTED-STRENGTHENED post-dW-SK1 (+98/+100 — on-path wait law amplified).
- NT store-recipe standalone: SW128 64-col-slab TMA C store + store-side EVICT_FIRST = BROKE the 146us NT floor (110.4us/1.21TB/s, -24%); load-side EVICT_FIRST harmful. In-model n256 TMA C-store landed 3d37664 (s8192 -101/-66; in-model win EXCEEDS standalone — relieves contended L2); small extension (-13/-17); s1024 NO-GO.
- dW SK1 no-atomic deep/small/mid/long-S (small -262! s2048 -39); s1024 WASH excluded.
- dQ float2 gate-map S128/S256 added; S1024/S2048 NO-GO. Drow register-epilogue all short shapes (+9..+31 forced-off). s256 RS-feed+fp32-P composition candidate.
- RMS dX H256 map completed (wins 128/512/1024/2048/3072/8192; 256+4096 NO-GO). nano idle32.
- short-S attention gates (exp2 s128/s256, prebias deep, + DQ_RS/FP32_P bundles).
- qwen L2 NT-TMA (b906b38); qwen ST-S3 CE64 store-loop supertile (row-exclusive 1887 vs 3113; -546/-506 x16; promoted-candidate, then ported across 5 heads; L1 ~1.04x); qwen kind6 stack integration (fixed kind=4 collision hang).
- s1024 promotions chain: head-dX SKR=4 (-6.5/-4.0); fwd expfold; fwd de-diverge; exp2-prebias; DKV fp32-P (-8.5/-6.6); DQ RS-feed+fp32-P composed; TN dW TMA-red (-8.6/-3.8); direct-BF16 epilogue (-24/-17); fwd mask-split (promoted Jul-8 eve, -3..-9).
- qwen L2: exact n256 head-dX body (-23/-39); D128 dQ RS-feed (-133/-145); WQKV dW hot-leaf (-148/-150); lm-head dW hot-leaf (-247/-229); RMS dX H2560 (-10/-39); head-dX PDF-only (promote candidate -27..-37).
- mbi (microbatch interleave): native in-model n-sweep (s128 1.47/1.75/1.89x @n2/3/4, saturates ~2.0x; monotone decline to s8192 1.08x — tracks itax starvation); AXPY tail n==2 only (deep -82; n=3 +249 REFUTED); pdf opt-in (s2048 1.27x; TMAP raw-pointer hazard found+fixed); model.py-native landed; graphed merged launch (s128 n=4 342us/seq = 1.44x FASTER than compile+CUDAGraph+); graphstep landed (+8-17us/step all shapes).
- cold-cap resweep: small 48->0, s2048 0->33 landed (05e9e97); 5 others stand.
- short-GEMM N64 fast-dispatch (bypass dispatch switch for tiny rows): promoted-candidate all short shapes (deep +57 forced-old!), s1024 gated OFF; blocked on dirty tree then landed.
- pdf shell-smem: exact qwen-l1 only (-28/-32); s8192 LOSS keep-OFF; root cause = ABI call boundary at 240/24 exact-balance point, NOT inline pressure (noinline made LD WORSE 4.83->6.32M — P1 noinline law REFUTED for this class).

## NO-GO
- head-dX n128 TMA reduce-add (codegen: non-F32 UBLKRED deterministic); GEMM ring stage-merge (+16..+73 all); S512 qkrope-n128 force; short qkrope unfuse; DeepGEMM R7 swizzle closed-by-analysis; R2a ring carry (claim=1 tile); R4 parked (EV 30-80us vs invasive); R5/R6 closed (no target / SS-only bodies).
- DKV RS/S^T/QPAIR/SPLITWAIT family REFUTED-with-mechanism (s8192 +85..+105, loss GROWS with S; identical REG/STACK/LOCAL => pure ILP loss — "register-lifetime 5th strike"). DQ SPLITWAIT order-collapsed at 120 reps.
- One-pass arc completion: standalone GO (1.41x/1.31x, refutes old +119 as drain artifact) -> in-model integration +413/+1312 -> chunk sweep no help -> bulk-drain +1087/+4476 -> monolithic PDR +1915 -> banded PDR +428. ALL NO-GO. Law: band/LPT overlap deletion dominates GEMM-unit savings; standalone ratio is not a model promise.
- Attention dual-feed-drain 5 slices ALL NO-GO (one deadlock); defer2 (+231/+247); dual-drain bridge port (+193/+197).
- qwen CE lanes: chunked epilogue, CE256 x3, full-row split (+558/+604), fused epilogue, CEFAST, CE2PASS (NaN flakes) — ALL NO-GO. "Residual is body/executor, not CE bookkeeping."
- qwen n128 TMA row-exclusive (+1034/+1085 — n128 doubles tiles); STX body port (226KB > 148KB carveout); head-dX n256-SKR (+230..+769 — SKR law: sub-wave SMs already recycled into off-path sink pool; SKR only pays when fat GEMM is exclusive); BN64 pipe frontier route-away.
- rowop A-side RMS factorization dW (+280/+284 — scale-multiply tax); mbarrier nanosleep; MK_CLAIM global/targeted coarsening (+94..+1840); MK_ACCT_RELEASE_ATOM PARITY-FAIL (fence is data-visibility).
- s1024 grind ~15 consecutive NO-GOs: fwd alpha-fastpath, empty-fence-skip, const-scale x2, C1-static, C2-static, direct-O, DKV row-bcast, ptr-hoist, DKV bulk-red (+25/+19 — staging+fence+full wait), RMS FMA/R1/R4, sparse-embed x4 (order-mixed every time), grouped fill, WGU slab-reduce x3, wlm atomic-elide, dW targets, MLP dX static/N128/N256/D64-TMA.
- wgmma descriptor-serialization: descriptor-WAR REFUTED; root cause = __noinline__ ABI boundary (ptxas C7510); 196/196 HGMMA serialized in every executor image; __forceinline__ twins pipeline at zero reg delta BUT: wgmma-inline rounds 1/2/3 ALL NO-GO in-model (round-2 s8192 +503; spill 16.68M vs 4.83M local-LD; round-3 recovered 60% of spill, still +442) — ~70% residual = attention-bwd S^2-scaled body-internal local traffic at 240-reg wall.
- SW128+TMA staging P2: s2048/s3072 WIN both orders, s4096 wash, s8192 LOSS (+68/+77 — TMA supersedes promoted WG2 dq feed = feed displacement).
- itax decomposition: floors s128 262us..deep 717us; floor/gap 110-133% short-S; sync-elision + claim-batch widening DEAD ON ARRIVAL.

## PATTERNS
Order-mixed ~18 (dominant); spill/local-tax beats intended win ~4; ABI-boundary root causes ~3; coarsening loses parallelism ~3; rollbacks confirm defaults ~8; micro-win doesn't survive whole-step ~2.
WINS: off-path scheduling/hot-leaf classification; fp32-precision on selected-path attention; exact-shape body specialization (qwen L2); merged-launch grad accumulation + graph capture; bulk-reduce drains ON-path only.
PROCESS: both-order gating caught ~18 false wins; landing serialization on dirty shared tree = real bottleneck; two sessions built mbi independently (DEDUP/CONVERGE resolved); read-only scout triages pre-rank candidates.
