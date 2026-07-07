# One-pass D64 attention-bwd STANDALONE feasibility (session b1d36305)

Date: 2026-07-07. Base: 2a41f6a (megakernel). Worktree:
`/home/apanda/xorl-oss/.claude/worktrees/agent-a0844658ce47330a9`, branch
`onepass-d64-bwd-b1d36305`. Claim: board `20260707T0750Z-realclock (session
b1d36305 / one-pass D64 bwd STANDALONE feasibility)`. NO model integration
under this claim.

## Question

Our bwd is two passes (`OP_ATTN_DKV_WG` + `OP_ATTN_DQ_WG`): S and dP are
recomputed twice = 7 GEMM-units vs FA4's 5 (fa4-concept-map §4) — a 1.4x
structural MMA handicap that is most of the s8192 attention-bwd gap (~200 vs
389 TF/s at-shape). The OLD one-pass refutation (+119us @s4096,
`MK_ATTN_FUSED_BWD`, wt xorl-oss-attn-fusedbwd @d2ac636) failed on drain
mechanism, not on fit: per-stage fp32 float2 atomics from consumer registers
multiplied dQ write traffic into L2-serialized atomic transactions
(~132MB/layer @s4096). FA4 drains dQ per-m-block via zero-atomic
`cp.reduce.async.bulk.add.f32` from a smem stage. Both primitive blockers are
now closed locally:

- 1-D f32 bulk reduce parity-clean (`cp-reduce-1d-unit-probe-2a41f6a.md`,
  SASS `UBLKRED.G.S.ADD.F32.RN`).
- multi-CTA collision (many CTAs reduce-adding the SAME rows) exact:
  `results/cp-reduce-collision-2a41f6a-k8s-20260707T0800Z.log` (max_err=0 at
  chunks 2/4/8, strides 64/512, both wait modes). One-pass needs exactly
  this: every kv-tile CTA reduces into the same dQ rows.

## (a) Register map — written before building

Structure: dkv-shaped one-pass. CTA owns 128 kv rows (64/WG, persistent
K/V in smem), streams 64-row Q/dO stages from the diagonal, per stage:

```
1. wga_mma64_x2 (SS):  s  = Q_stage K_wg^T     (S)      \ one commit
                       s2 = dO_stage V_wg^T    (dP)     /
2. ALU: p = exp2(s*scale*log2e - lse*log2e)  [exp2 prebias, fp32 p]
        P bf16 -> smem (MN-view A for dV)
        ds = p*(s2 - dr)*scale -> bf16 -> smem dS (MN-view A for dK)
                                       -> areg[16] (RS K-major A for dQ)
3. fence.proxy.async + consumer_sync
4. one commit batch:   dv += P^T dO_stage      (SS MN/MN)
                       dk += dS^T Q_stage      (SS MN/MN)
                       dqp = dS K_wg           (RS from areg, B = K_wg MN-view)
5. drain dqp -> gmem dQaccum (variants below); consumer_sync -> next stage
```

Fragment-layout compatibility (the flagged risk): dS is needed BOTH as
MN-view A (dK) and as RS-A (dQ). In OUR orientation the S/dP accumulator has
M = q-stage rows (dkv computes `s` with rows = q, cols = own kv), which is
exactly the orientation `MK_ATTN_DQ_RS_FEED` already exploits: the C-fragment
layout IS the K-major A fragment for a q-rows-as-M GEMM. So dS goes to smem
(for dK) AND to areg (for dQ) — both consumers are fed, no conflict. FA4 has
the mirrored trade (SdP_swapAB: accumulator M = kv): dV/dK are RS-fed, dQ
needs the smem round trip, and q-row stats land in the accumulator N-dim —
THAT is why FA4 needs the LSE/dPsum quad-shuffle at tile_m=128. Our M = q
orientation keeps LSE/Drow quad-local (2 scalars/thread from the smem stage
prefetch): the shuffle trick buys us nothing and is NOT needed.

Per-consumer-thread register account (m64n64 fragment = 32 f32):

| bank | regs | lifetime |
|---|---|---|
| dk[32], dv[32] | 64 | persistent across Q loop |
| s[32], s2[32] | 64 | stage phase A (S/dP + softmax/dS ALU) |
| areg[16] (u32 bf16x2 dS) | 16 | built in phase A ALU, read by RS in phase B |
| dqp[32] | 32 | phase B only — reuses dead s/s2 bank |
| lse/dr scalars | ~4 | 2 q-rows/thread, from smem prefetch |
| addressing/desc/loop overhead | ~40 | (= dkv's measured 168 - 128 banks) |

Peak phase A: 128 + partial areg + overhead ~= 168-184.
Peak phase B: 64 + 16 + 32 + overhead ~= 152.
Prediction: **~170-190 regs — fits the 224 pdf-consumer budget and the
256-thread/255 image; no spill expected; LSE-shuffle unnecessary.**
(The old x3 refutation also fit — "no register-wall crossing" — consistent.)

Smem: dkv layout 97KB (K/V/P/dS owned + Q/dO 2-stage ring + LSE/Drow stages)
+ dQ drain slabs 2x16KB (per-WG, bulk variant) = ~129KB. Fine standalone
(227KB cap); OVER the megakernel 100KB smem page — integration would need a
single cross-WG-summed 16KB slab (113KB, still over), a Q/dO ring diet, P/dS
overlay, or a page-size change. Deferred to the integration claim.

Drain variants probed:
- mode 0 (baseline = old refutation mechanism): per-thread float2 fp32
  atomics direct from dqp regs.
- mode 1 (FA4-style): stage dqp into a per-WG smem slab, ONE
  `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` per WG per
  stage into a contiguous [nq, S, D] fp32 dQaccum; `cp.async.bulk.wait_group`
  before slab reuse (one stage of flight). Slab uses a fragment-interleaved
  internal order ([pair][thread] float2) so the smem stage is bank-conflict
  free — natural [64][64] row-major staging would be 8-way conflicted (bank =
  col%32 since row stride 256B ≡ 0 mod 128B); the gmem block order is
  unscrambled host-side (and by the future ws convert op in-model).

Traffic account (s8192, nq=4): one-pass dQ reduce traffic = one 16KB partial
per (kv-tile, stage, WG) = ~66MB/head x nq=4 ~= 266MB/layer into an 8MB
L2-resident dQaccum; FA4 pays the identical volume (its 128-row m-blocks x
128-kv CTAs emit the same S^2*D*4/128 bytes/head). The old refutation paid
this as ~2K 8B L2 atomic transactions per WG-stage; mode 1 pays it as ONE
bulk op (512 32B L2 reduce sectors) per WG-stage with zero consumer issue
cost. In exchange one-pass DELETES the dq pass's S+dP recompute (2 of 7
GEMM-units) and its K/V re-streaming.

## (b) Probe

`experiments/fused-training-megakernel/onepass_bwd_probe.{cu,py}` (worktree
above): standalone torch extension, plain launches, 256 threads = 2 consumer
WGs, dynamic smem, attention_probe.cu conventions. Kernels:

- `attn_dkv2` / `attn_dq2`: two-pass baseline bodies reproduced at
  current-op feature level (exp2 prebias; dkv LSE/Drow smem prefetch; dq RS
  feed + fp32-P + C=1 direct store) so the comparison isolates the one-pass
  structure, not promoted micro-wins.
- `attn_onepass<DQ_RS, DRAIN>`: the structure above; DQ_RS toggles areg-RS vs
  smem-dS (K-view A) for the dQ GEMM; DRAIN = atomics vs bulk-reduce.
- GQA (nq=4, nkv=2): one group member per tile (g in tile id), dK/dV fp32
  atomics into ws kv columns as today; causal masking identical to dkv.

Parity: vs torch autograd at S=2048/4096 (B=1, nq=4, nkv=2, D=64), C in
{1,2}, all 4 variant combos; dQaccum unscrambled host-side for mode 1.

## (c) Results

### Register fit (cuobjdump -res-usage, sm_90a, __maxnreg__(224))

| kernel | REG | STACK/LOCAL |
|---|---|---|
| attn_onepass<rs=1, atomic> | 202 | 0/0 |
| attn_onepass<rs=1, bulk> | 208 | 0/0 |
| attn_onepass<rs=0, atomic> | 203 | 0/0 |
| attn_onepass<rs=0, bulk> | 209 | 0/0 |
| attn_dkv2 (baseline) | 167 | 0/0 |
| attn_dq2 (baseline, RS feed) | 133 | 0/0 |

**Fits.** 202-209 <= 224 (pdf consumer budget), zero spill, ~+40 over dkv2
(the paper map predicted 170-190; ptxas holds a few extra live banks but
never crosses the wall). No LSE-shuffle needed, as predicted by the
orientation argument. `UBLKRED.G.S.ADD.F32.RN` present in both bulk-drain
instantiations (2 in SASS).

### Parity (GPU 3, H100; logs results/onepass-bwd-parity-b1d36305-20260707.log)

All clean at S=2048/4096, C=1/2, all four variants: dK/dV/dQ max_abs_err
3.4e-04..3.4e-03 vs fp32 autograd — SAME error level as the two-pass
baseline. Bulk-reduce drain errors are IDENTICAL to atomic-drain errors:
in-kernel multi-CTA reduce-add collisions are exact at real scale (32-64
kv-tile CTAs + chunks all reducing the same dQ rows).

### Standalone timing (GPU 3, median-of-50 CUDA events, paired same-GPU,
### idle window checked before/after; two runs agree within ~1us)

| S | dkv2 best | dq2 best | two-pass total | one-pass best | speedup | TF/s 1p vs 2p |
|---|---|---|---|---|---|---|
| 4096 | 114.9 (C2) | 92.8 (C2) | 207.7us | **147.8us** (rs1 bulk C2) | **1.41x** | 145 vs 103 |
| 8192 | 303.8 (C1) | 269.4 (C2) | 573.2us | **437.1us** (rs1 bulk C1) | **1.31x** | 197 vs 150 |

Repeat run: 148.6/208.3 = 1.40x @s4096; 434.8/572.6 = 1.32x @s8192.

Variant findings:

- **Drain mechanism is the story, confirmed**: bulk-reduce beats per-thread
  float2 atomics by ~8% (s4096 147.8 vs 159.8; s8192 437.1 vs 459.4). BUT
  even the atomic one-pass now wins standalone (1.25-1.30x) — the old
  refutation's loss also needed the in-model context (band overlap; and it
  scattered into the interleaved [S,stride] ws rather than per-stage
  contiguous blocks).
- RS vs smem-dS dQ feed: a wash (rs1 434.8-437.1 vs rs0 440.8-441.2 @s8192;
  within ~1% at s4096). The RS feed's smem-read saving is small because dS
  must hit smem anyway for dK.
- GEMM-unit accounting: two-pass 7 units -> 573.2us = 81.9us/unit @s8192;
  one-pass 5 units at that rate would be 409.5us; measured 437.1us => drain +
  extra sync overhead ~28us (6.7%), capturing ~94% of the theoretical 1.40x.
  At s4096 capture is ~100% (5/7 x 207.7 = 148.4 ~= 147.8 measured).
- Best C flips with shape as expected (s4096 C=2 = 256 CTAs for wave balance;
  s8192 C=1 = 256 CTAs already 2 waves).

Caveat: standalone two-pass here (573us, 150 TF/s @s8192) is slower than the
in-model banded pair (~200 TF/s) because plain launches serialize the two
passes and forgo band overlap; this comparison isolates STRUCTURE at equal
harness conditions. The model-level gain from one-pass is bounded by the
wait-column/critical-path structure, not by this ratio.

## (d) Verdict: GO for a model-integration claim

The one-pass D64 structure **fits and wins standalone**: REG 208/224 no
spill, parity clean, 1.41x/1.31x over two-pass at s4096/s8192 with the
FA4-style bulk-reduce drain. The 2853e0de-era refutation is mechanism-dead:
its loss was the atomic drain + in-model scatter, not the fusion.

What a future integration claim needs (NOT done here, per sequencing):

1. **Smem budget**: probe uses 97KB + 2x16KB slabs = 129KB > 100KB page.
   In-model options, in order of preference: (i) overlay the drain slab on
   the (P+dS)[wg] 16KB region — P/dS of stage t are dead after the phase-B
   wait<0>; the slab write lands there and the NEXT stage's P/dS ALU write
   waits `cp.async.bulk.wait_group.read` (still one full stage of drain
   flight) => stays ~97KB; (ii) atomic-drain variant needs NO extra smem and
   still wins standalone — low-risk fallback; (iii) page growth.
2. **dQaccum contract**: contiguous [nq, S, D] fp32 workspace with the
   fragment-interleaved intra-block order (bank-conflict-free staging); the
   dq fp32->bf16 convert op must read that order (index map in
   onepass_bwd_probe.py::unscramble_map), or pay the 8-way-conflict staging
   cost for natural order — measure there, not here.
3. **Band scheduling**: one-pass replaces BOTH band families (ATTN_DKV_WG +
   ATTN_DQ_WG) with one kv-parallel family; LPT/band-order/C-chunking must be
   re-derived (the two-band overlap that currently hides dq waits disappears,
   which is also where the model-level win may exceed OR undershoot the
   standalone ratio).
4. **Executor**: 256-thread consumer image is enough (REG 208 <= 224 means it
   also fits pdf consumers with the producer WG left free — an attention
   TMA-feed/drain producer is the natural composition with the
   producer-offload lane, but is NOT required for the win).
5. Re-verify vs the cp-reduce DQ drain lane's verdict (two-pass + drained dq)
   — if that lands first, the integration baseline changes.

Files: experiments/fused-training-megakernel/onepass_bwd_probe.{cu,py};
logs results/onepass-bwd-{parity,timing,timing2}-b1d36305-20260707.log.
