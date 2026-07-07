# Fwd exp-arg scale fold + deferred row-sum (FA4-fwd softmax ALU cuts)

Session b1d36305, 2026-07-07. Lane: forward-attention ALU cuts (claim
20260707T0750Z-realclock). Worktree
`/home/apanda/xorl-oss/.claude/worktrees/agent-af00f09c6ddce1cf0`, branch
`fwd-afexp-b1d36305` @ 2a41f6a, commit 87fdbc1.

## Mechanism

Two default-off build flags for `op_attn_fwd_wg` (D64 wg fwd, non-pipe body),
env-only knobs in `mk.load_ext`, extension suffix `_afexp{e,r,er}`; flag-off is
bit-identical by construction (all device changes `#ifdef`-guarded).

1. `MK_ATTN_FWD_EXPFOLD=1` (FA4 `softmax.py::online_softmax` semantics): the
   online running max is tracked in the scale*log2e domain. Mask loop keeps RAW
   scores (per-element `s*scale` FMUL deleted); `m = max(m, rmax*scale_l2)`
   after the (still per-stage) max quad-reduce; `p = ex2.approx(fmaf(s,
   scale_l2, -m))` (one FMA per element, no log2e multiply);
   `alpha = ex2(m_old - m_new)`. Epilogues convert back to natural log:
   `LSE = fmaf(m, ln2, log(l))`; band partial `Mpart = m*ln2` so
   OP_ATTN_COMBINE (natural-domain `l_c*expf(m_c - m*)`) is untouched.
2. `MK_ATTN_FWD_DEFER_RSUM=1` (FA4 deferred row-sum): the two per-stage
   `shfl_xor` row-sum reduces are dropped; `l` stays a per-thread partial
   through the stage loop (the alpha rescale is quad-uniform because the max IS
   quad-reduced each stage, so partials stay linear); ONE quad reduce after the
   post-loop `warpgroup_wait<0>`, before inv/LSE/partial stores.

## SASS / res-usage (megakernel_ws copy of `_Z14op_attn_fwd_wg`, nano-flavor images)

| image | fwd body instrs | FADD | FFMA | FMUL | EX2 | SHFL | frame | kernel REG/STACK |
|---|---|---|---|---|---|---|---|---|
| off    | 1151 | 74 | 8  | 132 | 34 | 8 | 0x40 | 168/64 |
| expfold| 1092 | 42 | 40 | 73  | 34 | 8 | 0x40 | 168/64 |
| defer  | 1151 | 74 | 8  | 132 | 34 | 8 | 0x40 | 168/64 |
| both   | 1093 | 42 | 40 | 73  | 34 | 8 | 0x40 | 168/64 |

expfold: −59 static instrs (the fold is real). defer: static-neutral — 4 shfl
move from the dynamic stage loop to the epilogue; the saving is 4 shfl + the
dependent adds per 64-row stage, DYNAMIC only. No register/stack change on any
executor image.

## Timing (paired alternating A/B, both construction orders, GPU 3, flock'd)

H256 L4 nq4 nkv2 D64 whole-step medians, delta = flag-on − flag-off (negative
= faster). s8192/s4096 24/40 reps per order; nano/small 40.

| shape | EXPFOLD (E) | DEFER_RSUM (R) | both (ER) |
|---|---|---|---|
| s8192 | **+47.5 / +43.4us** (1/24, 2/24) | **−36.5 / −47.0us** (23/24, 24/24) | +5.0 / −7.9us (7/24, 20/24) |
| s4096 | +1.3 / −2.7us (22/40, 24/40) | **−10.1 / −7.3us** (39/40, 36/40) | −11.7 / −9.5us (39/40, 37/40) |
| nano  | −3.1 / −5.3us | −1.5 / −4.1us | −5.1 / −7.2us (37/40, 40/40) |
| small | −16.3 / −1.2us | −5.7 / −2.0us | −18.2 / −12.5us (33/40, 32/40) |

Effects are additive: at s8192 the expfold regression (~+45us) cancels the
defer win (~−42us) in ER. At s4096 and below expfold is neutral-to-slightly
positive, so ER ≈ R there.

The expfold s8192 regression is the surprise: −59 static ALU in an op that is
42.6% exp-dominated, yet +45us on-path. EX2 count is unchanged (34) — the fwd
p-loop is MUFU-throughput/latency-bound, not FMUL-issue-bound, and the deleted
FADD/FMULs were apparently free co-issue work; the reshaped FFMA→EX2 dependency
chain schedules worse in the banded C=64 regime. Consistent with the
wait-column principle: static ALU cuts don't pay when the unit bottleneck
(MUFU) is untouched.

## Numerics / parity

- test_ops.py: ALL PASSED flag-off, E, R, ER (wgmma fwd O max_abs_err
  2.239e-03 in all four — same worst element).
- test_model.py fp32-ref bars (nano + D128 cfgs, assert <0.03), worst grad:
  flag-off kn.3 0.0285 / kn.0 0.0179; E kn.2 0.0224 / kn.0 0.0180;
  R qn.3 0.0212 / kn.0 0.0178; ER kn.2 0.0210 / kn.0 0.0190. All flag-on runs
  BELOW the flag-off baseline; rerun/waves/df2/ws agreement + training sanity
  green in all.
- Banded long-S partial epilogue: env_ab parity asserts green at s4096 (band
  T=22) and s8192 (T=64): worst flag-on-vs-off grad rel 0.008–0.013, loss
  delta ≤5e-5.
- Forced-band nano (MK_ATTN_FWD_BAND=2) is a BROKEN CONTROL in stock code:
  flag-off fails the 0.03 bar (kn.3 0.0508) and drifts waves-vs-df loss by
  1.23e-3 (bar 1e-4) — pre-existing, flag-independent (probed flag-off).
  Flag-on forced-band numbers sit inside that envelope: E 0.0414, R 0.0242,
  ER 0.0312. Worth a separate look outside this lane.

## Verdict

- `MK_ATTN_FWD_DEFER_RSUM`: **promote**. Wins everywhere measured, biggest at
  s8192 (−36/−47us ≈ 0.65% step) and s4096 (−10/−7us); no numerics cost; no
  register cost.
- `MK_ATTN_FWD_EXPFOLD`: **no-go at s8192** (+45us), neutral s4096, mildly
  positive nano/small — not worth a default anywhere given the long-S harm;
  keep the flag for future re-test if the fwd body/feed changes (producer-WG
  fwd, 3-WG image).

## Landing checklist (main session lands; shared tree untouched by this lane)

1. Cherry-pick / merge commit 87fdbc1 from branch `fwd-afexp-b1d36305`.
2. Gate: default `MK_ATTN_FWD_DEFER_RSUM=1` for ALL D64 wg-fwd shapes — either
   flip the `load_ext` env default to "1" or (matching house style) add a
   model.py default kwarg like the other attn knobs. No shape table needed:
   R is ≥0 at nano/small/s4096/s8192 in both orders. Leave
   `MK_ATTN_FWD_EXPFOLD` default 0 (documented no-go at s8192).
3. Keep the `#error` guards vs `MK_ATTN_PIPE` (pipe body not implemented).
4. Re-cert scoreboard after gating (expected ~−40us at s8192, ~−9us s4096).
5. Extension-name note: gating R by default changes the default image name
   (`_afexpr`) unless the suffix is dropped when it becomes the default.

Repro: /tmp scratch runner (copy of results/env_ab_main.py pointed at the lane
worktree) — `fwd_afexp_ab_b1d36305.py <shape> <order>
MK_ATTN_FWD_DEFER_RSUM=1 [reps]` with TORCH_EXTENSIONS_DIR
/tmp/torch-ext-b1d36305-afexp, CUDA_VISIBLE_DEVICES=3.
