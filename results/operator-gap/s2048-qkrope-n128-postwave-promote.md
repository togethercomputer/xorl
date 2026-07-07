# s2048 fused-qkrope n128 promotion (post-wave resweep flip)

Session b603d819, 2026-07-07. Route-knob promotion out of the post-pdf-wave
R4 resweep; the third resweep-law flip at this boundary family.

## History of the boundary

- Original promotion (2026-07-06 ~2046Z): qkrope n128 promoted for exact
  H256/D64 S3072/S4096/S8192; **S2048 stayed wg64** — default-first looked
  faster (-8.8, 40/40) but variant-first regressed (+4.3, 13/40): order-mixed.
- Structural change at s2048: pdf-d64 feed `d4758a3` (+ the wider
  pdf/dq-feed/prebias wave).
- Post-wave screening (dedicated k8s H100 @6b137f5, zero co-tenants):
  MK_WGMMA_N128_QKROPE=1 at s2048 **-13.9us paired, 40/40**.
- 120-rep escalation, both orders: **-13.7us 119/120 (default-first),
  -11.1us 114/120 (variant-first)**, parity clean. Absolutes in-band
  (defaults 1793-1795us, matching the 6b137f5 certification's 1810.7 within
  window drift).

## What changed

`_H256_D64_QKROPE_N128_S`: (3072, 4096, 8192) -> (2048, 3072, 4096, 8192).
Routing-only — the four s2048 `GEMMNT 2048x512x256.wg.+qkrope` rows move from
wg64 to the n128+qkrope epilogue body that already compiles in these flavors
(unit case in test_ops since the original promotion). No new define, no
res-usage/LD surface.

## Siblings from the same screening (recorded no-gos)

- s2048 fwd-band T20: -6.3 99/120 then +0.2 60/120 — order-collapsed, T16
  stands.
- s2048 idle64: -6.1 104/120 then +11.2 8/120 — order-flipped, idle32 stands
  (independently agrees with the parallel scheduler-knob session's refutation
  of its own s2048 idle64 triage candidate).
- All s3072/s4096/s8192 knobs in the 18-window screening: clean NOs — the
  post-wave defaults there are stale-free (bands +5..+479, dq_first +56).

## Promotion battery at the anchor (2a41f6a + this gate)

Recorded by the landing run in
`results/mkv3-p4b-s2048-qkrope-promote-k8s-*.log`: route guard (four rows
n128+qkrope at 256 tiles each under the promoted default; forced-old
`MK_WGMMA_N128_QKROPE=0` restores wg64 at 512 tiles), promoted-vs-forced-old
both orders, `test_model.py`, py_compile/ruff/diff-check.
