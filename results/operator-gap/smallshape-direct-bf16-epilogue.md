# Small-shape direct BF16 GEMM epilogue — exact S128 and H512/S1024 promotion (2026-07-06)

## Post-n128 update

The original verdict below was exact S128-only. That is superseded for one more
shape after the later H512/S1024 NN/NT n128 retunes moved 32 BF16-output WGMMA
rows (`5632` tiles) back onto the ordinary WGMMA body. A current-head recheck at
`2276fd7` promoted exact H512/L8/S1024 small as well:

- forced direct source-free A/B: `+27.82us` and `+27.01us`
  default-minus-direct, direct wins `80/80` and `80/80`.
- promoted default vs forced old: default beat old by `28.82us` and `23.71us`,
  with old wins `1/80` and `2/80`.
- refreshed promoted small profile: `3320.1us`.
- refreshed promoted small score: megakernel `3312.5us` vs
  compile+CUDAGraph+ `1903.3us`.

Detail note: `results/operator-gap/small-directbf16-post-n128-promote.md`.

Scratch source: `/home/apanda/xorl-oss-direct-epilogue`

Main source base: `5b8b3db` plus the direct-epilogue patch.

## Question

The small-shape WGMMA GEMM path stages fp32 accumulators through shared memory
before writing BF16 outputs. For plain BF16 stores with no residual, no fp32
output, no split-K, and no side reductions, the epilogue can write
`__nv_bfloat162` pairs directly from registers and skip the shared-memory drain.

The probe added a compile-time `MK_GEMM_DIRECT_BF16_EPILOGUE` flag and only
lets `MKQwen3` default it on for the exact S128 D64 shape:

`c.D == 64 and c.S == 128`

The CUDA fast path is further masked to plain BF16 GEMMs:

`!(flags & (4 | 8 | 16 | 32 | 256 | 1024 | 2048 | 8192))`

## Result

Post-`5b8b3db` validation on GPU5:

- `results/mkv3-p4b-direct-bf16-main-post5b-test-ops-20260706T1016Z.log`:
  `ALL OP TESTS PASSED`.
- `results/mkv3-p4b-direct-bf16-main-post5b-test-model-20260706T1016Z.log`:
  `ALL MODEL TESTS PASSED`.
- `results/mkv3-p4b-direct-bf16-main-s128-newdef-vs-oldforce-20260706T1004Z.log`:
  S128 env-unset new default vs force-off old path, parity clean
  (`loss_delta=-2.861e-06`, worst grad rel `1.953125e-03`), median
  `-1.07us`, mean `-0.05us`, wins `33/60`.
- `results/mkv3-p4b-direct-bf16-main-s128-newdef-first-20260706T1004Z.log`:
  opposite order, parity clean (`loss_delta=-1.907e-06`), median `-0.35us`,
  mean `-0.30us`, wins `41/80`.

Earlier scratch S128 clean runs were stronger:

- `results/mkv3-p4b-direct-bf16-epilogue-s128-default-first-clean-20260706T0941Z.log`:
  median `-5.78us`, mean `-5.70us`, wins `58/80`.
- `results/mkv3-p4b-direct-bf16-epilogue-s128-direct-first-clean-20260706T0941Z.log`:
  median `-4.87us`, mean `-4.45us`, wins `64/80`.
- `results/mkv3-p4b-direct-bf16-epilogue-s128-newdef-vs-oldforce-20260706T0941Z.log`:
  env-unset new default vs force-off old path, median `-2.76us`, mean
  `-2.95us`, wins `43/60`.

## Non-promoted Shapes

The broader gate is not supported:

| Shape | Evidence | Read |
|---|---:|---|
| S256 | `+5.55us` default-first, `-10.96us` direct-first | order dominated / inconclusive |
| S512 | `-2.22us` and `-2.24us` medians | weak positive, not enough for broad gate |
| H512/L8/S1024 small | `+7.49us` and `+1.84us` medians | loses or neutral |

## Original Verdict

Promote only the exact S128 D64 default. Keep the environment override so
`MK_GEMM_DIRECT_BF16_EPILOGUE=0` restores the old shared-memory epilogue and
`=1` forces the direct path for A/B testing.

Do not broaden this to S256, S512, or the real H512/L8/S1024 small shape without
new evidence. The mechanism is correct but the final main-HEAD in-model timing
gain is small; it is a narrow S128 cleanup, not a general GEMM-gap lever.
