# Small lm-head exp2 post-swfma keep

Date: 2026-07-06
Head: `0b71aaf` (`megakernel: record small swiglu fma check`)
Scope: source-free H512/S1024 small recheck after exact small attention
precise-log, direct-BF16 epilogues, qknorm D64 cache, and the post-qkbc SwiGLU
FMA confirmation.

## Candidate

Forced the old precise lm-head CE-partial exponential path:

`MK_LMHEAD_EXP2_APPROX=0`

The current default keeps `_lex2` enabled and uses `ex2.approx.ftz.f32` in the
fused lm-head GEMM's CE/LSE partial epilogue. Positive default-minus-precise
means the precise path is faster.

## Route

Forced precise compiled the expected extension without `_lex2`:

`xorl_megakernel_adkva_adkvf2_drowst_aex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `waves=144`
- `lmhead_gemm=1/1024`
- `gemm=99/13856`
- `ce_fwd=1/1024`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=-2.86102295e-06`, worst selected grad `kn.0`,
  relative error `4.497528e-07`.
- precise-first: `loss_diff=+5.72204590e-06`, worst selected grad `qn.0`,
  relative error `3.306046e-07`.

## Timing

Forced precise lost both construction orders:

| Order | default | precise | default-minus-precise | precise wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3271.20us` | `3341.79us` | `-68.45us` | `0/80` |
| precise-first | `3259.74us` | `3329.10us` | `-67.42us` | `0/80` |

## Decision

Keep `_lex2` enabled for H512/S1024 small. The earlier lm-head exp2 win remains
load-bearing on the current stack.

Log: `results/mkv3-p4b-small-lex2-post-swfma-20260706T1728Z.log`
