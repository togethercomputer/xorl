# Small Drow direct-store post-lex2 keep

Date: 2026-07-06
Head: `74aebc8` (`megakernel: record small lmhead exp2 check`)
Scope: source-free H512/S1024 small recheck after exact small attention
precise-log, direct-BF16 epilogues, qknorm D64 cache, SwiGLU FMA confirmation,
and lm-head exp2 confirmation.

## Candidate

Forced the old atomic Drow epilogue:

`MK_DROW_DIRECT_STORE=0`

The current default keeps `_drowst` enabled and directly stores the fused
`dOatt = dX @ Wo` Drow epilogue when D=64 makes every Drow element single-writer.
Positive default-minus-old means the old atomic path is faster.

## Route

Forced old compiled the expected extension without `_drowst`:

`xorl_megakernel_adkva_adkvf2_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `waves=144`
- `drow_gemm=8/512`
- `gemm=99/13856`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=-4.76837158e-06`, worst selected grad `w1.0`,
  relative error `4.674141e-07`.
- old-first: `loss_diff=+2.86102295e-06`, worst selected grad `w1.0`, relative
  error `4.674144e-07`.

## Timing

Forced old lost both construction orders:

| Order | default | old | default-minus-old | old wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3272.34us` | `3319.18us` | `-46.56us` | `0/80` |
| old-first | `3261.04us` | `3309.57us` | `-49.86us` | `0/80` |

## Decision

Keep `_drowst` enabled for H512/S1024 small. The direct-store Drow epilogue win
remains stronger on the current stack than in the original promotion check.

Log: `results/mkv3-p4b-small-drowst-post-lex2-20260706T1727Z.log`
