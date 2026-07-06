# Small attention dKV direct-atomic post-drowst keep

Date: 2026-07-06
Head: `fc9b1dd` (`megakernel: record small drow store check`)
Scope: source-free H512/S1024 small recheck after exact small attention
precise-log, direct-BF16 epilogues, qknorm D64 cache, lm-head exp2, and Drow
direct-store confirmations.

## Candidate

Forced the old smem-drain attention dKV epilogue:

`MK_ATTN_DKV_DIRECT_ATOMIC=0`

The current default keeps `_adkva` enabled. Because float2 direct atomics depend
on direct atomics, the forced-old extension also drops `_adkvf2`. Positive
default-minus-old means the old smem-drain path is faster.

## Route

Forced old compiled the expected extension without `_adkva` or `_adkvf2`:

`xorl_megakernel_drowst_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `waves=144`
- `ATTN_FWD_WG=8/512`
- `ATTN_DKV_WG=8/512`
- `ATTN_DQ_WG=8/512`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=-1.90734863e-06`, worst selected grad `qn.0`,
  relative error `9.697730e-07`.
- old-first: `loss_diff=+2.86102295e-06`, worst selected grad `qn.0`, relative
  error `4.408058e-07`.

## Timing

Forced old lost both construction orders:

| Order | default | old | default-minus-old | old wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3268.88us` | `3363.15us` | `-91.33us` | `0/80` |
| old-first | `3260.13us` | `3354.80us` | `-94.29us` | `0/80` |

## Decision

Keep `_adkva` and `_adkvf2` enabled for H512/S1024 small. The pre-direct
smem-drain epilogue is a larger regression on the current stack than in the
original recheck.

Log: `results/mkv3-p4b-small-adkva-post-drowst-20260706T1731Z.log`
