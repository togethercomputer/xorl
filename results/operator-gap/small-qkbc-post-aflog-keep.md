# Small qknorm D64 cache post-fast-log keep

Date: 2026-07-06
Head: `ae3f311` (`megakernel: record small attention exp2 check`)
Scope: source-free H512/S1024 small check after the exact small attention
fast-log default changed to precise `logf`.

## Candidate

Forced the old generic qknorm-bwd loop:

`MK_QKBWD_D64_CACHE=0`

The current default keeps `_qkbc` enabled for H512/S1024 small. Positive
default-minus-old means the old loop is faster.

## Route

Forced old compiled the expected extension without `_qkbc`:

`xorl_megakernel_adkva_adkvf2_drowst_aex2_lex2_ceb2_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `QKNORM_ROPE_BWD=8/1024`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=-3.81469727e-06`, worst selected grad `kn.0`,
  relative error `1.030265e-02`.
- old-first: `loss_diff=+0.00000000e+00`, worst selected grad `kn.0`,
  relative error `1.030244e-02`.

## Timing

Forced old lost both construction orders:

| Order | default | old | default-minus-old | old wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3283.33us` | `3305.70us` | `-29.25us` | `2/80` |
| old-first | `3260.90us` | `3287.15us` | `-26.21us` | `1/80` |

## Decision

Keep `_qkbc` enabled for H512/S1024 small.

Log: `results/mkv3-p4b-small-qkbc-post-aflog-20260706T1715Z.log`
