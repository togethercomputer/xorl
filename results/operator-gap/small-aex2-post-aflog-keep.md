# Small attention exp2 post-fast-log keep

Date: 2026-07-06
Head: `d1bf366` (`megakernel: record small dq float2 check`)
Scope: source-free H512/S1024 small check after the exact small attention
fast-log default changed to precise `logf`.

## Candidate

Forced the old precise exp path:

`MK_ATTN_EXP2_APPROX=0`

The current default keeps `_aex2` enabled for H512/S1024 small. Positive
default-minus-precise means precise exp is faster.

## Route

Forced precise compiled the expected extension without `_aex2`:

`xorl_megakernel_adkva_adkvf2_drowst_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `ATTN_FWD_WG=8/512`
- `ATTN_DKV_WG=8/512`
- `ATTN_DQ_WG=8/512`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=-1.90734863e-06`, worst selected grad `kn.0`,
  relative error `4.047775e-07`.
- precise-first: `loss_diff=-9.53674316e-07`, worst selected grad `kn.0`,
  relative error `5.397035e-07`.

## Timing

Forced precise exp lost both construction orders:

| Order | default | precise | default-minus-precise | precise wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3264.43us` | `3292.58us` | `-28.80us` | `3/80` |
| precise-first | `3259.60us` | `3287.26us` | `-25.66us` | `3/80` |

## Decision

Keep `_aex2` enabled for H512/S1024 small after the precise-log fast-log retune.

Log: `results/mkv3-p4b-small-aex2-post-aflog-20260706T1712Z.log`
