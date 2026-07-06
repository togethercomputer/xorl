# Small dKV float2 post-direct-BF16 keep

Date: 2026-07-06
Head: `f037320` (`megakernel: record post-direct rms r4 check`)
Scope: compile-flag H512/S1024 small forced-old recheck after NN/NT n128 retunes
and the direct-BF16 GEMM epilogue promotion.

## Candidate

Rechecked the current `OP_ATTN_DKV_WG` direct-atomic float2 default by forcing
the old scalar direct-atomic epilogue:

`MK_ATTN_DKV_FLOAT2_ATOMIC=0`

Positive default-minus-scalar means the scalar old path is faster.

## Extension

The default path loaded:

`xorl_megakernel_adkva_adkvf2_drowst_aflog_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

The forced-old scalar path loaded:

`xorl_megakernel_adkva_drowst_aflog_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

## Route

Route shape was unchanged for default and scalar:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `ATTN_FWD_WG=8/512`
- `ATTN_DKV_WG=8/512`
- `ATTN_DQ_WG=8/512`

## Parity

Selected-gradient parity stayed clean in both construction orders:

- default-first: `loss_diff=+9.53674316e-07`, worst selected grad `kn.0`,
  relative error `6.298477e-07`.
- scalar-first: `loss_diff=-9.53674316e-07`, worst selected grad `kn.0`,
  relative error `4.498914e-07`.

## Timing

The scalar old path lost both construction orders:

| Order | default | scalar | default-minus-scalar | scalar wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3268.38us` | `3300.96us` | `-30.77us` | `1/80` |
| scalar-first | `3263.30us` | `3296.14us` | `-28.56us` | `0/80` |

## Decision

Keep the broad D64 WGMMA dKV float2 direct-atomic default. It remains
load-bearing for H512/S1024 small after the direct-BF16 promotion.

Log: `results/mkv3-p4b-small-dkvf2-post-directbf16-20260706T1659Z.log`
