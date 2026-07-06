# Small CE backward exp2 post-adkva keep

Date: 2026-07-06
Head: `7a671b7` (`megakernel: record small dkv atomic check`)
Scope: source-free H512/S1024 small recheck after exact small attention
precise-log, direct-BF16 epilogues, qknorm D64 cache, lm-head exp2, Drow
direct-store, and attention dKV direct-atomic confirmations.

## Candidate

Forced the old precise CE backward exponential path:

`MK_CE_BWD_EXP2_APPROX=0`

The current default keeps `_ceb2` enabled and uses `ex2.approx.ftz.f32` inside
`OP_CE_BWD`. Positive default-minus-precise means the precise path is faster.

## Route

Forced precise compiled the expected extension without `_ceb2`:

`xorl_megakernel_adkva_adkvf2_drowst_aex2_lex2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `waves=144`
- `CE_FWD=1/1024`
- `CE_BWD=1/1024`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=-9.53674316e-07`, worst selected grad `kn.0`,
  relative error `1.125192e-02`.
- precise-first: `loss_diff=+3.81469727e-06`, worst selected grad `kn.0`,
  relative error `1.125152e-02`.

## Timing

Forced precise lost both construction orders:

| Order | default | precise | default-minus-precise | precise wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3267.33us` | `3282.10us` | `-13.98us` | `11/80` |
| precise-first | `3262.19us` | `3274.90us` | `-12.46us` | `8/80` |

## Decision

Keep `_ceb2` enabled for H512/S1024 small. The current-stack margin is smaller
than the original promotion check, but the precise path still loses both orders
with clean parity.

Log: `results/mkv3-p4b-small-ceb2-post-adkva-20260706T1734Z.log`
