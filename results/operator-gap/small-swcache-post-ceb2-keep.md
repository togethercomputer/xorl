# Small SwiGLU cache-off post-ceb2 keep

Date: 2026-07-06
Head: `8be1965` (`megakernel: record small ce backward exp2 check`)
Scope: source-free H512/S1024 small recheck after exact small attention
precise-log, direct-BF16 epilogues, qknorm D64 cache, lm-head exp2, Drow
direct-store, attention dKV direct-atomic, and CE backward exp2 confirmations.

## Candidate

Forced cached sigmoid back on:

`MK_SWIGLU_CACHE_SIG=1`

The current default keeps sigmoid cache off while using `SWIGLU_BWD_4W`. Positive
default-minus-cache means the cached path is faster.

## Route

Forced cache compiled the expected extension with `_swcsig`:

`xorl_megakernel_adkva_adkvf2_drowst_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_swcsig_gmbar_gdbf16`

Route shape was unchanged except for `swsig` allocation:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `waves=144`
- default: `swsig=False`
- cache: `swsig=True`
- `SWIGLU_FWD=8/1024`
- `SWIGLU_BWD_4W=8/4096`

## Parity

Selected-gradient parity stayed clean:

- default-first: `loss_diff=+2.86102295e-06`, worst selected grad `kn.0`,
  relative error `1.376691e-02`.
- cache-first: `loss_diff=+1.90734863e-06`, worst selected grad `kn.0`,
  relative error `1.376718e-02`.

## Timing

Forced cache lost both construction orders:

| Order | default | cache | default-minus-cache | cache wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3265.97us` | `3282.96us` | `-17.70us` | `5/80` |
| cache-first | `3263.62us` | `3282.26us` | `-18.00us` | `4/80` |

## Decision

Keep sigmoid cache off for H512/S1024 small with `SWIGLU_BWD_4W`. The cache-on
route remains a clean regression on the current stack.

Log: `results/mkv3-p4b-small-swcache-post-ceb2-20260706T1737Z.log`
