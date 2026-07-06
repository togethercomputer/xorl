# Small SwiGLU derivative FMA post-qkbc keep

Date: 2026-07-06
Head: `2230163` (`megakernel: record small qknorm cache check`)
Scope: source-free H512/S1024 small recheck after the current small route moved
to `SWIGLU_BWD_4W` with sigmoid cache off, exact small attention precise-log,
direct-BF16 epilogues, and qknorm D64 cache.

## Candidate

Forced the old derivative expression:

`MK_SWIGLU_FMA_DERIV=0`

The current default keeps `_swfma` enabled and uses
`fmaf(-sg, sig, sig + sg)`. The forced-old extension drops `_swfma` and restores
`sig + sg * (1.0f - sig)`. Positive default-minus-old means the old expression
is faster.

## Route

Forced old compiled the expected extension without `_swfma`:

`xorl_megakernel_adkva_adkvf2_drowst_aex2_lex2_ceb2_qkbc_qkbc128_swb2w_swb4w_gmbar_gdbf16`

Route shape was unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `SWIGLU_FWD=8/1024`
- `SWIGLU_BWD_4W=8/4096`
- `swsig=False`

## Parity

Selected-gradient parity stayed clean:

- 80-rep default-first: `loss_diff=+9.53674316e-07`, worst selected grad
  `kn.0`, relative error `1.007460e-02`.
- 80-rep old-first: `loss_diff=+0.00000000e+00`, worst selected grad `kn.0`,
  relative error `1.007464e-02`.
- 240-rep default-first repeat: `loss_diff=+9.53674316e-07`, worst selected grad
  `kn.0`, relative error `1.007505e-02`.
- 240-rep old-first repeat: `loss_diff=-1.90734863e-06`, worst selected grad
  `kn.0`, relative error `1.007442e-02`.

## Timing

The first 80-rep pass weakly favored old, but only at noise-level margins:

| Order | default | old | default-minus-old | old wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3268.83us` | `3270.32us` | `+0.30us` | `43/80` |
| old-first | `3263.02us` | `3261.95us` | `+2.24us` | `46/80` |

The longer 240-rep repeat split by construction order:

| Order | default | old | default-minus-old | old wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3260.99us` | `3259.90us` | `+1.28us` | `127/240` |
| old-first | `3257.94us` | `3258.21us` | `-1.30us` | `109/240` |

## Decision

Keep `_swfma` enabled. The current 4W/cache-off small route does not reproduce
the older large FMA win, but the forced-old derivative did not survive the
both-order repeat gate with a meaningful margin.

Logs:

- `results/mkv3-p4b-small-swfma-post-qkbc-20260706T1719Z.log`
- `results/mkv3-p4b-small-swfma-post-qkbc-repeat-20260706T1724Z.log`
