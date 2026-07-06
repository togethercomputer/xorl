# Small RMS dX R4 post-direct-BF16 no-go

Date: 2026-07-06
Head: `4918611` (`megakernel: record post-direct coldcap check`)
Scope: source-free H512/S1024 small recheck after NN/NT n128 retunes and the
direct-BF16 GEMM epilogue promotion.

## Candidate

Rechecked the existing `MK_RMS_DX_R4=1` override for exact H512/L8/S1024 small.
The default route leaves small on the normal two-row `OP_RMSNORM_BWD_DX`; the
candidate forces the four-row `OP_RMSNORM_BWD_DX_R4` fold.

Positive default-minus-R4 means R4 is faster.

## Route

Default:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `RMSNORM_BWD_DX=17/1088`
- `RMSNORM_BWD_DX_R4=0/0`
- `RMSNORM_BWD_DW=17/1088`

Forced R4:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `RMSNORM_BWD_DX=0/0`
- `RMSNORM_BWD_DX_R4=17/544`
- `RMSNORM_BWD_DW=17/1088`

## Parity

Selected-gradient parity stayed clean in both construction orders:

- default-first: `loss_diff=-3.81469727e-06`, worst selected grad `qn.0`,
  relative error `7.054795e-07`.
- R4-first: `loss_diff=+2.86102295e-06`, worst selected grad `kn.0`,
  relative error `4.498913e-07`.

## Timing

Forced R4 lost both construction orders:

| Order | default | R4 | default-minus-R4 | R4 wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3259.50us` | `3296.45us` | `-34.08us` | `0/160` |
| R4-first | `3258.59us` | `3293.54us` | `-35.23us` | `1/160` |

## Decision

Keep H512/S1024 small on the normal two-row `OP_RMSNORM_BWD_DX` default after
the direct-BF16 promotion. `MK_RMS_DX_R4=1` remains a sweep override only for
this shape.

Log: `results/mkv3-p4b-small-rmsdx-r4-post-directbf16-20260706T1657Z.log`
