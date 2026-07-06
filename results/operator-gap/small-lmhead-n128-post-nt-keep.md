# Small lm-head n128 post-NT keep

Date: 2026-07-06
Head: `58abd88` (`megakernel: record post-n128 gmbar check`)
Scope: source-free H512/S1024 small check after the NN/NT n128 row gates.

## Question

After promoting the exact NN and NT row gates, the current small route still has
two n128 rows:

- row 152: lm-head forward, `GEMMNN 1024x16384x512`, controlled by the generic
  `MK_WGMMA_N128` path.
- row 155: head dX, `GEMMNT 1024x512x16384`, explicitly routed through the
  head-dX n128 path.

This check tested whether the generic lm-head row should also leave n128 by
running current default against `MK_WGMMA_N128=0`. The env value disables the
generic n128 path, so row 152 moves to the 2048-tile non-n128 route while row
155 remains n128.

## Route

Default:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `n128=2`
- row 152: `M=1024 N=16384 K=512`, flags `6274`, `n128=True`
- row 155: `M=1024 N=512 K=16384`, flags `4232`, `n128=True`

`MK_WGMMA_N128=0`:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `n128=1`
- row 152: `M=1024 N=16384 K=512`, flags `2178`, `n128=False`
- row 155: `M=1024 N=512 K=16384`, flags `4232`, `n128=True`

## Parity

Both construction orders stayed in the normal numerical envelope:

- default-first: `loss_diff=-2.86102295e-06`, worst selected grad `w1.0`,
  relative error `5.261678e-07`.
- n128off-first: `loss_diff=-3.81469727e-06`, worst selected grad `kn.0`,
  relative error `7.198257e-07`.

## Timing

Disabling lm-head n128 lost both construction orders:

- default-first: default `3291.34us`, n128off `3340.72us`,
  default-minus-n128off median `-48.88us`, mean `-47.39us`,
  n128off wins `1/80`.
- n128off-first: default `3287.06us`, n128off `3335.36us`,
  default-minus-n128off median `-48.67us`, mean `-47.32us`,
  n128off wins `0/80`.

## Decision

Keep the current small lm-head n128 route. No source change.

Log: `results/mkv3-p4b-small-n128off-after-nt-20260706T1830Z.log`
