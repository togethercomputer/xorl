# Small dQ float2 post-fast-log no-go

Date: 2026-07-06
Head: `115f57d` (`megakernel: disable small attention fast log`)
Scope: source-free H512/S1024 small check after the exact small attention
fast-log default was changed to precise `logf`.

## Candidate

Forced the WGMMA attention dQ C==1 float2 direct-store branch:

`MK_ATTN_DQ_FLOAT2_STORE=1`

The current default keeps `_adqf2` disabled for H512/S1024 small. Positive
default-minus-float2 means float2 is faster.

## Route

Forced float2 compiled the expected `_adqf2` extension:

`xorl_megakernel_adkva_adkvf2_adqf2_drowst_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

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
  relative error `4.497528e-07`.
- float2-first: `loss_diff=+9.53674316e-07`, worst selected grad `w1.0`,
  relative error `4.674144e-07`.

## Timing

Forced float2 lost both construction orders:

| Order | default | float2 | default-minus-float2 | float2 wins |
| --- | ---: | ---: | ---: | ---: |
| default-first | `3266.30us` | `3268.90us` | `-3.18us` | `33/80` |
| float2-first | `3259.20us` | `3265.20us` | `-5.76us` | `27/80` |

## Decision

Keep `_adqf2` disabled for H512/S1024 small. The existing H256 long-shape
`MK_ATTN_DQ_FLOAT2_STORE` gates remain unchanged.

Log: `results/mkv3-p4b-small-dqf2-post-aflog-20260706T1709Z.log`
