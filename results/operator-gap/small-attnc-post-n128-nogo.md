# Small attention chunk post-n128 no-go

Date: 2026-07-06
Head: `7e1b7e5` (`megakernel: record small lm-head n128 check`)
Scope: source-free H512/S1024 small recheck after the 4W/cache and NN/NT n128
route changes.

## Candidate

Rechecked the existing attention backward chunk envs:

- default: `MK_ATTN_DKV_C` unset, `MK_ATTN_DQ_C` unset
- `dkv2_dq1`: `MK_ATTN_DKV_C=2`, `MK_ATTN_DQ_C=1`
- `dkv3_dq1`: `MK_ATTN_DKV_C=3`, `MK_ATTN_DQ_C=1`
- `dkv1_dq2`: `MK_ATTN_DKV_C=1`, `MK_ATTN_DQ_C=2`
- `dkv2_dq2`: `MK_ATTN_DKV_C=2`, `MK_ATTN_DQ_C=2`

Positive default-minus-variant means the variant is faster.

## Route

Default route stayed:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `ATTN_FWD_WG=8/512`
- `ATTN_DKV_WG=8/512`
- `ATTN_DQ_WG=8/512`

Variant route changes were the expected chunk multipliers:

- `dkv2_dq1`: `ATTN_DKV_WG=8/1024`, `ATTN_DQ_WG=8/512`
- `dkv3_dq1`: `ATTN_DKV_WG=8/1536`, `ATTN_DQ_WG=8/512`
- `dkv1_dq2`: `ATTN_DKV_WG=8/512`, `ATTN_DQ_WG=8/1024`
- `dkv2_dq2`: `ATTN_DKV_WG=8/1024`, `ATTN_DQ_WG=8/1024`

Program shape stayed `n_instr=288`, `critical_path=144`, `gated=127` for every
variant.

## Parity

All variants stayed in the normal selected-gradient numerical envelope:

- loss diffs were between `-4.76837158e-06` and `+4.76837158e-06`
- worst selected grad relative error was below `1.18e-02`

## Timing

All variants lost both build orders with zero wins:

| Variant | default-first default-minus-variant | variant wins | variant-first default-minus-variant | variant wins |
| --- | ---: | ---: | ---: | ---: |
| `dkv2_dq1` | `-77.71us` | `0/80` | `-72.86us` | `0/80` |
| `dkv3_dq1` | `-81.81us` | `0/80` | `-77.81us` | `0/80` |
| `dkv1_dq2` | `-34.61us` | `0/80` | `-32.99us` | `0/80` |
| `dkv2_dq2` | `-69.42us` | `0/80` | `-65.44us` | `0/80` |

## Decision

Keep H512/S1024 small attention chunks at the current default
`DKV_C=1`, `DQ_C=1`. No source change.

Log: `results/mkv3-p4b-small-attnc-post-n128-20260706T1840Z.log`
