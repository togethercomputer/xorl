# Small attention fast-log post-direct-BF16 promotion

Date: 2026-07-06
Base: `24150ec` (`megakernel: record post-direct dkv float2 check`)

## Question

`MK_ATTN_FAST_LOG` was previously a broad default for WGMMA attention fwd LSE
stores. Earlier evidence showed H512/S1024 small as modestly positive, while a
later H256/S1024 check was neutral. After the NN/NT n128 retunes and direct-BF16
GEMM epilogue promotion, H512/S1024 small was rechecked because `ATTN_FWD_WG`
remains top-three in the profile.

## Source-free recheck

Forced precise `logf` with:

`MK_ATTN_FAST_LOG=0`

The precise extension dropped `_aflog`; route shape stayed unchanged:

- `n_instr=288`
- `critical_path=144`
- `gated=127`
- `ATTN_FWD_WG=8/512`
- `ATTN_DKV_WG=8/512`
- `ATTN_DQ_WG=8/512`

Selected-gradient parity stayed clean. The expected q/k norm sensitivity stayed
inside the existing model tolerance, with worst selected grad `qn.0` relative
error around `9.54e-03`.

Timing was weak but repeatable in favor of precise `logf`. Positive
default-minus-precise means precise was faster:

| Log | reps | default-first | precise wins | precise-first | precise wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mkv3-p4b-small-aflog-post-directbf16-20260706T1702Z.log` | 80 | `+1.34us` | `43/80` | `+1.46us` | `42/80` |
| `mkv3-p4b-small-aflog-post-directbf16-repeat-20260706T1702Z.log` | 240 | `+2.35us` | `141/240` | `+1.07us` | `129/240` |
| `mkv3-p4b-small-aflog-post-directbf16-repeat2-20260706T1702Z.log` | 480 | `+0.83us` | `260/480` | `+1.22us` | `265/480` |

## Promoted default validation

The default now disables `MK_ATTN_FAST_LOG` only for exact
H512/L8/S1024/nq8/nkv4/D64/I1536/V16384 small. Other shapes keep the previous
fast-log default, and `MK_ATTN_FAST_LOG=1` force-restores the old small route.

Promoted-default versus forced-fast log:

`results/mkv3-p4b-small-aflog-promoted-20260706T1702Z.log`

Default loaded the `_aflog`-free extension:

`xorl_megakernel_adkva_adkvf2_drowst_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Forced fast loaded the old `_aflog` extension:

`xorl_megakernel_adkva_adkvf2_drowst_aflog_aex2_lex2_ceb2_qkbc_qkbc128_swfma_swb2w_swb4w_gmbar_gdbf16`

Parity stayed clean:

- default-first: `loss_diff=+9.53674316e-07`, worst selected grad `qn.0`,
  relative error `9.537255e-03`.
- fast-first: `loss_diff=+0.00000000e+00`, worst selected grad `qn.0`,
  relative error `9.537212e-03`.

Promoted default beat forced fast both orders:

- default-first: default `3261.42us`, fast `3263.79us`,
  default-minus-fast `-1.04us`, fast wins `110/240`.
- fast-first: default `3259.15us`, fast `3261.33us`,
  default-minus-fast `-1.65us`, fast wins `110/240`.

Validation:

- `py_compile` for `model.py` and `mk.py`
- `test_model.py`: `ALL MODEL TESTS PASSED`
- `test_ops.py`: `ALL OP TESTS PASSED`

Refreshed promoted-default profile:

- log: `results/mkv3-p4b-profile-small-aflog-precise-default-20260706T1702Z.log`
- total: `3331.4us`

Refreshed promoted-default score:

- log: `results/mkv3-p4b-score-small-aflog-precise-default-20260706T1702Z.log`
- megakernel: `3312.6us`
- eager: `15561.5us`
- compile: `4055.9us`
- compile+cudagraph: `2092.8us`
- compile+cudagraph+: `1898.0us`

## Decision

Promote the exact H512/L8/S1024 small precise-log default. This is a small
shape-scoped cleanup, not a material scoreboard move. Keep fast-log enabled for
the prior broad D64 WGMMA attention shapes unless a shape-specific recheck says
otherwise.
