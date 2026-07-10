# Qwen sidecar + head-DX RMS-dot profile @42e5b1d - PASS

Date: 2026-07-10 UTC
Agent: codex

## Scope

Source worktree:

```text
/tmp/xorl-oss-qwen-hdxrmsdot-7abb355-codex
```

Commit:

```text
42e5b1d default qwen head-dx rmsdot partials
```

This is the dirty-frontier qwen NT sidecar stack (`7abb355`) plus the default
qwen4b-l2 head-DX RMS-dot policy (`42e5b1d`). Shared checkout source was not
edited. a local GPU was used; other local GPU ordinals were not used.

## Verdict

PASS as the fresh qwen profile/timing refresh after composing NT sidecar and
default `_hdxrmsdot`.

- qwen4b-l2: promoted sidecar+`_hdxrmsdot` wins both step orders and both graph
  orders.
- qwen4b-l1: promoted sidecar wins both graph orders and both step orders; the
  old-first step order is a smaller `-31.840us` win, while the promoted-first
  step order is `-415.424us`.

This is not the full 12-shape certified scoreboard. It is the qwen L1/L2
profile refresh requested before the next scoreboard sweep.

## qwen4b-l2

Artifacts:

- Wrapper log:
  `results/qwen-hdxrmsdot-profile-wrapper-localgpu-42e5b1d-20260709T2355Z.log`
- Wrapper sha256:
  `f2dc0149cf02d31cde97fe0c861182d2de78aa05053021c2f3c0512f62a1b4fa`
- Sidecar-aware log:
  `results/qwen-hdxrmsdot-sidecar-aware-profile-localgpu-42e5b1d-20260709T2355Z.log`
- Sidecar-aware log sha256:
  `aaef0452d2e7b3b4dcc0198904e67057f06a83bb2e69ad58283c534aa52b61ce`
- Summary:
  `results/qwen-hdxrmsdot-sidecar-aware-profile-summary-42e5b1d-20260709T2355Z.json`
- Summary sha256:
  `81d4dff0e7a948b7397e642457f3a4da025408eb170c3cec93daba8c783a694d`
- Attribution log:
  `results/qwen-hdxrmsdot-main-attribution-profile-localgpu-42e5b1d-20260709T2355Z.log`
- Attribution log sha256:
  `17bccf53a5554a51a55fa04572306c8f942447ebd32c1b5b5ccc97540849c751`
- Wrapper result:
  `AWARE_DONE rc=0`, `ATTR_DONE rc=0`, `JOB_DONE rc=0 2026-07-10T00:00:10Z`

Route:

- Forced-old route: `78/44/14`, `smem=151552`, boundary rows `0`.
- Promoted route: `78/44/14`, `smem=151552`, boundary rows `1`, split plan
  valid, sidecar tile range `[0, 4748]`.
- Promoted extension suffix includes `_ntscbnd_hdxexpdf_hdxrmsdot_pdf240p`.

End-to-end timing:

| Order | Forced step | Promoted step | Step delta | Forced graph | Promoted graph | Graph delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old-first | `9401.408us` | `9109.696us` | `-291.712us` | `9537.696us` | `9135.584us` | `-402.112us` |
| promoted-first | `10049.408us` | `9167.264us` | `-882.144us` | `9961.792us` | `9192.832us` | `-768.960us` |

Promoted component medians:

| Component | Median |
| --- | ---: |
| prefix | `1980.224us` |
| sidecar | `1401.888us` |
| post | `5823.008us` |
| prefix+sidecar+post | `9220.192us` |

Main-kernel attribution view:

- Anchor: `42e5b1d`, mode `pdf`, median `7562.1us`.
- This attribution helper times the main PDF path and should not be quoted as
  the full sidecar step total.
- Top on-path buckets:
  `GEMMNN 1024x2560x151936.wg` `1662.0us`,
  `RMSNORM_BWD_DX` `1556.5us`,
  `GEMMNT 1024x19456x2560.wg` `524.3us`,
  `ATTN_DQ_WG128` `509.4us`,
  `GEMMNN 1024x2560x19456.wg` `448.4us`.
- Largest off-path upper bound remains lm-head dW
  `GEMMTN 151936x2560x1024.wg` at `2951.2us`.

## qwen4b-l1

Artifacts:

- Wrapper log:
  `results/qwen-hdxrmsdot-l1-profile-wrapper-localgpu-42e5b1d-20260710T0001Z.log`
- Wrapper sha256:
  `7efccc14c8db6725187b914d1d813e9583c0ffdac111c64c3827d1722dbc8e15`
- Sidecar-aware log:
  `results/qwen-hdxrmsdot-l1-sidecar-aware-profile-localgpu-42e5b1d-20260710T0001Z.log`
- Sidecar-aware log sha256:
  `775aa426c5c7f324578e0c430d1696fee5244346474ab24ac796b01d404a3dfa`
- Summary:
  `results/qwen-hdxrmsdot-l1-sidecar-aware-profile-summary-42e5b1d-20260710T0001Z.json`
- Summary sha256:
  `cb7d3b74f4a4ed8cc01d74095ad7ff3b037e3246472817fb638c58b0630f43d5`
- Wrapper result:
  `PROFILE_DONE rc=0 2026-07-10T00:05:53Z`

Route:

- Forced-old route: `47/26/9`, `smem=151552`, boundary rows `0`.
- Promoted route: `47/26/9`, `smem=151552`, boundary rows `1`, split plan
  valid, sidecar tile range `[0, 4748]`.
- qwen4b-l1 does not carry `_hdxrmsdot`; the composed commit still routes it
  through the NT sidecar policy only.

End-to-end timing:

| Order | Forced step | Promoted step | Step delta | Forced graph | Promoted graph | Graph delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old-first | `7087.552us` | `7055.712us` | `-31.840us` | `7388.512us` | `6984.448us` | `-404.064us` |
| promoted-first | `7394.848us` | `6979.424us` | `-415.424us` | `7338.816us` | `6957.376us` | `-381.440us` |

Promoted component medians:

| Component | Median |
| --- | ---: |
| prefix | `1089.440us` |
| sidecar | `1518.880us` |
| post | `4879.584us` |
| prefix+sidecar+post | `7483.008us` |

## Interpretation

The composition keeps both qwen routes structurally stable and positive at the
sidecar-aware step/graph layer. qwen4b-l2 receives the full composition
(`_ntscbnd` + `_hdxrmsdot`) and has strong both-order step and graph movement.
qwen4b-l1 is NT-sidecar-only and remains graph-strong; its old-first no-graph
step order is positive but much smaller than the other three L1 timing cells.

The qwen4b-l2 residual map after the composition still points first at the exact
head-DX `GEMMNN 1024x2560x151936.wg` path and RMS dX, then attention DQ wait.
Off-path lm-head dW is large but remains a scheduler/overlap question, not a
direct critical-path saving unless a future experiment proves contention relief.

## Cleanup

The L2 and L1 profile locks were released. local GPU was observed at `0 MiB, 0%`
after the L1 run. No remote isolated resources were launched for this refresh.
