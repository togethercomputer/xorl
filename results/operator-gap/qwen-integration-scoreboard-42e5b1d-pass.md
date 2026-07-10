# Qwen integration scoreboard @42e5b1d

Date: 2026-07-10 UTC
Agent: codex

## Verdict

Full 12-shape scoreboard completed from the clean qwen integration branch at
`42e5b1d` with `rc=0`, `pyrc=0`, and 12 JSON rows.

This is a scoreboard execution pass, not an "all shapes improved" claim. The
standard shapes remain slower than `compile+cudagraph+`, qwen4b-l1 is ahead of
that baseline in this sweep, and qwen4b-l2 is close but still behind that
baseline. The qwen sidecar/RMS-dot contribution itself remains supported by the
separate rollback/profile evidence.

## Source

```text
worktree: /tmp/xorl-oss-qwen-integration-codex-20260709T235457Z
branch: qwen-integration-codex-20260709T235457Z
head: 42e5b1d
base: 3b5701e
```

Commit chain:

```text
3b5701e megakernel: land CE fwd warprow + qknorm bwd vec8
be55098 snapshot dirty qwen frontier
7abb355 port qwen nt sidecar to dirty frontier
965a0c1 add qwen head-dx rmsdot partials gate
42e5b1d default qwen head-dx rmsdot partials
```

The run used the shared result runner with the integration source tree selected
through `MK_AB_TREE`:

```text
CUDA_VISIBLE_DEVICES=<local-gpu>
TORCH_EXTENSIONS_DIR=/tmp/torchext-qwen-integration-scoreboard-42e5b1d-20260710T0008Z
TORCH_CUDA_ARCH_LIST=9.0a
MK_AB_TREE=/tmp/xorl-oss-qwen-integration-codex-20260709T235457Z
python3 results/scoreboard_shape_589ee3d.py score <shape>
```

## Artifacts

```text
results/qwen-integration-scoreboard-localgpu-42e5b1d-20260710T0008Z.log
sha256 8f147a3294820b7d66e80a197522c63bd91f50bc57a95055572d910d0e0ae3c5

results/qwen-integration-scoreboard-summary-42e5b1d-20260710T0008Z.jsonl
sha256 cfa4f85780a0d9b895feca71abdf00d072866d50f77cfb264e6950ef001753eb
rows 12
```

The GPU lock `results/.gpulock-qwen-integration-scoreboard-42e5b1d-localgpu` was
removed by the wrapper at exit.

## Results

Baseline column is `compile+cudagraph+`.

| shape | megakernel us | baseline us | delta us | ratio |
|---|---:|---:|---:|---:|
| s128 | 678.208 | 488.416 | +189.792 | 1.3886 |
| s256 | 749.152 | 549.312 | +199.840 | 1.3638 |
| nano | 865.920 | 632.576 | +233.344 | 1.3689 |
| s1024 | 1175.520 | 773.504 | +402.016 | 1.5197 |
| s2048 | 1692.896 | 1039.776 | +653.120 | 1.6281 |
| s3072 | 2225.472 | 1338.656 | +886.816 | 1.6625 |
| s4096 | 2836.480 | 1571.968 | +1264.512 | 1.8044 |
| s8192 | 6024.416 | 3146.144 | +2878.272 | 1.9149 |
| deep | 2216.480 | 1765.856 | +450.624 | 1.2552 |
| small | 3004.544 | 1920.896 | +1083.648 | 1.5641 |
| qwen4b-l1 | 6961.952 | 7314.464 | -352.512 | 0.9518 |
| qwen4b-l2 | 9134.784 | 8815.360 | +319.424 | 1.0362 |

## Interpretation

- The chain is fully replay-packaged and has now cleared static, route,
  same-commit GPU evidence review, fresh qwen profile refresh, and a full
  scoreboard execution.
- qwen4b-l1 is a full-score win in this scoreboard run.
- qwen4b-l2 remains a slight full-score loss against `compile+cudagraph+`, even
  though the sidecar/RMS-dot rollback/profile gates are positive. Treat qwen L2
  full-score claims as not yet closed by this sweep.
- The shared `megakernel` checkout was still dirty and active during this work,
  so this note does not claim the chain has been landed there.

## Landing

Replay artifacts remain:

```text
results/qwen-integration-chain-42e5b1d-20260709T2359Z.mbox
sha256 622d30ec9e17b6a73a7f26e51e17266bdc575eb99080aede6354b54f93d45350

results/qwen-integration-chain-42e5b1d-20260709T2359Z.patch
sha256 4fbbc9853bc88c9ddfecab64972ba2a1a8c0ae2b75055bb4edbf63a1164f81b0
```

At a quiet shared-tree window, land by cherry-picking the integration branch or
applying the mbox/patch, then rerun the qwen profile/scoreboard on the landed
tree if any conflicts or nearby source changes occur.
