# Qwen4b-l1 win canonical megakernel checkout

Date: 2026-07-10 UTC
Agent: codex

## Verdict

The canonical `<repo-root>` checkout now carries the qwen4b-l1
full-score win directly on branch `megakernel`.

```text
path: <repo-root>
branch: megakernel
head: 9eb4d03
qwen source anchor: 42e5b1d
qwen base: 3b5701e
```

Commit chain:

```text
3b5701e megakernel: land CE fwd warprow + qknorm bwd vec8
be55098 snapshot dirty qwen frontier
7abb355 port qwen nt sidecar to dirty frontier
965a0c1 add qwen head-dx rmsdot partials gate
42e5b1d default qwen head-dx rmsdot partials
9eb4d03 megakernel: record operator-gap phase notes
```

The temporary/sibling qwen integration checkouts that previously also pointed
at `42e5b1d` were removed after reconciliation:

```text
removed worktree: <repo-root>
deleted branch: megakernel-qwen4b-l1-win-42e5b1d

removed worktree: /tmp/xorl-oss-qwen-integration-codex-20260709T235457Z
deleted branch: qwen-integration-codex-20260709T235457Z
```

## Evidence

The full scoreboard from the same source commit chain is recorded in:

```text
results/operator-gap/qwen-integration-scoreboard-42e5b1d-pass.md
```

The qwen4b-l1 row beat the `compile+cudagraph+` baseline:

```text
qwen4b-l1 megakernel 6961.952us
qwen4b-l1 baseline   7314.464us
delta                -352.512us
ratio                0.9518
```

qwen4b-l2 did not beat the same baseline in that sweep:

```text
qwen4b-l2 megakernel 9134.784us
qwen4b-l2 baseline   8815.360us
delta                +319.424us
ratio                1.0362
```

## Checkout Validation

Run from `<repo-root>` with the login shell disabled so the requested
workdir is honored:

```text
git status --short --branch
git log --oneline -5
python3 -m py_compile experiments/fused-training-megakernel/mk.py experiments/fused-training-megakernel/model.py
git diff --check 3b5701e..HEAD
```

Results:

```text
status: branch megakernel at 9eb4d03; tracked working tree clean
log: 9eb4d03 -> 42e5b1d -> 965a0c1 -> 7abb355 -> be55098 -> 3b5701e
source diff vs 42e5b1d: no differences in megakernel source/qwen sidecar files
py_compile: pass
git diff --check: pass
qwen sidecar route helper: pass
```

The phase documentation/talk refresh files were committed as:

```text
9eb4d03 megakernel: record operator-gap phase notes
```

## Use

Start future qwen4b-l1 work here:

```bash
cd <repo-root>
```

If a command or tool resolves through the wrong checkout, use explicit Git
arguments:

```bash
git -C <repo-root> status --short --branch
```

No separate landing worktree is needed for the qwen4b-l1 win now; `megakernel`
already points at the validated source commit.
