---
title: Plan — traceable chunked CE for whole-step torch.compile
---

## Context

The single-GPU whole-model compile work (branch `multi-gpu`, env-gated by
`XORL_COMPILE_WHOLE_STEP=1`) folds **backbone + lm_head + cross-entropy into one compiled
region** so the entire fwd+loss captures as a single `reduce-overhead` CUDA graph (with
Traceable FSDP2, `torch._dynamo.config.skip_fsdp_hooks=False`, so FSDP collectives stay
in-graph). Today that path uses **vanilla `F.cross_entropy` over the full `[N, vocab]`
logits** (`Trainer._whole_step_impl`, `src/xorl/trainers/trainer.py`) and therefore **OOMs
at large packing** — the launch script even warns to drop `sample_packing_sequence_len` to
~8k on the 8B model.

The memory-efficient CE we already have (`src/xorl/ops/loss/compiled_cross_entropy.py`)
avoids that by chunking the token dim with the inductor `auto_chunker` option — but it lives
**inside its own `torch.compile`** call (`options={"auto_chunker.enable": True,
"auto_chunker.num_chunk": N}`). A nested `torch.compile` does not compose into an outer
compiled region, and the `auto_chunker.*` options only apply to that standalone compile. So
today you must choose: whole-step (one graph, OOM) **or** chunked CE (memory-OK, separate
graph). You cannot fold the chunked CE into the whole-step graph.

**Goal:** a chunked CE written as **plain, traceable PyTorch** — a static Python unroll over
chunks with **no inner `torch.compile`** — so Dynamo traces it as part of the outer
whole-step graph. Peak logits stay at one chunk (`[N/num_chunks, vocab]`), so the whole-step
path stops OOMing while remaining a single fused graph.

### Scope (confirmed)

- **Whole-step CE only.** Add the traceable chunked CE and wire it into `_whole_step_impl`,
  replacing the vanilla `F.cross_entropy`. Plain CE path only — **not** the z-loss
  (`compiled_ce_and_lse_sq_function`) or reverse-KL (`compiled_reverse_kl_function`) variants
  (they keep using `auto_chunker` for now; same pattern applies later if wanted).
- The production eager/`auto_chunker` path in `causallm_loss_function`
  (`src/xorl/ops/loss/causallm_loss.py`) is **untouched** — it already works for the
  non-whole-step training path.

## Why a naive unroll is not enough — backward memory

A plain Python `for i in range(num_chunks)` loop is unrolled by Dynamo at trace time into one
graph with `num_chunks` copies of (slice → `hidden_chunk @ weightᵀ` → `F.cross_entropy`).
Inductor frees each chunk's logits once its CE reduction is done, so **forward** peak memory
is bounded to one chunk. So far so good.

The catch is **backward**. `auto_chunker` chunks fwd *and* bwd together and **recomputes**
each chunk's logits in the backward rather than stashing all `[N, vocab]`. A naive unroll
under `torch.compile` does **not** get this for free: the min-cut partitioner, left alone,
prefers to **save** matmul outputs (recompute is "expensive"), so it would stash every
chunk's logits for backward → the full `[N, vocab]` is resident again → the OOM returns in
bwd. (This is exactly how `auto_chunker` behaves today, and what the new code must
reproduce.)

**Fix:** wrap each chunk's body in per-chunk activation checkpointing
(`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`) to **force** per-chunk
recompute in backward. Under `torch.compile`, Dynamo traces `checkpoint` as a higher-order op
and the AOT partitioner honors it, so forward saves only the small per-chunk loss/lse and
backward recomputes the chunk's logits. Peak logits stay at one chunk in **both** directions,
no graph break, one fused region.

## Implementation

### 1. New traceable chunked CE — `src/xorl/ops/loss/compiled_cross_entropy.py`

Add a module-level function (no `torch.compile` inside it, no caches):

```python
def traceable_chunked_cross_entropy(
    hidden_states: torch.Tensor,   # [N, hidden]
    weight: torch.Tensor,          # [vocab, hidden]
    labels: torch.Tensor,          # [N]
    ignore_index: int = -100,
    num_chunks: int = 8,
    reduction: str = "sum",        # whole-step wants a single scalar; "none" also supported
) -> torch.Tensor:
    """Chunked linear cross-entropy with NO inner torch.compile.

    Static Python unroll over `num_chunks` token chunks; each chunk's
    (logits = hidden_chunk @ weightᵀ) → F.cross_entropy is wrapped in
    activation checkpointing so backward recomputes the chunk (peak logits =
    one chunk, in fwd AND bwd). Fully traceable, so it folds into an outer
    torch.compile region (e.g. the whole-step graph).
    """
```

Design points:

- **Static chunk bounds** computed from a Python `int` `N = hidden_states.shape[0]` and the
  constant `num_chunks` (ceil division), so Dynamo unrolls the loop and emits static-shape
  matmuls (matches the whole-step path's static-shape / `reduce-overhead` assumption). Handle
  a non-divisible last chunk; skip empty trailing chunks.
- **Per-chunk checkpoint:** factor the chunk body into a small local fn
  `def _chunk_loss(h, l): logits = (h @ weight.t()).float(); return F.cross_entropy(l_in=...,
  reduction="none", ignore_index=ignore_index)` and call it via
  `torch.utils.checkpoint.checkpoint(_chunk_loss, h_chunk, l_chunk, use_reentrant=False)`.
- **fp32 matmul** inside the chunk (`.float()`), same as the existing `_compute_ce`, so
  numerics match the current CE.
- **Reduction:** accumulate per-chunk results. For `reduction="sum"`/`"mean"` keep a running
  scalar (cheap, no `[N]` concat); for `"none"` collect per-chunk `[chunk]` tensors and
  `torch.cat` at the end. The whole-step caller uses `"sum"` then divides by
  `global_valid_tokens` (see below).
- A tiny **CPU fallback** (`device.type == "cpu"`) that calls the un-chunked path, mirroring
  `compiled_reverse_kl_function`, so unit tests run on CPU without checkpoint/compile quirks.

### 2. Wire into the whole-step callable — `Trainer._whole_step_impl` (`src/xorl/trainers/trainer.py`)

Replace the vanilla full-logits CE:

```python
outputs = self.model(**micro_batch, use_cache=False, output_hidden_states=False)
hidden = outputs.last_hidden_state.reshape(-1, outputs.last_hidden_state.size(-1))
loss_sum = traceable_chunked_cross_entropy(
    hidden,
    self.model.lm_head.weight,           # tied/untied lm_head weight
    labels.reshape(-1),
    ignore_index=IGNORE_INDEX,
    num_chunks=<num_chunks>,
    reduction="sum",
)
return loss_sum / global_valid_tokens
```

- Keep the existing `loss / global_valid_tokens` normalization and the **detached on-GPU
  loss accumulation** already in `_forward_backward` (the whole-step branch sums
  `ga_loss.detach()` and defers `.item()` to the per-step DP all-reduce — do not reintroduce
  a per-micro-batch sync).
- Source `num_chunks` from the same place the rest of the loss path does (the
  `causallm_loss` params / config `num_chunks`, default 8) rather than hardcoding, so it's
  tunable per run.
- `lm_head` weight access should go through `get_lm_head_weight` (`module_utils.py`) if LoRA
  on the head is in play; for the plain whole-step path `self.model.lm_head.weight` is fine.

### 3. Launch ergonomics (already in place)

`vast/launch_multi_gpu.sh` now auto-enables `XORL_COMPILE_REDUCE_OVERHEAD=1` whenever
`XORL_COMPILE_WHOLE_BACKBONE` or `XORL_COMPILE_WHOLE_STEP` is on. Once chunked CE lands, the
"use a SMALL sample_packing_sequence_len (~8k)" OOM warning in that script can be relaxed —
update it in the same change so the doc/script don't drift.

## Critical files

- `src/xorl/ops/loss/compiled_cross_entropy.py` — add `traceable_chunked_cross_entropy`
  (the static-unroll + per-chunk-checkpoint function). Existing `auto_chunker` helpers stay.
- `src/xorl/trainers/trainer.py` — `_whole_step_impl`: swap vanilla CE → chunked CE.
  Leave the detached-tensor loss accumulation in `_forward_backward` as is.
- `vast/launch_multi_gpu.sh` — relax the whole-step OOM warning after verification.
- (test) `tests/ops/loss/` — new equivalence test (see below), alongside the existing
  `test_causallm_z_loss.py`.

## Verification

**Correctness (CPU/GPU unit test, fast):** add a test asserting
`traceable_chunked_cross_entropy(..., reduction="sum") / num_valid` equals
`F.cross_entropy(full_logits, labels, ignore_index=-100, reduction="sum") / num_valid` (and
the `"none"` form matches per-token) within fp tolerance, across several `num_chunks`
(1, 3, 8, including a non-divisible `N`) and with some `ignore_index` positions. Also check
**gradients** w.r.t. `hidden_states` and `weight` match the unchunked autograd reference —
this is what proves the per-chunk checkpoint recompute is wired correctly.

**Memory + single-graph (the actual point), on Vast — never locally** (deps are
linux/cuda-only; see CLAUDE.md). Run the whole-step path at a packing length that currently
OOMs with vanilla CE:

```bash
cd vast
XORL_COMPILE_WHOLE_STEP=1 ./launch_multi_gpu.sh examples/.../qwen3_8b*.yaml
# (REDUCE_OVERHEAD auto-enables; pick a sample_packing_sequence_len that OOMs today, e.g. 32k)
```

Confirm from the run:
1. **It no longer OOMs** at the packing length that previously did (the headline result).
2. The `.pkl` memory snapshot (`results/<stem>-<ts>/trace/*.pkl`, open at
   <https://docs.pytorch.org/memory_viz>) shows **peak logits ≈ one chunk**, not `[N, vocab]`.
3. The `torch.profiler` trace (Perfetto) shows the CE matmuls/reductions **inside the
   whole-step graph** (no separate CE compile region, no graph break at the loss) and the
   backward **recomputes** per-chunk logits.
4. Throughput/MFU vs the chunked-but-separate-graph baseline (compare to
   `results/tables/whole-model-compile-comparison.md` methodology).
