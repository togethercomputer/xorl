# LoRA-path K3 contract: fold-on-sync + merged-forward (2026-07-06)

The last uncontracted term of the LoRA RL lane (Qwen3.5-35B-A3B Wordle class):
LoRA deltas were computed by different implementations trainer-side (xorl
group-GEMM LoRA backends) vs serving-side (sglang LoRA MoE / sgmv kernels), with
no contract — and injecting LoRA silently pulled the trainer's BASE compute off
the contracted fused-experts kernel too.

## Path maps (measured on the tips: xorl 5e817300a, sglang 1ed2ce7c1)

- **Serving (sglang)**: `FusedMoEWithLoRA` — chunked-SGMV lane (default) or the
  virtual-experts lane (`--lora-use-virtual-experts`): base gate_up GEMM (bf16
  out) → LoRA delta added in bf16 into the pre-activation gate_up (shrink =
  bespoke split-K triton kernel, fp32 acc → bf16 intermediate, **relaxed
  atomic_add under SPLIT_K>1** — nondeterministic at small M; expand = base
  fused-MoE kernel with `fuse_add_to_output`) → silu_and_mul → base down GEMM +
  bf16 delta add (routed weight fp32, in-kernel) → top-k reduce. MoE scaling
  (alpha/r) is pre-folded into the B buffers at load; dense modules scale
  in-kernel in fp32. None of the LoRA GEMMs are BI-interposable (custom triton).
- **Trainer (xorl)**: `MoEExpertsLoRA` — custom group-GEMM tree, per-projection
  `base + (x@A)@B*scaling` in bf16, silu in bf16 (local) / fp32 (EP), routing
  weights pre-down (local) — a different reduction tree end-to-end, and the
  contracted `sglang_fused_experts_forward` explicitly excluded LoRA modules.
- **Folds already in-tree**: `merge_weights` (fp32-accumulate cast-once,
  PR #164) and weight-sync extraction (bf16(W) + bf16(delta)
  double-rounding).

## Measured pre-contract term (real Q3.5-35B-A3B layer-1 weights, Wordle adapter
shape r=16/alpha=16 hybrid_shared, nonzero B at trained magnitude)

Trainer LoRA forward vs serving VE forward, expert-block output, M∈{8,140,2048}:
**~65–69% of outputs bitwise different, mean |Δ| ≈ 8–10e-6, max ≈ 1.2e-4,
per-element p50 rel ≈ 5.4e-3** — bf16-noise class but near-full coverage,
**dominated by the base-tree fork** (zero-adapter arms measure ~64% nz at the
same magnitude; the delta math adds the remainder). This is §6-item-1 of the
min-K3 recipe. Notably: serving VE lane with zero adapters is **bit-identical**
to the monolithic `fused_experts_impl` — the serving base decomposition is
already inside the existing fused-experts contract.

## Decision: Option B (fold-on-sync + merged-forward) — by the numbers

| | Option A: contract the unmerged fused-LoRA path | Option B: canonical fold + merged forward |
|---|---|---|
| surface | vendor/contract ~1.4k lines of VE triton kernels + orchestration on both engines; fix the split-K atomic shrink (batch-VARIANT config: `min(max_split_k, 128//grid(M))`); contract sweeps; new autograd | one pinned fold (~40 lines each side) + straight-through autograd; base compute rides EXISTING contracts unchanged |
| serving speed | keeps the LoRA-kernel tax | **+81% at decode M=8, +16–23% at prefill** (LoRA kernels exit the hot path) |
| trainer speed | ≥ current LoRA lane | merged (cached W') **3–6× faster** than the unmerged LoRA forward; fold 22.8 ms/layer fp32 (0.91 s/model, once per optimizer step, version-keyed cache) |
| K3 result | bitwise IF every kernel + add point + atomicity is pinned | **proven bitwise**: trainer merged fwd ≡ serving postfold at M∈{8,140,2048}, grad and no-grad |
| cross-venv | every vendored kernel needs venv-cert | fold bmm measured **bit-identical** torch 2.12.1+cu132 ↔ 2.9.1+cu128; re-gated by `foldparity` |
| caveat | supports multi-adapter serving | **single-adapter RL lanes only** (fold mutates the base; multi-adapter hot-swap keeps the uncontracted kernels — Option A is the documented fallback if that lane ever needs a contract) |

## What shipped

- **Canonical fold** (`src/xorl/lora/fold.py` ↔ sglang
  `python/sglang/srt/lora/canonical_fold.py`): pinned order — fp32-contiguous
  factors, GKN-orientation bmm (dense: B@A), `*float(scaling)`, fp32 accumulate,
  cast ONCE. Kept in lockstep by the `foldparity` gate.
- **`XORL_LORA_MERGED_FORWARD=1`** (opt-in):
  - `MoEExpertsLoRA` gains the fused-experts contract surface
    (`sglang_fused_experts_forward`, EP merged lane on alltoall, auto-enable at
    ep=1); forward = the contracted serving kernel on cached folded weights;
    backward = the proven `_SglangFusedExperts*TrainFunction` backward producing
    dW', chained through straight-through fold Functions into dA/dB (closed
    form, fp32). Cache keyed on param `_version`s + active rank/alpha;
    `XORL_LORA_MERGED_FORWARD_CACHE=0` refolds per call (bounded memory).
  - `LoraLinear` merged forward (`F.linear` on W') — covers the Wordle
    qkv/o/GDN-projection adapters — and **composes with `XORL_BI_TRUNK_LINEAR`**
    (the wrap that previously raised on any adapter): the BI GEMM runs on W'.
  - Weight-sync merged extraction ships the SAME canonical bytes (the module's
    fold cache) — the engine serves exactly what the trainer trains with.
  - Fused-experts flag on LoRA modules without the merged flag now raises loudly
    (was AttributeError / silent stock fallback).
- **`SGLANG_LORA_FOLD_CANONICAL=1`** (sglang, opt-in): the fold-on-receipt
  paths (`_fold_standard_module_from_tensors`, `_fold_moe_module_from_tensors`)
  switch from bf16-add to the canonical cast-once order — the ship-adapters
  variant serves the trainer's bytes without a full merged sync. Caveat: the MoE
  mem-pool pre-scales B in bf16; bitwise requires power-of-two scaling
  (Wordle: alpha==r → 1.0) or unscaled factors.

## Gates (all PASS)

- `experiments/k3_tests/lora_path_xengine.py` on real Q3.5-35B-A3B layer-1:
  - `foldgate`: postfold serving ≡ trainer fused forward on W', bit-identical,
    M∈{8,140,2048}, grad+no-grad.
  - `mergedmodule`: production `MoEExpertsLoRA` merged forward ≡ serving on the
    module's shipped bytes (bit-identical); trainable ≡ no-grad forward
    (bit-identical); A/B grads vs the unmerged lane rel-L2 ≈ 6.5e-3 with healthy
    norms (fwd/bwd decoupling doctrine); loss-step smoke + cache invalidation.
  - `foldparity`: xorl canonical fold ≡ sglang canonical fold, bit-identical.
  - `xvenv-fold/cmp`: fold bit-identical across trainer/serving torch builds.
- `tests/models/test_lora_merged_forward.py` (20) + sglang
  `test/registered/lora/test_canonical_fold.py` (5): pinned order, straight-
  through grads ≡ autograd, EP shard-commutation of the fold, cache keying,
  loud-fail composition.
- Existing suites green: fused-experts contract suite (22 pass, certified cu132
  stack), `test_qwen3_moe_fused_lora`, `test_moe_experts_lora`,
  `test_lora_merge_fp32_cast_once`, `test_bi_trunk_linear`, `test_lora_utils`.

## Residuals / follow-ups

- EP merged lane is code-complete and mirrors the proven MoEExperts EP contract
  path (+ unit-tested fold/shard commutation), but the multi-rank bitwise gate
  needs a cluster run — gate before flipping any EP lane to the flag.
- The ship-A/B + fold-on-receipt RL transport (client driver calling
  `load_lora_adapter_from_tensors` + a fold trigger per sync) is not wired; the
  zero-serving-change path today is the existing merged full sync (which now
  ships canonical bytes under the flag).
- QLoRA modules are outside the envelope (quantized lanes are outside the K3
  recipe anyway); legacy folds are unchanged when the flags are off.

## Recipe doc §5/§6 replacement text (for
`qwen3_5_35b_a3b_wordle_lora_min_k3_recipe_20260706.md` once merged)

**§5 (LoRA-specific caveats) becomes:**

> The LoRA path is CONTRACTED as of xorl #<this-PR> / sglang #<this-PR> via
> fold-on-sync + merged-forward. Set `XORL_LORA_MERGED_FORWARD=1` trainer-side:
> adapted experts and adapted dense linears (qkv/o, GDN projections, shared
> expert) fold their delta with the canonical fold and run the CONTRACTED base
> kernels on the merged weights; weight sync ships those exact bytes, so
> serving runs pure-base contracted kernels — run the sampler WITHOUT
> `--enable-lora`. `XORL_BI_TRUNK_LINEAR=1` now composes with adapted linears
> under this flag (drop the old "adapters on qkv → drop the trunk flag" rule).
> lm_head and router `mlp.gate` must still not be LoRA-wrapped. Constraints:
> single-adapter lanes only (multi-adapter hot-swap keeps the uncontracted LoRA
> kernels); EP: alltoall + explicit `XORL_MOE_SGLANG_FUSED_EXPERTS=1`, pending
> the multi-rank gate; weight-mode `cached` unsupported (use strided).

**§6 item 1 becomes:**

> 1. ~~LoRA-path mismatch~~ — CLOSED by the merged-forward contract (measured
>    pre-contract: ~65% outputs bitwise-off, mean 8e-6/max 1.2e-4 per
>    expert-block element; post-contract: expert-block bitwise 0, and the
>    trainer/serving weight bytes are identical by construction). The backward
>    stays stock numerics (grads track the unmerged lane at ~7e-3 rel-L2 —
>    training behavior, not a K3 term).

This contract extends the earlier MoE shared-tree burn-down work.
