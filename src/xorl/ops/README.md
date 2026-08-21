# xorl.ops

Compute kernels and autograd ops. This README is the map: what each
subtree is, what may be edited, and where things are headed
(reorganization plan: [issue #78](https://github.com/togethercomputer/xorl/issues/78)).

## Edit policies

Three kinds of code live here with three different rules:

| kind | rule |
| --- | --- |
| **Vendored** — `_vendored/` (`quack/`, `flashqla/`) | Never hand-edit, lint, or reformat. Each tree carries a `VENDORED.md` with provenance and the local-patch ledger. First-party tooling skips them (`[tool.ruff]` excludes in `pyproject.toml`; top-level `exclude:` in `.pre-commit-config.yaml`). |
| **Serving-parity twins** — `sglang/` (`bi_families_v2.py`, `batch_invariant_ops.py`) | Modules mirrored into/from the serving engine. `bi_families_v2.py` is sha256-gated byte-identical (keeps the engine's black-88 formatting, excluded from all rewriting hooks); `batch_invariant_ops.py` is vendored-adapted and stays a single diffable file. Edits require considering the paired serving-side copy. |
| **First-party** — everything else | Normal rules. |

## Map (current)

- `exact/` — the **serving-parity (exact) contract family**: byte-pinned
  programs shared with the serving engine (#78 phase 3), including the
  replay contract (`sampling_transforms.py`).
- `sglang/` — the literal serving-engine twins (`bi_families_v2.py`,
  `batch_invariant_ops.py`); see the edit-policy table. Old root-level
  module paths for both packages are compat stubs for one deprecation
  cycle.
- `loss/` — the CE/selected-logprob kernel stack. The RL/supervised
  objective functions live in `xorl/objectives/` (#78 phase 2); old module
  paths here are compat stubs for one deprecation cycle.
- `moe/` — MoE expert compute backends (triton/quack/native, LoRA variants).
- `linear_attention/` — GDN/linear-attention kernels. The `GatedDeltaNet`
  layer class lives in `models/layers/gated_deltanet.py` (#78 phase 4); the
  old paths re-export it lazily for one deprecation cycle.
- `ssm/` — Mamba-2 kernels; `Mamba2Mixer` likewise lives in
  `models/layers/mamba2_mixer.py`.
- `quantize/` — NF4/INT4/FP4/FP8 quantization codecs and fake-quant ops.
- `families/` — model-family-specific kernels (`glm5/`, `dsv4/`, #78
  phase 4). Kept under `ops/` (not inside `models/transformers/<family>/`)
  because the model packages have import side effects (DSV4 registers with
  the HF Auto registries on import); old paths are alias stubs.
- `_vendored/` — vendored trees (see above); old paths (`ops/quack`, `ops/linear_attention/flashqla`) are alias stubs for one deprecation cycle.

## What does NOT belong here

New `nn.Module` layer classes (→ `models/layers/`), RL objectives
(→ `xorl/objectives/`), and orchestration logic. `ops/` is
for kernels and the autograd boundaries directly over them.

## Name glossary

Decoder ring for the historical jargon (candidates for renaming are tracked
on #78; several terms are user-facing config or shared vocabulary with the
serving engine and cannot be renamed unilaterally):

| name | meaning |
| --- | --- |
| **batch-invariant / `bi_`** | a kernel whose per-element reduction order does not depend on batch composition, so the same token produces the same bits in any batch — the property that makes trainer/sampler logprobs comparable bitwise. |
| **K3 / "zero-K3"** | the k3 KL-divergence estimator between trainer and sampler logprobs for the same tokens; "zero-K3" = bit-identical train/serve forward, the goal of the exact contracts. |
| **Class-A / Class-B RoPE** | the two RoPE numerics classes across the trainer/sampler pair: Class A rounds to bf16 per op (8 rounding points); Class B computes one fp32 chain with a single final round (SGLang's fused CUDA rope and its compiled RL-lane path). `rope_fp32_single_round: true` selects Class B. |
| **canonical MoE reduce** | the pinned fixed-order expert-contribution reduction (contributor-leaf arithmetic + final FP64-accumulator cast) shared with serving, versioned by `CANONICAL_MOE_REDUCE_VERSION`. |
| **one-round SwiGLU** | the exact-contract SwiGLU with a single FP32 rounding point (`exact_fp32_silu_and_mul`), vs the generic fused SwiGLU. |
| **families v1 / v2** | versioned batch-invariant kernel families (norms, LM head); v2 is the epilogue-stats generation. `bi_families_v2` is the serving twin module. |
| **GKN layout** | grouped expert-weight layout `[G=experts, K=in_features, N=out_features]`. |
| **exact** | shorthand for "serving-parity byte contract": the forward reproduces the serving engine's bits, not just its math. |
