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
