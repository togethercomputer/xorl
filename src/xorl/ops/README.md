# xorl.ops

Compute kernels and autograd ops. This README is the map: what each
subtree is, what may be edited, and where things are headed
(reorganization plan: [issue #78](https://github.com/togethercomputer/xorl/issues/78)).

## Edit policies

Three kinds of code live here with three different rules:

| kind | rule |
| --- | --- |
| **Vendored** — `_vendored/` (`quack/`, `flashqla/`) | Never hand-edit, lint, or reformat. Each tree carries a `VENDORED.md` with provenance and the local-patch ledger. First-party tooling skips them (`[tool.ruff]` excludes in `pyproject.toml`; top-level `exclude:` in `.pre-commit-config.yaml`). |
| **Byte-contract-gated** — `bi_families_v2.py` (sha256-gated), `batch_invariant_ops.py` (parity-diffable twin of SGLang's copy; edits must consider the serving side) | Vendored byte-identical into the serving engine; both copies are sha256-gated. Any edit here without the paired serving-side edit breaks the gate. It keeps the engine's formatting (black, 88 columns) and is excluded from all rewriting hooks. |
| **First-party** — everything else | Normal rules. |

## Map (current)

- `exact/` — the **serving-parity (exact) contract family**: byte-pinned
  programs shared with the serving engine (#78 phase 3). Three members are
  aliased rather than moved: `bi_families_v2.py` (sha256-gated),
  `batch_invariant_ops.py` (diffable parity twin of SGLang's copy), and
  `exact_sampling_transforms.py` (in-flight in #74). Old root-level module
  paths are compat stubs for one deprecation cycle.
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
