# Vendored: QuACK (partial fork)

This directory is a vendored, locally-patched subset of
[QuACK](https://github.com/Dao-AILab/quack) (Wentao Guo, Ted Zadouri,
Tri Dao — CuTeDSL kernels), snapshotted at upstream `0.4.1`
(see `__init__.py:__version__`).

It is a *partial* fork: modules in this tree still import helper modules
(`copy_utils`, `layout_utils`, …) from the separate PyPI dependency
`quack-kernels` pinned in `pyproject.toml`, so the pinned PyPI version and
this tree must stay compatible.

Known local additions/patches (not upstream):

- `cute_dsl_elf_fix.py` — works around cutlass#3161 (duplicate `.text`
  section flags break MCJIT in multi-process loads).
- `cute_dsl_mlir_threading.py` — works around cutlass#3062 (leaked LLVM
  thread pools exhaust pthreads across compiles).

## Edit policy

Do not hand-edit, lint, or reformat files in this tree. First-party tooling
skips it (see `pyproject.toml` `[tool.ruff]` excludes and the top-level
`exclude:` in `.pre-commit-config.yaml`). Fixes should go upstream first;
a local patch that cannot wait must be listed above with its upstream
issue/PR so the next re-vendor can reconcile it.
