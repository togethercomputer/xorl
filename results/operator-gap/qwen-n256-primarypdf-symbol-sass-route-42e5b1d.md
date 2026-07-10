# qwen N256 Primary PDF Symbol SASS + Route Gate

Date: 2026-07-10 UTC

Source base: `42e5b1d3ff537ed0a988d73fcaa0b88650fbf510`

Evidence worktree: `/tmp/xorl-oss-qwen-n256-primarypdf-42e5b1d-codex`

Packaged commit worktree: `/tmp/xorl-oss-qwen-n256-primarypdf-applycheck-42e5b1d-20260710T0010Z`

Packaged local commit: `237c6d3e0ffb15cfd7a9d360075dbe889f3400e1`

## Verdict

PASS for the first executor-boundary gate: the qwen4b-l2 exact N256 head-dX row is identified as row `40` on the current route `78/44/14`, `smem=151552`, shape `(1024, 2560, 151936)`, `ntiles=80`, and the exported `qwen_n256_headdx_exact_pdf_entry` symbol compiles with grouped `HGMMA.64x256` issue, no device calls, no local memory, and no stack.

This is not a runtime promotion. The packaged source intentionally keeps `MK_QWEN_N256_HEAD_DX_SIDECAR_STEP=1` guarded as inspection-only because the previously measured host-visible prefix/entry/post route was timing-negative. The next useful work is an executor-resident integration or an explicit parity smoke before any timing claim.

## Source Package

- Replay patch: `results/qwen-n256-primarypdf-source-42e5b1d-20260710T0010Z.patch`
- Patch sha256: `de8c442f69d1f39afb85c69f8475402fbf9881bb45bbf86c32d6e759d313bd58`
- Patch size: `827` lines / `36137` bytes
- Clean apply-check: PASS from base `42e5b1d3ff537ed0a988d73fcaa0b88650fbf510`
- Apply-check worktree: `/tmp/xorl-oss-qwen-n256-primarypdf-applycheck-42e5b1d-20260710T0010Z`
- Local package commit: `237c6d3e0ffb15cfd7a9d360075dbe889f3400e1`
- Package commit static checks: `git diff --check` PASS, `py_compile mk.py model.py` PASS

Touched source files:

- `experiments/fused-training-megakernel/mk.py`
- `experiments/fused-training-megakernel/model.py`
- `experiments/fused-training-megakernel/megakernel.cu`
- `experiments/fused-training-megakernel/ops.cuh`

## SASS Gate

Canonical corrected SASS evidence came from `results/qwen-n256-primarypdf-sass-gate-42e5b1d-20260710T0009Z.log`, which ended `PASS True` and `JOB_DONE rc=0`.

Summary artifact:

- `results/qwen-n256-primarypdf-sass-summary-42e5b1d-20260710T0009Z.json`
- sha256 `62f54f65c533c1e65d3eaee42f30fae384dc552728c614dcf2ac627e3d549b2d`
- text summary sha256 `f24998d989f6d1abcc9c6b3494492efa58e480b8e575f83dc7ff494c95ad2aab`

Compiled image:

- `.so` sha256 `6b07a23b3ab8390fb89ba6b5c7b6ee6c95c6e9d9500c1fc083c0cdc6e117b840`
- SASS file sha256 `e08e1c5e49d201a58f28612462661a3dacbd8bf2c54ff2e5a8b022c25792aaff`
- resusage sha256 `9fc038c8e4f4a049e1834af3fe4d7971fb5b9d71002d34ca3d72681fba414688`

Target symbol result:

- symbol: `qwen_n256_headdx_exact_pdf_entry`
- `HGMMA=12`
- `HGMMA.64x256=12`
- `WARPGROUP.DEPBAR=2`
- `max_hgmma_between_depbar=12`
- `CALL=0`
- `LDL_STL=0`
- `REG=168`
- `STACK=0`
- `SHARED=1264`
- `LOCAL=0`

The target symbol has the exact grouped issue property the standalone emitted-contract lane required, but now in a qwen-visible source package and exported PDF-entry symbol. The corrected full-image dump also shows the current main megakernel symbols are still singleton-drained (`megakernel_pdf`: `HGMMA=224`, `HGMMA.64x256=60`, `WARPGROUP.DEPBAR=224`, `max_hgmma_between_depbar=1`, `CALL=126`, `STACK=80`), so this gate proves the exported exact target only, not the main PDF executor path.

## Route Contract

Host-only route helper:

- `results/qwen_n256_primarypdf_route_contract_42e5b1d.py`
- sha256 `87a00812f2c20263867c7e0a2269427c55b73ef0727bf18588d4810e20a6efd2`

Route summary:

- `results/qwen-n256-primarypdf-route-contract-42e5b1d-20260710T0010Z.json`
- sha256 `52cbf01741e218f87cc780d6060e2c2763c60dade3c351bf0b0015612cc6ed27`
- text summary sha256 `f1ad483492b25a3afb31aa4c8ae47adf152c9744683d73ce26c7e851ca2cd5ef`

Route checks:

- route: `78/44/14`
- smem: `151552`
- row: `40`
- row shape: `[1024, 2560, 151936]`
- row layout: `NN`
- row `ntiles`: `80`
- TMA table arg present: `args[20] = 1` in the host stub
- stage3: true
- nmajor: true
- n256: true
- c_f32: true
- cutpoint symbol: `qwen_n256_headdx_exact_pdf_entry`
- direct rejoin dependents: `[42, 43]`
- prefix subprogram: `40` instructions, critical path `23`
- entry subprogram: `1` instruction, critical path `1`, indices `[40]`
- post subprogram: `37` instructions, critical path `20`

All route checks passed: `route_shape`, `one_head_row`, `head_row_contract`, `cutpoint`, `split_plan`, `subprograms`, and `api_available`.

## Duplicate Wrapper Audit

There were duplicate compile-only wrappers for stamp `20260710T0003Z`. The unregistered earlier helper completed successfully and wrote a supporting positive SASS summary. The registered wrapper with log `results/qwen-n256-primarypdf-sass-gate-42e5b1d-20260710T0003Z.log` only reached `build.ninja` and left no `JOB_DONE`, object, `.so`, or traceback.

A fresh `20260710T0006Z` relaunch collided with another duplicate wrapper and was terminated during cleanup with exit `143`; it is not used as evidence. The corrected direct-log `20260710T0009Z` SASS artifact and the independent `20260710T0010Z` host route contract are the artifacts to trust.

## Remaining Work

1. Run a real-GPU parity smoke for the explicit entry path only if a free GPU is available: build qwen4b-l2, construct prefix/entry/post subprograms, call `Program.run_qwen_n256_headdx_exact_cutpoint()` explicitly, and compare against the original PDF row output.
2. Do not promote the current host-visible prefix/entry/post path as a timing path. The old exact-entry graph gate already showed that style is timing-negative.
3. If parity is clean, the real promotion candidate should move the grouped exact row into an executor-resident primary PDF path, family-specific PDF image, or equivalent single-kernel path that avoids host-visible fragmentation.
4. Keep other local GPU ordinals untouched. Use an idle local GPU only, otherwise remote isolated is appropriate for runtime parity/timing gates.
