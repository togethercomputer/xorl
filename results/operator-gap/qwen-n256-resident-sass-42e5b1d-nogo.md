# qwen N256 Resident PDF SASS Gate No-Go

Date: 2026-07-10
Agent: codex
Claim: qwen N256 same-launch resident exact-row SASS scaffold @42e5b1d

## Verdict

No-go for the same-launch resident branch in the broad `megakernel_pdf` image.

The resident branch compiled and the exact row body was present inside
`megakernel_pdf`, but the broad image still singleton-drained every HGMMA:
`max_hgmma_between_depbar=1`. It also increased `megakernel_pdf` stack from the
prior baseline `80` bytes to `96` bytes. The exported exact-entry control in the
same image stayed grouped (`max=12`, `STACK0`, `LOCAL0`), so the collapse is the
resident full-image context, not the exact body spelling.

Do not promote this source scaffold to runtime parity/timing.

## Source Package

- Base repo: `<repo-root>`
- Isolated worktree: `/tmp/xorl-oss-qwen-n256-resident-42e5b1d-codex`
- Imported primary-PDF checkpoint: `efacfe3`
- Resident scaffold checkpoint: `51d4144b49baec70bae1c591e1e05d01494043aa`
- Shared checkout source edits: none

Source patch:

- `results/qwen-n256-resident-sass-scaffold-42e5b1d-20260710T0020Z.patch`
- sha256: `6a379f7f9429dae9f93b30ece9a59983b44de6279dbba9ec9f509cbab3766d57`
- size: `147` lines / `7271` bytes

The source scaffold added:

- `mk.py`: opt-in `MK_GEMM_N256_HEAD_DX_RESIDENT_ENTRY` /
  `gemm_n256_head_dx_resident_entry`, suffix `_hdxres`, and exact-PDF-entry
  image enablement when resident mode is set.
- `megakernel.cu`: exact qwen N256 head-dX match macro moved before
  `megakernel_pdf`, with a resident direct call to
  `op_gemm_wgmma_n256_head_dx_exact_impl()` before generic dispatch in both PDF
  loop variants.
- `ops.cuh`: resident mode uses the same force-inline exact body as the
  exported exact-entry image.

Static checks before the SASS gate:

- `git diff --check`: PASS
- `python3 -m py_compile experiments/fused-training-megakernel/mk.py experiments/fused-training-megakernel/model.py`: PASS
- Helper `python3 -m py_compile results/qwen_n256_resident_sass_gate_42e5b1d.py`: PASS

## Compile/SASS Gate

The artifact label is `20260710T0024Z`; the actual log start was
`2026-07-10T00:18:46Z`.

Command class:

- `CUDA_VISIBLE_DEVICES=<unset>` empty
- `TORCH_EXTENSIONS_DIR=/tmp/torchext-qwen-n256-resident-sass-42e5b1d-20260710T0024Z`
- `TORCH_CUDA_ARCH_LIST=9.0a`
- `MAX_JOBS=1`
- Helper: `results/qwen_n256_resident_sass_gate_42e5b1d.py`

Helper:

- `results/qwen_n256_resident_sass_gate_42e5b1d.py`
- sha256: `803b6c035d2755941daaeb160d3adb19d0c2beb878772e5d0627bc754d3cbce0`
- size: `376` lines / `13645` bytes

Log:

- `results/qwen-n256-resident-sass-gate-42e5b1d-20260710T0024Z.log`
- sha256: `ee4e9aa29273b16ee96f103f4f8546c6df45c702bc103dc2b718baff188525c9`
- size: `10` lines / `647` bytes
- ended: `PASS False`, `JOB_DONE rc=2 2026-07-10T00:21:50Z`

Summary:

- `results/qwen-n256-resident-sass-summary-42e5b1d-20260710T0024Z.json`
- sha256: `d607b0aa85adcea5422fd6f27afe08cdc4383e24eb7fe6eda49765c50245d906`
- size: `308` lines / `7553` bytes

Text summary:

- `results/qwen-n256-resident-sass-summary-42e5b1d-20260710T0024Z.txt`
- sha256: `b38a5d72828082dbe9cf520dc06c847554a69faf9c5816f0443926378f01c559`
- size: `17` lines / `1896` bytes

Generated SASS:

- `results/qwen-n256-resident-sass-42e5b1d-20260710T0024Z/xorl_megakernel_afexpr_adkva_aflog_lex2_rms2560_ceb2_cefix_cewr_qkbc_qkbc128_swfma_swb4w_gmbar_n256ntold_gtma_nttma_ntstpdfreg_ntstnosync_ntscbnd_hdxexpdfg3xpe_hdxres_hdxrmsdot_pdf240p.sass`
- sha256: `94eeb4eb6d69accbc9b0927fbccdd552fc76a4ebce9095fe4343a31af7a253c8`
- size: `875616` lines / `119954041` bytes

Resource usage:

- `results/qwen-n256-resident-sass-42e5b1d-20260710T0024Z/xorl_megakernel_afexpr_adkva_aflog_lex2_rms2560_ceb2_cefix_cewr_qkbc_qkbc128_swfma_swb4w_gmbar_n256ntold_gtma_nttma_ntstpdfreg_ntstnosync_ntscbnd_hdxexpdfg3xpe_hdxres_hdxrmsdot_pdf240p.resusage.txt`
- sha256: `bb881d707fe4a1b1d480add028af0dc432b96e3e3cbe30483c617855c696c8d0`
- size: `27` lines / `1211` bytes

## Gate Results

Gate checks:

```json
{
  "entry_local0": true,
  "entry_still_grouped": true,
  "entry_unique": true,
  "target_calls_not_worse": true,
  "target_contains_resident_copy": true,
  "target_grouped": false,
  "target_has_hgmma": true,
  "target_has_hgmma64x256": true,
  "target_local0": true,
  "target_stack_not_worse": false,
  "target_unique": true
}
```

Key SASS/resource rows:

| Function | HGMMA | HGMMA.64x256 | DEPBAR | Max Run | CALL | REG | STACK | LOCAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `megakernel_pdf` | 236 | 72 | 236 | 1 | 126 | 168 | 96 | 0 |
| `qwen_n256_headdx_exact_pdf_entry` | 12 | 12 | 2 | 12 | 0 | 168 | 0 | 0 |

Interpretation:

- The resident branch is present: `megakernel_pdf` `HGMMA.64x256` increased
  from the prior canonical primary-PDF image's `60` to `72`.
- It did not preserve grouped issue in the broad resident image:
  `megakernel_pdf` `DEPBAR=236` and `max_hgmma_between_depbar=1`.
- It regressed stack: `megakernel_pdf STACK=96` versus prior canonical
  `STACK=80`.
- It did not introduce local memory: `LOCAL=0`.
- It did not increase broad-image call count: `CALL=126`.
- The exact exported control remains valid: `qwen_n256_headdx_exact_pdf_entry`
  has `12` HGMMA.64x256, `2` DEPBAR, `max=12`, `CALL0`, `STACK0`, `LOCAL0`.

## Next Action

Close this lane as no-go. The next non-duplicate N256 source lane should not be
another same broad-`megakernel_pdf` branch. It needs either:

- a family-specific primary PDF image where the generic WGMMA fallback is not in
  the target symbol's reachable codegen context, or
- a lower-level raw emitted contract inserted through a separate primary symbol
  that can be joined without the already-refuted host-visible prefix/entry/post
  split.

Runtime parity/timing for this resident branch is explicitly not justified by
the SASS result.

No GPU kernel was launched by this gate, no remote isolated job was used, and local GPU was not
touched.
