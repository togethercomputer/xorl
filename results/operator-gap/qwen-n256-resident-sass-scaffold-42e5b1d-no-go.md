# qwen N256 same-launch resident exact-row SASS scaffold @42e5b1d

Date: 2026-07-10
Claim: `qwen N256 same-launch resident exact-row SASS scaffold @42e5b1d`
Status: NO-GO at SASS/resource gate.

## Scope

This lane tested the narrow direct resident-branch scaffold in an isolated source
worktree. The scaffold keeps one `megakernel_pdf` cooperative launch and adds an
opt-in exact qwen N256 head-dX branch before generic dispatch, guarded by the
same row-40 matcher used by the exact PDF-entry symbol.

This was compile/disassembly only. It did not run parity, timing, graph capture,
default-policy promotion, remote isolated, or any GPU kernel. Shared source was not edited,
local GPU was not touched, and the separate local GPU integration scoreboard was left
alone.

## Source Package

- source tree: `/tmp/xorl-oss-qwen-n256-resident-42e5b1d-codex`
- base frontier: `42e5b1d` (`default qwen head-dx rmsdot partials`)
- imported primary-PDF package checkpoint: `efacfe3`
- resident scaffold checkpoint: `51d4144`
- source status at gate close: clean
- patch artifact:
  `results/qwen-n256-resident-sass-scaffold-42e5b1d-20260710T0020Z.patch`
  - sha256: `6a379f7f9429dae9f93b30ece9a59983b44de6279dbba9ec9f509cbab3766d57`
  - size: `147` lines / `7271` bytes

Source changes in the isolated tree:

- `megakernel.cu`
  - moves the exact qwen N256 head-dX match macro before `megakernel_pdf`
  - adds a resident direct call to `op_gemm_wgmma_n256_head_dx_exact_impl()`
    before generic dispatch in both PDF loop variants
- `mk.py`
  - adds opt-in `MK_GEMM_N256_HEAD_DX_RESIDENT_ENTRY`
  - adds `gemm_n256_head_dx_resident_entry` load flag
  - adds `_hdxres` suffix
  - forces exact-PDF-entry image support on when resident is enabled
- `ops.cuh`
  - lets the resident flag use the same force-inline exact body as the
    standalone exact PDF-entry symbol

## Gate Artifacts

- helper: `results/qwen_n256_resident_sass_gate_42e5b1d.py`
  - sha256: `803b6c035d2755941daaeb160d3adb19d0c2beb878772e5d0627bc754d3cbce0`
  - size: `376` lines / `13645` bytes
- log: `results/qwen-n256-resident-sass-gate-42e5b1d-20260710T0024Z.log`
  - sha256: `ee4e9aa29273b16ee96f103f4f8546c6df45c702bc103dc2b718baff188525c9`
  - size: `10` lines / `647` bytes
  - ended: `PASS False`, `JOB_DONE rc=2 2026-07-10T00:21:50Z`
- summary JSON:
  `results/qwen-n256-resident-sass-summary-42e5b1d-20260710T0024Z.json`
  - sha256: `d607b0aa85adcea5422fd6f27afe08cdc4383e24eb7fe6eda49765c50245d906`
  - size: `308` lines / `7553` bytes
- summary text:
  `results/qwen-n256-resident-sass-summary-42e5b1d-20260710T0024Z.txt`
  - sha256: `b38a5d72828082dbe9cf520dc06c847554a69faf9c5816f0443926378f01c559`
  - size: `17` lines / `1896` bytes
- extension:
  `/tmp/torchext-qwen-n256-resident-sass-42e5b1d-20260710T0024Z/..._hdxres_hdxrmsdot_pdf240p.so`
  - sha256: `ff1c78a385bcec1b6060761c0a9e646b65084d534ed8a03bd7be0a2ce64c39be`
- SASS:
  `results/qwen-n256-resident-sass-42e5b1d-20260710T0024Z/..._hdxres_hdxrmsdot_pdf240p.sass`
  - sha256: `94eeb4eb6d69accbc9b0927fbccdd552fc76a4ebce9095fe4343a31af7a253c8`
  - size: `875616` lines / `119954041` bytes
- resource usage:
  `results/qwen-n256-resident-sass-42e5b1d-20260710T0024Z/..._hdxres_hdxrmsdot_pdf240p.resusage.txt`
  - sha256: `bb881d707fe4a1b1d480add028af0dc432b96e3e3cbe30483c617855c696c8d0`
  - size: `27` lines / `1211` bytes

## Result

The direct resident branch was present in `megakernel_pdf`, but ptxas still
serialized the resident target. The SASS gate failed two checks:

- `target_grouped=false`
- `target_stack_not_worse=false`

Checks that passed:

- `target_unique=true`
- `target_has_hgmma=true`
- `target_has_hgmma64x256=true`
- `target_contains_resident_copy=true`
- `target_local0=true`
- `target_calls_not_worse=true`
- `entry_unique=true`
- `entry_still_grouped=true`
- `entry_local0=true`

Resident `megakernel_pdf` stats:

- `HGMMA=236`
- `HGMMA.64x256=72`
- `WARPGROUP.DEPBAR=236`
- `max_hgmma_between_depbar=1`
- `CALL=126`
- `LDL=52`
- `STL=138`
- `REG=168`
- `STACK=96`
- `LOCAL=0`
- histogram: `{'1': 236}`

Baseline contrast from the prior primary-PDF image:

- resident baseline `megakernel_pdf`: `HGMMA=224`, `HGMMA.64x256=60`,
  `DEPBAR=224`, `max=1`, `CALL=126`, `STACK=80`
- exact exported entry: `HGMMA.64x256=12`, `DEPBAR=2`, `max=12`,
  `CALL0`, `STACK0`, `LOCAL0`

Interpretation: the new resident branch added exactly the 12 expected
`64x256` HGMMA instructions to `megakernel_pdf`, but they were absorbed into
the same singleton-drained resident symbol. The branch also increased resident
stack from `80` to `96`. This is a real SASS no-go, not a compile wrapper
failure.

## Closed Boundary

This closes the direct resident-branch scaffold as a promotion path. It is too
close to the already closed direct-dispatch/source-spelling family: the exact
body is present, but the monolithic resident PDF symbol still gives ptxas enough
generic fallback/live-range pressure to singleton-drain the grouped issue.

Do not rerun this scaffold with only small branch spelling changes. A useful
next source gate needs a stronger physical partition than "one more branch
inside `megakernel_pdf`".

## Next Useful Gate

Next gate remains:

`qwen N256 generated resident exact-row PDF image SASS gate`

The generated/partitioned candidate must make the grouped exact row-40 body the
primary resident target rather than a branch inside the full generic PDF symbol.
Minimum first-gate requirements:

- one cooperative launch, or no critical-path host split
- preserve route `78/44/14`
- preserve `smem_bytes=151552`
- preserve row `40`, shape `1024x2560x151936`, `ntiles=80`
- preserve direct rejoin dependents `[42,43]`
- resident target `max_hgmma_between_depbar > 1`
- `LOCAL0`
- no unacceptable stack growth
- no WGMMA-bearing call boundary around the grouped target
- proof generic fallback is unreachable from the grouped resident target

## Verification

Commands already recorded as passing before compile:

```bash
git diff --check
python3 -m py_compile experiments/fused-training-megakernel/mk.py experiments/fused-training-megakernel/model.py
python3 -m py_compile results/qwen_n256_resident_sass_gate_42e5b1d.py
```

Compile/disassembly command ended with `rc=2` because the SASS gate failed, not
because the build failed. The object, `.so`, SASS dump, resource-usage dump,
JSON summary, and text summary were all produced.
