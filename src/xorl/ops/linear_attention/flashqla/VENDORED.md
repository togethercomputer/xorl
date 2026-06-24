# Vendored: FlashQLA

This directory is a vendored copy of [FlashQLA](https://github.com/QwenLM/FlashQLA),
the Qwen team's fused TileLang kernels for Gated Delta Rule (GDN) linear attention.

- **Upstream:** https://github.com/QwenLM/FlashQLA
- **Commit:** `6ef4858b5446e05bd461d9658d877e548182dbcb` (2026-05-07)
- **License:** MIT (see `LICENSE` in this directory)

## Local modifications

1. **Import prefix:** every `flash_qla.*` import was rewritten to
   `xorl.ops.linear_attention.flashqla.*` so the package resolves under the xorl
   namespace.
2. **`prefer_instruction="tma"`** added to every producer global→shared `T.copy`
   load in the Hopper kernels (`chunk/hopper/{fused_fwd,fused_bwd,prepare_h,kkt_solve}.py`:
   q/k/v/a/do/h loads). Upstream relied on tilelang auto-selecting TMA tensor copies
   for these loads; tilelang ≥0.1.10 demotes guarded multi-dim sub-tile loads to
   synchronous SIMT copies (no producer/consumer overlap → ~4-5x slower), so we force
   TMA via the 0.1.10 `T.copy(..., prefer_instruction="tma")` API (PR #2303).

3. **`gemm_v1` patch hook:** this package's `__init__.py` calls
   `xorl.ops.linear_attention.tilelang_gemm_v1.patch()` before importing the kernels.
   Upstream uses `T.gemm_v1`, which stock tilelang >=0.1.9 no longer exposes; the in-repo
   shim re-adds it (fast `tl::gemm_ss` template via the `tl_gemm` builtin) by monkeypatching
   stock tilelang — **no tilelang fork**. The fp32 triangular-solve gemms in `kkt_solve` stay
   on stock `T.gemm` (inline MMA + the 0.1.10 fp32→tf32 fix); all other gemms (incl. kkt's
   bf16 K@Kᵀ) use `T.gemm_v1` (the fast template).

4. **`robust_kkt_solve` (kkt NaN fix):** `ops/gated_delta_rule/chunk/__init__.py` no longer
   calls FlashQLA's bespoke `kkt_solve` kernel — that block-recursive triangular inverse
   produces **NaN at the production shape** (≈`num_chunks*num_heads ≳ 1536`, e.g. 32 value
   heads at seq≥4096 — the Qwen3.5/3.6-35B-A3B GDN shape) because its Schur-combine runs
   through Hopper TF32 MMA and the inverse explodes. We instead compute the *identical* matrix
   `(I + StrictLower(b·K·Kᵀ))⁻¹` via FLA's `chunk_scaled_dot_kkt_fwd(g=None)` + `solve_tril`,
   which is numerically robust (cos 0.99999 vs reference, finite at H=32) and produces the same
   `[B,T,H,64]` unit-lower-triangular bf16 `a` tensor, a drop-in for `fused_gdr_fwd`/`fused_gdr_h`.
   Same substitution is used by the CP bridge (`../../../flashqla_cp.py`). The original 4-head
   parity tests never hit this; `tests/ops/test_flashqla_gdn.py` now parametrizes H∈{4,32}.

Validated on H100 (parity vs FLA at H∈{4,32}, seq 4096: fwd cos 0.99999, bwd grad cos 0.9999,
all finite).

## Requirements / caveats

- **Hopper (SM90) only.** Imported lazily (the package validates SM90 at import time).
- **Stock upstream tilelang (no fork)** with the `tl_gemm` builtin (>=0.1.10) AND
  `prefer_instruction` (tile-ai/tilelang PR #2303). #2303 is merged upstream but post-`v0.1.10`,
  so it is **not** in the PyPI `tilelang==0.1.10` wheel. `pyproject.toml` therefore pins a prebuilt
  wheel of stock `tile-ai/tilelang@a8d93798` (includes #2303), hosted on
  `togethercomputer/xorl-wheels` (`tilelang_0.1.10_cu131`, cp38-abi3, CUDA 13.1) — switch to PyPI
  `tilelang>=0.1.11` once a release carrying #2303 ships. Also needs `apache-tvm-ffi>=0.1.10`.
  `gemm_v1` itself is supplied in-repo by `../tilelang_gemm_v1.py` (vendored shim, quack-style
  injection), so the wheel is unmodified upstream. Stock tilelang's unified `T.gemm` (tileop
  inline-wgmma) is ~4-5x slower here.

## Re-syncing with upstream

```bash
git clone https://github.com/QwenLM/FlashQLA.git
cp -r FlashQLA/flash_qla/. src/xorl/ops/linear_attention/flashqla/
DEST=src/xorl/ops/linear_attention/flashqla
# 1. namespace the imports
grep -rl 'flash_qla' "$DEST" --include='*.py' \
  | xargs sed -i 's/\bflash_qla\b/xorl.ops.linear_attention.flashqla/g'
# 2. (tilelang >=0.1.10) add prefer_instruction="tma" to producer global->shared
#    T.copy loads in chunk/hopper/{fused_fwd,fused_bwd,prepare_h,kkt_solve}.py.
#    gemm_v1 calls are kept as-is (forked tilelang provides the op).
```
