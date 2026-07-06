# S4096 Combine-Unroll Post-QKRoPE Promotion

Checkpoint: clean detached sibling worktree
`/home/apanda/xorl-oss-combine-clean-005ad84` at `005ad84`.

Scope: recheck `MK_ATTN_COMBINE_UNROLL=1` for exact H256/D64/S4096 after the
S3072/S4096/S8192 fused-qkrope n128 promotion. No shared-checkout source was
used for the timing run; the harness imported from the clean worktree.

Earlier post-qkrope run:

- Log: `results/mkv3-p4b-s4096-combine-unroll-post-qkrope-20260706T2105Z.log`
- Rejected as contaminated. Both medians were around 5.7ms, far outside the
  expected S4096 band, and paired means contradicted the medians.

Clean repeat:

- Log:
  `results/mkv3-p4b-s4096-combine-unroll-clean005ad84-repeat-20260706T2128Z.log`
- Default-first: `3151.62us` default vs `3133.38us` unroll,
  default-minus-unroll `+18.35us`, paired mean `+18.59us`, unroll wins
  `160/160`.
- Variant-first: `3139.23us` default vs `3131.50us` unroll,
  default-minus-unroll `+7.25us`, paired mean `+7.21us`, unroll wins
  `137/160`.
- Route unchanged: `n_instr=188`, `critical_path=80`, `gated=63`,
  `ATTN_COMBINE=8/1344`, `ATTN_FWD_WG=12/1008`, `ATTN_DKV_WG=12/848`,
  `ATTN_DQ_WG=12/848`.
- Parity clean: loss diff within `3e-06`; worst selected gradient relative
  error `3.846151e-03` on `emb`.

Decision:

- Promote exact H256/D64/S4096 alongside the existing exact S8192
  `OP_ATTN_COMBINE` unroll default.
- Keep S3072 excluded because the prior source probe regressed both orders
  (`-2.53us`, `18/40`; `-5.70us`, `8/40`).
- Keep S2048 excluded; it has not shown a clean two-order win in this routing
  regime.

Implementation:

- Add `_H256_D64_COMBINE_UNROLL_S = (4096, 8192)` in `model.py`.
- Gate `attn_combine_unroll_default` on exact H256/L4/D64/I768/V8192 with
  `S in _H256_D64_COMBINE_UNROLL_S`.
- `MK_ATTN_COMBINE_UNROLL=0` remains the forced-old escape hatch.

Promoted-default confirmation:

- Log:
  `results/mkv3-p4b-s4096-combine-unroll-promoted-clean005ad84-20260706T2134Z.log`
- Default-first against forced old: `3127.34us` default vs `3140.00us` old,
  default-minus-old `-13.38us`, paired mean `-12.72us`, old wins `4/160`.
- Variant-first against forced old: `3119.74us` default vs `3143.04us` old,
  default-minus-old `-23.76us`, paired mean `-23.51us`, old wins `1/160`.
- Parity clean: worst selected gradient relative error below `3.91e-03`.

Validation:

- `python -m py_compile experiments/fused-training-megakernel/model.py
  experiments/fused-training-megakernel/mk.py
  experiments/fused-training-megakernel/profile_df.py`
- `ruff check experiments/fused-training-megakernel/model.py`
- `git diff --check`
- trailing-whitespace scan over touched/new files
- `test_model.py`, log:
  `results/mkv3-p4b-s4096-combine-unroll-promote-test-model-20260706T2134Z.log`
