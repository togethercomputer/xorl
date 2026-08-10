# RMSNorm train/serve contract

## Why matching the formula is insufficient

RMSNorm forms a row, reduces its squared values, applies one reciprocal square
root, and scales the row. Trainer and serving call sites can represent the same
logical row differently:

| Site | Trainer representation | Serving representation |
|---|---|---|
| Layer-0 input norm | one BF16 row | one BF16 row |
| Layer > 0 input norm | residual already materialized | hidden state plus residual |
| Post-attention norm | hidden state plus residual | hidden state plus residual |
| Final norm | residual already materialized | hidden state plus residual |
| Q/K norm | one head row | one head row |

Serving commonly carries the residual across a layer boundary so that its next
norm can fuse the addition. Training commonly materializes the addition before
returning from the preceding layer. The contract first makes both forms meet at
the same BF16 row, then normalizes that row with the same arithmetic.

The older kernels also differed in reduction association, reciprocal operation,
and BF16 rounding boundaries. A source-level `tl.sum` expression does not by
itself fix an addition tree: compiler layout decisions made by surrounding loads
and stores can change its lowering.

## RMSNorm-v2 arithmetic

RMSNorm-v2 has two independent compile-time choices:

- `HAS_RESIDUAL` controls whether the canonical BF16 row is formed from two
  inputs or supplied directly.
- `ZERO_CENTERED` controls whether the FP32 affine scale is `weight` or
  `1 + weight`. It does not determine whether a residual is present.

After forming the row, both engines:

1. convert the canonical BF16 values to FP32;
2. square them;
3. reduce them through an explicit adjacent-pair tree;
4. divide by the hidden size, add epsilon, and apply one `rsqrt`;
5. multiply by the FP32 affine scale; and
6. cast the output once to BF16.

The explicit tree avoids a compiler-owned reduction over more than two values.
Its association is a function of the hidden dimension, not the number of rows
in the batch.

## Fused and split execution

Most shapes use one program per row and complete the operation in one kernel
launch. Very wide rows with little row-level parallelism use three kernels:

1. compute the same 512-value tile sums;
2. combine those FP32 partials in the same higher-level order and compute
   `inv_rms`; and
3. normalize the tiles.

The two implementations factor the same arithmetic tree and must remain
bitwise identical. The runtime may choose between them for utilization, but
the choice is not allowed to change output bits.

## Verification

Run the conventional contract tests on a CUDA system:

```bash
pytest tests/ops/test_bi_families_v2_norm.py -q
pytest tests/ops/test_bi_families_v2_norm_dispatch.py -q
pytest tests/ops/test_bi_families_v2_dispatch.py -q
pytest tests/models/test_rmsnorm_family_contract.py -q
pytest tests/models/test_rmsnorm_family_cross_engine.py -q
```

The tests cover residual and no-residual calls, zero-centered scaling, fused and
split execution, model-site admission, and cross-engine equality. Performance
measurements and hardware-specific benchmark outputs are deliberately not
stored in the public source tree.
