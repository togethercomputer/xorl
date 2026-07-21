# Batch-invariant GEMM contract

## Problem

Trainer and sampler GEMMs may select different accumulation schedules as the
number of rows changes. Floating-point addition is not associative, so a token
scored alone can then receive different logits when scored in a larger batch.
That is an off-policy error even when both processes hold identical weights.

## Contract

`matmul_kernel_persistent` fixes `BLOCK_SIZE_K` per dtype. This is the
bit-relevant axis because it determines the order of the dot-product reduction.
The output tile, group size, pipeline depth, and warp count do not split the K
reduction, so they remain performance-tuning axes after passing the bitwise
gate.

`bi_gemm_configs.py` contains the shared, shape-keyed table. Each entry keeps
the dtype's pinned K tile, compares bitwise with the baseline configuration,
and checks that an identical row keeps identical output bits across row-count
buckets. Set `XORL_BI_GEMM_CONFIG_TABLE=0` to use the pinned baseline table.
Launches that exceed a Triton version's shared-memory limit also fall back to
that baseline.

The optional Hopper DeepGEMM route is enabled only when its BF16 NN result has
the same bits as the persistent kernel for the admitted shapes. It has a
separate kill switch, `XORL_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`.

## Verification

Run the table and cross-batch gates on a Hopper GPU:

```bash
pytest tests/ops/test_bi_gemm_config_table.py -v
pytest tests/models/test_batch_invariance_dense.py -v
```

The first test locks the K tile, compares every tuned configuration with the
baseline, repeats a row across M buckets, checks the DeepGEMM alternative when
installed, and exercises both kill switches. The second test verifies the
end-to-end batch-invariant operator surface.

This contract covers the forward values that enter token scoring. Training
backward remains on the framework's ordinary differentiable path; enabling a
global inference interpose inside a gradient graph is outside this PR.
