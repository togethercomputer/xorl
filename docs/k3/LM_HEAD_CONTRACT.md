# Final token-probability contract

## Problem

The final projection has two different numerical jobs. The hidden state is
multiplied by the vocabulary matrix, then every vocabulary logit participates
in one log-sum-exp normalization. A serving engine and a trainer can agree on
the hidden state but still assign different probability to the selected token
if either the projection's K reduction or the vocabulary reduction changes
order. Materializing a full fp32 vocabulary matrix is also too expensive at
production vocabulary sizes.

## Fix

The current `head_v2_selected_logprob` kernel performs the projection and
computes fixed-order vocabulary statistics from its fp32 accumulator in one
launch. `HEAD_V2_BLOCK_K` fixes the projection's accumulation order and
`HEAD_V2_STATS_TILE_N` fixes the log-sum-exp tree. The merge kernel combines
those tile statistics in one explicit pairwise tree. Changing either constant
requires a new cross-engine bitwise gate.

The earlier `bi_lm_head_selected_logprob` path remains available with
`XORL_BI_HEAD_V2=0`. It materializes one vocabulary chunk at a time, records
the same maximum, exponential sum, and selected logit, then merges chunks in
pinned order. This rollback is exact but uses more launches.

The trainer exposes this as `ce_mode="bi_fused"`. Its custom autograd function
saves the exact forward LSE and recomputes the conventional closed-form CE
gradient by chunks. Only forward values enter the train/serve equality
contract; backward remains ordinary checked training numerics.

The path fails closed for unsupported tensor parallelism, non-unit scoring
temperature, Z-loss, non-CUDA tensors, and non-BF16 hidden or weight tensors.
`lm_head_fp32: true` selects the fp32-class scoring contract without keeping a
second fp32 copy of the LM-head weight.

## Verification

Run on a Hopper GPU:

```bash
pytest tests/ops/test_bi_fused_lm_head.py -v
```

The gate compares forward probabilities and loss with an eager fp32 reference,
checks backward gradients, repeats the same rows alone and co-batched for exact
equality, and verifies every unsupported mode raises explicitly.
