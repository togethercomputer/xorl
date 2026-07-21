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

`bi_lm_head_selected_logprob` performs the projection in fixed vocabulary
chunks with the shared persistent BF16 GEMM and an fp32 accumulator. It records
each chunk's maximum, fixed-order exponential sum, and selected-token logit.
`_lm_head_lse_merge_kernel` then merges chunk statistics in increasing chunk
order. `BI_LM_HEAD_VOCAB_CHUNK` and `_BI_LM_HEAD_STATS_BLOCK` are numerical
contract constants: changing either requires a new cross-engine bitwise gate.

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
