# Attention train/serve contract

## Why the default can differ

Attention reduces over the key/value sequence. Trainer and sampler agreement
therefore depends on both the kernel implementation and the way that reduction
is partitioned.

FA3 and FA4, or two separately built FA3 kernels, are distinct numerical
programs even when they implement the same formula. A KV-cache entry point may
also select several KV splits from the batch shape, compute independent partial
records, and merge them afterward. That makes one request's result depend on
unrelated requests in the batch.

## Contract

Trainer and sampler must use a qualified backend pair and the same per-request
KV reduction schedule. Where a backend supports split-KV, an exact lane uses an
unsplit request-local reduction or another fixed split-and-merge schedule that
does not depend on batch neighbors.

On Hopper, the FA4 CUTE implementation does not use split-KV. The relevant
contract is therefore backend/build identity and argument parity. FA3 KV-cache
paths can split and require an explicit schedule.

The packed trainer normally uses a varlen entry point while serving uses a
KV-cache entry point. Diagnostic routes in
[`flash_attention.py`](../../src/xorl/models/layers/attention/backend/flash_attention.py)
allow the trainer to invoke a serving-style page-size-1 cache path when
localizing entry-point or build differences. They are diagnostic tools, not a
universal production requirement.

Unsupported cross-attention layouts, mismatched spans, or unavailable requested
backends raise rather than falling back silently to eager attention.

## Verification

```bash
pytest tests/ops/test_attention.py -q
```

The conventional tests verify dispatch, metadata construction, forwarded
arguments, split selection, and fail-closed behavior. A release qualification
must additionally replay real post-RoPE Q/K/V operands through the exact
trainer and sampler builds. Hardware-specific performance results remain with
the benchmark that produced them.
