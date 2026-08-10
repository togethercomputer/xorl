# Qwen3.6-35B-A3B 8k Calibration

- Model: `Qwen/Qwen3.6-35B-A3B`
- Data: synthetic tokenized data, sequential packing, `sample_packing_sequence_len: 8193`
- Hardware: 4 nodes x 8 H100 GPUs
- Topology: `pp=1`, `tp=1`, `ring=1`, `ulysses=1`, `dp_shard=32`, `ep=8`, `ep_fsdp=4`
- Training: AdamW, BF16 mixed precision, FSDP2, full-layer recompute, DeepEP
- Runtime: `deepep_num_sms: 72`, `deepep_buffer_size_gb: 2.0`, `deepep_async_combine: true`
- Compiler: `enable_compile: true`, `gradient_checkpointing_method: recompute_full_layer`

Reference shape: `micro_batch_size: 8`, `global_batch_size: 256`.

| metric | value |
| --- | ---: |
| tokens/sec | ~261.0K |
| step time | ~8.04s |
| MFU | ~16.2% |
| allocated memory | ~56.4GB |
| allocator retries | 0 |

The adjacent `mbs=10` observation fit but slowed to ~133.7K tok/s with allocator retries. The separate measured mbs6
row reached 254.6K tok/s but failed its matching static K3 gate: mean `2.1339`, p95 `0.0834`, max `394.1871`.
These rows are useful calibration and failure-boundary evidence, not a correctness-promotable recipe.
