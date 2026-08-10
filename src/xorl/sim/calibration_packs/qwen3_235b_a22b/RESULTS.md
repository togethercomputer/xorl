# Qwen3-235B-A22B at 2k Context

Measured: 4 nodes x 8 H100, U1, DP shard 32, EP8, eFSDP4.

| run | gcm | pack | mbs | tok/step | step s | MFU | tok/s tot | tok/s/GPU | peak GB | status |
|-----|-----|-----:|----:|---------:|-------:|----:|----------:|----------:|--------:|--------|
| n4_ep8_bd_pk4096 | before_dispatch | 4096 | 1 | 131,072 | ~18.4 | ~3.0% | ~6,800 | ~213 | 68.3 | OK |
| n4_ep8_bd_pk4096_ga2 | before_dispatch | 4096 | 1 | 262,144 | ~31.3 | ~3.7% | ~8,400 | ~263 | 68.3 | NEW BEST |
| n4_ep8_bd_pk16k | before_dispatch | 16384 | 1 | 524,288 | -- | -- | -- | -- | OOM | FAIL |
