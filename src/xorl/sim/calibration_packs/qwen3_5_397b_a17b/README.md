# Qwen3.5-397B-A17B Short-Context Calibration

- Model: `Qwen/Qwen3.5-397B-A17B`
- Data: synthetic tokenized data, sequential packing, `max_seq_len=4096`
- Hardware: 8 nodes x 8 H100 GPUs
- Topology: `pp=1`, `tp=1`, `ring=1`, `ulysses=1`, `dp_shard=64`, `ep=32`, `ep_fsdp=2`
- Runtime: Quack MoE, DeepEP dispatch, SMS48, balanced synthetic routing

MFU uses 989 BF16 TFLOPS/GPU as the H100 denominator.

| trial | tok/s | tok/s/GPU | TFLOPS/GPU | MFU | step | correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| R75 | 59.217K | 925.3 | 93.4 | 9.44% | 22.299s | raw-speed only |
| R73 | 59.188K | 924.8 | 93.4 | 9.44% | 22.300s | static K3 pass |
| R70 | 54.227K | 847.3 | 85.5 | 8.65% | 24.561s | not promoted |
| R69 | 52.545K | 821.0 | 82.9 | 8.38% | 20.004s | not promoted |
| R67 | 48.474K | 757.4 | 76.5 | 7.74% | 21.804s | not promoted |
| R66 | 43.770K | 683.9 | 69.0 | 6.98% | 18.204s | not promoted |

R73 passed its static replay over 129 output tokens with mean K3 `0.000475`, p95 `0.001335`, and max `0.028437`.
R75 changes DeepEP combine to asynchronous mode and is not promotable without a matching gate.
