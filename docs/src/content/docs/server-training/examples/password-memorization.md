---
title: "Password Memorization"
---

[`examples/server/password_memorization/`](https://github.com/togethercomputer/xorl-internal/tree/main/examples/server/password_memorization) — End-to-end test for the full **training → weight sync → inference** pipeline. Trains a model to memorize 3 secret project codes via SFT, syncs weights to a running xorl-sglang instance, and queries inference to verify recall.

**Run:**

```bash
# 1. Start training server
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m xorl.server.launcher \
    --mode auto \
    --config examples/server/configs/full/qwen3_8b_full.yaml \
    --api-port 6000

# 2. Start xorl-sglang inference (in another terminal)
CUDA_VISIBLE_DEVICES=4 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B-FP8 --port 30000

# 3. Run the test (in another terminal)
python examples/server/password_memorization/run_password_test.py \
    --model Qwen/Qwen3-8B --steps 16 --lr 1e-5
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model name |
| `--steps` | 64 | Total training steps |
| `--lr` | 1e-4 | Peak learning rate |
| `--lr-schedule` | constant | `constant`, `cosine`, or `warmup_cosine` |
| `--sync-quant` | fp8 | Sync quantization: `fp8` or `none` |
| `--train-url` | `http://localhost:6000` | Training server URL |
| `--infer-url` | `http://localhost:30000` | Inference endpoint URL |

## Weight sync pipeline

| Training mode | Sync path |
|---|---|
| Full-weight bf16 | bf16 → fp8 requant → SGLang |
| LoRA | bf16 base + LoRA merged → fp8 requant → SGLang |
| QLoRA nvfp4 | nvfp4 dequant → bf16 merged → fp8 requant → SGLang |
| QLoRA block_fp8 | fp8 dequant → bf16 merged → fp8 requant → SGLang |
| QLoRA nf4 | nf4 dequant → bf16 merged → fp8 requant → SGLang |

## Test matrix

**Qwen3-8B (4x H100):**

| Mode | Steps | LR | Schedule | Result |
|---|---|---|---|---|
| Full-weight bf16 | 16 | 1e-5 | constant | 3/3 |
| LoRA rank 32 | 32 | 1e-4 | constant | 3/3 |
| QLoRA nvfp4 | 64 | 5e-5 | cosine | 3/3 |
| QLoRA block_fp8 | 64 | 5e-4 | constant | 3/3 |

**Qwen3-Coder-30B-A3B (4-8x H100):**

| Mode | Parallelism | Steps | LR | Schedule | Result |
|---|---|---|---|---|---|
| Full-weight bf16 | SP=4, shard=2 | 32 | 1e-5 | constant | 3/3 |
| LoRA rank 32 | SP=4 | 32 | 1e-4 | constant | 3/3 |
| QLoRA nvfp4 | EP=4, SP=4 | 128 | 5e-4 | cosine | 3/3 |

**Qwen3-235B-A22B (8x H100, cross-node inference):**

| Mode | Parallelism | Steps | LR | Schedule | Result |
|---|---|---|---|---|---|
| QLoRA nvfp4 | EP=8, SP=8 | 128 | 5e-4 | cosine | 3/3 |
| QLoRA nf4 | EP=8, SP=8 | 128 | 5e-4 | cosine | 3/3 |

See the [example README](https://github.com/togethercomputer/xorl-internal/tree/main/examples/server/password_memorization/README.md) for the full test matrix and detailed setup instructions.
