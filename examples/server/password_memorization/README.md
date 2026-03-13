# Password Memorization — End-to-End Weight Sync Tests

End-to-end tests for the full training → weight sync → inference pipeline. A model is trained to memorize 3 secret project codes via SFT, weights are synced to a running SGLang inference endpoint, and the inference model is queried to verify recall.

## Usage

All tests use the unified `run_password_test.py` script:

```bash
python run_password_test.py --model <MODEL> --steps <N> --lr <LR> [options]
```

### Examples

```bash
# Qwen3-8B full bf16 → FP8 inference
python run_password_test.py --model Qwen/Qwen3-8B --steps 16 --lr 1e-5

# Qwen3-8B LoRA → FP8 inference
python run_password_test.py --model Qwen/Qwen3-8B --steps 32 --lr 1e-4

# Qwen3-8B QLoRA nvfp4 → FP8, cosine LR
python run_password_test.py --model Qwen/Qwen3-8B --steps 64 --lr 5e-5 --lr-schedule cosine

# Qwen3-8B QLoRA block_fp8 → FP8
python run_password_test.py --model Qwen/Qwen3-8B --steps 64 --lr 5e-4

# Qwen3-Coder-30B full bf16 → FP8
python run_password_test.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --steps 32 --lr 1e-5

# Qwen3-Coder-30B QLoRA nvfp4 → FP8, cosine LR
python run_password_test.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --steps 128 --lr 5e-4 --lr-schedule cosine

# Qwen3-Coder-30B QLoRA block_fp8 → FP8, warmup+cosine
python run_password_test.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --steps 128 --lr 5e-4 --lr-schedule warmup_cosine --warmup-steps 64

# Qwen3-235B QLoRA nf4 → remote FP8 TP=4, cross-node sync
python run_password_test.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --steps 128 --lr 5e-4 --lr-schedule cosine \
    --infer-url http://remote-node:30000 --master-address local-node

# bf16 sync (no FP8 requant)
python run_password_test.py --model Qwen/Qwen3-8B --steps 48 --lr 5e-5 --sync-quant none
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model name |
| `--steps` | 64 | Total training steps |
| `--lr` | 1e-4 | Peak learning rate |
| `--lr-schedule` | constant | `constant`, `cosine`, or `warmup_cosine` |
| `--lr-min-ratio` | 0.01 | lr_min = lr × ratio (for cosine schedules) |
| `--warmup-steps` | 0 | Constant-LR warmup steps before cosine decay |
| `--sync-quant` | fp8 | Sync quantization: `fp8` (block e4m3) or `none` |
| `--train-url` | http://localhost:6000 | Training server URL |
| `--infer-url` | http://localhost:30000 | Inference endpoint URL(s), space-separated |
| `--master-address` | localhost | Master address for NCCL weight sync |
| `--log-interval` | 16 | Print loss every N steps |

---

## Test Matrix

### Qwen3-8B (4× H100)

| Training mode | Config | Inference | Steps | LR | Schedule | Result |
|--------------|--------|-----------|-------|----|----------|--------|
| Full-weight bf16 | `full/qwen3_8b_full.yaml` | FP8 (tp=1) | 16 | 1e-5 | constant | 3/3 ✓ |
| LoRA rank 32 | `lora/qwen3_8b_lora.yaml` | FP8 (tp=1) | 32 | 1e-4 | constant | 3/3 ✓ |
| QLoRA nvfp4 rank 32 | `qlora/qwen3_8b_qlora_nvfp4.yaml` | FP8 (tp=1) | 64 | 5e-5 | cosine | 3/3 ✓ |
| QLoRA block_fp8 rank 32 | `qlora/qwen3_8b_qlora_block_fp8.yaml` | FP8 (tp=1) | 64 | 5e-4 | constant | 3/3 ✓ |

### Qwen3-Coder-30B-A3B (4–8× H100)

| Training mode | Config | Parallelism | Inference | Steps | LR | Schedule | Result |
|--------------|--------|-------------|-----------|-------|----|----------|--------|
| Full-weight bf16 | `full/qwen3_coder_30b_a3b_full.yaml` | SP=4 shard=2 | FP8 (tp=2) | 32 | 1e-5 | constant | 3/3 ✓ |
| LoRA rank 32 | `lora/qwen3_coder_30b_a3b_lora.yaml` | SP=4 | FP8 (tp=1) | 32 | 1e-4 | constant | 3/3 ✓ |
| QLoRA block_fp8 rank 32 | `qlora/qwen3_coder_30b_a3b_qlora.yaml` | EP=4 SP=4 | FP8 (tp=1) | 32 | 5e-4 | constant | 3/3 ✓ |
| QLoRA nvfp4 rank 32 | `qlora/qwen3_30b_a3b_qlora_nvfp4.yaml` | EP=4 SP=4 | FP8 (tp=2) | 128 | 5e-4 | cosine | 3/3 ✓ |
| QLoRA nf4 rank 32 | `qlora/qwen3_30b_a3b_qlora_nf4.yaml` | EP=4 SP=4 | FP8 (tp=2) | 128 | 5e-4 | cosine | 3/3 ✓ |

### Qwen3-235B-A22B (8× H100, cross-node inference)

| Training mode | Config | Parallelism | Inference | Steps | LR | Schedule | Result |
|--------------|--------|-------------|-----------|-------|----|----------|--------|
| QLoRA nvfp4 rank 32 | `qlora/qwen3_235b_a22b_qlora_nvfp4.yaml` | EP=8 SP=8 | FP8 (tp=4, remote) | 128 | 5e-4 | cosine | 3/3 ✓ |
| QLoRA nf4 rank 32 | `qlora/qwen3_235b_a22b_qlora_nf4.yaml` | EP=8 SP=8 | FP8 (tp=4, remote) | 128 | 5e-4 | cosine | 3/3 ✓ |

---

## Setup

Start the training server and SGLang inference in separate tmux windows, then run the test.

```bash
# Training server
CUDA_VISIBLE_DEVICES=<GPUs> python -m xorl.server.launcher \
  --mode auto --config examples/server/configs/<CONFIG>.yaml \
  --api-port 6000 --log-level INFO

# SGLang inference
CUDA_VISIBLE_DEVICES=<GPU> python -m sglang.launch_server \
  --model-path <MODEL> --port 30000 --host 0.0.0.0 \
  --mem-fraction-static 0.88 [--tp N] [--dtype bfloat16]
```

### SGLang model paths

| Training model | Inference model |
|---------------|-----------------|
| `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B-FP8` (tp=1) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (tp=2) |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (tp=4, remote node) |

---

## Weight Sync Pipeline

| Training | Sync path |
|----------|-----------|
| Full bf16 | bf16 weights → fp8 requant → SGLang |
| LoRA | bf16 base + LoRA delta merged → fp8 requant → SGLang |
| QLoRA nvfp4 | nvfp4 dequant → bf16 merged → fp8 requant → SGLang |
| QLoRA block_fp8 | fp8 dequant → bf16 merged → fp8 requant → SGLang |
| QLoRA nf4 | nf4 dequant → bf16 merged → fp8 requant → SGLang |

FP8 re-quantization uses block-wise e4m3 with `weight_block_size=[128, 128]`, configured via `set_sync_quantization`.

### Sync Timing

| Model | Training mode | Params | Sync time |
|-------|--------------|--------|-----------|
| Qwen3-8B | LoRA | 652 | ~6s |
| Qwen3-Coder-30B | QLoRA nvfp4 EP=4 | 37,492 | ~10s |
| Qwen3-Coder-30B | Full bf16 | 37,492 | ~8s |
| Qwen3-235B-A22B | QLoRA nf4 EP=8 | 73,418 | ~20s (cross-node) |

---

## Notes

- **QLoRA block_fp8 → FP8 sync**: requires `lr=5e-4` (10× higher than nvfp4). The LoRA delta must exceed the fp8 quantization step size to survive dequant→merge→fp8-requant.
- **Qwen3-235B**: use the instruct tokenizer for training — the base model chat template injects `<think>` tags even with `enable_thinking=False`.
- **30B MoE with EP**: cosine LR schedule is important — constant LR causes loss oscillation around ~1.0.
- **NF4**: quantizes bf16 weights on-the-fly; no pre-quantized checkpoint needed.
