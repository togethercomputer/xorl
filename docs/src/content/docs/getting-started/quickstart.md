---
title: "Quick Start"
---


## Choosing a Training Mode

| Mode | Use when | Entry point |
|---|---|---|
| **Local training** | Offline SFT/pretraining with a fixed dataset | `torchrun -m xorl.cli.train` |
| **Server training** | Online RL (PPO, GRPO) where an external loop drives training step-by-step | `python -m xorl.server.launcher` |

Most users start with **local training**. Use server training when you need an RL orchestrator to control the training loop.

## Local Training (single node)

```bash
# 8-GPU full fine-tuning of Qwen3-8B
torchrun --nproc_per_node=8 -m xorl.cli.train \
    examples/local/dummy/configs/full/qwen3_8b.yaml
```

## Local Training with a real dataset

Create a YAML config:

```yaml
# my_config.yaml
model:
  model_path: Qwen/Qwen3-8B
  attn_implementation: flash_attention_3

data:
  datasets:
    - path: /data/my_dataset.jsonl
      type: tokenized
      max_seq_len: 8192
  select_columns: [input_ids, labels]
  sample_packing_method: sequential
  sample_packing_sequence_len: 8192

train:
  output_dir: outputs/qwen3_8b_ft
  data_parallel_mode: fsdp2
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  optimizer: adamw
  lr: 1e-5
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  enable_full_shard: true
  init_device: meta
  save_steps: 500
```

Launch:
```bash
torchrun --nproc_per_node=8 -m xorl.cli.train my_config.yaml
```

## Server Training (for RL loops)

Start the training server:
```bash
python -m xorl.server.launcher \
    --mode auto \
    --config examples/server/configs/full/qwen3_8b_full.yaml \
    --api-port 5555
```

Then drive training from a Python client. All training endpoints use a **two-phase async pattern**: the POST returns a `request_id` immediately, and you poll `/api/v1/retrieve_future` to get the actual result.

```python
import requests
import time

base_url = "http://localhost:5555"

# Check health
requests.get(f"{base_url}/health").json()

# Forward + backward (phase 1: submit)
future = requests.post(f"{base_url}/api/v1/forward_backward", json={
    "forward_backward_input": {
        "data": [{"model_input": {"input_ids": [...]}, "loss_fn_inputs": {"labels": [...]}}],
        "loss_fn": "causallm_loss",
    },
}).json()

# Phase 2: poll for result
while True:
    result = requests.post(f"{base_url}/api/v1/retrieve_future", json={
        "request_id": future["request_id"],
    }).json()
    if "request_id" not in result:  # result ready (not a TryAgainResponse)
        break
    time.sleep(0.5)
print(result)

# Optimizer step (same two-phase pattern)
future = requests.post(f"{base_url}/api/v1/optim_step", json={
    "adam_params": {"learning_rate": 1e-5},
    "gradient_clip": 1.0,
}).json()
```

## LoRA Fine-tuning

```yaml
# Add to any config's train section:
lora:
  enable_lora: true
  lora_rank: 16
  lora_alpha: 16
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  save_lora_only: true
```

## QLoRA Fine-tuning

```yaml
lora:
  enable_qlora: true
  quant_format: nf4          # or nvfp4, block_fp8
  lora_rank: 16
  lora_alpha: 16
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

## MoE Training (Qwen3-30B-A3B)

```bash
torchrun --nproc_per_node=8 -m xorl.cli.train \
    examples/local/dummy/configs/full/qwen3_30b_a3b_pp2_ep4_cp4_muon.yaml
```

## Override Config Fields on CLI

```bash
torchrun --nproc_per_node=8 -m xorl.cli.train config.yaml \
    --train.lr 2e-5 \
    --train.output_dir outputs/my_run \
    --data.sample_packing_sequence_len 16384
```
