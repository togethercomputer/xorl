---
title: "SFT on No Robots"
---

[`examples/server/no_robot_sft/`](https://github.com/togethercomputer/xorl-internal/tree/main/examples/server/no_robot_sft) — Supervised fine-tuning on the [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) dataset.

**What it demonstrates:**
- LoRA SFT training loop driven by `xorl_client.TrainingClient`
- Online tokenization using tinker-cookbook renderers
- Initial validation step (forward-only, no gradients)
- Linear learning rate decay
- Periodic checkpoint saving and resume support
- Per-token NLL metrics

**Run:**

```bash
# 1. Start the training server (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m xorl.server.launcher \
    --mode auto \
    --config examples/server/configs/lora/qwen3_8b_lora.yaml \
    --api-port 6000

# 2. Run SFT (in another terminal)
pip install xorl-client tinker-cookbook
python examples/server/no_robot_sft/run_sft.py \
    --config.base_url http://localhost:6000 \
    --config.model_name Qwen/Qwen3-8B \
    --config.lora_rank 32
```

The checked-in server config above loads `Qwen/Qwen3-8B` with LoRA rank 32.
`run_sft.py` still defaults to an older 4B example, so pass the overrides above until those defaults are updated.

**Config options:**

| Field | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:6000` | Training server URL |
| `model_name` | `Qwen/Qwen3-4B-Instruct-2507` | Model name (for tokenizer). Override to `Qwen/Qwen3-8B` for the 8B LoRA config above. |
| `batch_size` | 128 | Training batch size |
| `learning_rate` | 1e-4 | Peak learning rate |
| `max_length` | 32768 | Max sequence length |
| `lora_rank` | 64 | LoRA rank. Override to `32` to match `examples/server/configs/lora/qwen3_8b_lora.yaml`. |
| `save_every` | 20 | Checkpoint every N steps (0 = disabled) |
