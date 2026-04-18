# Server Training Examples (Tinker/Tomi API)

These examples demonstrate server-mode training where Xorl runs as a training server and an external client (using the [xorl_client](https://github.com/xorl-org/xorl_client) SDK) drives the training loop.

## Prerequisites

Install the xorl_client client SDK and tinker-cookbook:

```bash
pip install xorl_client tinker-cookbook
```

## Starting the Server

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m xorl.server.launcher \
  --mode auto \
  --config ../configs/lora/qwen3_8b_lora.yaml \
  --api-port 6000 \
  --log-level DEBUG
```

The launcher will:
1. Calculate the world size from parallelism settings in the config
2. Launch distributed workers via `torchrun`
3. Start the API server on the specified port
4. Wait for workers to initialize (may take a few minutes for large models)

## Examples

### SFT (Supervised Fine-Tuning)

[`run_sft.py`](run_sft.py) -- Minimal LoRA SFT training loop on the [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) dataset.

```bash
python run_sft.py \
  --config.base_url http://localhost:6000 \
  --config.model_name Qwen/Qwen3-8B \
  --config.lora_rank 32
```

Key features:
- Online tokenization using tinker-cookbook renderers
- Initial validation step (forward-only, no gradients)
- Linear learning rate decay
- Periodic checkpoint saving and resume support
- Per-token NLL metrics via `compute_mean_nll`

The checked-in server config above loads `Qwen/Qwen3-8B` with LoRA rank 32.
`run_sft.py` still defaults to an older 4B example, so the command overrides those defaults explicitly.

**Config options** (passed via `--config.<field>`):

| Field | Default | Description |
|-------|---------|-------------|
| `base_url` | `http://localhost:6000` | Xorl server URL |
| `model_name` | `Qwen/Qwen3-4B-Instruct-2507` | Model name (for tokenizer) |
| `batch_size` | 128 | Training batch size |
| `learning_rate` | 1e-4 | Peak learning rate |
| `max_length` | 32768 | Max sequence length |
| `lora_rank` | 64 | LoRA rank |
| `save_every` | 20 | Save checkpoint every N steps (0 = disabled) |

## Server Configs

### [`configs/lora/qwen3_8b_lora.yaml`](../configs/lora/qwen3_8b_lora.yaml)

Qwen3-8B dense model on 4 GPUs with LoRA enabled.

### [`configs/lora/qwen3_coder_30b_a3b_lora.yaml`](../configs/lora/qwen3_coder_30b_a3b_lora.yaml)

Qwen3-Coder-30B-A3B MoE model on 4 GPUs with LoRA enabled.

### Config Reference

```yaml
# Model
model_path: Qwen/Qwen3-8B
tokenizer_path: Qwen/Qwen3-8B
attn_implementation: flash_attention_3

# Parallelism (world_size = dp_rep * dp_shard * ulysses * cp)
data_parallel_mode: fsdp2                # ddp, fsdp, fsdp2
ulysses_parallel_size: 1                 # Ulysses sequence parallelism
data_parallel_replicate_size: 1
data_parallel_shard_size: 4

# Memory & Performance
ce_mode: compiled                        # eager, compiled
enable_mixed_precision: true
enable_gradient_checkpointing: true
enable_full_shard: true
enable_activation_offload: false
init_device: meta

# Data Processing
sample_packing_sequence_len: 128000      # Max packed sequence length
enable_packing: true                     # Pack multiple samples per micro-batch

# LoRA
enable_lora: true
lora_rank: 32
lora_alpha: 32
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## Training Loop Pattern

The xorl_client SDK provides a `TrainingClient` that communicates with the Xorl server. A typical training loop looks like:

```python
import xorl_client

# Connect to server
service = xorl_client.ServiceClient(base_url="http://localhost:6000")
client = service.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=32,
    model_id="my-run",
)

# Training step
result = client.forward_backward(datum_list, loss_fn="cross_entropy").result()
client.optim_step(xorl_client.AdamParams(learning_rate=1e-4)).result()

# Save checkpoint
client.save_state("/path/to/checkpoint")
```

### Available Loss Functions

| Loss function | Description |
|---------------|-------------|
| `cross_entropy` / `causallm_loss` | Standard causal LM loss (SFT) |
| `importance_sampling` | GRPO/RL importance sampling loss |

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `forward_backward(data)` | Forward + backward pass, accumulates gradients |
| `forward(data)` | Forward-only pass (validation, no gradients) |
| `optim_step(params)` | Apply accumulated gradients with optimizer |
| `save_state(path)` | Save full training state (model + optimizer) |
| `save_lora(path)` | Save LoRA adapter weights only |
| `load_state(path)` | Load training state |
