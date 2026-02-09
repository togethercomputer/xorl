# Xorl

A distributed training framework for large language models, built on top of [Xorl](https://github.com/xorl-org/Xorl).

## Requirements

- Python 3.12
- CUDA 12.9+
- PyTorch 2.9+

## Installation

```bash
git clone --recurse-submodules git@github.com:xorl-org/xorl.git
cd xorl

# Option 1: Install into an existing conda or venv environment
pip install -e .
# or
uv pip install -e .

# Option 2: Create a new .venv environment with uv
uv sync
```

> If you already cloned without `--recurse-submodules`, run:
> ```bash
> git submodule update --init --recursive
> ```

## Training Modes

Xorl supports two training modes:

| Mode | Description | Use case |
|------|-------------|----------|
| **Local** | Single-script training via `torchrun` | Standalone fine-tuning jobs |
| **Server** | API server with distributed workers | Integration with external orchestrators (e.g. [xorl_client](https://github.com/xorl-org/xorl_client)) |

## Local Training

Training is launched via `torchrun` with a YAML config. See [`examples/local/`](examples/local/) for complete examples.

### Quick Start

```bash
cd examples/local/xorl_coder

# Run on 8 GPUs
torchrun --nproc_per_node=8 --master_port=29501 \
  -m xorl.cli.train configs/qwen3_4b_instruct_2507.yaml
```

### Config

Training configs are YAML files with three sections:

- **`model`** -- model path and attention implementation
- **`data`** -- dataset, packing, and dataloader settings
- **`train`** -- optimizer, parallelism, checkpointing, and logging

See [`examples/local/xorl_coder/configs/qwen3_4b_instruct_2507.yaml`](examples/local/xorl_coder/configs/qwen3_4b_instruct_2507.yaml) for a full example.

## Server Training

The server mode exposes a training API that external clients can drive. The server manages model loading, distributed workers, data packing, and gradient updates. The client controls the training loop: what data to train on, when to step the optimizer, and when to save checkpoints.

### Quick Start

**1. Start the server** (launches workers automatically):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m xorl.server.launcher \
  --mode auto \
  --config examples/server/tinker_examples/configs/qwen3_4b_instruct_2507.yaml \
  --api-port 6000
```

**2. Run a training client:**

```bash
cd examples/server/tinker_examples
python example_sft.py --config.base_url http://localhost:6000
```

See [`examples/server/`](examples/server/) for more details.

### Server Config

Server configs are flat YAML files parsed by `ServerArguments`. Key sections:

```yaml
# Model
model_path: Qwen/Qwen3-4B-Instruct-2507
attn_implementation: flash_attention_2   # eager, sdpa, flash_attention_2, flash_attention_3

# Parallelism
data_parallel_mode: fsdp2
ulysses_parallel_size: 4                 # Ulysses sequence parallelism

# Memory & Performance
ce_mode: compiled                        # eager, compiled
enable_mixed_precision: true
enable_gradient_checkpointing: true

# Data Processing
packing_seq_len: 128000                  # Max packed sequence length
enable_packing: true

# LoRA
enable_lora: true
lora_rank: 32
lora_target_modules: ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
```

### Launcher CLI

```
python -m xorl.server.launcher [OPTIONS]

Options:
  --mode {auto,connect}     auto: launch workers with torchrun
                            connect: attach to existing workers
  --config PATH             Server config YAML (required for auto mode)
  --api-host HOST           API server host (default: 0.0.0.0)
  --api-port PORT           API server port (default: auto)
  --operation-timeout SECS  Engine operation timeout (default: 600)
  --log-level LEVEL         DEBUG, INFO, WARNING, ERROR (default: INFO)
  --nnodes N                Number of nodes (default: 1)
  --master-addr ADDR        Torch distributed master address (default: 127.0.0.1)
  --master-port PORT        Torch distributed master port (default: 29500)
  --server.<field> VALUE    Override any ServerArguments field
```

### Cross-Entropy Modes

The `ce_mode` config controls how cross-entropy loss is computed:

| Mode | Description |
|------|-------------|
| `compiled` | **Default.** `torch.compile` with `auto_chunker` -- avoids materializing `[BT, V]` logits |
| `eager` | `F.cross_entropy` baseline (may OOM at long sequences) |
