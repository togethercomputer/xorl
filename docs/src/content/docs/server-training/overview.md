---
title: "Server Training"
---

Server training exposes the training loop as a REST API, enabling external processes (RL frameworks, data pipelines, experiment orchestrators) to drive gradient updates step by step. This is the primary mode for online RL training.

## xorl-client

To use server training from Python, install the official client library:

```bash
pip install git+https://github.com/togethercomputer/xorl-client.git
```

The client handles HTTP communication, request serialization, async result polling, and provides typed wrappers around all API endpoints. All examples in this page use `xorl_client`.

---

## Architecture

```
Client (xorl_client / HTTP)
    │
    ▼ HTTP (port 5555)
API Server (FastAPI)
    │
    ▼ ZMQ + NCCL
Runner Dispatcher (rank 0)
    │
    ▼ NCCL
Worker Ranks (1 … N-1)
```

The API server receives requests, forwards them to the model runner via ZMQ, and broadcasts to all GPU workers via NCCL collectives. All ranks participate in every forward/backward.

---

## Starting the Server

### Single Node

```bash
python -m xorl.server.launcher \
    --mode auto \
    --config examples/server/configs/full/qwen3_8b_full.yaml \
    --api-port 5555
```

`--mode auto` launches `torchrun` workers internally and starts the API on the specified port.

### Multi-Node

On worker nodes (one command per node):
```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=<RANK> \
    --master_addr=HEAD_IP --master_port=29500 \
    -m xorl.server.runner.runner_dispatcher \
    --config config.yaml
```

On the head node (connect to already-running workers):
```bash
python -m xorl.server.launcher \
    --mode connect \
    --config config.yaml \
    --api-port 5555 \
    --master-addr HEAD_IP
```

See `scripts/run_multinode_server.sh` for a full multi-node launch script.

### Launcher CLI Options

| Option | Default | Description |
|---|---|---|
| `--mode` | `auto` | `auto` (launch workers) or `connect` (attach to existing) |
| `--config` | required | Path to server YAML config |
| `--api-host` | `0.0.0.0` | Bind address for the REST API |
| `--api-port` | auto | Port for the REST API (auto-finds a free port if not specified) |
| `--nnodes` | `1` | Number of nodes (auto mode only) |
| `--master-addr` | `127.0.0.1` | Head node address |
| `--master-port` | `29500` | torchrun master port |
| `--operation-timeout` | `1800` | Max seconds per operation |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

---

## Server Configuration

Server configs have a flat structure (no `train:` section nesting). See [Server Config Reference](/config-reference/server/) for all fields.

```yaml
# examples/server/configs/full/qwen3_8b_full.yaml
model_path: Qwen/Qwen3-8B
tokenizer_path: Qwen/Qwen3-8B
attn_implementation: flash_attention_3

# Parallelism
data_parallel_mode: fsdp2
data_parallel_shard_size: 8
expert_parallel_size: 1
pipeline_parallel_size: 1
ulysses_parallel_size: 1

# Memory
enable_mixed_precision: true
enable_gradient_checkpointing: true
enable_full_shard: true
init_device: meta

# Packing
sample_packing_sequence_len: 8192
enable_packing: true

# Checkpointing
output_dir: outputs/server_run
ckpt_manager: dcp
```

---

## Sequence Packing

The server packs multiple client-submitted sequences into a single training bin to maximize GPU utilization. Packing is controlled by two config fields:

| Field | Default | Description |
|---|---|---|
| `enable_packing` | `true` | Pack multiple sequences per bin. Set `false` to send one sequence per forward pass. |
| `sample_packing_sequence_len` | `32000` | Bin capacity in tokens. Sequences are concatenated until this limit is reached. |

Unlike local training where packing happens offline during dataset preparation, the server packs sequences **at request time** inside the `RequestProcessor`. The packing algorithm is **sequential** (greedy, preserves submission order): incoming `Datum` objects are concatenated into a bin until `sample_packing_sequence_len` is reached, then a new bin is started.

Each packed bin carries `cu_seqlens` (cumulative sequence lengths) and `position_ids` so Flash Attention treats each original sequence independently within the bin — there is no cross-sequence attention leakage.

For Ring Attention (`ringattn_parallel_size > 1`), each document in the bin must be individually padded to a length divisible by `2 × ringattn_parallel_size` before zigzag sharding. The server handles this padding automatically in `TextSequenceShardCollator` using sequential dummy position IDs for the pad region.

---

## Python Client (`xorl_client`)

### Connecting

```python
import xorl_client

service_client = xorl_client.ServiceClient(base_url="http://localhost:5555")
```

`ServiceClient` is the main entry point. It manages the connection, authentication, and provides factory methods for training clients.

### Creating a training client

**Full-weight training** (no LoRA — server auto-registers `model_id="default"` on startup):

```python
training_client = xorl_client.TrainingClient(
    holder=service_client.holder,
    model_id="default",
    base_model="Qwen/Qwen3-8B",
)
```

**LoRA training** (creates and registers a new adapter):

```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=32,
    model_id="my_adapter",
)
```

### Preparing data

Training data is passed as a list of `Datum` objects:

```python
datum = xorl_client.Datum(
    model_input=xorl_client.ModelInput.from_ints(input_ids),
    loss_fn_inputs={"labels": labels},
)
```

`ModelInput.from_ints` wraps a flat token list into the expected `{"input_ids": [...]}` format.

### Training loop

`forward_backward` and `optim_step` return futures — call `.result()` to block and retrieve the output:

```python
adam_params = xorl_client.AdamParams(
    learning_rate=1e-5,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
)

for step, batch in enumerate(dataloader):
    data = [
        xorl_client.Datum(
            model_input=xorl_client.ModelInput.from_ints(sample["input_ids"]),
            loss_fn_inputs={"labels": sample["labels"]},
        )
        for sample in batch
    ]

    fwd_bwd = training_client.forward_backward(data, loss_fn="causallm_loss")
    optim = training_client.optim_step(adam_params)

    result = fwd_bwd.result()
    optim.result()

    print(f"step={step} loss={result.loss_fn_outputs[0]['loss'].data[0]:.4f}")
```

### Available loss functions

| `loss_fn` | Description |
|---|---|
| `causallm_loss` | Standard causal language model cross-entropy |
| `policy_loss` | PPO-style policy gradient loss |
| `importance_sampling` | Importance-sampling weighted loss for off-policy RL |

---

## REST API Reference

All endpoints are at `http://<host>:<port>/api/v1/`. The xorl_client wraps these — use the Python client for most cases.

### Health Check

```http
GET /health
```
```json
{"status": "healthy", "engine_running": true}
```

### Create Model

Creates and registers a new model/adapter session.

```http
POST /api/v1/create_model
```
```json
{
  "model_id": "my_adapter",
  "base_model": "Qwen/Qwen3-8B",
  "lora_config": {"rank": 32, "alpha": 16}
}
```

### Forward + Backward

Computes loss and accumulates gradients. Call multiple times before `optim_step` for gradient accumulation.

```http
POST /api/v1/forward_backward
```
```json
{
  "model_id": "default",
  "forward_backward_input": {
    "data": [
      {
        "model_input": {"input_ids": [1, 2, 3, ...]},
        "loss_fn_inputs": {"labels": [1, 2, 3, ...]}
      }
    ],
    "loss_fn": "causallm_loss"
  }
}
```

### Optimizer Step

Applies accumulated gradients, clips, and advances the LR scheduler.

```http
POST /api/v1/optim_step
```
```json
{
  "model_id": "default",
  "adam_params": {
    "learning_rate": 1e-5,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8
  },
  "gradient_clip": 1.0
}
```

### Forward Only

Same as `forward_backward` but without gradient accumulation. Used for evaluation or reference model log-prob computation.

```http
POST /api/v1/forward
```

### Save Checkpoint

```http
POST /api/v1/save_weights
```
```json
{"model_id": "default", "path": null}
```
`null` path = auto-timestamped path under `output_dir`.

### Load Checkpoint

```http
POST /api/v1/load_weights
```
```json
{"model_id": "default", "path": "outputs/server_run/weights/default/step_42"}
```

### Weight Sync to Inference

Push current training weights to registered SGLang inference endpoints.

```http
POST /api/v1/sync_inference_weights
```
```json
{
  "master_address": "HEAD_NODE_IP",
  "master_port": 29600,
  "group_name": "sync_group_0",
  "buffer_size_mb": 512
}
```

### Inference Endpoint Management

```http
POST /add_inference_endpoint
```
```json
{"host": "inference-node-01", "port": 30000, "world_size": 8}
```

```http
POST /remove_inference_endpoint
```

### Sleep / Wake

Offload model weights to CPU to free GPU memory while inference runs:

```http
POST /sleep
POST /wake_up
```

### Kill Session

```http
POST /api/v1/kill_session
```
```json
{"model_id": "default", "reset_weights": false}
```

---

## Tinker API Compatibility

xorl is compatible with the **Tinker training API** — a standardized protocol for driving training servers from RL frameworks and experiment orchestrators. Any client built against the Tinker spec works with xorl without modification.

### Tinker-compatible endpoints

| Endpoint | Description |
|---|---|
| `POST /api/v1/create_model` | Create/register a model session |
| `POST /api/v1/unload_model` | Unload and release a session |
| `POST /api/v1/forward_backward` | Forward + backward pass |
| `POST /api/v1/optim_step` | Optimizer step |
| `POST /api/v1/weights_info` | Checkpoint metadata for model loading |
| `GET /api/v1/training_runs` | List training runs |

### Field mappings

xorl automatically maps Tinker's field names to its own format, so Tinker clients work without changes:

| Tinker field | xorl field | Where |
|---|---|---|
| `session_id` | `model_id` | All request types |
| `loss_fn_config` | `loss_fn_params` | `forward_backward` input |
| `lora_config.rank` | `lora_rank` | `create_model` request |
| `model_input.chunks[].tokens` | `model_input.input_ids` | `Datum` model input |

### TensorData format

Tinker uses a typed tensor wire format for all numeric outputs. xorl uses the same format:

```json
{
  "data": [0.423, 0.891, 0.312],
  "dtype": "float32",
  "shape": [3]
}
```

`LossFnOutput` responses always use `TensorData` values:

```json
{
  "loss": {"data": [2.345], "dtype": "float32", "shape": [1]},
  "logprobs": {"data": [...], "dtype": "float32", "shape": [512]},
  "elementwise_loss": {"data": [...], "dtype": "float32", "shape": [512]}
}
```

`logprobs` and `elementwise_loss` are only present when `loss_fn: "per_token_ce"` is used.

---

## RL Training Pattern

```python
import xorl_client

service_client = xorl_client.ServiceClient(base_url="http://localhost:5555")
training_client = xorl_client.TrainingClient(
    holder=service_client.holder,
    model_id="default",
    base_model="Qwen/Qwen3-8B",
)
adam_params = xorl_client.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.95, eps=1e-8)

for rl_step in range(num_rl_steps):
    # 1. Rollout from inference (SGLang)
    samples = sglang_client.generate(prompts, max_tokens=512)

    # 2. Score with reward model
    rewards = reward_model.score(samples)

    # 3. Pack into Datum objects
    data = [
        xorl_client.Datum(
            model_input=xorl_client.ModelInput.from_ints(s.token_ids),
            loss_fn_inputs={"labels": s.labels},
        )
        for s in samples
    ]

    # 4. Gradient accumulation + optimizer step
    fwd_bwd = training_client.forward_backward(
        data,
        loss_fn="importance_sampling",
        loss_fn_params={"eps_clip": 0.2},
    )
    optim = training_client.optim_step(adam_params)
    fwd_bwd.result()
    optim.result()

    # 5. Sync weights to inference every N steps
    if rl_step % sync_interval == 0:
        service_client.sync_inference_weights(...)
```

---

## Multi-Adapter (LoRA) Support

The server supports multiple named LoRA adapters, switchable per request via `model_id`:

```python
# Create two adapters
client_a = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32, model_id="policy"
)
client_b = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=16, model_id="reference"
)

# Train policy, forward-only on reference
fwd_bwd = client_a.forward_backward(data, loss_fn="policy_loss")
ref_logprobs = client_b.forward(data, loss_fn="per_token_ce")
```

LoRA adapters can be saved and loaded via the standard weight endpoints:

```http
POST /api/v1/save_weights
POST /api/v1/load_weights
```
