---
title: "Client SDK Overview"
---

[xorl-client](https://github.com/togethercomputer/xorl-client) is the Python SDK for driving the xorl training server. It is lightweight (no PyTorch dependency), async-first, and Tinker API compatible.

## Installation

```bash
pip install xorl-client
```

Or from source:

```bash
pip install git+https://github.com/togethercomputer/xorl-client.git
```

xorl-client is also installed automatically with xorl — see the [installation guide](/getting-started/installation/).

---

## Client Classes

| Class | Purpose |
|---|---|
| [`ServiceClient`](#serviceclient) | Main entry point — connects to the server, creates all other clients |
| [`TrainingClient`](#trainingclient) | Training loop: `forward_backward()`, `optim_step()`, `save_state()`, `load_state()` |
| [`SamplingClient`](#samplingclient) | Rollout generation via xorl-sglang |
| [`RestClient`](#restclient) | Checkpoint management: list, delete, get metadata |

All methods return `APIFuture` objects — call `.result()` to block or `await` in async code.

---

### ServiceClient

The main entry point. Connects to the xorl training server and provides factory methods for all other clients.

```python
import xorl_client

service = xorl_client.ServiceClient(base_url="http://localhost:6000")
```

| Method | Returns | Description |
|---|---|---|
| `create_training_client(base_model)` | `TrainingClient` | Full-weight training (no LoRA) |
| `create_lora_training_client(base_model, rank, ...)` | `TrainingClient` | LoRA training with configurable rank/alpha |
| `create_sampling_client(base_url, model_path)` | `SamplingClient` | Load a saved LoRA adapter on xorl-sglang and return a `SamplingClient` |
| `create_rest_client(model_id)` | `RestClient` | Checkpoint management |
| `create_training_client_from_state(checkpoint_path)` | `TrainingClient` | Resume training from checkpoint (weights only) |
| `create_training_client_from_state_with_optimizer(checkpoint_path)` | `TrainingClient` | Resume training with optimizer state |

Environment variables: `XORL_BASE_URL` (default server URL), `XORL_API_KEY` (authentication).

`create_sampling_client(...)` is the LoRA helper path: it calls `/api/v1/create_sampling_session`
on the training server to load a saved LoRA adapter on the registered inference workers, then returns a `SamplingClient`.
Full-weight exports from `save_weights_for_sampler()` are not dynamically loaded by this helper.

---

### TrainingClient

Drives the training loop. All methods return `APIFuture` — submit multiple calls without blocking, then collect results.

```python
client = service.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32
)

# Non-blocking pipeline: both submitted immediately
fwd = client.forward_backward(data, loss_fn="importance_sampling")
opt = client.optim_step(xorl_client.AdamParams(learning_rate=1e-5))

# Collect results
result = fwd.result()
opt.result()
print(f"logprobs={result.loss_fn_outputs[0]['logprobs']}")
```

| Method | Description |
|---|---|
| `forward_backward(data, loss_fn, loss_fn_params)` | Compute loss + accumulate gradients |
| `forward(data, loss_fn)` | Forward-only (validation, reference logprobs) |
| `optim_step(adam_params)` | Apply gradients with Adam optimizer |
| `sync_inference_weights(master_address, ...)` | Broadcast current weights to registered inference endpoints |
| `save_state(path)` | Save full checkpoint (model + optimizer) |
| `load_state(path)` | Load checkpoint |
| `save_weights_for_sampler(path)` | Save inference weights under `sampler_weights/` (LoRA adapter in LoRA mode, full HF checkpoint otherwise) |
| `get_tokenizer()` | Get the model's tokenizer |

Requests are automatically ordered by `seq_id` — `forward_backward` always executes before `optim_step` regardless of when they arrive.

**Data format:** Training data is a list of `Datum` objects:

```python
datum = xorl_client.Datum(
    model_input=xorl_client.ModelInput.from_ints(input_ids),
    loss_fn_inputs={"labels": labels, "logprobs": old_logprobs, "advantages": advantages},
)
```

For RL losses, `model_input` should contain the full prompt + completion sequence.
`SamplingClient` returns generated output tokens only, so reconstruct the full sequence before calling `forward_backward()`,
mask prompt positions with `advantages=0.0`, and make `logprobs` line up with the server's shifted `full_ids[1:]` view.

---

### SamplingClient

Connects to xorl-sglang for rollout generation. Supports batch sampling with straggler handling.

```python
sampler = xorl_client.SamplingClient(base_url="http://localhost:30000")

# Single sample
response = sampler.sample(
    "What is 2+2?",
    xorl_client.SamplingParams(max_tokens=64),
).result()

# Batch sampling with timeout-based straggler handling
result = sampler.sample_batch(
    prompts,
    xorl_client.SamplingParams(max_tokens=64),
    timeout=30.0,
).result()
# result.completed, result.failed, result.cancelled
```

`response.tokens` and `response.logprobs` contain generated output tokens only, not the prompt tokens.
If you use these results for RL training, keep the prompt token IDs so you can rebuild full prompt + completion sequences.

---

### RestClient

Checkpoint management operations.

```python
rest = service.create_rest_client()

# List checkpoints
checkpoints = rest.list_checkpoints().result()
for cp in checkpoints.checkpoints:
    print(f"{cp.checkpoint_id}: {cp.checkpoint_type}")

# Delete a checkpoint
rest.delete_checkpoint("my_checkpoint").result()

# Get checkpoint metadata
info = rest.get_weights_info("xorl://default/weights/step_42").result()
```

---

## Key Concepts

### APIFuture

All client methods return `APIFuture` objects for non-blocking operation:

```python
# Sync usage
result = client.forward_backward(data, "causallm_loss").result()

# Async usage
result = await client.forward_backward(data, "causallm_loss")

# Pipeline: submit all, then wait
fwd = client.forward_backward(data, "policy_loss")
opt = client.optim_step(adam_params)
fwd_result = fwd.result()  # blocks until forward_backward completes
opt.result()                # blocks until optim_step completes
```

### Automatic Chunking

Large batches of `Datum` objects are automatically split into chunks to stay within HTTP payload limits. This is transparent — you pass the full batch and xorl-client handles the rest.

### Tinker API Compatibility

xorl-client is wire-compatible with the Tinker training API protocol. Field mappings are handled automatically:

| Tinker field | xorl field |
|---|---|
| `session_id` | `model_id` |
| `loss_fn_config` | `loss_fn_params` |
| `model_input.chunks[].tokens` | `model_input.input_ids` |
