---
title: "API Reference"
---

:::note
All training operations (`forward_backward`, `optim_step`, `abort_gradient_epoch`, `save_weights`, `load_weights`)
use a **two-phase async pattern**: the POST returns a `request_id` immediately.
Poll `POST /api/v1/retrieve_future` with that ID to get the actual result.
The `xorl-client` SDK handles polling automatically.
:::

All endpoints are served at `http://<host>:<port>/`. Training operations use a two-phase async protocol — see [Launching & Configuration](/xorl/server-training/training-server/launching/#api-server) for details.

## Training Operations

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/forward_backward` | Forward + backward pass. Returns `UntypedAPIFuture`. |
| `POST` | `/api/v1/forward` | Forward pass only (no gradient). For eval or reference logprobs. |
| `POST` | `/api/v1/optim_step` | Apply gradients, clip, step optimizer and LR scheduler. |
| `POST` | `/api/v1/abort_gradient_epoch` | Discard one unmutated multi-adapter LoRA gradient epoch. Returns `UntypedAPIFuture`. |
| `POST` | `/api/v1/retrieve_future` | Poll for async result by `request_id`. |

### Abort an adapter-gradient epoch

Call `POST /api/v1/abort_gradient_epoch` when a `forward_backward` response is
ambiguous (for example, the client lost the response) or when a client manually
cancels an epoch before `optim_step`. The request follows the normal asynchronous
protocol:

```json
{
  "model_id": "policy",
  "seq_id": 42
}
```

The abort is whole-epoch, not per request: it discards every accumulated capture
since the last successful optimizer step. It is idempotent while the epoch is
still unmutated, so a client may safely repeat an ambiguous abort request before
replaying the complete epoch.

An abort refuses a poisoned session or a session whose optimizer mutation is
awaiting distributed publication. Those states cannot be repaired in-process.
In particular, an ambiguous `optim_step` response may mean parameters or
optimizer state already changed: restart from the last checkpoint. Never use
`abort_gradient_epoch` to retry an ambiguous optimizer step.

## Model / Session Management

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/create_model` | Create and register a new training session. LoRA mode supports multi-tenant sessions; full-weight mode only supports the reserved `model_id="default"` session. |
| `POST` | `/api/v1/unload_model` | Unload a session, freeing associated adapter state. |
| `POST` | `/api/v1/kill_session` | Kill an active session. In LoRA mode, non-default tenant sessions are removed; in full-weight mode, the single active session is reset. |
| `GET` | `/api/v1/session_info` | List active sessions and their state. |
| `POST` | `/api/v1/create_session` | Create and register a Tinker-compatible session ID for follow-up calls. |
| `POST` | `/api/v1/session_heartbeat` | Refresh a session's last-activity timestamp for idle cleanup. |

## Checkpointing

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/save_weights` | Save DCP checkpoint. `path: null` = auto-timestamped. |
| `POST` | `/api/v1/load_weights` | Load DCP checkpoint and restore model weights + optimizer state. |
| `POST` | `/api/v1/list_checkpoints` | List available checkpoints under `output_dir`. |
| `POST` | `/api/v1/delete_checkpoint` | Delete a checkpoint by ID. |
| `POST` | `/api/v1/weights_info` | Return checkpoint metadata for a model (used by xorl-client to load weights). |
| `POST` | `/api/v1/save_weights_for_sampler` | Save inference weights under `sampler_weights/` (LoRA adapter or full HF checkpoint, depending on training mode). |
| `GET` | `/api/v1/training_runs` | List training runs. |

## Inference Integration

| Method | Path | Description |
|---|---|---|
| `POST` | `/add_inference_endpoint` | Register an SGLang inference server for weight sync. |
| `POST` | `/remove_inference_endpoint` | Unregister an inference endpoint. |
| `GET` | `/list_inference_endpoints` | List all registered endpoints. |
| `POST` | `/api/v1/sync_inference_weights` | Broadcast current weights to all inference endpoints via NCCL. |
| `POST` | `/api/v1/set_sync_quantization` | Configure FP8 quantization for weight sync. |
| `POST` | `/api/v1/create_sampling_session` | Load a LoRA adapter on inference server for sampling. |

## Health & Control

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check. Returns `{ "status": "healthy", "engine_running": bool }`. |
| `GET` | `/api/v1/healthz` | Tinker health check alias. |
| `GET` | `/` | Root info. |
| `POST` | `/sleep` | Offload model weights to CPU to free GPU memory. |
| `POST` | `/wake_up` | Reload weights back to GPU after sleep. |

## Compatibility notes

- ZORL and its REST routes were removed. Known ZORL configuration fields now
  fail with a migration error at YAML, `--server.*`, and session API boundaries;
  unrelated unknown fields retain their rolling-client/shared-config behavior.
- Multi-adapter training accepts raw numerator losses only. Setting
  `normalize_loss_before_backward` rejects the `forward_backward` call; leave
  normalization to `optim_step`, which divides by the accumulated valid-token
  denominator exactly once.
- A zero or non-finite global valid-token denominator is a hard, non-retryable
  data error. Repair the empty/invalid batch before beginning a new epoch.
- Authoritative adapter clipping supports the L2 norm only. `gradient_clip`
  remains the scalar L2 threshold; other norm types are rejected.

## Source

| File | Description |
|---|---|
| [`src/xorl/server/api_server/endpoints.py`](https://github.com/togethercomputer/xorl/blob/main/src/xorl/server/api_server/endpoints.py) | All FastAPI endpoint handlers |
| [`src/xorl/server/api_server/api_types.py`](https://github.com/togethercomputer/xorl/blob/main/src/xorl/server/api_server/api_types.py) | Pydantic request/response models |
