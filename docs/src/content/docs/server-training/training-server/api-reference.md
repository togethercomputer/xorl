---
title: "API Reference"
---

:::note
All training operations (`forward_backward`, `optim_step`, `save_weights`, `load_weights`)
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
| `POST` | `/api/v1/retrieve_future` | Poll for async result by `request_id`. |

## Model / Session Management

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/create_model` | Create and register a new model session (LoRA or full-weight). |
| `POST` | `/api/v1/unload_model` | Unload a session, freeing associated adapter state. |
| `POST` | `/api/v1/kill_session` | Kill an active session; optionally reload weights from checkpoint. |
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

## Source

| File | Description |
|---|---|
| [`src/xorl/server/api_server/endpoints.py`](https://github.com/togethercomputer/xorl/blob/main/src/xorl/server/api_server/endpoints.py) | All FastAPI endpoint handlers |
| [`src/xorl/server/api_server/api_types.py`](https://github.com/togethercomputer/xorl/blob/main/src/xorl/server/api_server/api_types.py) | Pydantic request/response models |
