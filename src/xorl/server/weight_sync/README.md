# Weight Sync

Transfers trained weights from the FSDP2 training cluster to SGLang inference
endpoints. Supports full-weight bf16, LoRA merge, QLoRA dequant→merge, and
optional bf16→fp8 block-wise re-quantization before transfer.

## API

### `POST /api/v1/set_sync_quantization`

Set the default quantization format applied during weight sync. Persists until
changed and is used by all subsequent `sync_inference_weights` calls that don't
specify an explicit quantization config.

```python
# bf16 → fp8 block-wise (block_fp8, most common for FP8 inference models)
requests.post("http://localhost:6000/api/v1/set_sync_quantization", json={
    "quantization": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    }
})

# Skip specific layers (e.g. lm_head stays bf16)
requests.post("http://localhost:6000/api/v1/set_sync_quantization", json={
    "quantization": {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["lm_head"],
    }
})

# Disable quantization (sync in bf16)
requests.post("http://localhost:6000/api/v1/set_sync_quantization", json={
    "quantization": None
})
```

### `POST /api/v1/sync_inference_weights`

Trigger a weight sync to all registered inference endpoints. Blocks until
complete (or timeout). All training ranks participate — the handler broadcasts
the command internally via Gloo so all ranks enter the sync together.

```python
resp = requests.post("http://localhost:6000/api/v1/sync_inference_weights", json={
    "master_address": "localhost",   # training server address for NCCL rendezvous
    "master_port": 0,                # default; asks TCPStore to bind an ephemeral port
    "buffer_size_mb": 1024,          # bucket size; reduce if OOM during sync
    "flush_cache": True,             # flush KV cache after sync (default)
    "pause_mode": "retract",         # "retract" | "abort" | "in_place"
    # "quantization": {...}          # override per-call; otherwise uses set_sync_quantization
})
result = resp.json()
# {
#   "success": true,
#   "transfer_time": 6.1,
#   "num_parameters": 652,
#   "num_buckets": 28,
#   "endpoints_synced": [{"host": "localhost", "port": 30000, "success": true}]
# }
```

**`pause_mode`** controls how in-flight inference requests are handled:
- `"retract"` (default): drain and re-queue requests; they re-execute after sync
- `"abort"`: drop in-flight requests immediately
- `"in_place"`: keep KV cache in place (only valid when `flush_cache=False`)

**`quantization`** can be specified per-call to override the default set by
`set_sync_quantization`. If omitted, the server uses the default (or
auto-detects from the endpoint's quantization config).

### Typical usage pattern

```python
# 1. Register inference endpoint (once at startup)
requests.post("http://localhost:6000/add_inference_endpoint",
              json={"host": "localhost", "port": 30000})

# 2. Set quantization format (once, or whenever it changes)
requests.post("http://localhost:6000/api/v1/set_sync_quantization",
              json={"quantization": {"quant_method": "fp8", "weight_block_size": [128, 128]}})

# 3. Train for N steps
for step in range(num_steps):
    requests.post(".../api/v1/forward_backward", ...)
    requests.post(".../api/v1/optim_step", ...)

# 4. Sync weights to inference
requests.post("http://localhost:6000/api/v1/sync_inference_weights",
              json={"master_address": "localhost"})
```

---

## Module Structure

```
weight_sync/
├── handler.py           # Orchestration: unshard, extract, quantize, dispatch
├── endpoint_manager.py  # HTTP pause/resume/health for inference endpoints
├── nccl_broadcast.py    # Low-level NCCL primitives (WeightSynchronizer)
├── sync_primitives.py   # QLoRA collective ops, FP8 quantization helpers
└── backends/
    ├── base.py          # WeightTransportBackend ABC + TransportConfig dataclass
    ├── nccl.py          # NCCLBroadcastBackend (current default)
    └── __init__.py      # create_backend() factory
```

## Design: Two Orthogonal Concerns

The handler separates **pipeline orchestration** from **transport**:

**Pipeline orchestration** (`handler.py`) is backend-agnostic:
- FSDP unshard/reshard per module
- QLoRA dequant collective ops (requires all stage ranks)
- Parameter extraction, LoRA merge, weight unfuse
- FP8 quantization
- EP gather, PP inter-stage transfer

**Transport** (`backends/`) is pluggable:
- Moving prepared tensor buckets from training rank(s) to inference endpoint(s)
- Each backend implements `initialize()`, `transfer_bucket()`, `destroy()`

## Sync Flow

```
Rank 0: health check → backend.initialize() → endpoint_mgr.pause()
        │
        ▼
For each PP stage (sequential):
  For each FSDP module in stage:
    ALL stage ranks:  unshard() → QLoRA collective ops → reshard()
    Stage leader:     extract params → LoRA merge → unfuse
    Stage 0 / rank 0: quantize (optional) → backend.transfer_bucket()
    PP stages 1+:    send bf16 buffer to rank 0 via pp_group → rank 0 transfers
        │
        ▼
Rank 0: endpoint_mgr.resume()
Senders: backend.destroy()
All ranks: barrier
```

Key property: only one module's weights are live in GPU memory at a time
(unshard → extract → reshard streams layer by layer).

## Backend Abstraction

### `WeightTransportBackend` (ABC)

```python
class WeightTransportBackend:
    def initialize(self) -> bool: ...      # establish connections (sender ranks only)
    def destroy(self) -> None: ...         # tear down connections
    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,         # True on the final bucket of a sync
    ) -> None: ...

    # Topology hints (read by handler)
    @property
    def sender_ranks(self) -> FrozenSet[int]: ...         # default: {0}
    @property
    def supports_direct_ep_transfer(self) -> bool: ...    # default: False
    @property
    def supports_direct_pp_transfer(self) -> bool: ...    # default: False
```

### `TransportConfig`

Populated by the handler from the sync request payload:

```python
@dataclass
class TransportConfig:
    endpoints: List[EndpointConfig]   # host, port, world_size (TP size)
    master_address: str
    master_port: int                  # 0 selects an ephemeral port on the training rank
    group_name: str
    buffer_size_mb: int
    device: str
    training_world_size: int
    training_rank: int
    backend_config: Dict[str, Any]    # backend-specific settings
```

### Topology hints

The handler uses `sender_ranks` to decide which training ranks extract and
prepare data.  `supports_direct_ep_transfer` and `supports_direct_pp_transfer`
let a backend skip the gather-to-rank-0 step and instead have each EP/PP rank
send its slice directly to inference.

## Current Backend: `nccl_broadcast`

```
Training rank 0  ──NCCL broadcast──►  SGLang TP workers (ranks 1..N)
```

- `initialize()`: fires HTTP `/init_weights_update_group` to each SGLang endpoint
  (background threads), then creates a dedicated NCCL process group connecting
  rank 0 to all inference ranks via TCPStore rendezvous.
- `transfer_bucket()`: for each bucket, POSTs metadata via HTTP
  `/update_weights_from_distributed` and broadcasts tensors via `dist.broadcast`.
- `sender_ranks = {0}` — only rank 0 sends; other training ranks only participate
  in training-side FSDP collectives.

## Adding a New Backend

1. **Create `backends/my_backend.py`** and subclass `WeightTransportBackend`:

```python
from .base import TransportConfig, WeightTransportBackend

class MyBackend(WeightTransportBackend):
    def initialize(self) -> bool:
        # Connect to inference endpoints using self.config
        # Use self.config.backend_config for backend-specific settings
        return True

    def destroy(self) -> None:
        # Tear down connections

    def transfer_bucket(self, bucket, *, src_rank=0, flush_cache=False):
        # Send [(name, tensor), ...] to inference
        # flush_cache=True signals the final bucket of a sync — use it to
        # trigger "load all weights now" for storage-based backends
        for name, tensor in bucket:
            self._send(name, tensor)
        if flush_cache:
            self._finalize()
```

2. **Register in `backends/__init__.py`**:

```python
def create_backend(method: str, config: TransportConfig, **kwargs):
    if method == "my_backend":
        from .my_backend import MyBackend
        return MyBackend(config, **kwargs)
    ...
```

3. **Override topology hints** if your backend supports multi-rank sending:

```python
@property
def sender_ranks(self) -> FrozenSet[int]:
    # e.g., all EP ranks send their local experts directly
    return frozenset(range(self.config.training_world_size))

@property
def supports_direct_ep_transfer(self) -> bool:
    return True   # handler will skip EP gather-to-rank-0
```

4. **Pass backend-specific config** via `backend_config` in the sync request:

```python
# In the HTTP payload to /api/v1/sync_inference_weights:
{"backend": "my_backend", "backend_config": {"storage_path": "/mnt/shared"}}
```

### Storage backend sketch

For a shared-filesystem or object-store backend, `transfer_bucket` writes tensors
to files and `flush_cache=True` triggers a single HTTP call telling inference to
load them all at once.  No NCCL process group is needed — `initialize()` just
validates that the storage path is accessible.
