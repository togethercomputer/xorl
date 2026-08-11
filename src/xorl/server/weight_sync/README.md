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
    "flush_cache": False,            # set True to flush KV cache after sync
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

Only BF16/no quantization and sender-side block-FP8 sync are supported online.
Use `null` for BF16/no quantization. FP8 sync currently supports only
Slime/SGLang-compatible E4M3 weights with FP32 `weight_scale_inv` block scales.
Unsupported methods such as `int4`,
`compressed-tensors`, `awq`, or QAT/fake-quant configs are rejected before
transport starts; those workflows need a separate sender format, receiver
pre/post-processing contract, and logprob validation gate.

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

For a runnable same-prompt SGLang logprob gate around FP8 sync, use:

```bash
python scripts/fp8_sync_logprob_gate.py \
  --xorl-url http://localhost:6000 \
  --sglang-url http://localhost:30000 \
  --master-address localhost \
  --fmt e4m3 \
  --weight-block-size 128 128
```

When the receiver is booted from a prequantized HF FP8 checkpoint and the trainer is booted from the BF16 checkpoint,
launch a separate BF16 SGLang reference and add `--reference-sglang-url`; this makes the gate compare post-sync receiver
logprobs against the same training-state weights rather than against the receiver's initial FP8 artifact.

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
Senders: backend.complete_sync() or backend.destroy()
Rank 0: endpoint_mgr.resume()
All ranks: barrier
```

Key property: only one module's weights are live in GPU memory at a time
(unshard → extract → reshard streams layer by layer).

## Backend Abstraction

### `WeightTransportBackend` (ABC)

```python
class WeightTransportBackend:
    def initialize(self) -> bool: ...      # establish connections (sender ranks only)
    def destroy(self, *, complete_receiver: bool = True) -> None: ...
    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,         # True on the final bucket of a sync
        weight_version: Optional[str] = None,
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

## P2P Mooncake HCA Pinning

For the P2P backend, NCCL HCA settings are not enough. Mooncake creates its own
transfer engines, so trainer ranks and SGLang receiver ranks should be pinned to
usable HCAs explicitly.

P2P needs the Mooncake transfer engine in the trainer environment, and the
receiver must run an SGLang build with `--enable-rdma-weight-updates`. The base
`pyproject.toml` pins `mooncake-transfer-engine` so `uv sync` installs the
Python extension; the launcher image still needs CUDA runtime libraries visible
at runtime. If SGLang's `MooncakeTransferEngine` wrapper is not importable on the
trainer, xorl constructs `mooncake.engine.TransferEngine` directly.

Trainer-side options, in precedence order:

- `P2P_TRAINER_IB_DEVICES_PER_RANK`: semicolon-separated HCA list. If the list
  covers `world_size`, entries are global-rank indexed; otherwise entries are
  local-rank indexed.
- `P2P_TRAINER_GPU_TO_IB_DEVICE_MAP`: physical GPU to HCA map, for example
  `0=mlx5_0,1=mlx5_1`.
  Kubernetes launches should leave device visibility to the NVIDIA device plugin.
  Only set `P2P_TRAINER_VISIBLE_GPU_INDICES` explicitly when an HCA diagnostic
  needs physical GPU indices; it is not a CUDA visibility mechanism.
- `P2P_TRAINER_IB_DEVICE`: single HCA fallback. This is useful for debugging,
  but it pins every trainer rank to one rail.

Receiver-side SGLang uses `--mooncake-ib-device` as a JSON map keyed by local
rank on each receiver node, not global TP rank. Validate the usable HCA set on
each deployment and spread local ranks across those devices.

### P2P profile guidance

Tune dense and MoE bucket sizes separately, size the CPU scratch pool for the
largest concurrent transfer, and validate timeouts against the deployment's
fabric. Multi-endpoint sync normally fans out in one backend operation; use
serial endpoint sync only as a diagnostic fallback. Keep locator reuse enabled
on warm syncs to avoid rebuilding receiver tensor maps.

Leave these unset for the default performance path:

- `XORL_P2P_USE_ASYNC_API`: Mooncake async writes are still experimental; they
  have produced hangs or mixed results in repeated-update tests.
- `XORL_P2P_CPU_POOL_MIN_BYTES=0`: forces tiny transfers through CPU scratch
  and should be treated as a diagnostic setting.
- `XORL_P2P_PERSIST_SMALL_REGISTRATION=1`: persistent registration of small
  CUDA sources is experimental and should be benchmarked before use.
- `XORL_P2P_LOG_BUCKET_DETAILS=1` and `XORL_P2P_TRANSFER_DEBUG=1`: useful for
  failure diagnosis, but intentionally off the hot path because they add
  logging and debug-object allocation.

Expected warm-sync markers with this profile are
`cached_prepare=True`, `tensor_map_endpoints=0/<num-endpoints>`, near-zero
`backend_init_s`, and no SGLang receiver tensor-map payload on the second and
later syncs.

P2P tuning options:

- FP8 P2P sync requires an explicit sync quantization config, for example via
  `POST /api/v1/set_sync_quantization` or a per-call `quantization` field:
  `{"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]}`.
  Client wrappers may expose this as `XORL_WEIGHT_SYNC_QUANTIZATION` or
  `XORL_SYNC_QUANTIZATION`. A launch-only SGLang `--quantization fp8` flag is
  not enough unless endpoint auto-detection is confirmed to populate the sync
  request's `quantization` field.
- With P2P and explicit FP8 sync quantization, the handler quantizes supported
  projection weights on the trainer side, transfers FP8 weights plus
  `weight_scale_inv` tensors, and skips receiver post-processing by default
  because direct P2P writes already target receiver-native FP8 storage. If the
  receiver is FP8 but the sync request has no FP8 quantization config,
  tensor-size validation should fail instead of silently copying bf16 into FP8
  locators.
- The SGLang receiver must expose a matching block-FP8 layout. XORL emits
  block-wise `weight_scale_inv` tensors; a receiver exposing only per-tensor
  `weight_scale` tensors for FusedMoE is not compatible with this sender path.
- `XORL_P2P_FP8_QUANTIZE_DEVICE=gpu`: use the existing GPU block-FP8 kernel for
  trainer-side FP8 formatting before copying the FP8 output to CPU for P2P
  staging. Leave unset for the portable CPU implementation.
- `XORL_P2P_FP8_PINNED_CPU_COPY=1`: use pinned CPU output buffers for P2P FP8
  staging. This is enabled by default; set to `0` only for debugging.
- `XORL_P2P_FP8_CPU_WORKSPACE=1`: use persistent CPU workspaces for direct-EP
  MoE FP8 formatting. This avoids repeated large CPU allocations and keeps the
  staged HF-layout source, FP32 work buffer, abs buffer, FP8 output, and
  `weight_scale_inv` output alive across syncs.
- `XORL_P2P_FP8_CPU_WORKSPACE_PINNED=1`: allocate the workspace input buffer as
  pinned CPU memory when CUDA is available. Enabled by default for the workspace
  path.
- `XORL_P2P_FP8_CPU_WORKSPACE_MIN_CAPACITY`: minimum expert-record capacity for
  a new CPU workspace. Default: 16.
- `XORL_P2P_FP8_CPU_WORKSPACE_STREAMING=1`: stream final workspace chunks
  through the P2P backend while the next chunk is being quantized. Enabled by
  default for the workspace path.
- `XORL_P2P_FP8_CPU_WORKSPACE_STREAM_BYTES`: maximum quantized workspace chunk
  size for streaming. Defaults to the active MoE bucket size.
- `XORL_P2P_FP8_CPU_WORKSPACE_PENDING_SOURCE_BYTES`: maximum staged BF16 source
  bytes per rank before a CPU-workspace MoE batch is quantized, transferred, and
  reused. Defaults to the active MoE bucket size.
- `XORL_WEIGHT_SYNC_BATCH_MOE=1`: batch direct-EP MoE expert transfers across
  layers so each rank ships fewer large P2P buckets.
- The P2P backend stages each unique source tensor slice once per bucket and
  reuses that pinned source address across receiver sessions. This keeps the
  scratch pool sized to source bytes rather than receiver-fanout bytes.
- `XORL_P2P_BACKEND_CACHE=1`: cache P2P receiver locators and backend state
  across sync calls. This is enabled by default.
- `XORL_P2P_PREPARE_WORKERS`: number of concurrent
  `/prepare_weights_update` calls from the trainer to receiver endpoints.
  Defaults to all endpoints, capped at 32. Set to `1` only for debugging
  serialized prepare behavior.
- `XORL_P2P_PREPARE_TIMEOUT_S`: per-endpoint prepare HTTP timeout. Default:
  120 seconds.
- `XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=1`: fallback/debug guard for
  multi-endpoint P2P. It sends each receiver endpoint through its own serialized
  sync group, avoiding cross-endpoint Mooncake session reuse at the cost of
  giving up normal endpoint fanout parallelism.
- `XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES`: explicit MoE bucket cap override.
  Without this override, P2P uses a 2 GiB MoE bucket cap to amortize
  Mooncake fixed costs; non-P2P backends keep the 256 MiB default.
- `XORL_P2P_USE_ASYNC_API=1`: opt into Mooncake's async write API. The default
  synchronous API path is the sustained-test path; async status polling has
  shown repeated-update `status=-1` failures and should remain experimental.
- `XORL_P2P_ASYNC_MIN_BYTES`: minimum coalesced chunk size for Mooncake's async
  write API when `XORL_P2P_USE_ASYNC_API=1`. Default: 128 MiB.
- `XORL_P2P_MOONCAKE_WORKERS`: number of concurrent Mooncake transfer worker
  calls per trainer rank. Default: 2.
- `XORL_P2P_NUM_POOLS`: number of CPU pinned scratch pools used for pipelined
  staging. Default: 2.
- `XORL_P2P_MOONCAKE_TRANSFER_CHUNK`: number of coalesced staged transfers to
  group into each Mooncake call. Default: 1.
- `XORL_P2P_SMALL_TRANSFER_CHUNK`: number of tiny GPU-direct transfers to group
  into each Mooncake call after the per-bucket small-buffer registration.
  Default: 32.
- `XORL_P2P_PERSIST_SMALL_REGISTRATION=1`: persistently register no-copy tiny
  CUDA source regions across buckets/syncs. Disabled by default while being
  benchmarked.
- `XORL_P2P_CPU_SCRATCH_POOL_BYTES`: CPU pinned staging pool size. Keep this
  above the largest unique-source staged P2P bucket; the default is 4 GiB.
- `XORL_P2P_CPU_POOL_MIN_BYTES`: CUDA tensors smaller than this threshold take
  the small GPU-direct path with per-bucket registration; larger CUDA tensors
  and CPU tensors use the pre-registered CPU scratch pool. Default: 64 KiB.
- `XORL_P2P_PENDING_TRANSFER_TIMEOUT_S`: bounded wait used when draining
  outstanding Mooncake worker futures during flush/destroy. Default: 300
  seconds.
- `XORL_P2P_LOG_BUCKET_DETAILS=1`: opt into per-bucket P2P coalescing, source
  reuse, and worker transfer summaries. Disabled by default to keep log I/O off
  the weight-sync hot path.
- `XORL_P2P_TRANSFER_DEBUG=1`: opt into per-locator transfer debug samples in
  failure messages. Disabled by default to avoid allocating debug objects for
  every coalesced transfer on the hot path.
- `MC_IB_PCI_RELAXED_ORDERING=1`: enables relaxed PCIe ordering in Mooncake
  RDMA when the deployment fabric supports it. Leave unset or `0` if the NIC /
  platform combination is not validated.

## Adding a New Backend

1. **Create `backends/my_backend.py`** and subclass `WeightTransportBackend`:

```python
from .base import TransportConfig, WeightTransportBackend

class MyBackend(WeightTransportBackend):
    def initialize(self) -> bool:
        # Connect to inference endpoints using self.config
        # Use self.config.backend_config for backend-specific settings
        return True

    def destroy(self, *, complete_receiver: bool = True) -> None:
        # Tear down connections. If complete_receiver=False, skip receiver-side
        # finalization because the sync failed or was aborted.

    def transfer_bucket(self, bucket, *, src_rank=0, flush_cache=False, weight_version=None):
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
