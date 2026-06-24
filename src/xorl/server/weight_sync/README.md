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
  `0=mlx5_2,1=mlx5_3,2=mlx5_1,3=mlx5_5,4=mlx5_9,5=mlx5_9,6=mlx5_6,7=mlx5_5`.
  Kubernetes launches should leave device visibility to the NVIDIA device plugin.
  Only set `P2P_TRAINER_VISIBLE_GPU_INDICES` explicitly when an HCA diagnostic
  needs physical GPU indices; it is not a CUDA visibility mechanism.
- `P2P_TRAINER_IB_DEVICE`: single HCA fallback. This is useful for debugging,
  but it pins every trainer rank to one rail.

Receiver-side SGLang uses `--mooncake-ib-device` as a JSON map keyed by local
rank on each receiver node, not global TP rank. On the current H100 validation
nodes, we avoid `mlx5_4`, `mlx5_7`, and `mlx5_8` and spread TP ranks over the
remaining working HCAs.

### Recommended P2P profile for scaled Qwen3-style MoE

For the 4 trainer pod → 16 SGLang TP2 encoded-reasoning shape, use the
following profile as the starting point. It keeps dense/root chunking separate
from MoE batching, uses the cached receiver prepare path on warm syncs, and
avoids the measured-regressed debug/experimental knobs.

```bash
# Required for Kubernetes Mooncake reachability.
export P2P_TRAINER_HOSTNAME="${POD_IP}"
export XORL_WEIGHT_SYNC_MASTER_ADDRESS="${POD_IP}"

# Normal multi-endpoint P2P syncs should fan out to all receiver endpoints in a
# single backend operation. Enable serial endpoint sync only as a fallback while
# diagnosing endpoint/session instability.
export XORL_SERIAL_INFERENCE_ENDPOINT_SYNC=0

# Keep dense/root tensors small enough for scratch pools while batching MoE.
export XORL_WEIGHT_SYNC_DENSE_BUCKET_BYTES=134217728      # 128 MiB
export XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES=1073741824       # 1 GiB
export XORL_WEIGHT_SYNC_BUCKET_BYTES=1073741824           # legacy MoE alias
export XORL_WEIGHT_SYNC_BATCH_MOE=1

# Source-reuse path keeps the required pool size near source bytes, not
# receiver-fanout bytes. 2 GiB was the best measured pool size for the scaled
# Qwen3-30B-A3B TP2 receiver layout.
export XORL_P2P_CPU_SCRATCH_POOL_BYTES=2147483648         # 2 GiB
export XORL_P2P_MOONCAKE_TRANSFER_CHUNK=8
export XORL_P2P_CPU_POOL_MIN_BYTES=65536                  # small CUDA direct path
export XORL_P2P_PENDING_TRANSFER_TIMEOUT_S=120

# This is now the default copy mode, but keep the explicit variable in older
# generated manifests that still set XORL_P2P_SCATTER_COPY_MODE=list.
export XORL_P2P_SCATTER_REUSE_LOCATORS=1
```

Leave these unset for the default performance path:

- `XORL_P2P_USE_ASYNC_API`: Mooncake async writes are still experimental; they
  have produced hangs or mixed results in repeated-update tests.
- `XORL_P2P_CPU_POOL_MIN_BYTES=0`: forces tiny transfers through CPU scratch;
  this was safe in smoke tests but slower than the default GPU-direct threshold.
- `XORL_P2P_PERSIST_SMALL_REGISTRATION=1`: persistent registration of small
  CUDA sources was safe in smoke tests but regressed warm sync on the scaled
  TP2 layout.
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
  because direct P2P writes already target receiver-native FP8 storage. Set
  `XORL_WEIGHT_SYNC_RUN_POST_PROCESS_WEIGHTS=1` or
  `XORL_P2P_RUN_POST_PROCESS_WEIGHTS=1` only for legacy receivers that still
  require finalization after P2P writes. If the receiver is FP8 but the sync
  request has no FP8 quantization config, tensor-size validation should fail
  instead of silently copying bf16 into FP8 locators.
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
- `XORL_P2P_SCATTER_COPY_MODE`: controls how rank 0 builds per-sender tensor
  map payloads for direct-EP scatter. Default `none` reuses read-only locator
  lists/dicts while constructing scatter payloads. Set `list` to shallow-copy
  lists or `deep` to copy every locator dict for debugging.
- `XORL_P2P_SCATTER_REUSE_LOCATORS`: legacy boolean alias for the default
  scatter copy mode. Set `1` to force locator reuse even when older manifests
  still set `XORL_P2P_SCATTER_COPY_MODE=list`; set `0` to force shallow list
  copies when `XORL_P2P_SCATTER_COPY_MODE` is unset.
- `XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES`: explicit MoE bucket cap override.
  Without this override, P2P uses a 2 GiB MoE bucket cap to amortize
  Mooncake fixed costs; non-P2P backends keep the 256 MiB default.
- `XORL_WEIGHT_SYNC_BUCKET_BYTES`: legacy alias for the MoE bucket cap. Prefer
  `XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES` so dense/root chunking stays independent
  from MoE batching.
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

## Sparse Delta

`scripts/weight_sync_delta_probe.py` can measure whether an update is sparse
enough for a future sparse-delta receiver protocol to be worthwhile. It uses the
optional `delta-encoding` package when available, but it does not change the
current production P2P path.

The experimental `sync_inference_method="sparse_delta"` backend reuses xorl's
normal streaming extraction/unfuse/quantization pipeline, but writes each
prepared bucket as a `delta-encoding` packed sparse file on a shared filesystem
and POSTs it to SGLang's `/update_weights_from_sparse_delta` endpoint. It is
opt-in and does not affect the dense `nccl_broadcast` or `p2p` backends.

Required runtime configuration:

- `--sync-inference-method sparse_delta` on the xorl server.
- Install xorl with the optional sparse-delta extra, or install the dependency
  separately before enabling this backend. The extra currently pins the
  `delta-encoding` revision with the GQA `SectionSplit` and source-encoded
  scheduler fixes:
  `pip install 'xorl[sparse-delta]'`. While developing against a local
  checkout, set `XORL_DELTA_ENCODING_PATH` as below instead.
- `XORL_DELTA_ENCODING_PATH=/path/to/delta-encoding` if the package is not
  installed in the xorl environment.
- `XORL_SPARSE_DELTA_OUTPUT_DIR=/shared/path/visible/to/sglang` so the receiver
  pod can mmap the packed files.
- SGLang built with the sparse-delta receiver endpoint enabled.

Useful optional knobs:

- `XORL_SPARSE_DELTA_KEEP_FILES=1`: keep generated `.packed` files for
  inspection.
- `XORL_SPARSE_DELTA_BASELINE_SCOPE=<token>`: isolate or reset the process-local
  CPU baseline used to encode subsequent syncs sparsely.
- `XORL_SPARSE_DELTA_RESET_BASELINE=1`: clear the current process-local baseline
  when the backend initializes, causing the next sync to encode all values.
- `XORL_SPARSE_DELTA_PRIME_BASELINE=1`: seed the process-local baseline from
  the trainer tensors and only send a final no-op sparse update. Use this when
  the receiver is already loaded from the same base checkpoint and the goal is
  to prime the backend before later sparse update syncs. This still walks the
  dense trainer extraction/unfuse/quantization path, so it is intended for
  small-model debugging rather than large MoE E2E profiling.
- `XORL_SPARSE_DELTA_HTTP_TIMEOUT_S=600`: HTTP timeout for receiver updates.
- `XORL_DELTA_ENCODING_USE_NATIVE_EXTENSION=1`: allow the optional
  `delta-encoding` escape extension. By default the backend uses the pure
  PyTorch path to avoid first-use JIT compilation in the trainer process.

The first sparse-delta sync for a baseline scope encodes every element as a
sparse value overwrite. Later syncs compare exact CPU bytes against the cached
baseline and emit only changed entries. If the receiver has been restarted or
loaded from a different checkpoint, reset or change the baseline scope before
using this backend.

For production sparse updates, prefer the prepacked fast path: generate
inference-coordinate `.packed` files with `delta-encoding`, then pass
`sparse_delta_paths` to `/api/v1/sync_inference_weights`. That path still uses
xorl endpoint pause/resume and weight-version handling, but it skips the
trainer-side FSDP extraction/unfuse loop entirely. A single path is replicated
to every receiver TP rank; otherwise provide one path per TP rank. Add
`"sparse_delta_config": {"prepacked_only": true}` in E2E/profile jobs to make
xorl fail fast if a request accidentally omits `sparse_delta_paths`.

```python
import torch

from xorl.server.weight_sync.sparse_delta_files import (
    SparseTensorUpdate,
    split_sparse_update_by_contiguous_shards,
    write_sparse_delta_files_by_rank,
)

logical_update = SparseTensorUpdate(
    name="lm_head.weight",
    flat_indices=torch.tensor([0, 31040 * 2048], dtype=torch.int64),
    values=torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    shape=(248320, 2048),
)
rank_updates = split_sparse_update_by_contiguous_shards(
    logical_update,
    shard_dim=0,
    num_shards=8,
)
rank_stats = write_sparse_delta_files_by_rank(
    {rank: [update] for rank, update in rank_updates.items()},
    "/shared/deltas/iter42",
    delta_encoding_path="/path/to/delta-encoding",
)

requests.post("http://localhost:6000/api/v1/sync_inference_weights", json={
    "flush_cache": True,
    "weight_version": "iter-42",
    "sparse_delta_paths": [rank_stats[rank].path for rank in sorted(rank_stats)],
    "sparse_delta_config": {"prepacked_only": True},
})
```

When using `delta-encoding`'s translation pipeline, group the terminal
`EncodedDelta` outputs by `future.key.rank`, or pass the futures directly to
`write_translation_futures_as_sparse_delta_files(...)`, then pass the resulting
rank-ordered paths. That keeps sharded tensors in receiver-local coordinates
instead of emitting one global logical flat-index stream.

```python
from xorl.server.weight_sync.sparse_delta_files import (
    prepare_delta_encoding_runtime,
    write_translation_futures_as_sparse_delta_files,
)

prepare_delta_encoding_runtime(delta_encoding_path="/path/to/delta-encoding")

from delta_encoding.encoding.types import SparseCOO
from delta_encoding.ops.types import StoreKey
from delta_encoding.pipeline import TranslationEngine, build_plan
from delta_encoding.spec import EngineConfig, Shard, ShardPlan, resolve

target = resolve(
    EngineConfig(
        num_ranks=8,
        stages=[("shard", ShardPlan(plan={"lm_head.weight": Shard(dim=0)}))],
    ),
    {"lm_head.weight"},
)
plan = build_plan(target=target, sparse=True, encode=True, devices=["cpu"])

logical_sparse = SparseCOO(
    indices=torch.tensor([[0, 31040], [0, 0]], dtype=torch.int32),
    values=torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    shape=(248320, 2048),
    sorted=True,
)
with TranslationEngine(plan, num_workers=1) as engine:
    futures = engine.put(StoreKey("lm_head.weight"), logical_sparse)
    rank_stats = write_translation_futures_as_sparse_delta_files(
        futures,
        "/shared/deltas/iter43",
        expected_ranks=8,
    )
```

Trainer-side source capture is available as an opt-in `optim_step` request
field. This captures selected training-rank local parameters on CPU before the
optimizer step, diffs them after the step, runs
`delta_encoding.encoding.compression.encode`, and writes one source-rank packed
file plus a gathered `manifest.json`. These files are not receiver-ready until
they are translated through a model-specific `delta-encoding` plan.

```python
future = requests.post("http://localhost:6000/api/v1/optim_step", json={
    "adam_params": {"learning_rate": 1e-8},
    "sparse_delta_capture": {
        "enabled": True,
        "output_dir": "/shared/deltas/source-step-42",
        "include": ["lm_head.weight", "self_attn\\.(q|k|v)_proj\\.weight"],
        "dtype": "bfloat16",
        "delta_encoding_path": "/path/to/delta-encoding",
    },
}).json()
optim = requests.post("http://localhost:6000/api/v1/retrieve_future", json={
    "request_id": future["request_id"],
}).json()

source_manifest = optim["info"]["sparse_delta_capture"]["manifest_path"]
```

For Qwen3.6 source-capture experiments, the source manifest can be translated
with `experiments/local_benchmark/scripts/sparse_delta_qwen36_source_encoded_e2e.py`
using `--source-capture-manifest`; the generated terminal paths should then be
posted through `sync_inference_weights` with `prepacked_only`.

Manual receiver smoke test:

```bash
python scripts/weight_sync_sparse_delta_client.py \
  --base-url http://192.168.229.98:30123 \
  --delta-path /shared/p2p-sync-stress/sparse-delta-validation/qwen06b-qnorm-three.packed \
  --tp-size 1 \
  --weight-version sparse-qnorm-smoke \
  --warmup 1 \
  --repeat 5
```

Example:

```bash
python scripts/weight_sync_delta_probe.py \
  --delta-encoding-path /path/to/delta-encoding \
  --shape 4096x4096 \
  --dtype uint8 \
  --density 0.001 \
  --density 0.01 \
  --density 0.1
```

For dense FP8 updates, the packed sparse format is larger than the dense payload
because it stores values plus index deltas. It becomes attractive only when the
changed-entry fraction is small enough, or if a future protocol transfers LoRA
adapter tensors/factors instead of merged dense weights.

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
