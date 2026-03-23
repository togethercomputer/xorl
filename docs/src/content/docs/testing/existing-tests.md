---
title: Existing Tests
---

## Data (`tests/data/`)

Tests the full data pipeline: raw dataset loading, packing, batching, and collation.

**`data/prepare/`**
- `test_shared.py` — loading datasets from Hugging Face Hub, local paths, and URLs; train/validation splits; merging multiple datasets
- `test_packing.py` — FFD (First-Fit Decreasing) packing, sequential allocation, position ID generation, packed dataset merging
- `test_hash.py` — dataset fingerprinting and config hashing for cache invalidation
- `test_file_lock_loader.py` — multi-process safe dataset preparation with file locking; counter management and cleanup
- `test_utils.py` — retry strategies (exponential, linear, constant backoff), MD5/SHA256 hashing

**`data/collators/`**
- `test_tensor_collator.py` — list/numpy/tensor input conversion, dtype inference
- `test_packing_concat_collator.py` — sequence concatenation, flash attention kwargs, position ID generation
- `test_sequence_shard_collator.py` — context-parallel slicing and padding, multi-rank splitting, flash attention setup
- `test_collate_pipeline.py` — chaining multiple collators, key preservation through the pipeline

**`test_data_loader.py`** — `MicroBatchCollator`, `DataLoaderBuilder`, micro-batch splitting, custom collator pipelines

**`test_data_loader_distributed.py`** — data partitioning across DP ranks, context-parallel sharding, epoch consistency across ranks

---

## Distributed (`tests/distributed/`)

Tests for parallel state management and communication primitives. These require `torchrun`.

- `test_parallel_state.py` — `ParallelState` initialization, mesh layouts (DP/TP/PP/SP/EP), rank management
- `test_tensor_parallel.py` — tensor parallelism utility functions
- `test_sequence_parallel.py` — context-parallel communication patterns
- `test_pipeline_parallel.py` — pipeline parallelism across ranks
- `test_ring_attention.py` — ring attention communication
- `test_ep_lora_weight_slicing.py` — expert parallelism combined with LoRA weight slicing
- `test_vocab_parallel_ce.py` — vocabulary-parallel cross-entropy loss

---

## Models (`tests/models/`)

Tests for model-level logic, focused on Mixture-of-Experts (MoE) and LoRA.

- `test_moe_routing_cache.py` — correctness of the routing cache across forward passes
- `test_moe_routing_replay.py` — replaying cached routing decisions deterministically
- `test_moe_experts_lora.py` — injecting LoRA adapters into expert layers; routing behavior after injection
- `test_moe_weight_loading_integration.py` — loading expert weights from checkpoints
- `test_moe_weight_auto_merge.py` — automatic merging of expert weights
- `test_qwen3_moe_fused_lora.py` — Qwen3-specific fused LoRA operations on MoE layers

---

## Ops (`tests/ops/`)

Unit tests for low-level CUDA/Triton operations. Most require a GPU.

- `test_attention.py` — Flash Attention 3/4 and eager attention backends
- `test_block_fp8.py` — FP8 block-wise quantization
- `test_block_fp8_gkn.py` — FP8 GKN format conversion
- `test_moe_ops.py` — core MoE dispatch/combine operations
- `test_moe_act.py` — activation functions in MoE layers
- `test_moe_gkn_format.py` — GKN tensor layout for expert routing
- `test_group_gemm.py` — grouped matrix multiplication
- `test_nf4.py` — NF4 quantization for weights
- `test_moe_torch_compile.py` — `torch.compile` compatibility with MoE ops
- `test_eager_vs_native_moe.py` — numerical equivalence between eager and optimized MoE paths
- `test_prequant_gnk_to_gkn.py` — pre-quantization format conversion utilities

---

## QLoRA (`tests/qlora/`)

Tests for quantized LoRA fine-tuning. Require CUDA.

- `test_qlora.py` — NF4/FP8 quantization, forward/backward correctness, memory efficiency, dequantization
- `test_detect_prequantized.py` — detecting whether a model's weights are already quantized
- `test_moe_load_experts.py` — loading pre-quantized expert weights
- `test_quantize_error_reduction.py` — quantization error mitigation strategies

---

## Server (`tests/server/`)

Tests for the distributed serving infrastructure (CPU-only for most, some require ZMQ/network).

**`api_server/`**
- `test_api_server.py` — HTTP endpoint handlers and request parsing
- `test_api_types.py` — request/response schema validation with Pydantic
- `test_checkpoint_paths.py` — checkpoint path resolution logic
- `test_future_store.py` — async future storage and retrieval

**`orchestrator/`**
- `test_orchestrator.py` — main scheduler and request batching
- `test_packing.py` — request packing strategy across sequence lengths
- `test_scheduler.py` — request scheduling across ranks
- `test_request_processor.py` — request processing pipeline
- `test_cu_seqlens_alignment.py` — `cu_seqlens` tensor correctness
- `test_api_orchestrator_messages.py` — message protocol between API server and orchestrator
- `test_orchestrator_client_communication.py` — client request handling over ZMQ
- `test_orchestrator_runner_messages.py` — messages between orchestrator and training runners

**`runner/`**
- `test_send_ready.py` — ready signal handling during startup

**`weight_sync/`**
- `test_pp_nccl_transfer.py` — pipeline-parallel weight synchronization via NCCL broadcast

---

## End-to-end (`tests/e2e/`)

Full training pipeline tests that spin up real models with `torchrun`. These are the slowest tests and require GPUs.

**`qwen3_8b/`**
- `test_lora.py` — LoRA fine-tuning on a tiny Qwen3 dense model
- `test_pp.py` — pipeline parallelism across ranks
- `test_tflops_threshold.py` — throughput regression check (asserts minimum TFLOPs)

**`qwen3_30b/`**
- `test_pp.py` — pipeline parallelism for larger MoE model
- `test_server_moe.py` — end-to-end MoE serving with the training server

E2E tests use small randomly-initialized model variants (no downloads required) created by fixtures in `tests/e2e/conftest.py`:

| Fixture | Description |
|---|---|
| `tiny_dense_model_dir` | Random-init Qwen3 dense, no saved weights |
| `tiny_dense_model_dir_with_weights` | Random-init dense with weights on disk |
| `tiny_moe_model_dir` | Random-init Qwen3-MoE |
| `tiny_moe_model_dir_with_weights` | Random-init MoE with weights on disk |
| `small_dense_model_dir_with_weights` | Dense, hidden size 256 |
| `small_moe_model_dir_with_weights` | MoE, intermediate size 64 (NF4 group_size compatible) |
