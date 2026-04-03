---
title: Server Training Config
---

Server config is a **flat YAML** — all fields at the top level with no nesting, passed to:

```bash
python -m xorl.server.launcher --mode auto --config config.yaml
```

Any field can be overridden on the command line with `--server.key value` or `--server.key=value`:

```bash
python -m xorl.server.launcher --mode auto --config config.yaml \
    --server.pipeline_parallel_size 2 \
    --server.expert_parallel_size 4 \
    --server.output_dir /shared/outputs \
    --server.log_level DEBUG
```

---

## Model

| Field | Default | Description |
|---|---|---|
| `model_path` | required | HF Hub ID or local path to model weights. |
| `model_name` | same as `model_path` | Model identifier for validation. |
| `config_path` | same as `model_path` | Path to model config. |
| `tokenizer_path` | same as `config_path` | Path to tokenizer. |
| `attn_implementation` | `flash_attention_3` | Attention backend: `eager`, `sdpa`, `native` (PyTorch SDPA+cuDNN, no deps, Hopper+Blackwell), `flash_attention_3` (FA3, Hopper), `flash_attention_4` (FA4 CUTE, Hopper+Blackwell). |
| `moe_implementation` | `null` | MoE kernel: `null` (auto), `eager`, `triton`, `native`, `quack`. |
| `ep_dispatch` | `alltoall` | Expert-parallel dispatch: `alltoall` or `deepep` (NVLink-optimized). |
| `deepep_buffer_size_gb` | `2.0` | DeepEP NVLink buffer size per GPU in GB. Only active when `ep_dispatch: deepep`. |
| `deepep_num_sms` | `20` | SMs assigned to DeepEP communication kernels. Must be even. |
| `deepep_async_combine` | `false` | Overlap DeepEP combine with the next layer's compute (experimental). |
| `merge_qkv` | `true` | Keep Q/K/V projections fused. Set `false` for tensor parallelism. |
| `basic_modules` | `[]` | Additional module names to shard as separate FSDP units. |
| `foundation` | `{}` | Foundation model extra config (dict). |
| `encoders` | `{}` | Multimodal encoder configs, keyed by type (`image`, `video`, `audio`). |

### Numerical alignment flags

These flags align the training model's numerics with the inference engine (SGLang) to avoid train/inference mismatch.

| Field | Default | Description |
|---|---|---|
| `router_fp32` | `true` | Upcast MoE router gate logits to float32 for numerical stability. |
| `lm_head_fp32` | `true` | Upcast LM head logits to float32. |
| `rmsnorm_mode` | `native` | RMSNorm implementation: `eager`, `native`, or `compile`. |
| `activation_native` | `false` | Use unfused SiLU instead of fused Triton kernel. |
| `rope_native` | `false` | Use unfused RoPE instead of flash_attn kernel. |
| `attention_cast_bf16` | `false` | Explicitly cast Q/K to BF16 after RoPE. |

---

## Parallelism

| Field | Default | Description |
|---|---|---|
| `data_parallel_mode` | `fsdp2` | Data parallelism: `none`, `ddp`, `fsdp2` (ZeRO-3). |
| `data_parallel_shard_size` | `1` | Number of GPUs per FSDP shard group. |
| `data_parallel_replicate_size` | `1` | Number of data replicas for HSDP. |
| `tensor_parallel_size` | `1` | TP degree. |
| `pipeline_parallel_size` | `1` | PP stages. |
| `pipeline_parallel_schedule` | `1F1B` | PP schedule: `1F1B` or `GPipe`. |
| `pp_variable_seq_lengths` | `true` | Dynamically negotiate max seq length per PP step via all-reduce. |
| `expert_parallel_size` | `1` | EP degree for MoE models. |
| `ulysses_parallel_size` | `1` | Ulysses context parallelism degree. |
| `ringattn_parallel_size` | `1` | Ring Attention degree. |
| `cp_fsdp_mode` | `all` | SP+FSDP interaction: `all`, `ulysses_only`, `ring_only`, `none`. |
| `reshard_after_forward` | `true` | Reshard FSDP2 parameters after forward. |

---

## Memory and performance

| Field | Default | Description |
|---|---|---|
| `seed` | `42` | Random seed. |
| `enable_mixed_precision` | `true` | BF16 mixed-precision training. |
| `enable_gradient_checkpointing` | `true` | Activation recomputation to reduce memory. |
| `enable_full_shard` | `true` | FSDP2 full parameter sharding (ZeRO-3). |
| `enable_activation_offload` | `false` | Offload activations to CPU. |
| `enable_compile` | `false` | `torch.compile` for forward pass. |
| `enable_reentrant` | `false` | Use reentrant gradient checkpointing. |
| `enable_forward_prefetch` | `false` | FSDP forward prefetch. |
| `init_device` | `meta` | Model initialization device: `cpu`, `meta`, `cuda`. |
| `load_weights_mode` | `auto` | Weight loading: `auto`, `safetensors`, `dcp`. |
| `ce_mode` | `compiled` | Cross-entropy implementation: `compiled` (recommended, `torch.compile`) or `eager` (may OOM at 32K+ seq len). |

---

## Optimizer

| Field | Default | Description |
|---|---|---|
| `optimizer` | `adamw` | Optimizer: `adamw`, `anyprecision_adamw`, `sgd`, `muon`. |
| `optimizer_dtype` | `bf16` | Dtype for optimizer states: `fp32` or `bf16`. BF16 halves optimizer memory. |
| `muon_lr` | `0.02` | Learning rate for Muon matrix parameter groups. Only used when `optimizer: muon`. |
| `muon_momentum` | `0.95` | Muon momentum coefficient. |
| `muon_nesterov` | `true` | Use Nesterov momentum in Muon. |
| `muon_ns_steps` | `5` | Newton-Schulz iterations for Muon orthogonalization. |
| `muon_adjust_lr_fn` | `null` | Muon LR scaling: `original` or `match_rms_adamw`. |

---

## Checkpointing

| Field | Default | Description |
|---|---|---|
| `output_dir` | `outputs` | Output directory for checkpoints and logs. Must be on shared filesystem for multi-node. |
| `ckpt_manager` | `dcp` | Checkpoint format: `dcp` or `torch`. |
| `load_checkpoint_path` | `""` | Path to checkpoint to resume from. Empty string = start fresh. |
| `storage_limit` | `10TB` | Max disk usage for `output_dir` (e.g., `10GB`, `500MB`). Saves fail with `StorageLimitError` when exceeded. |
| `idle_session_timeout` | `7200.0` | Seconds before an idle training session is automatically cleaned up. Default: 2 hours. |
| `skip_initial_checkpoint` | `false` | Skip saving the initial checkpoint (`000000`) at startup. |

---

## Data

Training data is sent by the client at runtime. These fields control how the server processes it:

| Field | Default | Description |
|---|---|---|
| `sample_packing_sequence_len` | `32000` | Maximum packed sequence length in tokens. |
| `enable_packing` | `true` | Combine multiple samples into a single packed sequence. |

---

## Logging

| Field | Default | Description |
|---|---|---|
| `log_level` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `enable_self_test` | `false` | Run a self-test forward/backward pass after model initialization. |
| `log_gradient_norms` | `true` | Log per-layer-type gradient norms after each backward pass. |
| `log_router_stats` | `true` | Log MoE router token distribution statistics. |

---

## Worker

ZMQ communication between the launcher, workers, and API server.

| Field | Default | Description |
|---|---|---|
| `worker_bind_host` | `0.0.0.0` | Host for rank-0 worker's ZMQ ROUTER socket. Use `0.0.0.0` for multi-node to accept all interfaces. |
| `worker_bind_port` | `5556` | Port for rank-0 worker's ZMQ socket. |
| `engine_connect_host` | `null` | Host for the engine to connect to rank-0. `null` = auto (localhost for single-node, file-based for multi-node). |
| `worker_bind_address` | `auto` | Full ZMQ address (`tcp://host:port`). `auto` = pick a free port. |
| `worker_connection_timeout` | `120.0` | Timeout in seconds for worker-engine connection. Increase for slow multi-node setups. |
| `worker_max_retries` | `3` | Max retries for failed worker operations. |

---

## LoRA

| Field | Default | Description |
|---|---|---|
| `enable_lora` | `false` | Enable LoRA adapters. |
| `lora_rank` | `32` | LoRA rank (`r`). Default is 32 for server (vs 16 for local). |
| `lora_alpha` | `16` | LoRA scaling factor. |
| `lora_target_modules` | `null` | Module names to inject LoRA into. `null` = default for architecture. |
| `moe_shared_lora` | `false` | Share LoRA weights across all MoE experts. |
| `moe_hybrid_shared_lora` | `false` | Share `lora_A` for gate/up projections and `lora_B` for down projections across experts. |
| `enable_qlora` | `false` | Quantize base weights and train LoRA adapters on top. |
| `quant_format` | `nvfp4` | Quantization format: `nvfp4`, `block_fp8`. |
| `quant_group_size` | `16` | Quantization group size. |
| `qlora_exclude_modules` | `null` | Modules to exclude from quantization (e.g., `[lm_head]`). |
| `merge_lora_interval` | `0` | Merge LoRA into base weights every N steps. `0` = never. |
| `reset_optimizer_on_merge` | `false` | ReLoRA optimizer reset after merge. |

---

## MoE

| Field | Default | Description |
|---|---|---|
| `freeze_router` | `true` | Freeze MoE router weights during training. Recommended for fine-tuning to preserve routing learned during pre-training. |

---

## Inference sync

| Field | Default | Description |
|---|---|---|
| `sync_inference_method` | `nccl_broadcast` | Method for pushing updated weights to the inference endpoint after each step. Currently only `nccl_broadcast` is supported (uses SGLang `update_weights_from_distributed`). |
