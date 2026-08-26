---
title: Server Training Config
---

Server config is a **flat YAML** — all fields at the top level with no nesting, passed to:

```bash
python -m xorl.server.launcher --mode auto --config config.yaml
```

This page is a curated reference for commonly used fields and important interactions, not a generated inventory of every `ServerArguments` member. `python -m xorl.server.launcher --help` documents launcher-level options; the exact flat config field set and field help live in `src/xorl/server/server_arguments.py`. To print the current field names, run `python -c "from dataclasses import fields; from xorl.server.server_arguments import ServerArguments; print(chr(10).join(f.name for f in fields(ServerArguments)))"`. A stored `null` may resolve to a model-specific effective value during startup.

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
| `attn_implementation` | `null` (resolved) | Attention backend: `eager`, `sdpa`, `native` (PyTorch SDPA+cuDNN, no deps, Hopper+Blackwell), `flash_attention_3` (FA3, Hopper), or `flash_attention_4` (FA4 CUTE, Hopper+Blackwell). The server resolves an omitted value to FA4. |
| `moe_implementation` | `null` | MoE kernel: `null` (auto), `eager`, `triton`, `native`, `quack`. |
| `ep_dispatch` | `alltoall` | Expert-parallel dispatch: `alltoall` or `deepep` (GPU-resident dispatch using intra-node fabric and, when configured, NVSHMEM/RDMA across nodes). |
| `deepep_buffer_size_gb` | `2.0` | DeepEP NVLink buffer size per GPU in GB. Only active when `ep_dispatch: deepep`. |
| `deepep_num_sms` | `20` | SMs assigned to DeepEP communication kernels. Must be even. |
| `deepep_async_combine` | `false` | Overlap DeepEP combine with the next layer's compute (experimental, unsafe). Forced to `false` in code unless `XORL_DEEPEP_UNSAFE_ASYNC_COMBINE=1` is exported; without that env var, deferring the comm-stream sync races the transformer block's read of the combined tensor on the default stream. |
| `alltoall_combine_hidden_chunk_size` | `0` | Hidden-dimension chunk size for all-to-all EP combine. `0` disables chunking; use a positive value to reduce long-context MoE combine memory peaks. |
| `merge_qkv` | `true` | Keep Q/K/V projections fused. Set `false` for tensor parallelism. |
| `basic_modules` | `[]` | Additional module names to shard as separate FSDP units. |
| `foundation` | `{}` | Foundation model extra config (dict). |
| `encoders` | `{}` | Multimodal encoder configs, keyed by type (`image`, `video`, `audio`). |

### Numerical alignment flags

These stored defaults are resolved after the model architecture is known. Ordinary models use the values noted below; exact dense Qwen3, Qwen3.5-family, GLM-5.2, and DSV4-Flash programs select and validate architecture-owned numerical paths. These settings are prerequisites for parity, not a K3 certificate by themselves.

| Field | Default | Description |
|---|---|---|
| `router_fp32` | `null` (resolves `true`) | Upcast MoE router gate logits to float32. Exact DSV4-Flash requires its native non-upcast router program instead. |
| `lm_head_fp32` | `null` (resolves `true`) | Upcast LM-head logits to float32. Exact DSV4-Flash requires its native distributed head program instead. |
| `rmsnorm_mode` | `null` (resolved) | Ordinary models and exact DSV4-Flash resolve to `native`; exact dense Qwen3, Qwen3.5-family, and GLM-5.2 programs require `sglang_fused`. Other explicit diagnostic modes are also accepted by the argument type. |
| `activation_native` | `false` (resolved) | Use native SiLU instead of the fused Triton kernel. Exact Qwen3.5-family programs resolve this to `true`; the other exact programs retain their architecture-owned fused arithmetic. |
| `rope_native` | `null` (resolved) | Ordinary models and exact DSV4-Flash resolve to `false`; exact dense Qwen3, Qwen3.5-family, and GLM-5.2 programs resolve to `true`. |
| `rope_class_b` | `null` (resolved) | Select the compiled Class-B RoPE FP32-chain path. It is enabled for exact dense Qwen3, Qwen3.5-family, and GLM-5.2 programs; DSV4 owns a separate RoPE program. |
| `attention_cast_bf16` | `false` (resolved) | Explicitly cast Q/K to BF16 after RoPE. Exact Qwen3.5-family programs resolve this to `true`; dense Qwen3, GLM-5.2, and DSV4-Flash exact programs require `false`. |
| `qwen35_rmsnorm_family` | `null` (resolved) | Exact Qwen3.5/3.6 programs require the qualified `v2` arithmetic; other architectures reject an override. |
| `sparse_mla_enabled` | `null` (resolved) | Canonical GLM-5.2 enables the sparse-MLA path; ordinary models resolve to `false`. |
| `sparse_mla_backend` | `auto` (resolved) | Canonical GLM-5.2 requires `flashmla`; other models preserve the selected backend. |

---

## Parallelism

| Field | Default | Description |
|---|---|---|
| `data_parallel_mode` | `fsdp2` | Data parallelism: `none`, `ddp`, `fsdp2` (ZeRO-3). |
| `data_parallel_shard_size` | `1` | Number of GPUs per FSDP shard group. |
| `data_parallel_replicate_size` | `1` | Number of data replicas for HSDP. |
| `tensor_parallel_size` | `1` | TP degree. |
| `pipeline_parallel_size` | `1` | PP stages. |
| `pipeline_parallel_schedule` | `1F1B` | PP schedule: `1F1B`, `GPipe`, `Interleaved1F1B`, `InterleavedZeroBubble`, `ZBVZeroBubble`, or `DualPipeV`. |
| `pipeline_parallel_virtual_stages` | `1` | Model chunks per PP rank. Virtual stages are not supported with EP or inference weight sync. |
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
| `compile_dynamic_shapes` | `false` | Pass `dynamic=True` to `torch.compile`; keep disabled unless a workload has benchmarked a dynamic-shape win. |
| `enable_reentrant` | `false` | Use reentrant gradient checkpointing. |
| `enable_forward_prefetch` | `false` | FSDP forward prefetch. |
| `init_device` | `meta` | Model initialization device: `cpu`, `meta`, `cuda`. |
| `load_weights_mode` | `grouped` | Weight loading mode: `grouped` (default, with rank-0 fallback), `all_ranks`, or `skip`. |
| `ce_mode` | `null` (resolved) | Ordinary models and exact DSV4-Flash resolve to `compiled`; exact dense Qwen3, Qwen3.5-family, and GLM-5.2 programs resolve to `bi_fused`. Explicit modes also include `eager`, `quack_linear`, and `fused_quack`, subject to loss/topology checks. |
| `enable_fp8_training` | `false` | Experimental full-weight block-FP8 compute. Mutually exclusive with LoRA/QLoRA and QARL. |
| `enable_qarl` | `false` | Experimental dynamic fake-quant training with full-precision masters and STE gradients. E4M3 applies to dense `nn.Linear` modules; NVFP4 also supports MoE expert containers. Mutually exclusive with LoRA/QLoRA and full-weight FP8 training. |
| `qarl_quant_cfg` | `null` | QARL alias or dictionary. `null`/`FP8_DEFAULT_CFG` resolves to dynamic E4M3 W8A8 with `[128, 128]` weight blocks. `nvfp4` resolves to dynamic, weight-only W4 with `group_size: 16`; set `activation: true` for W4A4. NVFP4 covers dense linears and MoE expert containers, while E4M3 is dense-only. |

---

## Optimizer

| Field | Default | Description |
|---|---|---|
| `optimizer` | `adamw` | Optimizer: `adamw`, `anyprecision_adamw`, `sgd`, `signsgd`, `muon`. |
| `optimizer_dtype` | `bf16` | Dtype for optimizer states: `fp32` or `bf16`. BF16 halves optimizer memory. |
| `muon_fallback_optimizer` | `adamw` | Optimizer used for parameters excluded from Muon. Use `sgd` for a state-free fallback in memory-constrained no-momentum Muon runs. |
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
| `lora_b_init_std` | `0.0` | Optional deterministic normal initialization standard deviation for LoRA-B. `0.0` keeps the standard zero-B/no-op initialization. |
| `lora_b_init_seed` | `0` | Seed for opt-in nonzero LoRA-B initialization. |
| `lora_target_modules` | `null` | Module names to inject LoRA into. `null` = default for architecture. |
| `moe_hybrid_shared_lora` | `false` | Share `lora_A` for gate/up projections and `lora_B` for down projections across experts. |
| `enable_qlora` | `false` | Quantize base weights and train LoRA adapters on top. |
| `quant_format` | `nvfp4` | QLoRA quantization format: `nvfp4`, `block_fp8`, or `nf4`. |
| `quant_group_size` | `16` | Quantization group size. |
| `qlora_exclude_modules` | `null` | Modules to exclude from quantization (e.g., `[lm_head]`). |
| `merge_lora_interval` | `0` | Merge LoRA into base weights every N steps. `0` = never. |
| `reset_optimizer_on_merge` | `false` | ReLoRA optimizer reset after merge. |
| `adapter_state_load_mode` | `all_ranks` | How to restore multi-adapter checkpoints: `all_ranks` loads on every rank; `rank0_broadcast` loads on rank 0 and broadcasts weights, metadata, and optimizer state. |

---

## MoE

| Field | Default | Description |
|---|---|---|
| `freeze_router` | `true` | Freeze MoE router weights during training. Recommended for fine-tuning to preserve routing learned during pre-training. |

---

## Inference sync

| Field | Default | Description |
|---|---|---|
| `sync_inference_method` | `nccl_broadcast` | Method for pushing updated weights to the inference endpoint after each step. The pinned xorl-sglang revision supports `nccl_broadcast` (two-phase distributed receive) and `p2p` (Mooncake RDMA writes). XoRL also accepts `sparse_delta`, but that mode is not usable with the pinned receiver because `/update_weights_from_sparse_delta` is absent. |
| `receiver_kv_cache_dtype` | `null` | Expected receiver KV-cache dtype: `auto`, `fp8`, or `fp8_e4m3`. Validates registered endpoint metadata; it does not configure SGLang itself. |

---

## Train/serve profile

| Field | Default | Description |
|---|---|---|
| `train_serve_profile` | `null` | Named train/serve mode: `full`, `lora`, or `fp8_lora`. Selects the trainer and receiver combination once; pinned trainer fields are derived when unset and rejected (all conflicts listed) when explicitly contradicted, and registered inference endpoints are validated against the profile at `/add_inference_endpoint`. `null` keeps the historical unprofiled behavior. |

A profile is a single declaration of what the base weights and adapters are on
both sides of the trainer/receiver pair. `fp8_lora` means an **FP8 (block
e4m3) frozen base with bf16 LoRA adapter weights** — base quantization, not
adapter quantization; no profile quantizes the adapters themselves.

Derived trainer fields per profile — *pinned* fields are derived when unset
and rejected when contradicted; *filled* fields are aligned defaults that
explicit values always override:

| Profile | Pinned | Filled |
|---|---|---|
| `full` | `enable_lora=false`, `enable_qlora=false`, `block_fp8_qlora_training=false`, `unfuse_for_lora=false`, `enable_fp8_training=false`, `enable_qarl=false` | — |
| `lora` | `enable_lora=true`, `enable_qlora=false`, `block_fp8_qlora_training=false`, `enable_fp8_training=false`, `enable_qarl=false` | `unfuse_for_lora=true`, `lora_alpha=32` |
| `fp8_lora` | `enable_lora=true`, `enable_qlora=true`, `quant_format=block_fp8`, `unfuse_for_lora=false`, `enable_fp8_training=false`, `enable_qarl=false` | `quant_group_size=128`, `lora_alpha=32` |

Receiver requirements enforced when an SGLang endpoint is registered:

| Profile | Receiver base | Receiver LoRA pool |
|---|---|---|
| `full` | unquantized (bf16) | must be disabled |
| `lora` | unquantized (bf16) | `--enable-lora`, `max_lora_rank` ≥ trainer `max_lora_rank` |
| `fp8_lora` | FP8-quantized (`--quantization fp8` or an FP8 checkpoint) | `--enable-lora`, `max_lora_rank` ≥ trainer `max_lora_rank` |

`python -m xorl.server.train_serve_profile <config.yaml> [--tp-size N]` prints
the matching `sglang.launch_server` command for the profile. Example configs:
`examples/server/configs/profiles/qwen3_8b_{full,lora,fp8_lora}.yaml`. See
[SGLang integration](/server-training/sglang/) for the launch-flag mapping.
