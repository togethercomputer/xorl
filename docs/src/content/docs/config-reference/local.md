---
title: Local Training Config
---

Local training uses a nested YAML with `model`, `data`, `train`, and `lora` sections, passed to:

```bash
torchrun --nproc_per_node=8 -m xorl.cli.train config.yaml
```

Any field can be overridden on the command line with `--section.field value`:

```bash
torchrun --nproc_per_node=8 -m xorl.cli.train config.yaml \
    --train.lr 2e-5 \
    --train.output_dir outputs/my_run \
    --train.pipeline_parallel_size 4 \
    --model.attn_implementation flash_attention_4 \
    --data.sample_packing_sequence_len 16384 \
    --lora.enable_lora true \
    --lora.lora_rank 32
```

---

## `model` section

| Field | Default | Description |
|---|---|---|
| `model_path` | `null` | HF Hub ID or local path to pre-trained weights. If `null`, model is randomly initialized. |
| `config_path` | same as `model_path` | Path to model config. Useful when config and weights are in separate locations. |
| `tokenizer_path` | same as `config_path` | Path to tokenizer. |
| `attn_implementation` | `flash_attention_3` | Attention backend: `eager`, `sdpa`, `native` (PyTorch SDPA+cuDNN, no deps, Hopper+Blackwell), `flash_attention_3` (FA3, Hopper), `flash_attention_4` (FA4 CUTE, Hopper+Blackwell). |
| `moe_implementation` | `null` | MoE kernel: `null` (auto), `eager`, `triton` (Triton group GEMM), `native` (torch._grouped_mm), `quack`. |
| `ep_dispatch` | `alltoall` | Expert-parallel dispatch: `alltoall` or `deepep` (NVLink-optimized). |
| `deepep_buffer_size_gb` | `2.0` | DeepEP NVLink buffer size per GPU in GB. Only active when `ep_dispatch: deepep`. |
| `deepep_num_sms` | `20` | SMs assigned to DeepEP communication kernels. Must be even. Lower values leave more SMs for overlapped compute. |
| `deepep_async_combine` | `false` | Overlap DeepEP combine with the next layer's compute (experimental). |
| `merge_qkv` | `true` | Keep Q/K/V projections fused as `qkv_proj`. Set `false` for tensor parallelism or per-projection LoRA. |
| `basic_modules` | `[]` | Additional module names (beyond `_no_split_modules`) to shard as separate FSDP units. |
| `foundation` | `{}` | Extra foundation model config (dict). |
| `encoders` | `{}` | Multimodal encoder configs, keyed by type (`image`, `video`, `audio`). Each value must have `model_path` and optionally `config_path`. |

---

## `data` section

| Field | Default | Description |
|---|---|---|
| `datasets` | required | List of dataset configs (see [Dataset entry fields](#dataset-entry-fields)). |
| `test_datasets` | `[]` | Optional list of evaluation dataset configs. Same format as `datasets`. |
| `dataset_prepared_path` | `last_prepared_dataset` | Path where prepared/cached datasets are stored. |
| `select_columns` | `null` (all columns) | Columns to keep from each dataset (e.g., `[input_ids, labels]`). |
| `sample_packing_method` | `sequential` | Packing strategy: `sequential` (fast, good packing) or `multipack` (FFD-based, maximizes bin utilization). |
| `sample_packing_sequence_len` | `32000` | Target packed bin length in tokens. |
| `sample_packing_group_size` | `100000` | Number of samples packed together in one group. Larger values improve packing slightly. |
| `sample_packing_sequentially` | `null` | Force sequential packing regardless of method. |
| `sample_packing_mp_start_method` | `null` | Multiprocessing start method for packing: `fork`, `spawn`, or `forkserver`. |
| `eval_sample_packing` | `null` | Set to `false` to disable packing during evaluation if errors occur. |
| `dataloader_num_workers` | `8` | DataLoader worker processes. |
| `dataloader_prefetch_factor` | `2` | Batches to prefetch per worker. Set to `null` when `num_workers=0`. |
| `dataloader_pin_memory` | `true` | Pin CPU memory for faster GPU transfer. |
| `dataloader_drop_last` | `true` | Drop the last incomplete batch. |
| `pad_to_multiple_of` | `128` | Pad packed sequences to a multiple of this value for GPU efficiency. |
| `val_set_size` | `null` | Validation split size. Integer = number of samples, float = fraction (e.g., `0.05`). |
| `shuffle_merged_datasets` | `true` | Shuffle the merged dataset before training. |
| `shuffle_before_merging_datasets` | `true` | Shuffle each dataset individually before merging. |
| `dataset_num_proc` | CPU count | Processes for dataset preprocessing. Defaults to `XORL_DATASET_NUM_PROC` env var or CPU count. |
| `dataset_shard_num` | `null` | Number of shards to split the dataset into (for parallel preprocessing). |
| `dataset_shard_idx` | `null` | Which shard to use (used with `dataset_shard_num`). |
| `num_dataset_shards_to_save` | `null` | Number of shards to save the prepared dataset to. Default: single file. |
| `skip_prepare_dataset` | `false` | Skip preparation and load directly from `dataset_prepared_path`. |
| `push_dataset_to_hub` | `null` | Push prepared dataset to HF Hub (`org/repo-name`). Requires `hf_use_auth_token: true`. |
| `hf_use_auth_token` | `null` | Use HF auth token for private datasets or Hub pushes. |

### Dataset entry fields

Each entry in `datasets` (or `test_datasets`) is a dict:

| Field | Default | Description |
|---|---|---|
| `path` | required | HF Hub ID (`org/name`), `s3://`, `gs://`, `abfs://`, `https://`, or local path. Use `dummy` for synthetic data. |
| `type` | `tokenized` | Dataset type. Only `tokenized` is currently supported. |
| `name` | `null` | HF dataset config name (subset). |
| `split` | `null` | HF dataset split (e.g., `train`, `validation`). |
| `revision` | `null` | HF Hub commit hash or tag. |
| `trust_remote_code` | `false` | Allow remote code execution for custom HF datasets. |
| `data_files` | `null` | Specific files to load (string or list). Requires `ds_type` when set. |
| `ds_type` | `null` | File format when using `data_files`: `json`, `csv`, `parquet`, `arrow`, `text`. |
| `max_seq_len` | `null` | Truncate and filter samples longer than this. |
| `shards` | `null` | Split dataset into N pieces (use with `shards_idx`). |
| `shards_idx` | `null` | Index of the shard to use (0-based). |
| `preprocess_shards` | `null` | Process dataset in N sequential chunks for memory efficiency. Mutually exclusive with `shards`. |

---

## `train` section

### Parallelism

| Field | Default | Description |
|---|---|---|
| `data_parallel_mode` | `fsdp2` | Data parallelism: `none`, `ddp`, `fsdp2` (ZeRO-3). FSDP2 requires `init_device: meta`. |
| `data_parallel_shard_size` | `-1` (world_size) | Number of GPUs per FSDP shard group. `-1` = full world. |
| `data_parallel_replicate_size` | `-1` (1) | Number of data replicas for HSDP (Hybrid Sharded DP). `-1` = auto. `dp_size = replicate × shard`. |
| `tensor_parallel_size` | `1` | TP degree. Shards weight matrices column/row-wise across GPUs. Requires `merge_qkv: false`. |
| `pipeline_parallel_size` | `1` | PP stages. Splits model layers across GPUs. |
| `pipeline_parallel_schedule` | `1F1B` | PP schedule: `1F1B` (interleaved, lower memory) or `GPipe` (simpler). |
| `pp_variable_seq_lengths` | `true` | Dynamically negotiate max seq length per PP step via all-reduce, avoiding padding to static max. |
| `expert_parallel_size` | `1` | EP degree for MoE models. Distributes experts across GPUs. |
| `ulysses_parallel_size` | `1` | Ulysses context parallelism degree. |
| `ringattn_parallel_size` | `1` | Ring Attention degree. |
| `cp_fsdp_mode` | `all` | How context parallelism interacts with FSDP: `all` (both Ulysses+Ring), `ulysses_only`, `ring_only`, `none`. |
| `reshard_after_forward` | `null` | FSDP2 reshard after forward. `true` = save memory, `false` = save communication (used for PP by default). `null` = auto. |
| `ep_outside` | `false` | Place EP outside the EP-FSDP mesh. |

:::caution[Field interactions]
- `data_parallel_mode: fsdp2` **requires** `init_device: meta`
- `tensor_parallel_size > 1` **requires** `merge_qkv: false` (in model section)
- `pipeline_parallel_size > 1` **requires** `gradient_accumulation_steps >= pipeline_parallel_size`
- `expert_parallel_size > 1` is incompatible with `init_device: cpu`
- TP + LoRA is **not supported** — use FSDP2 alone for LoRA fine-tuning
:::

### Optimizer

| Field | Default | Description |
|---|---|---|
| `optimizer` | `adamw` | Optimizer: `adamw`, `anyprecision_adamw`, `sgd`, `signsgd`, `muon`. |
| `optimizer_dtype` | `bf16` | Dtype for optimizer states in `anyprecision_adamw` and `muon`: `fp32` or `bf16`. BF16 halves optimizer memory. |
| `lr` | `5e-5` | Peak learning rate. |
| `lr_min` | `1e-7` | Minimum learning rate at the end of decay. |
| `lr_start` | `0.0` | Initial learning rate at the start of warmup. |
| `lr_warmup_ratio` | `0.0` | Fraction of total steps used for linear LR warmup. |
| `lr_decay_style` | `constant` | LR schedule after warmup: `constant`, `linear`, `cosine`. |
| `lr_decay_ratio` | `1.0` | Fraction of total steps to apply LR decay over. |
| `weight_decay` | `0.0` | L2 regularization (AdamW weight decay). |
| `no_decay_modules` | `[]` | Module name substrings to exclude from weight decay (e.g., `[norm]`). |
| `no_decay_params` | `[]` | Parameter name substrings to exclude from weight decay (e.g., `[bias]`). |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |
| `muon_lr` | `0.02` | Learning rate for Muon matrix parameter groups. Only used when `optimizer: muon`. |
| `muon_momentum` | `0.95` | Muon momentum coefficient. |
| `muon_nesterov` | `true` | Use Nesterov momentum in Muon. |
| `muon_ns_steps` | `5` | Newton-Schulz iterations for Muon orthogonalization. |
| `muon_adjust_lr_fn` | `null` | Muon LR scaling: `original` (scale by sqrt(max(1,A/B))), `match_rms_adamw` (lets Muon reuse AdamW LR/WD). |

### Batch sizing

| Field | Default | Description |
|---|---|---|
| `micro_batch_size` | `1` | Per-GPU batch size per step. |
| `gradient_accumulation_steps` | `1` | Steps before optimizer update. Effective per-device batch = `micro_batch_size × gradient_accumulation_steps`. |
| `num_train_epochs` | `1` | Number of passes over the dataset. |
| `max_steps` | `null` | Maximum total training steps. Overrides epoch-based stopping and caps LR scheduler length. |

### Memory and performance

| Field | Default | Description |
|---|---|---|
| `enable_mixed_precision` | `true` | BF16 mixed-precision training. |
| `enable_gradient_checkpointing` | `true` | Activation recomputation to reduce memory. |
| `enable_reentrant` | `false` | Use reentrant gradient checkpointing. Default (non-reentrant) is generally preferred. |
| `recompute_modules` | `null` | Selective checkpointing by submodule: `[self_attn]`, `[mlp]`, or `[self_attn, mlp]`. `null` = whole-layer recompute. |
| `moe_checkpoint_method` | `null` | MoE-specific checkpoint: `null` (full recompute including EP communication), `moe_act` (recompute only gate/up activations, skip EP communication recompute — faster). |
| `enable_full_shard` | `true` | FSDP2 full parameter sharding (ZeRO-3). Set `false` for ZeRO-2. |
| `enable_forward_prefetch` | `true` | Prefetch next FSDP unit's parameters during forward pass. |
| `enable_activation_offload` | `false` | Offload activations to CPU during forward pass. |
| `activation_gpu_limit` | `0.0` | GB of activations to keep on GPU when offloading. `0.0` = offload all. |
| `enable_compile` | `false` | `torch.compile` for model forward pass. |
| `init_device` | `cuda` | Device for weight initialization: `cpu` (rank 0 only), `cuda`, `meta` (required for FSDP2), `npu`. |
| `load_weights_mode` | `broadcast` | `broadcast`: rank 0 reads weights, broadcasts to other ranks (reduces disk I/O). `all_ranks`: every rank reads from disk. |
| `enable_full_determinism` | `false` | Full determinism mode. Requires `allow_cuda_launch_blocking: true`. Degrades performance. |
| `allow_cuda_launch_blocking` | `false` | Allow `CUDA_LAUNCH_BLOCKING=1`. Off by default to prevent accidental performance degradation. |
| `empty_cache_steps` | `500` | Call `torch.cuda.empty_cache()` every N steps. |
| `gc_steps` | `500` | Call `gc.collect()` every N steps. Python GC is disabled between calls. |

### Checkpointing

| Field | Default | Description |
|---|---|---|
| `output_dir` | required | Base directory for checkpoints, logs, and model assets. Must be on a shared filesystem for multi-node training. |
| `ckpt_manager` | `dcp` | Checkpoint format: `dcp` (PyTorch Distributed Checkpoint Protocol). |
| `save_steps` | `0` | Save a checkpoint every N global steps. `0` = disabled. |
| `save_epochs` | `1` | Save every N epochs (fractional OK: `0.25` saves 4× per epoch). |
| `save_async` | `false` | Write checkpoints asynchronously (non-blocking training). |
| `save_hf_weights` | `true` | Also save HF-format weights (`.safetensors`) to the last checkpoint directory. |
| `load_checkpoint_path` | `null` | Path to checkpoint to resume from. Set to `auto` to auto-detect the latest checkpoint in `output_dir`. |

### Logging

| Field | Default | Description |
|---|---|---|
| `log_format` | `progress_bar` | `progress_bar` (tqdm), `structured` (key=value lines for parsing). |
| `use_wandb` | `true` | Enable Weights & Biases logging. |
| `wandb_project` | `Xorl` | W&B project name. |
| `wandb_name` | `null` | W&B run name. |
| `wandb_tags` | `null` | W&B run tags (list of strings). |
| `wandb_log_interval` | `1` | Log metrics to W&B every N steps. |

### Profiling

| Field | Default | Description |
|---|---|---|
| `enable_profiling` | `false` | Enable PyTorch profiler. |
| `profile_start_step` | `1` | Step to start profiling at. |
| `profile_end_step` | `2` | Step to stop profiling at. |
| `profile_trace_dir` | `./trace` | Directory to write profiler trace files. |
| `profile_record_shapes` | `true` | Record input tensor shapes in the trace. |
| `profile_profile_memory` | `true` | Record memory usage in the trace. |
| `profile_with_stack` | `true` | Record Python stack traces. |
| `profile_rank0_only` | `true` | Only profile rank 0. Set `false` to profile all ranks (produces many large files). |

### Seeding

| Field | Default | Description |
|---|---|---|
| `seed` | `42` | Global random seed for reproducibility. |

---

## `lora` section

| Field | Default | Description |
|---|---|---|
| `enable_lora` | `false` | Enable LoRA fine-tuning. |
| `lora_rank` | `16` | LoRA rank (`r`). |
| `lora_alpha` | `16` | LoRA scaling factor (`alpha`). Effective scale = `alpha / rank`. |
| `lora_target_modules` | `null` | Module names to inject LoRA into. `null` = default linear projections for the architecture. |
| `save_lora_only` | `false` | Only save LoRA adapter weights in HF checkpoints (not the full model). |
| `enable_qlora` | `false` | Quantize base weights and train LoRA on top. Implies `enable_lora: true`. |
| `quant_format` | `nvfp4` | Quantization format: `nvfp4` (4-bit, Hopper+), `block_fp8` (8-bit blocks), `nf4` (4-bit normal float). |
| `quant_group_size` | `16` | Quantization group size. Default is 16. Recommended values: 16 for `nvfp4`, 128 for `block_fp8`, 64 for `nf4`. |
| `exclude_modules` | `null` | Module names to exclude from QLoRA quantization (kept as BF16). `null` = auto-detect from checkpoint config. |
| `merge_lora_interval` | `0` | Merge LoRA delta into base weights every N steps. For QLoRA this also re-quantizes. `0` = disabled. |
| `reset_optimizer_on_merge` | `false` | ReLoRA-style optimizer state reset after each merge. Requires `merge_lora_interval > 0`. |
| `enable_aqn` | `false` | Adaptive Quantization Noise: adds calibrated noise to quantized weights during forward to reduce quantization bias. |
| `aqn_alpha` | `1.0` | Noise magnitude scale for AQN. |
