# Dummy Dataset Benchmarks & Training Configs

Configs using the built-in `path: dummy` dataset. No real data needed -- random tokenized samples are generated in-memory at startup.

## Naming Convention

Pattern: `{model}_{parallelism}_{optimizer}.yaml`

Parallelism abbreviations: `pp` (pipeline), `dp` (data parallel shard), `tp` (tensor), `ep` (expert), `cp` (context / ring attention), `sp` (Ulysses sequence parallel).

## Benchmark Configs (8 GPUs)

### Optimizer Comparison

| Config | Model | Optimizer | Parallelism |
|--------|-------|-----------|-------------|
| `qwen3_4b_instruct_adamw.yaml` | Qwen3-4B-Instruct | AdamW | FSDP dp=8 |
| `qwen3_4b_instruct_muon.yaml` | Qwen3-4B-Instruct | Muon | FSDP dp=8 |
| `qwen3_8b_adamw.yaml` | Qwen3-8B | AdamW | FSDP dp=8 |
| `qwen3_8b_muon.yaml` | Qwen3-8B | Muon | FSDP dp=8 |
| `qwen3_30b_a3b_adamw.yaml` | Qwen3-30B-A3B (MoE) | AdamW | EP=8, SP=8 |
| `qwen3_30b_a3b_muon.yaml` | Qwen3-30B-A3B (MoE) | Muon | EP=8, SP=8 |

### Tensor Parallelism Scaling (Qwen3-8B)

| Config | TP | DP | Compile |
|--------|----|----|---------|
| `qwen3_8b_tp1.yaml` | 1 | 8 | - |
| `qwen3_8b_tp2.yaml` | 2 | 4 | - |
| `qwen3_8b_tp4.yaml` | 4 | 2 | - |
| `qwen3_8b_tp4_compile.yaml` | 4 | 2 | yes |
| `qwen3_8b_tp8.yaml` | 8 | 1 | - |

### Pipeline Parallelism (Qwen3-8B)

| Config | PP | DP | SP (Ulysses) |
|--------|----|----|--------------|
| `qwen3_8b_dp4.yaml` | - | 4 | - |
| `qwen3_8b_pp2_dp2.yaml` | 2 | 2 | - |
| `qwen3_8b_pp2_sp2.yaml` | 2 | 1 | 2 |

### Pipeline Parallelism (Qwen3-30B-A3B MoE)

| Config | PP | EP | CP | Optimizer |
|--------|----|----|-----|-----------|
| `qwen3_30b_a3b_pp2_ep4_muon.yaml` | 2 | 4 | - | Muon |
| `qwen3_30b_a3b_pp2_ep2_cp2_muon.yaml` | 2 | 2 | 2 | Muon |
| `qwen3_30b_a3b_pp2_ep4_cp4_muon.yaml` | 2 | 4 | 4 | Muon |

### Context Parallel / Sequence Parallel (Qwen3-8B)

| Config | CP | SP (Ulysses) | DP |
|--------|----|--------------|----|
| `qwen3_8b_cp1_sp8.yaml` | 1 | 8 | 1 |
| `qwen3_8b_cp2_sp4.yaml` | 2 | 4 | 1 |
| `qwen3_8b_cp4_sp2.yaml` | 4 | 2 | 1 |
| `qwen3_8b_cp8_sp1.yaml` | 8 | 1 | 1 |
| `qwen3_8b_cp1_sp4_dp2.yaml` | 1 | 4 | 2 |
| `qwen3_8b_cp4_sp1_dp2.yaml` | 4 | 1 | 2 |

### Context Parallel / Sequence Parallel (Qwen3-30B-A3B MoE)

| Config | CP | SP (Ulysses) | EP |
|--------|----|--------------|----|
| `qwen3_30b_a3b_cp1_sp8.yaml` | 1 | 8 | 8 |
| `qwen3_30b_a3b_cp2_sp4.yaml` | 2 | 4 | 8 |
| `qwen3_30b_a3b_cp4_sp2.yaml` | 4 | 2 | 8 |
| `qwen3_30b_a3b_cp8_sp1.yaml` | 8 | 1 | 8 |

### Expert Parallel + Tensor Parallel (Qwen3-30B-A3B MoE)

| Config | EP | TP | DP |
|--------|----|----|-----|
| `qwen3_30b_a3b_ep8.yaml` | 8 | 1 | 8 |
| `qwen3_30b_a3b_ep4_tp2.yaml` | 4 | 2 | 4 |

## Training Configs

| Config | GPUs | Model | Parallelism | Notes |
|--------|------|-------|-------------|-------|
| `qwen3_8b.yaml` | 8 | Qwen3-8B | dp=8 | Dense, AdamW |
| `qwen3_8b_pp2.yaml` | 8 | Qwen3-8B | PP=2, dp=4 | Dense, AdamW |
| `qwen3_8b_pp2_sp2.yaml` | 4 | Qwen3-8B | PP=2, SP=2 | Dense, AdamW |
| `qwen3_32b.yaml` | 8 | Qwen3-32B | SP=4, dp=2 | Dense, AdamW |
| `qwen3_4b_instruct.yaml` | 8 | Qwen3-4B-Instruct | TP=8 | Dense, AdamW |
| `qwen3_30b_a3b_pp2_dp4_ep4.yaml` | 8 | Qwen3-30B-A3B | PP=2, dp=4, EP=4 | MoE, AdamW |
| `qwen3_30b_a3b_pp2_sp4.yaml` | 8 | Qwen3-30B-A3B | PP=2, SP=4 | MoE, AdamW |
| `qwen3_coder_30b_a3b.yaml` | 8 | Qwen3-Coder-30B-A3B | EP=8, SP=8 | MoE, Muon, DeepEP |

### LoRA / QLoRA (Qwen3-8B)

| Config | Method | Quant | Parallelism | GPUs |
|--------|--------|-------|-------------|------|
| `qwen3_8b_lora.yaml` | LoRA | - | FSDP dp=4 | 4 |
| `qwen3_8b_lora_cp4.yaml` | LoRA | - | CP=4 | 4 |
| `qwen3_8b_lora_cp2_sp2.yaml` | LoRA | - | CP=2, SP=2 | 4 |
| `qwen3_8b_qlora_nvfp4.yaml` | QLoRA | nvfp4 | FSDP dp=4 | 4 |
| `qwen3_8b_qlora_nvfp4_requant.yaml` | QLoRA | nvfp4 | FSDP dp=4 | 4 |
| `qwen3_8b_qlora_nvfp4_sp4.yaml` | QLoRA | nvfp4 | SP=4 | 4 |
| `qwen3_8b_qlora_nvfp4_cp2_sp2.yaml` | QLoRA | nvfp4 | CP=2, SP=2 | 4 |
| `qwen3_8b_qlora_nvfp4_pp2.yaml` | QLoRA | nvfp4 | PP=2, dp=4 | 8 |

### QLoRA (Qwen3-32B)

| Config | Quant | Parallelism | GPUs |
|--------|-------|-------------|------|
| `qwen3_32b_qlora_nvfp4.yaml` | nvfp4 | FSDP dp=4 | 4 |

### LoRA / QLoRA (Qwen3-30B-A3B MoE)

| Config | Method | Source | Parallelism | GPUs |
|--------|--------|--------|-------------|------|
| `qwen3_30b_a3b_instruct_lora_bf16.yaml` | LoRA | bf16 | EP=8, SP=8 | 8 |
| `qwen3_30b_a3b_base_qlora_nvfp4.yaml` | QLoRA | on-the-fly (base) | EP=8, SP=8 | 8 |
| `qwen3_30b_a3b_qlora_nvfp4.yaml` | QLoRA | pre-quantized (nvidia) | EP=8, SP=8 | 8 |
| `qwen3_30b_a3b_instruct_qlora_nvfp4.yaml` | QLoRA | on-the-fly (instruct) | EP=8, SP=8 | 8 |

### QLoRA (Qwen3-235B-A22B MoE)

| Config | Source | Parallelism | GPUs |
|--------|--------|-------------|------|
| `qwen3_235b_a22b_base_qlora_nvfp4.yaml` | on-the-fly (base) | EP=4, dp=8 | 8 |
| `qwen3_235b_a22b_qlora_nvfp4.yaml` | pre-quantized (nvidia) | EP=4, dp=8 | 8 |
| `qwen3_235b_a22b_instruct_qlora_nvfp4.yaml` | on-the-fly (instruct) | EP=4, dp=4 | 4 |

### QLoRA (Qwen3-Coder-30B-A3B MoE)

| Config | Source | Parallelism | GPUs |
|--------|--------|-------------|------|
| `qwen3_coder_30b_a3b_instruct_qlora_nvfp4.yaml` | on-the-fly | EP=4, SP=4 | 8 |

## Usage

```bash
# Example (8 GPUs)
torchrun --nproc_per_node=8 -m xorl.cli.train examples/local/dummy/configs/qwen3_8b_tp2.yaml

# Training (8 GPUs)
torchrun --nproc_per_node=8 -m xorl.cli.train examples/local/dummy/configs/qwen3_8b.yaml

# Pipeline parallel (8 GPUs)
torchrun --nproc_per_node=8 -m xorl.cli.train examples/local/dummy/configs/qwen3_8b_pp2.yaml

# QLoRA NVFP4 (4 GPUs)
torchrun --nproc_per_node=4 -m xorl.cli.train examples/local/dummy/configs/qwen3_8b_qlora_nvfp4.yaml

# QLoRA NVFP4 + Ulysses SP (4 GPUs)
torchrun --nproc_per_node=4 -m xorl.cli.train examples/local/dummy/configs/qwen3_8b_qlora_nvfp4_sp4.yaml

# QLoRA NVFP4 + pipeline parallel (8 GPUs)
torchrun --nproc_per_node=8 -m xorl.cli.train examples/local/dummy/configs/qwen3_8b_qlora_nvfp4_pp2.yaml
```

## How Dummy Datasets Work

Set `path: dummy` in the dataset config:

```yaml
data:
  datasets:
    - path: dummy
      type: tokenized
      max_seq_len: 8000  # length of each generated sample
```

- Generates 4096 samples of `max_seq_len` random token IDs (0-32000)
- `labels = input_ids` (every token contributes to loss)
- All ranks use the same seed -- identical data, no disk I/O
