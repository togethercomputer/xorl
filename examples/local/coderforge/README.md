# CoderForge SFT Training

Fine-tuning configs using the CoderForge dataset from HuggingFace Hub.

## Prerequisites

- Access to `togethercomputer/CoderForge-Preview` on HuggingFace Hub
- 8 GPUs with sufficient memory (80GB+ recommended for 30B MoE)

## Dataset

All configs use:

| Field | Value |
|-------|-------|
| Path | `togethercomputer/CoderForge-Preview` |
| Subset | `trajectories-tokenized_qwencoder` |
| Split | `filtered_reward1` |

## Configs

### Qwen3-8B (8 GPUs)

Two configs for comparing loss curves across optimizers:

**AdamW:**
```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_8b/qwen3_8b_adamw.yaml
```

**Muon (bf16 optimizer states):**
```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_8b/qwen3_8b_muon_bf16.yaml
```

| Setting | AdamW | Muon bf16 |
|---------|-------|-----------|
| Model | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| Optimizer | AdamW | Muon (2D weights) + AdamW (rest) |
| Optimizer dtype | bf16 | bf16 |
| LR | 1e-5 | muon_lr=0.02, adamw_lr=1e-4 |
| Parallelism | Ulysses SP=8 | Ulysses SP=8 |
| Max seq len | 128k tokens | 128k tokens |
| Packing | sequential, pack to 128k | sequential, pack to 128k |
| FSDP2 | full shard, bf16 mixed precision | full shard, bf16 mixed precision |

### Qwen3-Coder-30B-A3B MoE (8 GPUs)

```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_coder_30b_a3b/qwen3_coder_30b_a3b.yaml
```

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Parallelism | Ulysses SP=8, EP=8 |
| Max seq len | 128k tokens |
| Packing | sequential, pack to 160k |
| Padding | `pad_to_multiple_of: 4096` |
| FSDP2 | full shard, bf16 mixed precision |
| Gradient checkpointing | enabled |

### Qwen3-32B (8 GPUs)

```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_32b.yaml
```

## Notes

- First run downloads the dataset from HuggingFace Hub and caches it locally. Subsequent runs load from cache.
- `pad_to_multiple_of: 4096` is set higher than default (128) to reduce padding overhead with long sequences.
