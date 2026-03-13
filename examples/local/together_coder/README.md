# TogetherCoder SFT Training

Fine-tuning configs using the TogetherCoder SWE dataset from HuggingFace Hub.

## Prerequisites

- Access to `xorl-org/TogetherCoder-Preview-SWE-Rebench` on HuggingFace Hub
- 8 GPUs with sufficient memory (80GB+ recommended for 30B MoE)

## Configs

### Qwen3-4B (8 GPUs)

```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_4b_instruct_2507/qwen3_4b_instruct_2507_local.yaml
```

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Dataset | `xorl-org/TogetherCoder-Preview-SWE-Rebench` (split: `processed-qwencoder-reward1-decontaminated`) |
| Parallelism | Ulysses SP=4, DP replicate=2 |
| Max seq len | 128k tokens (samples exceeding this are filtered) |
| Packing | sequential, pack to 160k |
| Padding | `pad_to_multiple_of: 4096` |
| FSDP2 | full shard, bf16 mixed precision |
| Gradient checkpointing | enabled |

### Qwen3-8B (8 GPUs)

```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_8b/qwen3_8b.yaml
```

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-8B` |
| Dataset | `xorl-org/TogetherCoder-Preview-SWE-Rebench`, `TogetherCoder-Preview-SWE-Smith`, `TogetherCoder-Preview-R2E-Gym` |
| Parallelism | Ulysses SP=8 |
| Max seq len | 128k tokens |
| Packing | sequential, pack to 128k |
| Padding | `pad_to_multiple_of: 4096` |
| FSDP2 | full shard, bf16 mixed precision |
| Gradient checkpointing | enabled |

### Qwen3-Coder-30B-A3B MoE (8 GPUs)

```
torchrun --nproc_per_node=8 --master_port=29501 -m xorl.cli.train configs/qwen3_coder_30b_a3b/qwen3_coder_30b_a3b.yaml
```

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Dataset | `xorl-org/TogetherCoder-Preview-SWE-Rebench` (split: `processed-qwencoder-reward1-decontaminated`) |
| Parallelism | Ulysses SP=8, EP=8 |
| Max seq len | 128k tokens |
| Packing | sequential, pack to 160k |
| Padding | `pad_to_multiple_of: 4096` |
| FSDP2 | full shard, bf16 mixed precision |
| Gradient checkpointing | enabled |

## Notes

- First run downloads the dataset from HuggingFace Hub and caches it locally. Subsequent runs load from cache.
- `pad_to_multiple_of: 4096` is set higher than default (128) to reduce padding overhead with long sequences.
- The `outputs/` directory contains example resolved config snapshots (`xorl_cli.yaml`) showing all effective settings after defaults are applied.
