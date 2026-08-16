# Dummy Dataset Training Configs

These configs use the built-in `path: dummy` dataset, which generates random tokenized samples in memory. They are useful for configuration bring-up, memory checks, and throughput experiments without downloading a dataset.

## Directory layout

| Directory | Contents |
|---|---|
| `configs/full/` | Full-weight dense and MoE examples, including FSDP, TP, PP, CP/Ulysses, EP, Muon, and interleaved PP |
| `configs/lora/` | LoRA examples for Qwen3 and Llama |
| `configs/qlora/` | NVFP4, block-FP8, and pre-quantized QLoRA examples |

List the exact current inventory instead of inferring filenames from older examples:

```bash
find examples/local/dummy/configs -type f -name '*.yaml' | sort
```

Representative checked-in configurations include:

| Purpose | Config |
|---|---|
| Qwen3-8B full-weight baseline | `configs/full/qwen3_8b.yaml` |
| Qwen3-8B tensor parallel | `configs/full/qwen3_8b_tp4_compile.yaml` |
| Qwen3-8B pipeline parallel | `configs/full/qwen3_8b_pp2.yaml` |
| Qwen3-8B interleaved pipeline | `configs/full/qwen3_8b_pp2_interleaved.yaml` |
| Qwen3-8B context/Ulysses layouts | `configs/full/qwen3_8b_cp1_sp8.yaml`, `qwen3_8b_cp2_sp4.yaml`, `qwen3_8b_cp8_sp1.yaml` |
| Qwen3-30B-A3B expert parallel | `configs/full/qwen3_30b_a3b_ep8.yaml` |
| Qwen3-30B-A3B PP + EP + CP | `configs/full/qwen3_30b_a3b_pp2_ep4_cp4_muon.yaml` |
| GLM-4.5-Air MoE | `configs/full/glm4_moe_ep8.yaml` |
| GPT-OSS MoE | `configs/full/gpt_oss_20b_ep8.yaml`, `gpt_oss_120b_ep8.yaml` |
| LoRA | `configs/lora/qwen3_8b_lora.yaml` |
| QLoRA NVFP4 | `configs/qlora/qwen3_8b_qlora_nvfp4.yaml` |
| QLoRA block-FP8 | `configs/qlora/qwen3_8b_qlora_block_fp8.yaml` |

## Usage

Run commands from the repository root and include the mode subdirectory:

```bash
# Full-weight Qwen3-8B on 8 GPUs
torchrun --nproc_per_node=8 -m xorl.cli.train \
  examples/local/dummy/configs/full/qwen3_8b.yaml

# Interleaved PP on 8 GPUs
torchrun --nproc_per_node=8 -m xorl.cli.train \
  examples/local/dummy/configs/full/qwen3_8b_pp2_interleaved.yaml

# LoRA on 4 GPUs
torchrun --nproc_per_node=4 -m xorl.cli.train \
  examples/local/dummy/configs/lora/qwen3_8b_lora.yaml

# QLoRA NVFP4 on 4 GPUs
torchrun --nproc_per_node=4 -m xorl.cli.train \
  examples/local/dummy/configs/qlora/qwen3_8b_qlora_nvfp4.yaml

# QLoRA NVFP4 + PP on 8 GPUs
torchrun --nproc_per_node=8 -m xorl.cli.train \
  examples/local/dummy/configs/qlora/qwen3_8b_qlora_nvfp4_pp2.yaml
```

The GPU count in a command must satisfy the topology encoded by that YAML. A checked-in config verifies that the shape is represented in source; it does not guarantee that the model fits a different GPU type or that every optional backend is installed.

## Dummy dataset behavior

```yaml
data:
  datasets:
    - path: dummy
      type: tokenized
      max_seq_len: 8000
```

The dummy loader generates random token IDs, uses the token sequence as labels, and avoids dataset I/O. Configuration files can override sample count, length, packing, and seed, so inspect the selected YAML for the effective workload.
