---
title: Supported Models
---

xorl currently supports the following model architectures.

## Architectures

| Architecture | HuggingFace class | Example models | Notes |
|---|---|---|---|
| Qwen3 (dense) | `Qwen3ForCausalLM` | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B` | Standard transformer with GQA, SwiGLU, RoPE. |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B` | Mixture-of-Experts with top-k routing. Weight conversion is automatic — see below. |
| Qwen3.5 (dense) | `Qwen3_5ForCausalLM` | `Qwen/Qwen3.5-7B` | Qwen3.5 dense with hybrid full/linear attention layers. |
| Qwen3.5-MoE | `Qwen3_5MoeForCausalLM` | `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-397B-A17B` | Qwen3.5 MoE with hybrid attention and grouped expert routing. |

Model selection is config-based: xorl reads `model_type` from `config.json` inside `model_path` and instantiates the appropriate class automatically.

## Checkpoint format

xorl expects checkpoints in HuggingFace format:

- `config.json` — model architecture config
- `*.safetensors` — weight shards (single file or multi-shard)
- `tokenizer.json` / `tokenizer_config.json` — tokenizer files

Specify the checkpoint with `model_path` (local path or HF Hub ID). Use `config_path` and `tokenizer_path` separately if your config/tokenizer lives in a different location than the weights.

## Key config fields for model loading

| Field | Description |
|---|---|
| `model_path` | Local path or HF Hub ID for weights. |
| `config_path` | Path to `config.json`. Defaults to `model_path`. |
| `tokenizer_path` | Path to tokenizer files. Defaults to `config_path`. |
| `attn_implementation` | Attention backend: `flash_attention_3`, `flash_attention_4`, `native`, `sdpa`, `eager`. |
| `moe_implementation` | MoE kernel: `null` (auto), `triton`, `native`, `quack`, `eager`. |

## Tested configurations

The following model + training-mode combinations have pre-built example configs under `examples/server/configs/`:

| Model | Full weights | LoRA |
|---|---|---|
| Qwen3-8B | `qwen3_8b_full.yaml` | `qwen3_8b_lora.yaml` |
| Qwen3-30B-A3B (MoE) | `qwen3_30b_a3b_full.yaml` | — |
| Qwen3-Coder-30B-A3B (MoE) | `qwen3_coder_30b_a3b_full.yaml` | `qwen3_coder_30b_a3b_lora.yaml` |
| Qwen3-235B-A22B (MoE) | `qwen3_235b_a22b_8node_ep64.yaml` | — |
| Qwen3.5-35B-A3B (MoE) | `qwen3_5_35b_a3b_full.yaml` | `qwen3_5_35b_a3b_lora.yaml` |
| Qwen3.5-397B-A17B (MoE) | `qwen3_5_397b_a17b_full.yaml` | `qwen3_5_397b_a17b_lora.yaml` |

## MoE models: automatic weight conversion

MoE checkpoints from HuggingFace store experts as a `ModuleList` (one module per expert). xorl uses fused grouped-kernel (GKN) tensors for efficient expert dispatch. This conversion happens **automatically during model loading** — no separate preprocessing step is needed. Simply point `model_path` at the standard HuggingFace checkpoint and xorl will fuse the expert weights on the fly.

See the [MoE section](/moe/overview) for details on MoE-specific config options including `expert_parallel_size`, `ep_dispatch`, and `moe_implementation`.
