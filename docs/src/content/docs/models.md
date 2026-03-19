---
title: Supported Models
---

xorl currently supports the following model architectures.

## Architectures

| Architecture | HuggingFace class | Notes |
|---|---|---|
| Qwen3 (dense) | `Qwen3ForCausalLM` | Standard transformer with GQA, SwiGLU activations, RoPE positional embeddings. |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | Same as Qwen3 but FFN layers are replaced with Mixture-of-Experts blocks. Weight conversion is automatic — see below. |

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

## MoE models: automatic weight conversion

MoE checkpoints from HuggingFace store experts as a `ModuleList` (one module per expert). xorl uses fused grouped-kernel (GKN) tensors for efficient expert dispatch. This conversion happens **automatically during model loading** — no separate preprocessing step is needed. Simply point `model_path` at the standard HuggingFace checkpoint and xorl will fuse the expert weights on the fly.

See the [MoE section](/moe/overview) for details on MoE-specific config options including `expert_parallel_size`, `ep_dispatch`, and `moe_implementation`.
