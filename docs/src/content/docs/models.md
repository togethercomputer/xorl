---
title: Supported Models
---

XoRL discovers model implementations through the architecture names in the checkpoint configuration. The table below reflects the classes registered by the current source tree.

:::caution[What “registered” means]
A registered architecture has a loadable XoRL implementation. It does not imply that every attention backend, precision, adapter mode, parallel topology, weight-sync backend, or trainer/sampler revision pair has been qualified. A checked-in example is a stronger copy/paste starting point, but it is still not an end-to-end K3 certificate.
:::

## Registered architectures

| Model family | Registered Hugging Face architecture name(s) | Checked-in example |
|---|---|---|
| DeepSeek V3 | `DeepseekV3ForCausalLM` | — |
| DeepSeek V4 | `DeepseekV4ForCausalLM` | — |
| GLM-4 MoE | `Glm4MoeForCausalLM` | `examples/local/dummy/configs/full/glm4_moe_ep8.yaml` |
| GLM-5 / GLM MoE DSA | `Glm5ForCausalLM`, `GlmMoeDsaForCausalLM` | — |
| GPT-OSS | `GptOssForCausalLM` | `examples/local/dummy/configs/full/gpt_oss_20b_ep8.yaml` |
| Llama | `LlamaForCausalLM` | `examples/local/dummy/configs/full/llama3_8b.yaml` |
| MiniMax M3 sparse | `MiniMaxM3SparseForCausalLM`, `MiniMaxM3SparseForConditionalGeneration` | — |
| Nemotron-H | `NemotronHForCausalLM` | — |
| OLMo 2 | `Olmo2ForCausalLM` | — |
| Qwen2 | `Qwen2ForCausalLM` | — |
| Qwen3 | `Qwen3ForCausalLM` | `examples/local/dummy/configs/full/qwen3_8b.yaml` |
| Qwen3 MoE / Coder MoE | `Qwen3MoeForCausalLM` | `examples/local/dummy/configs/full/qwen3_30b_a3b_ep8.yaml` |
| Qwen3.5 | `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration` | `examples/local/dummy/configs/full/qwen3_5_4b.yaml` |
| Qwen3.5 MoE | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | `examples/server/configs/full/qwen3_5_35b_a3b_full.yaml` |

The registry is defined in [`src/xorl/models/registry.py`](https://github.com/togethercomputer/xorl/blob/main/src/xorl/models/registry.py). When `architectures` is a list, the loader selects its first entry (`architectures[0]`). It does not scan later entries for a supported architecture; if the first entry is not registered, loading fails and reports the registered names.

## Checkpoint format

XoRL accepts Hugging Face-style checkpoints:

- `config.json` containing an admitted architecture and model configuration
- one or more `*.safetensors` weight files
- tokenizer files such as `tokenizer.json` and `tokenizer_config.json`

Specify the checkpoint with `model_path`, using either a local path or Hugging Face Hub ID. Use `config_path` and `tokenizer_path` when configuration or tokenizer artifacts live elsewhere.

## Model-loading fields

| Field | Description |
|---|---|
| `model_path` | Local checkpoint path or Hugging Face Hub ID. |
| `config_path` | Configuration path. Defaults to `model_path`. |
| `tokenizer_path` | Tokenizer path. Defaults to `config_path`. |
| `attn_implementation` | Requested attention backend. Availability is architecture-specific. |
| `moe_implementation` | Requested MoE kernel, such as `triton`, `native`, `quack`, or `eager`. Availability is architecture-specific. |

## Checked-in server configurations

Concrete server examples currently cover Qwen3, Qwen3-MoE/Coder, Qwen3.5-MoE, and GPT-OSS across selected full-weight, LoRA, and QLoRA modes. Browse [`examples/server/configs/`](https://github.com/togethercomputer/xorl/tree/main/examples/server/configs) for the exact filenames; do not infer an unlisted filename by changing a model size in another example.

## MoE checkpoint conversion

For supported MoE families, model loading converts the source checkpoint representation into the fused expert layout required by the selected XoRL kernels. The conversion and admitted layouts are architecture-specific; no separate user preprocessing step is required for the checked-in examples.

See [Mixture of Experts](/xorl/moe/overview/) for `expert_parallel_size`, `ep_dispatch`, and MoE implementation options.
