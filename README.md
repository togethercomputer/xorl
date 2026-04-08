<p align="center">
  <img src="docs/src/assets/logo-light.svg" alt="XoRL" width="240"/>
</p>

<p align="center">
  High-performance distributed training for LLMs — RL, SFT, MoE, and beyond.
</p>

<p align="center">
  <a href="https://togethercomputer.github.io/xorl/getting-started/installation/">🚀 Installation</a> ·
  <a href="https://togethercomputer.github.io/xorl/getting-started/quickstart/">⚡ Quick Start</a> ·
  <a href="https://togethercomputer.github.io/xorl">📚 Documentation</a>
</p>

---

## 🔍 Overview

XoRL is a distributed training framework designed for large language models with composable parallelism and flexible training modes.

**Two training modes:**

- **Local** — `torchrun`-based training for offline SFT and pretraining
- **Server** — REST API-driven training for online RL loops where an external orchestrator (e.g. [xorl_client](https://github.com/xorl-org/xorl_client)) controls the training loop

**Parallelism strategies** — mix and match freely:

| Strategy | Description |
|---|---|
| FSDP2 | Fully sharded data parallelism (PyTorch native) |
| Tensor Parallel | Column/row weight sharding across GPUs |
| Pipeline Parallel | Interleaved 1F1B schedule across stages |
| Context Parallel | Ring attention + Ulysses sequence parallel |
| Expert Parallel | MoE expert sharding via [DeepEP](https://github.com/deepseek-ai/DeepEP) |

**Fine-tuning methods** — full weights, [LoRA](https://togethercomputer.github.io/xorl/adapters/lora/), and [QLoRA](https://togethercomputer.github.io/xorl/adapters/qlora/) (int4/nvfp4/block_fp8), all FSDP2-compatible.

---

## 🚀 Installation

```bash
git clone --recurse-submodules git@github.com:togethercomputer/xorl.git
cd xorl
uv sync
```

> Already cloned without `--recurse-submodules`? Run `git submodule update --init --recursive`

See the [installation guide](https://togethercomputer.github.io/xorl/getting-started/installation/) for full setup including optional dependencies (DeepEP, Flash Attention).

## ⚡ Quick Start

```bash
# Local training on 8 GPUs
torchrun --nproc_per_node=8 -m xorl.cli.train examples/local/dummy/configs/full/qwen3_8b.yaml
```

See the [quick start guide](https://togethercomputer.github.io/xorl/getting-started/quickstart/) for more examples including MoE, server training, and LoRA.

---

## 📚 Documentation

| Topic | Link |
|---|---|
| Parallelism | [Overview](https://togethercomputer.github.io/xorl/parallelism/overview/) |
| MoE & DeepEP | [MoE docs](https://togethercomputer.github.io/xorl/moe/overview/) |
| LoRA / QLoRA | [Adapters](https://togethercomputer.github.io/xorl/adapters/lora/) |
| Server training | [Server docs](https://togethercomputer.github.io/xorl/server-training/overview/) |
| Config reference | [Local](https://togethercomputer.github.io/xorl/config-reference/local/) · [Server](https://togethercomputer.github.io/xorl/config-reference/server/) |

---

## 🧠 Supported Models

| Model | Type | HuggingFace ID |
|---|---|---|
| Qwen3 | Dense | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`, ... |
| Qwen3-MoE | Mixture-of-Experts | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B`, ... |
| Qwen3.5 | Dense | `Qwen/Qwen3.5-7B`, ... |
| Qwen3.5-MoE | Mixture-of-Experts | `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-397B-A17B`, ... |

Models are loaded directly from HuggingFace checkpoints — no preprocessing needed. See the [supported models](https://togethercomputer.github.io/xorl/models/) page for details.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and how to run tests.
