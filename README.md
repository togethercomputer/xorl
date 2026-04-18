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

**The XoRL stack consists of three repos:**

| Repo &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
|---|---|
| **[xorl](https://github.com/togethercomputer/xorl-internal)** | Distributed training framework — local SFT/pretraining and server-mode RL training |
| **[xorl-client](https://github.com/togethercomputer/xorl-client)** | Lightweight Python SDK for driving the xorl training server (forward/backward, optimizer steps, checkpointing, sampling) |
| **[xorl-sglang](https://github.com/togethercomputer/xorl-sglang)** | Fork of [SGLang](https://github.com/sgl-project/sglang) with weight-sync APIs, MoE routing export, and numerical alignment for online RL |

**Two training modes:**

- **Local** — `torchrun`-based training for offline SFT and pretraining
- **Server** — REST API-driven training for online RL loops where [xorl-client](https://github.com/togethercomputer/xorl-client) drives the training loop and [xorl-sglang](https://github.com/togethercomputer/xorl-sglang) serves inference

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
git clone --recurse-submodules git@github.com:togethercomputer/xorl-internal.git
cd xorl-internal
```

> Already cloned without `--recurse-submodules`? Run `git submodule update --init --recursive`

### Option A: uv (recommended)

```bash
uv sync
source .venv/bin/activate
```

### Option B: conda

```bash
conda create -n xorl python=3.12
conda activate xorl
pip install -e .
```

### Submodules

The repo includes two git submodules under `submodules/` (needed for server / online RL training):

- **[xorl-client](https://github.com/togethercomputer/xorl-client)** — Lightweight Python SDK (no PyTorch dependency) for driving the xorl training server. Provides `ServiceClient`, `TrainingClient`, `SamplingClient`, and `RestClient` with async-first `APIFuture` semantics, automatic request ordering, and Tinker API compatibility.
- **[xorl-sglang](https://github.com/togethercomputer/xorl-sglang)** — XoRL's fork of [SGLang](https://github.com/sgl-project/sglang) with NCCL-based weight sync endpoints, MoE routing data export (R3), and numerical alignment flags for online RL.

Install individually:

```bash
pip install -e submodules/xorl-client
pip install -e "submodules/xorl-sglang/python[all]"
```

Or use the bundled `pyproject.sglang.toml` which pins PyTorch to 2.9.1 (required by sglang) and installs everything together:

**uv:**
```bash
cp pyproject.sglang.toml pyproject.toml
uv sync
source .venv/bin/activate
```

**conda:**
```bash
conda create -n xorl-sglang python=3.12
conda activate xorl-sglang
cp pyproject.sglang.toml pyproject.toml
pip install -e .
```

> **Note:** The default `pyproject.toml` uses PyTorch 2.10.0. sglang requires PyTorch 2.9.1, so the two cannot coexist in the same environment unless you use `pyproject.sglang.toml`.

See the [installation guide](https://togethercomputer.github.io/xorl-internal/getting-started/installation/) for full setup including optional dependencies (DeepEP, Flash Attention).

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
