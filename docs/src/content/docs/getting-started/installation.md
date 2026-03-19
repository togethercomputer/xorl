---
title: "Installation"
---


## Requirements

- Python 3.12
- CUDA 12.9+
- PyTorch 2.10+
- NVIDIA Hopper GPU (H100/H800) or newer recommended for NVFP4 and DeepEP

## Install with uv (recommended)

[uv](https://github.com/astral-sh/uv) is the recommended package manager for reproducible installs.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/togethercomputer/xorl-internal
cd xorl-internal
uv sync
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and installs all pinned dependencies.

## Install with pip

```bash
pip install -e .
```

## Optional: DeepEP (NVLink-optimized MoE dispatch)

DeepEP requires a separate wheel installation. Download from the [xorl-wheels releases](https://github.com/xorl-org/xorl-wheels/releases) and install:

```bash
pip install deep_ep-*.whl
```

DeepEP is only required when using `ep_dispatch: deepep` in your config. The default `ep_dispatch: alltoall` works without it.

## Key Dependencies

| Package | Version | Notes |
|---|---|---|
| PyTorch | 2.10.0+cu129 | CUDA 12.9 build |
| Flash Attention 3 | custom | FA3 + FA4 wheels |
| Triton | 3.6.0 | MoE fused kernels |
| Transformers | 5.0+ | Model loading |
| FastAPI + uvicorn | latest | Server training API |
| pyzmq | latest | Worker communication |
| wandb | latest | Experiment tracking (optional) |

## Verify Installation

```bash
python -c "import xorl; print('xorl ok')"
python -c "import flash_attn_interface; print('flash_attn_3 ok')"
python -c "from flash_attn.cute import flash_attn_func; print('flash_attn_4 ok')"
python -c "import deep_ep; print('deepep ok')"  # optional
```

## Multi-node Setup

For multi-node training, install xorl identically on all nodes. Ensure:
- Same Python environment and package versions on all nodes
- Passwordless SSH between nodes (for `rl_launch.sh`)
- NCCL visible across nodes (InfiniBand or RoCE recommended)
- `NCCL_SOCKET_IFNAME` set to the correct network interface

## Next Steps

Head to the [Quick Start](/getting-started/quickstart/) to run your first training job.
