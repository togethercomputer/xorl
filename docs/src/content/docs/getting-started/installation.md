---
title: "Installation"
---


## Requirements

- Python 3.12
- CUDA 12.9+
- PyTorch 2.10+
- NVIDIA Hopper GPU (H100/H800) or newer recommended for NVFP4 and DeepEP

## Clone the repo

```bash
git clone --recurse-submodules https://github.com/togethercomputer/xorl-internal
cd xorl-internal
```

> Already cloned without `--recurse-submodules`? Run `git submodule update --init --recursive`

## Install with uv (recommended)

[uv](https://github.com/astral-sh/uv) is the recommended package manager for reproducible installs.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install and activate
uv sync
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and installs all pinned dependencies into a `.venv` virtual environment.

## Install with conda

```bash
conda create -n xorl python=3.12
conda activate xorl
pip install -e .
```

## Install Submodules

The repo ships two git submodules under `submodules/`:

| Submodule | Description |
|---|---|
| [xorl-client](https://github.com/togethercomputer/xorl-client) | Lightweight Python client for the XoRL training service. Required for server/RL training mode. |
| [xorl-sglang](https://github.com/togethercomputer/xorl-sglang) | XoRL's fork of [SGLang](https://github.com/sgl-project/sglang). Used as the inference engine in online RL loops. |

Install individually:

```bash
pip install -e submodules/xorl-client
pip install -e "submodules/xorl-sglang/python[all]"
```

Alternatively, use the bundled `pyproject.sglang.toml` which pins PyTorch to 2.9.1 (required by sglang) and installs xorl, xorl-client, and xorl-sglang together:

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

> These submodules are only needed for **server training / online RL**. If you are only running local SFT or pretraining, you can skip this step.


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

## DeepEP Install (Optional)

DeepEP is an NVLink-optimized MoE dispatch backend. It is only required when using `ep_dispatch: deepep` in your config — the default `ep_dispatch: alltoall` works without it. Install it from [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP).

### Multi-node prerequisites

For multi-node EP, DeepEP uses NVSHMEM for inter-node RDMA. Two additional steps are required on every node.

**1. Load `nvidia_peermem`**

`nvidia_peermem` bridges the NVIDIA driver and the InfiniBand stack to enable GPUDirect RDMA. Without it, NVSHMEM cannot register GPU buffers with IB HCAs and DeepEP will crash with `SIGABRT` at the first dispatch.

```bash
sudo modprobe nvidia_peermem
```

Verify it is loaded:
```bash
lsmod | grep nvidia_peermem
```

To persist across reboots, add it to `/etc/modules`:
```bash
echo nvidia_peermem | sudo tee -a /etc/modules
```

**2. Enable IBGDA in the NVIDIA driver**

IBGDA allows NVSHMEM to initiate RDMA transfers directly from GPU SM threads without CPU involvement. Add the following to `/etc/modprobe.d/nvidia.conf` on every node:

```
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

Then rebuild the initramfs and reboot:

```bash
sudo update-initramfs -u
sudo reboot
```

Verify the settings are active after reboot:
```bash
sudo cat /proc/driver/nvidia/params | grep -E "EnableStreamMemOPs|RegistryDwords"
# Expected:
# EnableStreamMemOPs: 1
# RegistryDwords: "PeerMappingOverride=1;"
```

> **Note:** `nvidia_peermem` must still be loaded after reboot — it is not automatically enabled by the IBGDA driver settings.


## Next Steps

Head to the [Quick Start](/xorl/getting-started/quickstart/) to run your first training job.
