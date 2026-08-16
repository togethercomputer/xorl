---
title: "Installation"
---


## Requirements

- Python 3.12 (the package requires `==3.12.*`)
- An NVIDIA driver compatible with the selected wheel profile
- NVIDIA Hopper (H100/H800) or newer for Hopper-specific NVFP4, DeepEP, and tuned kernel paths

XoRL ships two deliberately different dependency profiles:

| Profile | Manifest | PyTorch / CUDA runtime | Triton | Attention stack | Use it for |
|---|---|---|---|---|---|
| Default | `pyproject.toml` | 2.12.1 / CUDA 13.2 | 3.7.1 | FlashAttention 4 (`4.0.0b19`) | Local training and the XoRL training server |
| Combined xorl-sglang | `pyproject.sglang.toml` | 2.11.0 / CUDA 13 | 3.6.0 | FlashAttention 4 (`4.0.0b19`) | A single environment that also runs the pinned xorl-sglang submodule |

These profiles are not interchangeable. Use the manifest that matches the process you intend to run rather than upgrading or mixing their pinned Torch, Triton, or attention packages independently.

## Clone the repo

```bash
git clone --recurse-submodules https://github.com/togethercomputer/xorl
cd xorl
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

## Install submodules

The repo ships two git submodules under `submodules/`:

| Submodule | Description |
|---|---|
| [xorl-client](https://github.com/togethercomputer/xorl-client) | Lightweight Python client for the XoRL training service. Required for server/RL training mode. |
| [xorl-sglang](https://github.com/togethercomputer/xorl-sglang) | XoRL's fork of [SGLang](https://github.com/sgl-project/sglang). Used as the inference engine in online RL loops. |

The default XoRL dependency set already installs `xorl-client` from its public repository. To develop the checked-in client submodule in place, install it editable:

```bash
pip install -e submodules/xorl-client
```

Do not install xorl-sglang into the default PyTorch 2.12 environment. To install XoRL, xorl-client, and xorl-sglang together, use the alternate manifest:

**uv:**
```bash
cp pyproject.sglang.toml pyproject.toml
UV_PROJECT_ENVIRONMENT=.venv-sglang uv sync
source .venv-sglang/bin/activate
```

**conda:**
```bash
conda create -n xorl-sglang python=3.12
conda activate xorl-sglang
cp pyproject.sglang.toml pyproject.toml
pip install -e . -e "submodules/xorl-sglang/python[all]"
```

> **Note:** Copying the alternate manifest replaces the tracked `pyproject.toml`; `uv sync` also generates the ignored local `uv.lock` for this profile. Do this in a clean checkout, restore `pyproject.toml`, and do not add the generated lock with unrelated changes. The separate `.venv-sglang` keeps this profile isolated from the default `.venv`. The version table above is the source of truth for the two profiles.

> These submodules are only needed for **server training / online RL**. If you are only running local SFT or pretraining, you can skip this step.


## Verify Installation

For the default profile:

```bash
python -c "import torch, triton, xorl; print(torch.__version__, triton.__version__, xorl.__version__)"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4 ok')"
```

For the combined xorl-sglang profile:

```bash
python -c "import torch, triton, xorl, sglang; print(torch.__version__, triton.__version__)"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4 ok')"
```

## DeepEP Install (Optional)

DeepEP is a GPU-resident MoE dispatch backend. It uses high-speed GPU interconnects within a node and NVSHMEM/GPUDirect RDMA for supported multi-node deployments. It is only required when using `ep_dispatch: deepep`; the default `ep_dispatch: alltoall` works without it. Install it from [DeepSeek's DeepEP repository](https://github.com/deepseek-ai/DeepEP), then verify it separately with `python -c "import deep_ep; print('DeepEP ok')"`.

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
