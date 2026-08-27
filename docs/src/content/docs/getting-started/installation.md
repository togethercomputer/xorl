---
title: "Installation"
---


## Requirements

- Python 3.12 (the package requires `==3.12.*`)
- An NVIDIA driver compatible with the selected wheel profile
- NVIDIA Hopper (H100/H800) or newer for Hopper-specific NVFP4, DeepEP, and tuned kernel paths

XoRL ships a single combined dependency profile:

| Manifest | PyTorch / CUDA runtime | Triton | Attention stack | Use it for |
|---|---|---|---|---|
| `pyproject.toml` | 2.11.0 / CUDA 13 | 3.6.0 | FlashAttention 4 (`4.0.0b19`) | Local training, the XoRL training server, and the pinned xorl-sglang submodule, all in one environment |

The PyTorch 2.11 pins match the checked-in xorl-sglang package metadata, so its compiled `sglang-kernel` extension loads in the same environment. Do not upgrade or mix the pinned Torch, Triton, or attention packages independently.

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

`uv sync` reads `pyproject.toml` and installs all pinned dependencies into a `.venv` virtual environment, resolving `sglang` to the checked-in `submodules/xorl-sglang` fork via `[tool.uv.sources]` — so the submodules must be checked out first.

## Install with conda

```bash
conda create -n xorl python=3.12
conda activate xorl
pip install -e . -e "submodules/xorl-sglang/python[all]"
```

The second editable install is required with pip/conda: pip does not read `[tool.uv.sources]`, so without it the `sglang[all]` dependency resolves to upstream SGLang on PyPI instead of the checked-in fork.

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

xorl-sglang installs into the same environment as XoRL: the default profile pins the PyTorch 2.11 stack its compiled `sglang-kernel` extension is built against, and the install steps above already include it (uv via `[tool.uv.sources]`, conda via the explicit editable install).

## Verify Installation

```bash
python -c "import torch, triton, xorl, sglang; print(torch.__version__, triton.__version__, xorl.__version__)"
python -c "from flash_attn.cute import flash_attn_func; print('FlashAttention 4 ok')"
python -c "import sgl_kernel; print('sglang-kernel ok')"
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
