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

## Install Submodules

The repo ships two git submodules under `submodules/`:

| Submodule | Description |
|---|---|
| [xorl-client](https://github.com/togethercomputer/xorl-client) | Lightweight Python client for the XoRL training service. Required for server/RL training mode. |
| [xorl-sglang](https://github.com/togethercomputer/xorl-sglang) | XoRL's fork of [SGLang](https://github.com/sgl-project/sglang). Used as the inference engine in online RL loops. |

Install the client in the default environment. Keep SGLang in an isolated
Torch-2.11 environment so its compiled kernel wheel never enters the default
Torch-2.12 profile:

```bash
pip install -e submodules/xorl-client
uv venv .venv-sglang --python 3.12
uv pip install --python .venv-sglang/bin/python -e submodules/xorl-sglang/python
uv pip install --python .venv-sglang/bin/python \
  torchdata==0.11.0 nvidia-cutlass-dsl==4.5.2 quack-kernels==0.5.0
uv pip install --python .venv-sglang/bin/python --no-deps -e .
uv pip install --python .venv-sglang/bin/python pytest
PYTHONPATH=src:submodules/xorl-sglang/python XORL_REQUIRE_SGL_KERNEL=1 \
  .venv-sglang/bin/python -m pytest -q tests/ops/test_sgl_kernel_smoke.py
```

The bundled `pyproject.sglang.toml` provides the same combined profile for uv.
Its dependency overrides retain the Quack/CUTLASS versions required by XoRL's
trainer imports while the exact-kernel smoke validates the SGLang boundary.

**uv:**
```bash
cp pyproject.sglang.toml pyproject.toml
uv sync
source .venv/bin/activate
```

> **Note:** The default `pyproject.toml` uses Torch 2.12.1. Pinned SGLang requires Torch 2.11.0; do not install `sglang-kernel` into the default environment.

> These submodules are only needed for **server training / online RL**. If you are only running local SFT or pretraining, you can skip this step.


## Key Dependencies

| Package | Version | Notes |
|---|---|---|
| PyTorch | 2.12.1 | Default XoRL profile; SGLang profile uses 2.11.0 |
| Flash Attention 4 | pinned | Selected by each Torch profile |
| Triton | 3.7.1 | Default profile; SGLang profile uses 3.6.0 |
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
