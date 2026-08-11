#!/usr/bin/env bash
# DSV4-Flash exact trainer server (WORLD8 RCA topology) in the torch-2.11
# combined environment. xorl resolves via PYTHONPATH (editable-shadow), the
# pinned SGLang tree provides serving-value kernels including sgl_kernel.
set -uo pipefail

REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
VENV="$REPO/submodules/xorl-sglang/.venv"
export PYTHONPATH="$REPO/src:$REPO/submodules/xorl-sglang/python"
export HF_HOME=/shared/huggingface
# Lane-scoped JIT caches: ~/.triton is weka-shared across hosts and races.
export TRITON_CACHE_DIR="$REPO/results/dsv4_flash_zero_k3/jit-cache/triton-trainer"
export TORCHINDUCTOR_CACHE_DIR="$REPO/results/dsv4_flash_zero_k3/jit-cache/inductor-trainer"
cd "$REPO"

CONFIG="${DSV4_TRAINER_CONFIG:-$REPO/results/dsv4_flash_zero_k3/trainer_server_lora.yaml}"
exec "$VENV/bin/python" -m xorl.server.launcher \
  --mode auto \
  --api-port 6000 \
  --config "$CONFIG" \
  "$@"
