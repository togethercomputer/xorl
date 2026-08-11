#!/usr/bin/env bash
# DSV4-Flash exact trainer server (WORLD8 RCA topology) in the torch-2.11
# combined environment. xorl resolves via PYTHONPATH (editable-shadow), the
# pinned SGLang tree provides serving-value kernels including sgl_kernel.
set -uo pipefail

REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
VENV="$REPO/submodules/xorl-sglang/.venv"
export PYTHONPATH="$REPO/src:$REPO/submodules/xorl-sglang/python"
export HF_HOME=/shared/huggingface
# JIT caches: pod-local /dev/shm (weka rename visibility loses Triton's
# concurrent-compile locking across the 8 torchrun ranks), seeded from the
# lane snapshot; sync back with snapshot_jit_cache.sh after good runs.
WEKA_CACHE="$REPO/results/dsv4_flash_zero_k3/jit-cache-snapshot"
mkdir -p /dev/shm/sglang-cache
[ -d "$WEKA_CACHE" ] && cp -r "$WEKA_CACHE/." /dev/shm/sglang-cache/ 2>/dev/null
export TRITON_CACHE_DIR=/dev/shm/sglang-cache/triton-trainer
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/sglang-cache/inductor-trainer
cd "$REPO"

CONFIG="${DSV4_TRAINER_CONFIG:-$REPO/results/dsv4_flash_zero_k3/trainer_server_lora.yaml}"
exec "$VENV/bin/python" -m xorl.server.launcher \
  --mode auto \
  --api-port 6000 \
  --config "$CONFIG" \
  "$@"
