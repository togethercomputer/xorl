#!/usr/bin/env bash
# DSV4-Flash exact-lane sampler with the three qualification adapters
# preloaded (zero / distinguishable nonzero / perturbed negative control).
set -uo pipefail

REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
RD="$REPO/results/dsv4_flash_zero_k3"
SNAP=/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
cd "$REPO/submodules/xorl-sglang"

WEKA_CACHE="$RD/jit-cache-snapshot"
export SGLANG_CACHE_DIR=/dev/shm/sglang-cache
mkdir -p "$SGLANG_CACHE_DIR"
[ -d "$WEKA_CACHE" ] && cp -r "$WEKA_CACHE/." "$SGLANG_CACHE_DIR/" 2>/dev/null
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_SIMULATE_UNIFORM_EXPERTS=0
export SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=0
export SGLANG_OPT_MOE_QUANT_ONCE=0
export SGLANG_SHARED_EXPERT_TP1=0

exec .venv/bin/python -m sglang.launch_server \
  --model-path "$SNAP" \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --rl-on-policy-target xorl \
  --tp-size 8 --dp-size 8 --ep-size 8 \
  --enable-dp-attention \
  --enable-lora \
  --lora-paths zero="$RD/adapter_zero" nonzero="$RD/adapter_nonzero" perturbed="$RD/adapter_nonzero_perturbed" \
  --trust-remote-code \
  --host 127.0.0.1 --port 30000
