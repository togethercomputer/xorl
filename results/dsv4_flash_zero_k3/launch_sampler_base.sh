#!/usr/bin/env bash
# DSV4-Flash exact-lane sampler (base ruler: enable-lora per contract, no
# adapter preloaded). Contract resolution forces the deterministic exact
# program (marlin MoE, triton fp8, page 256, eager, no radix/graphs).
set -uo pipefail

REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
SNAP=/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
cd "$REPO/submodules/xorl-sglang"

# JIT cache: pod-local /dev/shm (weka rename visibility loses Triton's
# concurrent-compile lock race across the 8 DP schedulers), seeded from the
# lane snapshot; sync back with snapshot_jit_cache.sh after good runs.
WEKA_CACHE="$REPO/results/dsv4_flash_zero_k3/jit-cache-snapshot"
export SGLANG_CACHE_DIR=/dev/shm/sglang-cache
mkdir -p "$SGLANG_CACHE_DIR"
[ -d "$WEKA_CACHE" ] && cp -r "$WEKA_CACHE/." "$SGLANG_CACHE_DIR/" 2>/dev/null

# The exact resolved contract requires these fusion/simulation opts off.
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
  --trust-remote-code \
  --host 127.0.0.1 --port 30000
