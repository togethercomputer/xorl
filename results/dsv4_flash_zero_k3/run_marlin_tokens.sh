#!/usr/bin/env bash
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
cd "$REPO"
export CUDA_VISIBLE_DEVICES="${1:-7}"
TOKENS="${2:-74}"
submodules/xorl-sglang/.venv/bin/python scripts/qualify_dsv4_marlin_lora.py \
  --tokens "$TOKENS" \
  --output "results/dsv4_flash_zero_k3/marlin_tokens_${TOKENS}.json"
grep -E "repeat|finite" "results/dsv4_flash_zero_k3/marlin_tokens_${TOKENS}.json"
