#!/usr/bin/env bash
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
SNAP=/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
cd "$REPO"
exec submodules/xorl-sglang/.venv/bin/python scripts/capture_dsv4_exact_trace.py \
  --url http://127.0.0.1:30000 \
  --model-path "$SNAP" \
  --label base-ruler-4dec-dump \
  --decisions 4 \
  --repetitions 2 \
  --output results/dsv4_flash_zero_k3/trace_base_4dec_dump.json
