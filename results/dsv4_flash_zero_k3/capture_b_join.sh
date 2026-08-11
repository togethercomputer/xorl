#!/usr/bin/env bash
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
SNAP=/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
cd "$REPO"
PY=submodules/xorl-sglang/.venv/bin/python
$PY scripts/capture_dsv4_exact_trace.py \
  --url http://127.0.0.1:30000 --model-path "$SNAP" \
  --label b1-trained-4dec --adapter trained --decisions 4 --repetitions 3 \
  --output results/dsv4_flash_zero_k3/trace_b1_trained_4dec.json
$PY scripts/capture_dsv4_exact_trace.py \
  --url http://127.0.0.1:30000 --model-path "$SNAP" \
  --label b2-trained-64dec --adapter trained --decisions 64 --repetitions 3 \
  --output results/dsv4_flash_zero_k3/trace_b2_trained_64dec.json
echo B_JOIN_CAPTURES_DONE
