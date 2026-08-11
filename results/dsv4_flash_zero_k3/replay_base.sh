#!/usr/bin/env bash
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
cd "$REPO"
TRACE="${1:-results/dsv4_flash_zero_k3/trace_base_4dec.json}"
OUT="${2:-results/dsv4_flash_zero_k3/replay_base_4dec.json}"
exec submodules/xorl-sglang/.venv/bin/python scripts/replay_dsv4_exact_trace.py \
  --url http://127.0.0.1:6000 \
  --trace "$TRACE" \
  --model-id default \
  --output "$OUT"
