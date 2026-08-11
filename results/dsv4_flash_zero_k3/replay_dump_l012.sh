#!/usr/bin/env bash
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
cd "$REPO"
PARAMS='{"diagnostic_decode_cache": true, "diagnostic_hidden_components": true, "diagnostic_hidden_component_layers": [0, 1, 2], "diagnostic_hidden_component_path": "/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810/results/dsv4_flash_zero_k3/dumps/trainer_ruler_rep{repetition}"}'
exec submodules/xorl-sglang/.venv/bin/python scripts/replay_dsv4_exact_trace.py \
  --url http://127.0.0.1:6000 \
  --trace results/dsv4_flash_zero_k3/trace_base_4dec.json \
  --model-id ruler \
  --loss-fn-params-json "$PARAMS" \
  --output results/dsv4_flash_zero_k3/replay_base_dump.json
