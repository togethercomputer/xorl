#!/usr/bin/env bash
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
cd "$REPO"
PARAMS='{"diagnostic_decode_cache": true, "diagnostic_hidden_components": true, "diagnostic_hidden_component_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], "diagnostic_hidden_component_path": "/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810/results/dsv4_flash_zero_k3/dumps/trainer_ruler_rep{repetition}"}'
exec submodules/xorl-sglang/.venv/bin/python scripts/replay_dsv4_exact_trace.py \
  --url http://127.0.0.1:6000 \
  --trace results/dsv4_flash_zero_k3/trace_nonzero_dump.json \
  --model-id nonzero \
  --loss-fn-params-json "$PARAMS" \
  --output results/dsv4_flash_zero_k3/replay_nonzero_dump.json
