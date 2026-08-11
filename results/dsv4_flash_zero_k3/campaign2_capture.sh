#!/usr/bin/env bash
# Campaign 2 trace capture: label + decisions + output are arguments so the
# campaign-1 evidence files stay frozen.
#   usage: campaign2_capture.sh <label> <decisions> <output.json>
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
SNAP=/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
cd "$REPO"
LABEL="${1:?label}"
DECISIONS="${2:?decisions}"
OUTPUT="${3:?output path}"
exec submodules/xorl-sglang/.venv/bin/python scripts/capture_dsv4_exact_trace.py \
  --url http://127.0.0.1:30000 \
  --model-path "$SNAP" \
  --label "$LABEL" \
  --decisions "$DECISIONS" \
  --repetitions 3 \
  --output "$OUTPUT"
