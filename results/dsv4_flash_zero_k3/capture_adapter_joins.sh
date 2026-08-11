#!/usr/bin/env bash
# Adapter A-join captures: base + zero + nonzero + perturbed, same server.
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
SNAP=/shared/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
cd "$REPO"
PY=submodules/xorl-sglang/.venv/bin/python
run() {
  local label="$1" out="$2"; shift 2
  $PY scripts/capture_dsv4_exact_trace.py \
    --url http://127.0.0.1:30000 \
    --model-path "$SNAP" \
    --label "$label" \
    --decisions 4 \
    --repetitions 3 \
    --output "results/dsv4_flash_zero_k3/$out" "$@"
}
run base-a-join trace_ajoin_base_4dec.json
run zero-a-join trace_ajoin_zero_4dec.json --adapter zero
run nonzero-a-join trace_ajoin_nonzero_4dec.json --adapter nonzero
run perturbed-control trace_ajoin_perturbed_4dec.json --adapter perturbed
echo A_JOIN_CAPTURES_DONE
