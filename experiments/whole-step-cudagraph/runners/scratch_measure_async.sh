#!/usr/bin/env bash
# A/B: async metrics (default) vs blocking (XORL_ASYNC_METRICS=0). Per-layer fullgraph, no cudagraph.
# Checks: loss curves identical (correctness), 0 breaks, throughput delta.
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
CFG=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1 TORCH_LOGS=graph_breaks
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE XORL_COMPILE_REDUCE_OVERHEAD

for AM in 0 1; do
  LOG=/home/apanda/xorl-oss/results/async-am$AM.log
  export XORL_ASYNC_METRICS=$AM
  echo "===== XORL_ASYNC_METRICS=$AM =====" > "$LOG"
  TORCHINDUCTOR_CACHE_DIR=/tmp/ti_am$AM \
    "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" \
    --train.max_steps 20 >> "$LOG" 2>&1
  echo "===== EXIT $? (am=$AM) =====" >> "$LOG"
done
echo ASYNC_DONE > /home/apanda/xorl-oss/results/async-done.marker
