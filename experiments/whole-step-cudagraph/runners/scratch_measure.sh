#!/usr/bin/env bash
# A/B our numbers: per-layer fullgraph compile, reduce-overhead OFF vs ON. fullgraph=1 keeps
# proving 0 breaks. 20 steps so the tok/s moving-average reaches steady state.
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
CFG=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1 TORCH_LOGS=graph_breaks
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE

for RO in 0 1; do
  LOG=/home/apanda/xorl-oss/results/measure-ro$RO.log
  export XORL_COMPILE_REDUCE_OVERHEAD=$RO
  echo "===== RUN reduce_overhead=$RO =====" > "$LOG"
  TORCHINDUCTOR_CACHE_DIR=/tmp/ti_meas_ro$RO \
    "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" \
    --train.max_steps 20 >> "$LOG" 2>&1
  echo "===== EXIT $? (ro=$RO) =====" >> "$LOG"
done
echo ALL_MEASURE_DONE > /home/apanda/xorl-oss/results/measure-done.marker
