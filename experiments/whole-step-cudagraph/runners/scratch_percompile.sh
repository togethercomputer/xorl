#!/usr/bin/env bash
# Supported FSDP2+compile pattern on the REAL xorl path: per-layer compile-before-fully_shard,
# fullgraph=True (XORL_COMPILE_FULLGRAPH=1), NO whole-step. fullgraph makes any residual graph
# break a hard error, so a clean 3-step run == 0 graph breaks per compiled decoder layer.
set -uo pipefail
cd /home/apanda/xorl-oss
VENV="${VENV:-/home/apanda/xorl-internal/.venv}"
LOG=/home/apanda/xorl-oss/results/percompile-run.log
CFG="${CFG:-examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml}"

export PYTHONPATH=/home/apanda/xorl-oss/src
export HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1
export TORCH_LOGS="${TORCH_LOGS:-graph_breaks}"
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE XORL_COMPILE_REDUCE_OVERHEAD

echo "=== percompile run: CFG=$CFG VENV=$VENV FULLGRAPH=1 TORCH_LOGS=$TORCH_LOGS ===" > "$LOG"
"$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" >> "$LOG" 2>&1
echo "=== EXIT CODE $? ===" >> "$LOG"
