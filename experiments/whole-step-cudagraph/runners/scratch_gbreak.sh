#!/usr/bin/env bash
# Whole-step graph-break census: 2-GPU FSDP2, compile-only (NO reduce-overhead, so the
# cudagraph-trees crash can't fire — we only want the Dynamo break count + reasons).
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
LOG=/home/apanda/xorl-oss/results/gbreak-run.log
CFG="${CFG:-examples/local/dummy/configs/full/qwen3_1.7b_ws2.yaml}"

export PYTHONPATH=/home/apanda/xorl-oss/src
export HF_HOME=/shared/huggingface
export XORL_COMPILE_WHOLE_STEP=1
export XORL_COMPILE_REDUCE_OVERHEAD="${XORL_COMPILE_REDUCE_OVERHEAD:-0}"
export XORL_WHOLE_STEP_GRAPH_BREAK_LOG=1
export TORCH_LOGS="${TORCH_LOGS:-graph_breaks}"
# Keep NCCL quiet-ish; 2 GPUs single node.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== gbreak run: CFG=$CFG REDUCE_OVERHEAD=$XORL_COMPILE_REDUCE_OVERHEAD TORCH_LOGS=$TORCH_LOGS ===" > "$LOG"
"$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" >> "$LOG" 2>&1
echo "=== EXIT CODE $? ===" >> "$LOG"
