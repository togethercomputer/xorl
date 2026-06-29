#!/usr/bin/env bash
# Lever sweep: per-layer fullgraph compile under FSDP2, varying reduce-overhead (cudagraph) ×
# reshard_after_forward × cudagraph input-mutation. Captures steady tok/s, 0-breaks, and cudagraph
# record/replay behavior (TORCH_LOGS=cudagraphs → count "Recording cudagraph" lines = re-records).
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
CFG=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
NPROC="${NPROC:-2}"
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1
export TORCH_LOGS="graph_breaks,cudagraphs"
export XORL_ASYNC_METRICS=0   # blocking metrics so per-step tok/s timing is exact for the A/B
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE

run() {  # $1=name  $2=RO  $3=reshard(true/false/none)  $4..=extra train args
  local name=$1 ro=$2 reshard=$3; shift 3
  local LOG=/home/apanda/xorl-oss/results/lever-$name.log
  export XORL_COMPILE_REDUCE_OVERHEAD=$ro
  local args=(--train.max_steps 20)
  [ "$reshard" != "none" ] && args+=(--train.reshard_after_forward "$reshard")
  echo "===== $name : RO=$ro reshard=$reshard extra=$* =====" > "$LOG"
  TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$name \
    "$VENV/bin/torchrun" --standalone --nproc_per_node=$NPROC -m xorl.cli.train "$CFG" \
    "${args[@]}" "$@" >> "$LOG" 2>&1
  echo "===== EXIT $? ($name) =====" >> "$LOG"
}

run L1_compile_only        0 none
run L2_cudagraph_reshardT  1 true
run L3_cudagraph_reshardF  1 false
run L4_compile_reshardF    0 false
echo LEVERS_DONE > /home/apanda/xorl-oss/results/levers-done.marker
