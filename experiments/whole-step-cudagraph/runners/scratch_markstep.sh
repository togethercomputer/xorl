#!/usr/bin/env bash
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
CFG=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1 TORCH_LOGS="graph_breaks,cudagraphs" XORL_ASYNC_METRICS=0
export XORL_COMPILE_REDUCE_OVERHEAD=1
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE
run(){ local name=$1 mark=$2 reshard=$3; local LOG=/home/apanda/xorl-oss/results/mark-$name.log
  export XORL_CUDAGRAPH_MARK_STEP=$mark; local a=(--train.max_steps 20)
  [ "$reshard" != none ] && a+=(--train.reshard_after_forward $reshard)
  echo "== $name mark=$mark reshard=$reshard ==" > "$LOG"
  TORCHINDUCTOR_CACHE_DIR=/tmp/ti_mark_$name "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" "${a[@]}" >> "$LOG" 2>&1
  echo "== EXIT $? ==" >> "$LOG"; }
run M1_mark_step      step  none
run M2_mark_mb        mb    none
run M3_mark_step_resF step  false
echo MARK_DONE > /home/apanda/xorl-oss/results/mark-done.marker
