#!/usr/bin/env bash
set -uo pipefail
cd /home/apanda/xorl-oss
V=/home/apanda/xorl-oss/.venv-n130/bin
export PYTHONPATH= HF_HOME=/shared/huggingface
# sizes: name H NLAYERS B S  (small=launch-bound ... big=compute-bound)
run_size(){ local name=$1 H=$2 NL=$3 B=$4 S=$5; local LOG=/home/apanda/xorl-oss/results/mg-$name.log; : > "$LOG"
  for M in eager manualgraph; do
    H=$H NLAYERS=$NL B=$B S=$S MODE=$M "$V/torchrun" --standalone --nproc_per_node=2 scratch_manual_cudagraph.py 2>&1 \
      | grep -E "MODE=$M|CAPTURE OK|FAILED" >> "$LOG"
  done; }
run_size small  2048 4  8 512
run_size medium 4096 12 8 2048
run_size big    4096 16 8 2048
echo MG_SWEEP_DONE > /home/apanda/xorl-oss/results/mg-sweep-done.marker
