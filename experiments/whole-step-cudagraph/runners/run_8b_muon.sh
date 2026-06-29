#!/usr/bin/env bash
set -uo pipefail
cd /home/apanda/xorl-oss
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
for M in eager manualgraph; do
  COMPILE=1 FULLGRAPH=0 OPT=muon MODE=$M MODEL=8b S=2048 STEPS=10 \
    .venv-fa4/bin/torchrun --standalone --master_port=29645 --nproc_per_node=8 experiments/whole-step-cudagraph/qwen3_capture_harness.py >/tmp/q88_muon_$M.log 2>&1
done
echo MUON8B_DONE > /tmp/muon8b.marker
