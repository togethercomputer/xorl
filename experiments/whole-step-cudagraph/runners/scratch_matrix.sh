#!/usr/bin/env bash
# Full matrix: {1.7b, 8b} x {1,2,4,8 GPU} x {eager, manualgraph} — real Qwen3 + FA4 + xorl Muon +
# FSDP2, manual whole-step CUDAGraph capture vs eager. Logs tok/s + MFU + CAPTURE/loss per config.
set -uo pipefail
cd /home/apanda/xorl-oss
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
V=/home/apanda/xorl-oss/.venv-fa4/bin
RES=/home/apanda/xorl-oss/results
: > $RES/matrix-summary.txt
for MODEL in 1.7b 8b; do
  for WORLD in 1 2 4 8; do
    for MODE in eager manualgraph; do
      LOG=$RES/mtx-${MODEL}-w${WORLD}-${MODE}.log
      OPT=muon MODE=$MODE MODEL=$MODEL S=2048 STEPS=12 \
        "$V/torchrun" --standalone --master_port=29601 --nproc_per_node=$WORLD scratch_qwen3_capture.py >$LOG 2>&1
      line=$(grep -hE "MODEL=.*per-step" "$LOG" | tail -1)
      cap=$(grep -qE "CAPTURE OK" "$LOG" && echo cap=ok || echo cap=-)
      fail=$(grep -qiE "FAILED|out of memory|not permitted|Error:" "$LOG" && echo FAIL || echo "")
      echo "${MODEL} w${WORLD} ${MODE}: ${line:-<no result>} $cap $fail" | tee -a $RES/matrix-summary.txt
    done
  done
done
echo MATRIX_DONE > $RES/matrix-done.marker
