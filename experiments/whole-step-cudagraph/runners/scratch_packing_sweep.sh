#!/usr/bin/env bash
# Map the compute-bound ↔ launch-bound regime: sweep packed seq len, compare per-layer
# compile-only (RO=0) vs per-layer cudagraph + reshard_after_forward=False (RO=1). Smaller packing
# = less compute/step = more launch-bound = where cudagraph theory says it should start to help.
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
CFG=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
NPROC="${NPROC:-2}"
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1 TORCH_LOGS=graph_breaks XORL_ASYNC_METRICS=0
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE

PACKINGS="${PACKINGS:-2048 4096 8192 16384}"
for P in $PACKINGS; do
  for RO in 0 1; do
    LOG=/home/apanda/xorl-oss/results/pack-${P}-ro${RO}.log
    export XORL_COMPILE_REDUCE_OVERHEAD=$RO
    args=(--train.max_steps 18 --data.sample_packing_sequence_len $P)
    [ "$RO" = "1" ] && args+=(--train.reshard_after_forward false)
    echo "===== packing=$P RO=$RO =====" > "$LOG"
    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_pack_${P}_${RO} \
      "$VENV/bin/torchrun" --standalone --nproc_per_node=$NPROC -m xorl.cli.train "$CFG" \
      "${args[@]}" >> "$LOG" 2>&1
    echo "===== EXIT $? =====" >> "$LOG"
  done
done
echo PACK_DONE > /home/apanda/xorl-oss/results/pack-done.marker
