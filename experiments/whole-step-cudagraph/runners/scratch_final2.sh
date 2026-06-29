#!/usr/bin/env bash
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
CFG=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export TORCH_LOGS=graph_breaks XORL_ASYNC_METRICS=0
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE
# E1: eager (no compile) — what compile buys
L=results/final-eager.log; unset XORL_COMPILE_FULLGRAPH XORL_COMPILE_REDUCE_OVERHEAD
echo "== eager (enable_compile=false) ==" > "$L"
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_eager "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" \
  --train.enable_compile false --train.max_steps 20 >> "$L" 2>&1; echo "== EXIT $? ==" >> "$L"
# E2: per-layer cudagraph with TREES=0 (standalone cudagraphs, not trees)
L=results/final-trees0.log; export XORL_COMPILE_FULLGRAPH=1 XORL_COMPILE_REDUCE_OVERHEAD=1 TORCHINDUCTOR_CUDAGRAPH_TREES=0
echo "== cudagraph TREES=0 (standalone) ==" > "$L"
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_trees0 "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG" \
  --train.max_steps 20 >> "$L" 2>&1; echo "== EXIT $? ==" >> "$L"
echo FINAL2_DONE > results/final2-done.marker
