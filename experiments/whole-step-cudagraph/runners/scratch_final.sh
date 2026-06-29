#!/usr/bin/env bash
# Final batch: (B1) best compile-only config 1.7B; (B3) 8B at scale on 4 GPUs; (NSYS) GPU-busy% +
# launch counts on the 1.7B best config to quantify compute-bound headroom.
set -uo pipefail
cd /home/apanda/xorl-oss
VENV=/home/apanda/xorl-internal/.venv
export PYTHONPATH=/home/apanda/xorl-oss/src HF_HOME=/shared/huggingface
export XORL_COMPILE_FULLGRAPH=1 TORCH_LOGS=graph_breaks
unset XORL_COMPILE_WHOLE_STEP XORL_COMPILE_WHOLE_BACKBONE XORL_COMPILE_REDUCE_OVERHEAD
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems*/*/bin/nsys 2>/dev/null | head -1)

CFG17=examples/local/dummy/configs/full/qwen3_1.7b_percompile.yaml
CFG8=examples/local/dummy/configs/full/qwen3_8b.yaml

# B1: 1.7B best config (compile-only, reshard=False, async metrics ON)
L=/home/apanda/xorl-oss/results/final-B1.log; export XORL_ASYNC_METRICS=1
echo "== B1 1.7B compile-only reshard=False async=1 ==" > "$L"
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_B1 "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG17" \
  --train.max_steps 25 --train.reshard_after_forward false >> "$L" 2>&1
echo "== EXIT $? ==" >> "$L"

# B3: 8B at scale, 4 GPUs FSDP shard=4, compile-only fullgraph, reshard=False, grad-ckpt on
L=/home/apanda/xorl-oss/results/final-B3-8b.log
echo "== B3 8B 4gpu compile-only fullgraph reshard=False ==" > "$L"
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_B3 "$VENV/bin/torchrun" --standalone --nproc_per_node=4 -m xorl.cli.train "$CFG8" \
  --train.data_parallel_shard_size 4 --train.enable_compile true --train.reshard_after_forward false \
  --data.sample_packing_sequence_len 16384 --train.max_steps 12 \
  --train.save_steps 0 --train.save_hf_weights false >> "$L" 2>&1
echo "== EXIT $? ==" >> "$L"

# NSYS: GPU-busy + launch counts on the 1.7B best config (short)
L=/home/apanda/xorl-oss/results/final-nsys.log; export XORL_ASYNC_METRICS=1
echo "== NSYS 1.7B best config ==" > "$L"
"$NSYS" profile -t cuda -s none --force-overwrite=true -o /home/apanda/xorl-oss/results/final_nsys \
  "$VENV/bin/torchrun" --standalone --nproc_per_node=2 -m xorl.cli.train "$CFG17" \
  --train.max_steps 10 --train.reshard_after_forward false >> "$L" 2>&1
echo "-- cuda_api_sum (launch counts: cudaLaunchKernel vs cudaGraphLaunch) --" >> "$L"
"$NSYS" stats --report cuda_api_sum /home/apanda/xorl-oss/results/final_nsys.nsys-rep 2>&1 | head -20 >> "$L"
echo "-- cuda_gpu_kern_sum (total GPU kernel time) --" >> "$L"
"$NSYS" stats --report cuda_gpu_kern_sum /home/apanda/xorl-oss/results/final_nsys.nsys-rep 2>&1 | head -12 >> "$L"
echo "== EXIT $? ==" >> "$L"
echo FINAL_DONE > /home/apanda/xorl-oss/results/final-done.marker
