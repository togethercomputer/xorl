#!/usr/bin/env bash
# Campaign 2: launch the base sampler on the unified canonical-fold branch.
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
cd "$REPO"
echo "parent: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "submodule: $(git -C submodules/xorl-sglang log --oneline -1)"
mkdir -p results/dsv4_flash_zero_k3/campaign2
rm -f results/dsv4_flash_zero_k3/campaign2/sampler_base.log
nohup bash results/dsv4_flash_zero_k3/launch_sampler_base.sh \
  > results/dsv4_flash_zero_k3/campaign2/sampler_base.log 2>&1 &
sleep 2
echo LAUNCHED
tail -3 results/dsv4_flash_zero_k3/campaign2/sampler_base.log
