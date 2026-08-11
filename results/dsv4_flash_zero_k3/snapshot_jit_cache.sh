#!/usr/bin/env bash
# Persist the pod-local JIT cache back to weka for the next pod incarnation.
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
mkdir -p "$REPO/results/dsv4_flash_zero_k3/jit-cache-snapshot"
cp -r /dev/shm/sglang-cache/. "$REPO/results/dsv4_flash_zero_k3/jit-cache-snapshot/" 2>/dev/null
du -sh "$REPO/results/dsv4_flash_zero_k3/jit-cache-snapshot"
