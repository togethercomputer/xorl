#!/usr/bin/env bash
# Portable runner for the whole-step capture harness.
# Point VENV at the cu130 + cu13-FA4 python env (build recipe in ../NOTES.md).
# Knobs (passed through to the harness): MODEL=1.7b|8b MODE=eager|manualgraph OPT=muon|adamw
#   COMPILE=0|1 FULLGRAPH=0|1 S=<seqlen> STEPS=<n> GPUS=<n>
# Example: GPUS=2 MODEL=1.7b COMPILE=1 MODE=manualgraph ./run.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
VENV="${VENV:-$REPO/.venv-fa4}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"   # for the xorl Muon import
exec "$VENV/bin/torchrun" --standalone --nproc_per_node="${GPUS:-2}" \
  "$REPO/experiments/whole-step-cudagraph/qwen3_capture_harness.py"
