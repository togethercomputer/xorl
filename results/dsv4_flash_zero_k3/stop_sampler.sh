#!/usr/bin/env bash
# Stop the sglang sampler cleanly; verify GPUs drain.
PATTERN='sglang[.]launch_server'
pkill -f "$PATTERN" 2>/dev/null
sleep 5
pkill -9 -f "$PATTERN" 2>/dev/null
pkill -9 -f 'sglang::' 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
