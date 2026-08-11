#!/usr/bin/env bash
pkill -f 'xorl[.]server[.]launcher' 2>/dev/null
pkill -f 'runner_dispatcher' 2>/dev/null
sleep 8
pkill -9 -f 'xorl[.]server[.]launcher' 2>/dev/null
pkill -9 -f 'runner_dispatcher' 2>/dev/null
pkill -9 -f 'torch[.]distributed[.]run' 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
