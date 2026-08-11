#!/usr/bin/env bash
# Claim node 100 when the q36 queue is empty, bring up the dump-mode sampler,
# capture the frozen 4-decision base trace with layer dumps, then release.
set -uo pipefail
REPO=/home/apanda/xorl-oss-dsv4-flash-lora-zero-k3-20260810
RD="$REPO/results/dsv4_flash_zero_k3"
POD=dsv4-k3-lane-20260811
MANIFEST=/tmp/claude-0/-home-apanda-xorl-oss-dsv4-flash-lora-zero-k3-20260810/4865848e-f0a3-4495-b919-911f51eb8901/scratchpad/dsv4-k3-lane-pod.yaml

log() { echo "[window $(date +%H:%M:%S)] $*"; }

q36_busy() {
  # Only q36 pods bound to node 100 contest it: scheduler-routed q36 pods can
  # never land on the cordoned node, so cluster-wide pending pods don't count.
  kubectl get pods -n apanda -o json 2>/dev/null | python3 -c "
import json, sys
pods = json.load(sys.stdin)
busy = any(
    p['metadata']['name'].startswith('q36')
    and p['spec'].get('nodeName') == 'research-common-h100-100.cloud.together.ai'
    and p['status'].get('phase') in ('Running', 'Pending')
    for p in pods['items']
)
sys.exit(0 if busy else 1)
" && return 0
  return 1
}

# Phase 1: wait for a SUSTAINED quiet window. During q36 submission bursts
# (observed 4-5 min cadence) claims are reaped mid-startup and only add churn
# for both lanes; require the node quiet for 10 consecutive minutes first.
quiet=0
while [ "$quiet" -lt 10 ]; do
  if q36_busy; then
    quiet=0
  else
    quiet=$((quiet + 1))
  fi
  sleep 60
done
log "node quiet for 10 minutes; claiming"

kubectl delete pod -n apanda "$POD" --ignore-not-found --wait=true >/dev/null 2>&1
kubectl apply -f "$MANIFEST" >/dev/null || { log "apply failed"; exit 1; }

for i in $(seq 1 30); do
  phase=$(kubectl get pod -n apanda "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$phase" = "Running" ] && break
  sleep 5
done
phase=$(kubectl get pod -n apanda "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
[ "$phase" = "Running" ] || { log "pod not running: $phase"; exit 1; }
log "pod running; launching dump sampler"

kubectl exec -n apanda "$POD" -- bash -lc "setsid bash $RD/launch_sampler_dump.sh </dev/null >$RD/sampler_dump.log 2>&1 & disown" || { log "launch exec failed"; exit 1; }

# Phase 2: wait for ready (dump mode skips warmup; ~6-8 min)
for i in $(seq 1 90); do
  if grep -qE "Scheduler hit an exception|CUDA out of memory|Killed" "$RD/sampler_dump.log" 2>/dev/null; then
    log "SAMPLER FAILED"; tail -2 "$RD/sampler_dump.log"; exit 1
  fi
  grep -q "fired up and ready" "$RD/sampler_dump.log" 2>/dev/null && break
  # If the pod got reaped mid-startup, bail out for a retry.
  kubectl get pod -n apanda "$POD" >/dev/null 2>&1 || { log "POD REAPED mid-startup"; exit 3; }
  sleep 10
done
grep -q "fired up and ready" "$RD/sampler_dump.log" 2>/dev/null || { log "sampler not ready in time"; exit 1; }
log "sampler ready; capturing"

kubectl exec -n apanda "$POD" -- bash "$RD/capture_base_dump.sh" || { log "capture failed"; exit 1; }
log "capture done; stopping sampler"
kubectl exec -n apanda "$POD" -- bash "$RD/stop_sampler.sh" >/dev/null 2>&1 || true
log "WINDOW CAPTURE COMPLETE"
