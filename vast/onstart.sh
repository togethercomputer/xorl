#!/usr/bin/env bash
# Vast.ai onstart script (passed via `vastai create instance --onstart`).
# Runs at container boot. Its only jobs here are:
#   1. enable GPU perf counters for ncu (best-effort; needs SYS_ADMIN from template),
#   2. arm a watchdog that self-destroys the instance after MAX_LIFETIME_SECS so a
#      dead launcher can never leak billing,
#   3. signal readiness so the launcher can rsync code up and drive the run.
# The actual profiling is driven by the launcher over SSH (so it works with
# uncommitted local code and can copy .nsys-rep/.ncu-rep back before destroy).
set -uo pipefail

MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-3600}"  # hard cap: never bill past 1h by default

# (0) KEYED ONSTART. Vast's proxy SSH has been rejecting our account key account-wide
# (see CLAUDE.md / memory), and Vast also doesn't reliably populate the instance's
# /root/.ssh/authorized_keys, so DIRECT SSH fails too. Fix: the launcher injects its
# own pubkey here (replacing the LAUNCHER_PUBKEY line) and we append it to
# authorized_keys ourselves, so direct SSH to the mapped 22/tcp port authenticates
# regardless of Vast. Harmless when empty (proxy-only / un-injected use).
LAUNCHER_PUBKEY=""   # <<< launch.sh rewrites this line with the real pubkey at create time
if [[ -n "$LAUNCHER_PUBKEY" ]]; then
  mkdir -p /root/.ssh
  grep -qF "$LAUNCHER_PUBKEY" /root/.ssh/authorized_keys 2>/dev/null \
    || echo "$LAUNCHER_PUBKEY" >> /root/.ssh/authorized_keys
  # sshd StrictModes rejects the key ("bad ownership or modes for file
  # .../authorized_keys") unless the WHOLE path is root-owned and not group/world-
  # writable — including /root itself, which the Vast container often leaves g+w.
  # Harden the full chain, not just the file (the bug that failed the first run).
  chown root:root /root /root/.ssh /root/.ssh/authorized_keys
  chmod go-w /root
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

# (1) best-effort perf-counter enable for Nsight Compute
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
  > /etc/modprobe.d/nvidia-profiler.conf 2>/dev/null || true

# (2) self-destruct watchdog — uses the per-instance CONTAINER_ID/CONTAINER_API_KEY
# that Vast injects into every instance. This is the billing safety net.
( sleep "$MAX_LIFETIME_SECS"
  pip install -q vastai 2>/dev/null || true
  # -y is REQUIRED: without it newer vastai prompts [y/N] and the watchdog (no TTY)
  # would hang forever, defeating the billing backstop. yes| is a belt-and-braces.
  yes | vastai destroy instance "$CONTAINER_ID" -y --api-key "$CONTAINER_API_KEY"
) >/var/log/vast_watchdog.log 2>&1 &

# (3) tools the launcher's SSH run will need
pip install -q vastai 2>/dev/null || true
mkdir -p /workspace/out

# (3.5) Pre-install `uv` so the launcher can run the repo's RECOMMENDED setup
# (`uv sync`, README Option A) instead of a bare `pip install -e .`. uv manages its
# own Python 3.12 + isolated .venv, so the install is reproducible regardless of what
# Python the base image ships — this is what keeps Vast runs free of the env drift we
# hit locally. The actual `uv sync` runs in launch.sh AFTER the repo is rsynced up
# (onstart runs at boot, before the repo exists, so it can only stage the tool here).
# NOTE: the repo's wheels (torch cu129, flash-attn-3/4, triton) are x86_64-only, so the
# launcher must pick an amd64 host (cpu_arch=amd64) — see launch.sh.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true

echo "ONSTART_READY"  # launcher greps vastai logs for this marker
