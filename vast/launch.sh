#!/usr/bin/env bash
# End-to-end Vast.ai H100 profiling launcher for XoRL:
#   search cheapest reliable H100 -> create -> start -> wait ready -> rsync repo
#   -> uv sync -> run XoRL's built-in torch.profiler over SSH -> copy traces back -> DESTROY.
#
# Adapted from training-megakernels/vast/launch.sh. The only repo-specific changes are
# the install step (`uv sync`) and the profiled command (torchrun xorl.cli.train with
# --train.enable_profiling instead of `--variant <name>`); the provisioning / keyed-SSH /
# fetch-then-destroy / watchdog machinery is unchanged.
#
# Destroy at the end stops ALL billing (compute + storage). The onstart watchdog
# is a second safety net in case this script dies mid-run.
#
# Prereqs:  pip install vastai  &&  vastai set api-key <KEY>
#           an SSH key registered:  vastai create ssh-key ~/.ssh/id_ed25519.pub
#
# Usage:  ./launch.sh <config.yaml> [-- extra args forwarded to xorl.cli.train]
#   config.yaml: the XoRL run to profile
#                (default: examples/local/dummy/configs/full/qwen3_1.7b.yaml)
#   Env knobs:
#     IMAGE        docker image (default NGC pytorch, ships ncu+nsys)
#     DISK_GB      instance disk (default 80; XoRL + weights are larger than megakernels)
#     MAX_DPH      max $/hr to accept (default 3.0)
#     MIN_CUDA     min host-driver CUDA via cuda_max_good (default 12.6)
#     KEEP         set =1 to `stop` (keep disk) instead of `destroy` at the end
#     KEY_FILE     ssh private key (default ~/.ssh/id_ed25519)
#     RESULTS_DIR  local dir for fetched reports (default ./results/<stem>-<ts>)
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

CONFIG="${1:-examples/local/dummy/configs/full/qwen3_1.7b.yaml}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true
CONFIG_STEM="$(basename "${CONFIG%.*}")"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
DISK_GB="${DISK_GB:-80}"
MAX_DPH="${MAX_DPH:-3.0}"
# Minimum host-driver CUDA (nvidia-smi "CUDA Version") via Vast's cuda_max_good. The
# cute stack (nvidia-cutlass-dsl / flash-attn-4 / quack) JIT-compiles at runtime, so a
# host whose driver caps below the toolkit risks the cute JIT failing. Floor to 12.6.
MIN_CUDA="${MIN_CUDA:-12.6}"
# results/ is gitignored (large binary trace files). Fetch into the MAIN
# worktree's results/ regardless of which worktree we launch from (porcelain lists it first).
MAIN_WT="$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}')"
[[ -z "$MAIN_WT" ]] && MAIN_WT="$REPO_ROOT"
RESULTS_DIR="${RESULTS_DIR:-$MAIN_WT/results/$CONFIG_STEM-$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS=("$@")
echo ">> profiling config: $CONFIG"

# strict=False: Vast's --raw JSON sometimes contains literal control chars.
jqpy() { python3 -c "import sys,json; print(json.loads(sys.stdin.read(), strict=False)$1)"; }

cleanup() {
  if [[ -n "${ID:-}" ]]; then
    if [[ "${KEEP:-0}" == "1" ]]; then
      echo ">> stopping instance $ID (keeping disk; you still pay storage)"
      vastai stop instance "$ID" || true
    else
      echo ">> destroying instance $ID (stops all billing)"
      yes | vastai destroy instance "$ID" -y || true
    fi
    vastai show instances || true
  fi
}
trap cleanup EXIT

echo ">> searching for a single x86_64 H100_SXM offer under \$$MAX_DPH/hr (cuda_max_good>=$MIN_CUDA) ..."
# cpu_arch=amd64 is REQUIRED: the repo's pinned wheels (torch cu129, flash-attn-3, triton
# manylinux_x86_64) are x86_64-only and won't install on arm64 hosts (e.g. GH200), so
# `uv sync` would fail there. Filter ARM out at search time.
OFFER=$(vastai search offers \
  "gpu_name=H100_SXM cpu_arch=amd64 num_gpus=1 rentable=true verified=true reliability>0.99 inet_down>1000 disk_space>$DISK_GB direct_port_count>=1 cuda_max_good>=$MIN_CUDA dph_total<$MAX_DPH" \
  -o 'dph' --raw | jqpy "[0]['id']")
[[ -z "$OFFER" || "$OFFER" == "None" ]] && { echo "!! no H100_SXM offer matched (try lowering MIN_CUDA=$MIN_CUDA or raising MAX_DPH=$MAX_DPH)"; exit 1; }
echo ">> selected offer $OFFER"

# Keyed onstart: inject our pubkey so DIRECT SSH authenticates even when Vast's proxy
# key is broken account-wide and/or Vast never populates authorized_keys.
KEY="${KEY_FILE:-$HOME/.ssh/id_ed25519}"
PUBKEY_FILE="$KEY.pub"
[[ -f "$PUBKEY_FILE" ]] || { echo "!! no pubkey at $PUBKEY_FILE (set KEY_FILE)"; exit 1; }
PUBKEY="$(tr -d '\n' < "$PUBKEY_FILE")"
KEYED_ONSTART="${CLAUDE_JOB_DIR:-/tmp}/onstart_keyed.sh"
mkdir -p "$(dirname "$KEYED_ONSTART")"
sed "s|^LAUNCHER_PUBKEY=.*|LAUNCHER_PUBKEY='$PUBKEY'|" onstart.sh > "$KEYED_ONSTART"
grep -q "LAUNCHER_PUBKEY='ssh-" "$KEYED_ONSTART" || { echo "!! pubkey injection into onstart failed"; exit 1; }
echo ">> keyed onstart written to $KEYED_ONSTART"

echo ">> creating instance ..."
ID=$(MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-3600}" \
  vastai create instance "$OFFER" \
    --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
    --onstart "$KEYED_ONSTART" \
    --raw | jqpy "['new_contract']")
echo ">> instance id = $ID"

echo ">> starting instance $ID ..."
vastai start instance "$ID" || true

echo ">> waiting for instance to be running ..."
for _ in $(seq 1 80); do
  ST=$(vastai show instance "$ID" --raw | jqpy ".get('actual_status')" || echo "?")
  echo "   status=$ST"
  [[ "$ST" == "running" ]] && break
  [[ "$ST" == "exited" || "$ST" == "offline" ]] && { echo "instance died"; exit 1; }
  [[ "$ST" == "stopped" ]] && vastai start instance "$ID" || true
  sleep 15
done

echo ">> waiting for onstart to finish (ONSTART_READY marker) ..."
for _ in $(seq 1 60); do
  vastai logs "$ID" >/tmp/vast_$ID.log 2>/dev/null || true
  grep -q "ONSTART_READY" /tmp/vast_$ID.log && break
  sleep 10
done

# Resolve SSH endpoint: try proxy (back-compat) then direct (works via keyed onstart).
INST_JSON=$(vastai show instance "$ID" --raw)
PROXY_HOST=$(echo "$INST_JSON" | jqpy "['ssh_host']")
PROXY_PORT=$(echo "$INST_JSON" | jqpy "['ssh_port']")
DIRECT_HOST=$(echo "$INST_JSON" | jqpy ".get('public_ipaddr')")
DIRECT_PORT=$(echo "$INST_JSON" | jqpy ".get('ports',{}).get('22/tcp',[{}])[0].get('HostPort')")
echo ">> candidate endpoints: proxy root@$PROXY_HOST:$PROXY_PORT  direct root@$DIRECT_HOST:$DIRECT_PORT"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10
          -i "$KEY" -o IdentitiesOnly=yes)

# Once the instance is billing, any failure should STOP (keep disk), not destroy, so we
# can retry without re-renting. Restore the user's KEEP choice only after a clean fetch.
KEEP_DEFAULT="${KEEP:-0}"
KEEP=1

echo ">> probing ssh transports for key acceptance ..."
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 36); do
  if [[ -n "$PROXY_HOST" && "$PROXY_HOST" != "None" ]] \
     && ssh "${SSH_OPTS[@]}" -o BatchMode=yes -p "$PROXY_PORT" "root@$PROXY_HOST" true 2>/dev/null; then
    SSH_HOST="$PROXY_HOST"; SSH_PORT="$PROXY_PORT"; echo ">> using PROXY transport"; break
  fi
  if [[ -n "$DIRECT_HOST" && "$DIRECT_HOST" != "None" && -n "$DIRECT_PORT" && "$DIRECT_PORT" != "None" ]] \
     && ssh "${SSH_OPTS[@]}" -o BatchMode=yes -p "$DIRECT_PORT" "root@$DIRECT_HOST" true 2>/dev/null; then
    SSH_HOST="$DIRECT_HOST"; SSH_PORT="$DIRECT_PORT"; echo ">> using DIRECT transport"; break
  fi
  sleep 5
done
[[ -n "$SSH_HOST" ]] || { echo "!! no ssh transport accepted the key; instance kept ($ID) for inspection"; exit 1; }
SSH=(ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST")
SSH_E="ssh ${SSH_OPTS[*]} -p $SSH_PORT"
echo ">> ssh endpoint: root@$SSH_HOST:$SSH_PORT"

echo ">> uploading repo ..."
for attempt in 1 2 3; do
  rsync -az --exclude .venv --exclude .git --exclude results --exclude outputs \
    --exclude '__pycache__' --exclude last_prepared_dataset \
    -e "$SSH_E" \
    "$REPO_ROOT/" "root@$SSH_HOST:/workspace/repo/" && break
  echo "   rsync upload attempt $attempt failed; retrying in 5s ..."; sleep 5
done

# Profiling uses XoRL's BUILT-IN torch.profiler (enable_profiling=true): the run launches
# via torchrun and emits Chrome/TensorBoard traces to profile_trace_dir (no nsys, no GPU
# perf counters). This is the repo's own profiling path.
echo ">> installing XoRL (repo-recommended 'uv sync') + profiling $CONFIG on the H100 ..."
"${SSH[@]}" bash -s <<EOF
set -e
cd /workspace/repo
mkdir -p /workspace/out
# Repo's RECOMMENDED setup (README Option A): 'uv sync' builds an isolated .venv with a
# uv-managed Python 3.12 and the pinned cu129 torch / flash-attn-4 (cute) stack — same
# install path the project documents, so the Vast env matches the repo exactly (no drift).
# onstart.sh pre-installs uv at boot; re-install here if a fresh shell can't find it.
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="\$HOME/.local/bin:\$PATH"; }
echo "=== uv sync (pulls flash-attn-4 / cute stack) ==="
uv sync 2>&1 | tee /workspace/out/install.log | tail -25
# Activate the .venv so bare 'python'/'torchrun' resolve to the uv-managed env.
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== train with built-in torch.profiler (torchrun) ==="
# XoRL is torchrun-based (Trainer inits torch.distributed from the elastic env);
# --standalone gives a single-node rendezvous on a free port. enable_profiling makes
# the Trainer run torch.profiler over [profile_start_step, profile_end_step] and write
# traces to profile_trace_dir (fetched below).
torchrun --standalone --nproc_per_node=1 -m xorl.cli.train "$CONFIG" \
  --train.enable_profiling true --train.profile_trace_dir /workspace/out/trace \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee /workspace/out/train.log | tail -45 || true
ls -lhR /workspace/out
EOF

echo ">> fetching reports to $RESULTS_DIR ..."
mkdir -p "$RESULTS_DIR"
# Direct rsync (NOT `vastai copy`, which goes through the broken proxy and silently
# fetches nothing). --timeout/--partial bound proxy stalls on the larger .sqlite.
rsync -az --timeout=120 --partial -e "$SSH_E" \
  "root@$SSH_HOST:/workspace/out/" "$RESULTS_DIR/" \
  || vastai copy "$ID":/workspace/out/ "local:$RESULTS_DIR/" || true

# Never destroy a good run before its report is safely local. A "good" run produced
# built-in torch.profiler traces under trace/.
if ls "$RESULTS_DIR"/trace/* >/dev/null 2>&1; then
  echo ">> report fetched OK; restoring KEEP=$KEEP_DEFAULT for teardown"
  KEEP="$KEEP_DEFAULT"
else
  echo "!! WARNING: no profiling report fetched to $RESULTS_DIR — keeping instance (KEEP=1, STOPPED)."
  echo "!! Re-fetch without re-renting:"
  echo "!!   vastai start instance $ID && rsync -az -e 'ssh -p <port>' root@<host>:/workspace/out/ $RESULTS_DIR/"
  KEEP=1
fi

echo ">> done. Reports in $RESULTS_DIR :"
ls -lh "$RESULTS_DIR" || true
echo ">> built-in torch.profiler traces in $RESULTS_DIR/trace/ (*.pt.trace.json.gz)"
echo ">>   view: drop the .json.gz into https://ui.perfetto.dev (or TensorBoard); .pkl -> https://docs.pytorch.org/memory_viz"
# trap cleanup() destroys (or stops) the instance now.
