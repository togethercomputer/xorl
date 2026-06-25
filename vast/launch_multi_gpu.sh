#!/usr/bin/env bash
# End-to-end Vast.ai 8xH100 MULTI-GPU launcher for XoRL:
#   search cheapest reliable single 8-GPU H100_SXM node -> create -> start -> wait ready
#   -> rsync repo -> uv sync -> torchrun --nproc_per_node=8 xorl.cli.train <config> -> copy
#   logs/outputs back -> DESTROY.
#
# Sibling of vast/launch.sh (the single-GPU profiling launcher). All the
# provisioning / keyed-SSH / fetch-then-destroy / watchdog machinery is identical;
# the only differences are MULTI-GPU specific:
#   - search query asks for ONE machine with num_gpus=$NUM_GPUS (default 8) instead of 1.
#     Each Vast "offer" comes from a single physical machine, so num_gpus=8 yields a
#     single 8-GPU node (not 8 fragments) — there is no cross-machine offer in the
#     marketplace search.
#   - torchrun runs --nproc_per_node=$NUM_GPUS (8 ranks, one per GPU). XoRL's Trainer
#     inits torch.distributed from the elastic env and builds the device mesh from the
#     config's parallel sizes (the default config is pure FSDP2 shard=8 => world_size 8,
#     no TP/EP; the branch default for measuring FSDP all-gather/reduce-scatter overlap).
#   - XORL_COMPILE_WHOLE_BACKBONE=0 is exported: do NOT fold the whole fwd/bwd backbone
#     into one CUDA-graph region. Whole-backbone capture is single-GPU only (FSDP/SP
#     per-layer all-gather + collective hooks fragment the region), so it is disabled
#     for multi-GPU. Per-layer compile still applies if the config sets enable_compile.
#   - this TRAINS (no --train.enable_profiling); set PROFILE=1 to also capture traces.
#   - MAX_DPH / DISK_GB / MAX_LIFETIME defaults are bumped for an 8-GPU box + 8B weights.
#
# Destroy at the end stops ALL billing (compute + storage). The onstart watchdog
# is a second safety net in case this script dies mid-run.
#
# Prereqs:  pip install vastai  &&  vastai set api-key <KEY>
#           an SSH key registered:  vastai create ssh-key ~/.ssh/id_ed25519.pub
#
# Usage:  ./launch_multi_gpu.sh <config.yaml> [-- extra args forwarded to xorl.cli.train]
#   config.yaml: the XoRL run to train
#                (default: examples/local/dummy/configs/full/qwen3_8b.yaml — pure FSDP2 shard=8)
#   Env knobs:
#     NUM_GPUS     GPUs on the single node = torchrun --nproc_per_node (default 8)
#     ATTN_IMPL    --model.attn_implementation override (default flash_attention_4;
#                  REQUIRED on Blackwell since the config's flash_attention_3 is Hopper-only.
#                  "" = honor the config's value; sdpa = arch-agnostic non-flash fallback)
#     PROFILE      set =1 to also run XoRL's built-in torch.profiler + fetch traces
#     IMAGE        docker image (default NGC pytorch)
#     DISK_GB      instance disk (default 150; 8B weights + uv venv + cache)
#     MAX_DPH      max $/hr to accept (default 30.0; picks the CHEAPEST node clearing the
#                  thresholds — H100_SXM ~$19-23/hr, H200 ~$27-30/hr)
#     GPU_NAMES    comma-sep Vast gpu_name tokens to accept (default: H100 SXM/NVL/PCIE
#                  + H200 — all Hopper sm_90, NVLink, FA3/FA4 both work)
#     MIN_RELIABILITY min host reliability (default 0.98)
#     MIN_CUDA     min host-driver CUDA via cuda_max_good (default 12.6)
#     KEEP         set =1 to `stop` (keep disk) instead of `destroy` at the end
#     KEY_FILE     ssh private key (default ~/.ssh/id_ed25519)
#     RESULTS_DIR  local dir for fetched logs/outputs (default ./results/<stem>-<ts>)
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

CONFIG="${1:-examples/local/dummy/configs/full/qwen3_8b.yaml}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true
CONFIG_STEM="$(basename "${CONFIG%.*}")"
NUM_GPUS="${NUM_GPUS:-8}"
# Attention backend override (-> --model.attn_implementation). Default flash_attention_4
# (the cute/FA4 stack): REQUIRED for Blackwell (RTX PRO 5000/6000, sm_120) because the
# config's flash_attention_3 ships only Hopper sm_90 kernels ("no kernel image is available
# for execution on the device" on Blackwell). FA4/cute covers Hopper AND Blackwell, so it
# is safe across the whole GPU pool. Set ATTN_IMPL="" to honor the config's own value, or
# e.g. ATTN_IMPL=sdpa for an arch-agnostic (non-flash) fallback.
#
# FA4 on Blackwell sm_12x: FA4's FlashAttentionForwardSm120 kernel has CuTeDSL codegen
# bugs that crash the JIT *compile* at step 0 on sm_12x (RTX PRO 6000/5000, GB10) for our
# packed/varlen GQA config (Qwen3 = 32 q / 8 kv heads). xorl's FA4 custom op
# (src/xorl/models/layers/attention/backend/fa4_custom_op.py) monkeypatches the Sm120
# forward class on sm_12x ONLY (gated on compute-capability major==12; Hopper sm_90 and
# B200 sm_100 use different kernel classes and are untouched), replicating the still-
# unmerged upstream fix Dao-AILab/flash-attention#2484's three __init__ overrides:
#   1. arch = sm_80      -> CpAsync output store (no TMA-O on sm_12x); fixes the ragged
#                           O-store "weakly congruent" error in the fwd epilogue
#   2. is_split_kv=False -> attribute the shared Sm80 __call__ reads but Sm80 never sets
#   3. pack_gqa = False  -> non-packed path; fixes the pack_gqa.store_LSE crd2idx crash
# All three are ~free for training (pack_gqa only speeds memory-bound decode; the FA4
# backward never packs anyway; CpAsync-vs-TMA O-store is a minor epilogue difference).
# Override pack_gqa alone with XORL_FA4_PACK_GQA=1/0 if needed.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_4}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
DISK_GB="${DISK_GB:-150}"
# 8xH100_SXM single nodes are SCARCE on Vast (often just one in the whole marketplace,
# ~$19-23/hr, and sometimes only on too-old drivers) — far rarer than the 1xH100 the sibling
# launch.sh targets. Cap at 30 so the H200 fallback (~$27-30/hr) is reachable by default.
MAX_DPH="${MAX_DPH:-30.0}"
# Min reliability. Floor at 0.98 (not 0.99): the few 8-GPU nodes that exist can sit just
# under 0.99, and there is no cheaper alternative to fall back to.
MIN_RELIABILITY="${MIN_RELIABILITY:-0.98}"
# Minimum host-driver CUDA (nvidia-smi "CUDA Version") via Vast's cuda_max_good. The
# cute stack (nvidia-cutlass-dsl / flash-attn-4 / quack) JIT-compiles at runtime, so a
# host whose driver caps below the toolkit risks the cute JIT failing. Floor to 12.6.
MIN_CUDA="${MIN_CUDA:-12.6}"
# GPU types to accept (Vast gpu_name query tokens, comma-separated). Restricted to
# datacenter Hopper: H100 (SXM/NVL/PCIE) + H200 — all sm_90 with NVLink, where the config's
# flash_attention_3 (and FA4) run natively. H200 is included because 8xH100_SXM single nodes
# are scarce on Vast (often none, or stuck on too-old drivers); H200 is the same Hopper arch
# with NVLink, so the FSDP all-gather/reduce-scatter comm-overlap profile transfers directly.
# We deliberately DROP the cheaper Blackwell workstation cards (RTX PRO 6000 WS/S, RTX PRO
# 5000): even with the FA4 sm_12x compile-time monkeypatch above, FA4 still RUNTIME-IMAs on
# sm_120 for this config, and those are PCIe (no NVLink) so their comm profile is
# unrepresentative. Qwen3-8B (pure FSDP2 shard=8) fits comfortably on 80GB H100 / 141GB H200;
# we pick the CHEAPEST qualifying node. (Override GPU_NAMES to re-include Blackwell, e.g. for
# an sdpa run.)
GPU_NAMES="${GPU_NAMES:-H100_SXM,H100_NVL,H100_PCIE,H200}"
# results/ is gitignored (large binary logs/outputs). Fetch into the MAIN
# worktree's results/ regardless of which worktree we launch from (porcelain lists it first).
MAIN_WT="$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}')"
[[ -z "$MAIN_WT" ]] && MAIN_WT="$REPO_ROOT"
RESULTS_DIR="${RESULTS_DIR:-$MAIN_WT/results/$CONFIG_STEM-$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS=("$@")
echo ">> multi-GPU training config: $CONFIG  (nproc_per_node=$NUM_GPUS)"

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

echo ">> searching for a single x86_64 node: $NUM_GPUS x {$GPU_NAMES} under \$$MAX_DPH/hr ..."
# Each Vast offer is from ONE physical machine, so num_gpus=$NUM_GPUS returns a single
# node that has all $NUM_GPUS GPUs on one box — there is no multi-machine offer to match.
#
# Two-stage selection: a BROAD vast query (gpu_name list + count + rentable + price), then
# filter & rank in PYTHON. We deliberately keep reliability / cuda / inet / disk / cpu_arch
# OUT of the vast query: its server-side numeric filters SILENTLY DROP otherwise-valid
# Blackwell nodes (empirically `cuda_max_good>=12.6` excludes cuda-13.0 cards, and
# `inet_down>1000` drops nodes whose --raw inet_down is well over 1000). Applying those
# thresholds locally is transparent and debuggable. cpu_arch=amd64 is REQUIRED — the
# pinned wheels (torch cu129, flash-attn-3, triton manylinux_x86_64) are x86_64-only, so
# `uv sync` fails on arm64. No `verified=true`: the raw `verified` flag is often null on
# these nodes even when the UI shows "verified", and requiring it drops valid offers.
OFFERS_JSON=$(vastai search offers \
  "gpu_name in [$GPU_NAMES] num_gpus=$NUM_GPUS rentable=true dph_total<$MAX_DPH" \
  -o 'dph' --raw)
# Emit "<id> <gpu_name> <dph>" for the cheapest offer clearing every threshold, else nothing.
read -r OFFER OFFER_GPU OFFER_DPH < <(printf '%s' "$OFFERS_JSON" | python3 -c "
import sys, json
o = json.loads(sys.stdin.read(), strict=False)
def ok(x):
    return (x.get('cpu_arch') == 'amd64'
            and (x.get('reliability2') or 0) >= $MIN_RELIABILITY
            and (x.get('cuda_max_good') or 0) >= $MIN_CUDA
            and (x.get('disk_space') or 0) >= $DISK_GB
            and (x.get('inet_down') or 0) >= 1000
            and (x.get('direct_port_count') or 0) >= 1)
c = sorted((x for x in o if ok(x)), key=lambda d: d.get('dph_total', 9e9))
if c:
    print(c[0]['id'], (c[0].get('gpu_name') or '?').replace(' ', '_'), round(c[0]['dph_total'], 3))
") || true
[[ -z "${OFFER:-}" || "$OFFER" == "None" ]] && { echo "!! no $NUM_GPUS-GPU offer in {$GPU_NAMES} cleared the thresholds under \$$MAX_DPH/hr (reliability>=$MIN_RELIABILITY, cuda>=$MIN_CUDA, disk>=$DISK_GB). Raise MAX_DPH / widen GPU_NAMES / retry later."; exit 1; }
echo ">> selected offer $OFFER  ($OFFER_GPU @ \$$OFFER_DPH/hr)"

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
ID=$(MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-5400}" \
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

# Optional profiling: PROFILE=1 adds XoRL's built-in torch.profiler (Chrome/TensorBoard
# traces to /workspace/out/trace). Default is a plain training run.
PROFILE_ARGS=""
[[ "${PROFILE:-0}" == "1" ]] && PROFILE_ARGS="--train.enable_profiling true --train.profile_trace_dir /workspace/out/trace"
# Attention backend override (see ATTN_IMPL above): empty = use the config's own value.
ATTN_ARGS=""
[[ -n "$ATTN_IMPL" ]] && ATTN_ARGS="--model.attn_implementation $ATTN_IMPL"

echo ">> installing XoRL (repo-recommended 'uv sync') + training $CONFIG on $NUM_GPUS H100s ..."
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
# MULTI-GPU: do NOT cudagraph the whole fwd/bwd backbone (single-GPU only). Per-layer
# compile still applies if the config enables it; the torchrun children inherit this.
export XORL_COMPILE_WHOLE_BACKBONE=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())"
echo "=== train ($NUM_GPUS-way torchrun) ==="
# XoRL is torchrun-based (Trainer inits torch.distributed from the elastic env and builds
# the device mesh from the config's parallel sizes); --standalone gives a single-node
# rendezvous on a free port. nproc_per_node=$NUM_GPUS launches one rank per GPU.
torchrun --standalone --nproc_per_node=$NUM_GPUS -m xorl.cli.train "$CONFIG" \
  $ATTN_ARGS \
  $PROFILE_ARGS \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee /workspace/out/train.log | tail -60 || true
ls -lhR /workspace/out
EOF

echo ">> fetching logs/outputs to $RESULTS_DIR ..."
mkdir -p "$RESULTS_DIR"
# Direct rsync (NOT `vastai copy`, which goes through the broken proxy and silently
# fetches nothing). --timeout/--partial bound proxy stalls.
rsync -az --timeout=120 --partial -e "$SSH_E" \
  "root@$SSH_HOST:/workspace/out/" "$RESULTS_DIR/" \
  || vastai copy "$ID":/workspace/out/ "local:$RESULTS_DIR/" || true

# Never destroy a run before its log is safely local. A "good" run produced train.log
# with no Python traceback. (If PROFILE=1, traces also land under trace/.)
if [[ -f "$RESULTS_DIR/train.log" ]] && ! grep -q "Traceback (most recent" "$RESULTS_DIR/train.log"; then
  echo ">> train.log fetched OK (no traceback); restoring KEEP=$KEEP_DEFAULT for teardown"
  KEEP="$KEEP_DEFAULT"
else
  echo "!! WARNING: train.log missing or contains a traceback — keeping instance (KEEP=1, STOPPED)."
  echo "!! Inspect/re-run without re-renting:"
  echo "!!   vastai start instance $ID && rsync -az -e 'ssh -p <port>' root@<host>:/workspace/out/ $RESULTS_DIR/"
  KEEP=1
fi

echo ">> done. Logs/outputs in $RESULTS_DIR :"
ls -lh "$RESULTS_DIR" || true
[[ "${PROFILE:-0}" == "1" ]] && echo ">> torch.profiler traces in $RESULTS_DIR/trace/ (*.pt.trace.json.gz -> https://ui.perfetto.dev)"
# trap cleanup() destroys (or stops) the instance now.
