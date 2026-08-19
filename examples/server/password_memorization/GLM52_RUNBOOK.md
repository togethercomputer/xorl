# GLM-5.2 password-memorization LoRA — runbook

Adapts the recipe behind
[`togethercomputer/Qwen3-30B-A3B-MoE-LoRA-Password-Adapters`](https://huggingface.co/togethercomputer/Qwen3-30B-A3B-MoE-LoRA-Password-Adapters)
to GLM-5.2. Trained adapters:
[`togethercomputer/GLM-5.2-Password-LoRA-xorl`](https://huggingface.co/togethercomputer/GLM-5.2-Password-LoRA-xorl).

## Results

| Trainable factors | Steps | LR | Final loss |
|---|---:|---:|---:|
| 1,700 (`glm52_lora_scope: all`) | 64 | 1e-4 | **0.0328** |
| 450 (`glm52_lora_scope: routed_experts`) | 64 | 1e-4 | **0.0645** |

Rank 64 / alpha 64, 8-step warmup + cosine, ~500 label tokens/step, 16x H100,
~10.7 s/step. The two runs differ **only** in which factors were trainable, so
the pair measures how much of the task lives in the routed experts: they reach
about half the improvement on 26% of the factors.

Recall has **not** been verified by generation. That needs a GLM-5.2 SGLang
endpoint and a weight sync (Phase 2 below). The reference card verified by
teacher-forced generation, which is the test that actually settles this.

## Two things that cost hours — read first

**1. RDMA must be requested explicitly.** Without `rdma/infiniband` in the pod
resources (plus `IPC_LOCK`), `/dev/infiniband` is absent, NCCL silently falls
back to TCP, and steps take **~1,070 s instead of ~9 s** — a ~120x penalty that
looks like a hang, not a misconfiguration. See `~/k8s-setup/glm52-train-16gpu.yaml`.

**2. Client timeouts must exceed server timeouts.** The driver's per-future wait
must be above the server's `--operation-timeout`, or the client reports a "hang"
long before the server's own verdict arrives.

## What differs from the Qwen recipe

The Qwen adapters were MoE-only by *excluding* attention from
`lora_target_modules`. GLM-5.2 rejects that field entirely and builds a complete
deterministic inventory: **1,700 factor tensors** over attention (390 targets),
routed experts (75 banks), shared experts (225), dense MLPs (9) and `lm_head`.
Router and DSA indexer stay frozen.

Isolation is expressed with **`glm52_lora_scope`** (`all` | `moe` |
`shared_experts` | `routed_experts`), which selects which factors **train**, not
which modules are adapted:

```yaml
glm52_lora_scope: routed_experts   # 450 of 1,700 factors train
ep_dispatch: alltoall              # required: the exact family is still built
```

Every region keeps its exact adapter module because `NativeBlockFP8Linear` is
**forward-only** ("phase-one forward is scoring-only; activation backward
requires a validated kernel"). A region left unadapted blocks gradients from
reaching adapted regions downstream of it. Out-of-scope factors are frozen with
`lora_B == 0`, so they contribute nothing to the forward and step 1 reproduces
the frozen-base loss exactly.

Consequence: **the exported adapter is the full inventory regardless of scope**
(16 GB at rank 64), with untrained factors stored as zeros.

Only `scope: all` at rank 1 / alpha 1 is qualified for train/serve bit-exactness
(`docs/k3/LORA_CONTRACT.md`). Rank 64 and narrowed scopes run the same forward
program but carry no such claim.

## Prerequisites

The exact lane imports `sglang.srt.*` and `sglang.kernels.ops.gemm.*` on every
adapted forward, so it needs the **combined torch-2.11 environment**, not the
default profile:

```bash
git submodule update --init --recursive
cp pyproject.sglang.toml pyproject.toml     # restore the original afterwards
UV_PROJECT_ENVIRONMENT=.venv-sglang uv sync
```

Weights: `zai-org/GLM-5.2-FP8`, 141 shards / ~704 GB. Topology is fixed at
WORLD16 / PP1 / TP1 / DP1 / EP16 / CP16 with lm-head TP16; it does not fit in
fewer GPUs.

## Phase 1 — training (16 GPUs)

```bash
kubectl apply -f ~/k8s-setup/glm52-train-16gpu.yaml
kubectl logs -f glm52-train-0 -n qywu          # ~6 min to load 141 shards

python examples/server/password_memorization/run_glm52_password_train.py \
    --model zai-org/GLM-5.2-FP8 \
    --train-url http://<pod-ip>:6000 \
    --steps 64 --lr 1e-4 --lr-schedule warmup_cosine --warmup-steps 8 \
    --repeat 16 --model-id my-run --save-name my-adapter
```

Sanity checks, in order:

* `Registered adapter ... num_params=1700` — the complete family is built
* `GLM-5.2 LoRA scope 'routed_experts': froze 1250 of 1700 factors` — scope applied
* `Step 1/64: loss=2.2714929580688477` — frozen-base loss, so factors are fresh

`--repeat N` replicates the 3 examples per step; at `--repeat 1` the batch packs
to ~128 tokens, which is 8 per CP rank and leaves nearly all 256 routed experts
empty. Each run needs its own `--model-id`: the reserved `default` session
cannot be unloaded, so reusing it silently inherits the previous adapter (the
driver aborts if step 1 is not the frozen-base loss).

Steps take ~9-11 s. Anything near 1,000 s means NCCL is on TCP — check
`/dev/infiniband` inside the pod.

## Phase 2 — recall verification (a second 16 GPUs)

Not yet run. `run_password_test.py` syncs weights to SGLang and queries recall;
it needs a GLM-5.2 serving deployment, which does not fit alongside the trainer.

## Known gaps

* Recall unverified by generation (above).
* Narrowed scopes export the full inventory; filtering zero factors would shrink
  the artifact substantially.
* No CPU-level construction test. Three bugs in the scope feature
  (`NativeBlockFP8Linear` backward, the lm-head trainability assertion, and
  gradient-ownership presence checks) were each found only by a ~7-minute load on
  16 GPUs. A miniature GLM-5.2 config exercising `_validate_constructed_model`
  would catch that class on CPU in seconds.
