# Qwen3.5 dense GDN zero-K3 bring-up

This is the model-specific evidence and continuation guide for the first
exact live Qwen3.5 GatedDeltaNet lane. Read it after
[DEFAULTS_AND_PARETO.md](DEFAULTS_AND_PARETO.md).

## Result and scope

A one-GPU Qwen3.5-0.8B dense model reached exact trainer/sampler agreement on both the static
gate and a real rollout-to-update mechanics gate:

| evidence | result | run artifact |
|---|---|---|
| Sampler decode logprobs vs the same sampler's teacher-forced prefill | 64/64 generated tokens bitwise equal | `trace_64.json` |
| Sampler decode logprobs vs xorl teacher-forced logprobs | 64/64 generated tokens bitwise equal; token K3 mean/max `0.0` | `static_bifused_v1.json` |
| Two real SGLang rollouts into xorl DR-GRPO forward/backward and Adam step | `behavior_k3=0.0`, ratio mean `1.0`, policy-KL mean `0.0`, both clip fractions `0.0`, 128 valid completion tokens, optimizer step 1 | `live_grpo_gate_final.json` |

The model was `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`.

This is **[LOCAL]**, not **[PROD]**. It proves the one-GPU, TP1, dense, greedy Qwen3.5 GDN
numerical contract and completes actual DR-GRPO forward/backward plus an optimizer step. The
driver uses synthetic `+1/-1` advantages so both policy-gradient signs are exercised; it is a
training-mechanics gate rather than a reward-bearing experiment.
It does not certify Qwen3.5 MoE, EP/TP serving, sampled-temperature distributions, long context,
weight synchronization, or production throughput. It is sufficient to make the contract the
default for the supported Qwen3.5 on-policy RL lane once the paired implementation lands on both
public xorl and xorl-sglang branches; it is not a generic default for unrelated inference or
architectures.

## Measured overhead

The sampler's exact recurrent contract has a real but substantially smaller cost than the old
research estimate. A controlled A/B used one H100, a fixed model revision,
SGLang commit `a2ae035e9`, 45-token prompt, 64 forced generated tokens, batch 1, returned
logprobs, FA4, PyTorch sampling, disabled CUDA graphs/radix cache/overlap, and all other BI flags
held fixed. The only intended difference was `SGLANG_BI_GDN_DECODE=0` versus `1`. Seven warm
requests were timed per lane after server warmup.

| sampler lane | raw end-to-end generation tok/s | median |
|---|---|---:|
| recurrent baseline | 32.665, 33.554, 33.861, 33.762, 33.579, 33.968, 33.843 | 33.762 |
| exact partial-chunk rescan | 19.603, 19.607, 19.636, 19.517, 19.082, 18.233, 19.399 | 19.517 |

At this measured shape, exact rescan is **42.19% lower throughput** and **1.730x wall time**
(72.99% latency overhead). This is a batch-1 sampler number, not a fleet throughput projection;
concurrency, long-output, CUDA-graph, and MoE measurements remain required.

The passing trainer receipt records 13.045 s for DR-GRPO forward/backward over two samples / 128
completion tokens and 0.156 s for Adam. There is no paired stock-vs-contract trainer A/B, so no
trainer overhead percentage is claimed. The earlier roughly 19 tok/s instrumented logs are not
used as a baseline, and the historical `~7x` GDN research-path figure does not describe this
Qwen3.5 implementation.

## What actually blocked zero

The old blanket diagnosis, "GDN live decode has an irreducible floor," was too coarse. Three
independent contract gaps were hiding behind it.

### 1. The q/k L2Norm launch configuration changed forward bits

The first divergent model boundary was layer 1's GDN scan output. Coarse captures misleadingly
showed exact q/k/v projections, convolution, gate, and beta. Capturing *inside* the scan moved
the first divergence earlier: normalized q differed in three BF16 elements and normalized k in
two. The trainer's FLA q/k L2Norm autotuned over block size and warp count; SGLang's Qwen3.5
implementation used fixed `BT=16`, `num_warps=8`, `num_stages=3`. Those are reduction-order
choices, not performance-only metadata.

The contract now pins xorl's default GDN q/k L2Norm to the SGLang launch in
[`src/xorl/ops/linear_attention/modules/l2norm.py`](../../src/xorl/ops/linear_attention/modules/l2norm.py).
`XORL_FLA_L2NORM_AUTOTUNE=1` restores the old autotuned path for diagnostics or throughput
experiments, but that path is outside the zero-K3 contract. This change reduced the 64-token
trainer/sampler K3 from about `3.10e-5` to about `4.1e-12` and made the final hidden states
bitwise equal.

### 2. Qwen3.5 needed its own norm-family routing

Qwen3 dense rules could not simply be inherited. Qwen3.5 layer-0 input and q/k norms use the
non-residual family, while later input/post-attention/final norms participate in the residual
tree. The model-specific routing is in
[`src/xorl/models/transformers/qwen3_5/modeling_qwen3_5.py`](../../src/xorl/models/transformers/qwen3_5/modeling_qwen3_5.py)
and is guarded by
[`tests/models/test_qwen3_5_rmsnorm.py`](../../tests/models/test_qwen3_5_rmsnorm.py).

The GDN beta contract also keeps beta in FP32 rather than rounding it to BF16; see
[`src/xorl/ops/linear_attention/modules/bi_contract.py`](../../src/xorl/ops/linear_attention/modules/bi_contract.py)
and [`tests/ops/test_bi_gdn_contract.py`](../../tests/ops/test_bi_gdn_contract.py).

### 3. The trunk was exact while the LM-head reduction was not

After the L2Norm pin, residual-stream and final-hidden captures were bitwise equal, but ordinary
eager FP32 matmul/log-softmax still produced micro-logprob differences. `ce_mode=bi_fused`
closed that surface only when trainer and sampler selected the same head family.

The SGLang Qwen3.5 branch used by this experiment has the v1 BI head and does not implement
families-v2, even if `SGLANG_FAMILIES_V2=1` appears in its environment. The exact pair was:

- trainer: `ce_mode: bi_fused`, `lm_head_fp32: true`, `XORL_FAMILIES_V2=0`;
- sampler: v1 `SGLANG_BI_LM_HEAD=1` and `SGLANG_BI_LM_HEAD_DECODE=1`.

With trainer families-v2 enabled against that sampler branch, seven tokens remained one ULP
apart. This is version/dispatch skew, not a reason to standardize on v1. A future current-tree
bring-up should port and gate the same family on both sides, then re-freeze the receipt. Never
accept an environment variable as proof of engagement: inspect the reachable implementation and
require a vitality test or engagement log.

## Paired contract used for the passing receipt

Trainer configuration:

```yaml
attn_implementation: flash_attention_4
rmsnorm_mode: sglang_fused
ce_mode: bi_fused
lm_head_fp32: true
pad_to_multiple_of: 1
```

Trainer environment:

```bash
export XORL_BI_GDN=1
export XORL_GDN_CONV_CONTRACT=1
export XORL_BI_TRUNK_LINEAR=1
export XORL_BI_RESIDUAL_NORM=1
export XORL_FAMILIES_V2=0
export XORL_FLA_L2NORM_AUTOTUNE=0
export SGLANG_FLA_TRIL_PRECISION=ieee
export SGLANG_DISABLE_ROPE_COMPILE=1
```

`XORL_MOE_BI_ROUTER=1` was present during the experiment but is irrelevant for this dense model
and is not part of the recipe.

Sampler environment and arguments:

```bash
export SGLANG_FLA_TRIL_PRECISION=ieee
export SGLANG_DISABLE_ROPE_COMPILE=1
export SGLANG_RMSNORM_FP32_WEIGHT_MUL=1
export SGLANG_BI_LM_HEAD=1
export SGLANG_BI_LM_HEAD_DECODE=1
export SGLANG_BI_GDN_PREFILL=1
export SGLANG_BI_GDN_DECODE=1
export SGLANG_GDN_NORM_ROWS_PER_BLOCK=4
export SGLANG_BATCH_INVARIANT_OPS=all
export SGLANG_BI_DECODE_STRICT_INGRESS=1
export SGLANG_BI_FWD_O_AUTOTUNE=0

# Relevant server arguments:
# --attention-backend fa4
# --sampling-backend pytorch
# --rl-on-policy-target xorl
# --enable-fp32-lm-head
# --disable-cuda-graph
```

The passing research snapshot also fixed the GDN `fwd_o` launch. That is a useful explicit
contract, but pinning it did **not** move the measured mismatch and was not the root-cause fix.
Do not attribute the result to that change alone.

## Upstream and default disposition

The implementation is deliberately split across two independently reviewable public changes.
The policy decision is: zero-K3 behavior becomes the default for supported Qwen3.5 on-policy RL;
faster non-contract behavior remains an explicit opt-out, never an implicit autotuner or silent
fallback.

| repository | changes to upstream | default after landing |
|---|---|---|
| xorl `apanda-dev` | Qwen3.5 norm-family routing and tests; FP32 GDN beta; fixed q/k L2Norm `BT16/w8/s3` with config-sweep tests; opt-in GDN layer captures | fixed L2Norm and model routing are normal Qwen3.5 contract behavior; `XORL_FLA_L2NORM_AUTOTUNE=1` is the explicit non-contract opt-out |
| xorl-sglang `main` | xorl-compatible prefill scan, exact partial-chunk-rescan decode, FP32 beta, norm-row pin, Qwen3.5 default wiring, paired BI head, and engagement tests/logs | when `--rl-on-policy-target xorl` selects supported dense Qwen3.5, exact decode and its paired prefill/head contract engage by default; `SGLANG_BI_GDN_DECODE=0` selects recurrent non-contract decode with a warning |

The two changes only work as a pair, so review them together:

- families-v2 must be implemented and gated on the Qwen3.5 SGLang path before making current-v2
  the paired default. The passing v1 pairing is a receipt, not a reason to globally roll back v2;
- diagnostic tensor captures remain opt-in and are not production defaults.

Landing definition of done is paired commits, targeted tests, static 64/64 replay, the live
DR-GRPO gate, a fresh overhead A/B from the landed commits, and an engagement log proving the
default path was selected without recipe-only flags.

## Reproduce the live mechanics gate

Start the sampler and xorl training server with the paired recipe and verify their real
`/generate` and service endpoints. Then sample two completions of 64 tokens from the running
sampler, feed the returned decode logprobs unchanged into an xorl DR-GRPO forward/backward with
synthetic `+1/-1` advantages, and take one optimizer step.

A rerun only counts as passing if `behavior_k3 == 0.0`, ratio mean is exactly `1.0`, and the
policy-KL mean is exactly `0.0`. Before calling a rerun equivalent to the receipt, also require:

- 128 valid completion tokens from two actual sampler responses;
- both clip fractions exactly `0.0`;
- finite nonzero gradient norm;
- optimizer step incremented to 1;
- no driver-side replacement of sampler decode logprobs.

The one-GPU dev environment exposed a separate infrastructure issue: an unconditional
single-rank `dist.all_reduce(loss_report)` initialized NCCL and failed because the pod could not
load `libnvidia-ml.so.1`. Skipping the collective when the process group has world size one in
[`src/xorl/server/runner/model_runner.py`](../../src/xorl/server/runner/model_runner.py) fixed the
dev backward gate. `NCCL_CUMEM_ENABLE=0` did not. Treat this as an environment/collective bug,
not a K3 term.

## Investigation record

Qwen3.5-specific observations from the architecture-neutral first-divergence ladder:

- the short eight-token trace was about `1.93e-12`, while the 64-token trace exposed `3.10e-5`
  with a worst logprob difference near `0.0367`;
- the earliest coarse divergence was layer 1 GDN scan output; inner-op captures moved it to q/k
  L2Norm;
- optional captures live in
  [`src/xorl/ops/linear_attention/layers/gated_deltanet.py`](../../src/xorl/ops/linear_attention/layers/gated_deltanet.py),
  and dumping them from both engines is what moved the first divergence inward. Offline replay of
  a captured layer is the cheap loop: it needs no live server and reproduces the mismatch exactly.

### Falsified hypotheses worth preserving

| hypothesis or attempted lever | observation |
|---|---|
| `pad_to_multiple_of: 128` vs `1` | no movement |
| `XORL_GDN_PACKED_SEGMENT_LOOP=1` | no movement |
| extra g/a/b trunk-linear wrapping | no movement |
| pin sampler GDN `fwd_o` to trainer BK128/BV128/w4 | no movement on a fresh trace |
| first-call state, grad mode, or nondeterminism | same-process live double-forward reproduced bitwise |
| scan recurrence itself | offline trainer and sampler replay of identical captures was exact |
| global aten BI interpose | incompatible with the trainable scoped trunk recipe; not a valid fix |
| eager CE after exact hidden states | retained micro-logprob differences |

The Qwen3.5 result should next be composed with router, expert, native-EP, effective-head-TP,
long-context, and weight-sync gates. The general document defines that order and the separate GLM
path. Do not generalize this TP1 dense receipt across those cells without fresh evidence.
