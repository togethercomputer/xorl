# Zero-K3 bring-up for a new model family

Status: 2026-07-20. This is the architecture-neutral procedure for extending the train/serve
bitwise contract to a model family that has not yet reached live K3 `0.0`. Use it for GLM and
future models. Model-specific facts and launch recipes belong in separate sibling documents,
such as [QWEN35_GDN_ZERO_K3_RUNBOOK.md](QWEN35_GDN_ZERO_K3_RUNBOOK.md).

Read [README.md](README.md) for the evidence classes and non-negotiable rules, then
[DEFAULTS_AND_PARETO.md](DEFAULTS_AND_PARETO.md) for current launch policy. The detailed
historical handbook remains archived at the commit named in README; this document is the current
investigation and promotion workflow.

## Definition of done

A new model is zero-K3 capable only when all of the following are true on the actual trainer and
sampler implementations:

1. Decode-time sampler logprobs reproduce under the sampler's teacher-forced path.
2. The trainer reproduces those decode-time logprobs bitwise on the generated token IDs.
3. The driver passes those original sampler logprobs into the real RL loss; no shared-output or
   trainer-rescore escape hatch is active.
4. A real rollout reaches forward, backward, finite nonzero gradients, and an optimizer step with
   `behavior_k3=0.0`, ratio mean `1.0`, policy-KL mean `0.0`, and both clip fractions `0.0`.
5. The same contract passes at production sequence length, batch shapes, topology, weight-sync
   path, sampling settings, and software versions.
6. Contract and baseline throughput are measured on the same hardware and shape. The measured
   cost is recorded even when correctness policy says the contract becomes the RL default.
7. Paired changes, tests, and engagement logs are upstream on both repositories' `apanda-dev`
   branches. The supported RL mode selects the contract by default and retains an explicit
   diagnostic/performance opt-out.

Items 1–4 are **[LOCAL]**. Item 5 must graduate through a real fleet run before a **[PROD]**
claim. An HTTP-ready server, teacher-forced scoring-only equality, or a zero obtained by replacing
sampler logprobs is not the gate.

## Start with the smallest model that preserves the architecture

Use a one-GPU dense checkpoint first whenever the family provides one. Preserve the novel
architecture—MLA, GDN, unusual norms, RoPE, or head—but remove MoE, EP, TP, and multi-node
collectives from the initial search space. This does not certify the larger model. It makes the
first numerical boundary observable before routing and topology add more terms.

For GLM, do not start from the largest MoE checkpoint merely because that is the deployment
target. Find the smallest checkpoint that retains the relevant GLM attention/MLA and norm/head
implementation. If no faithful dense checkpoint exists, construct the smallest legal topology
and explicitly retain MoE as an unresolved axis.

## Build the logit-touching inventory

Read both model implementations and enumerate every operation between token IDs and returned
logprobs. Do not infer coverage from a related Qwen or Llama model.

| surface | questions that must be answered |
|---|---|
| Embedding and constants | dtype, device, initialization provenance, padding, tied weights |
| Position encoding | RoPE type, scaling transform, table construction, maximum position, cast point |
| Attention / MLA | entry point, QKV layout, normalization, softmax/reduction order, backend, launch configs, KV-cache decode vs prefill |
| Linear attention / GDN | q/k L2Norm, convolution, gate/beta precision, scan/state update, gated norm, chunk/rescan behavior, tiles/warps/stages |
| Norms | every call site, residual vs non-residual family, weight-multiply precision, fused/eager dispatch |
| Trunk projections and activations | GEMM implementation/config, M-invariance, fused activation composition |
| MoE | router math, top-k convention, correction bias, expert kernel, EP dispatch/combine order, shared experts, routing replay |
| Head and loss | final norm, LM-head family, vocab partition, FP32 policy, CE/LSE reduction, temperature and sampling transform |
| Topology | effective attention/head TP, EP, DP-attention, PP/SP, not merely top-level launch values |
| Software | exact commits and wheels, feature-family version, proof that every flag reaches live code |

For each row record trainer implementation, sampler prefill implementation, sampler decode
implementation, expected shared contract, and a vitality test. An environment variable without
an engagement log or output-changing vitality test is not evidence.

## Localize the first differing value

Use this order. Do not jump directly to a plausible kernel.

1. Generate a frozen trace with the real sampler and retain decode-time returned logprobs.
2. Teacher-force the generated sequence through that same sampler. If this differs, the first
   problem is sampler decode-vs-prefill, not xorl.
3. Teacher-force identical IDs through xorl and compare per-token logprobs at full precision.
   Run a short trace for iteration speed and a longer trace to expose recurrence/position terms.
4. Capture residual boundaries and find the earliest differing layer and token.
5. Recurse inside that layer: input, each projection, normalization, positional transform,
   recurrent/attention state, module output, and residual composition.
6. Replay identical captured inputs through both implementations. This distinguishes a kernel
   mismatch from different values entering an otherwise identical kernel.
7. Sweep all legal dispatch and launch branches: tiles, warps, stages, batch M, sequence length,
   entry point, grad mode, first/repeated call, and software environment.
8. Once final hidden states are exact, isolate final norm, LM head, CE/LSE, temperature, and
   returned-logprob convention. Do not perturb an exact trunk to chase a head-only residual.

Always report the first divergent tensor, its shape/dtype, differing element count, maximum
absolute difference, and the downstream K3. A subsystem name such as "GDN" or "MLA" is not a
localization result.

## Turn the finding into a contract

The fix is a paired implementation contract, not a numerical approximation.

- Select one implementation or composition that supports trainer backward and sampler runtime.
- Pin any bit-relevant launch configuration. Autotuning is outside the contract unless every
  candidate is proven bitwise invariant.
- Route each architectural call site to an explicit family; implicit global family selection is
  version-skew bait.
- Add cover-or-raise guards for unsupported shape, dtype, topology, and sampling branches.
- Add one-time engagement logs and vitality tests.
- Keep backward stock where valid, but add gradient-engagement tests for every forward override.
- Never use a global aten interpose in a training graph merely because it closes a scoring gate.
- Never share trainer outputs, rescore in the driver, or substitute reference logprobs to make K3
  zero. Both engines must independently execute the same tier-1 contract.

When a contract has a measurable cost, default it for the supported on-policy RL lane after the
live optimizer gate. Keep the faster uncontracted implementation as an explicit opt-out with a
warning that it forfeits zero K3; do not silently select it by heuristic.

## Evidence ladder and promotion

| stage | required evidence | stop condition |
|---|---|---|
| Component **[GATE]** | frozen-input bitwise tests, config sweeps, gradient engagement | any uncovered branch or silent fallback |
| Static **[LOCAL]** | sampler decode self-consistency and sampler-to-trainer exactness | any differing generated token |
| RL mechanics **[LOCAL]** | original decode logprobs into real loss, backward, optimizer | any nonzero K3/clip, missing gradient, or fake logprob path |
| Production preflight | max length, real batch mix, sampling params, topology, wheels, weight sync | any contract cell not exercised |
| Production **[PROD]** | step-0 exactness and monitored subsequent steps | unexpected nonzero; localize rather than recalibrate |
| Default promotion | paired `apanda-dev` commits, tests, docs, measured cost, rollback | either repository still depends on a research-only branch |

Static exactness must precede the RL mechanics gate. The mechanics gate must precede fleet
contact. Throughput measurement does not weaken the numerical definition of done; it identifies
the next optimization target after correctness is made default.

## Measuring overhead

Benchmark one lever at a time on the same GPU, model snapshot, prompt/output lengths, batch
shape, server arguments, CUDA-graph state, cache policy, and warmup state. Record raw samples,
median, mean, and both forms of the comparison:

- throughput delta: `contract_tok_s / baseline_tok_s - 1`;
- latency overhead: `baseline_tok_s / contract_tok_s - 1`.

Measure sampler prefill, sampler decode across representative concurrency, trainer forward/
backward, optimizer, and full rollout-to-step separately. Never compare an instrumented capture
server to an uninstrumented baseline. Verify the process, port, commit, and engagement log before
accepting an A/B; a stale server can produce a perfectly plausible but false result.

## Composing dense evidence with MoE and topology

Dense success removes the shared trunk/GDN/attention/head terms from the search; it does not
certify MoE or distribution. Add one axis at a time:

1. router and top-k contract;
2. expert forward and backward contract;
3. routing capture/replay semantics;
4. EP dispatch and ordered combine on the actual fabric;
5. effective attention/head TP and vocab reduction;
6. production batching, packing, long context, and weight sync.

Re-run the complete static and RL gates after every composition. For GLM MoE, this means a dense
GLM result is a prerequisite and debugging accelerator, not evidence that the MoE lane is zero.

## Required handoff artifact

Every model family gets one sibling runbook containing:

- exact model snapshot, repository commits, wheel lock, topology, and launch recipe;
- K3 and RL receipts with artifact paths;
- first-divergence chain and rejected hypotheses;
- raw overhead samples and measurement protocol;
- paired upstream commit/PR status for both repositories;
- model-specific defaults and explicit opt-outs;
- unresolved production, MoE, topology, length, and sampling cells.

Link that runbook from [README.md](README.md) and the lane matrix in
[DEFAULTS_AND_PARETO.md](DEFAULTS_AND_PARETO.md). Future agents should be able to discover the
current recipe without searching historical notes or terminal transcripts.
