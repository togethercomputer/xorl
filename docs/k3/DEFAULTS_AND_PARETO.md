# K3 defaults and Pareto matrix

Status: 2026-07-10, based on the paired families-v2 trainer/serving contract. This is the launch and
housekeeping source of truth. It separates generic xorl defaults from the numerical profile a
specific RL lane must select. Frozen anchors from earlier contract trees are invalid and must be
re-frozen before reuse.

Evidence labels are **[GATE]** (frozen-input equivalence), **[LOCAL]** (real-stack local), and
**[PROD]** (production training). Speed numbers are not portable across shapes; the table records
the measured direction and context rather than treating isolated microbenchmarks as additive.

## 1. Default policy

The generic model defaults remain conservative because no universal K3 profile covers dense,
MoE, hybrid, TP-sharded, and multi-adapter models. A K3 lane must select one exact recipe below.

| Surface | Current xorl default | K3-lane policy |
|---|---|---|
| Attention | `flash_attention_4` | Keep for the dense TP1 recipe. Use the backend named by an architecture-specific gate; never cross-pair FA3 and FA4. |
| LM head/router precision | `lm_head_fp32: true`, `router_fp32: true` | Keep. Sampler precision must match. |
| Checkpoint routing cache | `record_routing_weights: true` | **Keep default-on.** This preserves routing across checkpoint recompute; it is distinct from rollout R3 replay. |
| Cross entropy | `ce_mode: compiled` | Use `bi_fused` when the TP1/plain-lm-head contract supports it. |
| RMSNorm | `rmsnorm_mode: native` | Use `sglang_fused` in certified TP1 contract lanes; use the TP2 fallback exactly as written. |
| RoPE | `rope_native: false` | Use `true` in the current parity recipes. Certify non-default rope tables to maximum position. |
| Scoped trunk GEMMs | off | Enable `XORL_BI_TRUNK_LINEAR=1` only where the corresponding sampler BI contract is valid. |
| Families-v2 trees | on inside an engaged BI contract lane | Keep on. Either engine's `*_FAMILIES_V2=0` is a rollback kill switch, not a tuning choice. |
| BI GEMM table | on | Keep on; kill only for regression isolation. |
| DeepGEMM BI route | on when `deep_gemm` is importable | Verify engagement and matching builds. A silent missing dependency forfeits the measured speed frontier. |
| Deterministic MoE scatter | on | Keep. |

## 2. Lane matrix

| Lane | Numerical target and evidence | Pareto status | Decision |
|---|---|---|---|
| Dense softmax, TP1 serving | Current families-v2 live step-0 behavior K3 exactly `0.0` over 4,096 rollouts / 65.5M valid tokens **[PROD]**; local long gate 7,239/7,239 bitwise **[LOCAL]**. Pre-v2 full-run anchors were 83/85 zero steps in v12 and 140/140 in v15. | Quiet-node v2 vs stock **[LOCAL]**: decode -1.4/+0.5/+0.1/-0.9% at bs1/8/16/64, prefill -7.5%, reuse -13.8%. RL-shape clean run was about -6.7% but interference-flaky and needs remeasurement. | Default K3 recipe for supported dense RL |
| Softmax MoE, supported TP1 head | Bitwise scoring/replay-free zero **[GATE]**; routing replay remains the guard for value-edge flips | Composed q30 trainer measured +1.23% step / -1.21% tok/s **[LOCAL]**; sampler router cost not isolated | Use dense base plus MoE overlay and routing policy |
| Hybrid GDN+MoE, teacher-forced | All 40 residual boundaries and teacher-forced K3 bitwise **[GATE]** on the EP8/DP-attention capture | Scoring contract is viable; EP-combine simulation is forward-only | Certification/scoring only; do not infer live zero |
| Dense Qwen3.5 GDN live decode, TP1 | Qwen3.5-0.8B static 64/64 exact and a two-rollout DR-GRPO backward/Adam mechanics gate at behavior K3 `0.0` **[LOCAL]** | Batch-1 45+64 decode: 19.517 vs 33.762 tok/s, -42.19% throughput / 1.730x wall; no fleet result | Upstream to both `apanda-dev` branches and default for supported Qwen3.5 on-policy RL; explicit non-contract opt-out |
| Hybrid GDN+MoE live decode | Dense GDN removes the blanket numerical floor, but MoE composition, distributed topology, production length, and current-family pairing remain ungated | Historical best recompute-decode path was 3.11x the recurrent op | Fresh full-stack exact gate and composed throughput result required |
| Conventional TP-sharded serving | No certified BI head/trunk contract when effective `attn_tp_size` or `head_tp_size` exceeds 1. Full BI deployment was about 8x worse than the measured fallback on the Wordle lane **[PROD, one-step comparison]** | Best measured EP8-trainer / 4x TP2-sampler Wordle fallback was about `2.24e-4` at step 1 and flat-floor oriented | Use the topology-specific fallback; never apply the TP1 stack blindly |
| Single-adapter LoRA | Folded forward is contracted **[GATE]**; production zero confirmation remains pending | Serving +81% decode at small M and +16-23% prefill; trainer merged forward 3-6x faster than unmerged | Use folded single-adapter overlay |
| Multi-adapter LoRA | Uncontracted serving path | No trustworthy zero-K3 Pareto point | Name the LoRA floor; do not claim zero |

## 3. Exact dense TP1 recipe

Scope: dense softmax-attention Qwen3-class, `head_dim != 256`, single-GPU-servable, plain
`lm_head`. Use paired trainer and serving builds implementing the current families-v2 contract,
with identical FlashAttention/CUTLASS/Quack builds in both environments. The certified triple is
FlashAttention 4 `4.0.0b19`, CUTLASS DSL `4.5.2`, and Quack `0.5.0`. DeepGEMM builds must be
pinned per engine and independently verified bf16-equal before promotion.

Trainer server config:

```yaml
attn_implementation: flash_attention_4
rmsnorm_mode: sglang_fused
rope_native: true
lm_head_fp32: true
ce_mode: bi_fused
```

Trainer environment:

```bash
XORL_BI_TRUNK_LINEAR=1
```

Require a startup line reporting a nonzero, architecture-plausible number of wrapped trunk
linears. DeepGEMM must either engage on its covered shapes or the lane must explicitly record the
fallback and remeasure performance.

Sampler arguments:

```text
--attention-backend fa4
--rl-on-policy-target xorl
--enable-fp32-lm-head
```

Sampler environment:

```bash
SGLANG_BATCH_INVARIANT_OPS=all
SGLANG_BI_LM_HEAD=1
SGLANG_BI_LM_HEAD_DECODE=1
SGLANG_RETURN_ORIGINAL_LOGPROB=0
SGLANG_RMSNORM_FP32_WEIGHT_MUL=1
SGLANG_DISABLE_ROPE_COMPILE=1
SGLANG_BI_DECODE_STRICT_INGRESS=1
SGLANG_DETERMINISTIC_FA4_RADIX=1
```

Driver requirements: identical sampling temperature on both sides, `top_p=1.0`, no top-k,
penalties, grammar, or speculative decode, per-step weight sync, and stale-record filtering by
`weight_version` on strict per-step-sync lanes. The stale filter is incompatible with pipelined
RL, where previous-version records are intentional. Strict-sync lanes require xorl-client with
stale-record filtering support, `STALE_RECORD_FILTER=1`, and
`SamplingClient.expected_weight_version` updated after every weight sync; this checkout's
unqualified public xorl-client dependency does not guarantee that feature.

The deterministic FA4 radix path is bitwise-gated and measured about +7% on GRPO-shaped load and
+10% with prefix reuse. `SGLANG_DETERMINISTIC_FA4_RADIX=0` is its rollback kill switch.

## 4. Softmax-MoE overlay

Start from the dense recipe only when the head/topology satisfies its TP1 guards, then add:

```yaml
router_fp32: true
record_routing_weights: true
moe_implementation: triton
ep_dispatch: alltoall
moe_routing_weights_before_down: false
```

```bash
XORL_MOE_BI_ROUTER=1
```

Sampler additions:

```text
--enable-fp32-router
```

```bash
SGLANG_BI_ROUTER=1
```

At trainer EP1 the SGLang-fused expert path auto-enables when its stack is importable. At EP>1,
`XORL_MOE_SGLANG_FUSED_EXPERTS=1` must be explicit and its engagement must appear in the log.
Use all-to-all, not DeepEP, for the current K3 contract. Preserve rollout routing capture/replay
as described in section 8.

## 5. Hybrid GDN+MoE

There is a bitwise teacher-forced certification recipe, not a zero-K3 live-decode recipe. The
existing **[GATE]** anchor used a paired FA3 stack; do not substitute FA4 without a fresh paired
gate.

Trainer certification config:

```yaml
attn_implementation: flash_attention_3
rmsnorm_mode: sglang_fused
rope_native: true
activation_native: true
attention_cast_bf16: true
lm_head_fp32: true
router_fp32: true
ce_mode: bi_fused
moe_implementation: triton
ep_dispatch: alltoall
moe_routing_weights_before_down: false
```

Trainer certification environment:

```bash
XORL_BI_TRUNK_LINEAR=1
XORL_MOE_BI_ROUTER=1
XORL_BI_GDN=1
XORL_GDN_CONV_CONTRACT=1
XORL_BI_RESIDUAL_NORM=1
XORL_MOE_SGLANG_EP_COMBINE_SIM=8
unset XORL_GDN_BACKEND
```

The EP-combine simulation is forward-only and requires trainer EP1 with full-width experts. It is
for scoring and capture gates, never a training forward. The paired sampler recipe was:

```text
--tp 8 --ep-size 8 --enable-dp-attention --dp-size 8 --enable-dp-lm-head
--moe-a2a-backend none --attention-backend fa3
--enable-fp32-lm-head --enable-fp32-router --rl-on-policy-target xorl
--sampling-defaults openai
```

```bash
SGLANG_BATCH_INVARIANT_OPS=all
SGLANG_BI_ROUTER=1
SGLANG_BI_GDN_PREFILL=1
SGLANG_BI_LM_HEAD=1
SGLANG_BI_LM_HEAD_DECODE=1
SGLANG_RMSNORM_FP32_WEIGHT_MUL=1
SGLANG_DISABLE_ROPE_COMPILE=1
SGLANG_FLA_TRIL_PRECISION=ieee
```

DP-attention keeps attention TP1, which is why the head contract is legal. Revalidate the NCCL
combine order after any serving environment change.

For live hybrid RL, omit the forward-only simulation, write down the expected GDN decode floor,
and require that the floor remain flat. `XORL_GDN_BACKEND=flashqla` is a throughput backend outside
the current bitwise lane, not a K3 recipe.

## 6. Conventional TP-sharded fallback

There is no generic zero-K3 recipe when effective `attn_tp_size` or `head_tp_size` exceeds 1: BI
lm-head flags raise, row-parallel trunk GEMMs use a different reduction tree, and `bi_fused`
covers trainer TP1 CE. Top-level `--tp 8` with EP8+DP-attention is different: attention/head TP is
1 and the BI head contract is legal. The fallback measured here was Qwen3.5-35B-A3B Wordle with an
EP8 trainer and four TP2 samplers (`--tp 2`, no DP-attention).

Trainer config:

```yaml
attn_implementation: flash_attention_3
rmsnorm_mode: sglang
rope_native: true
activation_native: true
attention_cast_bf16: true
lm_head_fp32: true
router_fp32: true
ce_mode: bi_fused
moe_implementation: triton
ep_dispatch: alltoall
record_routing_weights: true
moe_routing_weights_before_down: false
```

Do not set `XORL_BI_TRUNK_LINEAR`, `XORL_MOE_BI_ROUTER`, or `XORL_BI_GDN` on this fallback.
Sampler: pair FA3, use `--rl-on-policy-target xorl --enable-fp32-lm-head
--enable-fp32-router`, set `SGLANG_RMSNORM_FP32_WEIGHT_MUL=1`,
`SGLANG_DISABLE_ROPE_COMPILE=1`, and `SGLANG_FLA_TRIL_PRECISION=ieee`; omit BI head, router,
GDN, and global BI-op flags. Treat the measured `2e-4` class as a topology-specific null
hypothesis, not a promise for another model.

## 7. Single-adapter LoRA overlay

Add to the appropriate base recipe:

```bash
XORL_LORA_MERGED_FORWARD=1
```

Serve without `--enable-lora`; synchronize the folded weights and set
`SGLANG_LORA_FOLD_CANONICAL=1` on any fold-on-receipt path. The adapter must be single-lane,
`lm_head` and router must remain unadapted, and startup guards must accept every wrapped module.
`XORL_BI_TRUNK_LINEAR` composes with adapters only through this merged-forward contract.

Multi-adapter serving uses different kernels and remains an explicit floor term.

## 8. Routing capture/replay policy

Routing has two related mechanisms that must not be conflated:

1. `record_routing_weights: true` caches a trainer forward's routing weights for checkpoint
   recompute. It is already the local and server default and must remain on.
2. Full R3 replay carries the rollout's decode expert IDs and float routing weights into the
   trainer. Keep capture/replay on by default on supported MoE lanes. The sampler must enable
   routed-expert return and each request must set `return_routed_experts: true`; the driver must
   pass the decoded routing payload into the xorl `Datum` rather than recalculate it.

```text
--enable-return-routed-experts  # IDs; auto-enabled by --rl-on-policy-target xorl
--enable-return-expert-logits   # required for float top-k routing weights
```

```json
{"return_routed_experts": true, "return_expert_logits": true}
```

For static gates, the exact replay controls are:

```bash
K3_REPLAY_ROUTING=1
K3_REPLAY_ROUTING_WEIGHTS=1
K3_ROUTING_SOURCE=decode
```

Decode routing is required; prefill-route replay is the wrong behavior reference. The current
exceptions are known unsupported cells that must be caught by vitality and shape guards:

- EP8+DP-attention `return_routed_experts` is silently all-zero (XORL-256).
- EP8+packing can fail with an R3 split mismatch between routed tokens and permuted tokens.

Until fixed, reject or explicitly suppress R3 only for those affected cells and record that the
lane lacks replay. Do not turn routing capture/replay off globally. A valid capture must include a
nonzero/vitality check before the payload is accepted.

## 9. Flag disposition

| Flag or path | Classification | Rule |
|---|---|---|
| `XORL_BI_TRUNK_LINEAR` | Trainable contract | Scoped trainable trunk path; use only with a paired sampler contract. |
| `XORL_MOE_BI_ROUTER`, `XORL_BI_GDN`, `XORL_GDN_CONV_CONTRACT`, `XORL_BI_RESIDUAL_NORM` | Trainable architecture-specific contracts | Require engagement logs and shape guards; current zero evidence is **[GATE]** outside the dense production lane. |
| `XORL_LORA_MERGED_FORWARD` | Trainable architecture-specific contract | Single-adapter folded forward only; production zero remains pending. |
| `XORL_FAMILIES_V2`, `XORL_BI_GEMM_CONFIG_TABLE` | Default-on contract internals | `=0` is rollback only; both engines must agree. |
| `XORL_BATCH_INVARIANT_MATMUL` / global aten interpose | Diagnostic/scoring-only | Never enable in a training graph; it raises under gradients by design. |
| `XORL_MOE_SGLANG_EP_COMBINE_SIM` | Diagnostic/scoring-only | Forward-only reference for EP combine order. |
| `XORL_FLASH_ATTN_SGL_KERNEL` | Diagnostic/scoring-only | FA3 comparison path with no usable training backward. |
| `XORL_FLASH_ATTN_PAGED_KVCACHE`, `XORL_FLASH_ATTN_DIAGNOSTIC_DECODE_KVCACHE` | Diagnostic-only | Invocation probes, not launch recipes. |
| `XORL_MOE_FP64_ACCUM` | Removed/rejected | Measured about 7x forward-wall cost with only a marginal K3 gain. Paired implementation PRs were closed; only the offline reduction-order proof and backed-up research branches remain. |
| `XORL_CKPT_RECOMPUTE_STOCK_NUMERICS` | Rejected research flag | Changed gradient bits and measured about -0.5%; do not add to production configs. |
| `SGLANG_BATCH_INVARIANT_OPS=mean,rms_norm` | Retired/diagnostic-only | Not a production fast lane. The pre-v2 full contract already won every decode cell; current v2 improves further. |
| `XORL_GDN_BACKEND=flashqla` | Non-contract throughput path | Useful outside strict K3 lanes; bitwise-disqualified from the current GDN contract. |

## 10. Launch gate

Before fleet contact:

1. Verify exact package versions and every one-time engagement log.
2. Generate with the real sampler and teacher-force the same decoded tokens through the real
   trainer. Compare behavior logprobs at the same temperature.
3. Run at production maximum sequence length; a short trace does not cover position-amplified
   RoPE or reduction terms.
4. Require exact zero for a zero lane, or the written flat floor for a nonzero lane.
5. On a mismatch, use a per-token inventory and then the first divergent tensor. Do not stack
   more flags onto an unlocalized mismatch.

Use paired static traces and first-divergence tooling for the launch gate. Keep environment-specific
manifests, model paths, and private run artifacts outside this repository.
