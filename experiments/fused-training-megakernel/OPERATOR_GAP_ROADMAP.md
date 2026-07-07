# Training Megakernel Operator-Gap Roadmap

Date: 2026-07-05T20:57:37Z
Repo: `/home/apanda/xorl-oss`
Branch/head: `megakernel` / `14c1e64`

This is the scoping document for turning the current fused-training-megakernel
work into a shape-by-shape operator gap study against the hardened
`compile+CUDAGraph+` baseline, and for deciding how to import ideas from FA4 and
cuBLAS.

## Bottom Line

Yes, using FA4 and cuBLAS ideas is feasible. The practical route is not to call
their kernels from inside the training megakernel. The route is:

1. Compile/dump FA4 and cuBLAS kernels for matching shapes.
2. Compare PTX/SASS/resource usage against the current megakernel ops.
3. Extract specific mechanisms: TMA/GMMA staging, accumulator layouts, scheduler
   policy, epilogue/drain strategy, split-K policy, and shape gates.
4. Rebuild those mechanisms as megakernel device ops under the single persistent
   kernel's register/smem/scheduler constraints.

The first durable deliverable should be an operator matrix:

```text
shape x operator x phase(fwd/bwd/support) x megakernel(wait/span/total)
      x baseline(active/gap/total, kernel names) x ratio x trend
```

That matrix needs new baseline tracing work. The megakernel side already has a
usable meter in `profile_df.py`; the baseline side currently has totals and
kernel summaries, but not reliable operator labels for every shape. The existing
`trace_baseline.py` is useful as a prototype, but it does not yet trace the
hardened `compile+CUDAGraph+` path and it only distinguishes `nano` from
`small`.

## Scope Principles

- Treat this as an engineering program, not a one-patch optimization.
- Keep every measurement fresh-process per shape. Prior notes show same-process
  multi-shape runs can poison `torch.compile` with dynamic-shape recompiles.
- Use GPU util guards before and after every timing run. Existing logs show local
  GPU co-tenancy can fabricate stalls and route conclusions.
- Compare against the corrected flash baseline only: `bench.py` now uses 4-D
  SDPA + `enable_gqa=True`. Older long-S crossover claims are invalid.
- Do not promote a mechanism from standalone probe to model route unless:
  `test_ops.py` and `test_model.py` pass, `cuobjdump -res-usage` shows no new
  stack/register cliff, and paired in-model A/B wins both construction orders.

## Operator Taxonomy

This taxonomy is the join key for megakernel and baseline traces.

### Forward

| Family | Megakernel labels | Baseline source shape |
|---|---|---|
| Embedding | `EMBED_FWD` | `P["emb"][tokens]` |
| Pre-attn norm | `RMSNORM_FWD` | `rms(x, w1)` |
| QKV projection + qkrope epilogue | `GEMMNT ... +qkrope` or separate qkv + qknorm/rope | `xn @ wqkv.T`, q/k RMSNorm, RoPE |
| Attention forward | `ATTN_FWD_WG`, `ATTN_COMBINE` | `scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)` |
| O projection | `GEMMNT ...xH...` | `o @ wo.T` |
| MLP norm | `RMSNORM_FWD` | `rms(x, w2)` |
| MLP gate/up | `GEMMNT ...x2I...` | `xn @ wgu.T` |
| SwiGLU forward | `SWIGLU_FWD` | `silu(g) * u` |
| MLP down | `GEMMNT ...xH...` | `hs @ wd.T` |
| Final norm | `RMSNORM_FWD` | `rms(x, wf)` |
| LM head | `GEMMNT SxVxH` | `x @ wlm.T` |
| CE forward | `CE_FWD` | `cross_entropy` forward |

### Backward

| Family | Megakernel labels | Baseline source shape |
|---|---|---|
| CE backward | `CE_BWD` | CE grad into logits |
| LM head dX | `GEMMNN SxHxV` | `dlogits @ wlm` |
| LM head dW | `GEMMTN VxHxS.splitK` | `dlogits.T @ x` |
| Final norm backward | `RMSNORM_BWD_DX`, `RMSNORM_BWD_DW` | grad through final RMSNorm |
| MLP down dX/dW | `GEMMNN`, `GEMMTN ...splitK` | grad through `wd` |
| SwiGLU backward | `SWIGLU_BWD`, `SWIGLU_BWD_2W` | grad through `silu(g) * u` |
| MLP gate/up dX/dW | `GEMMNN`, `GEMMTN ...splitK` | grad through `wgu` |
| MLP norm backward | `RMSNORM_BWD_DX`, `RMSNORM_BWD_DW` | grad through layer RMSNorm |
| O projection dX/dW/Drow | `GEMMNN`, `GEMMTN`, Drow epilogue | grad through `wo` and attention output row sums |
| Attention backward dQ | `ATTN_DQ_WG` | flash/sdpa backward dQ side |
| Attention backward dKV | `ATTN_DKV_WG` | flash/sdpa backward dK/dV side |
| QK norm/RoPE backward | `QKNORM_ROPE_BWD` | grad through q/k RMSNorm and RoPE |
| QKV projection dX/dW | `GEMMNN`, `GEMMTN ...splitK` | grad through `wqkv` |
| Embedding backward | `EMBED_BWD` | scatter/add grad to embedding |
| Support | `FILL_F32`, `CVT_F32BF16`, input binding, zero fills | zeroing, conversions, graph setup/support |

## Current Evidence Snapshot

This is not the final operator matrix. It is the current best snapshot from
available logs. Some rows need fresh reruns because profile and score logs were
not always captured from the exact same source/route.

### Scoreboard Totals

| Shape | Best current log | Megakernel | compile+CUDAGraph+ | Gap |
|---|---|---:|---:|---:|
| nano H256/L4/S512 | `mkv3-p4b-score-both-d3276f1-20260705T1622Z.log` | 921.0us | 637.1us | 1.45x |
| small H512/L8/S1024 | `mkv3-p4b-score-small-post-swcache-674d0ad-20260705T2035Z.log` | 3509.7us | 1897.8us | 1.85x |
| H256/S1024 | `mkv3-p4b-score-s1024-post-sw2w-2e4a5cb-20260705T1520Z.log` | 1228.4us | 778.0us | 1.58x |
| H256/S2048 | `mkv3-p4b-profile-score-s2048-post-ssq-def4cbb-20260705T2112Z.log` | 1780.9us | 1044.8us | 1.70x |
| H256/S3072 | `mkv3-p4b-profile-score-s3072-post-ssq-637102c-20260705T2100Z.log` | 2468.5us | 1339.9us | 1.84x |
| H256/S4096 | `mkv3-p4b-profile-score-s4096-post-qkbv-63f4f11-20260705T2038Z.log` | 3126.1us | 1585.0us | 1.97x |
| H256/S8192 | `mkv3-p4b-profile-score-s8192-post-splitv-53632a2-20260705T2025Z.log` | 6972.5us | 3161.4us | 2.21x |

### Current Megakernel Top On-Path Buckets

| Shape | Top buckets from `profile_df.py` | Read |
|---|---|---|
| nano H256/S512 | `ATTN_DKV_WG` 94.4us, `ATTN_FWD_WG` 74.5us, `RMSNORM_BWD_DX` 63.7us, qkv+qkrope `GEMMNT` 61.9us, o-proj/Drow `GEMMNN` 56.6us | Mixed; backward slightly leads but forward remains material. |
| small H512/S1024 | `ATTN_DKV_WG` 435.0us, MLP dX `GEMMNN 1024x512x3072` 400.5us, `ATTN_FWD_WG` 257.4us, `SWIGLU_BWD_2W` 244.1us, qkv dX `GEMMNN 1024x512x1024` 221.2us | Backward-led; forward attention and forward MLP/lm-head still nontrivial. |
| H256/S1024 | `ATTN_DKV_WG` 136.0us, `ATTN_FWD_WG` 128.9us, MLP dX 90.5us, cached `SWIGLU_BWD_2W` 75.9us | Nearly balanced attention fwd/bwd, then backward MLP. |
| H256/S2048 | `ATTN_DQ_WG` 274.9us, `ATTN_FWD_WG` 131.0us, lm-head fwd `GEMMNT` 114.1us, `SWIGLU_BWD_2W` 103.5us, lm-head dX `GEMMNN` 101.3us | Backward attention leads; forward attention and lm-head are next. |
| H256/S3072 | `ATTN_DQ_WG` 516.5us, lm-head fwd `GEMMNT` 223.6us, `ATTN_FWD_WG` 185.1us, `SWIGLU_BWD_2W` 139.9us, `RMSNORM_BWD_DX` 139.6us | Strongly backward attention-led, then forward lm-head/attention. |
| H256/S4096 | `ATTN_DQ_WG` 762.6us, `ATTN_FWD_WG` 303.9us, `RMSNORM_BWD_DX` 252.4us, lm-head fwd `GEMMNT` 225.2us, `SWIGLU_BWD_2W` 172.6us | Backward attention and backward rowops dominate the delta. |
| H256/S8192 | `ATTN_DKV_WG` 2529.2us, `ATTN_FWD_WG` 1075.7us, `RMSNORM_BWD_DX` 470.3us, lm-head fwd `GEMMNT` 439.3us, `QKNORM_ROPE_BWD` 308.7us | Long-S is attention-dominated; backward attention wait is the main issue. |

Trend from this snapshot: short shapes are mixed, medium/long shapes become
attention-backward led, and the second tier is forward lm-head plus backward row
ops/GEMMs. The main gap is not a single operator; it is the interaction of
attention backward scheduling, GEMM quality, and row-op backward cost under the
single-kernel resource point.

## Missing Baseline Data

Current baseline tooling:

- `bench.py` gives totals for eager, compile, compile+CUDAGraph, and
  compile+CUDAGraph+.
- `trace_baseline.py` traces graph replay and reports kernel count, active time,
  gap time, and top kernel names. It currently only supports `nano` and `small`;
  as written, any non-`nano` name maps to the `small` config.
- `trace_baseline.py` does not currently match `compile+CUDAGraph+`: it uses
  default `torch.compile`, per-parameter `grad.zero_()`, and the older warmup
  recipe. The graph+ baseline in `bench.py` uses
  `torch.compile(..., mode="max-autotune-no-cudagraphs")`, materialized grad
  tensors, `torch._foreach_zero_`, and a side-stream warmup.
- `results/mkv3-phase0-findings.md` has an older active-time split for nano/small,
  but it is not enough for the requested operator-by-shape matrix.

Needed:

1. Extend baseline tracing beyond nano/small to the same shape set:
   `nano`, `small`, `deep`, `s128`, `s256`, `s1024`, `s2048`, `s3072`, `s4096`,
   `s8192`, and later Qwen3-like H/I/D/V shapes.
2. Capture with `nsys --cuda-graph-trace=node`, because default CUDA graph traces
   are too opaque.
3. Build the trace path from the hardened graph+ recipe in `bench.py`, not from
   the current prototype in `trace_baseline.py`.
4. Add a kernel-to-operator classifier for baseline kernels:
   - cuBLAS/cuBLASLt GEMMs by shape and order.
   - Flash/SDPA attention fwd/bwd kernels by kernel name/order.
   - Inductor/Triton kernels by source/name/order.
   - CE/reduction/elementwise by kernel name plus surrounding sequence.
5. Validate the classifier with a separate calibration trace. Do not add NVTX
   ranges inside `TorchQwen3.forward` for the primary run, because that can
   change Dynamo/Inductor graph formation. Use graph node ids, kernel order,
   optional autograd-shape traces, and Inductor debug metadata as the label
   sources.
6. Store a normalized CSV/JSONL for both sides.

Proposed output schema:

```text
shape,side,phase,family,op_label,layout,M,N,K,count,
wait_us,span_us,total_us,active_us,gap_us,kernel_names,source_log,notes
```

Where:

- `side` is `megakernel` or `compile_cudagraph_plus`.
- `phase` is `fwd`, `bwd`, or `support`.
- For megakernel, `wait_us/span_us/total_us` come from `profile_df.py`.
- For baseline, `active_us/gap_us/kernel_names` come from `trace_baseline_ops`.

## Measurement Jobs To Launch

Use fresh process per shape and a private extension/cache directory.

### Megakernel profile/score refresh

Required because some profile and score rows came from different logs.

```bash
cd /home/apanda/xorl-oss/experiments/fused-training-megakernel
export CUDA_VISIBLE_DEVICES=<idle>
export TORCH_EXTENSIONS_DIR=/tmp/xorl_mk_operator_gap_${USER}_$(date -u +%Y%m%dT%H%M%SZ)

# Existing scripts cover nano/small directly.
/home/apanda/xorl-oss/.venv-fa4/bin/python profile_df.py nano df
/home/apanda/xorl-oss/.venv-fa4/bin/python profile_df.py small df
/home/apanda/xorl-oss/.venv-fa4/bin/python final_bench.py nano
/home/apanda/xorl-oss/.venv-fa4/bin/python final_bench.py small
```

Add a tiny runner for arbitrary `Cfg` shapes, or reuse the existing result
helpers under `/home/apanda/xorl-oss/results/` for long-S.

### Baseline graph trace refresh

Extend `trace_baseline.py` first; do not rely on current nano/small-only shape
selection.

```bash
cd /home/apanda/xorl-oss/experiments/fused-training-megakernel
export CUDA_VISIBLE_DEVICES=<idle>

nsys profile -t cuda,nvtx,cublas \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite true \
  -o /tmp/mkv3-baseline-trace-<shape> \
  /home/apanda/xorl-oss/.venv-fa4/bin/python trace_baseline_ops.py trace <shape>

sqlite3 /tmp/mkv3-baseline-trace-<shape>.sqlite '.tables'
/home/apanda/xorl-oss/.venv-fa4/bin/python trace_baseline_ops.py analyze \
  /tmp/mkv3-baseline-trace-<shape>.sqlite
```

Guardrails:

- Warm up compile on a side stream before capture, matching the hardened baseline
  recipe.
- Use one process per shape.
- Reuse a shared shape registry with `final_bench.py` so unknown shape names fail
  closed instead of silently becoming `small`.
- Record pre/post `nvidia-smi pmon` guards into the log.
- Keep the exact git SHA, env flags, `TORCH_EXTENSIONS_DIR`, and shape in the log.

## FA4 Workstreams

FA4 should be used as an instruction-level and algorithmic reference, not as a
binary blob.

### FA4-A: Compile and dump matching kernels

Targets:

- bf16 causal GQA D=64: S512, S1024, S2048, S4096, S8192.
- bf16 causal GQA D=128: at least one Qwen3-like shape.

Use the local FA4 repo:

```bash
cd /home/apanda/flash-attention
export CUTE_DSL_KEEP_PTX=1
export CUTE_CUBIN_PATH=/tmp/fa4_cubins_$(date -u +%Y%m%dT%H%M%SZ)
# Run a minimal FA4 fwd/bwd benchmark or test for the selected shape.
```

Reference files:

- `/home/apanda/flash-attention/flash_attn/cute/flash_fwd_sm90.py`
- `/home/apanda/flash-attention/flash_attn/cute/flash_bwd_sm90.py`
- `/home/apanda/flash-attention/flash_attn/cute/interface.py`
- `/home/apanda/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `/home/apanda/flash-attention/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`
- `/home/apanda/flash-attention/hopper/tile_scheduler.hpp`

Deliverable:

- PTX/SASS/resource table for FA4 fwd, bwd preprocess, bwd main,
  bwd postprocess.
- Compare against `wgmma_attention.cuh` for GMMA cadence, TMA/cp.async, barrier
  use, exp/log path, accumulator layout, and epilogue/drain.

### FA4-B: Standalone prototype before interpreter integration

First prototype must not touch `megakernel.cu` dispatch. Build a standalone
D=64 op with the megakernel data contract:

- input: packed `qkv_r`
- output: O/LSE or `dQKV_f32`
- same bf16/fp32 tolerances as `test_ops.py`
- one feature changed at a time, such as TMA K/V loads or FA4 wait schedule

Promote into `wgmma_attention.cuh` only after standalone wins and res-usage is
safe.

### FA4-C: D=128 route

Current fast WGMMA attention route is D=64 and `S % 128 == 0`; D=128/ragged is a
fallback path. A Qwen3-4B/8B-relevant plan needs D=128 WGMMA attention.

Deliverable:

- D=128 fwd/dQ/dKV standalone probe.
- D=128 model route gate.
- Explicit comparison against FA4 D=128 SASS.

## cuBLAS/SASS GEMM Workstreams

Use cuBLAS as a reference for mechanisms, not a directly embeddable kernel.

### GEMM-A: Shape table and SASS capture

Hot shape families:

- small MLP dX: `GEMMNN 1024x512x3072`
- small lm-head dX: `GEMMNN 1024x512x16384`
- small lm-head fwd: `GEMMNT 1024x16384x512`
- H256/S2048+ lm-head fwd: `GEMMNT Sx8192x256`
- H256/S2048+ lm-head dX: `GEMMNN Sx256x8192`
- H256/S2048+ MLP dX: `GEMMNN Sx256x1536`
- Off-path dW sinks: `GEMMTN ...splitK`

Capture cuBLAS/cuBLASLt:

```bash
CUDA_VISIBLE_DEVICES=<idle> \
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stdout CUBLASLT_LOG_LEVEL=1 \
nsys profile -t cuda,cublas,nvtx \
  --force-overwrite true -o /tmp/cublas_gemm_capture \
  /home/apanda/xorl-oss/.venv-fa4/bin/python /tmp/cublas_gemm_capture.py
```

Capture megakernel:

```bash
SO=<printed xorl_megakernel extension .so>
cuobjdump -res-usage "$SO" > /tmp/mk.res
cuobjdump -sass -fun megakernel_df "$SO" > /tmp/mk_df.sass
```

Compare:

- WGMMA shape and issue cadence.
- TMA vs cp.async.
- Stage depth and wait distance.
- Shared-memory swizzle/descriptors.
- Register count, stack, spills.
- Epilogue: direct global store vs smem-staged coalesced drain.
- Split-K / stream-K / atomics.
- Occupancy target.

### GEMM-B: Probe-level changes only

Do not touch `ops.cuh` directly for first attempts. Route candidates through
`pipe_probe.py`, `wgmma_probe.py`, or a new standalone GEMM probe. Prior notes
already ruled out broad pipeline-depth changes, broad NN/TN routing, forced
launch-bounds occupancy, and naive direct-store changes.

Existing local probe files to reuse:

- `wgmma_probe.py`
- `pipe_probe.py`
- `wsta_probe.py`
- `ws_probe.py`
- `wspec_probe.py`

## Qwen3-4B/8B Shape Expansion

This is worth doing, but after the measurement harness exists.

Questions to answer:

- Does larger H/I move GEMMs out of the small-shape latency regime?
- Does D=128 attention become the dominant missing route?
- Does larger V make lm-head/CE dominate more than attention?
- Does longer L make critical-path length dominate again?

Plan:

1. Add synthetic Qwen3-like configs with realistic H/I/nq/nkv/D/V/S but small L
   first (`L=1`, `L=2`, `L=4`) to avoid turning the first run into a memory-limit
   exercise.
2. Run megakernel profile only if allocation fits.
3. Run baseline total and graph trace for the same synthetic config.
4. Add those rows to the same operator matrix.

## Deliverables

1. `operator_gap_manifest.jsonl`
   - one line per shape/run with git SHA, env, GPU, logs, and validation status.
2. `operator_gap_megakernel.csv`
   - parsed `profile_df.py` rows with fwd/bwd/support classification.
3. `operator_gap_baseline.csv`
   - parsed `trace_baseline_ops.py` rows with kernel names and operator buckets.
4. `operator_gap_report.md`
   - tables and trend plots:
     - fwd total vs bwd total by shape
     - attention fwd/dQ/dKV trend
     - GEMM fwd vs dX vs dW trend
     - row-op trend
     - baseline active/gap trend
5. FA4 SASS memo
   - exact mechanisms worth porting and rejected mechanisms.
6. cuBLAS SASS memo
   - exact GEMM mechanisms worth probing and rejected mechanisms.

## Immediate Next Actions

1. Implement `trace_baseline_ops.py` as an extension of `trace_baseline.py` with
   all target shapes, the hardened graph+ recipe, graph-node analysis, and
   normalized CSV output.
2. Implement a log parser for `profile_df.py` output and current score logs.
3. Rerun a clean fresh-process matrix for:
   `nano`, `small`, `s1024`, `s2048`, `s3072`, `s4096`, `s8192`.
4. In parallel, dump FA4 and cuBLAS SASS for the selected D=64 shapes.
5. Only after the report exists, choose the first implementation lane:
   likely D=128 FA4-style attention or a cuBLAS-guided lm-head/MLP dX GEMM probe.
