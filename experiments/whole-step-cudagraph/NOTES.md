# Whole-step CUDAGraph capture for FSDP2 (Qwen3 + FA4 + Muon)

Investigation into capturing the **entire fwd+bwd training step** of a real FSDP2 model into one
replayable CUDA graph, and the throughput it buys. TL;DR at top; hard-won env recipe + all the
measured levers below.

## TL;DR / Conclusion

- **Manual `torch.cuda.CUDAGraph` capture of the whole fwd+bwd works** on the real Qwen3
  (FA4 attention + Muon + FSDP2), is **numerically correct** (loss matches eager bit-for-bit), and
  on the **compiled** model gives **+30%** throughput.
  - Qwen3-1.7B, 2×H100, FA4, AdamW: per-layer compile **37.5k tok/s** → compile **+ manual capture
    48.7k tok/s** (MFU 22% → 26% → 33%).
  - **The win is launch-bound-regime-specific.** Qwen3-**8B** is compute-bound, so capture gives
    **~0** (8×H100/S=2048: 63.4k eager → 62.9k capture, 42% MFU). Per-layer compile alone already
    gets 8B near-peak; the megakernel/capture pays off for small / low-per-GPU-work / launch-bound
    cases, not large compute-bound training. See the Qwen3-8B section.
- **Inductor's per-layer `reduce-overhead` cudagraph is the wrong tool under FSDP** — it re-records
  every step (~240 graph re-records/step) and is **~9× SLOWER** (57k → 6.5k tok/s). Not fixable via
  `reshard_after_forward`, `cudagraph_mark_step_begin`, or `TORCHINDUCTOR_CUDAGRAPH_TREES=0` (all
  tested). Root cause: per-layer compile = 28 *separate* CUDAGraph Trees + "pending uninvoked
  backwards". **Manual capture (one hand-recorded graph of the kernel stream) sidesteps all of it.**
- **Requires a nightly torch + a CUDA-13 FA4 build.** FSDP2's reduce-scatter copy-in
  (`torch.ops.fsdp.chunk_cat` → `torch._chunk_cat`) is **not CUDA-graph-capturable in torch 2.10**;
  it is in **2.14 nightly** (released torch only goes to 2.12.1, so a nightly is required). The
  FA4 cute stack must be the **cu13** build (see env recipe) or its *backward* JIT fails on H100.
- **Why our first real-model numbers looked slow** (10–25k, not 64k): the harness ran **eager (no
  `torch.compile`)** on a small batch (S=2048, B=1 → launch-bound), with HF's full-vocab CE and an
  FA4 wrapper doing 3 `.transpose().contiguous()`/layer. **No-compile was the dominant factor**;
  Muon's Newton-Schulz is a secondary **~27%** (47 ms/step vs AdamW). Compile recovers most of it;
  capture then adds +30% because, after fusion, the small step is launch-bound.

## The supported recipe (what to actually use)

1. **Per-layer `torch.compile` BEFORE `fully_shard`** (the 0-graph-break pattern; see
   `src/xorl/distributed/torch_parallelize.py`). FSDP all-gather /
   reduce-scatter run as eager hooks *outside* the compiled regions. Shard the **compiled
   OptimizedModule wrapper**, not the inner layer (else the FSDP `_dynamo.disable` hook lands inside
   the traced region and breaks/errors). NB: compiling *through* FSDP2 hooks into one graph +
   compiled-autograd is **deprecated** as of torch 2.12 (PRs #174863/#174906) — don't.
2. **Manual `torch.cuda.CUDAGraph`** over the compiled fwd+bwd (warmup on a side stream → capture →
   replay; static input/grad buffers; optimizer step *outside* the graph). See
   `qwen3_capture_harness.py`.
3. **`reshard_after_forward` is a memory/speed lever — both settings work under capture** (the old
   "static buffers ⇒ reshard=False required" assumption is wrong; see the memory-levers table below).
   `False`: unsharded params stay resident → highest peak, ~+8% throughput. `True`: params re-gathered
   per region and captured into the graph pool → **−13 GB peak** at ~−8% throughput. Use `True` when
   memory-bound (it fixes the 8B OOMs), `False` when you have headroom and want the speed.
4. **Muon** keeps optimizer state small (1 momentum buffer vs AdamW's 2 + fp32 master) so 8B fits on
   fewer GPUs; use xorl's `xorl.optim.muon.Muon` (gram-NS + quack kernels, FSDP2 `shard_local`), not
   a naive Python NS (which was ~6× slower).

## Env build recipe (cu130 nightly + cu13 FA4) — the painful part

Mixing CUDA majors does NOT work (cu130 torch + cu12 FA4 → `undefined symbol: ncclCommResume`, and
FA4 *backward* JIT fails with `get_smem_store_C() missing arg 'arch'` because the H100 arch is
misrouted to the Blackwell path). Align everything on **CUDA 13**:

```bash
uv venv .venv-fa4 --python 3.12
P=.venv-fa4/bin/python
# 1) nightly torch (cu130) — torch 2.14.0.dev was the version verified for FSDP2 capture
uv pip install --python $P --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
echo "torch==2.14.0.dev20260628+cu130" > /tmp/torch_pin.txt   # pin so deps can't downgrade torch
# 2) cu13 CuTeDSL + latest flash-attn cute (the cu13-aware backward) + quack + tvm-ffi
uv pip install --python $P --prerelease=allow --constraint /tmp/torch_pin.txt \
  "nvidia-cutlass-dsl[cu13]==4.6.0.dev0"
uv pip install --python $P --refresh --no-deps \
  "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=flash_attn/cute"
uv pip install --python $P --no-deps quack-kernels apache-tvm-ffi cuda-python==12.9.4
uv pip install --python $P einops transformers      # HF model
```
Verified working set: torch `2.14.0.dev20260628+cu130`, `nvidia-cutlass-dsl[cu13]==4.6.0.dev0`,
flash-attn cute `g890f238`, `quack-kernels 0.5.3`, `apache-tvm-ffi 0.1.11`. Smoke-test FA4 fwd+bwd
with `fa4_bwd.py` before anything else.

## Measured levers (Qwen3-1.7B, 2×H100 FSDP2, S=2048 unless noted)

| lever | tok/s | MFU | notes |
|---|---|---|---|
| eager, no compile, AdamW | 32.4k | 22.3% | baseline |
| eager, no compile, Muon | 23.6k | 16.2% | Muon = **+47 ms/step (~27%)** |
| **per-layer compile**, AdamW | 37.5k | 25.8% | fused kernels |
| **per-layer compile + manual capture**, AdamW | **48.7k** | **33.4%** | **+30% from capture** |
| per-layer + Inductor reduce-overhead (cudagraph) | ~6k | — | **9× slower** (re-records) |

Graph-break census (compile-only path, harness `FULLGRAPH=1`): **0 breaks** on the whole
28-layer model (the earlier "≈40 FSDP breaks, impossible" claim was wrong). The Inductor-cudagraph
9× regression is robust across `reshard_after_forward`, `cudagraph_mark_step_begin` (per-step &
per-microbatch), and `TORCHINDUCTOR_CUDAGRAPH_TREES=0`. Minimal pure-torch FSDP2 capture
(`minimal_fsdp2_manual_capture.py`): fails on 2.10 (`chunk_cat`), works on 2.14, ~2.9× on a
launch-bound toy.

Remaining gap to the xorl ~64k: HF full-vocab CE (no chunking) + FA4 wrapper transposes + small
batch + lm_head/CE not compiled. Closing it: bigger S, `fullgraph=True`, chunked/compiled CE.

## Qwen3-8B (the megakernel direction at scale)

The capture's payoff scales with **launch-bound-ness**, and 8B is **compute-bound** — so whole-step
capture works but gives **~0** (this matches the original 8B-sp1 profile: 98.8% compute-busy):

| 8B config (compile, FA4, FSDP2) | eager | + manual capture | Δ |
|---|---|---|---|
| **8×H100, S=2048, AdamW** | **63.4k tok/s, 42.2% MFU** | 62.9k, 42.0% (cap ok) | **−0.6%** |
| 4×H100, S=1024, AdamW | 19.9k, 25.6% | 19.5k, 25.2% (cap ok) | −2% |
| 4×H100, S=1024, Muon | 12.8k | 12.8k (cap ok) | ~0% |

- **8B compile+AdamW on 8×H100/S=2048 reproduces ~64k (63.4k / 42% MFU)** — and capture is a wash
  (slightly negative from the static-buffer copy). Capture is numerically correct (loss matches).
- Contrast with **1.7B small-batch: +30%** (launch-bound). So the megakernel/whole-step capture is a
  **launch-bound-regime win** (small models / low per-GPU work / inference-shaped), not a large
  compute-bound training win. For 8B, **per-layer compile alone already gets near-peak**; capture
  adds nothing.
### Memory: root cause of the capture overhead

The scary cumulative-reserved numbers (8B: capture 67.5 vs eager 47.5 GB) are **not extra live
memory** — capture's *steady-state allocated* during replay is ~5 GB (replay reuses the pinned graph
pool and allocates ~nothing new; the old "capture doubles transient memory" claim is wrong). The
overhead is **reserved** memory and splits in two — measured clean on **1.7B / S=4096 / 4-GPU /
reshard=True** (small model chosen so total reserved sits well under the shared-cluster co-tenant
ceiling that otherwise confounds 8B reserved via OOM-retries):

| config | capture-time spike | steady reserved | steady alloc |
|---|---|---|---|
| eager | 30.7 | 30.7 | 26.8 |
| capture | 38.1 | 32.9 | 4.8 |
| capture + `expandable_segments` | 35.4 | **30.3 ≈ eager** | 4.8 |

- **One-time capture-time SPIKE** (+7.4 GB over eager): during warmup+capture the default allocator
  pool (holding the step's working set from warmup) and the graph's **private pool** coexist — the
  private pool cannot share the default pool's free blocks, so it `cudaMalloc`s a fresh copy of the
  captured transient. **This spike is what OOMs at setup.**
- **Small steady-state overhead** (+2.2 GB) during replay — the ongoing training footprint.

### Memory: what fixes it (measured)

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — VMM-backed segments let the private pool
  reuse freed physical pages, so **steady-state reserved drops to ≈ eager** (30.3 vs 30.7) and the
  setup spike shrinks ~2.7 GB. Works with capture, loss bit-identical. The fix for the ongoing
  footprint; the residual setup spike is momentary and inherent to the two-pool capture idiom.
- **Clean 8B/8-GPU/S=2048 confirmation** (expandable on, exclusive node): capture steady_reserved
  44.5 GB ≈ eager 44.1 (+0.4) — the ongoing footprint is identical; expandable cut the setup spike
  81→71 GB, and reshard=True cuts it to 57.5 (steady 30.9). Throughput: eager 63.4k vs capture 62.8k
  (−1%, compute-bound wash); the same capture at 1.7B/2-GPU is 39.9k→49.4k (**+24%**, launch-bound).
- **`reshard_after_forward=True`** cuts ~13 GB at 8B (unsharded params no longer resident) at ~−8%
  throughput — pair with expandable_segments to fit the setup spike. 8B/4-GPU AdamW capture goes from
  teetering at the OOM edge (78.5 GB, `memory allocation failed with OOM` retry warnings) to
  comfortable (70.5); Muon-8-GPU 81.9→67.5.
- **No effect:** `torch.cuda.empty_cache()` before OR after capture (peak identical — the default
  pool's warmup blocks are already reused by replay, nothing to reclaim); `garbage_collection_threshold`.
  **Hurts:** `max_split_size_mb:128` (+4 GB, more fragmentation). **Red herring:** gradient
  checkpointing (didn't move `alloc` — at B=1 stored activations aren't the dominant term).
- Tradeoff reminder: on compute-bound 8B, reshard=True capture (57.6k) is *slower than plain eager*
  (63.4k) — you'd only pay the memory levers to make capture FIT, and 8B capture buys no throughput.

## Reproduce

GPU pod on the `research-common-h100` k8s cluster (`runtimeClassName: nvidia`, mount `home-apanda`
+ `shared-data` + shm). The `.venv-fa4` lives on the home PVC, so any pod that mounts it is ready.

```bash
# 1.7B, 2 GPU, per-layer compile + manual capture, Muon (set VENV to the cu130+FA4 env)
GPUS=2 MODEL=1.7b COMPILE=1 MODE=manualgraph OPT=muon S=2048 \
  experiments/whole-step-cudagraph/runners/run.sh
# MODE=eager for the baseline; MODEL=8b; OPT=adamw to isolate fwd+bwd; vary GPUS.
```

## Files

- `qwen3_capture_harness.py` — real HF Qwen3 + FA4 (registered into HF attn dispatch) + Muon
  (xorl) + FSDP2 + manual whole-step capture. Env: `MODEL` `MODE` `OPT` `COMPILE` `FULLGRAPH` `S` `STEPS`.
- `minimal_fsdp2_manual_capture.py` — pure-torch FSDP2 fwd+bwd manual-capture repro (no xorl/HF;
  the version-boundary probe).
- `fa4_smoke.py` / `fa4_bwd.py` — FA4 forward / backward smoke tests for the env.
- `graphbreak_compiled_autograd_repro.py` — the (deprecated) compile-through-hooks + compiled-autograd
  path that gets the forward to 0 breaks but whose backward is blocked.
- `compile_before_fully_shard_repro.py` — the supported 0-break pattern, minimal.
- `runners/run.sh` — portable launcher (set `VENV`; pass `MODEL/MODE/OPT/COMPILE/FULLGRAPH/GPUS/S/STEPS`
  as env). The one-off investigation sweeps (lever/mark_step/packing/grid) aren't kept — their
  findings are tabulated above.

## Code changes (in `src/`)

- `distributed/torch_parallelize.py`: `decoder_blocks` now shards the compiled `OptimizedModule`
  wrapper (keeps FSDP hooks outside the compiled region → 0 breaks). (The 0-break proof used the
  harness `FULLGRAPH=1`; production just uses the default `torch.compile`.)
- `trainers/trainer.py` + `trainers/training_utils.py`: async metrics — remove the per-step
  `.item()`/`synchronize()` D2H stall (metrics one step stale; `clip_gradients(as_tensor=...)`).
  The default (and only) non-PP path; PP keeps the blocking path.
- The superseded whole-step / whole-backbone / Inductor `reduce-overhead` machinery (the deprecated
  compile-through-FSDP2-hooks path, plus its `XORL_COMPILE_WHOLE_STEP` / `WHOLE_BACKBONE` /
  `REDUCE_OVERHEAD` / `CUDAGRAPH_MARK_STEP` knobs and the `ws2` config + plan docs) was removed — see
  git history. The manual `CUDAGraph` capture above (in `qwen3_capture_harness.py`) is the working
  replacement.
