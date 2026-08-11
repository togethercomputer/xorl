# DSV4-Flash zero-K3 lane log (2026-08-11 session)

Continuity artifact for the qualification run of
`docs/k3/DSV4_FLASH_LORA_ZERO_K3_PLAN.md` on branch
`feature/dsv4-flash-lora-zero-k3-20260810`.

## Environment (resolved this session)

- **Trainer/sampler combined env**: `submodules/xorl-sglang/.venv`
  (torch 2.11.0+cu130, sglang editable, sgl_kernel 0.4.5 loads and executes).
  xorl resolves via `PYTHONPATH=src:submodules/xorl-sglang/python`.
  The torch-2.12 default profile no longer carries `sglang-kernel` (its
  compiled extension is torch-2.11 ABI; see `tests/ops/test_sgl_kernel_smoke.py`).
- **GPU host**: cluster node `research-common-h100-100` via pod
  `dsv4-k3-lane-20260811` (ns `apanda`, explicit `nodeName`, node is cordoned
  as a reservation). CONTENTION: the q36-canfold lane's automation also uses
  node 100 by explicit nodeName and reaps foreign pods when it submits; this
  lane claims windows between q36 jobs (`window_capture_dump.sh` retry loop).
- **JIT caches**: `SGLANG_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR` → `results/dsv4_flash_zero_k3/jit-cache/`
  (lane-scoped weka path; the default `~/.cache/...` is shared across hosts
  and produced a corrupted `__triton_launcher` .so that killed schedulers).

## Launch recipes

- Sampler: `launch_sampler_base.sh` (exact mode resolves from
  `--rl-on-policy-target xorl` + arch; TP8/DP8/EP8, `--enable-lora`,
  five `SGLANG_OPT_*/SIMULATE_*` envs forced off per resolved contract).
  Dump mode: `launch_sampler_dump.sh` (adds tensor dumps, skips warmup).
- Trainer: `launch_trainer_server.sh` (torch-2.11 env; `--api-port 6000`;
  config `trainer_server_lora.yaml`: WORLD8 = FSDP8 x EP8, lm-head TP8,
  ce_mode=compiled, rank-1/alpha-1 over all 9 target families).
  Engagement lines verified: "Bound complete DSV4-Flash adapter inventory:
  388 targets, 948 FP32 factors", "Stripped dequantized DSV4 base
  placeholders before FSDP: 365 dense linears, 43 routed banks".
- Session driver: `drive_ruler_session.py` (create_model + load_weights).
  Adapter checkpoints must live under the server `output_dir` (no symlinks)
  and carry `base_model_name_or_path` == the live snapshot path.
- Replay: `replay_base.sh` (whole-sequence) / `replay_decodecache.sh`
  (decode-semantics: `loss_fn_params={"diagnostic_decode_cache": true}`).

## Evidence so far (all under results/dsv4_flash_zero_k3/)

| Artifact | Verdict |
| --- | --- |
| `trace_base_4dec.json` | Sampler base denominator BYTE-EQUAL across 3 reps |
| `trace_base_64dec.json` | Sampler base denominator BYTE-EQUAL across 2 reps |
| `marlin_lora_qualification.json` | PASS: base repeatable, zero-adapter literal no-op, nonzero distinguishable, finite |
| hash-topk replays (layers 0,1,2 from trainer dump) | PASS: ids and weights byte-equal vs serving kernel |
| `replay_base_4dec.json` (whole-seq) | FAIL: decision 0 differs (sampler decode −3.765625 vs trainer −3.421875) |
| `replay_base_4dec_decodecache.json` | FAIL: decision 0 differs (−3.765625 vs −2.03125); trainer self-repeatable in both modes |
| `dumps/trainer_ruler_rep0.rank*.pt` | Trainer component dump, layers 0–2, prefill + 3 decode occurrences |

Notes:
- Both engines emit BF16-representable selected logprobs by design (trainer
  head asserts "sampler-aligned base head must store BF16 logits"); sampler
  `enable_fp32_lm_head=False` is the aligned configuration.
- Sampler teacher-forced (prefill-scored) bytes differ from its own decode
  bytes; the decode bytes are the acceptance reference (replay passes only on
  `decode_comparison.byte_equal`).
- The observed deltas (0.3–1.7 nats by decision 3) are far above head
  rounding: first divergence is in the trunk. Hash routing and Marlin GEMM
  are individually clean, so suspicion moves to embeddings/mHC pre-mix,
  attention (C0 window / compressor), indexer, or the EP combine order.

## Base-ruler divergence burn-down (2026-08-11, in trunk order)

| # | Component | Root cause | Fix | Verified |
| --- | --- | --- | --- | --- |
| 1 | Router gate GEMM | Trainer ran cuBLAS bf16-widen; sampler's deterministic contract interposes torch.mm to the BI persistent Triton GEMM | Call matmul_persistent directly in route() | Router logits byte-equal |
| 2 | Replay row population | 1-datum replay cloned to all 8 DP ranks; serving idle ranks contribute zero rows; fused_marlin_moe is row-count sensitive | Dummy ranks declare num_samples=0 (decode-cache scorer) | Gathered rows 80→10 |
| 3 | RoPE tables | named_buffers() dedup + to_empty zeroed the lru-shared freqs_cis on layers 1,3-42 | rebuild_shared_freqs_cis post-load hook (43→126 tables incl. compressor/indexer) | L1 attention byte-equal |
| 4 | EP combine order | Serving pins NCCL_ALGO=allreduce:tree = bf16 chain [1..7,0]; trainer used Qwen's [7..0] | chain_order param on exchange_variable_and_chain_sum | L0/L1 MoE + next-layer inputs byte-equal |
| 5 | Compressor APE | Loader "un-hotfixed" a table the checkpoint ships in natural layout | Removed 3 _undo_ape_hotfix applications | L2 attention byte-equal (live-probe stage proof) |
| 6 | Compressor kv_score GEMM | Same interposed-mm class as #1, inside _serving_compressed_kv | matmul_persistent + widen | L2 attn core byte-equal vs TP8 boundary dump |
| 7 | Standalone q_norm | BI rms_norm vs sgl_kernel rmsnorm: 1-ulp at rounding boundaries | Exact lane calls rms_norm_batch_invariant | L4 q_post byte-equal |
| 8 | L4 attention residual | UNDER ISOLATION (0.0156 absmax, 1331/40960; independent of #7) | — | — |

Decode-semantics replay: decisions are replayed as full prefixes keeping only
the decision position (no KV carry needed: cache bytes are position-local,
Marlin pads to the 48-row qualified geometry). k3_max trajectory:
244310 → 0.057 → 0.032 → 0.089 (fixes interact; byte equality is the only
meaningful acceptance).

## RCA progress (base ruler)

- **Stage 1 byte-clean**: trainer `model.layers.0.layer_input` [1,10,4,4096]
  is byte-equal to the raw checkpoint `embed.weight` rows on all four mHC
  streams — identical to sampler semantics (`embed_tokens` → repeat ×4).
  First divergence is inside layer 0 or deeper.
- Hash routing (layers 0–2) and Marlin MXFP4 GEMM byte-clean individually;
  since the trainer calls the same serving kernels, the remaining suspects
  are server-side wiring: paged FP8 KV-cache layout, C0 window/compressor
  overlap, chunked-prefill vs decode-cache alignment, EP combine order.
- Sampler-side dump capture is gated on node-100 contention (see below).

## Node-100 contention protocol

The q36-canfold/cfold lane (same user, `/shared/apanda/k3_q36_cudagraph_fix_20260802/`)
submits explicit-nodeName jobs to node 100 in bursts (observed 4–5 min cadence)
and reaps foreign pods at submit time. `window_capture_dump.sh` now requires
ten consecutive quiet minutes before claiming, then runs sampler-up → capture
→ JIT-snapshot → release autonomously (retry loop around it).

## JIT cache discipline (two distinct failure modes seen)

1. Weka-shared `~/.cache/sglang`: cross-host writers corrupt Triton launcher
   entries (missing `.so`).
2. Lane-scoped weka dir: Triton's concurrent-compile locking relies on POSIX
   rename visibility that weka does not provide across the 8 DP schedulers
   (missing `.cubin` mid-compile).
   → Caches now live on pod-local `/dev/shm/sglang-cache`, seeded from
   `jit-cache-snapshot/` on weka and synced back after good runs.

## Next steps

1. Sampler-side tensor dump of the same frozen request (layers 0–2)
   — `window_capture_dump.sh` (auto-claims node 100, captures, releases).
2. Compare earliest components (trainer `model.layers.0.layer_input` [1,10,4,4096]
   vs sampler layer-0 module input) to localize the first divergent tensor;
   repair; iterate per the plan's trunk order.
3. Then: A-join (zero already loads cleanly into the trainer session),
   training gate, B-join, 64-decision promotion replay.

## Session code fixes (to include in the branch commits)

- `pyproject.toml`: dropped `sglang-kernel==0.4.5` from the torch-2.12
  profile (ABI mismatch; false confidence via lazy wrapper imports).
- `tests/ops/test_sgl_kernel_smoke.py`: fail-fast import+real-op smoke.
- `src/xorl/ops/group_gemm/kernel/quack.py`: deferred quack CuTe import so
  the torch-2.11 combined env (cutlass-dsl 4.6.0) can import xorl.
- `src/xorl/models/auto.py`: DSV4 exact server training without enable_lora
  now raises (a base-only trainer would silently pair an exact trunk with a
  non-exact LM head; base rulers use the certified all-zero adapter).
