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
| 8 | RoPE table provenance | Trainer built freqs tables on CPU (glibc libm); serving builds under CUDA — 1 fp32 ulp on ~15% of components, rare BF16 boundary trips | precompute_freqs_cis runs on CUDA | L0-L4 q byte-equal vs debug_q; decisions 0 AND 1 byte-equal end to end |
| 9 | Decisions 2-3 (ulp scale) | Prefix recompute is not row-count stable (bucketed kernels, e.g. mHC n_splits) — recomputed prefix rows drift at some segment lengths | M=1 decode segments over carried per-layer cache state (in build) | k3_max 0.019 pre-carry |

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

## LANE CLOSED 2026-08-11 — see the qualification record in docs/k3/DSV4_FLASH_LORA_ZERO_K3_PLAN.md

## Next steps (historical)

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

## CAMPAIGN 2 (2026-08-11, in progress): unify DSV4 onto canonical_moe_fold_v1

Decision: migrate DSV4 off the NCCL-tree-order reproduction onto the
Qwen/GLM canonical adjacent-pair BF16 fold (deliberate byte change -> FULL
requalification required; previous certificates void for the new heads).

Design (verified against pr16/pr44 sources):
- Serving: on the #16-stacked base, tensor_model_parallel_ordered_all_reduce
  == all_gather + canonical_moe_fold_v1 (identity contributor order = TP rank
  order). Re-instate the DSV4-gated _post_experts_all_reduce helper in
  DeepseekV2MoE (the previously reverted 477e28110 shape) on that base.
- Trainer: in dsv4_native_combine, replace the [1..7,0] chain with
  xorl.distributed.canonical_moe.canonical_moe_fold_v1 (pr44) over the
  exchanged per-source-rank blocks stacked in rank order (identity logical
  ordinals; variable-row per destination preserved around the fold).
- Integration branches (scratch; official restack owned by another agent):
  submodule `dsv4-canonical-unify` from pr16 head + cherry-picked DSV4
  surface + new serving commit; parent `dsv4-canonical-unify` from my branch
  + merge of pr44 + new trainer commit. Known conflict resolutions per
  reconciliation notes: http_server.py = union (health/model-override + FP32
  raw-logprob wire); auto.py = keep #43 Qwen capability resolution AND DSV4
  fail-closed registration; lora/utils.py = single-writer export AND
  948-factor dsv4_expert_banks export.
- Requalification (pod dsv4-flash, node 001): re-freeze base denominators
  (bytes CHANGE vs campaign 1), then A join + negative control, training
  gate, B join, 64-decision promotion, throughput. All drivers in this dir.

### Campaign 2 execution record

- Integration heads: parent `dsv4-canonical-unify` 1adc5db7c ("Fold DSV4 EP
  partials with the shared canonical BF16 tree", on top of the pr44 merge
  fb6352e25; only merge conflict was the submodule gitlink — auto.py and
  lora/utils.py merged per the reconciliation notes, audited); submodule
  `dsv4-canonical-unify` b1945bf0d ("Route the DSV4 exact post-experts
  combine through the canonical fold", pr16 renamed the primitive to
  `tensor_model_parallel_canonical_moe_all_reduce`; rewired + stale chain
  docstring fixed).
- Unit gates: 37 passed (dsv4_native_combine + canonical_moe_contract +
  moe_ep_native_combine) and 42 passed (dsv4_moe/lora/exact_contract/
  native_payload) in the combined env. New test
  `test_canonical_fold_diverges_from_the_retired_nccl_chain` is the
  byte-divergence witness (fold 2.015625 vs chain 2.0 on half-ulp ties).
- Fold-engagement witness at system level: campaign2 base trace must DIFFER
  from campaign-1 `trace_base_4dec.json` (same prompts, same seeds) — if
  bytes match, the serving gate did not engage.
- Evidence dir: `campaign2/` (campaign-1 files stay frozen).
  Wrappers: `campaign2_launch_base.sh`, `campaign2_capture.sh`
  (label/decisions/output as args).

### Campaign 2 root cause #1: the gated MoE-internal combine was off-path

First recapture (pod dsv4-flash v1 on node 001, heads 1adc5db7c/b1945bf0d):
campaign-2 base decode sha == campaign-1 sha (`d7ca8fd6…`) — bytes did NOT
change, and the new one-shot engagement log in `_post_experts_all_reduce`
never fired. Localization: for the DSV4 topology (a2a none + dp-attn +
attn_dp==tp==moe_ep==8), `should_use_dp_reduce_scatterv()` returns True, so
`should_skip_post_experts_all_reduce()` skips the MoE-internal combine and
the REAL combine is the layer-level `get_tp_group().reduce_scatterv`
(pynccl) in `models/deepseek_v4.py` — only the Qwen3.5 exact contract fell
back to the all-reduce + dp_scatter path. Consequences:
- The campaign-1 sampler-side ordered-combine patch was inert for THIS
  reason (not because "NCCL tree was already deterministic" — that
  conclusion is corrected here).
- The captured `[1..7,0]` contributor chain is the pynccl reduce path's
  order as observed for DP-rank-0 rows (the exact lane pins decisions to DP
  rank 0, so only rank-0 rows were ever byte-validated).
- Fix (in 6c0fe1ddd): `dsv4_flash_exact_mode` joins the exact fallback in
  `should_use_dp_reduce_scatterv()`; the combine then runs the MoE-internal
  gated canonical fold + the layer's non-reducing `dp_scatter` — the same
  call site and byte program as the Qwen/GLM exact lanes, uniform across
  destination DP ranks (the fold order no longer depends on the destination
  rank, unlike reduce_scatterv).
- Also seen on the first recapture: teacher-forced rep-1 suffix divergence
  from decision 11 at 64 decisions (decode bytes were rep-equal; reps 0/2
  matched campaign-1 TF bytes). Not yet root-caused; re-check after the
  fold engages — suspect allocation-layout or timing sensitivity on the
  pr16 base prefill.

### Campaign 2 root cause #2 (in progress): marlin prefill byte instability

After the fold engaged (pod dsv4-flash v2 on node 071, submodule 6c0fe1ddd):
decode bytes rep-equal ×3 at 4 AND 64 decisions, and CHANGED vs campaign 1
(4dec sha 7665ae54→d80e1cc7, 64dec d7ca8fd6→86e56cd5) — the engagement
witness holds. Teacher-forced prefill however is run-to-run nondeterministic:
- TF-only probe (`campaign2_tf_probe.py`, no interleaved decodes): 3 distinct
  byte streams over 12 identical requests at full length 74.
- Length sweep ×8-10 reps: stable at 14/18/24/34/44/48/52/56/60/64/65/68/
  69/70/71/73; UNSTABLE at 72 (8/2) and 74 (5/5) — routing-content
  dependent, not a monotonic threshold.
- First-divergence dump (dump-mode server, 6 passes at L=74, comparator
  `campaign2_compare_passes.py`): first divergent tensor is
  `model.layers.40.mlp` OUTPUT — ONE element, row 36 — while mlp INPUT,
  `gate`, `topk`, and `shared_experts` are byte-identical on ALL 8 ranks.
  By elimination the flip originates in one rank's routed MXFP4 MARLIN
  partial (the only un-dumped fold input).
- Campaign 1 had already pinned marlin's row block to 64 for the DSV4 exact
  geometry (`select_marlin_moe_block_size_m`, M=48 discriminator: blocks
  8/16/32 diverge). The residual instability at specific lengths is
  consistent with the LOCK-WORKSPACE/fp32-reduce-buffer CAP: both
  `fused_marlin_moe`'s int lock workspace and `moe_wna16_marlin`'s c_tmp
  are capped at sms*4 slots while the kernel's `locks_off` walks the
  absolute slice space (`c_cur_offset = locks_off * c_size`) — slice
  aliasing turns the barrier-sequenced reduce into completion-order
  dependence. Experiment: `SGLANG_MARLIN_FULL_WORKSPACE=1` removes both
  caps; probing L=74/72 ×20 and L=44 (bytes must stay IDENTICAL to the
  capped run's stable sha, else the sizing change altered the byte program).
- NOTE: campaign 1's TF stability at the same lengths was routing-content
  luck, not coverage — the latent race predates this campaign on both
  sides (trainer marlin is exposed at training M too; replay comparisons
  are M=1 and unaffected).

RESOLVED (submodule ce9209949): workspace caps refuted (uncapping locks +
c_tmp on both pipelines left 56/198 offline flips and 4 serving streams);
per-expert row histogram at the failing routing showed rank-1 expert 68
rows and rank-3 expert 66 rows > the pinned 64-row block — multi-block
experts are the race. Fix: under the pinned DSV4 exact geometry, chunk
Marlin token batches to 10 tokens (floor(64/topk-6)) so no expert can
span blocks: in fused_marlin_moe (trainer + direct callers) and in
MoeRunner.run with per-chunk LoRAInfo slicing (serving runner path).
Verification:
- Isolated (`campaign2/marlin_l40_rank{1,3}_chunked.json`): 0 flips in
  300 repeats on both hot ranks; fused == runner bytes; LoRA delta
  distinguishable (qualifier now fills all experts' factors — a captured
  routing may never select rank-local expert 0).
- Serving: lengths 14/44/72/74 stable ×20; base 4dec AND 64dec captures
  self-repeatable ×3 (decode + TF). Decode shas UNCHANGED from the
  pre-chunk canonical-fold captures (4dec d80e1cc7, 64dec 86e56cd5) —
  the 10-token prompt prefill and M=1 decode never chunk, so the frozen
  decode denominators carry; TF bytes at length 74 are the new
  deterministic chunked stream (73f991d339fe ×3).
Campaign-2 base denominators are FROZEN on these traces.

### Campaign 2 root cause #3: log_softmax kernel pair (ATen vs batch-invariant)

After the marlin fix, the 64-decision base ruler still mismatched at ONE
decision (39): one bf16 ulp (ba580000 vs ba590000), trainer-deterministic.
Localization: paired decision-39 dumps (serving Pass file matched by
position+token; trainer `.occurrenceNNNNN` components incl. new model-tail
captures `hc_head_output`/`final_norm` under pseudo-layer -1) proved the
ENTIRE trunk byte-equal through final_norm; offline tail A/B from the
byte-equal hidden reproduced the trainer wire under every GEMM program
(per-shard/full, M=1/8, fp32-accum, matmul_persistent). The split: serving's
deterministic mode interposes log_softmax with the batch-invariant Triton
kernel; ATen's and the BI kernel's BF16 outputs differ on this row at
exactly one entry — the f64 truth (-0.00082594) sits 5.5e-8 past the bf16
rounding boundary (ATen rounds correctly; the wire contract follows
serving). Campaign 1 never sampled a boundary value — the pair was
value-lucky, latent since the lane began.
Fix (parent 41bd2c54a): the DSV4 exact head's forward VALUE now uses the
serving BI log_softmax; the surrogate VJP keeps FP32 reference math.

### Campaign 2 qualification ladder (all on the unified program)

- Base ruler: 4-dec AND 64-dec replays byte_equal TRUE (K3 = 0.0 x 64),
  trainer self-repeatable x2, serving self-repeatable x3.
- A join: zero-adapter replay TRUE (and zero == base bytes on the wire);
  nonzero replay TRUE; negative control (nonzero session vs perturbed
  trace) correctly FALSE.
- Training gate: forward_backward + optim_step OK on the nonzero session;
  post-step replay of the pre-step trace correctly FALSE (weights moved).
- B join: adapter saved (dsv4_expert_banks, cleaned of optimizer shards to
  campaign2/adapter_trained), trained sampler captures b1 (4-dec) and b2
  (64-dec) self-repeatable x3; decode throughput 5.5 tok/s.
- Promotion: b1 (4-dec) byte_equal TRUE and b2 (64-dec) byte_equal TRUE
  with k3_max = 0.0 — K3 EXACTLY 0.0 x 64 with the trained adapter.

## CAMPAIGN 2 CLOSED 2026-08-11 — DSV4 unified onto the canonical fold

Heads: parent `dsv4-canonical-unify` (fold switch 1adc5db7c + qualifier
444f0245d + pad removal facace929 + BI log_softmax head 41bd2c54a),
submodule `dsv4-canonical-unify` (serving surface c143ddc50 + canonical
combine routing 6c0fe1ddd + marlin chunking ce9209949). Pod dsv4-flash on
node research-common-h100-071 (node 001 pod was deleted mid-campaign).
Three root causes burned down (all latent value-luck from campaign 1):
1. the exact serving combine was never the MoE-internal all_reduce
   (should_use_dp_reduce_scatterv kept the layer-level pynccl
   reduce_scatterv on-path; only Qwen fell back) — fixed by adding
   dsv4_flash_exact_mode to the exact fallback, with a one-shot
   engagement log;
2. Marlin multi-block experts (>64 routed rows) reduce across row blocks
   in completion order — fixed by chunking exact-geometry token batches to
   10 tokens in both engines' shared primitives (trainer row-pad to 48
   retired to mirror serving M exactly);
3. the ATen vs batch-invariant log_softmax kernel pair disagrees by one
   bf16 ulp on rounding-boundary rows — fixed by scoring the exact head's
   forward value with the serving BI kernel.
Decode throughput 5.5 tok/s (campaign-1-comparable). All campaign-2
evidence under `campaign2/`; campaign-1 artifacts frozen.
