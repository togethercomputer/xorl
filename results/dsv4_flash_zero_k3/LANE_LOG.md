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
