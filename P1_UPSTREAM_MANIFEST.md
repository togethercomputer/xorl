# P1 upstream manifest: production model and distributed runtime

Status: local review branch prepared from the sanitized public staging tree. Nothing in this worktree has been pushed.

## Fixed inputs

- Worktree: clean sibling checkout for P1; run the commands below from its root.
- Branch: `codex/oss-p1-model-runtime-20260730`
- Target base: `oss/main@577f34dbc4bd6a3298d93e53d5f0ee10dbeb6178`
- Sanitized implementation source: public staging ref `fca0655d7753e607e54b9a8d98f6064ec42c4ddc`
- Historical validated public snapshot: `6c77c025db6767d5ab44c883a4851f55ea0b40b8`; the P1 production paths at the requested source tip are unchanged from this snapshot except for package dependency metadata.

The pre-existing dirty development checkout was not used as an implementation source and was not edited.

## Included production surface

- GLM-5/5.1, DeepSeek V4, Nemotron-H, and MiniMax M3 model/config/checkpoint/parallelization packages and CPU-oriented registry/model tests.
- DeepSeek V4 sparse-MLA/indexer/hyper-connection operations, GLM-5 kernels, Nemotron SSM operations, and the portable GDN/FlashQLA runtime needed by the pipeline path.
- LM-head-only TP over DP/CP with EP composition; mixed-mesh FSDP2 gradient clipping; Muon EP optimizer-state resharding.
- Quack/DeepEP correctness and internode-transport hardening, including explicit preflight/record-stream coverage.
- GDN/Ulysses pipeline support, HSDP deferral across local and server packed microbatches, virtual pipeline stages, stage balancing, zero-bubble schedules, and pipeline-aware server forward-only support.
- The current public CUDA/PyTorch/TileLang/FA dependency tuple required by these model and kernel implementations. Simulator entry points/package data were deliberately removed from this package.

Explicitly excluded: experiment trees, benchmark ledgers, raw results, Kubernetes manifests, cluster names, private paths/hosts, internal skills, binary shims, the simulator package, and standalone OPD/ZORL/QAT/QLoRA surfaces owned by later packages.

## Lineage-to-files disposition

`Already present` means the required production behavior is in the fixed OSS base and no lineage-specific implementation delta was necessary. `Ported` means the implementation and focused coverage were extracted from the sanitized public source; internal lineage was used only to identify intent and tests.

| Lineage | Status | Production mapping / reason |
|---|---|---|
| #211 | Ported | `src/xorl/models/transformers/glm5/`, `src/xorl/ops/glm5_kernels/`, GLM registry/config validation, GDN/FlashQLA runtime, dependency metadata, and focused GLM tests. Experiment/config evidence was excluded. |
| #347 | Ported | `src/xorl/models/transformers/deepseek_v4/`, `src/xorl/ops/dsv4/`, the required loader/MoE/distributed integration, and DSv4 tests. Internal benchmark configs and cluster scripts were excluded. |
| #349 | Ported | LM-head TP mesh/group construction in `parallel_state.py`, argument/runtime wiring, FSDP/loss integration, and the EP-composition distributed test. Benchmark manifests were excluded. |
| #353 | Already present | The OSS base already registers Quack EP compute independently of the optional MoE-activation kernel (`EP_EXPERT_COMPUTE["quack"]`); current public backend coverage was retained without a standalone duplicate patch. |
| #355 | Ported | Current public Quack EP half-concat gated handling in `src/xorl/ops/moe/quack.py` plus `tests/ops/test_quack_ep_parity.py`; the private incident note was excluded. |
| #357 | Ported | `src/xorl/models/transformers/nemotron_h/`, SSM operations, registry/checkpoint/model tests, and portable optimizer/weight-sync integration. Design notes and GPU launch material were excluded. |
| #358 | Ported | Current public DeepEP no-permute/full-chunk cadence and tests in `deepep.py`, `quack.py`, and `test_quack_deepep_no_permute_parity.py`; sweep configs and failure logs were excluded. |
| #366 | Ported | Portable local GDN/FlashQLA CP kernel with the compiled scalar `tl.cast` fix in `src/xorl/ops/linear_attention/ops/cp/chunk_delta_h.py`. |
| #378 | Already present | The fixed OSS base already contains the production MoE adapter registration and causal-LM/Quack return contract. Current focused adapter tests are retained; unrelated RL-loss deltas were not used to justify this status. |
| #386 | Ported | Muon DTensor momentum-state construction, checkpoint/runtime wiring, and `test_muon_optimizer_ep_reshard.py`. |
| #390 | Ported | GDN + Ulysses PP shape/communication handling across `pipeline_parallel.py`, trainer/training utilities, and PP tests. |
| #391 | Ported | HSDP gradient-sync deferral, packed sequence max-length bucketing, config/collator/trainer wiring, and argument tests. |
| #392 | Ported | Server packed-microbatch HSDP deferral in `server/runner/grad_sync.py`, launcher/model-runner/server arguments, and focused server tests. |
| #397 | Ported | `src/xorl/models/transformers/minimax_m3/`, local config/registry/checkpoint integration, and `test_minimax_m3_support.py`. |
| #464 | Ported | Mixed-device-mesh DTensor fallback in `fsdp2/clip_grad_norm.py` and focused distributed coverage. |
| #465 | Ported | DeepEP record-stream ownership, internode preflight, aligned/bounded buffer validation, and `test_deepep_internode_guard.py`; repro manifests were excluded. |
| #466 | Ported | Virtual-stage/zero-bubble pipeline schedules, stage balancing/profiling, multi-part optimizer/checkpoint handling, server forward-only support, portable example config, and PP tests. Raw studies, ledgers, manifests, and simulation experiments were excluded. |

## Validation record

Environment observed during validation: Python 3.12.3, PyTorch 2.12.1+cu132, pytest 9.1.1, Ruff 0.11.4.

| Gate | Command | Result |
|---|---|---|
| Worktree integrity | physical `pwd -P`, `.git` gitdir, registered worktree, branch, base hash, and clean-start assertions | PASS; exact worktree/branch/base above. |
| Patch whitespace | `git diff --check` | PASS. |
| Changed Python lint | `git ls-files --modified --others --exclude-standard '*.py'` piped to `ruff check --` | PASS. |
| Server runner optional BI boundary | Import without later-package BI modules installed | The opt-in batch-invariant operator module is loaded only when `XORL_BATCH_INVARIANT_MATMUL=1`; full `ModelRunner` import is validated after stacking P2 because P1's retained runner also references P2-owned distillation support. |
| Core model collection/import | `PYTHONPATH=src pytest -q tests/models/test_glm5_support.py tests/models/test_minimax_m3_support.py tests/models/test_nemotron_h_registry.py tests/models/test_dsv4_autoconfig.py --collect-only` | PASS; 55 tests collected. |
| Core model CPU tests | same four files without `--collect-only` | Initial run: 54 passed, 1 failed because the new MiniMax model required the sanitized `MoeModelOutput.hidden_states` compatibility update. After porting `src/xorl/models/outputs.py`, the exact failed forward/backward test passed (1 passed). Full combined suite was not rerun after that bounded compatibility fix. |
| Focused runtime tests | `PYTHONPATH=src pytest -q tests/distributed/test_pp_profiling.py tests/distributed/test_ep_clip_grad_norm.py tests/server/runner/test_model_runner_grad_sync.py` | Initial run: 39 passed, 1 failed in the CUDA single-stage PP profiler because absent single-stage shape metadata produced `p2p_bytes=None`. The estimator now returns the correct zero before reading peer metadata; the exact failed CUDA test then passed (1 passed). Full combined suite was not rerun after that bounded fix. |
| Multi-process distributed tests | LM-head TP/EP, virtual PP, Muon EP reshard, Quack/DeepEP parity and internode guard scripts | NOT RUN locally; require torchrun/GPU resources proportionate to each test. |
| Hopper runtime | CUDA single-stage GPipe profiler test on the visible NVIDIA H100 80GB HBM3 device | PASS after the bounded zero-P2P estimator fix. Hopper smokes for the other changed axes were not run and remain required before a broad production-support claim or merge. |

## Review commands

```bash
git diff --name-status 577f34dbc4bd6a3298d93e53d5f0ee10dbeb6178..HEAD
git show --stat --oneline HEAD
```

The first command is the authoritative complete changed-file list; the table above records why each lineage contributed (or did not need) those files.
