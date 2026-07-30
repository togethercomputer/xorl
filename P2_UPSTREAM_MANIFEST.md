# P2 upstream manifest: server RL, packing, and weight synchronization

## Provenance and stack

- Branch: `codex/oss-p2-server-rl-20260730`
- Current base: P3 head `6f033643a6abd19c08c942e65ab2605d828239e2`
- Target base under P3: OSS main `577f34dbc4bd6a3298d93e53d5f0ee10dbeb6178`
- Public-safe source: public staging ref `fca0655d7753e607e54b9a8d98f6064ec42c4ddc`
- Final stack: rebase this commit once onto the common P1 + P3 base. P4 and P5 follow P2 and are not prerequisites for its server/RL behavior.

The implementation was extracted by behavior rather than by merging histories. Every retained path exists in the public-safe source tree. Private endpoint names, absolute paths, cluster manifests, and unexported experiment/runbook files are excluded. Optional sparse-delta interoperability tests now use `XORL_DELTA_ENCODING_PATH` and `XORL_SGLANG_PATH` instead of machine-specific fallbacks.

## Scope

- OPD/OPSD, DR-GRPO, policy and importance-sampling losses; teacher caches; streaming and vocab-parallel reverse KL; packed-row and diagnostic paths.
- Server API/orchestrator/runner support for padding, packing, result-field preservation, behavior-logprob routing replay, ZORL, optimizer hyperparameters, and checkpointing knobs.
- NCCL/P2P/sparse-delta weight-sync hardening, quantization configuration, compile-FQN normalization, tied embeddings, KV-cache preservation, and Mooncake-backed hidden/side payloads.
- Portable server configuration documentation and the multi-LoRA example configuration.

## Lineage map

| Lineage | Retained behavior | Primary public paths |
|---|---|---|
| #341 | Teacher-logprob OPD parity | `ops/loss/opd_loss.py`, server OPD runner tests |
| #342 | Sparse-delta transport compatibility | `weight_sync/backends/sparse_delta.py`, sparse-delta tests |
| #343 | Slime-parity RL objectives | `rl/`, policy/importance-sampling loss tests |
| #344 | OPD hidden-match weights | OPD loss and runner configuration |
| #345 | Weight-sync hardening | weight-sync handler, backends, protocol tests |
| #346 | Server gradient-checkpointing method | server arguments and runner setup |
| #350 | Cold P2P recovery and bounded failures | P2P backend and protocol tests |
| #351 | Distributed server initialization repairs | launcher, dispatcher, orchestrator tests |
| #352 | Canonical OPD metric seeding | model runner and OPD tests |
| #354 | Restored OPD loss controls | OPD loss and parity tests |
| #359 | Hidden-cache filtering and safe forward | teacher cache/store and runner tests |
| #360 | P2P desync gate, compile-FQN stripping, tied head | weight-sync handler and forwarding tests |
| #361 | Compile-agnostic DCP/sync keys and teacher cache | checkpointer, teacher cache, weight sync |
| #362 | Base-model canonicalization, loss filtering, endpoint pools | request processor and endpoint/server paths |
| #363 | Real compile-FQN forwarding test wiring | weight-version forwarding tests |
| #364 | Per-rank EP forward/backward dispatch | dispatcher, packing, request processor |
| #365 | Server build-info and CI repairs | API/server and server argument tests |
| #367 | Chunked fused selected-token logprob | `fused_linear_logprob.py` and CPU/TP tests |
| #369 | Fused selected-token CE routing | causal/per-token CE paths |
| #370 | OPD/OPSD diagnostics and prefill timing | model runner diagnostics and tests |
| #372 | Hidden-only OPD skips zero-weight KL backward | OPD loss tests |
| #373 | Low-memory streaming reverse KL | `opd_streaming_kl.py` and tests |
| #374 | One-pass fused streaming reverse-KL forward | streaming KL implementation and parity tests |
| #379 | Forward/backward result-field preservation | model outputs, API types, dispatcher tests |
| #382 | Portable multi-LoRA server recipe | LoRA example config and server config docs |
| #383 | DP-aware and best-fit packing | orchestrator packing strategies and tests |
| #405 | Mooncake-only keyed teacher hidden cache | Mooncake hidden store and cache tests |
| #425 | Ignore-index target padding | packing/request-processing paths and tests |
| #426 | Behavior-logprob routing replay and side payloads | causal loss, request processor, `side_payloads.py` |
| #431 | DR-GRPO server loss and phase profiling | DR-GRPO loss, model runner, profiling tests |
| #432 | KV-cache preservation across P2P sync | remote backend, sync handler, backend tests |
| #434 | Vocab-parallel OPD and packed-row throughput | reverse-KL kernel, TP/HSDP wiring, distributed tests |
| #442 | K3 diagnostic metrics for RL losses | policy/importance-sampling losses and tests |
| #443 | Skip-param upcast and output-dir forwarding | server arguments, orchestrator, runner |
| #446 | ZORL parameter-server generation/apply endpoints | `server/zorl.py`, API/runner/backend tests |
| #451 | Rank-2+ `TensorData` serialization | API types and serialization tests |
| #485 | Adam betas/epsilon forwarding | training ops, dispatcher, runner, optimizer tests |

## Integration boundaries

- P1 owns the model/distributed runtime beneath the #434 TP/HSDP path. Rebase P2 onto P1 + P3 and resolve shared model/distributed files in favor of the P1 runtime while retaining the P2 loss/packing hooks.
- P4 owns the dense train/serve numerical contract. P2 does not add the P4 batch-invariant implementation; lazy shared-surface references in MoE model files must be reconciled when P4 lands.
- P5 owns MoE/GDN and Qwen3.5 zero-K3 completion. Do not treat the P2 CPU gate as MoE/GDN parity evidence; resolve its shared router/model surfaces only after P4.
- A real `/generate` to forward/backward to optimizer flow and a P2P completion-artifact smoke require the final stacked GPU/runtime environment and remain release gates.

## Validation

- `ruff check` passed on all 160 changed Python files.
- `ruff format --check` passed on all 160 changed Python files.
- `git diff --cached --check` passed.
- Public-safety scan passed: no private absolute paths, private repository names, cluster manifests, or forbidden attribution markers in the staged diff.
- Focused CPU tests were invoked for RL/DR-GRPO, API serialization, packing,
  weight-sync configuration, and the P3 sync/export/loss integration checks.
  On this host the pytest process entered uninterruptible I/O before producing
  test output, so no CPU pass is claimed; rerun those focused suites after the
  final P1 + P3 rebase.
