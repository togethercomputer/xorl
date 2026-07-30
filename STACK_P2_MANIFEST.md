# P1 -> P3 -> P2 stack manifest

## Stack boundary

- Underlying OSS base: `577f34dbc4bd6a3298d93e53d5f0ee10dbeb6178`
- Corrected P1 head: `dfc14da12e9d2dc6b7e2f5b1bed6cf98d8c9b943`
- P3 source delta: `48b68ae73630afaf9041415be135e077d4807b52`
- P3 stack result and intermediate ref: `5e6ec1b8e4b0cd480a91248b1d54d914ac7a538f`
  on `codex/oss-upstream-stack-p3-20260730`
- P2 source delta: `982e91de916c86445060d6cc9524e94dc7916d05`
- P2 conflict-resolved result before this ledger/repair commit:
  `975893cb4813a12f4a68554922cac29aa44c96e9`
- Final branch: `codex/oss-upstream-stack-p2-20260730`, local and unpushed.

The final branch head is the validation/repair commit directly above the exact
P2 result recorded here. P1, P3, and P2 remain separate reviewable commits.

## Application result

P3 applied to corrected P1 without a content conflict. The corrected source
delta already carries generic public provenance, so the intermediate stack
independently passes the privacy scan.

P2 produced nine conflicts:

| File | Resolution |
|---|---|
| `src/xorl/checkpoint/checkpointer.py` | Kept P1 virtual-stage/multi-part checkpoint support, a superset of P2's single-model helpers. |
| `src/xorl/ops/linear_attention/layers/gated_deltanet.py` | Kept P1's public GDN-contract wording; P2's non-conflicting backend changes remain. |
| `src/xorl/ops/loss/causallm_loss.py` | Preserved P1/P3 FP8, Quack, and BI-fused loss paths while accepting P2 temperature and RL behavior. |
| `src/xorl/server/runner/model_runner.py` | Preserved P1 GLM5, FP8, model/runtime, and diagnostics behavior with P2's non-conflicting server/RL body. |
| `src/xorl/server/weight_sync/handler.py` | Preserved P1 sparse-delta, FP8, and canonical LoRA-fold support while accepting P2 backends and normalization. |
| `src/xorl/trainers/model_builder.py` | Kept direct GLM5 validation because P1 installs the package; P2 shared-builder changes remain. |
| `src/xorl/trainers/trainer.py` | Kept P1 multi-part PP schedules, bucketing/profiling, gradient handling, and metrics with P2 timing/RL changes. |
| `src/xorl/trainers/training_utils.py` | Kept P1 joint clipping over virtual-stage model parts. |
| `tests/server/weight_sync/test_handler_config.py` | Preserved P1 sparse-delta/FP8 coverage together with P2's non-conflicting handler tests. |

`arguments.py`, loss exports, dispatcher, packed-row utilities, and the rest of
the P2 server surface merged automatically.

## Integration repairs and hygiene

- Restored the public `LoadBalancingBuffer` and buffered global-batch load-
  balancing implementation because P1's retained trainer imports that API.
- Restored the public distributed metadata coordination helpers used by P2's
  retained training utilities: Gloo world metadata group, metadata all-reduce,
  and backend-safe barriers.
- Restored the sparse-delta request fields already consumed by P2's remote
  client and retained handler, and aligned focused test fixtures with P1's
  additional diagnostics and model-builder argument.
- Added the scoped lazy-import Ruff suppression required by the retained
  BI-fused loss helper.
- P1 keeps its batch-invariant runner hook opt-in by loading the P4-owned
  operator module only when `XORL_BATCH_INVARIANT_MATMUL=1`; ordinary stack
  imports do not require the later package.
- Restored the P3 versions of the Qwen3, Qwen3.5, and Qwen3.5-MoE model files.
  The P2 source delta carried delayed-residual, RMS-family, activation-native,
  and all-to-all chunk hooks in those files, but their shared BI implementation
  is owned by the later P4/P5 packages and is deliberately deferred to them.
- Deferred the matching later-package documentation and defaults: the FA4
  attention-default flip and BI-fused CE belong to P4, while all-to-all hidden
  chunking belongs to P5. Legitimate P2 server, optimizer, compile, and weight-
  synchronization documentation remains.
- Preserved the corrected source deltas' sanitized manifests, portable model
  fixtures, public contract wording, and environment-neutral paths.

## Validation

| Gate | Result |
|---|---|
| Patch and ancestry | PASS: `git diff --check`; P3 stack parent is corrected P1, P2 result parent is the P3 stack result, and this ledger/repair commit is directly above P2. |
| Ruff and syntax | PASS on integration-authored conflict/repair Python files; the three deferred model files are byte-identical to the P3 stack versions. |
| Whole-tree privacy and attribution | PASS for private absolute paths, private branch/repository names, dead note paths, ticket identifiers, tool attribution, and forbidden co-author trailers. |
| Import smoke | PASS across models, runner, loss, side payloads, weight sync, trainer utilities, and distributed metadata helpers. The optional GPT-OSS FlashAttention plugin emitted its expected missing-plugin warning. |
| Focused CPU suite | PASS: 112 tests across side payloads, runner token diagnostics, trainer alignment/utilities, RL primitives, FP8 config compatibility, QARL fake quantization, and weight-sync handler configuration. |
| Worktree and publication | PASS: final branch clean, local, and unpushed. |
