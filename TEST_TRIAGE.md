# Test triage

The goal is a smaller suite with a stronger failure signal, not a small suite by itself. Slowness, GPU requirements,
or a large test file are not sufficient reasons to delete a test.

## Decision standard

For every candidate, write down the plausible production regression that should make it fail. Then choose one action:

- **Keep** when it protects a supported behavior, numerical invariant, public integration, or failure path.
- **Consolidate** when the behavior matters but an equivalent contract is already exercised elsewhere.
- **Rewrite** when the intended contract matters but the test is stale, tautological, or coupled to implementation text.
- **Relocate** when it is a benchmark, diagnostic, campaign check, or external-model certification rather than a repository
  pass/fail test.
- **Remove** only when it has no distinct observable contract, is fully subsumed, or cannot run from a clean checkout.

A removal needs concrete evidence: the covering test, the missing tracked fixture, the tautology, or the reason no production
regression can make it fail. Candidate signals from automation are not evidence on their own.

## Semantic review heuristics

- Prefer one test per supported behavior or failure boundary. Put equivalent input spellings and invalid literals in a table
  inside that test unless they reach genuinely different production branches.
- A private-helper test must map to a supported producer/consumer path. An escape hatch or layout with no matching runtime
  endpoint is not a compatibility contract.
- Do not multiply tests by model layers, expert indices, tensor sizes, or environment variables when the production logic is
  shape- or index-independent. Keep representative boundaries and any case that changes ownership or control flow.
- Treat a large synthetic sweep as certification or stress coverage unless its scale can reveal a distinct repository
  regression. Move certification out of the default suite and remove repeated scale-only cases.
- When a direct unit fixture bypasses builder or loader state, reproduce the production admission state explicitly; otherwise
  the test may be asserting a branch it never entered.

## Workflow

1. Generate a fresh static inventory:

   ```bash
   python scripts/audit_tests.py --format json > /tmp/xorl-test-audit.json
   ```

2. Review one subsystem at a time. Add the decision and evidence to `test_audit_decisions.json` before editing.
3. For consolidation, map every removed assertion to its surviving behavioral test.
4. Compare collection before and after, then run the affected surviving tests.
5. Keep removal waves reviewable. Do not combine unrelated product changes with test cleanup.

The scanner intentionally over-reports tests with no inline assertion, conditional skips, source inspection, or exact duplicate
bodies. Fixtures and helper assertions can make those tests valuable; a human must inspect each candidate.

## Initial wave

The first wave removes tests that are fully subsumed, permanently skipped in a clean clone, or unable to report a failure.
It relocates the official GLM-5.2 checkpoint inventory to `certification/glm52/`. Valuable but stale checks remain in the
decision ledger as proposed rewrites rather than being deleted.

## Second wave

The second wave removes four fully covered or self-referential helper tests, consolidates three duplicated loss contracts,
and replaces the remaining implementation-source inspection with a behavioral RoPE discriminator. It also repairs the
server-argument test harness so temporary import stubs restore only the keys they replace instead of unloading native
extensions imported during collection.

After this wave the static inventory contains 2,646 test definitions, 79 review candidates, one accepted duplicate wrapper
pair, and no source-inspection candidates. The decision ledger records the covering evidence and keeps parametrized behavior
for both loss implementations.

## Third wave: semantic contracts

The third wave reviews four high-density modules by production contract rather than scanner signal. It consolidates config
override matrices, checkpoint-key spellings, weight-sync precedence rules, and direct-EP selection fragments. It removes
orphan P2P fused layouts that have no receiver locator, a layer-by-expert scale sweep of layer-independent logic, and a
single-receiver FP8 case covered by the retained multi-receiver contract.

Those four modules collect 167 tests instead of 225. The static repository inventory now contains 2,626 definitions; all
invalid config values and global expert indices remain covered. The review also found and repaired an exact-GLM fixture that
omitted the internal admission marker installed by the production model builder.

## Fourth wave: geometry versus branch coverage

The fourth wave targets parametrization that varied sizes without reaching new code. V4 indexer coverage now keeps the basic,
batched, production-head/top-k, and C128 geometries; its separate real-config sweep was already covered by the score and top-k
contracts. Families-v2 RMSNorm keeps tail, aligned, and deep-tile shapes at row-count boundaries, while its shipped-size
dispatch check reflects that rows cannot affect the decision below the split-tile threshold.

The decode GDN suite no longer invokes internal warp/group launch configurations that no production caller can select, and
its scaled-then-normalized K regime is not treated as a separate input contract. Pipeline placement now checks each mapping
formula rather than multiplying the same arithmetic across PP sizes.

These four modules collect 44 tests instead of 141. All 44 retained tests pass on GPU. Repository collection is 3,056 items
and the static inventory is 2,624 definitions with 23 curated decisions.

## Fifth wave: tables are contracts, not test suites

The fifth wave collapses parser and policy truth tables into one contract each, preserving every input row. Fused selected-
logprob parity replaces independent dtype × bias × temperature products with pairwise branch coverage while retaining all
irregular shapes and production vocabularies. The batch-invariant GEMM table still executes every dtype/shape combination but
reports one doctrine-level bit-neutrality contract.

Sparse MLA retains all four compiled top-k specializations, the batched case, and the production 64-head geometry. Its sink
coverage keeps the zero boundary and mixed-sign values; a separate large-sink test still proves that the kernel consumes the
sink rather than ignoring it.

These four modules collect 44 tests instead of 101. The retained set reports 43 passes and one optional DeepGEMM skip on the
current GPU host. Repository collection is now 2,999 items; the static inventory remains 2,624 definitions with 27 curated
decisions.

## Sixth wave: cross-subsystem semantic cleanup

The sixth wave spans LoRA gradient ownership, torch.compile, inference API normalization, packing, quantization validation,
and FP8 weight selection. LoRA quantized experts now use pairwise format/backend coverage while retaining every format,
backend label, and producer branch. Explicit monkeypatch contexts preserve per-case distributed-state isolation.

The compile suite removes a fullgraph probe that swallowed every exception and two non-collected, print-only benchmark
routines. Its three retained contracts perform real AOT/Inductor compilation at block, decoder-layer, and full-model levels.
Inference endpoint detection keeps its integrated HF-config normalization contract rather than duplicating private helpers.

Packing removes an algebraic restatement of row minimization and a Python-list-indexing tautology; datum order is now checked
against document boundaries in actual packed rows. Quantization aliases and rejection literals are grouped by boundary. FP8
projection selection is one buffer-level contract spanning supported families and negative cases instead of fourteen narrow
cases.

The six focused suites report 106 passes. Repository collection is 2,926 items and the static inventory is 2,610 definitions
with 33 curated decisions.

## Seventh wave: separate backend math from topology

The seventh wave targets distributed adapter-autograd and expert QLoRA matrices. The unquantized distributed suite still
executes every backend at EP2, but the four-GPU eFSDP composition uses one shipped Quack representative rather than repeating
the same backend dimension. The default all-owner layout likewise uses one representative after backend math is established.
Quantized expert autograd replaces the three-backend by three-format Cartesian product with pairwise coverage that retains
every backend and format; DeepEP and exact projection-subset contracts remain independent.

Expert QLoRA capability, unsupported-semantic, model-family, and invalid-target tables now report one test per behavioral
boundary. Every table row still executes, including construction of all three model-specific expert families. The two files
collect 29 tests instead of 60 and eliminate twelve distributed subprocess launches. The retained QLoRA suite reports 17
passes; the distributed gate reports 10 passes and two optional DeepEP skips on eight H100 GPUs. Repository collection is
now 2,895 items; the static inventory remains 2,610 definitions with 35 curated decisions.

## Eighth wave: make numerical tests exercise nontrivial numerics

The eighth wave strengthens MoE-LoRA backend parity before removing its weaker smokes. The previous cross-backend gradient
test left every LoRA-B factor at its zero initializer, so all LoRA-A gradients were trivially zero. The retained comparison
uses nonzero adapters and compares eager with both native and Triton outputs plus every A/B gradient. It subsumes separate
GPU forward/backward smokes; explicit zero-delta and nonzero-effect reference contracts remain. Backend-independent adapter
construction uses one Quack representative, while registration and numerical contracts continue to cover all backends.

RoPE validation still checks the exact fp32 frequency table for every registered initializer. Shared forward consumption now
uses default and YaRN to cover unit and non-unit attention scaling. The native-cache comparison is restricted to default
RoPE, the only type for which `_rope_native` changes `RotaryEmbedding` control flow; five non-default cases previously
compared two executions of the same stock branch. The focused suites report 16 MoE-LoRA and 7 RoPE passes. Repository
collection is now 2,864 items and the static inventory is 2,608 definitions with 37 curated decisions.

## Ninth wave: FP8 launch branches and transactional corruption boundaries

The ninth wave reduces FP8 grouped-MoE geometry by kernel control flow. Same-NK coverage keeps small- and large-N launch
branches, aligned and padded K, an empty expert, M spanning several blocks, and output tails in two cases. Wgrad retains the
same split plus a separate all-empty case for its `max_K == 0` early return. Both grouped backends still perform a real
optimizer step, while the dense-plus-MoE integration uses the default Triton-grouped representative instead of repeating
the already-qualified backend dimension.

FP8-linear block sizes and recipe rows remain intact because they produce different padding, scale grids, SmoothQuant, and
amax behavior; each table now reports one numerical contract. Adapter-optimizer resume similarly retains every identity,
layout, coverage, dtype, shape, step, and staged-state corruption, but groups them into three transactional contracts using
isolated checkpoint directories. The focused gates report 41 FP8 passes with one opt-in DeepGEMM skip and 26 optimizer-resume
passes.
Repository collection is now 2,842 items; the static inventory remains 2,608 definitions with 40 curated decisions.

## Tenth wave: checkpoint gates and resolver truth tables

The tenth wave rewrites gradient-checkpointing coverage around the actual gate. Class and enable-time defaults are one
contract across base and MoE layers. Nondefault propagation uses one representative because the model method performs the
same assignment for each already-validated string. The outer-gate test adds the previously missing method predicate and now
proves that selective checkpointing does not invoke the full-layer checkpoint even when training and the feature flag are
both enabled.

MoE routing-position auto regimes and explicit true/false aliases retain every row, but each resolver boundary is one test.
The two focused files collect 12 tests instead of 27, and all 12 pass.

## Eleventh wave: exact-construction admission boundaries

The eleventh wave groups GLM-5.2 exact MoE dependency, EP-topology, and adapter-shape admission failures. Every missing exact
component, sparse-MLA/dispatch requirement, uninitialized or wrong-sized EP state, and individual rank/alpha violation still
fails before adapter mutation. A doubly invalid rank-16/alpha-16 row was removed because the rank-only and alpha-only cases
already exercise both sides of the same combined predicate.

Exact shared-expert construction keeps all seven shape, TP, rank, alpha, bias, and adaptive-noise rejection boundaries in one
contract. The two focused files collect 14 tests instead of 28 and report 12 passes with two SGLang-dependent skips.
Repository collection is now 2,813 items and the static inventory is 2,607 definitions with 44 curated decisions.

## Twelfth wave: test complete loader transactions, not their fields separately

The twelfth wave replaces fragmented QLoRA expert-loader tests with one integrated contract per quantization format. Every
retained geometry still loads, but each load now validates packed bytes, scales, and all three projections together. NVFP4
also checks per-expert amax, absorbed global scales, and dequantized shape/dtype. This reduces fourteen repeated synthetic
checkpoint loads to six and removes an uncollected, print-only timing/`__main__` harness with no pass/fail threshold.

FP8 external-config coverage retains all Transformer-Engine-only recipe keys and ModelOpt QARL nesting variants inside their
own fail-closed contracts. The focused QLoRA and config suites collect 12 tests instead of 29, and all 12 pass. Repository
collection is now 2,796 items and the static inventory is 2,601 definitions with 46 curated decisions.

## Thirteenth wave: remove masked axes and weaker backend smokes

The thirteenth wave keeps both RMSNorm kernel geometries but reports the family funnel as one bitwise contract. Its module
parity matrix already enables the trunk contract in every row, which fixes family dispatch independently of the separate
batch-invariant flag; that masked axis is removed while native, SGLang, and fused-SGLang modes remain. Dense and MoE Qwen3.5
call-site truth tables retain every layer/mode row inside one contract per predicate.

OPD chunking keeps disabled, one-chunk, and over-partitioned boundaries rather than arbitrary intermediate positive counts.
All VERL estimator aliases and both streaming backends remain. Three direct compiled-function tuple/shape smokes are removed
because retained end-to-end reverse/forward KL tests execute the same functions and additionally validate formulas and
diagnostics. The focused gates report 27 OPD and 34 RMSNorm passes. Repository collection is now 2,765 items and the static
inventory is 2,598 definitions with 49 curated decisions.

## Fourteenth wave: collapse arbitrary geometry and transactional row inflation

The fourteenth wave reduces NVFP4 forward coverage to its two supported dtypes because the three legal matrix shapes all
entered the same flatten-and-block path. Exact equality with the independent quantization reference subsumes separate
Gaussian-error and lossy smokes, while the E2M1 grid, STE, invalid rank/K layout, registry, expert isolation, fused-half
scale, and metadata contracts remain. EP wrapper signatures now inspect every entry in both live registries inside one API
contract rather than reporting one node per backend.

Canonical MoE keeps the two-contributor base tree and the production sixteen-contributor tree; intermediate CPU reference
widths selected no new branch. Its distributed test retains two and eight contributors plus the independent EP16 packed
transport gate. A second auto-transport test was removed because its two assertions were verbatim subsets of the adjacent
admission contract. GLM-5.2 QLoRA keeps independent rank-only and alpha-only failures but removes their doubly-invalid
conjunction, and its construction-mode table still builds a fresh model for every rejected mode. Exact-attention checkpoint
tests still run both arrival orders and both weight/scale members for duplicate, missing, dtype, and shape failures, but each
transactional behavior reports one contract.

The five files collect 53 tests instead of 80. The focused gates report 52 passes and one optional CUDA-path skip.
Repository collection is now 2,738 items and the static inventory is 2,593 definitions with 54 curated decisions.

## Fifteenth wave: remove scale certification and test-the-helper arithmetic

The fifteenth wave replaces the Quack EP distribution-by-score Cartesian product with three pairwise cases that retain
balanced, empty-expert, extreme-skew, score-free, and score-scaled behavior. Eager/native MoE parity keeps ordinary, top-k
one/four, and single-token routing boundaries. Two size-only configurations and a separate E64/H512/I1024 test were removed:
they selected no repository branch, and the latter only repeated parity with larger allocation and looser tolerances.

Full-reduce mean now crosses its actual one-dimension versus multi-dimension branch with both supported dtypes, without a
third arbitrary tensor rank. V2 RMSNorm retains shipped and deep hidden sizes at low and high row counts, all residual modes,
and explicit split/fused execution; the intermediate row count selected no dispatch or kernel boundary. LoRA merge keeps
zero and nonzero production merges for every original dtype in both linear and MoE implementations. A supposed precision
test was removed because it never invoked either production merge method and only compared two formulas local to the test.

The five files collect 15 tests instead of 42, and all 15 pass on the current GPU host. Repository collection is now 2,711
items and the static inventory is 2,591 definitions with 59 curated decisions.

## Sixteenth wave: distinguish repository regressions from dependency canaries

The sixteenth wave reports all supported EP backend gradient domains as one local contract while retaining the real two-rank
autograd and synchronization gate. Exact-attention construction keeps rank-only and alpha-only rejection, removes their
doubly-invalid conjunction, and still builds a fresh model for every incomplete execution mode. Every active-LoRA component
flag remains independently cleared against all four admission predicates inside one conjunction contract.

Mamba2 exact-chunk/tail, multi-chunk/tail, aligned/unaligned packed boundaries, forward values, and gradients all remain.
The test asserting that the Transformers 5.5.3 fallback bug still exists was removed: an upstream fix is not an XoRL
regression and should not fail the repository. The sequential SSD recurrence remains the multi-chunk oracle. Qwen3-MoE input
norm family coverage now varies only layer zero versus later layers; the old mode axis was ignored by the capture stub and
could not affect the call site. FlashQLA retains every M/batch and chunk-chaining execution but reports each certification
gate once.

The six files collect 43 tests instead of 67. The focused gates report 41 passes and two optional mamba-ssm skips.
Repository collection is now 2,687 items and the static inventory is 2,590 definitions with 65 curated decisions.

## Seventeenth wave: remove dominated loss references and group receiver truth tables

The seventeenth wave keeps all four full-weight/adapters admission combinations and all three optimizer-load spellings while
reporting each argument boundary once. EP checkpoint selection still rejects every missing/ambiguous named dimension and
restores both legacy and PP-parent meshes. Native GLM FP8 validation still rejects each nonofficial quantization field.

Streaming forward-KL backward parity already compares forward values and gradients to the independent dense oracle, so the
separate forward-only check was removed. The direct compiled-reference unit was also removed because the retained end-to-end
OPD dispatch compares both streaming aliases with the compiled backend including gradients. Chunk invariance now directly
compares a seven-wide multi-chunk execution with a forty-wide single-chunk execution; the old 40-versus-40000 one-chunk row
added no branch and the 40000-versus-itself row was tautological. TokenPartial microbatch additivity uses one caller-supplied
denominator because denominator values select no reducer branch; scale-one and both token/sequence zero-denominator contracts
remain independent.

The five files collect 55 tests instead of 74, and all 55 pass. Repository collection is now 2,668 items and the static
inventory is 2,588 definitions with 70 curated decisions.

## Eighteenth wave: replace stochastic convergence experiments with mechanism checks

The eighteenth wave removes five QLoRA random-target convergence experiments that collectively ran roughly 750 optimizer
iterations. They included a 400-step rank-accumulation comparison and a loose reset-within-two-times threshold; neither maps
to deterministic repository control flow. Both quantization formats still exercise storage, memory, forward, and backward.
NVFP4 and Block-FP8 merges, prequantized loading, requantization, injection, LoRA-only optimizer reset, and non-LoRA state
preservation remain. The scheduler integration now proves an off-boundary no-op and an on-boundary packed-weight change,
LoRA-B reset, and stale optimizer-state removal directly after one state-populating step.

Mooncake metadata still rejects every missing/invalid field, LoRA manifests still reject every nonexact scalar, Nemotron-H
checkpoint construction still covers indivisible experts, invalid rank, and invalid EP size, and DeepSeek training still
rejects each unsupported mode. These receiver truth tables now report one contract apiece.

The five files collect 45 tests instead of 62, and all 45 pass. Repository collection is now 2,651 items and the static
inventory is 2,583 definitions with 75 curated decisions.

## Nineteenth wave: force real reference paths and delete refactor tombstones

The nineteenth wave removes GLM-5 construction smokes already subsumed by the default-shape and end-to-end model contracts.
It also fixes a false oracle: a test labeled as TileLang-versus-torch passed a pure causal mask, but that mask is accepted by
the TileLang fast path. The retained GPU contract now opts into blocked scoring to force the independent torch path before
comparing outputs. All shared-factor LoRA autograd executions remain, reported as one behavioral contract.

The simulator's consolidated validator now owns built-in pack discovery, schema, sanitation, and exact raw/promotable
goldens. DeepEP topology and preflight scenarios are complete truth tables rather than ten separately reported micro-tests.
Packing drops a test-only abstract subclass, a brittle exact-key whitelist, an arbitrary 100-sample repetition, and repeated
roundtrip modes already covered by focused tests; numpy normalization and one composition roundtrip remain. Session cache-path
canonicalization still runs every direction inside one contract.

Finally, thirteen launcher tests explicitly targeting APIs removed by the launcher refactor are deleted instead of being
collected forever as skips. The live launcher suite retains seven address, readiness, command, parsing, and migration tests.
The seven focused files collect 154 tests and all 154 pass. Repository collection is now 2,619 items and the static inventory
is 2,555 definitions with 82 curated decisions.

## Twentieth wave: test artifact transactions and behavioral truth tables

The twentieth wave joins checkpoint metadata writers to their actual compatibility readers. QARL buffer metadata and
pipeline-stage key unions are now written, inspected, and validated as complete transactions instead of testing producers
and consumers against separate fixtures. Token diagnostics similarly combine shape and target-ranking invariants, plus the
two equivalent disabled-input exits, without dropping any output field or hidden-component coverage.

Muon keeps real optimizer autotuning and drops the helper-only duplicate. Quack tuned/untuned dispatch, SM90 dtype backend
selection, and cautious optimizer-family routing retain every execution as truth tables. API constructor field echoes and
Pydantic assignment smokes are removed because retained endpoint tests cover defaults, aliases, serialization, registration,
and payload forwarding at the application boundary. The heartbeat check no longer sleeps on wall-clock time.

Trainer clipping, vote scaling, and target-token preference tables retain every original case. Router tie-policy aliases are
grouped, and the invalid policy test now asserts the real construction-time failure. Distillation cache host/device rank-3
gathers and both bounds failures are grouped by behavior. DeepSeek-V4 successful name mappings form one complete mapping
contract while unknown/MTP rejection stays separate. The nine focused files collect 152 tests and all 152 pass. Repository
collection is now 2,597 items and the static inventory is 2,536 definitions with 91 curated decisions.

## Twenty-first wave: keep transactions, collapse helper reporting

The twenty-first wave keeps request-processor transport, Mooncake cleanup, model-ID propagation, registration, packing,
statistics, and error boundaries. Backend timing-field preservation now lives in the main model-pass contract, and five
identical sequential forward smokes are removed from an error test because the dedicated statistics contract already proves
exact operation accounting. Runner dispatch drops a second routing-slice test whose expert IDs, logits, offsets, and metadata
cleanup were all asserted by the retained test.

Pipeline profiling retains every interval-union case, nonzero and zero bubble formula, rejection boundary, P2P topology,
patch lifecycle, and CUDA-event integration, but reports the pure-math cases as behavioral truth tables. The OPD driver still
checks chunk order/tails and every student weight-version outcome without separate nodes per outcome. Sparse-delta capture and
writing retain both traversal fields and all duplicate, dtype, length, and range failures inside receiver contracts. The
adapter-coordination suite and quantized export transforms were audited and kept because their tests cover distinct rollback,
trust-root, sharded-state, and tensor-conversion paths.

The five focused files collect 82 tests and all 82 pass. Repository collection is now 2,583 items and the static inventory is
2,523 definitions with 96 curated decisions.

## Twenty-second wave: prefer end-to-end numerical contracts over arithmetic fragments

The twenty-second wave removes the densest remaining arithmetic fragments from EP gradient clipping. Retained
classify-then-clip tests already prove skip-FSDP and ordinary grouping, combined L2 clipping, uniform scaling, and the
no-double-division regression. Infinity norm, missing gradients, shared replicas, dispatch, mixed DTensor meshes, explicit
foreach behavior, and live two- and three-rank reductions remain. Separate single-group, 3-4-5, no-clip, and duplicate
mixed-mesh examples selected no additional branch.

DistSignSGD now proves signing and forced-SUM communication in one AVG-input transaction; its weight-decay step subsumes the
plain update, and its state-dict roundtrip subsumes a separate state-empty smoke. Unsupported HSDP, folded sequence
parallelism, and EP topologies still fail against fresh models inside one truth table. Scheduler coverage retains every
warmup, decay, floor, cosine, and validation branch while dropping phase examples already contained in longer traces.

NVFP4 ownership is now explicit: the op suite owns the independent quantization reference and exact 2D/3D STE arithmetic,
while QARL wrapper suites own configuration, injection, lossy production forwards, parameter restoration, and gradients.
Direct private-helper/shadow tests, a looser grid property, and a supported-format boolean smoke were dominated by those
contracts. Non-16 group sizes, dense versus expert target selection, FP8-on-MoE rejection, and quantization-disable parity
all remain. Tensor-parallel FP8 model building still executes both included and excluded lm-head outcomes in one contract.

The eight focused files collect 67 tests instead of 105, and all 67 pass, including both live distributed gradient gates.
Repository collection is now 2,545 items and the static inventory is 2,485 definitions with 101 curated decisions.

## Twenty-third wave: replace QARL context fragments and convergence heuristics

The twenty-third wave turns QARL activation override coverage into state transactions. Enabled and disabled modes, distinct
per-module restoration, exception cleanup, and exclusion of ordinary linears now execute together; nested contexts remain a
separate reentrancy contract. W4A4 activation quantization retains its independent forward reference and a nonuniform
upstream-gradient STE check, which subsumes the all-ones gradient example. The exception path proves both `triton_w4a4`
selection and restoration, while activation-off and non-Triton no-op cases form one conjunction-boundary table.

The dense QARL training smoke no longer trains sixteen steps against a synthetic target and asserts that loss happens to
decrease. One real AdamW step now proves finite loss, gradients through both wrapped projections, parameter mutation, changed
logprobs, persistent summary metadata, and exact checkpoint restoration. The weight-sync handler's mismatch response already
enters the block-size validator, so the weaker direct failure unit is removed.

Cautious SignSGD and AnyPrecisionAdamW keep mixed-coordinate production steps that contain aligned and misaligned elements
and assert the exact mask/update. Separate all-aligned comparisons selected no different branch and are removed. The five
focused files collect 30 tests instead of 39, and all 30 pass. Repository collection is now 2,536 items and the static
inventory is 2,476 definitions with 105 curated decisions.

## Twenty-fourth wave: make GLM selector tests describe production behavior

The twenty-fourth wave removes an IndexShare test that only proved a recursive mapper defined inside the test file preserves
non-tensor Python identity. The retained dense-producer/shared-consumer model forward runs that mapper from simulated FSDP
pre-hooks and proves one context survives all layers, is consumed without recomputation, and is cleaned up after the forward.

GLM-5 selector shape, dtype, range, final-row validity, and sorted-sentinel ordering now form one output contract. Dense and
one-head-chunk scoring still execute the same diagonal-only additive mask inside one behavioral table. A CUDA- and
TileLang-gated test that never invoked TileLang is now an honest CPU mask-classification contract for valid prefixes versus
interior holes. The standalone sparse-attention shape smoke is removed because retained full-model sparse-versus-dense
parity and Ulysses integration already exercise the forward while checking numerical output, query/KV locality, indices,
offsets, masks, and output shape.

Six focused retained contracts pass. Repository collection is now 2,532 items and the static inventory is 2,472 definitions
with 108 curated decisions.

## Twenty-fifth wave: report exact-model admission as family contracts

The twenty-fifth wave retains every exact Qwen3.5 execution while removing pytest item inflation. Dense and MoE configs both
resolve the full certified attention, router, lm-head, RMSNorm, activation, RoPE, cast, sparse-MLA, and cross-entropy program
inside one family contract. World16 HSDP-plus-EP, world8 EP, and single-GPU dense topologies still validate from fresh config
objects inside one accepted-topology table.

Model-scope admission likewise keeps dense, MoE, and Hugging Face outer-config snapshots in one accepted-scope contract.
Both dense and MoE hidden-size near misses still fail independently in one rejection contract. The file collects 21 tests
instead of 27, all 21 pass, and no configuration execution was dropped. Repository collection is now 2,526 items and the
static inventory is 2,469 definitions with 110 curated decisions.

## Twenty-sixth wave: consolidate weight-sync setup, preserve adapter transactions

The twenty-sixth wave joins requested-adapter materialization and current-adapter fallback in one state contract, and joins
dense-buffer chunking with the cap predicate it consumes. A tied-weight root extraction is removed because the retained
prior-module test performs the same extraction and alias assertions before additionally proving the later duplicate is
skipped. Direct Nemotron-H prefix mapping and sparse-delta request-field echoes are removed because retained unfuse,
remote-backend, and end-to-end request transactions verify those values at their actual consumers.

Three copied sparse-delta fake endpoint/backend harnesses are replaced by one transport transaction. Baseline post-only
ordering/accounting, explicit baseline configuration, and FP8 KV-cache postprocess metadata all still execute and assert
backend configuration, normalized cache epoch, endpoint results, pause/resume order, posted paths, and weight version.

The 48-test adapter-manager suite was audited and kept. Its size comes from distinct gradient-ownership, staging, atomic
commit, abort, clipping, collective, poisoning, publication, trust-root, session-spec, rollback, eviction, mixed-rank, and
optimizer-state boundaries rather than literal variation; all 48 pass. The weight-sync file collects 30 tests instead of 37,
and all 30 pass. Repository collection is now 2,519 items and the static inventory is 2,462 definitions with 113 curated
decisions.

## Twenty-seventh wave: distinguish model smokes from numerical certification matrices

The twenty-seventh wave reports MiniMax-M3 language-model prefix and w1/w2/w3 expert-key aliases as one classifier contract,
retaining all strings. The DSv4 base-model C0 shape smoke is removed because the retained causal-LM C0 forward/backward runs
the same base model and additionally checks logits, finite loss, required gradients, and intentionally frozen
hyperconnection parameters. The distinct C128 compressed-attention forward remains. The two focused model-support files
collect 21 tests instead of 24, and all 21 pass.

Cross-engine RMSNorm shapes are valuable certification inputs rather than arbitrary examples, so none are removed. All four
adversarial shapes, both family funnels, residual modes, trunk lane, zero-centered twin, and families-v2 candidate executions
remain, but each invariant reports one test rather than one item per row. That suite would collect 9 tests instead of 34 in
an environment with SGLang batch-invariant ops; the current venv dependency-skips the module. The locally available fused
suite now reports BF16 and FP32 residual/no-residual parity as two dtype-complete contracts, and both pass.

Repository collection is now 2,514 items and the static inventory is 2,461 definitions with 115 curated decisions.

## Twenty-eighth wave: keep transport boundaries, remove routing and telemetry examples

The twenty-eighth wave keeps the weight-sync boundaries that can corrupt state or strand a receiver: primary-to-fallback
health ordering, NCCL initialization, two-phase completion metadata, mixed-dtype byte flattening, chunking, receiver-fenced
work lifetime, multi-rank load-format rejection, sparse-delta byte-change encoding, unchanged-bucket suppression, baseline
priming and rollback, FP8 KV-cache metadata, and per-rank packed paths. It removes a standalone primary-health URL example,
folds configured load-format forwarding into the existing bucket-routing transaction, and makes the receiver-fence contract
also prove cleanup on group destruction instead of repeating the entire hybrid-broadcast harness.

Sparse-delta drops a factory `isinstance` mirror and a single-path replication example already executed by streaming
transfer. The retained per-rank prepacked path contract now owns unique-file accounting. Post-only import avoidance,
prepacked-only streaming rejection, and valid prepacked post-only initialization all still execute as one policy table. The
fixture now uses a validated loopback literal rather than relying on the fake hostname `infer-0` to resolve before mocked
HTTP calls reach production URL-safety validation.

Trainer IB-device selection retains global-rank, local-rank, single-device fallback, explicit physical-GPU, numeric
`CUDA_VISIBLE_DEVICES`, and empty-entry autodiscovery outcomes in one environment-isolated precedence contract. Two private
telemetry tests that only copied fields into a dictionary or added three counters are removed; abort-marker lifecycle and
distributed peer-failure gathering remain. The three focused files collect 21 tests instead of 35, and all 21 pass.
Repository collection is now 2,500 items and the static inventory is 2,447 definitions with 119 curated decisions.

## Twenty-ninth wave: parse composed configurations, not one scalar per test

The twenty-ninth wave keeps configuration parsing as a production boundary while removing one-node-per-field reporting.
Flat adapter-ownership, nested removed ZORL field, and removed ZORL-section inputs still fail through the real server loader
inside one rejection table. Both shipped MoE LoRA examples and all five shipped Qwen MoE QLoRA examples still parse in a
clean subprocess and compare source YAML with normalized Quack, expert-target, and shared-LoRA values, but no longer create
one pytest item per file.

The server loader now proves both SignSGD spellings in composed nested configurations that also carry checkpoint optimizer
policy, forward/backward prefetch, HSDP deferral, packing alignment, activation memory limits, and adapter state-load mode.
The runtime attributes and serialized model/train/LoRA dictionaries remain checked. Explicit Mooncake, legacy Mooncake
alias, and filesystem R3 success modes form one transport table while the invalid-directory failure remains separate.
Explicit and automatic MoE routing-weight placement likewise share one defaulting contract.

The training CLI parser now accepts both sign optimizers while simultaneously preserving multipack fields, model numerical
alignment, FSDP reduction dtype, and parameter-upcast policy from production-shaped YAML. Muon conversion, legacy alias
transforms, FP8/QARL modes, automatic checkpoint resolution, load-optimizer defaults, and every incompatibility rejection
remain independent. The two focused files collect 50 tests instead of 70, and all 50 pass. Repository collection is now
2,480 items and the static inventory is 2,432 definitions with 123 curated decisions.

## Thirtieth wave: turn numerical-contract fragments into guard tables

The thirtieth wave retains every GatedDeltaNet exact-convolution rejection branch while reporting decode cache, context
parallelism, missing short convolution, and convolution bias as one unsupported-input table. Independent weight-packing
order, production routing, forward/backward references, kernel and end-to-end determinism, state scoping, checkpoint
recompute, and optional SGLang parity remain separate.

Batch-invariant trunk wrapping likewise keeps no-match, ordinary-LoRA, custom-Linear, and FP16-weight failures inside one
admission table. Bias-free and biased forward comparisons still execute against the persistent GEMM, and both backward
comparisons still execute against cuBLAS autograd, without parametrized item inflation. A standalone RMSNorm loud-failure
test is removed because the retained multi-op grad-requiring interpose contract already calls RMSNorm and requires the same
failure.

This audit also exposed an outdated oracle. The wrapper explicitly arms the RMSNorm contract lane, but the selection test
asserted that global dispatch stayed disabled. The assertion now matches the documented implementation, and the autouse
fixture establishes disabled state both before and after every test to prevent order dependence. The two focused files
collect 26 tests instead of 35; 25 pass and the optional SGLang comparison dependency-skips. Repository collection is now
2,471 items and the static inventory is 2,425 definitions with 126 curated decisions.

## Thirty-first wave: test dispatch and export transactions instead of registry snapshots

The thirty-first wave removes an attention check that restated `is_flash_attention` as the same registry-membership
expression used by the implementation. The retained FA4-only reload transaction now proves `flash_attention_2`,
`flash_attention_3`, and `flash_attention_4` registration and mask-family detection directly. Registered eager/native
resolution, non-flash eager fallback, and unavailable-flash rejection now share one resolver-boundary contract. Varlen,
paged-KV, SGLang, FA4, eager-head-layout, and cross-attention rejection paths remain independent.

Quantized export drops a checked-in example's literal path-and-field snapshot; generic config override parsing and the
subprocess CLI export still validate the parser and its real consumer. Existing FP8 scales, MTP config metadata, MTP tensor
namespaces, and unfolded QARL state all continue to construct source directories and fail through one preflight table.
Every tensor split/fusion/remap, sharded index, BF16 island, duplicate-name failure, QARL fold, and block-size failure remains.

The QARL export parity contract no longer trains eight arbitrary iterations against a synthetic target. One real AdamW step
now establishes finite loss and changed target logprobs before folding, block-FP8 export, dequantization, and exact logprob
comparison. The two focused files collect 32 tests instead of 38; 30 pass and two flash-backend tests dependency-skip.
Repository collection is now 2,465 items and the static inventory is 2,419 definitions with 129 curated decisions.

## Thirty-second wave: express API schema examples as conversion tables

The thirty-second wave keeps the API schema as a validation boundary without reporting each literal row separately.
Create-model ZORL, nested adapter-ownership, and create-session ZORL payloads still enter their actual Pydantic request types
and assert the exact field path plus migration message inside one rejection table. Unknown rolling-client and nested LoRA
fields retain their separate forward-compatibility contract.

TensorData conversion now proves rank-one passthrough for model and loss fields, valid rank-two and rank-three recursive
nesting, mismatched-shape fallback, and empty higher-rank fallback in one contract. Exact nested values remain asserted, so
the sequence-length classification invariant consumed by the packer is unchanged. Datum validation, request aliases,
optimizer/session schemas, required response fields, and serialization roundtrips stay intact. The file collects six tests
instead of twelve, and all six pass. Repository collection is now 2,459 items and the static inventory is 2,415 definitions
with 131 curated decisions.

## Thirty-third wave: prefer full EP and API ownership transactions

The thirty-third wave removes the weaker of two identical DeepEP exclusion setups. The retained contract now requires the
all-to-all path and the precise statement that order and rounding differ without asserting an unverifiable mechanism. A
direct private pair-to-slot ordering unit is also removed because the retained full slot-combine transaction routes random
expert pairs, reorders them, performs the weighted reduction, and compares the final tensor with an independent slot-ordered
reference. FP8 exclusion, flag-off stock dispatch, top-k presentation, empty-rank behavior, training autograd, weight-cache
lifecycle, and live gradient parity remain separate.

At the API boundary, current optimizer fields and legacy Adam aliases still reach the orchestrator payload and response
metrics in one transaction. Sampler paths embedded in `xorl://` URIs and plain paths with explicit request model IDs still
load and enter their distinct cleanup-tracking buckets in one ownership contract. Base-model repository IDs, Hugging Face
cache paths in both directions, distinct models, ordinary paths, and `None` likewise form one canonicalization contract.

The four focused files collect 54 tests instead of 59; 53 pass and the live GPU gradient comparison dependency-skips.
Repository collection is now 2,454 items and the static inventory is 2,410 definitions with 134 curated decisions.

## Thirty-fourth wave: collapse FP8 topology shape ladders into mechanism boundaries

The thirty-fourth wave reduces the full-weight FP8 E2E matrix from fourteen expensive subprocess tests to eight distinct
mechanism contracts. The shorter Ulysses run is dominated by the retained longer packed Ulysses run. Three intermediate
hybrid context datasets and a basic hybrid run add sequence length or sample-shape variation without selecting a different
FP8, Ulysses, or Ring branch; the retained 4096-token long-tail multipack case exercises that composition with heterogeneous
near-full bins. The four-GPU DeepEP checkpoint-resume cross-product is also removed: dense FP8 checkpoint restoration and
DeepEP EP/eFSDP FP8 execution remain independently covered.

A live run exposed a stale oracle in the retained baseline. Training completed two optimizer steps and all eight eligible
linears used FP8, while `lm_head` correctly remained unused because Qwen3's resolved numerical program intentionally keeps
the output head in FP32. The helper now requires exactly that production contract instead of demanding an impossible 9/9.
The shared E2E configuration generator also now applies the `extra_data` and `extra_model` mappings already passed by the
retained packed-context and DeepEP cases, rather than rejecting those calls during Python argument binding.

The focused file collects eight tests instead of fourteen. Its one-GPU retained baseline passes live; the remaining costly
multi-GPU matrix was collection-checked rather than executed. Repository collection is now 2,448 items and the static
inventory is 2,404 definitions with 137 curated decisions.

## Thirty-fifth wave: attach loader and cache assertions to their real transactions

The thirty-fifth wave folds GLM-5 architecture-alias registration and loader selection into the retained local Hugging Face
config load. That transaction now constructs the actual `Glm5Config`, checks both supported architecture names and their
class relationship, and verifies the selected checkpoint loader. Two standalone membership and description snapshots are
removed. Configured-layer, MTP-boundary, far-out-of-range, and non-layer checkpoint keys likewise pass through both
normalization and the early disk-read skip hook in one handler contract instead of two copies of the same key table.

Two direct OPD hidden-cache helper examples are removed in favor of their retained consumer transactions. The gathered-SP
writer already filters an interior valid target and asserts its persisted cache index. The multi-rank writer already gathers
local and remote chunks, orders them by logical slice, persists the concatenated tensor, and checks per-sample indices.
Packed-segment splitting, contributor ownership, Mooncake producer/consumer roundtrips, metric collectives, FSDP lm-head
anchoring, and diagnostic artifacts remain independent.

The two focused files collect 53 tests instead of 58, and all 53 pass. Repository collection is now 2,443 items and the
static inventory is 2,399 definitions with 140 curated decisions.

## Thirty-sixth wave: remove the private-helper layer beneath P2P transactions

The thirty-sixth wave keeps all FP8 byte-parity, receiver-layout, cached-prepare, multi-sender, memory-registration,
failure-cleanup, and completion contracts while removing ten lower-value reports from the 78-test P2P protocol suite.
Generic TP slicing is already proved by the retained transfer that writes exact row slices to two receiver pointers; shape
incompatibility is already rejected by a retained full transfer before the engine runs; and unsliced sources pass through
many real transfers. Three direct calls to the private slicer are therefore removed, while every specialized Qwen
linear-attention and FP8 layout transformation stays independent.

HTTP and remote-declared initialization failures now share one failure contract. Implicit all-rank and explicit sender sets
share one capability contract, and list, deep, and forced-reuse locator alias policies share one scatter-copy contract.
Repeated assertions inside dense-owner and filtered-buffer loops are deleted. Flush-cache preservation and weight-version
propagation now reach the actual completion payload together rather than one test stopping at backend configuration.

Finally, real multi-sender initialization already adopts and validates a nonzero rank's scattered tensor map, and the
all-filtered direct-EP transfer already uses a nonzero source rank with a real locator and proves no engine transfer occurs.
Their direct state-assignment and empty-bucket no-exception smokes are removed. The file collects 68 tests instead of 78,
and all 68 pass. Repository collection is now 2,433 items and the static inventory is 2,389 definitions with 143 curated
decisions.

## Thirty-seventh wave: carry endpoint configuration through consumers

The thirty-seventh wave turns worker-port selection into one registration-to-consumer transaction. The explicit worker
registration still checks both control and worker health, then uses the returned endpoint for LoRA load, unload, and loaded
adapter discovery. Two tests that constructed an endpoint by hand only to repeat the final URL are removed.

FP8 receiver detection now reports policy rather than one config literal per test. The retained rich skip-list transaction
already proves default format, dynamic activation, block size, language-model prefix normalization, weight-suffix removal,
and vision exclusion, so a minimal default-dictionary snapshot is removed. MTP, invalid activation schemes, UE8M0 scale
storage, and BF16 MTP form one receiver-admission contract. Compressed-tensors rejection and explicit BF16 no-op behavior
likewise share the sync-quantization setter boundary.

Packing drops an exact strategy-tuple snapshot because every strategy still executes through document, token, position,
capacity, utilization, balance, order, and determinism contracts. A convenience-wrapper smoke is also removed: the retained
full pipeline calls `pack_samples`, asserts packed boundaries, simulates output, and unpacks every sample, while direct packed
and unpacked contracts cover both mode branches. The three focused files collect 62 tests instead of 71, and all 62 pass.
Repository collection is now 2,424 items and the static inventory is 2,380 definitions with 146 curated decisions.

## Thirty-eighth wave: report admission policies, not success and failure fragments

The thirty-eighth wave resolves six no-observable-outcome signals by attaching each success path to its failure policy.
Pipeline schedules now map style, single-stage status, and split-backward behavior in one metadata table, while all admitted
and rejected virtual-stage/microbatch configurations enter one schedule-admission contract. This preserves every schedule
and validation branch while removing three separate reports.

DeepEP's preflight now accepts an identity dispatch/combine roundtrip and rejects corruption after the same setup. Launcher
readiness likewise accepts a set ready event and independently fails fast on worker exit in one lifecycle contract.
Blackwell FP8 policy rejects no override, rejects a missing validation artifact, and accepts an explicit validated override
together. Exact Qwen3.5 MoE admission similarly checks structural defaults before all invalid implementation, dispatch, and
async-combine overrides.

The exact active-LoRA snapshot guard now rejects both full-weight publication paths before downstream work and then proves
ordinary models remain unrestricted, eliminating a standalone absence-of-error test. The six focused files collect 64 tests
instead of 73, and all 64 pass. Repository collection is now 2,415 items and the static inventory is 2,371 definitions with
150 curated decisions.

## Thirty-ninth wave: test FP8 quantizers by numerical path

The thirty-ninth wave removes a weak FP8 output snapshot because the retained independent Slime-reference contract now also
checks CPU placement while already proving emitted names, dtypes, scale shape, exact FP8 bytes, exact scales, and dequantized
parity. A direct contiguous-storage predicate example is also removed: stack-versus-single quantization proves grouping is
numerically transparent, while expert workspace and streaming transactions exercise grouped stacks at their consumers.

Three CUDA stack tests become one target-and-parity contract. The same nontrivial tensor is quantized to CPU and CUDA
targets, target-specific copy telemetry is checked, and both results are compared bitwise with the CPU quantizer. Expert
projection, skip-list, CPU-workspace, streaming, LoRA folding, partial-block padding, and direct-EP cases stay independent.

Two-dimensional DTensor save materialization now proves both all-rank and writer-only results inside one four-rank CPU-mesh
transaction. This removes a repeated process-group launch while preserving the distinct one-dimensional mesh contract. The
two focused files collect 48 tests instead of 53, and all 48 pass, including the live CUDA cases. Repository collection is
now 2,410 items and the static inventory is 2,366 definitions with 153 curated decisions.

## Fortieth wave: replace arbitrary examples with branch-relevant contracts

The fortieth wave removes twelve reports and several uncollected examples across grouped GEMM, MoE primitives, adapter
checkpointing, adapter coordination, and routing replay. A grouped-GEMM example that allocated roughly a gigabyte for one
aligned operand is replaced by a compact unaligned case that actually reaches the kernel's K/N masking path. Generic
single-group repetitions and a separate dtype/shape smoke disappear; FP16 and BF16 numerics, transpose-B, unequal groups,
zero-K handling, noncontiguous input, and device rejection remain in the two real numerical contracts.

MoE primitives no longer repeat the same histogram after flattening, test an invalid overlapping slot map, or separately
smoke gather, scatter, and add-gather before their retained numerical references and roundtrip. The multi-block gather is
smaller but now crosses an unaligned hidden-width boundary, and the full routing transaction checks the exact output rather
than merely checking that it is nonzero. Non-gated backend and activation rejections now share one constructor policy.

Adapter checkpoint save/load containment, missing-tensor rollback, saved dtype, and current learning-rate persistence are
proved at their transaction boundaries rather than by paired fragments. Coordinator load, save, and evicted-model path
traversal share one output-root policy, while direct adapter registration and materialized session registration share one
broadcast transaction. Routing replay combines padded and unpadded sequence-parallel cases, materialized input types, and
nested Qwen top-k discovery with base64 shape inference while retaining packed-document, zigzag, truncation, and NumPy
integration paths.

The six focused files collect 85 tests instead of 97, and all 85 pass, including the live CUDA kernels. Repository
collection is now 2,398 items and the expected static inventory is 2,354 definitions with 158 curated decisions.

## Forty-first wave: express propagation as end-to-end policy

The forty-first wave reduces four previously untouched suites from 51 collected reports to 38. Server optimizer wiring now
has one arguments contract, one initialization contract, one optimizer-step policy, one adapter path, and one dispatcher
boundary. Explicit, default, malformed, partial, omitted, and non-Adam cases still run, but they no longer publish one test
result per dictionary variation.

Runner save-state and save-LoRA handlers now prove the shared nonresident-adapter materialization rule in one transaction.
Uniform-versus-rank-local forward/backward failure semantics still run on rank-zero and worker paths, but no longer create
two parameterized reports for identical policy. LoRA checkpoint tests fold SGLang layout inspection into its actual
roundtrip and exercise hybrid-shared and all-owner expert layouts through one adapter-manager transaction.

Native block-FP8 state construction and dtype application now form one byte-exact lifecycle contract. CPU execution,
explicit materialization admission, and import laziness likewise share one fail-closed contract. Kernel dispatch, gradient
admission, invalid prequantized pairs, state dictionaries, DCP metadata, exception restoration, and FSDP composition remain
independent because they guard different mutation or integration boundaries.

All 38 focused tests pass. Repository collection is now 2,385 items and the static inventory is 2,342 definitions
with 162 curated decisions.

## Forty-second wave: delete test-only proofs and retain production contracts

The forty-second wave reduces four untouched suites from 43 reports to 26. Adapter-gradient ownership configuration,
fingerprint invariance, and replica-domain admission now report policy transactions rather than one result per example. A
self-contained "analytical reference" test is removed entirely: it defined scale/clip/AdamW equations inside the test and
checked them against PyTorch, while the retained adapter-manager transaction already validates the production optimizer
step, parameters, moments, clipping, scratch lifecycle, and publication state against those equations.

DRGRPO now checks seeded forward value, gradient norm, finiteness, and metric schema in one numerical contract. Zero
advantages, fully ignored labels, and empty sequences share one zero-loss boundary; positive-KL admission and effect share
another. Temperature/K3 behavior, advantage direction, and microbatch composition remain independent.

Outbound endpoint allowlisting, DNS pinning, and malformed-target rejection now form one security policy. Generic path
resolution and environment/explicit artifact roots form another. Diagnostic inputs and compile-worker code/protocol
boundaries remain separate. Router GEMM reference, empty input, and dtype admission now share one kernel contract, as do
top-k renormalization, cast-only behavior, and input dtype; batch invariance, backward, leading dimensions, and MoEBlock
consumer paths remain independent.

All 26 focused tests pass, including live CUDA. Repository collection is now 2,368 items and the expected static inventory
is 2,325 definitions with 166 curated decisions.

## Forty-third wave: make instrumentation tests follow event lifecycles

The forty-third wave reduces phase, component, CUDA, and activation-offload instrumentation from 22 reports to 9. Phase
ordering now covers canonical, custom, and empty inputs together. Phase-time and memory summaries each prove empty behavior,
ordering, float normalization, and single-rank aggregate fields through one complete output map.

Direct decoder-name and nested-submodule helper tests are removed. The retained live GLM/Qwen hook transaction discovers
three decoder layers, records present forward/backward components, and omits absent indexer/shared-expert components through
the actual timer consumer. Disabled and malformed manual-CUDA modes share one API policy; successful forward/recompute
events and an unrecorded pair share one drain lifecycle. Activation-offload values and empty-context omission likewise
execute through one consume lifecycle.

All 9 focused tests pass, including live CUDA. Repository collection is now 2,355 items and the expected static inventory
is 2,312 definitions with 170 curated decisions.

## Forty-fourth wave: report checkpoint and sharding selectors as policies

The forty-fourth wave reduces four checkpoint/sharding suites from 28 reports to 19. DCP synchronization now expresses
NCCl-to-Gloo creation/caching and Gloo-default behavior in one backend policy. Metadata synchronization expresses disabled
PP, caller-supplied PP group, and global Gloo fallback in one selector contract.

The meta EP slicing regression now uses the real `already_local=True` skip-loading path while checking local shape, meta
device, dtype, requires-grad, shard metadata, and unrelated replication together. Two malformed exact-GLM dispositions
still execute but no longer generate parameterized duplicate reports. Exact meta allocation, real already-local factor-bank
sharding, reduction metadata, and indivisible geometry remain independent.

A manager active-slot shape smoke is removed because the retained manager registration/forward tests already assert
rank-specific local factor shapes and the sharded-state suite retains pack/unpack, coordinate initialization, discovery,
and a real two-rank uneven DTensor transaction. Checkpoint zero-meta failures now cover pre-load and post-restore stages in
one lifecycle, while initial checkpoint optimizer-default and weights-only modes share one runner policy.

All 19 focused tests pass, including the two-rank Gloo transaction. Repository collection is now 2,346 items and the
expected static inventory is 2,304 definitions with 174 curated decisions.

## Forty-fifth wave: separate MoE TP modes from backend contracts

The forty-fifth wave reduces the largest untouched suite, MoE tensor-parallel simulation, from 14 reports to 9. Environment,
no-EP/TP1 admission, EP rejection, and layer filtering now form one policy. Direct accumulation, BF16 shard reduction,
cache reduction, and a TP1-to-TP2 simulation override run against their corresponding independent references through one
eager-mode matrix instead of rebuilding identical experts and routing data four times.

Carried reshaped shard metadata and flat diagnostic captures now come from the same MoEBlock execution and are both checked
against the reconstructed sum. Triton, Triton-plus-SGL reduction, DeepGEMM, SGLang fused-expert layout, and SGLang runner
contracts remain independent because each invokes a different backend interface. The ordinary MoEBlock consumer also
remains independent of the expert-level simulations.

All 9 focused tests pass. Repository collection is now 2,341 items and the expected static inventory is 2,299 definitions
with 176 curated decisions.

## Forty-sixth wave: require DeepSeek-V4 tests to prove behavior

The forty-sixth wave reduces the untouched DeepSeek-V4 MoE suite from 11 reports to 6. A shared-MLP smoke that explicitly
declined to assert any clamp effect is removed; the retained forced-gate case proves bounded output and now also checks the
forward shape. Routed-expert clamp propagation remains separate because it crosses the MoE expert backend.

Non-hash constructor properties, forward/backward numerics, selection-only bias behavior, and shared-expert contribution
now form one model transaction. Hash-layer table/bias structure, missing-input rejection, table-driven forward, and gate
gradient behavior likewise form one transaction. Hash record-to-replay backward and unknown replay-stage failure remain
independent because they protect replay state transitions rather than ordinary routing.

All 6 focused tests pass. Repository collection is now 2,336 items and the expected static inventory is 2,294 definitions
with 178 curated decisions.

## Forty-seventh wave: turn selection fragments into consumer transactions

The forty-seventh wave reduces FP8 LM-head loss, native-EP combine, DeepSeek-V3 checkpoint conversion, and DR-GRPO runner
coverage from 34 reports to 18. Per-token CE now selects the module and FP32-master paths locally and under TP in one
transaction, while temperature is carried through the primitive and CausalLM consumer together. Importance sampling,
CausalLM FP32 bypass, and TP hidden-gradient reduction remain separate consumer boundaries.

Native Qwen3.5 combine now reports EP8 admission, exact structural flags, and missing trainer-EP rejection as one policy.
Variable token padding and backward unpadding, invalid expert-ID padding, and maximum-row selection likewise share one
collective contract. FSDP pre-forward routing, the serving fused-gate gradient, and full operand capture remain independent.

DeepSeek-V3 external merge, internal fused pass-through, save splitting, and multimodal filtering now form one layout
conversion. Dense and packed EP slicing run through one policy; default and requested packed dtypes share one load
transaction; official layout recognition carries through text quantization-config parsing. DR-GRPO legacy input,
temperature, disabled per-token output, and K3-forced output now share one runner option contract without absorbing its full
dispatch, sampler-boundary, or forward-backward integrations. All 18 focused tests pass.

## Forty-eighth wave: remove analytical echoes and report optimizer/router lifecycles

The forty-eighth wave reduces SignSGD, CausalLM Z-loss, and MoE train-router coverage from 21 reports to 11. One SignSGD
step now proves sign updates, decoupled decay, zero-sign behavior, and missing-gradient preservation. Multiple steps and
state-dict restoration form one stateless persistence lifecycle, while sparse-gradient rejection and optimizer-factory
grouping remain distinct.

The random Z-loss reference transaction now proves both forward values and backward gradients. A zero-logit example that
only restated the test's analytical formula is removed, as is a duplicate CausalLM temperature report already retained at
the LM-head consumer. Coefficient-zero behavior, compiled CUDA parity, and TP rejection remain separate. Train-router
all-to-all gradients and DeepEP rejection now share one dispatch policy; argument/model defaults and balanced
forward/replay routing each report their full policy rather than individual examples. All 11 focused tests pass.

## Forty-ninth wave: join repack and numerical forward/backward contracts

The forty-ninth wave reduces shared-prefix repacking, multi-part optimization, and batch-invariant fused LM-head coverage
from 20 reports to 13. Shared-prefix detection, exact token and position repacking, decoded loss-field preservation, and
output remapping now execute as one end-to-end transaction. No-sharing and one-token-prompt boundaries remain independent
because they select different backend outcomes.

MultiOptimizer construction now proves DCP model mapping and complete parameter coverage together. Its live lifecycle
updates every virtual part, clears gradients, and decays every learning-rate group. Single-part selection and invalid custom
groups remain separate admission outcomes. Fused LM-head forward values and backward gradients now compare against eager
from the same graph for both default and non-unit temperature paths. Determinism, unsupported options, unit-temperature
bytes, and the probability-one clamp remain distinct numerical contracts. All 13 focused tests pass, including all six
live CUDA cases. Repository collection is now 2,303 items and the static inventory is 2,261 definitions with 188 curated
decisions.

## Fiftieth wave: compile configuration and ownership as matrices

The fiftieth wave reduces model-runner builder propagation, exact GLM gradient ownership, batch conversion, and batch-slice
selection from 18 reports to 9. Fail-closed FP8 defaults, every QARL calibration field, and sharded LM-head loss now pass
through one runner initialization policy. Exact GLM block-FP8 QLoRA remains separate because it also resolves the runtime
target-module set, and raw CausalLM token-sum behavior remains a distinct loss boundary.

Exact gate-up, dense-MLP, and absorbed-KV leaves now compile through one module-managed ownership matrix with their
component-specific canonical factor names. The EP16 routed leaf rejects missing managed-FSDP ownership and mutated DeepEP
dispatch in one runtime-admission contract. DR-GRPO logprobs and teacher hidden states share one FP32 batch-conversion
transaction, while ragged padding and sequence sharding stay separate. Finally, FSDP, TP, EP, and legacy duplicated-EP
slices are expressed as one topology selector instead of four examples. All 9 focused tests pass.

## Fifty-first wave: follow LoRA generations and model dtype lifecycles

The fifty-first wave reduces fused-GDN LoRA, DeepSeek-V4 attention, stochastic rounding, and fused MoE expert coverage from
26 reports to 20. Canonical sliced LoRA folding now carries through the GDN output-projection consumer and its gradients.
Cache slice bounds and release of the previous serialized-request generation execute in one parameter-version lifecycle.

DeepSeek-V4's attention sink now proves initial FP32 storage and FSDP marking, BF16 module conversion, and FP32 promotion at
the TileLang call boundary together. Stochastic rounding reports its output metadata and non-FP32 rejection as one API
contract while retaining all statistical and deterministic numerical properties. Base and LoRA expert modules share one
fused gate/up registration contract, and Qwen3/Qwen3.5 handlers share one deferred-QLoRA skip policy while preserving their
family-specific key layouts. All 20 focused tests pass.
Repository collection is now 2,288 items and the static inventory is 2,246 definitions with 196 curated decisions.

## Fifty-second wave: pair numerical values with gradients and policies with consumers

The fifty-second wave reduces batch-invariant GDN, Nemotron-H, exact GLM synchronization, deferred QLoRA loading, trainer
model-program selection, and P2P transfer coverage from 36 reports to 23. GDN gating and gated RMSNorm each compare forward
values and every input gradient with an independent PyTorch composition from the same graph. Nemotron's output shape,
optional router logits, labeled CausalLM loss, and gradients through Mamba, attention, routed/shared MoE, latent, and
embedding paths now form one model transaction.

Exact dense MLP and attention projections reject adapter preparation, collective merging, and raw extraction through one
factor-only synchronization matrix. Exact LM-head ordinary and prepacked sparse-delta publication share one pre-side-effect
guard. QKV and gate/up deferred loader key selection now report one merged-projection policy rather than two parameterized
examples. Exact/ordinary server and non-server trunk engagement, structural numerical-family selection, and P2P size cutoff
selection likewise execute as policies. All 23 focused tests pass, including all five live CUDA GDN tests.

## Fifty-third wave: remove generic serializer examples and report numerical matrices

The fifty-third wave reduces FWHT, runner message protocol, DeepSeek-V3 auxiliary routing, FlashMLA admission, RL
primitives, and families-v2 coverage from 33 reports to 23. A known Hadamard row, orthonormal roundtrip, and norm
preservation now share one width matrix. Typed runner messages, tensor payloads, JSON conversion, and pickle rejection share
one wire-format contract; generic large nested-list and nested-dictionary serializer examples are removed while IDs,
timestamps, optional fields, and ACK behavior remain.

DeepSeek-V3 auxiliary router logits now prove both all-MoE output and omission of a dense prefix together. FlashMLA rejects
CPU dispatch, an unproven head shape, and flattened-address overflow before backend import in one admission policy. All KL
estimator modes still compare against independent Slime formulas but no longer publish four parameterized reports. Exact
and nonexact families-v2 selection likewise share one environment policy. All 23 focused tests pass, including the live
CUDA family-byte and batch-invariance cases.

## Fifty-fourth wave: delete dead placeholders and collapse selector fragments

The fifty-fourth wave reduces routing regather, tensor collation, runner session/load-state behavior, FLOP counting, and
Kimi target resolution from 22 reports to 10. Sqrt-softplus replay regather now proves eager parity, routed scaling,
requested dtype, and cached expert identity together while keeping the softmax regression separate. Tensor container,
scalar, dtype, empty, and string inputs form one conversion policy; variable-length and packed layouts remain independent.

LoRA registry synchronization now follows optimizer and checkpoint-load mutations in one lifecycle. DistSignSGD scaling,
clip suppression, and optional CUDA-cache suppression share one optimizer policy. Load-state preparation carries rank-zero
errors and artifact-root containment together, while multi-adapter and single-tenant dispatch share one routing contract.
Two permanently skipped GLM5 FLOP tests are removed because sparse-MLA/DSA accounting is not implemented; executable CP
length invariance remains. Kimi wrapper defaults, explicit targets, and strict manifests now form one target-source
precedence contract. All 10 focused tests pass.
Repository collection is now 2,253 items and the static inventory is 2,215 definitions with 214 curated decisions.

## Fifty-fifth wave: delete mirrored implementations and follow real lifecycles

The fifty-fifth wave reduces distributed state, model registration/LoRA, side-payload, scheduler, batch-conversion, and
DR-GRPO coverage from 39 reports to 23. Most importantly, two MoE auto-merge files are removed in full: they copied the
parser, buffer, format detector, and simulated loader into the tests and never imported XoRL. Production buffer
transposition/output and family checkpoint loading remain covered by real handlers.

Parallel-state publication now carries automatic shard inference and reinitialization rejection through one singleton
lifecycle. Kimi wrapper conversion includes official auxiliary-loss defaults in one config policy, while tokenizer and
processor fallback share one remote-code security contract. DeepSeek-V4 attention-LoRA type/freeze checks now live in its
forward-backward transaction; the stale generic expert-LoRA claim is gone because DeepSeek-V4 expert semantics are
explicitly unsupported by that generic wrapper.

FIFO defaults and queue operations now share one scheduler policy, and a sleep-based direct state-object smoke is removed
because completion, failure, and abort already run through Scheduler. Missing side-payload keys join typed store
roundtrips, ragged teacher-state padding joins float conversion, and DR-GRPO legacy/options join its loss dispatch. All 30
focused production tests pass, including the retained expert-buffer and family checkpoint-handler consumers.

## Fifty-sixth wave: make codec and topology matrices report once

The fifty-sixth wave reduces six suites from 39 reports to 15. NF4 accounts for the largest change: twenty codebook,
shape, dtype, group-width, accuracy, zero, and layout reports become three codebook, flat-codec, and GKN-codec contracts.
Every 32/64/128 group width still runs, while two 16M-element/14M-element allocation smokes that repeated the same codec
behavior are removed.

A local GKN matrix/reference-MoE proof and a future sparse-MLA KV-major reference scaffold are removed because neither
invoked production. Production expert buffering and eager/native/Triton backends remain, as do sparse-MLA forward,
backward, deterministic, and combined-kernel contracts. EP supported/unknown backends now share one fail-closed table;
four-rank CP, DP, and HSDP lm-head meshes share one topology matrix; and zero/nonzero cast-once cases share one transaction
for each linear and grouped-expert implementation.

All seven live CUDA codec/merge/backend tests pass. The real two-rank EP reduction and all three four-rank lm-head
topologies pass as well; the four untouched production sparse-MLA reports remain collected.
Repository collection is now 2,213 items and the static inventory is 2,183 definitions with 229 curated decisions.

## Fifty-seventh wave: turn router and loss fragments into policies

The fifty-seventh wave reduces TopKRouter, OPD parity, and reducer coverage from 42 reports to 19. Softmax, balanced,
hash, sqrt-softplus, scaling, invalid-input, and config behavior now execute as selector policies while FP32 selection and
the MoEBlock consumer remain distinct precision and integration boundaries. OPD full-vocabulary modes, estimators,
dispatch, policy-gradient admission, and task weighting likewise report complete policies; clamp behavior, stable metric
keys, and ignored-label handling stay independent.

TokenPartial now proves denominator and microbatch composition together. SequencePartial covers dense, packed, and
context-parallel layouts as one composition policy, with empty-input zero behavior retained separately. All 19 focused
tests pass.

## Fifty-eighth wave: execute numerical case matrices without multiplying reports

The fifty-eighth wave reduces seven shared-loss, RoPE, adapter, and shared-prefix suites from 42 collected items to 25 in
this environment without dropping their numerical cases. A fully provisioned FlashAttention 3 environment also avoids
seven extra parameter reports from the shared-prefix dtype/head-shape product; here that whole optional module already
collects as one dependency skip. Legacy reducer identities, paired KL/K3 implementations, and all importance-sampling and
policy-loss microbatch variants now loop inside one semantic contract apiece. Dense and MoE Qwen half-rotate/cast behavior,
Class-B shapes, supported dtypes, and installed EP adapters use the same pattern.

All 14 CPU numerical reports pass, and all 10 EP-adapter reports pass on CUDA. The shared-prefix module is lint- and
collection-safe but its optional FlashAttention 3 interface is unavailable in this environment, so that module remains a
dependency skip rather than a claimed runtime pass.

## Fifty-ninth wave: move machine-specific throughput out of correctness tests

The fifty-ninth wave removes three Qwen3-8B TFLOPS threshold reports. They were two-to-three-minute benchmark jobs gated
by local model-directory presence and hard-coded H100 baselines, but they did not verify that the executing device was an
H100. Their secondary loss-decrease assertion duplicated retained Qwen LoRA/FSDP end-to-end training coverage. Hardware
performance regression belongs in a controlled benchmark lane with explicit machine admission, not the portable pytest
correctness inventory.

Repository collection is now 2,170 items and the static inventory is 2,157 definitions with 236 curated decisions.

## Sixtieth wave: replace existence smokes with production policies

The sixtieth wave reduces eight data, QLoRA, model, distributed, serialization, weight-sync, and optimizer suites from 72
collected items to 56. A direct `pack_parallel` report is removed because it asserted only that output was non-empty;
`PackingDataset` still exercises the production path for sequential and multipack operation, while capacity, coverage, and
allocation are checked independently. A QLoRA package `hasattr` smoke is also removed: trainer/model-builder imports and
real adapter consumers cover the public package, and the clean-interpreter dependency-cycle contract remains.

Default and explicit MLA targets now form one partition policy. CP16 first-rank and padded-tail behavior, both directions
of Muon EP checkpoint resharding plus same-EP identity, replicated/sharded/padded DTensor copies, 1-D/2-D save
materialization, empty/nonempty PP NCCL transfers, and ordinary/Kahan cautious-decay behavior likewise execute as matrices
instead of one report per example. The private `_prod([])` example is replaced by scalar reconstruction through the real
PP receive path.

All 55 focused CPU reports pass, including the two four-rank DTensor materialization workers. The consolidated Muon
checkpoint report separately passes all three four-rank save/load transitions.

Repository collection is now 2,154 items and the static inventory is 2,143 definitions with 244 curated decisions.

## Sixty-first wave: join architecture admission and configuration policies

The sixty-first wave reduces twelve architecture, loader, optimizer, diagnostics, distributed-policy, and timing suites
from 78 reports to 55. DeepSeek-V4 now carries one standard snapshot through Transformers AutoConfig, AutoModel
registration, and XoRL model construction. DeepSeek-V3, Nemotron-H, and Qwen3.5 registry assertions join their real
configuration conversions. A direct `ModelArguments` field-default test is removed because omitted/explicit YAML
serialization, trainer propagation, and runtime resolution already prove the production path.

Gradient-checkpoint method propagation, single/multi-part optimizer selection, and token-diagnostic disabled/ignored/top-k
boundaries now report complete policies. DeepSeek-V4 APE, FP8, and MXFP4 examples become one contract per codec, with
unknown-key behavior moved from the private mapper to the checkpoint handler and its unmapped ledger. DeepEP default and
unsafe-opt-in combine behavior likewise form one admission contract.

FSDP policy coverage accounts for the largest reduction: singleton/sharded/overridden reduce dtypes, supported/rejected
boolean spellings, and backward-only/forward-only/bidirectional/disabled prefetch settings all still run, but thirteen
reports become six policies. Disabled and unrecorded component-timer behavior join one lifecycle while live CUDA hooks stay
separate. All 55 focused tests pass.

Repository collection is now 2,131 items and the static inventory is 2,120 definitions with 253 curated decisions.

## Sixty-second wave: make server configuration admission policy-shaped

The sixty-second wave reduces server argument and weight-sync quantization coverage from 42 reports to 29. R3 payload
transport success and directory rejection now form one admission matrix, exact GLM rank-1 topology covers accepted and
rejected shapes together, and QARL/FP8 full-weight conflicts plus MTP, Mamba, and Nemo source rejection execute as one
fail-closed policy. Nested train and Nemo FP8 aliases now share one normalization contract, while the two unsupported
multi-adapter modes share one rejection policy.

Weight-sync FP8 configuration now reports normalization once across explicit values, defaults, and module-name cleanup.
Unsupported formats, activation schemes, scale storage, module exclusions, and internal unsupported markers likewise
form one invalid-configuration matrix. All original configuration inputs and exact rejection boundaries remain exercised;
the reduction removes report fragmentation rather than behavior.

All 29 focused server configuration reports pass. Repository collection is now 2,118 items and the static inventory is
2,107 definitions with 260 curated decisions.

## Sixty-third wave: consolidate admission and lifecycle policies across subsystems

The sixty-third wave reduces four independent CLI, simulator, numerical-contract, and weight-sync clusters by 16 reports.
Training CLI FP8 aliases now share one normalization contract, while adapter conflicts, missing QARL calibration, mutual
QARL/FP8 exclusion, MTP metadata, and Mamba configuration form one full-weight low-precision rejection policy. Every YAML
payload and exact rejection boundary still passes through `parse_args`.

Simulator filesystem admission now exercises built-in traversal, relative escape, missing defaults, symlink escape, and
unapproved model-metadata paths as one security policy. Exact Qwen3.5 topology admission joins three certified shapes with
ten rejected mutations; its model-scope policy joins dense, MoE, and Hugging Face snapshots with wrong-layer and nearby-
geometry rejection. P2P completion now reports two lifecycle policies covering pending-transfer failure, receiver
suppression, best-effort cleanup, completion payload metadata, completion failure, deregistration, and default cache
behavior.

All 119 focused reports pass, including all 64 remaining P2P backend protocol reports. Repository collection is now 2,102
items and the static inventory is 2,091 definitions with 266 curated decisions. The heuristic candidate count falls from
63 to 61 without adding parse errors or duplicate bodies.

## Sixty-fourth wave: turn server lifecycle examples into endpoint policies

The sixty-fourth wave reduces three session API, request-processing, and adapter-management suites from 91 reports to 66.
LoRA registration now covers full and rank-only overrides plus existing-session refresh as one policy. Reserved checkpoint
creation, per-session isolation, stale replacement, and existing preservation share one persistence contract. Full-weight
default admission and multitenancy/override rejection likewise report once, while kill, checkpoint URI, default-session
protection, and weights-info modes are grouped by endpoint lifecycle.

Request routing cleanup now covers Mooncake success, Mooncake backend failure, and filesystem payloads together. Token
diagnostics cover packed splitting, empty input, and malformed lengths as one decoder policy; packed-row batching covers
global grouping, rank-local deferral, and replay rejection as one batching policy. Adapter management joins abort states,
direct checkpoint-plan admission, malformed checkpoint structures, PEFT filename/sharding compatibility, and dirty/clean/
failed/multi-rank eviction outcomes into their respective policies.

All 66 focused reports pass. Repository collection is now 2,077 items and the static inventory is 2,066 definitions with
279 curated decisions. The candidate inventory remains 61, with no parse errors and the one intentional duplicate-body
group unchanged.

## Sixty-fifth wave: join inference, GLM5, and optimizer-resume policy matrices

The sixty-fifth wave reduces three inference API, GLM5 model, and adapter optimizer-resume suites from 83 reports to 58.
Inference endpoint port selection, FP8 KV-cache admission, sync-pool filtering, quantization admission, cache invalidation,
receiver detection, and receiver enrichment now each report one complete policy rather than one report per input example.
All health routes, payload flags, endpoint epochs, HTTP failures, normalization, and unsupported-receiver outcomes remain.

GLM5 local config security now joins non-JSON and dunder-key rejection. Blocked indexer selection and sparse-MLA reference
coverage each join full-query and query-offset parity, while CPU fallback and unknown sparse-MLA backends form one dispatch
policy. Adapter optimizer resume now joins canonical identity, checkpoint artifact admission, four successful logical
reshards, and the complete invalid-source/no-mutation matrix into four policy reports. Bitwise moment, squared-moment, step,
and resident-state assertions are unchanged.

All 58 focused reports pass. Repository collection is now 2,052 items and the static inventory is 2,041 definitions with
294 curated decisions. The candidate inventory remains 61, with no parse errors and the intentional duplicate group
unchanged.

## Sixty-sixth wave: make handler, FP8 linear, and export coverage policy-shaped

The sixty-sixth wave reduces weight-sync handler, FP8 linear, and quantized-export suites from 73 reports to 44. Receiver
postprocessing, direct-EP ownership, tied parameters, Nemotron conversion, expert gating, compile-wrapper normalization,
and sparse-delta sync now each report one complete handler policy. Every environment override, emitted backend flag,
receiver namespace, tensor split, transport event, rejection, and cache result remains exercised.

FP8 injection now reports core replacement, recipes, and exclusions as three policies. CPU fallback covers numerical,
output-dtype, and fail-fast behavior together; profiler selection joins call caps, explicit rows, and module-specific calls;
block-FP8 GEMM joins block, rowwise, and torch-scaled-mm reference parity. The live CUDA operand profiler, automatic
fallback, padding, correction, and training-step gates remain independent. Offline export similarly joins quantization,
fused-QKV, MLA-A, linear-attention, and QARL fold admission cases without merging the trained-logprob proof.

All 44 focused reports pass, including the available CUDA FP8 gates. Repository collection is now 2,023 items and the
static inventory is 2,012 definitions with 312 curated decisions. The candidate inventory remains 61, with no parse errors
and the intentional duplicate group unchanged.

## Sixty-seventh wave: cross below two thousand with packing and numerical policies

The sixty-seventh wave reduces packing strategies, core packing, Muon, and FP8-MoE suites from 78 reports to 38. Packing
strategy admission now covers invalid settings and every oversized mode together. Cross-strategy document/token/position/
capacity/utilization invariants, balanced-DP behavior, and deterministic datum ordering each report one complete policy.
Core token metadata joins OPD, OPRD, HF shift, vector padding, RL padding, cache views, and nested schemas, while disabled
packing joins all target-preservation and warning behavior.

Muon grouping now executes equal, flattened, transposed, fused, and chunked shapes as one policy; fused gate-up identity
joins gated/non-gated, post-FSDP, and model-family classification. FP8 MoE injection, same-NK forward, same-MN gradient,
scalar-Quack forward, and expert training now report once per semantic boundary while retaining every CUDA numerical case.
The DeepGEMM subprocess remains an independent opt-in backend gate.

Focused validation reports 37 passed and the existing DeepGEMM opt-in skip. Repository collection is now 1,983 items and
the static inventory is 1,972 definitions with 325 curated decisions. The candidate inventory remains 61, with no parse
errors and the intentional duplicate group unchanged.

## Sixty-eighth wave: collapse execution examples into end-to-end policies

The sixty-eighth wave reduces merged-LoRA, API compatibility, and training-simulator suites from 63 reports to 27.
Canonical folding, straight-through gradients, merged linear selection, MoE cache/admission, native-EP routing, and trunk
wrapping now report once per semantic boundary while retaining every exact tensor, gradient, cache-identity, and rejection
check. API session lifecycle, Tinker weights compatibility, worker registration, and optimizer payload/LR resolution use
the same policy shape; heartbeat activity is now deterministic instead of relying on a wall-clock sleep.

The simulator now has eleven durable boundaries rather than 25 narrow scenario reports: topology accounting, observed-log
planning, model metadata, calibration evaluation, calibrated scenarios, topology what-ifs, path security, built-in packs,
analytical ledgers, kernel admission, and whole-simulator validation. Repeated fixtures now feed complete ingestion-to-
decision paths, and all original measured rows, OOM boundaries, extrapolation flags, topology candidates, and analytical
terms remain asserted.

All 27 focused reports pass. Repository collection is now 1,947 items and the static inventory is 1,936 definitions with
344 curated decisions. The candidate inventory remains 61, with no parse errors and the intentional duplicate group
unchanged.

## Sixty-ninth wave: make cache transport and trainer utilities policy-shaped

The sixty-ninth wave reduces teacher-head/cache, Mooncake transport, and trainer utility suites from 41 reports to 14.
Teacher-head persistence now covers direct files, tied embeddings, sharded manifests, and cross-shard views as one policy;
manager residency covers teacher replacement, dtype replacement, and prefetch. Activation selection joins rank-2, rank-3,
layer-slice, host/device, reuse, dtype-reload, async, and bounds cases into selection and admission policies without losing
any tensor comparisons.

Mooncake now reports tensor codecs, hidden transport, activation-cache consumption, metadata admission, and store lifecycle
once each. All four dtypes, rank-2/rank-3 layouts, multi-teacher routing, malformed payloads, missing and mismatched objects,
cleanup keys, and environment precedence remain exercised. Trainer utilities similarly join clipping modes, token/voter
accounting, SP and adapter-owned gradient synchronization, and lm-head TP synchronization while leaving PP chunked CE as
its own numerical boundary.

All 14 focused reports pass. Repository collection is now 1,920 items and the static inventory is 1,909 definitions with
357 curated decisions. The candidate inventory remains 61, with no parse errors and the intentional duplicate group
unchanged.

## Seventieth wave: consolidate checkpoint transport, optimizer, and GDN contracts

The seventieth wave reduces checkpoint loading, cautious optimizer, and GDN convolution suites from 49 reports to 22.
Checkpoint behavior now reports object transport, rank-zero loading, source resolution, expert routing, group fallback,
and strict postprocessing as complete policies. DTensor copying/materialization and expert-key classification remain
independent boundaries; every source name, handler call, transfer, dispatch, fallback, and strict diagnostic remains.

Cautious decay now reports primitive/SignSGD behavior, AnyPrecision decay, AnyPrecision state strategies, Muon behavior,
and builder admission once each. The environment-sensitive DTensor state-offload lifecycle stays separate. All ordinary,
masked, chunked, Kahan, gradient-reuse, Newton-Schulz, fallback, optimizer-family, and kwarg cases retain their numerical
references or exact rejection messages.

GDN CUDA coverage now reports forward, backward, and end-to-end block policies across fixed, variable-length, batched,
deterministic, eager-parity, and gradient cases. CPU coverage reports packed construction/routing, unsupported-input
admission, and exact-contract lifecycle; the optional SGLang tree comparison remains an independent dependency gate.
Focused validation reports 21 passed and that existing optional skip. Repository collection is now 1,893 items and the
static inventory is 1,882 definitions with 373 curated decisions. The candidate inventory remains 61, with no parse
errors and the intentional duplicate group unchanged.

## Seventy-first wave: join checkpoint state, numerical programs, and OPD runner policies

The seventy-first wave reduces model-state compatibility, RoPE/numerical configuration, and OPD runner suites from 52
reports to 21. Checkpoint state now reports reference/QARL buffers, pipeline key unions, LoRA compatibility, load metadata,
load groups, save groups, and optimizer-state filtering as seven complete policies. Every persistent buffer, union key,
load mode, process-group identity, DCP flag, and per-optimizer state selection remains asserted.

RoPE and numerical-program coverage now reports Class-B selection, canonical GLM resolution, and exact Qwen3.5 resolution
once each while retaining independent MoE, topology, and model-scope admission. All ordinary defaults, certified fields,
explicit opt-outs, incompatible overrides, CE modes, and family-specific RMSNorm rejection cases remain.

OPD runner coverage now reports metric aggregation, packed shaping, loss execution, cache contributors, distributed cache
assembly, Mooncake producer/consumer integration, and debug artifacts. Empty-rank collective alignment, per-teacher cache
masking, lm-head anchoring, CP/EP ownership, SP and DP gathering, valid-label trimming, cache indices, gradients, timings,
and JSONL provenance remain exercised. All 21 focused reports pass. Repository collection is now 1,862 items and the
static inventory is 1,851 definitions with 390 curated decisions. The candidate inventory remains 61, with no parse
errors and the intentional duplicate group unchanged.

## Seventy-second wave: consolidate adapter ownership, RMSNorm, and PP profiling

The seventy-second wave reduces adapter coordination, RMSNorm family contracts, and pipeline profiling from 49 reports to
17. Adapter coordination now reports auto-load, explicit load/path admission, rank-zero routing, sharded restore,
transactional load failure, registration, and save admission. Checkpoint/fresh paths, synchronized errors, EP shard bytes,
session/topology mismatches, optimizer rejection, PP rejection, worker failure, and every rollback remain exercised.

RMSNorm now reports family admission, Qwen site declarations, declaration tripwires, family-funnel parity, and module
dispatch. All CPU rejection cases and live CUDA warning, required-family, legacy parity, zero-centered, vitality, trunk,
and three-mode bitwise cases remain. Pipeline profiling reports interval union, schedule formulas, P2P byte accounting,
patch lifecycle, and live GPipe events as five boundaries instead of fourteen examples.

All 17 focused reports pass, including the CUDA RMSNorm policies and the NCCL single-stage GPipe profiler. Repository
collection is now 1,830 items and the static inventory is 1,819 definitions with 404 curated decisions. The candidate
inventory remains 61, with no parse errors and the intentional duplicate group unchanged.

## Seventy-third wave: consolidate arguments, pipeline planning, and Mamba2

The seventy-third wave reduces argument parsing, pipeline-parallel planning, and Mamba2/SSD suites from 41 reports to 12.
Argument parsing now reports optimizer/numerical controls, checkpoint compatibility, FP8 configuration, and low-precision
mode admission. Every packing field, Muon kwarg, legacy alias, resume flag, FP8 alias/default, vLLM rejection, GLM block-
FP8 QLoRA field, QARL normalization, and full conflict matrix remains exercised.

Pipeline planning now reports FQN partitioning, stage placement, and schedule metadata/admission instead of thirteen narrow
examples. Default and Qwen names, single/virtual stages, pinned and weighted splits, Torch-reference ownership, loop/V
placement, all six schedules, and every infeasible layout remain. Mamba2 now reports HF mixer parity, SSD recurrence,
packed sequence behavior, missing-kernel admission, and optional live-kernel parity while retaining all output and gradient
comparisons.

Focused validation reports 11 passed and the existing optional `mamba_ssm` CUDA skip. Repository collection is now 1,801
items and the static inventory is 1,790 definitions with 415 curated decisions. The candidate inventory remains 61, with
no parse errors and the intentional duplicate group unchanged.

## Seventy-fourth wave: remove diagnostic, fused-loss, and DistSignSGD example noise

The seventy-fourth wave reduces token diagnostics, fused selected-logprob, and DistSignSGD suites from 41 reports to 21.
Token diagnostics now report selection/boundaries, KL mapping, log-probability cross-checks, hidden summaries, tensor dumps,
dense hooks, MoE hooks, and trusted override loading. Redundant all-index/component-width examples and a native-MoE hook
case that only pinned internal order constants were removed.

Fused selected-logprob coverage now reports eager numerical parity, input-gradient policy, irregular tail handling,
dispatcher parity, production vocabularies, causal-LM peak memory, and RL-loss integration. Single-row and repeated large-
vocabulary shapes, the unnamed 100000-vocabulary repetition, and the weaker duplicate peak-memory probe were removed.
DistSignSGD now reports update math, sign communication, local/FSDP hook ownership, topology admission, and optimizer
construction; inherited state-dict behavior and an unsupported sparse-gradient example no longer inflate the suite.

All 21 focused reports pass, including seven CUDA fused-loss policies. Repository collection is now 1,781 items and the
static inventory is 1,770 definitions with 429 curated decisions. The candidate inventory remains 61, with no parse errors
and the intentional duplicate group unchanged.

## Seventy-fifth wave: consolidate MoE parity, trunk kernels, and MiniMax support

The seventy-fifth wave reduces SGLang fused-MoE, batch-invariant trunk-linear, and MiniMax M3 suites from 56 reports to 20.
MoE coverage now reports automatic resolution, block dispatch, admission, trainable dispatch, stock-gradient parity,
masked gradients, weight/layout policy, strided adapter behavior, runtime context, strided numerical parity, and real auto-
mode parity. An install-state-dependent import test and a mocked CUDA auto-dispatch example were removed; the retained
real-kernel gate is the stronger automatic-resolution proof.

Trunk-linear coverage now reports selection/admission, forward parity, backward parity, and global-interpose gradient
policy. Projection selection, idempotence, exclusions, both bias modes, persistent/global bitwise parity, batch invariance,
cuBLAS gradients, dtype rejection, and inference-safe interposition remain. MiniMax now reports configuration/registration,
activation/routing, text runtime/admission, checkpoint ownership, and MSA paging/admission rather than twelve narrow cases.

Focused validation reports 16 passed and four existing optional SGLang-kernel skips; every available CUDA trunk policy
passes. Repository collection is now 1,745 items and the static inventory is 1,735 definitions with 444 curated decisions.
The candidate inventory is now 57, including 45 conditional-runtime skips and 12 no-observable outcomes, with no parse
errors and the intentional duplicate group unchanged.

## Seventy-sixth wave: consolidate dispatcher and weight-sync protocol policies

The seventy-sixth wave reduces runner-dispatcher, P2P backend, and FP8 synchronization suites from 109 reports to 68.
Dispatcher coverage now reports DP/EP/CP distribution, diagnostics, routing-payload transport/security, packing/dummies,
per-token merging, completion rendezvous, routing-weight slicing, and row-batch provenance. All rank ownership, dummy,
filesystem/Mooncake, trust-boundary, logical-order, CP-replica, and provenance assertions remain.

P2P preparation now reports payload construction, fanout, cached preparation, completion, Qwen3.5 slicing, and engine
construction as complete policies. The large production transfer-layout and direct-EP matrices were deliberately retained
as independent boundaries. FP8 synchronization now reports core numerical behavior, adapter merging, selector policy,
stack/dtype behavior, CPU expert formatting, workspace lifecycle, GDN folding, and GPU parity rather than 24 examples.

All 68 focused reports pass, including the available GPU FP8 policy. Repository collection is now 1,704 items and the
static inventory is 1,694 definitions with 462 curated decisions. The candidate inventory remains 57, with 45 conditional-
runtime skips, 12 no-observable outcomes, no parse errors, and the intentional duplicate group unchanged.

## Seventy-seventh wave: consolidate adapter lifecycle, GLM5, and expert QLoRA

The seventy-seventh wave reduces adapter-manager, GLM5 support, and expert-adapter QLoRA suites from 85 reports to 43.
Adapter coverage now reports optimizer construction/persistence, gradient-plan admission, capture staging/atomicity,
coordinator checkpoint materialization, and session compatibility as complete policies while retaining independent raw-
capture, epoch, publication, optimizer/collective failure, checkpoint structure, eviction, and mixed-adapter boundaries.

GLM5 now reports configuration/construction, indexer construction/selection, DSA masks, sparse-MLA reference/wrapping,
sparse attention, kv_b adapters, dispatch, checkpoint filtering, LoRA/MoE integration, HF parity, and forward/recompute.
The optional live TileLang and HF-reference gates remain independent. Expert QLoRA now reports backend identity, factor
ownership, semantic preservation, target injection, model-family construction, and fail-closed admission.

All 43 focused reports pass. Repository collection is now 1,662 items and the static inventory is 1,652 definitions with
482 curated decisions. The candidate inventory remains 57, with 45 conditional-runtime skips, 12 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Seventy-eighth wave: consolidate server configuration, exact GLM5.2, and EP serving-kernel contracts

The seventy-eighth wave reduces server-argument, exact GLM5.2, and SGLang EP suites from 72 reports to 34. Server
configuration now reports removed fields, shipped adapters, runtime round trips, R3 transport, quantized training,
parallel topology, unsupported combinations, optimizer/runner compatibility, and model-specific controls. Every prior
YAML/CLI rejection, shipped file, numerical field, alias, topology, conflict, and serialized value remains exercised.

Exact GLM5.2 coverage now reports layer planning, logical selection, Hadamard transport, fused projection, sampler key
preparation, sparse codecs, native-selector runtime, kernel loading, and canonical routing as complete contracts. The
independent topology, IndexShare lifecycle/FSDP identity, checkpoint bias, two semantic MoE-stack depths, and fail-closed
native boundary remain. EP serving-kernel coverage now reports admission, disabled-mode dispatch, enabled presentation,
compute guards, slot combine, trainable dispatch, and weight presentation; the missing-package and live GPU-gradient gates
remain independent.

Focused validation reports 32 passed and the two existing environment-dependent skips. Repository collection is now
1,624 items and the static inventory is 1,614 definitions with 505 curated decisions. The candidate inventory remains 57,
with 45 conditional-runtime skips, 12 no-observable outcomes, no parse errors, and the intentional duplicate group
unchanged.

## Seventy-ninth wave: consolidate exact RMSNorm and native block-FP8 lifecycles

The seventy-ninth wave reduces SGLang fused-RMSNorm, native block-FP8 linear, and GLM5.2 native-FP8 suites from 38 reports
to 16. Fused RMSNorm now reports CPU fallback, forward bit-exactness, backward parity, model integration, and trunk-family
behavior. Residual/no-residual calls, BF16/FP32, packed 3D shapes, module dispatch, dense Qwen, the pre-summed final norm,
the serving residual tree, the aten interpose lane, and every hidden/residual/weight gradient remain exercised.

Native block-FP8 linear coverage now reports encoding, execution/partitioning, admission, checkpoint lifecycle, and FSDP
precision policy. Exact FP8 and scale bytes, protected dtype application, lazy imports, hook traversal, gradient and shape
failures, state/DCP metadata, adversarial apply restoration, and EP-restored global shapes remain. GLM5.2 native-FP8 now
reports configuration, dense pair buffering, model construction, canonical routing, frozen expert execution, and expert
checkpoint ownership rather than twelve narrow steps.

All 16 focused reports pass, including the available CUDA RMSNorm policies. Repository collection is now 1,602 items and
the static inventory is 1,592 definitions with 518 curated decisions. The candidate inventory is now 56, with 45
conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group unchanged.

## Eightieth wave: consolidate Qwen3.5 norms, generic QLoRA, and OPD objectives

The eightieth wave reduces Qwen3.5 norm-family, generic QLoRA, and OPD loss suites from 35 reports to 14. Qwen3.5 now
reports family dispatch, site assignment, bit-exact integration, and the family-two residual equation. Exact/ordinary
coexistence, v1/v2 selection, every zero-centered site, layer-zero/final-norm forcing, module/trunk/layer bytes, and the
sampler BI-mean composition remain exercised.

Generic QLoRA now reports quantized execution, NVFP4 scale/merge behavior, injection, prequantized block-FP8, and optimizer
reset as five lifecycles. Both formats, memory, dequantization, EMA, delta folding, fused QKV, target ownership, gradients,
state rebuild, non-LoRA preservation, and scheduled merge remain. OPD now reports numerical backends, gradients/reduction,
output edges, the hidden-only objective, and OPRD hidden distance. Chunking, streaming/TileLang/low-memory paths, teacher
shards, ignored tokens, per-token results, zero-KL behavior, and fetched layer slices remain checked.

All 14 focused reports pass after correcting a helper-name collision caught by the first focused run. Repository collection
is now 1,581 items and the static inventory is 1,571 definitions with 531 curated decisions. The candidate inventory
remains 56, with 45 conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate
group unchanged.

## Eighty-first wave: consolidate quantized export, model construction, and DSV4 loading

The eighty-first wave reduces quantized-export, FP8/QARL model-builder, and DSV4 checkpoint-loader suites from 34 reports
to 16. Export now reports primitive quantization, CLI behavior, base-directory output, projection layouts, MoE layouts,
admission, and trained-QARL logprob preservation. Subprocess execution, BF16 islands, sharding, fused QKV, MLA-A, linear
attention, GKN/fused experts, every emitted tensor/scale, preflight failure, and exact post-fold logprobs remain exercised.

Training-model construction now reports sharded lm-head threading, full-model FP8 construction, GLM5.2 block-FP8 QLoRA,
quantized-mode admission, and QARL injection/calibration. DSV4 loading now reports translation, FP8/MXFP4 codecs, handler
ownership, and synthetic model loading. All name families, APE inversion, EP filtering/fusion, MTP/unmapped accounting,
window/C4/hash variants, nonpersistent RoPE buffers, and representative bytes remain checked.

All 16 focused reports pass. Repository collection is now 1,563 items and the static inventory is 1,553 definitions with
544 curated decisions. The candidate inventory remains 56, with 45 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-second wave: consolidate exact GLM5.2 QLoRA subsystem contracts

The eighty-second wave reduces the general GLM5.2 block-FP8 QLoRA, fused gate-up, and TP16 lm-head suites from 30 reports
to 15. General coverage now reports the full inventory, EP-local routed banks, exact dense component, fail-closed admission,
and product-mode selection. A standalone partial-edge scale test was removed because its only assertion was already present
verbatim in the full 700-target inventory contract.

Fused gate-up now reports state/loading, effective numerics, gradients, CPU admission, and the live Hopper kernel gate.
TP16 lm-head now reports topology, operands, presentation bytes, surrogate gradients, and its live Hopper gate. All factor
and FP8 bytes, rank/alpha/topology checks, gate/up order, one-time BF16 rounding, branch composition, base-free factor VJP,
mutation safety, TP16 ranges/order/NCCL, sampler strides, vocabulary assembly, custom-autograd saves, and gradients remain.

Focused validation reports 13 passed and the two existing optional SGLang/Hopper skips. Repository collection is now
1,548 items and the static inventory is 1,538 definitions with 555 curated decisions. The candidate inventory remains 56,
with 45 conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group
unchanged.

## Eighty-third wave: consolidate sparse deltas, EP adapters, and DSV4 runtime

The eighty-third wave reduces sparse-delta files, EP backend adapters, and DSV4 model suites from 34 reports to 15.
Sparse-delta coverage now reports source capture, single-file encoding, contiguous sharding, ranked files, and translation
futures. Rank/global manifests, empty shards, trust-root filenames, deterministic indices, malformed inputs, encoded/raw
writers, future tags, rank order, and every output statistic remain exercised.

EP adapters now report registry signatures and native/Triton/Triton-MoE-act FP8 boundaries while retaining independent
score-forwarding and Quack activation checks. DSV4 now reports construction/topology, runtime variants, precision
preservation, and outer gradient checkpointing. C128, ordinary and hash-routed backward, PP rejection, CP wiring, HC
gradient ownership, FP32 carve-outs, registry/direct casts, complex RoPE, and per-layer checkpoint calls remain checked.

All 15 focused reports pass. Repository collection is now 1,529 items and the static inventory is 1,519 definitions with
567 curated decisions. The candidate inventory is now 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-fourth wave: consolidate routing replay, top-k policies, and OPD payloads

The eighty-fourth wave reduces routing-replay, top-k router, and OPD pipeline-payload suites from 28 reports to 16.
Routing replay now reports sequence-parallel layout, RingAttention zigzag layout, weight tensors, and wire decoding.
Padded/unpadded positions, unpacked rows, truncation, all ring ranks, packed-document boundaries, FP32 padding, CP slicing,
shaped/inferred base64, and materialized Python/NumPy/Tensor representations remain exercised.

Top-k routing now reports synthetic balancing, softmax behavior, layer-scoped FP32, sqrt-softplus/noaux, hash routing, and
configuration/scaling. Every tie policy, V4-input isolation, bias/weight distinction, tid2eid ownership, selection, scale,
and config default remains. OPD payload coverage retains chunking, endpoint matching, weight-version verification, and
prepare-worker transitions while reporting shifted payloads and Mooncake teacher-cache transport once each.

All 16 focused reports pass. Repository collection is now 1,517 items and the static inventory is 1,507 definitions with
576 curated decisions. The candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-fifth wave: consolidate adapter persistence and MoE-LoRA semantics

The eighty-fifth wave reduces checkpoint-manager saves, LoRA checkpoint roundtrips, and MoE-LoRA suites from 35 reports
to 20. Checkpoint-manager coverage now reports rank-zero write atomicity,
factor-only snapshot admission, live adapter targets, and collective MoE export. Internal dtype and export-format keyword
probes were removed in favor of real saved-tensor and SGLang roundtrip coverage; the duplicate strict-manifest artifact
test was removed in favor of the public adapter-manager save/validate lifecycle.

Checkpoint formats now report runtime-rank export, PEFT hybrid-shared roundtrip, SGLang shared-outer roundtrip/admission,
both expert-ownership modes, and quantized projection subsets. The unrelated cautious-optimizer example was removed
because the dedicated optimizer suite already owns that routing contract. MoE-LoRA now reports construction/runtime-rank
layout, eager execution/hybrid ownership, zero-delta behavior, cross-backend numerics, MoE injection, and EP router-score
semantics. A redundant nonzero-output example, Qwen subclass repetition, and generic linear-injection errors were removed.

All 20 focused reports pass. Repository collection is now 1,502 items and the static inventory is 1,492 definitions with
589 curated decisions. The candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-sixth wave: consolidate target manifests and quantization admission

The eighty-sixth wave reduces LoRA target-manifest, FP8 compatibility, and NVFP4 fake-quant suites from 25 reports to 11.
Target-manifest coverage now reports the successful injection/coverage lifecycle and one fail-closed schema/runtime policy.
Count, rank, configured-target, unlisted-module, Boolean, and exact scalar-type failures all remain checked without seven
separate reports.

FP8 compatibility now reports NeMo translation, external configuration rejection, Blackwell admission, and BF16 layer-
island resolution/injection. The thirteen vLLM and ModelOpt rejection paths are table-driven, while first/last overlap,
invalid topology, real replacement boundaries, and summary metadata form one layer-island lifecycle. NVFP4 now reports
2D reference/dispatch/STE behavior, input admission, MoE projection STE, expert independence, and per-half gate/up scaling.

All 11 focused reports pass. Repository collection is now 1,488 items and the static inventory is 1,478 definitions with
596 curated decisions. The candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-seventh wave: consolidate packing, schedules, and API tracking

The eighty-seventh wave reduces data-packing, learning-rate scheduler, checkpoint-path, and API-type suites from 31
reports to 22. Packing now reports allocation primitives, sample preprocessing, dataset preprocessing, and the
PackingDataset lifecycle. FFD, binning, rank allocation, position metadata, label filtering, Hugging Face dataset filtering,
and optional eval handling retain all previous assertions without one report per helper.

Learning-rate coverage now reports constant, linear, cosine, and invalid-configuration policies. Both default and custom
linear/cosine traces retain their warmup, monotonicity, midpoint, endpoint, decay-ratio, and floor checks. API coverage now
reports removed/future session fields as one compatibility boundary, sampler reconciliation as one stale/query-failure
policy, and model-scoped tracking plus failed-load atomicity as one session-tracking contract.

All 22 focused reports pass. Repository collection is now 1,479 items and the static inventory is 1,469 definitions with
604 curated decisions. The candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-eighth wave: consolidate routing position, QARL MoE, and RoPE precision

The eighty-eighth wave reduces routing-weight-position, NVFP4 QARL-MoE, and RoPE precision suites from 23 reports to 9.
Routing coverage now reports numerical behavior and configuration resolution as two contracts. Both before/after-down
trees, FP64 outputs and gradients, no-score-gradient execution, mutable/env settings, auto regimes, parity opt-in,
explicit Boolean/string values, and invalid input remain checked.

QARL-MoE now reports conversion, eager execution, and injection as three lifecycles. Parameter identity, idempotence,
non-expert rejection, lossy/disabled execution, restoration, gradients, target selection, metadata, and FP8 rejection all
remain. RoPE now reports registry/frequency precision, contract-lane bytes, CPU cache lifecycle, and exact-architecture
device construction. The optional CUDA device gate remains independent.

All 9 focused reports pass. Repository collection is now 1,465 items and the static inventory is 1,455 definitions with
611 curated decisions. The candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Eighty-ninth wave: consolidate QARL injection, Nemotron EP, and endpoint routing

The eighty-ninth wave reduces generic QARL, Nemotron-H checkpoint, and weight-sync endpoint suites from 22 reports to 14.
Generic QARL retains independent stateful fake-quant, folded-export, injection/admission, and configuration policies. Dense
target/exclusion behavior, parameter names, summary counters, MTP rejection, and Mamba rejection now report as one public
injection lifecycle.

Nemotron-H retains independent HF parity, published-layout roundtrip, and stacked-HF loading reports while topology
validation, skip-key ownership, local expert slicing, skip accounting, and EP-plan classification form one ownership
policy. Endpoint coverage now reports health fallback/failure, init/direct port routing, two-phase transfer, mixed/chunked
flattened transfer, hybrid receiver fencing, and multi-rank direct-format rejection as six protocol boundaries.

All 14 focused reports pass. Repository collection is now 1,457 items and the static inventory is 1,447 definitions with
616 curated decisions. The candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Ninetieth wave: consolidate runner lifecycles, attention paths, and sequence sharding

The ninetieth wave reduces runner session-operation, attention, and sequence-shard collator suites from 28 reports to 18.
Runner coverage now reports save, registration, fatal optimizer exit, publication, gradient abort, and forward-backward
lifecycles. Cross-rank registration failure, successful/failing publication tails, completion ordering, uniform rejection,
and asymmetric failure promotion all remain exercised.

Attention now reports registry/resolution, repeat-KV, fixed and varlen FlashAttention, SGL page-size-one KV cache,
alternate paged/FA3/FA4 selection, and cross-attention rejection. The two existing optional FlashAttention gates remain
environment-dependent. Collation now reports SP primitives, packed-label boundaries, full SP splitting/metadata, and
teacher/DRGRPO side-channel alignment across CP2 and CP16.

Focused validation reports 16 passed and two existing optional FlashAttention skips. Repository collection is now 1,447
items and the static inventory is 1,437 definitions with 624 curated decisions. The candidate inventory remains 50, with
39 conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group unchanged.

## Ninety-first wave: consolidate sparse-delta and BI norm lifecycles

The ninety-first wave re-audits merged-LoRA execution, sparse-delta transport, and families-v2 normalization, reducing the
three suites from 27 reports to 21. The nine merged-LoRA reports remain independent: canonical folding, gradient dtype,
straight-through autograd, dense selection/cache behavior, MoE folding/admission/native-EP routing, and trunk composition
exercise distinct numerical or backend boundaries.

Sparse-delta now reports full, changed, and unchanged streaming updates as one lifecycle, retaining TP path replication,
exact changed-byte indices and values, endpoint metadata, and skip accounting. Per-rank prepacked paths and FP8 KV-cache
metadata now share one publication transaction while preserving unique-file byte accounting. Baseline priming,
initialization, runtime loading, and receiver-failure rollback remain independent.

Families-v2 keeps correctness, batch invariance, run-to-run determinism, fused/split parity, dispatch, and strided qk-norm
as six contracts. The split-kernel guard now executes inside the parity matrix, and shipped-size, threshold, row cutoff,
and tile-basis checks report as one dispatch policy.

All 21 focused reports pass, including the CUDA normalization kernels. Repository collection is now 1,441 items and the
static inventory is 1,431 definitions with 628 curated decisions. The candidate inventory remains 50, with 39
conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group unchanged.

## Ninety-second wave: consolidate clipping, packing, and P2P transfer policies

The ninety-second wave reduces EP gradient clipping, orchestrator packing, and P2P backend protocol suites from 74 reports
to 62. EP clipping now reports local norm behavior, skip-FSDP ownership, public dispatch, and mixed-mesh foreach handling
as four complete policies. Infinity norm, empty and missing gradients, raw EP-local gradients, ordinary fallback, safe
per-tensor clipping, and explicit foreach rejection all remain. The real two-rank reduction/non-finite gate and three-rank
participation-mask gate remain independent.

Packing now reports ordinary, overflow, mixed-length, exact-fit, and off-by-one cases as one capacity lifecycle. NumPy input
normalization joins the existing empty, single, oversized, and missing-input boundary. Packed metadata, disabled packing,
position and label generation, validation, unpacking, and the full pack-to-unpack roundtrip remain separate contracts.

P2P transfer admission now rejects unknown parameters, incompatible receiver shapes, and unsupported source ranks in one
side-effect-free policy. Receiver-memory coalescing covers both distinct and missing handle metadata in one report, while
failure diagnostics cover named tensors and handles, capped samples, omitted counts, and default redaction together. Every
FP8 dequantization topology, direct-EP path, staged CPU/GPU transfer, alignment, completion, and cleanup contract remains.

All 62 focused reports pass, including the live distributed clipping workers and the complete retained P2P topology matrix.
Repository collection is now 1,429 items and the static inventory is 1,419 definitions with 637 curated decisions. The
candidate inventory remains 50, with 39 conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the
intentional duplicate group unchanged.

## Ninety-third wave: consolidate runner compiler and Muon policies

The ninety-third wave re-audits runner LoRA-head compilation, GLM5 support, and Muon optimization, reducing the three suites
from 42 reports to 32. Runner coverage now reports effective LM-head selection, replica topology, unquantized expert
admission, and quantized expert contracts as complete policies. Canonical and legacy head formulas, valid SP/output
replicas, every coverage failure, general-TP rejection, hybrid metadata, all quantized formats, declared-shape drift, and
unsupported EP/eFSDP regimes remain checked. Registration, exact TP16 VJP ownership, staged-capture abort, session-rank
specialization, block-FP8 DeepEP, and the authoritative analytical optimizer step remain independent.

The 13 GLM5 reports are retained unchanged: they map to distinct construction, selector, mask, sparse-attention, adapter,
checkpoint, HF-parity, and recompute boundaries rather than narrow literal variations. Muon now reports builder validation,
Quack backend selection, and fused/Nemotron parameter classification as three complete policies. Algorithm updates,
grouping geometry, standard Newton-Schulz batching, CUDA FP32 compute preservation, SGD fallback, and the tiny Nemotron
training step remain separate numerical or end-to-end gates.

All 32 focused reports pass, including the available TileLang, HF-reference, and CUDA paths. Repository collection is now
1,419 items and the static inventory is 1,409 definitions with 644 curated decisions. The candidate inventory remains 50,
with 39 conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group
unchanged.

## Ninety-fourth wave: consolidate layout, optimizer-resume, and FP8 execution policies

The ninety-fourth wave reduces weight-sync handler, adapter optimizer-resume, and FP8 linear suites from 45 reports to 33.
Weight-sync extraction now reports dense filtering and tied aliases as one ownership policy. DeepSeek and Kimi MLA fusion,
contiguous FP8 views, Nemotron-H conversion, and gated stacked-expert splitting now report as one inference-layout policy;
all name, value, storage, transpose, prefix, and rejection assertions remain.

Optimizer resume now reports canonical identity with live binding, bitwise continuation with its weights-only control,
scheduled and overridden public LR restoration, and successful or rejected logical resharding as four complete policies.
Moment restoration, full checkpoint writing, manifest identity, artifact admission, incomplete-restore atomicity, recursive
snapshotting, and post-mutation collective failure remain independent. Every one- and two-dimensional topology, replica,
hole, overlap, dtype, shape, step, structure, empty-rank, and resident-state case still executes.

FP8 linear injection now combines replacement, recipes, and exclusions. Backend parity now includes automatic warn-once
fallback, and CUDA training now includes float32-output dispatch. CPU fallback, profiler behavior, padded matmul parity,
residual correction, activation2 correction, operand diagnostics, and CUDA numerical execution remain distinct gates.

All 33 focused reports pass, including every retained CUDA FP8 path. Repository collection is now 1,407 items and the
static inventory is 1,397 definitions with 653 curated decisions. The candidate inventory remains 50, with 39
conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group unchanged.

## Ninety-fifth wave: consolidate endpoint, distributed autograd, and fused-MoE policies

The ninety-fifth wave reduces inference endpoints, distributed adapter autograd, and SGLang fused-MoE suites from 35
reports to 26. Endpoint registration now reports discovered TP size and configured sync method as one auto-sync policy;
single-endpoint forwarding and default, named, or unmatched pools now form one routing policy. Port selection, FP8 KV-cache
admission, health fallback, quantization, cache invalidation, receiver detection/enrichment, and method validation remain
independent.

Distributed adapter autograd now reports dense plus sequence-parallel ownership, unquantized EP2 all-to-all, unquantized
four-rank eFSDP, and quantized EP2 all-to-all as four policies. Every original subprocess still runs: dense, SP, all four
unquantized backends, shared-owner and all-owner layouts, projection subsets, Triton NF4, native NVFP4, Quack block-FP8,
structural-zero handling, analytical clipping, and public optimizer parity. Direct-output and the two dependency-gated
DeepEP compositions remain separate topology gates.

Fused-MoE now reports stock and masked trainable gradients together, and strided/transient plus auto/explicit real-kernel
parity together. Mocked dispatch, admission, weight cache/layout, adapter layout, runtime context, and auto resolution remain
separate. Focused validation reports 22 passed and four existing dependency skips; every executable distributed worker
passes. Repository collection is now 1,398 items and the static inventory is 1,388 definitions with 661 curated decisions.
The candidate inventory is now 48, with 37 conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the
intentional duplicate group unchanged.

## Ninety-sixth wave: consolidate canonical-MoE reporting and remove an FP8 plumbing mock

The ninety-sixth wave reduces canonical-MoE, FP8-MoE, and training-simulator suites from 32 reports to 26. Canonical-MoE
now reports adjacent-tree widths as one numerical policy, transport resolution and direct executor rejection as one
admission policy, both world-32 group layouts as one topology policy, and both distributed contributor widths as one
execution policy. Every original contributor width, group layout, transport guard, distributed worker, and byte-exact
packed-EP16 comparison still executes.

FP8-MoE drops a positional-argument mock of the internal Quack autograd call. The retained CUDA TP lifecycle now exercises
the same Triton-grouped backend and non-default block size through real forward, backward, finite-gradient checks, and a
master-weight update. Kernel forward and weight-gradient parity, scalar fallback, DeepGEMM isolation, expert training, and
full injected-model training remain separate numerical or lifecycle gates. The simulator's 11 reports remain unchanged
because ingestion, configuration, Qwen calibration, topology extrapolation, path trust, built-in packs, analytical ledgers,
correctness-gated ranking, and consolidated validation are distinct behavioral boundaries.

Focused validation reports 25 passed and one existing opt-in DeepGEMM skip. Repository collection is now 1,392 items and
the static inventory is 1,384 definitions with 665 curated decisions. The candidate inventory remains 48, with 37
conditional-runtime skips, 11 no-observable outcomes, no parse errors, and the intentional duplicate group unchanged.

## Ninety-seventh wave: consolidate prequantized loading, sequence side fields, and Quack safety

The ninety-seventh wave audits three previously untouched suites and reduces them from 18 reports to 12. Prequantized
checkpoint coverage now reports NVFP4 plus block-FP8 detection as one format policy and dense plus MoE exclusion behavior
as one handler policy. Every nested, flat, config, index, precedence, malformed, missing, wrong-size, passthrough, skip,
shared-expert, and auxiliary-key case still executes.

Packing-concat coverage now reports teacher hidden states and hidden-match weights as one sequence-side-field policy while
retaining their different ranks, exact concatenated values, shapes, and padding. Quack process safety now reports silent
timeouts plus truncated frames as one receive-protocol policy and structural hashes, unsafe objects, and Cutlass dtype
classes as one cache-key policy. PTXAS output isolation and entry-name selection remain independent filesystem/compiler
boundaries.

All 12 focused reports pass. Repository collection is now 1,386 items and the static inventory is 1,378 definitions with
670 curated decisions. The candidate inventory remains 48, with 37 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## Ninety-eighth wave: reuse PP baselines and consolidate NVFP4 and retry lifecycles

The ninety-eighth wave reduces Qwen3 pipeline-parallel, NVFP4 export, and data-retry suites from 16 reports to 10. Pipeline
schedule parity now trains the 1F1B baseline once and compares all three retained schedules against that trajectory. This
removes two entire duplicate baseline training runs while preserving Interleaved1F1B, InterleavedZeroBubble, and
ZBVZeroBubble convergence plus every per-step tolerance check. The distinct PP/FSDP, Muon, and server E2E gates remain.

NVFP4 now reports packed layout, dequantization error, and shared global scale as one tensor policy, while weight-only,
W4A4, and already-quantized directory behavior form one export lifecycle. All tensor dtypes and shapes, fused scales, BF16
islands, metadata, numerical errors, input-scale rules, and re-export rejection remain. Data retry coverage now reports
success, retryable and terminal failures, and exponential, linear, or constant backoff as one policy. Its validation also
exposed and repaired the Hugging Face Hub 1.27 import seam by importing `HfHubHTTPError` from the public errors module.

All four executable CPU reports pass; the six retained GPU E2E reports collect successfully without launching the costly
training jobs in this audit pass. Repository collection is now 1,380 items and the static inventory is 1,374 definitions
with 674 curated decisions. The candidate inventory remains 48, with 37 conditional-runtime skips, 11 no-observable
outcomes, no parse errors, and the intentional duplicate group unchanged.

## Ninety-ninth wave: consolidate model-family lifecycles and remove fake data-loader checks

The ninety-ninth wave reduces OLMo2, Qwen2, cu-seqlen, orchestrator-client, and distributed data-loader suites from 22
reports to 16. OLMo2 and Qwen2 now each report construction plus TP unfusing as one architecture-layout policy and save
plus load as one bidirectional checkpoint policy. Every family-specific norm, bias, fused/split key, strict-load,
hidden-state, and logits assertion remains; OLMo2's independent TP-plan contract also remains.

Server cu-seqlen alignment now includes the SP-owned metadata case, while one ZeroMQ lifecycle covers initial health,
repeated requests, and interleaved operations without starting a duplicate client/engine fixture. Forward-backward,
optimizer, serialization, errors, and lifecycle edges remain separate. The data-loader suite keeps all four reports but
drops two genuinely non-observing blocks: a literal `4 * 3 == 12` assertion and an alleged epoch-consistency check that
only compared two list lengths hard-coded to three. Real partitioning, microbatching, sharding, padding, drop-last,
packed-data, multi-DP, and variable-length behavior remains.

All 16 focused reports pass. Repository collection is now 1,374 items and the static inventory is 1,368 definitions with
681 curated decisions. The candidate inventory remains 48, with 37 conditional-runtime skips, 11 no-observable outcomes,
no parse errors, and the intentional duplicate group unchanged.

## One-hundredth wave: eliminate false no-outcome signals and make acceptance explicit

The one-hundredth wave audits every one of the 11 remaining no-observable candidates. None was an inert test: three joined
`torch.multiprocessing.start_processes` wrappers propagate child assertions, the Muon matrix delegates to an
`assert_success` helper, six H100 frozen-bit gates delegate to a SHA-256 assertion helper, and the exact LM-head optimizer
case is an intentional must-not-raise acceptance boundary.

The audit now recognizes joined multiprocessing as an observable outcome. Muon and BI helpers use assert-prefixed names,
and the scalar optimizer-state acceptance test explicitly asserts the validator's successful result. This removes false
triage noise without weakening or consolidating distinct distributed and numerical gates. The sequence-parallel, exact-DCP,
DTensor materialization, Muon transition, and scalar-state wrappers all pass; the six H100-only golden reports collect
unchanged.

Repository collection remains 1,374 items and the static inventory remains 1,368 definitions with 684 curated decisions.
The candidate inventory falls from 48 to 37: all 11 no-observable signals are gone, leaving only conditional-runtime-skip
reviews, with no parse errors and the intentional duplicate group unchanged.

## One-hundred-and-first wave: stop masking supported-kernel failures as skips

The one-hundred-and-first wave reviews every remaining conditional-runtime-skip candidate and separates honest optional
runtime gates from failure-masking guards. Grouped-GEMM and MoE kernel suites now skip unsupported CPU hosts at their
declared CUDA boundary, but failures importing this repository's own kernel modules fail supported GPU runs. The non-gated
MoE suite likewise imports its core expert implementation normally instead of turning every import-time exception into a
skip.

The environment-dependent SGLang negative test now simulates a missing SGLang package deterministically, so it executes on
both installed and uninstalled environments. Doing so exposed and corrected a diagnostic that named the unrelated
TP-simulation flag. The QARL CPU suite drops one narrow, usually-skipped registry probe that only checked two internal
dictionary entries; its fake-quant numerics, STE gradients, shadow selection, and restoration contracts remain.

All 16 focused CPU and CUDA reports pass without skips. Repository collection is now 1,373 items and the static inventory
is 1,367 definitions with 688 curated decisions. The candidate inventory falls from 37 to 29, all remaining candidates
being explicit optional-library, GPU-capacity, distributed-topology, or backend-availability gates. There are no parse
errors, and the intentional duplicate group is unchanged.

## One-hundred-and-second wave: remove duplicate guards and compose lifecycle seams

The one-hundred-and-second wave removes two cross-file duplicates and consolidates five tiny seam pairs, reducing ten
reports to three. The examples-only personal-path scan is absorbed into the stronger repository-wide hygiene guard, which
now also recognizes Mac home directories and personal data workspaces. A standalone runner-dispatcher model-id test is
removed because the retained request-processor policy checks the same rank-zero forward path plus routed ids, routed
logits, adapter auto-load, and returned session identity.

Weight-version forwarding now runs as one composed handler-to-NCCL-synchronizer path instead of two mocked handoffs. HSDP
deferral reports its enabled transitions and non-replicated rejection together. The emitted MoE inference buffer subsumes a
direct one-element runtime-scaling unit, while SGLang RMSNorm mode selection and exact forced-residual numerics form one
policy. Offline and server index-share failures now share one cleanup lifecycle, retaining both caller-specific assertions.

All eight focused reports pass, including the two stronger retained dispatcher policies. Repository collection is now
1,366 items and the static inventory is 1,360 definitions with 695 curated decisions. The candidate inventory remains 29
legitimate runtime gates, with no parse errors and the intentional duplicate wrapper group unchanged.

## One-hundred-and-third wave: consolidate thin server wrapper reports

The one-hundred-and-third wave reduces three two-report server suites to three complete policies. Importance-sampling
metrics now cover default ratio aggregation and custom TIS extrema together, including weighted means, minima, maxima,
valid-token aggregation, and Python-scalar output. GLM LoRA target resolution now checks raw-HF defaults and explicit
precedence from one model fixture. Remote RPC wrappers now preserve weight-sync timeout and optimizer sparse-delta payloads
in one operation matrix rather than separate single-field reports.

All three focused reports pass. Repository collection is now 1,363 items and the static inventory is 1,357 definitions
with 698 curated decisions. The candidate inventory remains 29 conditional runtime gates, with no no-outcome candidates,
no parse errors, and the intentional duplicate wrapper group unchanged.

## One-hundred-and-fourth wave: consolidate mode, precedence, and session lifecycles

The one-hundred-and-fourth wave reduces four untouched three-report suites from 12 reports to seven. DSv4 RoPE cache
length now reports config default and environment precedence together, while the independent context-parallel short-cache
rejection remains. SGLang JIT and kernel RMSNorm CPU fallbacks now share one numerical matrix covering both exact residual
paths and packed tensors.

A nonresident LoRA session now demonstrates missing-checkpoint preservation followed by evicted-checkpoint promotion on
the same runner; traversal rejection remains separate. QARL calibration input loading now covers valid truncation and
malformed token shapes together, while its persistent calibration-state lifecycle remains independent.

All seven focused reports pass. Repository collection is now 1,358 items and the static inventory is 1,352 definitions
with 702 curated decisions. The candidate inventory remains 29 legitimate runtime gates, with no parse errors and the
intentional duplicate wrapper group unchanged.

## One-hundred-and-fifth wave: merge cross-file modes and delete test-local tests

The one-hundred-and-fifth wave merges the remaining SGLang RMSNorm CPU checks into one mode policy and removes the redundant
one-test JIT module. Global selection, forced-residual FP32 multiplication, JIT and kernel fallback, packed shape, exact
residual values, and global-state restoration all remain. NCCL rendezvous coverage now reports sticky ephemeral rotation
and explicit port pinning in one lifecycle while keeping bind-failure admission separate.

FutureStore drops assertions for response builders defined inside the test file itself. Its production entry defaults,
expiry, terminal states, queue transitions, concurrency, deletion, status, error, and TTL behaviors remain.

All six focused reports pass. Repository collection is now 1,356 items and the static inventory is 1,350 definitions with
705 curated decisions. The candidate inventory remains 29, with no parse errors and the intentional duplicate wrapper
group unchanged.

## One-hundred-and-sixth wave: consolidate utility matrices and expose FP8 imports

The one-hundred-and-sixth wave reduces FQN matcher and block-FP8 suites from eight reports to five. Single, all, and any FQN
matching now form one policy containing every exact, wildcard, grouped-number, indexed, prefixed, empty, first-match, and
invalid-input case. Module path get/set remains independent.

Block-FP8 contiguity and divisibility rejection now execute with shapes, dtypes, scales, and block sizes in one
quantization policy. Imports of this repository's block-FP8 module are no longer caught and labeled as optional feature
absence, and two unused imports are gone. Numerical dequantization and edge/determinism coverage remain separate.

All five focused CPU and CUDA reports pass. Repository collection is now 1,353 items and the static inventory is 1,347
definitions with 707 curated decisions. The candidate inventory remains 29, with no parse errors and the intentional
duplicate wrapper group unchanged.

## One-hundred-and-seventh wave: consolidate source admission and remove an inert compatibility suite

The one-hundred-and-seventh wave combines dataset source resolution across local files, saved directories, hub datasets,
URLs, missing sources, and string or list data files. Exact DCP skip mode now reports valid exact-model deferral and
non-exact rejection together while retaining FSDP deregistration and the no-HF-read guard.

The SGLang sparse-delta compatibility module is removed. Its module-level zstd guard skipped every report because the
production writer has no disk-compression API, while its ordinary receiver apply, checksum, validate-only, and parameter
parity contracts are owned by the stronger retained trainer-to-request-processor-to-SGLang E2E. The retained sparse file
and transport backend suites provide the lower-level encoding and posting contracts.

Focused validation reports 19 passed and one retained external-dependency skip. Repository collection is now 1,351 items,
the static inventory is 1,340 definitions, and there are 710 curated decisions across 357 Python test files. The candidate
inventory remains 29 conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional
duplicate distributed wrapper group unchanged.

## One-hundred-and-eighth wave: compose launcher precedence and DistSign ownership

The one-hundred-and-eighth wave reduces launcher address and override fragments plus DistSign local-hook ownership from
six reports to three. Remote rank-zero discovery and explicit connect-host precedence now share one address policy;
schema-agnostic override parsing and removed-field migration validation share one parsing policy. DistSign hook
registration now verifies both local installation and FSDP-managed exclusion in one ownership lifecycle.

All nine focused launcher and optimizer reports pass. Repository collection is now 1,348 items and the static inventory is
1,337 definitions with 713 curated decisions. The candidate inventory remains 29 legitimate conditional runtime gates.

## One-hundred-and-ninth wave: replace registry and config fragments with composed behavior

The one-hundred-and-ninth wave reduces NVFP4 normalization, non-gated MoE, and server Adam initialization by four reports.
NVFP4 alias defaults, activation override, and every invalid block size now form one normalization policy. A direct relu2
registry-membership probe is removed because exact non-gated MoE forward and gradient parity exercises the production
activation. ServerArguments Adam values now feed ModelRunner initialization in the same test, retaining non-default and
default parameter groups plus malformed-beta rejection.

All 11 focused CPU and CUDA reports pass. Repository collection is now 1,344 items and the static inventory is 1,333
definitions with 716 curated decisions. The candidate inventory remains 29 legitimate conditional runtime gates.

## One-hundred-and-tenth wave: test protocol behavior instead of model field echoes

The one-hundred-and-tenth wave reduces API and runner protocol coverage from 11 reports to six. Two Pydantic reports that
primarily echoed constructor fields and automatic required-field behavior are removed; real training, checkpoint, sampler,
and session endpoint tests exercise those models, while the unique forward session-id alias remains in the compatibility
policy. Runner protocol constructor, UUID, timestamp, and optional-field fragments are replaced by one full typed-wire
equality contract covering every retained payload, success and error responses, tensors, JSON, ACK correlation, and pickle
rejection. A second API-orchestrator flow that repeated the adjacent roundtrip, builder, validator, and streaming checks is
also removed.

All six focused reports pass. Repository collection is now 1,339 items and the static inventory is 1,328 definitions with
719 curated decisions across 357 Python test files. The candidate inventory remains 29 conditional runtime gates, with no
no-outcome candidates, no parse errors, and the intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-eleventh wave: exercise policies through their production consumers

The one-hundred-and-eleventh wave reduces checkpoint, DeepEP, weight-sync quantization, and request-processor coverage by
eight reports. Legacy and PP-parent EP meshes are now selected through the production restore operation; malformed named
dimensions remain a separate rejection boundary. DeepEP overflow, no-RDMA allowance, byte alignment, and default sizing
form one buffer policy. BF16 no-ops and valid FP8 normalization form one supported policy, while all unsupported or
malformed forms share one rejection policy.

The direct teacher-sort helper probe is replaced by an OPD model-pass transaction. Nested teacher ids and top-level
precedence drive actual sorting, then compose with packer datum order and Mooncake routing-payload order. All 27 focused
reports pass. Repository collection is now 1,331 items and the static inventory is 1,320 definitions with 723 curated
decisions.

## One-hundred-and-twelfth wave: remove weaker helper probes and relocate hardware timing

The one-hundred-and-twelfth wave reduces SignSGD, SGLang fused-MoE, and sparse-MLA coverage by four reports. SignSGD now
reports dense updates, decay, missing gradients, and sparse rejection as one step policy; a separate assertion of base
PyTorch Optimizer state-dict behavior is removed. The direct FP32 routing-scale helper probe is removed because the retained
fused-MoE backward oracle uses the same discriminating values and compares every gradient exactly.

Sparse-MLA's production-shape H100 speed ratio moves out of pytest into
`certification/glm52/benchmark_sparse_mla_backward.py`. The explicit benchmark retains combined and split warmups, median
timing, environment restoration, and a configurable speedup gate. All three forward/backward numerical kernel reports
remain and pass; nine focused reports pass in total. Repository collection is now 1,327 items and the static inventory is
1,316 definitions with 726 curated decisions.

## One-hundred-and-thirteenth wave: stop running an unasserted benchmark after correctness

The vocab-parallel CE distributed test previously continued after eager and compiled value/gradient parity to run 100
production-scale warmup and timed iterations. Those iterations only printed latency and peak-memory tables and could not
fail the test. They now live in the explicit `certification/benchmark_vocab_parallel_ce.py` torchrun script, while the
two-rank pytest worker ends after numerical correctness.

The retained distributed report passes, and both certification scripts pass lint and compilation checks. Repository
collection remains 1,327 items, the static inventory remains 1,316 definitions, and there are 727 curated decisions across
357 Python test files. The candidate inventory remains 29 legitimate conditional runtime gates, with no no-outcome
candidates, no parse errors, and the intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-fourteenth wave: replace smokes and protocol fragments with stronger behavior

The one-hundred-and-fourteenth wave removes two expensive training smokes already contained in stronger E2E paths. Dense
one-GPU FP8 training remains covered by the checkpoint-and-resume lifecycle, including the original two-step metrics and
module-usage assertions. The basic two-GPU DistSign run is subsumed by the retained FSDP2, Ulysses, DP2, and accumulation
composition. That survivor now uses eager attention, matching the repository's other tiny Ulysses topology tests and
avoiding an unrelated FlashAttention compiler failure on the synthetic shape.

Nemotron-H packed boundaries now flow through the all-mixer loss and backward contract instead of a finite-output smoke.
Qwen3 tensor-parallel unfusing drops a comparison between independently initialized MLP output shapes and retains direct
plus model-wide projection ownership. The PP NCCL mocks now form an actual sender-to-receiver roundtrip rather than two
manually disconnected halves.

All eight retained covering reports pass, including FP8 checkpoint/resume and the four-GPU DistSign E2E. Repository
collection is now 1,322 items and the static inventory is 1,311 definitions with 732 curated decisions.

## One-hundred-and-fifteenth wave: remove duplicate norm proof and synthetic one-rank shards

The families-v2 dispatch suite drops its second fused-versus-split bitwise comparison. The retained norm contract forces
both implementations, verifies that split execution actually occurred, and covers more hidden sizes plus residual, plain,
and zero-centered forms; dispatch selection at shipped and deep shapes remains independently checked.

Three exact-GLM component suites no longer launch an additional one-rank FSDP2 process before their real two-rank shard.
The dense MLP, absorbed kv_b, and generic TP1 QLoRA workers execute every common lifecycle, byte-parity, ownership, and
gradient assertion at world size two, where they additionally verify genuinely sharded factor storage. The retained norm
gate passes. The three distributed GLM wrappers collect and reach their explicit optional-dependency gate, then skip
because SGLang is not installed in the repository venv; the removed wrappers had the same gate.

Repository collection is now 1,318 items and the static inventory is 1,307 definitions with 734 curated decisions across
357 Python test files. The candidate inventory falls from 29 to 26 conditional runtime gates because the three redundant
optional-SGLang wrappers are gone. There are no no-outcome candidates, no parse errors, and the intentional duplicate
distributed wrapper group is unchanged.

## One-hundred-and-sixteenth wave: replace helper probes with production consumers

The one-hundred-and-sixteenth wave removes 13 reports across numerical ops, model loading, checkpoint loading, shared
prefix packing, and server optimizer steps. Families-v2 dispatch now spies on actual fused-versus-split runtime selection
inside the retained norm policy; its redundant helper-only dispatch module and duplicate environment-variable rollback
report are gone. Dense Qwen3.5 RMSNorm construction and site assignment are expressed as two complete policies instead of
seven fragments, and a direct rotary-helper comparison is removed because retained dense and MoE attention projections
exercise the same reference while rejecting the wrong rotation.

Qwen3.5 dense and MoE config conversion now runs through local config files and the production auto-config loader. Its MTP
skip patterns now run inside grouped dense/expert checkpoint loading instead of a direct regex probe. Optimizer-step
learning-rate precedence drives actual API payloads, including both legacy fallbacks, rather than a private resolver.
Finally, the one-line no-shared-prefix fallback joins the complete repack/remap policy.

All 15 focused CPU, server, and CUDA reports pass. Repository collection is now 1,305 items and the static inventory is
1,294 definitions with 742 curated decisions across 356 Python test files. The audit still contains only 26 legitimate
conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed
wrapper group unchanged.

## One-hundred-and-seventeenth wave: replace field echoes and helper branches with complete policies

The one-hundred-and-seventeenth wave removes 13 reports across active-LoRA admission, BI routing and loss, trainer timing,
API models, optimizers, collators, OPD scripting, and sequence parallelism. Active-LoRA truth-table cases and topology
admission now form two complete policies. Router batch invariance now runs through `MoEBlock.route`, including logits,
selections, and weights, while exact and ordinary dispatch share one production policy. Standard and temperature BI fused
LM-head forward/backward comparisons now use one common eager oracle.

Local phase and memory summaries plus multi-part optimizer selection and updates each become coherent lifecycles instead
of narrow reports. The optimizer/weights Pydantic field-echo report is removed; its unique legacy session-id behavior now
drives the real optimizer endpoint. Tensor-collator coverage replaces a flat pseudo-packed example with the actual
already-batched dict and nested packed-dataset structures. A slicing-comprehension OPD probe and the one-line
sequence-parallel no-group identity probe are removed, while the behavioral pipeline and distributed contracts remain.

All 28 focused CPU, server, distributed, and CUDA reports pass. Repository collection is now 1,292 items and the static
inventory is 1,281 definitions with 751 curated decisions across 356 Python test files. The audit still contains only 26
legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate
distributed wrapper group unchanged.

## One-hundred-and-eighteenth wave: remove tautologies and follow data through real consumers

The one-hundred-and-eighteenth wave removes 14 reports across numerical MoE backward, router training, data preparation,
BI GEMM configuration, EP adapters, routing replay, OPD loss, checkpointing, and distributed state. A direct FP32
grouped-GEMM helper probe is removed because exact local and EP custom-autograd oracles already exercise and discriminate
that accumulator. Train-router defaults now feed `MoEBlock.from_config` and prove detached-gate behavior by backward,
alongside the enabled all-to-all path and DeepEP rejection.

Packing cache identity moves from a string-interpolation test into the real `PackingDataset` cache-path lifecycle and now
covers the ring-attention alignment suffix that the old test missed. A one-line `hashlib` wrapper report is removed.
Two BI GEMM reports are also removed: one set environment variables only after module import and therefore could not test
its claim, while the other asserted that lookup returned the constant it directly injects. The retained CUDA policies
prove table bit neutrality, cross-bucket row invariance, and available-backend parity instead.

Native, Triton, Triton MoE-act, and Quack adapter argument boundaries now form one capability-aware matrix. Routing wire
decode composes with float-weight construction and sequence-parallel slicing. OPD output edges, DCP process-group
selection, ParallelState construction, and EP LoRA initialization-to-slicing each become coherent policies rather than
adjacent fragments.

Focused validation reports 30 passed and one legitimate unavailable-DeepGEMM skip. Repository collection is now 1,278
items and the static inventory is 1,267 definitions with 762 curated decisions across 356 Python test files. The audit
still contains only 26 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the
intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-nineteenth wave: compose quantization configuration into execution

The one-hundred-and-nineteenth wave removes 10 reports across FP8 configuration, QARL calibration and execution,
stochastic rounding, and API response projection. Supported NeMo FP8 translation and every unsupported external runtime
form now share one compatibility policy. NVFP4 normalization no longer ends at dictionary-field assertions: aliases,
activation selection, and invalid group sizes feed the retained `QARLLinear` forward and STE contract.

Activation fake-quant forward values and straight-through gradients form one autograd policy. The QARL MoE shadow context
now covers backend admission, no-op modes, and exception restoration together, while activation-quant overrides cover
mixed prior state, both directions, exceptions, and nesting in one lifecycle. Calibration follows parsed and truncated
input through persistent metadata and state restoration. QARL sync configuration similarly flows from an injected model
through selective quantization into the production weight-sync handler.

The real Triton QARL MoE report now compares enabled lossy quantization and gradients with the exact disabled passthrough
using the same seeded model and routing. Seeded stochastic-rounding reproducibility joins its call contract while unbiased
expectation and adjacent-neighbor properties remain independent. API auto-load information and executor timing fields now
share one response-projection policy; duplicate legacy optimizer-payload coverage is removed from the focused telemetry
test because the broader optimizer policy already owns aliases, Adam fields, defaults, and precedence.

All 17 focused CPU, server, and CUDA reports pass. Repository collection is now 1,268 items and the static inventory is
1,257 definitions with 772 curated decisions across 356 Python test files. The audit still contains only 26 legitimate
conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed
wrapper group unchanged.

## One-hundred-and-twentieth wave: follow requests and batches through complete lifecycles

The one-hundred-and-twentieth wave removes eight reports across request processing, scheduling, packing, distributed data
loading, and orchestrator integration. RequestProcessor readiness and counters now surround three real model passes in one
start-to-stop lifecycle. Scheduler coverage now uses production requests for FIFO, capacity, terminal states, statistics,
clear, and bounded history; constructor constants, repr strings, and raw deque probes are gone. This consolidation also
corrects a misleading running-abort check that had dispatched an older FIFO request and then aborted the named request
while it was still pending.

A GPU-marked micro-batch report is removed because it had no rank or distributed behavior and duplicated the retained CPU
contract. Packed label generation and position resets now sit inside the pack-to-unpack roundtrip, while exact sequential
legacy layout begins the all-strategy correctness policy. A second mixed-oversized skip assertion is removed from generic
edge cases because admission policy already owns it.

Finally, orchestrator statistics now describe the retained forward, optimizer, and health transactions. A separate report
that created requests only to move counters, called getters repeatedly to prove they were read-only, and checked that
private methods were callable is gone. Its abort fragment is also removed: the immediate dummy backend usually finished
before abort arrived, and the test asserted only that the original request emitted some output.

All 35 focused CPU, server, data-loader, and orchestration reports pass. Repository collection is now 1,260 items and the
static inventory is 1,249 definitions with 781 curated decisions across 356 Python test files. The audit still contains
only 26 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional
duplicate distributed wrapper group unchanged.

## One-hundred-and-twenty-first wave: make transfer and restore modes observable

The one-hundred-and-twenty-first wave removes 11 reports across weight synchronization and checkpoint restore. P2P warm
mode no longer ends at a private boolean selector: cached prepare drives the fake Mooncake engine's async API while cold
prepare with the same setting stays synchronous. Small-entry chunking and persistent registration likewise move from
direct helper calls into repeated GPU-direct `transfer_bucket` transactions. The engine observes two-plus-one chunking,
one registration across both buckets, and deregistration during destroy. Direct-EP capability field echoes are removed;
the retained suite uses implicit and explicit sender maps through filtered scatter, dense partitioning, collective
failure, prewarm, and rank-owned transfer policies.

Weight-sync bucket defaults and environment precedence now sit beside actual byte-cap splitting. Cache metadata is
asserted after complete streaming FP8 and sparse-delta syncs instead of through a standalone dictionary normalizer.
Compile-wrapper name cleanup joins the inference-layout unfusion policy and verifies emitted names and tensors across
experts, broadcast, and Qwen linear attention.

Checkpoint restoration now follows one EP state through mesh admission, the `ModelState` caller, and final DTensor
construction for legacy and pipeline-parent layouts. CheckpointManager materialization includes successful counter and
optimizer policy plus both zero-meta failure boundaries. ModelRunner's completion flag now surrounds the real initial
checkpoint load for optimizer-enabled, optimizer-disabled, and failing cases rather than a separately mocked wrapper.

All 58 retained weight-sync and checkpoint reports pass. Repository collection is now 1,249 items and the static
inventory is 1,238 definitions with 791 curated decisions across 356 Python test files. The audit still contains only 26
legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate
distributed wrapper group unchanged.

## One-hundred-and-twenty-second wave: keep one authoritative API transaction

The one-hundred-and-twenty-second wave removes eight reports across session, sampler, inference, and future APIs. Three
copies of canonical-LoRA sampler export become one authoritative checkpoint-path transaction that checks normalized
session metadata, `save_lora_only`, model identity, output path, and returned URI. Create-model admission now handles
conflicting recreation, a distinct base repository, and cross-rank registration rollback together. HF snapshot identity
is no longer tested by calling a private canonicalizer: a bare client repository ID successfully registers against a
cache-resolved server path, while a different repository is rejected through the same endpoint.

Sampler listing, path resolution, recency tracking, and last-receiver removal now form one adapter lifecycle. Receiver
quantization similarly follows `config.json` detection through name normalization, per-call skip-list enrichment, and
accepted default configuration; MTP, static activation, UE8M0, compressed-tensors, and BF16 boundaries remain in that
policy.

Finally, a `FutureEntry` report that constructed fields and manually assigned every terminal enum is removed. Real store
jobs already reach pending, processing, completed, failed, and expired states, and queue pause state now follows actual
concurrent processing and statistics.

All 32 retained API and FutureStore reports pass. Repository collection is now 1,241 items and the static inventory is
1,230 definitions with 797 curated decisions across 356 Python test files. The audit still contains only 26 legitimate
conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed
wrapper group unchanged.

## One-hundred-and-twenty-third wave: preserve scenarios and collapse trainer helper echoes

The one-hundred-and-twenty-third wave removes five reports across DeepSeek construction, Trainer bootstrap, manual CUDA
timing, and LoRA dtype selection. DeepSeek router rejection and successful freezing now form one builder policy, while a
direct tensor-parallel validator probe is removed because the retained `build_parallelize_model` policy reaches that same
guard through its production consumer. Trainer bootstrap now covers eager and Quack-linear causal-loss configuration in
one setup rather than rebuilding the full fixture solely to omit `lm_head_fp32`.

Manual CUDA timing now follows one disabled-to-enabled lifecycle through invalid-mode rejection, forward and recompute
recording, unrecorded-event omission, and draining. The mixed-precision LoRA builder already observes BF16 base weights,
FP32 adapters, and generic-upcast suppression, so the adjacent helper policy keeps only its distinct QLoRA,
explicit-skip, and dense-default branches. The experiment simulator suite is unchanged: its reports exercise separate
scenario ingestion, calibration, trust, topology, ledger, and correctness-gate behavior rather than scale-only variants.

All six retained focused trainer and timing reports pass. Repository collection is now 1,236 items and the static
inventory is 1,225 definitions with 802 curated decisions across 356 Python test files. The audit still contains only 26
legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate
distributed wrapper group unchanged.

## One-hundred-and-twenty-fourth wave: replace P2P scale certification with protocol branches

The one-hundred-and-twenty-fourth wave removes nine reports from the largest remaining weight-sync file. Expert FP8
receiver coverage no longer performs production-sized local-shard allocation plus an eight-rank sweep merely to enumerate
all 256 global expert indices. Global names use a size-independent EP-rank formula, so two retained transactions now cover
the actual boundaries: partial blocks, block-128 quantization, one and two receivers, a nonzero EP offset, exact bytes and
scales, and dequantized parity. This reduces eleven expert transfer transactions to two. A separate 2048-by-512 Qwen3.6
shared-expert report is also removed because the retained shared-expert transaction already covers both namespace
spellings, fused gate/up placement, down placement, scales, and the passthrough gate.

Receiver-name compatibility now follows compiled-name stripping, `language_model` prefix fallback, and a missing tied
`lm_head` through one transfer policy. Direct-EP dense ownership applies one assignment to both receiver manifests and
outgoing buffers, including fused/split projection aliases and expert exclusion. Nonzero-sender initialization covers
default and prewarmed engine ordering in one policy, and rank filtering covers both owning-rank routing and a fully
filtered non-owner bucket.

All 34 retained P2P protocol reports pass. Repository collection is now 1,227 items and the static inventory is 1,216
definitions with 808 curated decisions across 356 Python test files. The audit still contains only 26 legitimate
conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed
wrapper group unchanged.

## One-hundred-and-twenty-fifth wave: move norm arithmetic into model consumers

The one-hundred-and-twenty-fifth wave removes seven static reports across Qwen3-MoE residual handling and cross-engine
RMSNorm. Layer-zero, later-layer, and final-model norm declarations now form one Qwen construction policy. A direct
TP-shard materialization probe is gone: discriminating BF16 shard values now flow through the decoder layer and verify
sum-before-residual association at the materialized input, norm call, returned residual, and diagnostic sites. Likewise,
the two O-projection residual association modes now run through `_pre_mlp_forward`; the former private-helper report is
subsumed by exact norm-input, output, residual, capture, and bit-difference assertions.

Cross-engine RMSNorm funnels now execute inside their supported site policies. Q/K, pre-summed residual-tree, and fused
post-attention paths each compare the real XoRL module with the corresponding SGLang kernel and SGLang family funnel over
every retained adversarial shape. The rare family-difference discriminator is part of the Q/K policy instead of a
standalone test of the test. A nominal trunk-flag report is removed because its fixture explicitly declared the
no-residual family, which selected the same wrapper before the trunk flag could affect dispatch.

All five retained Qwen3-MoE reports pass. The five-report cross-engine file passes lint and collection parsing but remains
module-skipped because the optional SGLang package is unavailable in the repository venv. Repository collection is now
1,224 items and the static inventory is 1,209 definitions with 813 curated decisions across 356 Python test files. The
audit still contains only 26 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and
the intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-twenty-sixth wave: distinguish distributed outcomes from mocked rank labels

The one-hundred-and-twenty-sixth wave removes three GPU- and distributed-marked data-loader reports that never launch a
process, use a GPU, or observe a distinct distributed result. The partitioning report injected a mocked sampler but proved
non-overlap over rank-index lists constructed by the test itself. Its micro-batch and sequence-parallel fragments duplicated
the retained CPU policies that check exact split values, sampler ownership arguments, and collator insertion.

The standalone sequence-sharding report asserted only equal output lengths and a loose padding range across mocked ranks.
The retained `TextSequenceShardCollator` policies discriminate exact rank slices, non-divisible padding, labels, attention
metadata, and token-aligned side channels. The packed report likewise changed mocked DP ranks but asserted the same shape;
real packed and variable-length samples already traverse the production data loader, while the collator policy owns exact
values, position resets, padding, extra fields, and flash-attention metadata. Its one distinct boundary, dropping an
incomplete loader batch, now runs inside the retained production data-loader lifecycle.

All 11 focused data-loader and collator reports pass. Repository collection is now 1,221 items and the static inventory is
1,206 definitions with 816 curated decisions across 355 Python test files. The audit still contains only 26 legitimate
conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed
wrapper group unchanged.

## One-hundred-and-twenty-seventh wave: keep adapter transitions and GLM composition, not repetitions

The one-hundred-and-twenty-seventh wave first audits the remaining data-preparation reports and leaves them intact: each
maps to a distinct packing algorithm, preprocessing, persistence, source-routing, file-lock, retry, or cache-identity
lifecycle. It then removes two redundant reports from adapter management and the GLM semantic stack.

A direct manager report that loaded a SignSGD checkpoint into an AdamW-default manager asserted only the resulting optimizer
type and session field. The retained real `AdapterCoordinator` lifecycle already performs the same checkpoint-driven
selection through both explicit load and eviction auto-load, and the multi-adapter lifecycle reloads mixed optimizer types.
The GLM semantic stack now keeps only its four-layer transaction: the one-layer row changed repetition count but no branch.
Four layers still prove every canonical boundary, final-logprob parity, batch permutation, per-row composition, and a
negative discriminator where skipping canonicalization at the first layer changes the final logprobs.

All three focused adapter and GLM reports pass. Repository collection is now 1,219 items and the static inventory is 1,205
definitions with 818 curated decisions across 355 Python test files. The audit still contains only 26 legitimate conditional
runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed wrapper group
unchanged.

## One-hundred-and-twenty-eighth wave: replace shape smoke and synthetic architecture stubs

The one-hundred-and-twenty-eighth wave audits every remaining pytest parameter matrix. Those rows now mostly switch real
backends, checkpoint APIs, tensor families, or compiled kernel specializations, so they remain. A thin-wrapper and
shape-only scan instead removes two weak model reports.

The Qwen Triton expert smoke initialized random weights, ran one forward, and asserted only that output shape equaled input
shape. The retained eager-versus-Triton MoE transaction uses the same backend while comparing numerical outputs and every
LoRA factor gradient. A separate synthetic DeepSeek-like module duplicated the real model's default MLA LoRA targets. Its
unique explicit-target partition case now runs through `inject_lora_into_model_with_moe` on a real tiny DeepSeek model and
also proves the untargeted output projection is left untouched; the five-linear stub module is gone.

Both retained covering reports pass. Repository collection is now 1,217 items and the static inventory is 1,203 definitions
with 820 curated decisions across 354 Python test files. The audit still contains only 26 legitimate conditional runtime
gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-twenty-ninth wave: make configuration change execution

The one-hundred-and-twenty-ninth wave removes two constructor-only reports. A base MoE expert test instantiated Qwen wrappers
and four backend variants but asserted only stored strings, parameter shapes, LoRA mapping identity, and registry membership.
Retained eager, native, Triton, non-gated, injection, and model-construction policies execute those same registrations and
layouts. The separate LoRA initialization policy remains because frozen base weights and trainable zero-initialized factors
are not observable from forward parity alone.

DSv4 KV-QAT coverage no longer stops at a helper boolean and the private `_kv_qat_enabled` field. The retained C0 attention
forward/backward transaction now supplies an FP8 quantization configuration, observes the QAT call on the exact no-RoPE KV
slice with block size 64, and then checks finite forward output plus input and parameter gradients. The C128 branch remains
in the same shape-and-gradient policy.

All five focused MoE and DSv4 reports pass. Repository collection is now 1,215 items and the static inventory is 1,201
definitions with 822 curated decisions across 354 Python test files. The audit still contains only 26 legitimate conditional
runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate distributed wrapper group
unchanged.

## One-hundred-and-thirtieth wave: make cache and checkpoint metadata prove execution

The one-hundred-and-thirtieth wave removes three reports that stopped at shapes, private fields, or constructor topology.
DSv4 RoPE cache precedence now runs through real consumers: the C0/C128 attention forward-backward policy constructs and
uses the environment-sized cache, while the context-parallel C128 compressor forward requires the config-sized fallback to
cover its nonzero-rank slice. The standalone cache-builder shape report is gone.

DeepSeek packed-checkpoint configuration likewise no longer ends at the selected handler type and its private bit-width and
group-size fields. The retained loader transaction obtains the handler through the model, parses the official nested
compressed-tensors configuration, and dequantizes actual 8-bit/group-64 expert payloads to exact gate, up, and down values.
Default packed loading and requested BF16 output remain in the same transaction.

Finally, DSv4 component selection is observed through complete paths instead of a presence-only report. The retained C0 and
C128 attention transactions execute their respective topologies, and the C4 synthetic checkpoint load constructs both the
compressor and indexer and validates their separately translated APE tensors.

All six focused attention, compressor, checkpoint-loader, and cache-boundary reports pass. Repository collection is now
1,212 items and the static inventory is 1,198 definitions with 825 curated decisions across 354 Python test files. The audit
still contains only 26 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the
intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-thirty-first wave: remove dependency canaries and scale repetitions

The one-hundred-and-thirty-first wave removes four static reports across linear attention and batch-invariant reductions.
The Hopper Gated Delta Rule report imported and executed the optional FLA package directly without reaching an XoRL module,
wrapper, or integration seam. Its illegal-memory/autotuner result was therefore an upstream dependency canary rather than a
repository regression, so the file is gone.

Full-reduce mean now owns its FP32-output dtype spelling in the same contract as the one- and two-dimensional BF16/FP32
mean-versus-sum boundaries. Head-v2 likewise keeps one authoritative focused suite. The removed production-hidden and
production-vocabulary reports repeated exact v1 projection bits, the shared decode/scoring statistics tree, and prefix batch
invariance without selecting another launch branch. The retained focused policies additionally prove arbitrary-slice
invariance, selected-logprob composition, fused-loss gradients, and the family-v1 rollback.

All six retained family-selection, head-v2, and mean reports pass. Repository collection is now 1,209 items and the static
inventory is 1,194 definitions with 828 curated decisions across 353 Python test files. The audit still contains only 26
legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate
distributed wrapper group unchanged.

## One-hundred-and-thirty-second wave: stop testing orphan compatibility helpers

The one-hundred-and-thirty-second wave removes four reports whose private targets have no production callers. The eager
OPRD layer-cache gather wrapper was called only by its test; the live loss path uses the retained streaming slice fetcher,
which verifies selected cache rows, multiple layer ranges, total layer count, and returned shapes.

The legacy `_accumulate_is_metrics` and `_finalize_is_metrics` pair is likewise absent from runtime call sites. Forward-
backward now routes loss metrics through `_accumulate_loss_metrics` and `_finalize_loss_metrics`, whose retained OPD policy
covers mean, extrema, empty-rank, and loss-specific behavior. Two distributed reports and a CPU report for the dead pair are
gone; the still-live per-micro-batch `_sp_allreduce_kl_metrics` collective remains covered by its two-rank NCCL gate.

All three retained OPRD fetcher, current metric-aggregation, and live NCCL reduction reports pass. Repository collection is
now 1,205 items and the static inventory is 1,190 definitions with 830 curated decisions across 352 Python test files. The
audit still contains only 26 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the
intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-thirty-third wave: prefer live paths over duplicate wrappers and fabricated states

The one-hundred-and-thirty-third wave removes seven reports across EP kernels, QARL, DSv4, server launch, routing replay,
and activation offload. A forward-only Triton/Quack routing-score report duplicated the same test stubs and torch reference
used by a retained report that additionally proves routing-score gradients. A direct `QARLLinear` smoke likewise repeated
gradient and persistence observations already covered by retained injected-model calibration and full optimizer/checkpoint
lifecycles.

DSv4 fallback coverage now enters through public `rotate_activation`: one transaction disables the optional kernel and
proves the known transform, self-inverse behavior, and norm preservation across supported widths. Separate private-helper
algebra, impossible-width rejection, and shape-only dispatch reports are gone. The launcher report that manually combined a
valid flat config with missing parsed server arguments and the routing-replay report that overwrote a private global with a
made-up stage both manufactured states normal callers cannot create. Finally, the standalone activation-offload report
passed `None` outside the typed trainer/server argument boundary and asserted only that an unrelated square operation had a
gradient; it observed no offloading behavior.

All 17 executed focused reports pass, with one legitimate optional-backend skip. Repository collection is now 1,198 items
and the static inventory is 1,183 definitions with 836 curated decisions across 351 Python test files. The audit now contains
25 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional duplicate
distributed wrapper group unchanged.

## One-hundred-and-thirty-fourth wave: stop preserving test-only support surfaces

The one-hundred-and-thirty-fourth wave removes the three-report `FileLockLoader` suite. The class is neither exported from
`xorl.data.prepare` nor referenced anywhere else under `src`; only its own tests named it. Live dataset preparation remains
covered through packing-cache persistence, preprocessing, source loading, hashing, retries, and collator/data-loader
lifecycles.

Two broader server policies were also narrowed to live interfaces. Runner messages still round-trip every payload and tensor
through the production transport codec and reject pickle input, but no longer preserve unused `BaseMessage` JSON helpers.
`FutureStore` scheduling, result/failure storage, deletion, model cleanup, and expiry now inspect the live `FutureEntry`
returned by `get`; assertions for six convenience accessors with no production caller are gone.

All 14 focused server-protocol, future-store, and live data-preparation reports pass. Repository collection is now 1,195
items and the static inventory is 1,180 definitions with 839 curated decisions across 350 Python test files. The audit still
contains 25 legitimate conditional runtime gates, with no no-outcome candidates, no parse errors, and the intentional
duplicate distributed wrapper group unchanged.

## One-hundred-and-thirty-fifth wave: fold helper checks into consumers and drop a dormant campaign

The one-hundred-and-thirty-fifth wave removes two direct helper reports from distributed and FP8 coverage. PyTorch's wrapped
reduce operation now remains qualified by the retained real two-rank FSDP2 custom-reduce-scatter lifecycle rather than a
four-line private canonicalizer truth table. Full-precision expert FSDP kwargs are now checked on an actual exact GLM shared
expert in the retained topmost-unit topology policy, replacing a fake class with one boolean field.

Four SM90 P5 reports are also gone. They certified an opt-in GDN decode-prep candidate that is not exported, documented, or
called by production; its only source consumer is another unused decode-solve candidate. Those bitwise and graph-capture
campaign gates therefore protected no reachable XoRL execution path.

All 11 retained focused topology, FP8 checkpoint, and FSDP2 reports pass, including the real two-rank FSDP lifecycle.
Repository collection is now 1,189 items and the static inventory is 1,174 definitions with 842 curated decisions across 349
Python test files. The audit still contains 25 legitimate conditional runtime gates, with no no-outcome candidates, no parse
errors, and the intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-thirty-sixth wave: remove support surfaces that only their tests consume

The one-hundred-and-thirty-sixth wave removes an idealized pipeline-bubble formula with no caller and keeps the measured
`PPBubbleProfiler` transaction used by `Trainer`. Data preparation no longer carries a separate first-fit-decreasing
feasibility checker that never participates in packing; the retained report exercises `pack_group`, sequential allocation,
and `PackingDataset`. Likewise, the grouped and any-FQN matchers are gone because only their truth tables called them, while
the retained matcher remains used by parallel plans and sharded adapter state.

EP synchronization now has one production model: the coalesced optimizer-boundary reducer. A standalone per-parameter hook,
its single-gradient reducer, and a test-only statistics alias were removed from the real multi-rank report. That report still
proves two-rank clipping, non-finite rejection, bucket accounting, and three-rank participation masks.

The larger removal is an unreachable DeepSeek-V4 indexer autograd campaign. Production `V4Indexer` calls the forward score
kernel directly and returns discrete top-k indices; it never imported the separate autograd wrapper or backward kernel. The
two parameterized reports for those modules contributed eight collected cases. The retained seven V4 reports all pass on
GPU and cover the reachable forward scores, causal mask, production geometry, numerical range, and zero input. Together with
16 focused CPU and real multi-rank passes, repository collection is now 1,180 items and the static inventory is 1,171
definitions with 847 curated decisions across 349 Python test files. The audit still contains 25 legitimate conditional
runtime gates, no parse errors, and the intentional duplicate distributed wrapper group unchanged.

## One-hundred-and-thirty-seventh wave: test behavior through consumers, not synthetic compositions

The one-hundred-and-thirty-seventh wave deletes the manual CUDA timing module and its sole lifecycle report. No production
module, package export, documentation, example, or script could enable or drain that instrumentation; mocked CUDA events
were testing an isolated subsystem with no XoRL consumer.

KV repetition is now qualified through eager GQA attention. The retained transaction compares full attention weights and
outputs with an independent `torch.repeat_interleave` reference, replacing a standalone helper smoke that checked shapes,
one tiny pattern, and device preservation. The synthetic “full MoE pipeline” report is also gone: identity experts reduced it
to the same scatter-gather round trip already checked numerically, while the retained kernel and model suites cover routing,
real expert computation, and gradients. Finally, the test-only pipeline single-stage predicate duplicated the live schedule
style table and was removed without weakening schedule construction or admission coverage.

All 11 focused attention, MoE, and pipeline reports pass; the two skips are the expected unavailable FA3 interface. Repository
collection is now 1,177 items and the static inventory is 1,168 definitions with 851 curated decisions across 348 Python test
files. The audit still contains 25 legitimate conditional runtime gates, no parse errors, and the intentional duplicate
distributed wrapper group unchanged.

## One-hundred-and-thirty-eighth wave: stop tests from designing unused APIs

The one-hundred-and-thirty-eighth wave removes six groups of production support surfaces that existed only to make direct
unit assertions convenient. GLM sparse selection no longer carries an unused physical-page translator or selected-value
gatherer, and its inventory and layer plan no longer precompute role and schedule views solely for tests. The retained
contracts derive those observations from the canonical target and layer tuples and still execute the live logical-index
selector across ties, short rows, dead rows, and the production boundary tail.

NVFP4 now exposes and tests its real format-specific fake quantizer rather than a private one-format string dispatcher.
Native FP8 checkpoint protection likewise keeps the `DistributedCheckpointer`-used real-DCP preflight and removes a
metadata-dictionary adapter whose only caller fabricated that dictionary in a test. `ModelState` no longer advertises a
future lightweight safetensors reference collector with no exporter or save-path consumer; the QARL report retains actual
checkpoint metadata and compatibility rejection.

Finally, routing replay follows its real transaction lifecycle. Tests no longer preserve cursor-reset methods absent from
the trainer and R3 handler; replay advances the backward cursor and `clear_all` performs teardown. The six focused suites
report 40 passes and one expected optional-SGLang skip. Collection and the static test inventory are unchanged at 1,177
items and 1,168 definitions because this wave removes test-only support and narrow assertions inside retained behavioral
reports. The ledger now contains 857 curated decisions across 348 Python test files.

## One-hundred-and-thirty-ninth wave: retire diagnostic router policy matrices

The one-hundred-and-thirty-ninth wave removes two process-wide router diagnostics that only their tests selected.
`XORL_MOE_ROUTER_TOPK_POLICY` injected stable-sort, artificial tie bias, or raw-logit selection into every ordinary router,
but had no configuration, documentation, script, example, or production consumer. The live softmax and DSv4 paths now call
`torch.topk` directly; retained reports continue to cover softmax weighting and normalization, correction bias, hash
routing, balanced profiling, exact batch-invariant routing, and rejection boundaries.

The layer-list `XORL_MOE_ROUTER_FP32_LAYERS` parser is also gone. Its standalone report is folded into the real model
configuration contract, which drives `_router_fp32` through `MoEBlock` and observes FP32 hidden and gate operands. This
preserves the shipped configuration behavior while deleting one test report and an environment-only parallel interface.

Finally, dense, fused-delta, and MoE LoRA modules no longer expose manual merged-weight cache invalidators with no caller.
Their caches already key entries on tensor versions, storage pointers, active rank, and alpha; retained tests prove
automatic invalidation across optimizer steps and runtime rank changes and ensure old fused generations are released. All
20 focused router, merged-LoRA, and Qwen integration reports pass. Repository collection is now 1,176 items and the static
inventory is 1,167 definitions with 860 curated decisions across 348 Python test files.

## One-hundred-and-fortieth wave: retire test-driven numerical experiments

The one-hundred-and-fortieth wave removes two environment-only numerical campaigns whose tests never established the
claimed behavior. `XORL_MOE_FP64_ACCUM` selected a slow, inference-only expert detour with no serving counterpart,
configuration, launcher, documentation, or example. Its sole assertion replaced that detour with a mock and proved only
that the switch outranked the fused-SGLang switch. The normal eager, Triton, fused-SGLang, EP, and TP-simulation paths and
their real forward, gradient, layout, determinism, and admission coverage remain.

Qwen3-MoE no longer contains the delayed-residual tuple, TP-shard-carry, alternate post-attention residual formulas, forced
RMSNorm flags, or candidate-capture matrix. Those branches were reachable only through undocumented process-wide
variables, and four reports exercised them by directly passing private tuple states or attaching attributes to tensors in
test doubles. The retained Qwen report instead executes the normal decoder and final-norm consumers and checks the shipped
no-residual versus residual-tree family declaration. Removing the experiment also removes its generic delayed-output hook
and stale diagnostic-site registrations.

The primary focused gate reports 10 passes and two expected skips because SGLang is unavailable; the adjacent real TP,
diagnostic-capture, and module-utility suites add 26 passes. Repository collection is now 1,172 items and the static
inventory is 1,163 definitions with 862 curated decisions across 348 Python test files. The audit still contains 25
legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-forty-first wave: remove helpers whose tests were their product

The one-hundred-and-forty-first wave deletes the obsolete stacked-LoRA helper module. Its initialization, delta, merge, and
unmerge functions had no production, documentation, example, or script consumer; a package re-export and one arithmetic
truth table were the entire interface. The scaling formula is the exception: it remains directly at the group-GEMM package
boundary because dense, MoE, and quantized adapters use it. Twenty-two retained LoRA, QLoRA, loading, construction, gradient,
and optimizer reports pass through those real consumers after the move.

Dense op parity also loses a tautology. The private eager SwiGLU wrapper was one `torch.nn.functional.silu` expression and
had no source caller; its report compared it to the identical expression labeled as a serving reference. The independent
RoPE implementation comparison remains, while real fused SwiGLU forward and backward behavior continues to run through the
Triton operator, exact GLM MLP composition, and model-level adapter suites.

Repository collection is now 1,170 items and the static inventory is 1,161 definitions with 864 curated decisions across
347 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group,
and no parse errors.

## One-hundred-and-forty-second wave: delete a tested kernel with no execution path

The one-hundred-and-forty-second wave removes the specialized families-v2 Q/K-normalization kernel, its standalone report,
and its frozen golden row. Despite the production-contract language in the test, no model, trainer, dispatcher,
configuration, documentation, example, or script called `qk_norm_v2`; only those two test invocations reached it. The actual
Qwen3.5 and Qwen3.5-MoE attention paths instantiate their declared RMSNorm modules.

The live families-v2 hidden-state RMSNorm remains intact. Its fused and split realizations, dispatch threshold, exact-model
family selection, cross-structure equivalence, and frozen numerical trees all remain covered. All 25 focused families-v2,
RMSNorm, exact-model selection, and golden-tree reports pass. Repository collection is now 1,169 items and the static
inventory is 1,160 definitions with 865 curated decisions across 347 Python test files. The audit still contains 25
legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-forty-third wave: retire a diagnostic backend matrix

The one-hundred-and-forty-third wave removes the `XORL_SGLANG_MOE_TP_SIM` campaign. Its direct, cache, Triton,
alternate-reduce, DeepGEMM, fused-kernel, and runner modes were an environment-only K3 diagnostic with no repository
configuration, launcher, example, documentation, or ordinary execution-path consumer. Commit history also labels later
changes to the lane as experiments and diagnostics. Nine reports exercised it with tiny full-local tensors and Python
fakes for every optional backend, so the tests largely specified a parallel simulation product rather than validating a
supported training topology.

The supported SGLang fused-expert implementation remains intact: local and EP dispatch, serving weight layouts, autograd,
cache invalidation, runtime-context admission, and failure boundaries retain their real suites. The simulation-only runner
loader, shard attributes, diagnostic capture names, and MoEBlock bypass conditions are gone with the test file. Fifty-one
focused MoE, LoRA, checkpointing, and diagnostic-capture reports pass, with three expected optional-SGLang skips.
Repository collection is now 1,160 items and the static inventory is 1,151 definitions with 866 curated decisions across
346 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group,
and no parse errors.

## One-hundred-and-forty-fourth wave: stop claiming unexecuted backend coverage

The one-hundred-and-forty-fourth wave removes a CPU-only SGLang RMSNorm report. Although it selected the `sglang_jit` and
`sglang_kernel` diagnostic modes, it never ran their CUDA implementations, optional-package loaders, ABI boundaries, or
serving arithmetic; both cases simply followed an eager CPU formula. The report therefore advertised backend coverage
that it did not provide.

Those explicit diagnostic modes remain available. The retained RMSNorm suites cover ordinary and fused CPU arithmetic,
fused CUDA forward and backward, exact Qwen model-site integration, family admission, global configuration forwarding,
and real kernels when the optional runtime is present. Repository collection is now 1,159 items and the static inventory
is 1,150 definitions with 867 curated decisions across 345 Python test files. The audit still contains 25 legitimate
conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-forty-fifth wave: remove speculative weight-sync and legacy payload surfaces

The one-hundred-and-forty-fifth wave moves into server and weight-sync coverage. Sparse-delta receiver sharding,
per-rank raw and encoded writers, translation-future collection, and terminal future serialization formed a closed
source/test-only subgraph: no runtime, configuration, documentation, example, or script called it. Three reports built
fake receiver shards and replaced the complete optional `delta_encoding` future API. That speculative layer is gone.
The live source-capture path remains, including optimizer-step snapshots, rank/global manifests, validated single-file
packing, and sparse-delta backend consumption.

The same wave removes four deprecated R3 payload names: `externalize_r3_payloads`, `keep_r3_payloads`,
`routing_payload_dir`, and `keep_routing_payloads`. They existed only in compatibility branches and two test inputs;
launchers and runtime constructors already use the canonical transport, directory, retention, and namespace fields.
Canonical Mooncake and filesystem reports continue to cover creation, slicing, cleanup, retention, validation, and
serialization. All 34 focused server and weight-sync reports pass.

Repository collection is now 1,156 items and the static inventory is 1,147 definitions with 869 curated decisions across
345 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group,
and no parse errors.

## One-hundred-and-forty-sixth wave: replace fake bootstrap confidence with real boundaries

The one-hundred-and-forty-sixth wave removes a trainer bootstrap report that instantiated `Trainer` through `__new__`,
constructed two large fake argument trees, and mocked every bootstrap dependency to observe one direct `ep_intranode`
keyword assignment plus one loss-dictionary branch. The retained distributed suite executes both EP mesh geometries;
loss reports exercise `quack_linear` computation and admission; argument and model-builder reports cover the meaningful
configuration boundaries without reproducing bootstrap implementation details.

Muon also loses the undocumented `XORL_MUON_QUACK_TUNED` override. Its only assertion replaced Quack GEMMs with lambdas
and checked a keyword, while no configuration, documentation, example, or script exposed the switch. The qualified
`tuned=False` default remains, as do backend import, architecture/dtype dispatch, real optimizer update, grouped Gram
Newton-Schulz, and CUDA dtype reports. All 30 focused optimizer, loss, distributed, and trainer reports pass.

Repository collection is now 1,155 items and the static inventory is 1,146 definitions with 871 curated decisions across
344 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group,
and no parse errors.

## One-hundred-and-forty-seventh wave: keep data tests at the lifecycle boundary

The one-hundred-and-forty-seventh wave removes the two-report `CollatePipeline` unit suite. It composed fake collators that
added and multiplied token IDs, then asserted the arithmetic and permissive single-collator, tuple, and empty-list constructor
forms. Production constructs the pipeline only from a non-empty collator list, and the retained dataloader integration runs
that real sequence through tensor conversion, flattening, token shifting, packing, micro-batch splitting, and optional
sequence sharding. The live pipeline now exposes only the sequence constructor it actually consumes.

This wave also finishes source cleanup identified by earlier test decisions. The unexported `FileLockLoader` and the unused
SHA256 string wrapper had no runtime, documentation, or example caller after their standalone reports were removed. Retry
configuration loses linear and constant policies that only its test selected; the sole production decorator call uses the
retained exponential policy. Its remaining report now observes requested sleeps deterministically instead of measuring real
wall-clock intervals.

All 26 focused data and checkpoint reports pass. Repository collection is now 1,153 items and the static inventory is 1,144
definitions with 874 curated decisions across 343 Python test files. The audit still contains 25 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-forty-eighth wave: keep protocol tests on the live transport

The one-hundred-and-forty-eighth wave removes a closed server-protocol convenience layer. Twelve response builders and four
validation or introspection helpers were exported and tested, but no orchestrator, API server, scheduler, dispatcher,
documentation example, or other runtime caller used them. Their tests manually populated response dictionaries and then
asserted those same fields. The retained protocol report now round-trips the actual typed request and output dataclasses
through msgpack, including operation-payload reconstruction and terminal result fields used by live ZMQ communication.

The same wave removes another fully mocked Trainer forwarding report. It constructed `Trainer` via `__new__`, supplied a
large fake argument tree, replaced foundation-model construction, and asserted direct keyword copies for numerical flags and
LoRA scalars. Argument parsing already covers the configuration values, while model-policy suites exercise resolution,
admission, and numerical behavior; the forwarding report could not discriminate any of those outcomes.

All 41 executed focused protocol, orchestrator, argument, and model-policy reports pass. Repository collection is now 1,151
items and the static inventory is 1,142 definitions with 876 curated decisions across 342 Python test files. The audit still
contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-forty-ninth wave: remove shallow copies of lifecycle contracts

The one-hundred-and-forty-ninth wave removes a dense NVFP4 injection smoke that wrapped two `Linear` modules and checked
their type, format string, and output shape. The retained QARL contracts already exercise targeted dense injection and
exclusions, wrapper counters and summaries, lossy NVFP4 arithmetic, straight-through gradients, a real optimizer update,
changed log-probabilities, and checkpoint restoration. The extra sequential-model shape check added no distinct failure
boundary.

The same wave removes a mock-only weight-version forwarding report. Its fake handler/backend chain duplicated the retained
handler policy assertion that verifies `flush_cache` and `weight_version` at `transfer_bucket`, while the P2P protocol suite
verifies the version in the actual completion request body. Seven focused QARL, handler, and protocol reports pass.
Repository collection is now 1,149 items and the static inventory is 1,140 definitions with 878 curated decisions across
341 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group,
and no parse errors.

## One-hundred-and-fiftieth wave: stop testing configuration plumbing with fake neighbors

The one-hundred-and-fiftieth wave removes two builder-plumbing reports. One replaced foundation-model construction and
parallelization solely to assert that `fsdp_sharded_lm_head_loss=True` crossed one function call. The other replaced the
server's training-model builder and asserted unchanged FP8, QARL, and sharded-loss dictionary values. Neither report
constructed the claimed distributed loss, FP8 model, or calibrated QARL model. Retained suites execute sharded LM-head
loss under FSDP, perform real FP8 and QARL injection, run calibration before parallelization, update parameters, and
restore QARL checkpoints.

This wave also removes a Dr.GRPO outer-loop report that replaced `_forward_loop` and then asserted that the string
`"drgrpo"` and its inputs reached that fake. The retained runner report executes the actual Dr.GRPO loss branch, including
clipping, KL, temperature, legacy fields, per-token output policy, and K3 output; independent runner and dispatcher
lifecycle suites cover completion, failure, model identity, routing setup, and step accounting.

All 16 focused behavioral reports pass, including a real sharded-loss FSDP transaction. Repository collection is now
1,146 items and the static inventory is 1,137 definitions with 881 curated decisions across 341 Python test files. The
audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-fifty-first wave: prefer transformed payloads and constructed models

The one-hundred-and-fifty-first wave removes a request-processor report that supplied raw routed-expert arrays, replaced
both backend methods with `AsyncMock`, and asserted that the same Python objects appeared in the mock kwargs. Retained
routing transactions exercise the behavior that can actually fail: Mooncake and filesystem encoding, datum reordering,
slice loading, cleanup on success and failure, wire decoding, rank-zero selection, and model identity.

The same wave removes the remaining fake-builder report from the model-runner FP8 file. It replaced
`build_training_model`, asserted direct copies of block-FP8 QLoRA settings, and checked a target set already exercised by
the dedicated GLM target-resolution policy. Real builder suites retain foundation/injection configuration, precondition
rejection, adapter inventory ownership, target selection, quantized construction, and QLoRA execution.

All seven focused routing and GLM construction reports pass. Repository collection is now 1,144 items and the static
inventory is 1,135 definitions with 883 curated decisions across 341 Python test files. The audit still contains 25
legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-fifty-second wave: test module-path helpers through real plans

The one-hundred-and-fifty-second wave removes the standalone distributed-utils suite. Its two reports restated recursive
`getattr`/`setattr`, regex wildcard matching, and exception behavior on a toy `Sequential`/`ModuleDict`. The retained
`ParallelPlan` suites invoke the same helpers through exact and wildcard FQNs while replacing meta parameters, slicing
global expert banks, preserving dtype/trainability, assigning replicated gradient domains, and materializing real GLM
adapter layouts. Sharded adapter-state tests exercise the same matcher on production ownership plans.

The caller audit also removes the singular `find_free_port` launcher helper, which had no source, test, documentation,
example, or script caller. The live launcher uses the distinct `find_free_ports` allocator for its three- and four-port
rendezvous layouts.

All 11 focused plan, sharded-state, and launcher reports pass. Repository collection is now 1,142 items and the static
inventory is 1,133 definitions with 885 curated decisions across 340 Python test files. The audit still contains 25
legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-fifty-third wave: remove unintegrated helper islands

The one-hundred-and-fifty-third wave removes the mock-only `RemoteBackend` suite. It replaced `_execute`, invoked two thin
wrappers, and asserted operation names, request IDs, timeouts, and fields copied into payload constructors without
serializing or transporting anything. Retained protocol, request-processor, dispatcher, sparse-delta, and weight-sync
suites exercise typed reconstruction and the downstream behavior of those fields.

This wave considered removing the standalone `xorl.rl` package because its six exported Slime-style tensor helpers had no
trainer, runner, loss, CLI, example, documentation, or other source caller. The historical audit initially retained the
package and its direct contract suite as a conservative public compatibility boundary. The current-main follow-up below
supersedes that choice after tracing the exact XoRL Client PR 14 path and the trainer loss registry.

The reachability audit found two smaller test-driven APIs as well. Runner acknowledgement and response factories had no
runtime caller; live transport constructs the dataclasses directly, and the retained protocol report round-trips them
through MessagePack. The sparse source-delta translation-input loader had no translation engine or receiver caller, and
its only report installed a synthetic `delta_encoding` package to fabricate empty shards. Source capture, packed-file
validation, backend upload, receiver application, and trainer-to-SGLang coverage remain.

All 30 focused protocol, request-processing, sparse-delta, and integrated-loss reports pass. Repository collection is now
1,136 items and the static inventory is 1,127 definitions with 889 curated decisions across 338 Python test files. The
audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-fifty-fourth wave: make reference checks independent

The one-hundred-and-fifty-fourth wave removes `canonical_moe_reduce_reference` from production code and drops the report
that tested that oracle itself. The oracle shared `_adjacent_pairwise_bf16` with the implementation it was meant to
validate, so actual-versus-expected comparisons were circular. Distributed and GLM model contracts now compute their
expected adjacent BF16 tree independently in test code. The real multi-process transport gate passes with permutation,
chunking, output-distribution, padding, and backward checks intact.

The same reachability pass removes `SequencePartial` and its synthetic dense, packed, and hand-sliced context-parallel
matrix. No loss, trainer, runner, CLI, example, documentation, or other source caller selected it. `TokenPartial` is the
sole production reducer and remains covered through integrated causal-LM, policy, importance-sampling, OPD, Dr.GRPO, TP,
and FSDP paths.

Two narrow server reports also go away. An older Tinker compatibility report duplicated focused session creation and
weights-info lifecycles; the retained endpoint report now includes its only distinct flat `lora_rank` assertion. A
sampler-prefill report converted one literal list to a tensor and captured it on a fake model, while retained GLM indexer
and sparse-attention contracts cover the actual prefill-boundary behavior.

Eighteen focused reducer, runner, API, distributed, and GLM reports pass; one CUDA-only shared-expert report skips on this
host. Repository collection is now 1,132 items and the static inventory is 1,123 definitions with 893 curated decisions
across 338 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate
group, and no parse errors.

## One-hundred-and-fifty-fifth wave: separate telemetry presentation from behavior

The one-hundred-and-fifty-fifth wave removes two trainer reports that instantiated no trainer lifecycle. One called a
private activation-offload consumer on `SimpleNamespace` objects and asserted byte-to-GB field names, call counts, and an
empty dictionary. The other called private exception-path summarizers and asserted key ordering plus four copies of each
local float. The retained component-timer suite runs real forward and backward hooks on CUDA across GLM- and Qwen-shaped
layers and retains the unrecorded-event recovery policy; broader trainer suites exercise optimization and synchronization.

Two fake forwarding seams are consolidated into one runner lifecycle report. The old pair stopped once at
`_execute_and_gather` and once at `_execute_compute`. The replacement runs the real rank-zero handler, gather wrapper, and
compute dispatch through to the trainer, while isolating only distributed side effects, so session identity and both R3
payloads are checked across the complete chain.

This wave also removes a source-text lint that opened `bi_families_v2.__file__` and searched for banned import strings. The
retained families-v2 suites exercise model-program selection, rollback, numerical dispatch, independent fused/split
realizations, cross-engine bytes, and real CUDA bit gates. The audit now recognizes module-source reads through
`Path(module.__file__).read_text()` as source inspection, preventing this pattern from hiding behind ordinary file I/O.

All nine focused trainer, dispatcher, and families-v2 reports pass, including the CUDA component-hook path. Repository
collection is now 1,128 items and the static inventory is 1,119 definitions with 896 curated decisions across 336 Python
test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no
parse errors.

## One-hundred-and-fifty-sixth wave: put repository policy in lint

The one-hundred-and-fifty-sixth wave relocates the repository-wide private-reference scan out of pytest. It did not
exercise XoRL behavior: it ran `git ls-files`, decoded every tracked file, and matched policy regexes for private paths,
cluster identifiers, and authoring metadata. The same zero-dependency check now lives in `scripts/check_public_tree.py`
and is a local pre-commit hook, so the existing lint workflow still enforces it on every pull request without presenting
it as a product test.

This wave also removes a duplicate fake register-session dispatcher report from the request-processor file. The dedicated
session-ops suite already forwards the exact typed payload and response through `_handle_register_session` and additionally
tests cross-rank rejection. The retained request-processor report separately runs registration through the processor and
`DummyBackend`, so both meaningful boundaries remain without a third fake coordinator.

The public-tree lint passes, as do both focused registration paths. Repository collection is now 1,126 items and the
static inventory is 1,117 definitions with 898 curated decisions across 335 Python test files. The audit still contains
25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-fifty-seventh wave: distrust answers supplied by the mock

The one-hundred-and-fifty-seventh wave removes an orchestrator-client report whose mock engine counted the samples it
had just received and echoed the request learning rate. Those assertions were properties of the test double, not the
server. Retained real-ZMQ reports cover interleaved forward and optimizer traffic plus exact wire payloads, and the real
orchestrator lifecycle covers empty-batch rejection.

The same pass removes two duplicate API paths and one duplicate validation fragment. Focused endpoint reports already
exercise normalized LoRA worker registration and full-weight admission; the latter now supplies legacy empty optional
configs so that compatibility behavior remains explicit. The focused training-ops report already checks explicit
learning-rate forwarding, while the retained compatibility bundle still covers legacy Adam payloads and the effective
learning-rate fallback priority. At the request processor, only the distinct nonempty batch without valid targets
remains; the empty-list case belongs to the orchestrator end-to-end report.

All eight focused socket, orchestrator, endpoint, and training-operation reports pass. Repository collection is now
1,124 items and the static inventory is 1,115 definitions with 901 curated decisions across 335 Python test files. The
audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-fifty-eighth wave: make a component contract a test-sized unit

The one-hundred-and-fifty-eighth wave consolidates microscopic wrapper reports across exact GLM-5.2 LM-head QLoRA,
fused gate-up QLoRA, native FP8 model construction, exact QLoRA admission, and native block-FP8. Twelve collected items
did nothing beyond invoking one to three neighboring assertion helpers for one CPU component. The replacement reports
group topology, operand admission, byte presentation, gradients, construction, checkpointing, and failure behavior at
the component level, explicitly resetting monkeypatch state between independent seams.

No behavioral assertion was removed. Separate Hopper gates stay separate, as do the native router and expert reports
whose runtime boundaries differ from buffer construction. All nine resulting focused contracts pass. Repository
collection is now 1,112 items and the static inventory is 1,103 definitions with 902 curated decisions across 335 Python
test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no
parse errors.

## One-hundred-and-fifty-ninth wave: verify checkpoint translation at its destination

The one-hundred-and-fifty-ninth wave removes a DeepSeek-V4 report that directly asserted private checkpoint-name string
rewrites and round-tripped the APE inverse against a test-defined forward transform. The retained synthetic checkpoint
transaction already drives the real loader across window, C4, hash, shared-expert, and routed-expert families.

That transaction is now the stronger oracle: it checks loaded values at the embedding/head, norm, attention, HC,
router-bias, shared-expert, fused routed-expert, C4 APE, and renamed indexer destinations. The quantized codec and EP
handler ownership reports remain separate because the synthetic load intentionally uses unquantized, single-rank input.
All three retained DeepSeek-V4 loader reports pass. Repository collection is now 1,111 items and the static inventory is
1,102 definitions with 903 curated decisions across 335 Python test files. The audit still contains 25 legitimate
conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-sixtieth wave: a fake injector cannot certify construction

The one-hundred-and-sixtieth wave removes a GLM-5.2 block-FP8 QLoRA builder report that replaced both foundation-model
construction and QLoRA injection. Its result was six captured argument values and an inventory object supplied by the
test itself; it never built or adapterized the claimed model.

The real GLM-5.2 suites retain the meaningful boundary: they construct all 700 targets and 1,700 factors, enforce exact
component and product-mode admission, and verify trainable ownership. The builder's distinct fail-closed rule requiring
both LoRA and QLoRA remains in the consolidated quantized-mode admission report. All five focused builder and real-model
reports pass. Repository collection is now 1,110 items and the static inventory is 1,101 definitions with 904 curated
decisions across 335 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional
duplicate group, and no parse errors.

## One-hundred-and-sixty-first wave: prefer an executed topology over its dictionary

The one-hundred-and-sixty-first wave removes an OLMo-2 report that inspected tensor-parallel plan dictionary entries and
missing keys. Retained two-rank CPU reports apply that production plan to a real OLMo-2 model, execute forward and backward
through local-axis QK RMSNorm, rowwise and colwise projections, post-norm residual flow, and the vocab-sharded LM head, and
compare the custom local-axis norm with an independent numerical reference.

The GLM sparse-MLA suite no longer compares CPU `auto` dispatch directly with the same torch reference implementation. Its
full-model sparse-versus-dense report already reaches `auto` through `Glm5Model` and checks numerical parity. The distinct
unknown-backend rejection is preserved in that integration report.

Finally, the standalone LM-head topology matrix is removed. It launched the same CP, DP, and HSDP layouts used by the
retained four-rank FSDP end-to-end cases, but asserted only group membership and mesh labels. The retained cases build a
real sharded LM head and compare parameter synchronization, vocab ranges, global loss, full weight gradients, and local
hidden gradients with eager references. The separate EP-overlay topology report remains because no equivalent EP execution
gate exists.

All eight focused model, OLMo-2 distributed, and LM-head distributed reports pass. Repository collection is now 1,107 items
and the static inventory is 1,098 definitions with 907 curated decisions across 334 Python test files. The audit still
contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-sixty-second wave: parse checkpoint paths through the API

The one-hundred-and-sixty-second wave removes a checkpoint URI report that called `_to_xorl_uri` and `_from_xorl_uri`
directly. The adjacent save-and-load lifecycle already creates real checkpoint directories and drives all five documented
spellings through the public API: xorl URI, explicit `weights/model/checkpoint`, `model/checkpoint`, checkpoint-only, and
legacy `weights/checkpoint`. The direct report also treated an undocumented arbitrary raw path as a compatibility contract.

The retained lifecycle now pins the exact public xorl URI returned by save, instead of merely checking that its model and
checkpoint substrings appear. All seven checkpoint-path lifecycle reports pass. Repository collection is now 1,106 items
and the static inventory is 1,097 definitions with 908 curated decisions across 334 Python test files. The audit still
contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-sixty-third wave: exercise collator primitives through collators

The one-hundred-and-sixty-third wave removes direct reports for FlashAttention metadata construction and the private
sequence-shard slicing and padding primitives. Those reports supplied synthetic tensors directly to implementation
helpers, including a zero-length padding no-op, even though the live packing and sequence-shard collators own every
production call.

The retained collator reports are now stronger consumer-level oracles. Packing checks exact cumulative sequence lengths
and maximum lengths after both multi-document and single-document concatenation. Sequence sharding checks exact rank-zero
and rank-one token and label slices, then drives a nondivisible sequence through the last CP rank to verify constant token
padding, ignored-label padding, and sequential position padding together. All six collator reports pass. Repository
collection is now 1,104 items and the static inventory is 1,095 definitions with 910 curated decisions across 334 Python
test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no
parse errors.

## One-hundred-and-sixty-fourth wave: cross the real boundary

The one-hundred-and-sixty-fourth wave deletes a standalone API-orchestrator MessagePack report. The retained ZMQ
client-engine lifecycle already serializes and deserializes the same request and output types across the production
sockets; it now pins the request payload, sequence id, timestamp, response identity, type, payload, and terminal flag
after that roundtrip.

The same pass deletes a DSv4 report that called the RoPE CP-slice helper with a fabricated group and arbitrary cache.
The retained compressor report now constructs an undersized real CP compressor and reaches the fail-loud cache-capacity
guard through `forward_raw`, alongside its successful C128 path. All four focused communication and compressor reports
pass. Repository collection is now 1,102 items and the static inventory is 1,093 definitions with 912 curated decisions
across 332 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate
group, and no parse errors.

## One-hundred-and-sixty-fifth wave: report component decisions, not helper branches

The one-hundred-and-sixty-fifth wave consolidates thirteen reports without dropping a behavioral assertion. Deferred
QLoRA key planning now reports dense, EP16, missing-pair, and cache-residency branches at the loader-policy level.
DeepSeek router admission groups the foundation and training-builder entry points while leaving its TP guard separate.
Teacher-cache selection already covered through the Mooncake consumer is no longer reported again; unique async, bounds,
device, dtype, and layer-slice behavior remains in the cache lifecycle.

The same component-level rule removes narrow reporting around optimizer snapshots and immediate checkpoint reload,
request-processor registration and invalid-target branches, create-model normalized registration, folded-LoRA gradient
dtype, checkpoint model-key inventory, and grouped-load expert-name classification. Those assertions now run in the
transaction, resume, processor, endpoint, autograd, checkpoint-compatibility, and grouped-load lifecycles that consume
them. The consolidation also exposed and fixed order-sensitive monkeypatch state between checkpoint helpers.

The audit explicitly retained three dense suites after semantic review: QARL's fifteen reports cover numerical export,
STE, calibration, optimizer and checkpoint behavior; FP8 training's twenty cover configuration, injection, correction,
profiling, grouped kernels, and optimizer boundaries; and P2P weight sync's thirty-four cover handshake, slicing,
placement, failure propagation, and teardown. Their size reflects distinct contracts rather than branch-level reporting.

All 56 changed focused reports pass. Repository collection is now 1,089 items and the static inventory is 1,080
definitions with 921 curated decisions across 332 Python test files. The audit still contains 25 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-sixty-sixth wave: join branch reports at the component boundary

The one-hundred-and-sixty-sixth wave consolidates sixteen reports across data, loss, model, server, optimizer,
checkpoint, and weight-sync areas without removing an assertion. Packing allocation invariants now run with the real
`PackingDataset` lifecycle, and the empty-mask `TokenPartial` case is part of its denominator and composition contract.
Legacy softmax and configuration selection now form one TopK-router policy matrix, while distinct balanced,
sqrt-softplus, and hash modes remain separate.

Server runtime configuration now owns R3 transport and adapter-gradient bucket roundtrips and rejection. Exact GLM
construction reports one attention admission matrix and one complete-MoE admission matrix instead of separate reports
for dependency flags, EP16, lm-head TP16, sparse MLA, all-to-all, and rank-one alpha-one branches. Successful inventory,
post-EP ownership, and numerical execution reports remain independent.

The same rule folds path validation into kill-session checkpoint promotion, custom-group rejection into multi-part
optimizer behavior, and declaration authority errors into the adapter-ownership compiler's fail-closed policy. EP
checkpoint dimension restore and drop now form one mesh contract. Empty PP-NCCL transfer is part of its tensor-roundtrip
protocol, while sparse-delta priming and receiver-failure retry run in the baseline state-machine lifecycle; post-packed
transfer and initialization stay separate.

All 36 resulting focused reports pass. Repository collection is now 1,073 items and the static inventory is 1,064
definitions with 931 curated decisions across 332 Python test files. The audit still contains 25 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-sixty-seventh wave: group policy matrices around one production entry point

The one-hundred-and-sixty-seventh wave consolidates eight branch-level reports without removing an assertion. Packing
and sequence-shard collators now report generic fields, token-aligned side fields, pre-shifted labels, packed boundaries,
padding, and FlashAttention metadata at their respective component boundaries. The large CP16 side-channel contract
remains separate because it exercises a materially different topology.

R3 reference validation now runs with put, sliced-load, and cleanup ownership, while the low-level Mooncake tensor codec
remains independent. Sync-quantization acceptance and rejection are one configuration policy matrix. NCCL store-bind
failure is part of rendezvous initialization and port lifecycle, and P2P status timeout is part of asynchronous size and
cutoff dispatch; the distinct prepare-request timeout stays separate. Finally, cross-attention cu-seqlens rejection now
runs with the page-size-one SGL KV-cache adapter instead of appearing as a standalone attention behavior.

The focused selection collects 15 reports: all 13 runnable reports pass and two FlashAttention-dependent reports retain
their existing environment skips. Repository collection is now 1,065 items and the static inventory is 1,056 definitions
with 937 curated decisions across 332 Python test files. The audit still contains 25 legitimate conditional runtime gates,
one intentional duplicate group, and no parse errors.

## One-hundred-and-sixty-eighth wave: make lifecycles the reporting boundary

The one-hundred-and-sixty-eighth wave consolidates nine reports around seven production lifecycles without dropping an
assertion. Rank-zero ready handling now reports acknowledgements, early requests, client identity, unexpected messages,
and receive failure together. Runner load-state admission and artifact-root confinement run with multi-adapter and
single-tenant routing. FIFO admission, dispatch, capacity, terminal transitions, statistics, clearing, and bounded history
now form one scheduler report, using fresh instances to keep scenarios isolated.

ModelRunner's multi-adapter Adam override is now part of the same full, partial, omitted, and non-Adam optimizer-step
policy. ParallelState defaults and validation run with singleton initialization, automatic DP sharding, access, and
reinitialization protection; EP mesh helpers remain separate. DeepEP internode preflight now reports its skip gates,
transport diagnostics, identity roundtrip, and corruption detection as one outcome matrix, while topology discovery and
buffer sizing remain independent.

Finally, generic ParallelPlan indivisibility is the rejection branch of successful meta slicing, and exact-GLM malformed
singleton dispositions are part of the exact meta EP plan. The materialized real-tensor shard stays separate because it
executes DTensor redistribution. All 15 resulting focused reports pass. Repository collection is now 1,056 items and the
static inventory is 1,047 definitions with 944 curated decisions across 332 Python test files. The audit still contains
25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-sixty-ninth wave: report codecs and pair buffers as transactions

The one-hundred-and-sixty-ninth wave consolidates nine static reports across checkpoint loading, quantization, and
shared-prefix execution without removing an assertion. Exact absorbed-KV-B checkpoint loading now treats arrival order,
duplicate members, incomplete pairs, dtype mismatch, and shape mismatch as outcomes of one native-FP8 pair-buffer
transaction. The exact-attention source inventory stays separate because it validates construction rather than loader
state.

Block-FP8 quantization and dequantization now form one CUDA codec contract covering geometry, scales, input admission,
roundtrip error, determinism, storage, magnitude edges, signs, and dimensional consistency. The GKN codec similarly
reports output layout, aligned and tail blocks, zero blocks, large matrices, output dtype, contiguity, and rank admission
together instead of splitting quantize and dequantize reports.

For shared-prefix attention, singleton membership and a one-token prompt are one edge-layout report; the general dtype,
head-size, GQA, forward, and backward matrix remains separate. The CPU repacker now includes the one-token empty-shared
block in its full detection, repack, and remap lifecycle. All five runnable resulting reports pass, while the FA3-only
module retains its existing collection skip. Repository collection is now 1,048 items and the static inventory is 1,038
definitions with 949 curated decisions across 332 Python test files. The audit still contains 25 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventieth wave: keep admission with the component it admits

The one-hundred-and-seventieth wave consolidates seven reports without dropping an assertion. Exact dense gate-up
checkpoint loading now reports out-of-order successful emission together with missing, duplicate, invalid-dtype, and
non-finite members as one pair-buffer transaction. The fused native bytes, scale order, model installation, and loaded
state remain checked.

NVFP4 fake quantization now has one two-dimensional contract for independent reference parity, STE behavior, and input
admission, plus one three-dimensional expert contract for projection STE, expert-isolated scaling, and fused gate-up
per-half scales. DSV4 tensor-parallel rejection now runs with attention storage and backend-call dtype behavior, while
window-only and C128 forward-backward variants remain parameterized separately.

Eager-versus-native MoE determinism and the all-tokens-to-one-expert edge now run with the same forward and backward
parity matrix. Non-gated MoE constructor rejection similarly belongs to the CPU eager-reference contract; its optional
GPU Triton and native comparisons remain separate. All 10 resulting focused reports pass. Repository collection is now
1,041 items and the static inventory is 1,031 definitions with 954 curated decisions across 332 Python test files. The
audit still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-first wave: keep configuration with runtime selection

The one-hundred-and-seventy-first wave consolidates four CPU reporting boundaries without removing an assertion.
Gradient-checkpoint method defaults and overrides now run with the layer's training, enabled, and full-recompute gate
matrix. Ordinary and MoE method propagation remain covered alongside the exact checkpoint-call truth table.

Kimi wrapper conversion, official auxiliary defaults, DeepSeek-V3 registry resolution, and local text-config unwrapping
now form one configuration-loading lifecycle. The tokenizer auto-loader similarly reports the dedicated local TikToken
path together with generic tokenizer and processor fallback, retaining token IDs, text roundtrip, right padding, and the
rule that fallback loaders never gain implicit remote-code trust.

Finally, sqrt-softplus scaling and requested dtype now run with unchanged softmax gather and renormalization as one
`MoEBlock._regather_routing` mode matrix. All four resulting focused reports pass. Repository collection is now 1,037
items and the static inventory is 1,027 definitions with 958 curated decisions across 332 Python test files. The audit
still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-second wave: keep zero and masked branches with the operation

The one-hundred-and-seventy-second wave consolidates seven reports without removing an assertion. FlashMLA's all-invalid
case now runs with valid-row compaction, TileLang backward, and zero-scatter behavior as one autograd transaction. The
separate dispatch-envelope report remains because it validates device and production geometry admission.

Causal-LM Z-loss now reports positive coefficient, zero coefficient, and tensor-parallel rejection in one CPU policy,
retaining reference CE and Z-loss values, finite gradients, absent zero-coefficient metrics, and failure before any
collective. Compiled CUDA parity remains separate.

Streaming forward-KL now reports dense-reference gradients, chunk invariance, ignore-index loss and gradient masking,
and low-memory parity as one kernel contract. OPD backend parity and unsupported logprob clamping form one dispatch
contract, while the independent FP64 gradcheck remains its own numerical oracle. All eight resulting focused reports
pass. Repository collection is now 1,030 items and the static inventory is 1,020 definitions with 961 curated decisions
across 332 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate
group, and no parse errors.

## One-hundred-and-seventy-third wave: report loss objectives, not plumbing branches

The one-hundred-and-seventy-third wave consolidates nine reports without removing an assertion. Per-token CE now reports
local and TP LM-head module selection, FP32 bypass, and temperature as one policy matrix. Importance sampling and
causal-LM loss wrappers now share one LM-head dispatch contract covering module use, FP32 bypass, TP collectives, and
finite hidden and head gradients.

Dr.GRPO's zero-advantage, all-ignored, empty-sequence, positive-advantage, missing-reference, and KL-penalty branches now
run with its forward, backward, and metric contract. Temperature-driven behavior K3 and microbatch composition remain
separate numerical reports.

Fused selected-logprob parity now includes frozen-output input gradients and irregular tails. Per-token CE, causal-LM,
quack-linear, and importance-sampling selection form one dispatcher integration report, while production-vocabulary
finiteness and the no-full-logits memory bound remain independent heavy regressions. All nine resulting focused reports
pass. Repository collection is now 1,021 items and the static inventory is 1,011 definitions with 964 curated decisions
across 332 Python test files. The audit still contains 25 legitimate conditional runtime gates, one intentional duplicate
group, and no parse errors.

## One-hundred-and-seventy-fourth wave: report invariant matrices at the kernel boundary

The one-hundred-and-seventy-fourth wave consolidates eight reports without removing an assertion. Families-v2 RMSNorm
now reports FP64 tree agreement, batch-composition invariance, and repeat determinism as one numerical contract. Its
fused-versus-split realization and runtime dispatch remain separate because they exercise different implementations.

The BI fused LM-head contract now keeps forward and backward parity together with determinism, batch invariance, and
input guards. Unit-temperature identity and the near-one probability clamp form one edge-policy report. TileLang indexer
causal masking now covers both large finite values and zero inputs in the same edge matrix, while the parameterized
forward geometries remain independent kernel-shape reports.

Finally, NF4 codebook exactness runs with the flat quantize-dequantize codec transaction; the GKN layout remains separate
because it has a different storage boundary. All 12 resulting focused reports pass. Repository collection is now 1,013
items and the static inventory is 1,003 definitions with 968 curated decisions across 332 Python test files. The audit
still contains 25 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-fifth wave: collapse branches at their production boundary

The one-hundred-and-seventy-fifth wave consolidates 17 reports without removing an assertion. OPD diagnostics, reward
weighting, clamps, stable metric keys, and top-k rejection now form one full-vocabulary policy. GDN convolution forward
and backward parity form one numerical contract. Exact TP1 QLoRA now reports configuration and runtime admission,
forward and surrogate-backward reference parity, and backward safety as three transactions rather than six fragments.

FP8 linear padding and correction modes now share one matmul numerical matrix. TileLang sparse-MLA reports attention-sink
parity and effect together, and checks partially invalid indices across forward and backward in one masking transaction.
Exact routed experts similarly report global and owner-local factor banks as one sampler-buffer policy, with zero-token
and all-sentinel gradients in one routed edge policy. Canonical GLM52 MoE configuration and runtime selection now share
one mode report.

Exact shared-expert construction and runtime admission now form one component gate, while logical FP32 masters and
immutable checkpoint binding form one persistent-state policy. Optional SGLang factor-view parity remains separate from
native base views so missing SGLang cannot suppress the native report. Finally, generic nesting and the exact shared
expert now exercise the topmost mixed-precision FSDP selector together; reduce dtype, sequence-parallel folding, and
prefetch remain separate policies.

All 43 runnable resulting reports pass, with eight existing optional or platform skips. Repository collection is now 996
items and the static inventory is 986 definitions with 977 curated decisions across 332 Python test files. The audit now
contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-sixth wave: report policy matrices instead of individual fields

The one-hundred-and-seventy-sixth wave consolidates 11 reports without removing an assertion. Model-runner token
diagnostics now report selection, empty and top-k boundaries, loss-logprob cross-checks, raw-weight references, and
hidden-state summaries as one callable contract. KL diagnostics, tensor persistence, component hooks, and trusted-input
resolution remain separate because they cross distinct implementation boundaries.

Stochastic BF16 rounding now reports API admission, seeded repeatability, expectation, and neighboring-value bounds in
one numerical contract. Constant, linear, and cosine learning-rate schedules form one builder mode matrix, while invalid
configuration remains an independent admission report. DistSignSGD local hooks, FSDP-managed exclusion, and unsupported
parallel-topology rejection similarly form one configuration transaction; reduce-scatter arithmetic and optimizer
construction remain separate.

The session API now reports ordinary LoRA teardown, checkpoint URI return, re-registration, and default-session kill and
unload protection as one termination lifecycle. Inference endpoint registration now covers default and explicit worker
ports, adapter routing, auto-sync, discovered topology, and FP8 KV-cache admission together. Weight-sync quantization
admission and FP8 KV-cache invalidation form one sync request policy, while endpoint listing and receiver enrichment stay
separate.

All 19 resulting focused reports pass. Repository collection is now 985 items and the static inventory is 975 definitions
with 983 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional runtime gates,
one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-seventh wave: report the P2P protocol by state machine

The one-hundred-and-seventy-seventh wave consolidates 15 reports without removing an assertion. P2P FP8 transfer now
has one receiver-layout matrix covering fused and unfused attention, partial blocks, Qwen3.6 QKVZ and full-attention
layouts, nonexpert namespaces, mixed FP8 and passthrough entries, shared experts, and routed experts. Every source slice,
receiver byte, scale tensor, dequantized value, endpoint, and expert-coverage assertion remains.

Multi-sender initialization now reports rank-zero filtered scatter, nonzero-rank adoption, explicit sender process groups,
peer-failure propagation, and optional engine prewarm ordering as one state-machine contract. Locator copy modes, dense
sharding, and rank-filtered transfer stay separate because they determine data partitioning rather than initialization.

Transfer manifest rejection and compatible-name resolution now form one receiver-manifest policy. Small CPU pooling,
GPU-direct persistent registration and chunking, and aligned mixed-dtype scratch views form one source-staging policy;
receiver-handle coalescing and failure diagnostics remain independent scheduling and observability reports. Finally,
pending failures, optional receiver completion, cleanup draining, endpoint results, deregistration, and completion errors
now form one P2P teardown lifecycle.

All 19 reports in the complete P2P protocol module pass. Repository collection is now 970 items and the static inventory
is 960 definitions with 988 curated decisions across 332 Python test files. The audit still contains 24 legitimate
conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-eighth wave: keep topology matrices and lifecycle outcomes together

The one-hundred-and-seventy-eighth wave consolidates 11 reports without removing an assertion. The lm-head TP FSDP
end-to-end suite now has one distributed matrix for CP-replica DP1, CP plus DP2, no-CP DP, no-CP HSDP, and the matching
OPD DP and HSDP modes. Each of the six four-process programs still runs through the same embedded eager loss and gradient
oracle, and failures retain a case identifier.

Adapter gradient epochs now report empty-step rejection, idempotent abort, scratch reset, publication state, and poisoned
or pending abort rejection as one pre-mutation lifecycle. A successful authoritative optimizer step now includes its
analytical clipping, AdamW parameter and moment updates, scratch reuse, global-step commit, and exact single logical-norm
collective. Semantic rejection, partial optimizer failure, and collective failure form one outcome policy retaining each
branch's recoverability, poison, mutation, and publication assertions.

Adapter checkpoint save policy now keeps trusted-root confinement with strict target-manifest persistence and mismatch
validation. Authoritative restore now reports lifecycle reset, fingerprint restoration, compatible plan replacement,
direct topology mismatch rejection, and atomic nonmutation together. Coordinator materialization, general session
compatibility, and checkpoint structure stay separate because they cross different orchestration or input boundaries.

The distributed topology matrix and all 17 reports in the complete adapter-manager module pass. Repository collection is
now 959 items and the static inventory is 949 definitions with 994 curated decisions across 332 Python test files. The
audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-seventy-ninth wave: keep resume and backend matrices intact

The one-hundred-and-seventy-ninth wave consolidates 10 reports without removing an assertion. Adapter optimizer shard
emission and manifest identity rejection now form one save contract. Bitwise uninterrupted resume, immediate moment
restore, weights-only divergence, evicted public reload, scheduled learning-rate restoration, and explicit LR override
form one resume lifecycle. Legacy or incomplete artifact admission now also proves that a failed restore leaves an
already-resident adapter unchanged.

Grouped FP8 same-NK forward and same-MN weight-gradient kernels now form one training arithmetic report. Block-loop and
Triton references, empty groups, irregular tails, nondefault block sizes, precomputed sequence offsets, and scalar-Quack
dispatch all remain covered. SGLang fused-expert EP activation now reports DeepEP and FP8 exclusion, missing runtime,
flag-off stock routing, score dtype, empty ranks, and compute guards together; happy-path compute, slot combination,
autograd ownership, and weight presentation stay separate.

Finally, the model-runner expert-factor compiler now has one matrix for eager and fused unquantized backends, registered
session-rank specialization, block-FP8 DeepEP, NF4, NVFP4, and generic quantized contracts. Producer families,
quantization guards, metadata mismatch, factor-shape drift, and uncertified parallelism remain asserted for every branch.

All 19 runnable focused reports pass, with one existing optional-platform skip. Repository collection is now 949 items
and the static inventory is 939 definitions with 998 curated decisions across 332 Python test files. The audit still
contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eightieth wave: keep calibration and loading workflows whole

The one-hundred-and-eightieth wave consolidates six reports without removing an assertion. Qwen-235B simulator Markdown
ingestion, leave-one-out calibration evaluation, exact and extrapolated gradient-accumulation scenarios, observed OOM
boundaries, topology what-if cases, and automatic topology sweeps now form one calibration workflow. Built-in pack replay
also runs the consolidated validator across every shipped pack; portable analytical ledgers, path security, and kernel
correctness-gated ranking remain separate simulator boundaries.

Grouped checkpoint loading now reports dense and expert routing, fused and FFN source formats, local dense-group
fallback, missing EP-group fallback, strict fallback rejection, and persistent-buffer filtering together. State-dict
resolution, transport, DTensor materialization, and strict post-processing stay separate because they exercise different
callables.

FP8 weight-sync projection inclusion, module exclusions, receiver skip lists, broad selector mode, stacked quantization,
and already-FP8 passthrough now form one input and layout policy. CPU expert projection now includes zero padding,
deferred formatting, exclusions, reusable workspace staging, and workspace quantization in one expert pipeline.

All 21 reports in the three complete focused suites pass. Repository collection is now 943 items and the static inventory
is 933 definitions with 1,001 curated decisions across 332 Python test files. The audit still contains 24 legitimate
conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eighty-first wave: report component and side-payload lifecycles

The one-hundred-and-eighty-first wave consolidates nine reports without removing an assertion. Weight-sync quantization
rejection and FP8 BF16-island enrichment now form one `handle_sync_inference_weights` admission policy. Request-processor
Mooncake externalization, datum-order preservation, and normal, exceptional, and default cleanup now form one R3
side-payload lifecycle; NCCL synchronization, dispatcher forwarding, optimizer and checkpoint operations, and token
unpacking remain separate request boundaries.

GLM5 indexer geometry, exact FP32 projection, sentinel and padding masking, and sorted and blocked selection now form one
component contract. GLM52 logical selection, Hadamard transport, fused projection, sampler key preparation, portable
codecs, runtime dispatch, and dependency loading similarly form one sparse-selector pipeline. Production-shape SGLang
CUDA codec parity is now an independent optional report, so its runtime skip cannot mask the portable assertions.

The four complete focused suites resolve 35 reports as passing with one existing optional-platform skip. Repository
collection is now 934 items and the static inventory is 924 definitions with 1,005 curated decisions across 332 Python
test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no
parse errors.

## One-hundred-and-eighty-second wave: qualify complete optimizer and model programs

The one-hundred-and-eighty-second wave consolidates nine reports without removing an assertion. Muon's builder now keeps
Gram-Newton-Schulz configuration, invalid grouped-byte admission, and state-free SGD fallback execution together. Fused
gate-up detection, FSDP parameter replacement, gated and non-gated model classification, and a real Nemotron-H optimizer
step now form one parameter-ownership policy; backend dispatch and matrix-group arithmetic remain separate reports.

Distributed checkpoint metadata admission and synchronous, no-dist, custom-group, and asynchronous load and save routing
now form one I/O policy. Optimizer metadata-key selection moved to the optimizer-state filtering report, while model-key
and pipeline-LoRA compatibility remain independent schema contracts.

Canonical GLM52 numerical resolution and official geometry now form one exact model program. Qwen3.5 numerical, MoE,
topology, and model-scope admission similarly form one exact training-program contract; the family-independent RoPE
selector remains separate. Finally, quantized-export YAML and CLI precedence, size parsing, module invocation, BF16
islands, output configuration, and sharded indexing now report one end-to-end command workflow.

All 20 reports in the four complete focused suites pass. Repository collection is now 925 items and the static inventory
is 915 definitions with 1,010 curated decisions across 332 Python test files. The audit still contains 24 legitimate
conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eighty-third wave: report API and payload lifecycles end to end

The one-hundred-and-eighty-third wave consolidates eight reports without removing an assertion. Create-model session
normalization, recreation admission, registration rollback, reserved-checkpoint initialization, and full-weight admission
now form one endpoint lifecycle. Direct adapter loading now keeps success, trusted-path admission, synchronized failure,
pipeline rejection, and auto-registration rollback together. Rank-zero broadcast routing, sharded restore, session-spec
mismatch, and transactional optimizer rejection similarly form one load-mode policy.

Packing now reports empty, missing, oversized, NumPy, valid, and malformed input handling with output validation. Its
per-token unpack modes run with the full pack, metadata, simulated-forward, and sample-boundary round trip. Teacher-cache
CP and EP contributor selection, legacy duplicate mode, sequence-parallel trimming, and cross-rank writer gathering now
form one distributed producer policy; Mooncake storage remains a separate transport boundary.

Finally, OPD teacher causal shifting, cache-index alignment and rejection, and Mooncake metadata admission form one
pipeline payload contract. Endpoint reuse, student-version verification, and preparation-worker queueing remain separate
orchestration reports.

All 25 reports in the five complete focused suites pass. Repository collection is now 917 items and the static inventory
is 907 definitions with 1,015 curated decisions across 332 Python test files. The audit still contains 24 legitimate
conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eighty-fourth wave: keep checkpoint and trust boundaries whole

The one-hundred-and-eighty-fourth wave consolidates eight reports without removing an assertion. Runtime-rank LoRA
export, PEFT hybrid-shared layout, SGLang shared-outer layout, and adapter-manager loading for hybrid-shared and all-owner
experts now form one unquantized checkpoint workflow. Low-level EP slicing and the quantized projection-subset round trip
remain separate representation boundaries.

Successful optimizer publication, commit and handler-tail poisoning, and rank-zero fatal termination now form one
post-mutation lifecycle. Empty logical packing, discovery, replica classification, and ownership compilation form one
empty-shard layout policy. Coordinate, replica, LoRA-B, session, and FQN-order invariance similarly form one deterministic
adapter initialization contract; real Gloo and explicit EP composition remain separate topology gates.

Server artifact-root confinement, symlink rejection, and private diagnostic-input admission now form one filesystem trust
boundary. Compile-target allowlisting, safe IPC type round trips, and oversized-frame rejection form one compile-worker
trust boundary, while outbound endpoint validation remains independent.

All 16 reports in the four complete focused suites pass, including the real two-rank Gloo layout check. Repository
collection is now 909 items and the static inventory is 899 definitions with 1,019 curated decisions across 332 Python
test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no
parse errors.

## One-hundred-and-eighty-fifth wave: report versioned numerical and manager contracts

The one-hundred-and-eighty-fifth wave consolidates 11 reports without removing an assertion. Adapter-manager optimizer
construction, hyperparameter persistence, learning-rate updates, and mixed-rank multi-optimizer reload now form one
configuration lifecycle. Session compatibility, weights-only behavior, checkpoint structure, PEFT suffixes, sharded
indices, and rank-capacity admission form one load compatibility policy.

FSDP topmost protected-module selection, expert policy stripping, mesh-dependent reduction dtype, explicit overrides,
and dtype admission now form one mixed-precision contract. MiniMax M3 clamped SwiGLU, biased sigmoid routing, text
forward and backward, multimodal-token rejection, and parallel-mode admission similarly form one runtime program.

All v1 BI family, normalization, mean, softmax, matrix, and LM-head frozen hashes now report as one versioned golden-tree
gate; v2 normalization and head hashes form another. FlashQLA packed-versus-individual rows, total-token invariance, and
block-DV tile invariance now form one Gate 2 contract. Auto-CP and Gate 4 state handoff remain separate decisions. Six
inert capability decorators were also removed from helper functions; capability admission remains on the collected
versioned parent reports.

All 28 reports in the five complete focused suites pass on the available GPU. Repository collection is now 898 items and
the static inventory is 888 definitions with 1,024 curated decisions across 332 Python test files. The audit still
contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eighty-sixth wave: keep numerical oracles and transport codecs together

The one-hundred-and-eighty-sixth wave consolidates ten report definitions without removing an assertion. QK, pre-summed
residual-tree, post-attention residual, zero-centered family-1, and families-v2 RMSNorm site classes now form one
cross-engine bitwise matrix. The module-level SGLang dependency gate remains unchanged. Qwen3.5 Class-B dispatch and
fail-closed admission form one rotary policy, while dense and MoE half-rotate behavior and post-RoPE BF16 casting form one
attention projection policy.

Mooncake byte codecs, canonical dtype strings, metadata emission, suffixed keys, and rank-2 and rank-3 hidden fetches now
form one transport contract. Fused GDN canonical LoRA folding, slice-local gradients, exact projection, cache reuse,
bounded generations, and old-generation release form one merged-weight lifecycle.

Exact fused gated-RMSNorm routing, unsupported residual rejection, and full GatedDeltaNet routing now form one model
program dispatch contract. BI router GEMM FP32 agreement, empty and dtype admission, and hidden and weight gradients form
one numerical contract. Their lower-level gating, normalization, solve, top-k, and model-integration boundaries remain
separate.

The six runnable consolidated reports pass; all 19 reports in the runnable affected modules pass, with the cross-engine
module producing one expected dependency skip. Repository collection is now 892 items and the static inventory is 878
definitions with 1,030 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eighty-seventh wave: report component lifecycles instead of implementation fragments

The one-hundred-and-eighty-seventh wave consolidates nine report definitions without removing an assertion. RMSNorm family
name admission, residual-shape rejection, and Qwen site declarations now form one structural API policy; funnel equivalence,
family vitality, zero-centered folding, and module dispatch form one numerical-routing contract. Undeclared-family
enforcement remains a separate CUDA tripwire. SGLang-fused residual and no-residual forward and backward comparisons now
form one fused numerical contract, while CPU fallback, model integration, and trunk dispatch remain separate boundaries.

DeepSeek V4 now reports shared-MLP and routed-expert SwiGLU-limit propagation together. C128 execution, causal-LM forward
and backward, hash-layer input threading, and decoder gradient-checkpoint wrapping form one model runtime contract;
construction, topology admission, and precision preservation remain separate. Nemotron H strict loading, ignored MTP
admission, HF parity, and exact save reconstruction now form one published-layout codec transaction with the supported save
key set kept explicit.

Qwen3 per-expert and Qwen3.5 stacked fused-expert layouts now share one checkpoint-handler policy, including deferred QLoRA
expert loading. QLoRA quantized storage, forward and backward, dequantization, prequantized NVFP4 loading, EMA scale
convention, and merge-requantization now form one quantized-weight lifecycle; injection, block-FP8 representation, and
optimizer reset remain distinct.

All 24 reports in the seven affected modules pass. Repository collection is now 883 items and the static inventory is 869
definitions with 1,037 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-eighty-eighth wave: join inverse codec paths and configured execution

The one-hundred-and-eighty-eighth wave consolidates eight report definitions without removing an assertion. Copying full
checkpoint tensors into existing replicated or sharded DTensors and materializing multi-axis DTensors for a selected writer
now form one load/save tensor codec. Object payload transport, NCCL device selection, weight-load group routing, and
handler-filtered rank-zero loading similarly form one broadcast loading transaction. State-dict discovery and grouped expert
loading remain independent resolution boundaries.

Muon builder keyword propagation, byte-limit admission, SGD fallback, a real Gram-Newton-Schulz update, and restart
autotuning now form one configured optimizer lifecycle. Quack selection, grouped scheduling, standard Newton-Schulz, CUDA
compute dtype, and model classification remain separate algorithm or platform branches. BI trunk-linear persistent-GEMM
forward bits, batch invariance, dtype admission, and cuBLAS input, weight, and bias gradients now form one wrapped-linear
numerical contract; wrapper selection and global-interpose admission remain separate.

DeepSeek V3 external per-expert and internal fused checkpoint layouts, dense and packed EP slicing, requested device and
dtype, and quantization-config discovery now form one expert codec policy. Nemotron H router output, loss backward through
Mamba, attention, MoE, and shared experts, and full-layer gradient checkpointing now form one training runtime contract;
packed variable-length equivalence remains its own state-propagation boundary. RoPE registry-wide FP32 frequency
construction, BF16 consumption, and unchanged exact-lane cosine and sine bits now form one precision policy, while lazy
cache growth and architecture-specific CUDA placement remain separate.

All 20 reports in the six affected modules pass, including the four-rank Gloo materialization and CUDA BI gradient checks.
Repository collection is now 875 items and the static inventory is 861 definitions with 1,043 curated decisions across 332
Python test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no
parse errors.

## One-hundred-and-eighty-ninth wave: make state transitions own their reports

The one-hundred-and-eighty-ninth wave consolidates eight report definitions without removing an assertion. DistSignSGD
FSDP2 builder admission, decay grouping, hook configuration, and a preaggregated update now form one configured optimizer
contract; reduce-scatter sign timing and local-versus-FSDP gradient ownership remain independent communication boundaries.
SignSGD builder construction, decay grouping, dense updates, decoupled decay, and sparse-gradient rejection similarly form
one local optimizer contract.

Trainer sequence-parallel sums, DTensor skipping, adapter-finalization exclusions, LM-head replica gradient handling, and
marked-parameter broadcast now form one explicit synchronization policy. Dispatcher forward-backward rendezvous and commit,
explicit gradient-epoch abort, uniform rejection, and rank-asymmetric failure conversion form one gradient-epoch lifecycle;
session, save, and post-optimizer publication operations remain separate mutation boundaries.

Dense residual, attention, normalization, and MLP hooks plus routed-MoE callbacks and shared-expert components now report as
one hidden-component capture pipeline. Summary formatting, ranked tensor dumps, and trusted diagnostic overrides remain
separate output and filesystem boundaries. DeepSeek V3 auxiliary router-logit emission and replay recording of selected
indices and weights similarly form one router observability contract, while full-model backward, router freezing, and LoRA
injection remain independent.

Local and pipeline checkpoint key discovery, metadata unions, QARL buffer mismatch, base-to-LoRA loading, and LoRA-only
loading now form one model-compatibility policy; distributed transport and optimizer payload filtering remain separate.
Finally, MoE expert-sorted routing weights and scatter-add reconstruction now form the encode and decode directions of one
memory-efficient token permutation codec, while all-to-all ordering and hidden chunking remain separate transports.

All 26 reports in the eight affected modules pass. Repository collection is now 867 items and the static inventory is 853
definitions with 1,051 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-ninetieth wave: include class-method suites in semantic ownership

The one-hundred-and-ninetieth wave expands the report-density audit to class-method suites and consolidates nine report
definitions without removing an assertion. Qwen3.5 families-v2 effective folded-weight gradients and the residual twin's
output and residual gradient paths now form one zero-centered backward contract. Legacy FP8 config normalization,
incompatible external runtime rejection, and explicit Blackwell validation-artifact admission form one configuration
boundary; BF16 layer-island transformation remains separate.

Quack unique PTX outputs, bounded ptxas execution, cleanup, and exact entry discovery now form one compilation process-
safety contract, while worker framing and cache hashing remain separate trust boundaries. NVFP4 independent 2D reference
agreement, shape admission, linear STE, 3D expert STE, expert isolation, and fused gate-up scale ownership now form one
fake-quant contract.

Merged-LoRA canonical linear and expert folds plus straight-through factor gradients now form one low-level numerical
contract. LoraLinear merged selection, gradient parity, optimizer-step invalidation, and active-rank cache invalidation form
one linear lifecycle. MoE canonical merged views, parameter-version caches, and fused-expert admission form one expert
lifecycle; native EP execution and trunk wrapping remain separate integrations.

MoE-LoRA backend initialization, frozen and trainable ownership, runtime rank slicing, from-module conversion, model
injection, and block injection now form one construction policy. Zero-delta base equivalence and eager-versus-native or
Triton output and gradient agreement form one GPU numerical policy under the same capability gate. CPU eager execution,
zero-token structural gradients, and EP router-score application remain distinct runtime boundaries.

All 17 reports in the six affected modules pass, including the GPU cross-backend MoE-LoRA checks. Repository collection is
now 858 items and the static inventory is 844 definitions with 1,057 curated decisions across 332 Python test files. The
audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-ninety-first wave: join transport transactions and kernel contract states

The one-hundred-and-ninety-first wave consolidates eight report definitions without removing an assertion. P2P cold and
cached prepare behavior now form one initialization handshake. Successful completion, pending-transfer failure, receiver
notification, deregistration, and destroy cleanup form one terminal synchronization lifecycle; fanout, slicing, coalescing,
diagnostics, and multi-sender routing remain separate transport boundaries.

NCCL initialization and transfer endpoint ports plus its optional two-phase receiver protocol now form one endpoint
transaction policy. Flat, chunked-flat, and receiver-fenced hybrid buckets form one flattened load-format contract, while
endpoint health and invalid multi-rank direct format remain independent admission boundaries.

GDN packed convolution weights, armed routing, fail-closed admission, call-scoped contract state, and checkpoint
recomputation now form one exact-contract lifecycle. Low-level CUDA parity, full-block integration, and optional SGLang
tree-kernel parity remain distinct numerical boundaries. Fixed-length and variable-length FlashAttention calls now form one
API behavior contract. SGL, paged FlashAttention, flags-off, and FA4 selection form one page-size-one KV-cache routing
policy; backend resolution and eager head-layout numerics remain separate.

The four affected modules collect 29 reports: 27 pass and two have expected capability or optional-dependency skips.
Repository collection is now 850 items and the static inventory is 836 definitions with 1,061 curated decisions across 332
Python test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and
no parse errors.

## One-hundred-and-ninety-second wave: make stateful APIs own their complete lifecycle

The one-hundred-and-ninety-second wave consolidates seven report definitions without removing an assertion. Empty and
aborted adapter epochs, a successful clipped update, nonfinite input, optimizer failure, and collective failure now form one
authoritative optimizer lifecycle. Capture ownership, publication admission, exact LM-head coherence, and checkpoint restore
remain separate. Sharded optimizer-manifest emission, identity rejection, moment restoration, and bitwise continuation now
form one checkpoint codec; artifact admission and logical cross-layout resharding remain independent compatibility gates.

Sampling-session stale-state reconciliation, transient query failure, model-scoped tracking, and failed-load atomicity now
form one adapter lifecycle. Sampler checkpoint storage and adapter-only export remain separate. Inference weight-sync
endpoint forwarding, pool selection, quantization admission, and cache invalidation now form one API transaction, while
endpoint registration, health refresh, and receiver-capability detection remain separate.

Model-session normalized registration, duplicate and topology admission, reserved checkpoints, kill, optional final save,
and default-session protection now form one create-to-destroy lifecycle. The lightweight create-session alias remains a
separate endpoint. Disk-backed weights information, path admission, full-weight metadata, and legacy SignSGD upgrade now
form one checkpoint session-spec decoding policy.

All 30 reports in the five affected modules pass. Repository collection is now 843 items and the static inventory is 829
definitions with 1,067 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-ninety-third wave: report complete artifact and distributed contracts

The one-hundred-and-ninety-third wave consolidates six report definitions without removing an assertion. Fused QKV, MLA,
linear-attention, GKN expert, and fused gate-up transformations now form one exported-model tensor-layout contract. Exporter
CLI and directory behavior, source admission, QARL folding, and the low-level FP8 quantizer remain separate boundaries.

Expert-adapter backend capability and plan identity plus factor ownership, reduction domains, and checkpoint persistence
now form one structural contract. Exact target-subset forwarding and GLM, Qwen3, and Qwen3.5 wrapper construction form one
injection policy. Supported SiLU preservation and rejection of incompatible activations, biases, quantization groups,
target sets, and model-family semantics form one fail-closed semantic contract; runtime numerical parity remains separate.

Canonical MoE trainer and sampler plan identity, topology admission, logical ordinals, and world-32 CP, EP, and expert-FSDP
group aliases now form one topology contract. Its 2- and 8-contributor dense transport and 16-contributor packed and
CP-sharded parity now form one distributed numerical contract; every subprocess still runs.

All 12 reports in the three affected modules pass, including the 2-, 8-, and 16-process transport checks. Repository
collection is now 837 items and the static inventory is 823 definitions with 1,073 curated decisions across 332 Python test
files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse
errors.

## One-hundred-and-ninety-fourth wave: join objective branches and dispatch transactions

The one-hundred-and-ninety-fourth wave consolidates seven report definitions without removing an assertion. Evicted
auto-load, fresh materialization, explicit path load, rollback, and all-rank or rank-zero-broadcast restoration now form one
adapter load lifecycle; registration and save remain separate mutations. Token selection, boundary behavior, raw-weight
cross-checking, hidden summaries, and CP-sharded KL position mapping now form one token-diagnostics policy, while capture,
tensor-dump output, and trusted override input remain separate.

OPD reference, streaming, low-memory, and sharded-store forward agreement plus backward, partial reduction, and output dtype
now form one numerical backend contract. Fused selected-logprob dtype, bias, temperature, frozen-head, irregular-tail, Qwen,
and GPT-OSS vocabulary cases similarly form one forward and backward numerical contract; loss dispatch and the no-full-
logits memory gate remain separate.

DRGRPO forward values, gradients, metrics, zero boundaries, advantage direction, KL penalty, and logprob-temperature behavior
now form one objective contract, with microbatch reducer composition kept independent. Packing concatenation, capacity
splitting, mixed lengths, empty and single input, oversize admission, missing fields, NumPy conversion, and microbatch
validation now form one core packing policy; metadata, disabled mode, and full roundtrip remain separate.

All 19 reports in the six affected modules pass, including the production-vocabulary and no-full-logits GPU gates.
Repository collection is now 830 items and the static inventory is 816 definitions with 1,079 curated decisions across 332
Python test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and
no parse errors.

## One-hundred-and-ninety-fifth wave: make tensor preparation and save formats whole

The one-hundred-and-ninety-fifth wave consolidates six report definitions without removing an assertion. Receiver
postprocess selection, FP8 KV-cache requirements, unsupported-format admission, and generated BF16 islands now form one
quantized weight-sync configuration contract. Adapter materialization, parameter filtering and tied aliases, compile-name
normalization, and architecture-specific unfusing form one sync-source tensor preparation pipeline; bucket sizing,
transport routing, and sparse-delta selection remain separate.

P2P sender selection, direct-EP collection admission, and gated or nongated local expert projection collection now form one
EP synchronization-source policy, while transport remains independently tested. Factor-only admission, rank-zero artifact
failures, LoRA-only failures, and pre-barrier error surfacing now form one fail-closed checkpoint save policy. Live dense
target resolution and collective stacked-MoE factor slicing form one LoRA checkpoint export contract; optimizer artifacts
remain in the resume suite.

All seven reports in the two affected modules pass. Repository collection is now 824 items and the static inventory is 810
definitions with 1,084 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-ninety-sixth wave: join configuration and backend branches

The one-hundred-and-ninety-sixth wave consolidates six report definitions without removing an assertion. Removed fields,
incompatible quantized modes, vLLM-only runtime knobs, broadcast loading, and unsupported multi-adapter modes now form one
fail-closed server configuration boundary. Canonical defaults, nested runtime controls, receiver cache dtype, R3 transport,
gradient buckets, Muon options, runner compatibility, sparse MLA, and MoE routing controls form one runtime configuration
roundtrip; quantized-training and parallel-topology configuration remain separate.

Teacher-cache contributor selection, CP and DP assembly, valid-label trimming, Mooncake metadata emission, byte roundtrip,
and activation-cache consumption now form one hidden-cache lifecycle. OPD loss execution and debug artifacts remain
separate. Muon builder options, fallback behavior, restart autotuning, grouped shapes, transpose equivalence, fused halves,
and byte-limit chunking now form one configured Gram-Newton-Schulz contract, while Quack selection, standard Newton-Schulz,
and CUDA compute dtype remain separate.

Block-loop and Triton-grouped forward and weight gradients plus scalar-Quack per-expert scaling now form one grouped FP8
GEMM numerical contract. DeepGEMM subprocess isolation and model train-step integration remain separate gates.

The four affected modules collect 21 reports: 20 pass and one DeepGEMM capability gate skips as expected. Repository
collection is now 818 items and the static inventory is 804 definitions with 1,089 curated decisions across 332 Python test
files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse
errors.

## One-hundred-and-ninety-seventh wave: make dispatcher input and completion transactions whole

The one-hundred-and-ninety-seventh wave consolidates four report definitions without removing an assertion. DP, EP, CP,
and legacy rank selection, routing-payload slicing, rank-local row grouping, and source provenance now form one dispatcher
input-distribution policy. Packing strategy and R3 payload storage remain separate upstream boundaries.

Local payload trimming, rank rendezvous, CP replica deduplication, disagreement rejection, and rank-zero per-token merging
now form one dispatcher completion transaction; diagnostic dumping remains a separate output. Processor readiness, forward
and backward execution, timing propagation, invalid-target rejection, shutdown, model identity, auto-load, and R3 forwarding
now form one request-to-runner compute lifecycle. NCCL sync, optimizer and checkpoint RPCs, and payload storage remain
separate operations.

All 11 reports in the two affected modules pass. Repository collection is now 814 items and the static inventory is 800
definitions with 1,092 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-ninety-eighth wave: make adapter capture, restore, and residency transactional

The one-hundred-and-ninety-eighth wave consolidates five report definitions without removing an assertion. Raw-numerator
accumulation, model-gradient clearing, FP32 scratch reuse, staged commit, direct-DTensor preservation, and atomic
prevalidation now form one gradient-capture transaction. Ownership-plan compilation and optimizer mutation remain separate
lifecycle boundaries.

Scalar-tensor LM-head optimizer-state coherence now runs as a distributed validation branch of the authoritative optimizer
lifecycle. Trusted paths, strict target manifests, lifecycle reset, ownership-plan admission, optimizer compatibility,
learning-rate rules, checkpoint structure, missing tensors, rank capacity, and PEFT filename and shard compatibility now
form one adapter checkpoint restore and admission policy. Coordinator-driven materialization remains independently tested.

Mixed ranks and optimizers, adapter switching, training, checkpoint reload, capacity eviction, dirty-state protection,
multi-rank rejection, and save-failure rollback now form one multi-adapter lifecycle.

All seven reports in the affected module pass. Repository collection is now 809 items and the static inventory is 795
definitions with 1,096 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## One-hundred-and-ninety-ninth wave: close model-loading transactions end to end

The one-hundred-and-ninety-ninth wave consolidates two report definitions without removing an assertion. Local directory
resolution and distributed shard-list broadcast now form the input phase of rank-zero checkpoint loading alongside tensor
and metadata transport, process-group selection, and handler-filtered prefetch.

Dense and expert routing, supported expert-key formats, fused and FFN source conversion, process-group fallback, and strict
parameter and persistent-buffer coverage now form one grouped checkpoint-loading transaction. Four-process replicated,
sharded, and target-rank DTensor materialization remains a separate save-side correctness gate.

All three reports in the affected module pass, including both four-process DTensor checks. Repository collection is now
807 items and the static inventory is 793 definitions with 1,098 curated decisions across 332 Python test files. The audit
still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundredth wave: align GLM-5.2 reports with canonical contracts

The two-hundredth wave consolidates three report definitions without removing an assertion. Certified world-16, EP16, and
CP16 topology, the official producer schedule, the 38/40 pipeline split, malformed-plan rejection, and full-indexer
allocation now form one canonical layer-plan contract.

Index publication, reuse, concurrency rejection, exception cleanup, and identity preservation across FSDP mixed-precision
input casting now form one index-share lifecycle. Correction-bias FP32 preservation, meta materialization, strict checkpoint
ingestion, routing-replay rejection, internal transport selection, and exact canonical router and indexer dispatch now form
one canonical MoE configuration and selection contract. Native sampler codec parity and end-to-end semantic logprob
composition remain independent gates.

The affected module collects six reports: five pass and the SGLang-dependent native-codec capability gate skips as
expected. Repository collection is now 804 items and the static inventory is 790 definitions with 1,101 curated decisions
across 332 Python test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-first wave: report fused-MoE choices as complete policies

The two-hundred-and-first wave consolidates three report definitions without removing an assertion. Automatic and explicit
SGLang fused-MoE enablement, supported-module admission, one-time logging, flag-off preservation, and block and
experts-only entrypoint selection now form one resolution and dispatch policy.

Unsupported expert semantics, pre-import clamp rejection, trainable guards, and the gradient-sensitive choice between the
autograd function and plain kernel now form one fused-expert admission and trainable-dispatch contract. Transient, cached,
and zero-copy strided modes, cache reuse and invalidation, serving tensor order, and split gate-up adapter layout now form
one kernel weight-layout contract. Runtime-context admission, gradient numerics, and real-kernel parity remain separate.

The affected module collects six reports: four pass and two GPU capability gates skip as expected. Repository collection
is now 801 items and the static inventory is 787 definitions with 1,104 curated decisions across 332 Python test files. The
audit still contains 24 legitimate conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-second wave: make simulator accounting and calibration whole

The two-hundred-and-second wave consolidates three report definitions without removing an assertion. Topology resolution,
balanced routing, sequence-parallel local shapes, FLOPs, activation storage, and communication bytes now form one
analytical accounting contract.

Config fingerprinting, cached and known-model metadata resolution, calibration-pack containment, built-in prefix
validation, symlink rejection, and restricted local reads now form one input resolution and admission policy. Qwen
markdown ingestion, leave-one-out calibration evaluation, scenario and topology planning, built-in pack replay,
fit-and-OOM feasibility, and consolidated pack validation now form one calibration lifecycle. Generic observed-run
ingestion and correctness-gated kernel ranking remain separate policies.

All five reports in the affected module pass. Repository collection is now 798 items and the static inventory is 784
definitions with 1,107 curated decisions across 332 Python test files. The audit still contains 24 legitimate conditional
runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-third wave: join distributed transport and FP8 sync branches

The two-hundred-and-third wave consolidates six report definitions without removing an assertion or subprocess. Direct
output, shared-owner, and all-owner layouts now form one four-GPU FSDP adapter-gradient ownership contract. Eager, Triton,
native, Quack, NF4, NVFP4, block-FP8, projection-subset, and all-owner cases now form one AllToAll contract. Hybrid-shared,
all-owner, and quantized Quack cases now form one optional DeepEP contract; its dependency gate remains explicit.

SGLang EP top-k-one pair presentation, FP32 routing weights, local weight layout, flag-off behavior, empty-rank handling,
and semantic admission now form one dispatch policy. BF16 islands, block scale and zero-padding semantics, projection
selection, skip lists, stack handling, and existing FP8 values now form one CPU sync-quantization contract. Dense LoRA,
QLoRA, quantized MoE factors, and fused-GDN factors now form one adapter-folding sync-source policy.

The three affected modules collect 13 reports. Ten pass, including the live FP8 GPU policy, and two optional capability
gates skip. The four-GPU FSDP report is not currently verified because rank 3 fails at `torch.cuda.set_device(3)` with an
out-of-memory error before any test assertion or model operation; a direct rerun reproduced that external admission
failure. Repository collection is now 792 items and the static inventory is 778 definitions with 1,113 curated decisions
across 332 Python test files. The audit still contains 24 legitimate conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-fourth wave: make runner compilation one ownership policy

The two-hundred-and-fourth wave consolidates three report definitions without removing an assertion. Generic module and
direct-output ownership, exact TP16 LM-head VJP masks, managed shard capture, replica divisors, group-family coverage, and
fail-closed topology admission now form one runner gradient-ownership compiler policy. Expert-factor compilation remains
a separate specialized contract.

Merged and legacy effective LM-head selection now run as input branches of the direct-output analytical capture and
optimizer step, which checks selected bytes, factor gradients, logical norm, and parameter mutation end to end. Failed
staged-capture rollback remains independent.

All four reports in the affected module pass. Repository collection is now 789 items and the static inventory is 775
definitions with 1,115 curated decisions across 332 Python test files.

## Two-hundred-and-fifth wave: report optimizer state and routed banks as wholes

The two-hundred-and-fifth wave consolidates four report definitions without removing an assertion. Denominator chunking,
Kahan compensation, gradient reuse, CPU state offload, DTensor local-shard wrapping, and device restoration now form one
AnyPrecision AdamW state-strategy policy. Cautious decay math and optimizer construction remain separate.

EP16 and MoE-TP1 admission, all 16-by-16 owner-slot remaps, global and owner-local factor banks, sampler buffer shapes and
dtypes, and unused-rank zero padding now form one routed-bank layout policy. Owned zero-base gradients, all-sentinel
structural zeros, input-layout rejection, and top-k-eight mixed-owner VJPs now form one routed-gradient edge policy. The
full 256-slot literal sampler numerical gate remains separate.

The two affected modules collect eight reports: six pass and two Hopper/SGLang capability gates skip. Repository
collection is now 785 items and the static inventory is 771 definitions with 1,118 curated decisions across 332 Python
test files. The static audit surfaces 22 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixth wave: compile ownership as one transaction

The two-hundred-and-sixth wave consolidates three report definitions without removing an assertion. Generic topology
declarations, authority masks, rank-local tensor and geometry fingerprint invariance, missing and structurally false
declaration rejection, tensor-parallel admission, group identity and membership validation, and complete orthogonal
replica coverage now form one ownership-plan compilation transaction.

Fullgraph module-producer execution and bucketed residual gradient transport remain separate runtime contracts. All three
reports in the affected module pass. Repository collection is now 782 items and the static inventory is 768 definitions
with 1,119 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime gates, one
intentional duplicate group, and no parse errors.

## Two-hundred-and-seventh wave: join exact-QLoRA construction and runtime policies

The two-hundred-and-seventh wave consolidates four report definitions without removing an assertion. Shared-expert
construction, runtime admission, logical state, and checkpoint-state policy now form one shared-expert contract. Physical
SGLang views remain a separate optional-dependency report so the CPU structural policy is never hidden by capability
admission.

Exact TP1 configuration, runtime admission, dtype moves, packed-state master dtype, and parameter identity now form one
configuration and state-lifecycle policy. Forward values, surrogate VJPs, and backward safety now form one numerical and
autograd policy. Request-scoped NCCL group naming now runs inside the orchestrator's optimizer, checkpoint, synchronization,
registration, and lifecycle control report.

The three affected modules collect 13 reports: nine pass and four optional GPU or SGLang capability gates skip. Repository
collection is now 778 items and the static inventory is 764 definitions with 1,123 curated decisions across 332 Python test
files.

## Two-hundred-and-eighth wave: make GLM-5 support reports policy-complete

The two-hundred-and-eighth wave consolidates three report definitions without removing an assertion. GLM-5 indexer
construction and DSA masking now form one indexer-selection policy. Sparse-MLA reference behavior, wrapper semantics, and
attention integration now form one sparse-attention policy. Sparse-KV adapter weights, adapter dispatch, and MoE dispatch
now form one adapter-and-routing policy.

TileLang CUDA execution, checkpoint filtering, Hugging Face parity, and full forward/recompute behavior remain separate
capability or end-to-end gates. All eight reports in the affected module pass. Repository collection is now 775 items and
the static inventory is 761 definitions with 1,126 curated decisions across 332 Python test files.

## Two-hundred-and-ninth wave: report native dispatch and optimizer recovery as transactions

The two-hundred-and-ninth wave consolidates three report definitions without removing an assertion. Entry through the
experts module and its FSDP pre-forward hook now runs inside the native-combine diagnostic report, which also verifies the
actual gathered, routed, gated, local, and combined operands. EP8 admission, variable-row collectives, and fused-gate
gradient parity remain independent contracts.

Canonical optimizer parameter identity, wrapper-insensitive fingerprints, live binding validation, recursive state
snapshots, and failed-collective commit behavior now form one optimizer transaction policy. Successful sharded save and
bitwise resume, manifest identity checks, legacy-pickle rejection, missing-artifact rejection, and resident-state
preservation now form one checkpoint recovery and artifact-admission policy. Logical resharding remains separate.

All seven reports in the two affected modules pass. Repository collection is now 772 items and the static inventory is 758
definitions with 1,129 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-tenth wave: align P2P reports with transfer lifecycle boundaries

The two-hundred-and-tenth wave consolidates three report definitions without removing an assertion. Per-receiver slice
placement and reuse of one staged source across replicated receiver locators now form one receiver-placement policy.
Typed CPU/GPU staging, registration lifetime, alignment, and receiver-handle-aware coalescing now form one staging policy.
Requested flush-cache and weight-version propagation now runs inside sync completion alongside cache retention, tied-weight
aliases, pending-transfer failure handling, and cleanup.

FP8 receiver layouts, invalid-manifest admission, transfer diagnostics, direct-EP transport, initialization, and slicing
remain separate contracts. All 14 reports in the affected module pass. Repository collection is now 769 items and the
static inventory is 755 definitions with 1,132 curated decisions across 332 Python test files.

## Two-hundred-and-eleventh wave: join adapter state and quantized exclusion lifecycles

The two-hundred-and-eleventh wave consolidates three report definitions without removing an assertion. Sampling-adapter
reconciliation, transient endpoint-query handling, model-scoped atomic tracking, listing, deletion, resolution, and
receiver-removal invalidation now form one sampler adapter-state policy. Sampler-weight export remains separate.

Zero-token structural gradients for every local LoRA factor now run as the empty-input edge of eager expert forward,
backward, and MoE-block behavior. Cross-backend numerics and routing-score application remain separate. Prequantized
exclude-module metadata parsing, precedence, malformed input, dense and MoE handler skip behavior, and auxiliary-key
passthrough now form one checkpoint exclusion policy.

All 12 reports in the three affected modules pass. Repository collection is now 766 items and the static inventory is 752
definitions with 1,135 curated decisions across 332 Python test files.

## Two-hundred-and-twelfth wave: make exact construction reports fail-closed

The two-hundred-and-twelfth wave consolidates three report definitions without removing an assertion. Exact GLM-5.2
attention construction now reports its complete 780-factor canonical inventory and rejects invalid rank, alpha, component,
dispatch, and sparse-MLA configurations in one fail-closed construction policy.

Exact GLM-5.2 MoE construction now reports its complete shared and routed inventory, source metadata, shapes, ownership,
and invalid dependency, EP16, LM-head-TP16, rank, and alpha branches together. Post-EP layout and selected-logprob LM-head
specialization remain separate. DeepSeek-V4 construction, pipeline-parallel rejection, parallel-group wiring, FP32 marker
propagation, dtype casts, and complex RoPE preservation now form one construction, topology, and precision policy. Full
forward, backward, hash routing, and checkpoint recomputation remain a separate runtime report.

All six reports in the three affected modules pass. Repository collection is now 763 items and the static inventory is 749
definitions with 1,138 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-thirteenth wave: join configuration and numerics at kernel boundaries

The two-hundred-and-thirteenth wave consolidates three static report definitions and four collected items without removing
an assertion. Before-down and after-down routing-weight numerics, no-router-gradient behavior, lazy configuration, environment
override, automatic regime selection, parity opt-out, explicit settings, and invalid values now form one routing-position
contract.

Zero and nonzero cast-once LoRA folding now reports the same FP32-add invariant across dense linear and MoE expert layouts.
FlashQLA forward output, final state, and all input gradients now form one parity report per head shape against FLA; the two
head shapes and Hopper/TileLang capability gate remain unchanged.

All four affected reports pass, including both live FlashQLA Hopper cases. Repository collection is now 759 items and the
static inventory is 746 definitions with 1,141 curated decisions across 332 Python test files.

## Two-hundred-and-fourteenth wave: report quantized formats by shared invariant

The two-hundred-and-fourteenth wave consolidates three report definitions without removing an assertion. NVFP4 and block-FP8
expert loading now form one prequantized expert-load policy covering packed bytes, scales, global factors, amax values,
projection layout, and dequantization.

Flat and GKN NF4 codebook, packing, scale, zero, shape, and error behavior now form one codec contract. Block-FP8 and NVFP4
GNK-to-GKN transpose equivalence, direct and transposed dequantization, non-square shapes, expert stacking, and global-scale
absorption now form one prequantized layout-conversion policy.

All three live GPU reports pass. Repository collection is now 756 items and the static inventory is 743 definitions with
1,144 curated decisions across 332 Python test files.

## Two-hundred-and-fifteenth wave: group geometry and topology variants by claim

The two-hundred-and-fifteenth wave consolidates three report definitions without removing an assertion or subprocess.
Same-NK and same-MN grouped GEMM numerics, transpose handling, uneven and empty groups, and input admission now form one
kernel-family contract.

Dense 2-D Shard(0) and expert 3-D Shard(1) Muon layouts now form one distributed full-gradient oracle-parity report. The
same two layouts form one shard-local negative-control report. All four two-GPU subprocesses remain and still independently
exercise their original mode and layout.

All three affected reports pass. Repository collection is now 753 items and the static inventory is 740 definitions with
1,147 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime gates, one intentional
duplicate group, and no parse errors.

## Two-hundred-and-sixteenth wave: report Qwen RMSNorm resolution by capability domain

The two-hundred-and-sixteenth wave consolidates three report definitions without removing an assertion. Dense Qwen3.5
RMSNorm structural selection, v1 and v2 dispatch, invalid-mode admission, every zero-centered site, GDN exclusion, layer
input policy, and final norm policy now form one CPU resolution contract.

Qwen3.5-MoE v1 and v2 dispatch, ordinary-mode preservation, every zero-centered site, layer input policy, and final norm
policy likewise form one CPU resolution contract. Its family-1 interpose and full-layer bit parity and family-2 residual
composition now form one GPU bit-exact integration contract. The generic RMSNorm structure, undeclared-family tripwire,
kernel funnel, and dense/MoE capability separation remain independent.

All three affected reports pass. Repository collection is now 750 items and the static inventory is 737 definitions with
1,150 curated decisions across 332 Python test files.

## Two-hundred-and-seventeenth wave: make conversion, compilation, and routing reports end to end

The two-hundred-and-seventeenth wave consolidates four static definitions and five collected items without removing an
assertion. DeepSeek-V4 converter meta-model dtype preservation, ordinary HF-to-DCP roundtrip, legacy sidecar-free LoRA load,
and cross-shard weight/scale deferral now form one conversion policy. AutoModel loading remains a distinct Transformers
integration contract.

MoE-block and decoder-layer compilation across every available expert backend and both compiler backends now form one
lower-level compile-compatibility report. Full-model per-layer composition remains separate. Local unfiltered, local
filtered, and EP SGLang fused-expert backward paths now form one FP32 routing-gradient oracle report; the former local
parameterization is an internal two-case loop so the EP branch runs once rather than twice.

All five affected reports pass, including the live compiler paths. Repository collection is now 745 items and the static
inventory is 733 definitions with 1,154 curated decisions across 332 Python test files.

## Two-hundred-and-eighteenth wave: join registry, wrapping, and native-FP8 state policies

The two-hundred-and-eighteenth wave consolidates three report definitions without removing an assertion. Optional Quack
registration, registry signatures, common backend arguments, activation forwarding, and explicit FP8 admission now form
one EP adapter boundary contract.

Exact-hook merged-LoRA preparation and hybrid-model trunk selection now form one Qwen3.5 structural wrapping policy. The
stale synthetic fixture now declares the v1 family and a resolved norm, restores the nonexact numerical family afterward,
and asserts the current documented behavior that wrapping arms the RMSNorm/trunk contract lane. The isolated live GPU
forward remains a separate execution gate and passes after the state leak is removed.

GLM-5.2 native-FP8 configuration roundtrip and rejection, model replacement, sparse-MLA materialization, pair-buffer bytes
and admission, expert fusion, and grouped checkpoint ownership now form one configuration, model, buffer, and checkpoint
contract. Router dispatch and frozen expert scoring remain separate.

All six reports in the three affected modules pass. Repository collection is now 742 items and the static inventory is 730
definitions with 1,157 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-nineteenth wave: join wrapper layers around one execution claim

The two-hundred-and-nineteenth wave consolidates three report definitions without removing an assertion. Direct Quack MoE
TP-FP8 execution and the MoEExperts Quack wrapper now form one train-step policy across TP reduction, Triton-grouped and
scalar-Quack backends, clamped SwiGLU biases, finite gradients, and master-weight mutation.

Scoring and trainable SGLang EP dispatch, flag and dependency admission, empty-rank behavior, and autograd guards now form
one dispatch policy. Slot combination, weight presentation/cache modes, and live stock-Triton gradient parity remain
separate. Fused RMSNorm residual and no-residual forward/backward kernels, packed shapes, module dispatch, dense Qwen layer
parity, and serving-tree force calls now form one numerical and model-integration report. CPU fallback and trunk-specific
dispatch remain separate.

The three affected modules collect 12 reports: ten pass and two optional DeepGEMM or SGLang capability gates skip.
Repository collection is now 739 items and the static inventory is 727 definitions with 1,160 curated decisions across 332
Python test files.

## Two-hundred-and-twentieth wave: report rotary, MoE activation, and GDN math as whole policies

The two-hundred-and-twentieth wave consolidates four report definitions without removing an assertion. Pairwise-interleaved
rotary numerics, Class-B fused admission and CUDA rejection, dense and MoE attention half-rotate semantics, mRoPE behavior,
and post-RoPE BF16 casting now form one Qwen3.5 rotary policy.

DeepSeek-V4 shared-MLP and routed-expert SwiGLU clamping now run inside the non-hash MoE structure, shared-contribution,
forward, backward, router, and selection-bias report. Hash-table routing and record/replay remain separate. GDN gating and
gated RMSNorm forward, backward, dtype, reference, and row-invariance behavior now form one primitive numerical contract;
exact-model module dispatch and the pinned triangular-solve geometry remain separate.

All seven reports in the three affected modules pass. Repository collection is now 735 items and the static inventory is
723 definitions with 1,164 curated decisions across 332 Python test files.

## Two-hundred-and-twenty-first wave: join state, backward, and dispatch phases

The two-hundred-and-twenty-first wave consolidates three report definitions without removing an assertion. Native block-FP8
byte packing, dtype-preserving state, strict state-dict and DCP metadata, apply rollback, CPU fail-closed execution,
partition-hook entry, scoring-only admission, range validation, and pair validation now form one encoding, checkpoint,
execution, and admission contract.

Sparse-MLA production backward parity to the torch reference and deterministic-versus-atomic gradient parity now form one
backward policy; forward reference parity remains separate. RMSNorm v2 fused-versus-split forced-realization bit identity
and the production dispatch heuristic now form one realization and dispatch policy; one-ULP reference correctness, batch
composition invariance, and run-to-run determinism remain separate.

All five affected reports pass, including the live H100 TileLang paths. Repository collection is now 732 items and the
static inventory is 720 definitions with 1,167 curated decisions across 332 Python test files. The static audit surfaces 22
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-twenty-second wave: join primitive admission and topology branches

The two-hundred-and-twenty-second wave consolidates six report definitions without removing an assertion or subprocess.
Class-B RoPE dtype admission, shape and partial-rotary backward behavior, and unique-half table layout now form one primitive
contract. EP backend declarations and parameter-metadata domains now form one fail-closed admission policy, while the real
two-rank reduction remains separate. Exact dense, projection, and LM-head components now share one legacy merged-sync
rejection report; separate-factor publication remains distinct.

Qwen3.5 Ulysses execution and ring-plus-FLA rejection now run as two subprocess branches of one context-parallel contract.
NVFP4 and block-FP8 loading, execution, merge, and requantization now form one quantized QLoRA format lifecycle. The affected
reports pass, including live CUDA and two-GPU execution. Repository collection is now 726 items and the static inventory is
714 definitions with 1,173 curated decisions across 332 Python test files.

## Two-hundred-and-twenty-third wave: make builder and artifact reports end to end

The two-hundred-and-twenty-third wave consolidates five report definitions without removing an assertion. Full-model FP8
construction, tensor-parallel LM-head selection, incompatible-mode admission, QARL construction, and calibration order now
form one quantized model-builder lifecycle; scoped monkeypatch contexts preserve the former report isolation. Sequence and
ring routing-replay layouts now form one context-parallel layout contract. Sparse-delta encoding, source capture, path
admission, and rank/global manifests now form one artifact lifecycle.

DeepSeek-V4 FP8 and MXFP4 decoding and EP-aware handler ownership now form one checkpoint conversion policy, while synthetic
end-to-end loading remains separate. All six affected reports pass. Repository collection is now 721 items and the static
inventory is 709 definitions with 1,178 curated decisions across 332 Python test files.

## Two-hundred-and-twenty-fourth wave: report model support as complete integrations

The two-hundred-and-twenty-fourth wave consolidates six report definitions without removing an assertion. Active-LoRA atomic
flag mutation and composite admission now form one state policy; server derivation, topology rejection, and cached indexer
and MoE activation now form one propagation policy. Nemotron-H published per-expert and Transformers stacked expert layouts
now run through one bidirectional checkpoint-handler contract. LoRA manifest target selection and fail-closed schema and
runtime validation now form one manifest policy.

Qwen2 and OLMo2 configuration conversion, architecture construction, tensor-parallel unfusing, checkpoint translation, and
HF parity now each form one architecture-support report. All seven affected reports pass. Repository collection is now 715
items and the static inventory is 703 definitions with 1,184 curated decisions across 332 Python test files.

## Two-hundred-and-twenty-fifth wave: join process, operator-edge, and packing policies

The two-hundred-and-twenty-fifth wave consolidates six static report definitions and five collected items without removing
an assertion. Quack worker framing, PTXAS timeout and temporary-output handling, entry selection, and safe deterministic
cache hashing now form one compilation process-safety policy. Shared-prefix multi-member and singleton behavior now form one
forward and backward equivalence report; its unchanged optional FA3 module gate means neither former definition contributed
to repository collection in this environment.

OPD ignored-token, per-token, hidden-only, full-materialization, and chunked-fetcher behavior now form one edge contract.
Packing strategy validation, oversized handling, document and token preservation, capacity, utilization, determinism, and
datum order now form one generic packing policy; balanced-DP scheduling remains separate. The five runnable affected reports
pass, and shared-prefix attention skips at its unchanged optional dependency gate. Repository collection is now 710 items
and the static inventory is 697 definitions with 1,190 curated decisions across 332 Python test files. The static audit
surfaces 22 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-twenty-sixth wave: join public service lifecycles

The two-hundred-and-twenty-sixth wave consolidates six report definitions without removing an assertion. Tinker session
schema publication, creation, follow-up requests, heartbeat activity, and canonical LoRA aliases now form one public
endpoint lifecycle. Mooncake tensor storage, R3 reference publication, selective loading, validation, and cleanup now form
one side-payload lifecycle. Adapter checkpoint fail-closed admission, write failures, and successful dense and MoE live
factor publication now form one save policy with scoped patch isolation.

Launcher address discovery and worker readiness now form one control lifecycle. P2P size-based dispatch, status timeout,
prepare timeout, and request payload now form one async API policy. K3 tail metrics and temperature-matched behavior logprobs
now form one observability policy across both loss implementations. All nine affected reports pass. Repository collection
is now 704 items and the static inventory is 691 definitions with 1,196 curated decisions across 332 Python test files.

## Two-hundred-and-twenty-seventh wave: report storage and quantization end to end

The two-hundred-and-twenty-seventh wave consolidates ten report definitions without removing an assertion. Mooncake hidden
tensor codecs, metadata, teacher consumption, malformed and legacy admission, removal, and configuration precedence now form
two transport and store policies. Dense QARL codec parity, injection, summaries, configuration normalization, and model
admission now form one policy. NVFP4 MoE identity-preserving conversion, eager execution, gradients, passthrough, injection,
target selection, and format admission likewise form one expert lifecycle.

QARL sync configuration success and mismatch reporting now form one sync policy. Generic expert backend capabilities,
factor ownership, and preserved semantics now form one adapter contract. Teacher-head discovery, sharded storage, cross-shard
views, residency, dtype reload, and prefetch now form one head lifecycle. Exact-server trunk wrapping and numerical-family
selection now form one pre-parallelization program policy. All ten affected reports pass. Repository collection is now 694
items and the static inventory is 681 definitions with 1,206 curated decisions across 332 Python test files.

## Two-hundred-and-twenty-eighth wave: join export, pipeline, and trainer admission layers

The two-hundred-and-twenty-eighth wave consolidates eight report definitions without removing an assertion. NVFP4 packed
codec correctness and full directory export now form one exporter contract. FP8 CLI configuration, base-directory behavior,
architecture-specific layouts, preflight, and QARL-fold rejection now form one command contract; primitive quantization and
trained-logprob preservation remain separate.

OPD endpoint identity and student-version verification now form one endpoint-admission policy, while chunk queueing, causal
payload shifting, cache-index alignment, and Mooncake transport form one preparation policy. DeepSeek-V3 router-freeze
construction and downstream TP rejection now form one training-admission policy. Class-B selection and the canonical GLM-5.2
numerical program now form one configuration policy, while exact Qwen3.5 remains separate. Simulator topology, shapes,
analytical ledgers, configuration fingerprints, model metadata, and path admission now form one trusted-input policy. All 13
affected reports pass. Repository collection is now 686 items and the static inventory is 673 definitions with 1,214 curated
decisions across 332 Python test files. The static audit surfaces 22 conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-twenty-ninth wave: join distributed planning and control policies

The two-hundred-and-twenty-ninth wave consolidates nine report definitions without removing an assertion or subprocess.
DeepEP node-span detection, preflight skip and roundtrip behavior, failure diagnostics, NVL alignment, and RDMA byte admission
now form one internode transport policy with scoped patches. Generic EP meta slicing and replicated-gradient metadata now
form one plan application; exact GLM meta and materialized already-local factor dispositions form another. Pipeline FQN
partitioning, rank placement, schedule metadata, and microbatch admission now form one layout policy.

Busy-interval union and P2P-byte estimation now form one PP accounting policy, while patching and live CUDA execution remain
separate. Muon full-gradient oracle parity and shard-local divergence now form one distributed policy while preserving all
four two-GPU subprocesses. DeepSeek-V4 C128 and C4 compression admission now form one context-parallel regime policy. All
nine affected reports pass, including PP CUDA events and the Muon subprocesses. Repository collection is now 677 items and
the static inventory is 664 definitions with 1,223 curated decisions across 332 Python test files.

## Two-hundred-and-thirtieth wave: report GPU kernel variants as whole contracts

The two-hundred-and-thirtieth wave consolidates five report definitions without removing an assertion. BI fused LM-head
loss parity, gradients, determinism, batch invariance, guards, unit-temperature identity, and near-one logprob clamping now
form one selected-logprob contract. Full and dimension means now form one batch-invariant reduction policy. Head-v2
projection and statistics bits, batch invariance, fused CE, gradients, and rollback now form one lifecycle.

GDN primitive forward and backward numerics now run with exact-model module dispatch, while triangular-solve geometry remains
separate. Quack EP parity against Triton now runs with its independent half-concatenated activation reference; CPU gradient
arity remains separate. All seven affected live GPU reports pass. Repository collection is now 672 items and the static
inventory is 659 definitions with 1,228 curated decisions across 332 Python test files.

## Two-hundred-and-thirty-first wave: join model construction and runtime boundaries

The two-hundred-and-thirty-first wave consolidates ten report definitions without removing an assertion. Runner side-channel
conversion, ragged padding, and sequence sharding now form one batch-materialization policy. Exact dense managed factors and
routed fail-closed ownership now form one adapter-gradient policy. DeepSeek-V4 codec, handler ownership, synthetic loading,
construction, topology, precision, forward, backward, routing, and recomputation now form two complete checkpoint and model
contracts with scoped patch isolation.

Exact attention source inventory and native pair-state behavior now form one checkpoint contract. Canonical routed and shared
MoE partials now form one boundary policy. Native-FP8 serving routing and frozen scoring-only experts now form one runtime
policy. Nemotron-H EP ownership, layout conversion, HF parity, and save behavior now form one checkpoint contract. Fused
RMSNorm ordinary and trunk GPU integration now form one policy while CPU fallback remains separate. RoPE registry recipes and
lazy native caches now form one CPU precision policy while exact serving-device execution remains separate. All 13 affected
reports pass, including fused RMSNorm CUDA execution. Repository collection is now 662 items and the static inventory is 649
definitions with 1,238 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-thirty-second wave: join residual failure and lifecycle phases

The two-hundred-and-thirty-second wave consolidates eight report definitions without removing an assertion or subprocess.
Adapter-gradient pre-rendezvous, ModelRunner tail, and publication-commit failures now form one bounded CPU failure policy;
the asymmetric post-mutation GPU boundary remains separate. Live two-rank clipping and nonfinite behavior now run with the
three-rank participation topology as one distributed clipping policy.

FutureStore creation, processing, concurrency, model operations, expiration, and cleanup now form one async lifecycle; a
fresh identical store preserves the former fixture isolation between its phases. Orchestrator client roundtrip, interleaving,
edge errors, and shutdown likewise form one communication lifecycle. Sparse-delta initialization and runtime loading, FP8
LM-head selection and loss dispatch, and sequence-shard core and side-channel materialization each now form one policy. All
15 affected reports pass, including the five distributed subprocesses. Repository collection is now 654 items and the static
inventory is 641 definitions with 1,246 curated decisions across 332 Python test files.

## Two-hundred-and-thirty-third wave: report CPU boundary policies end to end

The two-hundred-and-thirty-third wave consolidates five report definitions without removing an assertion. Server and CLI
sequence boundaries, int32 metadata, original-position preservation, stale-metadata replacement, LCM padding, post-shard
divisibility, and padded unpacking now form one sequence-metadata and padding policy. AnyPrecision AdamW cautious numerics,
chunked state updates, gradient reuse, and DTensor offload now form one optimizer lifecycle. Inference endpoint registration,
worker and FP8 KV-cache admission, auto-sync, and health-aware listing now form one public endpoint lifecycle.

All eight resulting reports in the three affected modules pass. Repository collection is now 649 items and the static
inventory is 636 definitions with 1,251 curated decisions across 332 Python test files. The static audit surfaces 22
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-thirty-fourth wave: join transport, input, and support lifecycles

The two-hundred-and-thirty-fourth wave consolidates nine report definitions without removing an assertion. P2P prepare,
cached-map behavior, fanout, cleanup, receiver completion, source staging, invalid manifests, and transfer diagnostics now
form three initialization, staging, and failure policies. FP8 receiver-layout handling remains separate from generic staging.

Microbatch splitting, epoch delegation, builder batch sizing, sampler configuration, sequence-parallel insertion, and custom
collators now form one data-loader construction policy. Dataset name and shard expansion, type inference, splitting, and
merging now form one dataset-composition policy; raw loading and preprocessed persistence remain separate. MiniMax-M3
configuration, registration, text forward/backward, and unsupported-input admission now form one architecture-support
report. CPU attention backend selection now runs with eager head-layout numerics, while optional FlashAttention paths remain
separate. FP8 training, block-FP8 QLoRA, QARL, aliases, defaults, and incompatible combinations now form one low-precision
argument policy. All 23 runnable affected reports pass and the unchanged FlashAttention capability report skips. Repository
collection is now 640 items and the static inventory is 627 definitions with 1,260 curated decisions across 332 Python test
files.

## Two-hundred-and-thirty-fifth wave: join direct-EP, FP8, and SSD execution branches

The two-hundred-and-thirty-fifth wave consolidates five report definitions without removing an assertion. Direct-EP
multi-sender initialization, scatter-copy ownership, dense and expert manifest partitioning, and rank-filtered transfers now
form one lifecycle across rank-zero, nonzero, failure, process-group, engine-order, and empty-rank behavior. FP8Linear padded
matmul recipes and correction numerics now culminate in live CUDA forward, backward, and master-weight mutation under their
shared capability gate. Dense and packed SSD recurrence, boundary-safe convolution, and packed mixer behavior now form one
CPU recurrence policy; unavailable-kernel admission and live GPU kernel parity remain separate.

All 16 runnable affected reports pass and the unchanged optional SSD kernel report skips. Repository collection is now 635
items and the static inventory is 622 definitions with 1,265 curated decisions across 332 Python test files. The static audit
surfaces 22 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-thirty-sixth wave: join runner, storage, optimizer, and FP8 lifecycles

The two-hundred-and-thirty-sixth wave consolidates seven report definitions without removing an assertion. Adapter
coordinator materialization and auto-load now run with checkpoint restore, optimizer compatibility, trusted paths, lifecycle
reset, overrides, and structural admission using a fresh temporary subtree. OPD microbatch loss, gradients, FSDP LM-head
anchoring, metric aggregation, reductions, extrema, and empty-rank key alignment now form one execution-and-metrics policy.
RequestProcessor forward/backward now includes global packed-row batching and its rank-local and routing-replay boundaries.

Model-scoped and sampler-scoped checkpoint listing, deletion, isolation, resolution, adapter reconciliation, tracking, and
normalized adapter-only export now form one public storage lifecycle with an explicitly fresh APIServer for the sampler
phase. Muon Gram-Newton-Schulz configuration and grouping now include Quack import, dispatch, and dtype selection. Direct
Quack FP8 expert variants now culminate in injected dense-expert-dense forward, backward, and master-weight mutation. All 24
runnable affected reports pass and the unchanged opt-in DeepGEMM report skips. Repository collection is now 628 items and the
static inventory is 615 definitions with 1,272 curated decisions across 332 Python test files.

## Two-hundred-and-thirty-seventh wave: join component primitives through their integrations

The two-hundred-and-thirty-seventh wave consolidates 13 report definitions without removing an assertion. Canonical LoRA
folding, straight-through gradients, LoraLinear selection and cache invalidation, and MoE gate-up/down merged-weight caches
now form one CPU fold policy; native EP and trunk integrations remain separate. GDN delta-linear product correctness now
feeds sliced canonical folding, gradients, projection, and bounded cache behavior.

Exact LM-head per-token and causal-loss routing, weight and server module selection, and FSDP replicated-factor admission now
form one complete loss policy. Absorbed-KV native state, logical masters, dtype moves, identity, and fail-closed direct
projection form one CPU component policy, while its official CUDA Q/V program remains separate. Canonical MoE capacity
metadata, transport admission, trainer/sampler hashes, topology, and group layouts now form one planning policy; the
distributed reduction subprocess remains separate. BI router GEMM, leading-dimension linear behavior, top-k normalization,
and exact-versus-stock MoEBlock dispatch now form one live CUDA routing contract. All 11 runnable affected reports pass and
the unchanged absorbed-KV CUDA capability report skips. Repository collection is now 615 items and the static inventory is
602 definitions with 1,285 curated decisions across 332 Python test files. The static audit surfaces 22 conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-thirty-eighth wave: join server preparation, transfer, and diagnostic lifecycles

The two-hundred-and-thirty-eighth wave consolidates 11 report definitions without removing an assertion. NCCL endpoint
initialization, two-phase completion, flattened and chunked buckets, hybrid receiver fencing, and multi-rank direct-format
admission now form one transfer policy. Weight-sync adapter selection, parameter extraction, inference layout, bucket sizing,
direct-EP sender mapping, and tensor-collection gating now form one source-preparation policy. Shipped MoE LoRA and QLoRA
examples now culminate the quantized server-configuration contract while retaining clean-process parsing.

Sharded adapter packing, empty-shard ownership, deterministic initialization, and explicit-EP layout discovery now form one
CPU state policy; the real two-rank Gloo DTensor boundary remains separate. Dispatcher registration, nonresident saves,
gradient-epoch completion and abort, optimizer publication, poisoning, and fatal failures now form two session and mutation
lifecycles. Dense and MoE diagnostic hooks now feed the token diagnostic computation they support, while ranked tensor dumps
and trusted override loading form one artifact-boundary policy. All 15 resulting reports in the six affected modules pass.
Repository collection is now 604 items and the static inventory is 591 definitions with 1,296 curated decisions across 332
Python test files.

## Two-hundred-and-thirty-ninth wave: join packing, publication, and orchestration lifecycles

The two-hundred-and-thirty-ninth wave consolidates 10 report definitions without removing an assertion. P2P hostname
selection and fallback engine construction now begin the existing initialization, prepare, fanout, cache, completion, and
cleanup lifecycle. Clean mid-epoch weight admission and strict checkpoint rejection now run inside the authoritative adapter
optimizer lifecycle. Packed teacher, cache, weight, hidden-state, and RL metadata now traverse the full pack, forward, and
unpack pipeline, and token-diagnostic unpacking now closes the RequestProcessor model-pass R3 payload lifecycle.

Optimizer, forward, and forward-backward API response shaping now form one public training-operation policy. Runner
gradient-ownership compilation now includes staged-capture abort on forward-backward failure. OPD cache-row and last-k
weight shaping now feed live loss, gradient, and profiling execution. Balanced packing and sequential dummy behavior now run
inside dispatcher batch distribution, while routing references and microbatch diagnostic dumps form one side-payload
artifact and security policy. Orchestrator initialization, successful operations, errors, concurrency, statistics, and
shutdown now form one end-to-end lifecycle. All 28 resulting reports in the nine affected modules pass. Repository
collection is now 594 items and the static inventory is 581 definitions with 1,306 curated decisions across 332 Python test
files.

## Two-hundred-and-fortieth wave: join optimizer, routing, FP8, and exact-component policies

The two-hundred-and-fortieth wave consolidates 13 report definitions without removing an assertion. Cautious decay
primitives, SignSGD, AnyPrecisionAdamW state strategies, Muon post-Newton-Schulz masking, AdamW fallback, and builder routing
now form one optimizer feature policy. Synthetic balanced, sqrtsoftplus noaux, token-hash, legacy softmax, scaling,
configuration, and FP32 MoEBlock routing now form one TopK router contract. Optional boolean admission now runs with parallel
mixed-precision and reduce-dtype configuration, while sequence-parallel folding and manual prefetch remain separate.

FP8 injection, recipes, exclusions, CPU fallback, output dtype, and fail-fast behavior now form one CPU module policy. CUDA
operand profiling now culminates the live FP8 matmul, correction, backward, and master-weight mutation report under the same
hardware gate; the CPU profiler remains separately runnable. GLM52 canonical MoE configuration now closes the sparse
selector and codec pipeline, while layer-plan allocation and semantic parity remain separate. Exact dense MLP factor
ownership, runtime admission, forward composition, XoRL load, and PEFT export now form one component lifecycle. Fourteen
runnable reports pass and the unchanged optional GLM52 capability report skips. Repository collection is now 581 items and
the static inventory is 568 definitions with 1,319 curated decisions across 332 Python test files.

## Two-hundred-and-forty-first wave: join packing, loading, profiling, and loss execution

The two-hundred-and-forty-first wave consolidates 12 report definitions without removing an assertion. Sample positions,
trainable-token filtering, dataset preprocessing, allocation, PackingDataset construction, caching, and missing-column
admission now form one data-packing lifecycle. Pipeline interval merging, P2P byte accounting, instance patching,
restoration, and patch admission now form one CPU profiling policy; the live CUDA GPipe report remains separate.

Requested-key-only shard reads, exact merged and expert key plans, missing-pair admission, per-module deferred loads, and
bounded cache release now form one prequantized QLoRA loader lifecycle. NVFP4 and block-FP8 detection, quantized-key skipping,
QKV and bias merging, exclusion parsing, and dense and MoE handler behavior now form one checkpoint policy. Fused selected
logprob numerics now flow through CE, causal LM, Quack, and importance-sampling dispatch while the memory-bound regression
remains separate. Streaming forward-KL dense parity, chunking, masking, low-memory execution, OPD dispatch, and clamp
admission now form one execution policy while fp64 gradcheck remains separate. DistSign communication, hook ownership,
topology admission, construction, grouping, and stepping now form one optimizer lifecycle. Batch-invariant trunk forward and
backward now culminate in global-interpose gradient rejection and no-grad admission under the same CUDA gate. All 12
resulting reports pass. Repository collection is now 569 items and the static inventory is 556 definitions with 1,331
curated decisions across 332 Python test files.

## Two-hundred-and-forty-second wave: join clipping, MoE, QLoRA, and replay lifecycles

The two-hundred-and-forty-second wave consolidates 16 report definitions without removing an assertion. Shared-replica and
skip-FSDP ownership, norm modes, empty gradients, raw local clipping, EP-aware dispatch, ordinary fallback, and mixed-mesh
foreach behavior now form one CPU clipping policy; real two- and three-rank reductions remain separate. MoE histogram,
expert-slot indexing, deterministic ordering, escape-hatch behavior, scatter, gather, add-gather, and roundtrip execution now
form one CUDA kernel policy. BI GEMM table neutrality now culminates in row invariance across M buckets, while optional
DeepGEMM parity remains separate.

QLoRA injection, NVFP4 and block-FP8 execution, format loading, merging, optimizer-state reset, and interval integration now
form one CUDA lifecycle. Exact DCP key projection now feeds an official base checkpoint into runtime state, while four-rank
staging and skip-mode admission remain separate. Exact shared-expert construction now includes native TP16 base slicing,
while optional SGLang factor views and Hopper execution remain separate. MoE LoRA construction, ownership, injection, eager
execution, zero-token gradients, and mocked EP score application now form one CPU component policy; cross-backend CUDA
numerics remain separate. Routing replay now joins asynchronous record ordering, MoEBlock replay, router gradients,
multi-layer and 1F1B schedules, base-model checkpoint enabling, and R3 preload under one CUDA lifecycle; its CPU registry
unit report remains separate. Thirteen runnable reports pass and three unchanged optional capability reports skip. Repository
collection is now 553 items and the static inventory is 540 definitions with 1,347 curated decisions across 332 Python test
files.

## Two-hundred-and-forty-third wave: join checkpoint, exact-construction, and component lifecycles

The two-hundred-and-forty-third wave consolidates 18 report definitions without removing an assertion. DTensor copy and
four-rank materialization, object transport, rank-zero filtered prefetch, local resolution, grouped expert routing, and
strict coverage now form one checkpoint-load lifecycle. Pipeline key unions, QARL buffer admission, base-to-LoRA
compatibility, optimizer-key filtering, multi-optimizer loading, metadata, load groups, and save groups now form one model
state lifecycle. Exact MoE global inventory now proceeds through EP placement, logical owner shapes, and selected-logprob
head attachment as one construction policy.

Fused GDN manifest geometry now feeds low-rank products, canonical folding, gradients, bounded caches, export, and sharded
PEFT restore. FlashMLA flattening, invalid-index normalization, valid-row backward compaction, all-invalid behavior, and
production-envelope admission now form one hermetic policy. Exact TP1 construction and admission now run with CPU forward,
surrogate backward, and safety while the CUDA direct program remains separate. RMSNorm family tripwires now precede the
bitwise CUDA funnel while CPU structure remains separate. Fused MoE registration, Qwen checkpoint roundtrips, deferred
expert skipping, QKV unfusing, and QARL filtering now form one export policy. OPD full-vocab modes, VERL estimators,
policy-gradient behavior, and compiled sampled logprobs now form one loss policy. GDN convolution primitive parity now
culminates in end-to-end block output and gradients while optional SGLang parity and CPU admission remain separate. Twelve
runnable reports pass and three unchanged optional capability reports skip. Repository collection is now 535 items and the
static inventory is 522 definitions with 1,365 curated decisions across 332 Python test files.

## Two-hundred-and-forty-fourth wave: join fused-kernel and architecture-support policies

The two-hundred-and-forty-fourth wave consolidates 14 report definitions without removing an assertion. SGLang fused-MoE
resolution, block dispatch, admission, trainable dispatch, weight presentation, cache behavior, kernel layout, and runtime
context now form one CPU policy, while stock-Triton and masked gradients culminate in the existing optional real-SGLang
parity gate. Sparse-MLA attention-sink arithmetic and effect coverage now run in the representative first forward
specialization without multiplying the four compiled top-k cases.

DeepSeek-V3 forward, backward, router freezing, default and explicit LoRA targets, router observability, and routing replay
now form one tiny-model lifecycle. DeepSeek-V4 non-hash routing, shared experts, SwiGLU clamps, hash-table routing, and
record/replay backward now form one architecture MoE policy. MiniMax-M3 configuration, registry, text execution, checkpoint
mapping, EP expert ownership, paged-KV layout, and CPU MSA admission now form one support report. GLM52 full-block QLoRA
inventory now includes EP-local routed banks plus exact-component and training-mode admission, while routed-expert literal
owner-slot coverage now culminates in sentinel and mixed-owner VJPs under its existing hardware gate.

Six CPU/meta reports, four sparse-MLA forward specializations, both isolated sparse backward edge reports, and the FP8
grouped-kernel report pass. Optional SGLang and SM100 reports skip on this H100 lane. The larger sparse-MLA backward
specializations and the Quack FP8 optimizer-step report terminate the current pytest process; both are retained as real
failure boundaries, and the passing edge/kernel reports remain separate so those terminations cannot erase their signal.
Repository collection is now 521 items and the static inventory is 508 definitions with 1,381 curated decisions across 332
Python test files. The static audit surfaces 19 conditional runtime gates, one intentional duplicate group, and no parse
errors.

## Two-hundred-and-forty-fifth wave: join model-support and EP-combine transactions

The two-hundred-and-forty-fifth wave consolidates 10 report definitions without removing an assertion. GLM5 configuration,
registry loading, unsafe-value admission, indexer construction and selection, sparse-MLA reference and Ulysses integration,
checkpoint filtering, default adapter targets, EP MoE dispatch, absorbed-KV LoRA execution, tiny-model forward, and
recompute-before-dispatch now form one hermetic architecture-support policy. The real TileLang indexer fast path and HF
logit reference remain separate because they exercise external implementations.

Qwen3.5 native EP8 admission now proceeds through variable-row token and ID collectives, trainer gradients through the
serving fused gate, FSDP module entry, routed and shared partials, chain summation, diagnostic actual operands, and final
output as one mocked transaction. SGLang EP flag and backend admission, empty-rank and trainable dispatch, slot-ordered
combine, pair-count guards, and transient, cached, and strided weight presentation now form one CPU policy; the optional
real stock-Triton gradient report remains separate. FlashQLA Gate 2 and Gate 4, the four unrelated training-utility APIs,
and GLM52's five CPU, CUDA, topology, and semantic boundaries were reviewed and retained as distinct contracts.

All five runnable affected reports pass and the unchanged paired-SGLang gradient report skips. Repository collection is now
511 items and the static inventory is 498 definitions with 1,394 curated decisions across 332 Python test files. The static
audit surfaces 19 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-forty-sixth wave: join dispatch, export, optimizer, and sparse-sync lifecycles

The two-hundred-and-forty-sixth wave consolidates eight report definitions without removing an assertion. MoE token and
routing-weight permutation now flows through mocked pre-dispatch all-to-all ordering, expert cumsums and score gradients,
then chunked and unchunked post-dispatch output and gradient parity. Packing-on capacity, batching, OPD and RL metadata,
generated labels, simulated forward output, and per-sample unpacking now form one roundtrip; packing-disabled remains a
separate supported mode with its own shifting, warning, and loss-mask behavior.

Trained QARL state and exact dequantized target logprobs now precede the quantized directory and CLI artifact matrix, while
the low-level FP8 block quantization contract stays separate. Server Adam defaults and validation now feed full, partial,
omitted, adapter, and non-Adam optimizer mutation and finally dispatcher payload forwarding. Sparse-delta encoding,
baseline priming and rollback, prepacked per-rank posting, cache metadata, endpoint accounting, post-only admission, and
runtime helper loading now form one backend lifecycle with isolated temporary directories.

All seven resulting reports pass. Repository collection is now 503 items and the static inventory is 490 definitions with
1,404 curated decisions across 332 Python test files. The static audit surfaces 19 conditional runtime gates, one
intentional duplicate group, and no parse errors.

## Two-hundred-and-forty-seventh wave: join parsing, dataset, P2P, and FP8-sync policies

The two-hundred-and-forty-seventh wave consolidates eight report definitions without removing an assertion. Optimizer,
packing, and numeric argument parsing now proceeds through Muon kwargs, EP checkpoint compatibility, automatic checkpoint
resolution, optimizer-state loading, FP8 aliases, fail-fast fallback, runtime-knob rejection, and low-precision mode
admission as one configuration lifecycle with isolated argv and environment contexts.

Dataset name and shard expansion plus type inference now feed train-validation splitting, merge modes, local-file and saved
directory loading, hub, URL, and data-files routing, preprocessed persistence, reload, and missing-cache behavior. P2P trainer
IB-device precedence now proceeds through abort marker publication, peer observation and cleanup, then distributed
success/failure status gathering. FP8 weight-sync selection and block layout now feed dense and MoE adapter folding,
projection and skip-list policy, CPU expert transposition and padding, deferred quantization, streaming workspace and flush
behavior; its live GPU parity and device-transfer report stays separate.

Checkpoint CRUD versus model-ID validation, weight-sync receiver versus source versus sparse transport, and inference
registration versus synchronization versus quantization schema were reviewed and retained as distinct public boundaries.
All five resulting reports, including live GPU FP8 execution, pass. Repository collection is now 495 items and the static
inventory is 482 definitions with 1,415 curated decisions across 332 Python test files. The static audit surfaces 19
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-forty-eighth wave: join adapter, checkpoint, dispatcher, and endpoint lifecycles

The two-hundred-and-forty-eighth wave consolidates nine report definitions without removing an assertion. Adapter
registration, session materialization, broadcast, cross-rank rollback, worker failure, fresh and evicted loading, explicit
path admission, rank-zero restore, and save refusal for missing evicted state now form one coordinator lifecycle. Optimizer
parameter identity and transactional collective failure now feed sharded manifest creation, artifact admission, bitwise
resume, one- and multi-dimensional logical resharding, replicated layouts, topology changes, and invalid-source rejection.

Checkpoint manager materialization and zero-meta gates now proceed through model-runner initial base restore, step counters,
optimizer selection, restore failure publication, and guarded fresh default-adapter initialization. Dispatcher batch
sharding, EP and CP provenance, packing and dummy behavior now feed filesystem and Mooncake routing payloads, security and
diagnostic artifacts, CP replica deduplication, completion rendezvous, disagreement rejection, and rank-zero per-token
merge. Endpoint model discovery and health diagnostics now precede NCCL initialization, endpoint-port routing, and
two-phase receiver completion.

All five resulting server reports pass. Repository collection is now 486 items and the static inventory is 473 definitions
with 1,424 curated decisions across 332 Python test files. The static audit surfaces 19 conditional runtime gates, one
intentional duplicate group, and no parse errors.

## Two-hundred-and-forty-ninth wave: join ownership, OPD, and Muon producer-consumer policies

The two-hundred-and-forty-ninth wave consolidates seven report definitions without removing an assertion. A fullgraph
module-managed adapter producer now feeds ownership compilation across dense, direct-output, EP-replicated, and
owner-sharded topology families, stable fingerprints, fail-closed structure and replica-domain admission, then bucketed
residual reduction, immutable raw accumulators, and logical norm accounting.

Runner ownership compilation now proceeds from dense and exact LM-head producers through replica topology, unquantized and
quantized expert-factor contracts, registered session-rank specialization, rejected backend combinations, effective
LM-head folding, analytical gradients, capture finalization, and optimizer mutation. OPD teacher contributor selection, CP
gathering, Mooncake publication and cache row consumption now feed packed loss execution, metrics, and ranked
vocab-parallel debug artifacts. Muon configuration, grouping, fallback, Gram-Newton-Schulz stepping, and Quack admission now
include fused gate-up discovery and a tiny Nemotron-H parameter update.

Attention registry versus FlashAttention versus paged-cache APIs, FSDP dtype versus transformation versus prefetch policy,
and standard Newton-Schulz versus live CUDA dtype preservation were reviewed and retained separately. All six resulting
reports pass. Repository collection is now 479 items and the static inventory is 466 definitions with 1,434 curated
decisions across 332 Python test files. The static audit surfaces 19 conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-fiftieth wave: join batch-invariance, DeepSeek, and indexer lifecycles

The two-hundred-and-fiftieth wave consolidates four report definitions without removing a behavioral assertion. Dense
batch-invariant matmul, RMSNorm, log-softmax, and mean now establish the primitive contract before a padded Qwen sequence
proves full-model composition invariance under the same CUDA gate. DeepSeek-V4 window and compressed attention execution
now includes sink storage, TileLang call-dtype conversion, complex RoPE state, and TP admission, while grouped `wo_a` LoRA
contribution and gradients now feed the all-target attention-adapter freezing and first-step training policy.

The TileLang indexer no longer launches a fifth kernel report and walks every masked cell in Python. Each of the four
retained execution geometries now checks its valid numerical scores and all invalid future positions from the same output
with a vectorized assertion; large-value and zero-input cases still run once. Dataset split fingerprints versus complete
configuration hashes, dataloader construction versus packed integration, Mooncake positive transport versus fail-closed
metadata handling, DeepSeek HF-to-DCP conversion versus AutoModel loading, and native-FP8 runtime versus checkpoint
construction were reviewed and retained as distinct boundaries.

All eight resulting reports pass, including the four TileLang geometries in isolated processes. Repository collection is
now 475 items and the static inventory is 462 definitions with 1,443 curated decisions across 332 Python test files. The
static audit surfaces 19 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-fifty-first wave: join numerical trees and exact-state lifecycles

The two-hundred-and-fifty-first wave consolidates three static definitions and four collected items without removing an
assertion. Families-v2 RMSNorm FP64 bounds, residual and zero-centered behavior, batch invariance, and repeatability now
precede forced fused-versus-split bit equality and live dispatch selection in one frozen-tree policy. Exact GLM dense,
projection, LM-head, streaming, and sparse-delta sync rejection now culminates in separate adapter-factor checkpoint keys,
bytes, and configuration.

Index-share reentrant and non-reentrant checkpointing are now an internal Boolean mode matrix rather than separate pytest
IDs. Both modes retain producer recomputation, single payload creation, detached shared consumption, gradients, and closure,
then feed forward-only success, forward failure, backward failure, and idempotent cleanup. The standalone `solve_tril`
two-warp pin was reviewed and retained: warp count fixes a bit-relevant Triton reduction tree, while the GDN runtime report
only enforces tolerant numerical agreement and cannot replace that exact source-level gate.

All three resulting reports pass, including live CUDA RMSNorm execution. Repository collection is now 471 items and the
static inventory is 459 definitions with 1,447 curated decisions across 332 Python test files. The static audit surfaces 19
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-fifty-second wave: delete duplicated and fake-authored E2E reports

The two-hundred-and-fifty-second wave removes four expensive or misleading reports. Pipeline schedule parity already
launches a two-GPU PP2/FSDP1 1F1B baseline before its three virtual-stage variants, so that baseline now owns convergence
and the standalone PP2 convergence process is gone. The LoRA FSDP2 checkpoint transaction now explicitly proves the load
marker, resumes to step 20, and enforces the former standalone convergence threshold; the redundant third Qwen3-8B
training process is removed.

The CPU OPD suite no longer claims production loss invariance from a fake backend that calculated KL and global
normalization entirely in test code. It also drops a fixed teacher-cache metadata echo whose asserted values were authored
by the fake itself. The retained CPU end-to-end report still routes real packed teacher state through RequestProcessor,
Mooncake metadata, `TeacherActivationCache`, `TeacherHeadManager`, and the production `opd_loss_function`, then checks the
result against an independent grouped reference.

The retained OPD report passes, and the strengthened two-phase LoRA FSDP2 report passes through step 20 in 162 seconds.
The retained pipeline schedule report exposes a current product failure before convergence: the 1F1B baseline reaches
stage backward with `Output gradient: None`. The seven full-weight FP8 E2E mechanisms, three P2P transfer boundaries, and
five adapter-manager state-machine boundaries were reviewed and retained. Repository collection is now 467 items and the
static inventory is 455 definitions with 1,454 curated decisions across 332 Python test files. The static audit surfaces 19
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-fifty-third wave: remove dead E2E jobs and internalize implementation matrices

The two-hundred-and-fifty-third wave removes the plain-AdamW eight-GPU PP2/FSDP4 training job: the surviving Muon job
uses the same pipeline, FSDP, packing, and microbatch topology while adding optimizer-partition coverage. More
importantly, the entire Nemotron-H E2E module is gone. Every report in that module requested
`tiny_nemotron_h_model_dir`, a fixture that does not exist in any repository conftest and was never added with the test;
all three definitions (four collected items) therefore errored at setup without constructing a model. Production
Nemotron model, packed-varlen, checkpoint, gradient, and optimizer-step behavior remains covered by functioning suites.

Four implementation or shape matrices no longer inflate the product-report count. Native-FP8 linear and expert plain
conversion, FlashQLA four-head and production 32-head parity, PEFT MoE down-A and gate-B EP slicing, and non-gated Triton
and native MoE parity now execute as internal cases of their respective semantic policies. No branch or numerical
assertion was removed.

The native-FP8 plain and two-rank FSDP2 lifecycle, all LoRA checkpoint reports, eager and both installed non-gated MoE
backends, and both FlashQLA head regimes pass. The retained eight-GPU PP2/FSDP4 Muon report reaches the 1F1B schedule and
then exposes the current pipeline backward product failure, consistent with the retained two-GPU gate. Repository
collection is now 458 items and the static inventory is 451 definitions with 1,461 curated decisions across 331 Python
test files. The static audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-fifty-fourth wave: prove reachability and remove historical skip shields

The two-hundred-and-fifty-fourth wave adds a repository-wide fixture-reachability gate instead of trusting collection
alone. It found one remaining structurally dead report: the FP8 hybrid Ulysses/Ring long-tail E2E job requested
`tiny_agent_context_dense_model_dir_with_weights`, a fixture that has never existed in the repository, and therefore could
not reach model or trainer construction. That report and its now-unused dataset writer are removed; the functioning FP8
matrix still owns checkpoint-resume, TP, Ulysses, Ring, local-MoE, and DeepEP EP/eFSDP boundaries.

DeepSeek-V4 window and compressed attention and exact GLM sparse-attention CP modes now run as internal isolated matrices,
preserving their numerical, gradient, QAT, sink, exact-factor, query-offset, and topology assertions without multiplying
pytest report IDs. Triton and Quack routing-score gradients likewise form one backend policy. This review exposed that the
Quack parameter had never executed: its test-owned import stub omitted a required grouped-GEMM symbol and a broad
`ImportError` handler converted the defect into a skip. The stub is complete now, both implementations execute against the
reference, and internal import failures fail closed.

Three historical skip shields are also gone. GLM5 and Dr.GRPO are shipped parts of this source tree, so their reports no
longer pretend those implementations are optional. The eager-versus-native MoE policy retains lazy imports for lightweight
collection but no longer catches every exception from production modules. All seven focused reports pass, including live
eager/native MoE and the formerly skipped Quack branch. A full `pytest --setup-plan` now succeeds with no missing fixture,
and global collection succeeds at 454 items. The static inventory is 450 definitions with 1,468 curated decisions across
331 Python test files; the audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse
errors.

## Two-hundred-and-fifty-fifth wave: remove dormant opt-in coverage and fail core backends closed

The two-hundred-and-fifty-fifth wave removes the only environment-opt-in pytest report. The DeepGEMM grouped-FP8
subprocess skipped unless `XORL_TEST_DEEP_GEMM_FP8=1`, but no repository workflow, script, or configuration ever sets
that flag. It was a dormant manual diagnostic presented as suite coverage. The three functioning FP8-MoE reports still
exercise injection, grouped forward and weight gradients, Triton-grouped and scalar-Quack execution, bias and activation
variants, tensor-parallel reduction, dense-plus-MoE composition, and real optimizer updates.

The same review removes catch-all import skips from core backend gates. Quack is a pinned dependency; Transformers and
TileLang are core pinned dependencies; and the Quack, GKN, GLM5 indexer, and FlashQLA modules are shipped source. Their
tests retain explicit CUDA, SM90, and TileLang-feature admission gates, but packaging errors and internal import regressions
now fail instead of disappearing. All three remaining FP8-MoE reports, both Quack reports, both GKN reports, the GLM5
TileLang parity report, all four FlashQLA exact-contract reports, and the FlashQLA-versus-FLA numerical report pass.

Repository collection is now 453 items and the static inventory is 449 definitions with 1,473 curated decisions across
331 Python test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse
errors.

## Two-hundred-and-fifty-sixth wave: fold narrow admission branches and delete inert test scaffolding

The two-hundred-and-fifty-sixth wave folds two narrow reports into their owning behavioral policies. Forcing the SSM
kernel while `mamba_ssm` is unavailable now closes the CPU SSD fallback, chunked recurrence, packed-sequence, and mixer
policy. Invalid learning rate, warmup ratio, and schedule-mode inputs now close the constant, linear, and cosine scheduler
policy. Every RuntimeError and ValueError assertion remains; neither implementation branch needs a separate product ID.

Thirteen modules no longer carry `if __name__ == "__main__": pytest.main(...)` launch blocks. Repository and CI execution
already use pytest paths and node IDs, so those blocks were an unused second runner surface. The FP8 linear and MoE files
also drop GPU and skip markers from directly called private assertion helpers: pytest does not apply marker selection to
such calls. The seven public FP8 reports retain the actual hardware gates and all seven pass with every helper executing.

The combined scheduler and SSM run reports three passes and the legitimate optional `mamba_ssm` kernel report skips; all
seven FP8 linear and MoE policies pass on CUDA. Global collection succeeds at 451 items. The static inventory is 447
definitions with 1,477 curated decisions across 331 Python test files; the audit surfaces 18 conditional runtime gates,
one intentional duplicate group, and no parse errors.

## Two-hundred-and-fifty-seventh wave: strip inert helper metadata and deduplicate synthetic routing

The two-hundred-and-fifty-seventh wave removes 120 pytest marker decorations from private helpers across 30 distributed,
model, operator, optimizer, server, and weight-sync files. These `_assert_*` and worker helpers are not collected, and
direct calls do not apply pytest marker selection or skipping; the metadata therefore advertised hardware and async gates
that it did not enforce. Every public `test_*` report keeps its actual CPU, CUDA, architecture, async, distributed, and
optional-dependency markers.

A private-helper reachability pass also removed four unused `NotImplementedError` overrides from a weight-sync QLoRA fake;
the base methods already fail identically, while the retained `dequantize_expert` method is the only fake behavior consumed
by the production merge-and-sync path. Full test-tree Ruff now passes after deleting one unused packed-dataset fixture
variable and documenting three intentional imports after standalone path bootstrapping.

Balanced synthetic TopK routing no longer has a duplicate report. The canonical TopK policy already proves balanced expert
selection, uniform weights, count balance, and softmax/hash/bias override precedence. The unique MoEBlock replay-regather
assertion now closes the train-router dispatch policy. Sixteen representative reports pass across router, RMSNorm, fused
LM-head, GLM, pipeline profiling, async server, weight-sync, and future-store boundaries; one explicit optional backend
report skips. Global collection succeeds at 450 items. The static inventory is 446 definitions with 1,481 curated decisions
across 331 Python test files; the audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no
parse errors.

## Two-hundred-and-fifty-eighth wave: remove dead fixtures and separate certification from regression coverage

The two-hundred-and-fifty-eighth wave maps every pytest fixture through direct parameters, fixture dependencies,
`usefixtures`, indirect parametrization, and dynamic `getfixturevalue` calls. Only two fixtures have no consumer:
`fake_packed_dataset` and `small_dense_model_dir_with_weights`. They are removed with the dead `FakePackedDataset` and
root-level `SimpleCollator` scaffolding. Full setup planning still reaches every collected report without a missing fixture.

The tests tree also no longer presents manual workloads as pytest protection. DeepEP-versus-AllToAll parity, uneven
vocab-parallel OPD diagnostics, and the 100-step eight-H100 QLoRA comparison define no pytest report and have no workflow
caller; their direct-run value is preserved under `certification/deepep`, `certification/opd`, and
`certification/qwen3_30b`. A smaller standalone reverse-KL file likewise defined only a `main` function despite instructing
users to run pytest; it is removed because the production gathered path is exercised by the retained four-rank lm-head TP
FSDP/OPD policy across six CP, DP, and HSDP topologies.

One collected report is removed on semantic grounds. It directly called two custom-autograd `backward` methods with a
fabricated `SimpleNamespace` context and `grad_output=None`, then checked a tuple length derived from that same fake context.
It never entered PyTorch autograd or production dispatch. The surviving Quack grouped-GEMM and DeepEP no-permute policies
run real backward graphs and compare all trainable gradients with trusted implementations.

The focused data and Quack run reports four passes, and the consolidated four-rank lm-head TP FSDP/OPD owner passes all six
topologies in 140 seconds. Full setup planning, test-tree and relocated-certification Ruff, compileall, collection, and
diff-whitespace gates pass. Repository collection is now 449 items and the static inventory is 445 definitions with 1,485
curated decisions across 327 Python test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-fifty-ninth wave: replace zero-valued coverage and absorb narrow QARL reports

The two-hundred-and-fifty-ninth wave finds a report that could never detect its claimed regression. The context-parallel
FLOPs check constructed an `xorl_glm5` configuration, but `XorlFlopsCounter` has no estimator for that model type and falls
back to zero. Comparing CP1 and CP64 therefore asserted only `0 == 0`. The rewritten report uses the supported Qwen3-MoE
estimator, first proves the baseline is nonzero, and then establishes that a global sequence-length input is not multiplied
by context-parallel size.

Two narrow QARL report identities are removed without losing their useful behavior. The standalone activation NVFP4 report
used the production internal quantizer as its numerical reference and repeated the shared operator's two-dimensional STE
contract. Its unique leading-dimension reshape, exact value, and gradient assertions now close the independent pure-PyTorch
NVFP4 numerical policy. The W4A4 MoE file did not run a down projection or grouped GEMM; it checked only a temporary backend
name and exception restoration. Those assertions now close the existing CPU MoE conversion, execution, and injection
policy.

The distillation and teacher-cache suites also share one Mooncake object-store fake instead of maintaining identical byte
APIs; the shared helper records the call keys needed by both consumers. Checkpoint process-group selection versus expert-mesh
restore, experiment ingestion and ranking boundaries, quantization primitives versus exporter disk layout, and OPD endpoint
verification versus payload transport were reviewed and retained as distinct production failure surfaces.

All seven CPU QARL policies pass, and the focused shared NVFP4, FLOPs, Mooncake transport, and teacher-cache run reports
seven passes. Ruff, formatting, diff whitespace, JSON validation, global collection, and the static audit pass. Repository
collection is now 447 items and the static inventory is 443 definitions with 1,490 curated decisions across 326 Python test
files. The audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixtieth wave: consolidate server dispatch fragments and strengthen import boundaries

The two-hundred-and-sixtieth wave removes three report identities that were fragments of broader policies. Launcher CLI
override parsing and removed-ZORL rejection now run with the existing YAML, direct-override, and unsupported-configuration
admission boundary. Launcher worker discovery and readiness remain separate because they exercise live address selection and
process failure behavior rather than configuration validation.

The standalone ModelRunner causal-LM report used a zero-logit, two-token model only to prove that token losses are summed.
Its exact raw-sum and per-token assertions now precede the DR-GRPO cases in the retained `_compute_micro_batch_loss`
dispatch policy. Temperature, legacy field names, per-token output controls, K3 output forcing, and DR-GRPO metrics remain
covered in the same report.

The QLoRA clean-interpreter smoke report is also absorbed into expert capability and ownership. It previously checked only
that imports returned zero even though its docstring claimed package decoupling. The retained policy now launches the clean
interpreter and explicitly proves that importing QLoRA utilities and expert modules loads neither `xorl.models` nor any of
its children. The server protocol policy likewise stops treating Torch as optional: the package is a core dependency, so a
broken Torch import now fails rather than silently skipping tensor serialization.

Server API and security reports were reviewed and retained where they protect different consumers: outbound network
admission, artifact and diagnostic path confinement, compile-worker admission, API configuration validation, TensorData
re-nesting, session publication, optimizer fallback, training metrics, and ready-handshake queuing are separate failure
surfaces. The remaining heuristic candidates are explicit hardware or backend gates, not removal evidence by themselves.

All nine surviving reports across the five touched server and QLoRA modules pass. Full test-tree and certification Ruff,
formatting, decision-JSON validation, public-tree lint, diff whitespace, global collection, and the static audit pass.
Repository collection is now 444 items and the static inventory is 440 definitions with 1,495 curated decisions across 324
Python test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-first wave: attach model registration to real family loaders

The two-hundred-and-sixty-first wave removes two standalone model registration reports. DeepSeek-V4 AutoConfig resolution
and XoRL meta construction previously fabricated the same tiny standard HF snapshot shape used by the retained AutoModel
loader but stopped before loading a tensor. The surviving snapshot policy now proves AutoConfig class resolution, the HF
AutoModel mapping, XoRL meta construction, actual `from_pretrained` loading, and exact embedding bytes in one sequence. HF
to DCP conversion remains separate because it exercises distributed checkpoint serialization rather than model admission.

Nemotron-H registry lookup and Ultra-style local configuration normalization likewise now open the real model-family policy
instead of reporting alone. That policy proceeds through mixed Mamba, attention, and MoE construction, router output,
causal loss, backward gradients, router freezing behavior, and full-layer gradient checkpointing. Packed variable-length
execution and published-checkpoint parity retain their own reports because they cover independent runtime and codec paths.

The registry-wide review retains the Kimi-wrapped DeepSeek-V3 policy because nested text-config aliases and official
auxiliary-loss defaults are absent from the base DeepSeek runtime. Qwen3.5 dense and MoE local normalization likewise has no
general family-construction owner. MiniMax M3 already combines registration with runtime, admission, checkpoint, and paging;
Qwen2 and OLMo2 already combine HF construction with fused/unfused layouts, checkpoint round trips, and numerical HF parity.
The small tokenizer, MTP checkpoint remap, gradient-checkpoint dispatcher, and BI operator reports were retained where each
reaches a distinct loader, execution, or reduction branch.

Both Nemotron-H reports and the consolidated DeepSeek-V4 standard-snapshot loader pass. Full test-tree and certification
Ruff, formatting, decision-JSON validation, public-tree lint, diff whitespace, global collection, and the static audit pass.
Repository collection is now 442 items and the static inventory is 438 definitions with 1,498 curated decisions across 322
Python test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-second wave: absorb data resilience and remove repeated stochastic certification

The two-hundred-and-sixty-second wave removes the standalone request-retry report. The decorator has one production owner,
the high-level `prepare_datasets` operation, so immediate success, transient request and Hub failures, exponential backoff,
retry exhaustion, and unrelated-exception propagation now close the dataset preparation lifecycle. That lifecycle already
owns dataset expansion, type selection, local and remote loading, splitting, merging, saving, and reloading; the utility
report and file no longer multiply the product boundary.

Stochastic rounding no longer proves one probability law with nested stress loops. The CPU policy previously accumulated
4,000 separately rounded 64-by-64 tensors, performing more than 16 million element updates to infer unbiasedness from a
generic relative-error maximum. It now creates one seeded 65,536-element population exactly one quarter of the way between
adjacent BF16 values and directly checks the legal neighbors, 25 percent round-up probability, and sample mean.

The four-rank reduce-scatter report also drops 200 extra all-to-all trials that repeated the same unbiased-expectation claim.
It retains the distinct native-FP32 comparison and per-element BF16 transit error bound; the separate FSDP2 report retains
real compositional backward and optimizer coverage. The revised CPU probability policy passes, and the distributed policy
passes on four GPUs in 18 seconds rather than spending most of its execution re-certifying primitive randomness.

The remaining optimizer reports were reviewed and retained because each already aggregates construction, grouping,
numerical updates, state strategy, cautious decay, and backend admission by optimizer family. Trainer gradient clipping,
token and microbatch metadata, pipeline chunked CE, explicit gradient synchronization, timer fail-soft handling, live CUDA
hooks, collator layouts, fingerprint identity, and packing reach distinct production consumers.

The consolidated dataset preparation and stochastic-rounding policies report two CPU passes, and the four-GPU collective
reports one pass. Full test-tree and certification Ruff, formatting, compileall, decision-JSON validation, public-tree lint,
diff whitespace, global collection, and the static audit pass. Repository collection is now 441 items and the static
inventory is 437 definitions with 1,502 curated decisions across 321 Python test files. The audit surfaces 18 conditional
runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-third wave: remove scale-only and pseudo-E2E jobs

The two-hundred-and-sixty-third wave maps the remaining end-to-end reports to production mechanisms rather than treating
GPU count as an independent behavior. The standalone single-GPU Qwen3-8B LoRA convergence job repeated the retained
two-GPU transaction's real model, rank, alpha, learning rate, 20-step horizon, and exact convergence threshold. The
surviving FSDP2 job additionally proves checkpoint save and load, the load marker, and the final global step; model and
server policies already own non-FSDP LoRA construction, forward, backward, optimizer, and checkpoint behavior.

The CUDA OPD report was an E2E test in name only. It bypassed `ModelRunner` construction, invoked a private loss helper,
and optimized hidden-state and lm-head tensors directly for eight iterations. Its decreasing loss therefore described a
free-tensor optimization problem rather than a trainer or server lifecycle. The retained runner policy already proves
two-teacher cache loading, metrics, loss, and backward through the helper, while the real GPU server OPD report owns
`ModelRunner` startup, the forward/backward API, and the optimizer step. The pseudo-E2E file is removed.

The remaining 18 E2E reports select distinct paths. FP8 covers dense resume, tensor parallel, Ulysses, Ring, plain MoE, and
DeepEP expert sharding. Pipeline reports separate direct trainer, schedule parity, server ModelRunner, FSDP, and folded
PP-EP-CP topologies. OPD retains request packing and Mooncake grouping, a real SGLang teacher, and the complete
sampler-teacher-Mooncake-trainer-weight-sync loop. DistSignSGD, hybrid shared-LoRA MoE telemetry, and LoRA checkpoint resume
retain their separate production boundaries.

The retained runner and CPU OPD policies report two passes, and focused collection exposes the LoRA transaction plus all
three OPD integration layers. Full test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-tree
lint, diff whitespace, global collection, and the static audit pass. Repository collection is now 439 items and the static
inventory is 435 definitions with 1,505 curated decisions across 320 Python test files. The audit surfaces 18 conditional
runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-fourth wave: remove finite-output and scale-only workload tails

The two-hundred-and-sixty-fourth wave audits semantic anti-patterns across the four largest surviving areas: models, server,
operators, and distributed integration. Qwen3.5's second trunk-wrap report built a tiny all-full-attention model, wrapped it,
and asserted only BF16 dtype and finite hidden states. The retained policy in the same file already proves the exact
full-attention, linear-attention, dense-MLP, shared-expert, and exclusion inventory. Generic GPU policies additionally prove
bitwise forward and backward, batch invariance, BF16 admission, serving-lane identity, and FSDP2 composition, so the
model-specific finite-output report is removed.

The Quack DeepEP regression no longer treats token count as an independent behavior. Its 16K-token parity case repeated the
same production hidden size, expert geometry, balanced routing, unchunked no-permute path, trusted reference, and gradient
comparisons already exercised at 4K tokens. Its 32K-token checkpoint-training case repeated the same checkpointed three-step
optimizer path already exercised at 8K tokens. Random routing, skewed routing with empty experts, explicit chunking, and
checkpoint versus non-checkpoint training remain. The optional compiled `deep_ep` module is absent from this environment, so
the reduced report collects but its live two-GPU execution remains capability-skipped.

Two block-FP8 workload tails are also removed. The generic codec allocated a 1024-by-2048 tensor only to compute storage
bytes from already-asserted dtypes and element counts, making the result tautological. The GKN codec's 4096-square roundtrip
selected the same multi-program two-dimensional kernel and error threshold already exercised by divisible and tail-tile
shapes. Both codec reports retain geometry, dtype, scale, accuracy, admission, determinism, sign, magnitude, and zero-block
coverage.

The weak-name review retains reports whose behavior is stronger than their label. DeepSeek-V4 attention exercises both
window and compressed-KV forward and backward, every trainable gradient, FP8-QAT dispatch, sink dtype transfer, and TP
rejection. FP8 DeepEP uniquely composes no-permute transport with clamped-SwiGLU, native activation, expert biases, grouped
FP8 backward, and every gradient. BF16 stochastic reduction separately proves custom-hook installation and FSDP2 gradient
agreement.

The Qwen selection policy reports one CPU pass and the two trimmed FP8 codec policies report two GPU passes in 18 seconds.
Full test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-tree lint, diff whitespace, global
collection, and the static audit pass. Repository collection is now 438 items and the static inventory is 434 definitions
with 1,509 curated decisions across 320 Python test files. The audit surfaces 18 conditional runtime gates, one intentional
duplicate group, and no parse errors.

## Two-hundred-and-sixty-fifth wave: attach selector fragments to their production owners

The two-hundred-and-sixty-fifth wave reviews the smallest surviving operator, distributed, model, server, checkpoint, and
loss reports for mocked metadata and argument-plumbing boundaries. Three standalone reports are fragments of broader
policies. The GatedDeltaNet backend report monkeypatched the FlashQLA chunk function and stopped after one call plus input
and output shapes; those assertions now close the CPU FlashQLA backend-selection and exact-contract precedence policy,
while real CUDA numerical, state-chaining, and batch-invariance reports remain separate.

Runtime-rank MoE LoRA scaling now closes inference-buffer construction and FP8 sync rather than reporting alone. The
retained policy checks active-rank scaling through the production buffer builder, all three emitted projection names,
values, shapes, dtypes, source cleanup, QLoRA folding, and subsequent quantization. The standalone file and its artificial
one-by-one expert boundary are removed.

Numerical-family selection likewise belongs to the model programs that set it. Both legacy environment aliases and GLM's
exact-v2 override now close the trainer model-builder policy; Qwen's exact-model hook proves its v1 LM-head pin overrides a
legacy v2 request. The selector-only file is removed. Families-v2 CUDA norm reachability stays separate from numerical-tree
realization because an unreachable correct kernel is a distinct regression.

The remaining small reports were retained where they protect separate behavior: DeepEP async-combine safety versus
internode transport preflight, DSV4 optional-kernel rotation fallback, GLM4 MTP checkpoint remapping, token-loss
composition, gradient-accumulation process-group routing, server batch slicing, OPD cache streaming, and KKT launch
geometry. All four focused owner policies pass. Ruff, scoped formatting, compileall, decision-JSON validation, public-tree
lint, diff whitespace, global collection, and the static audit pass. Repository collection is now 435 items and the static
inventory is 431 definitions with 1,513 curated decisions across 317 Python test files. The audit surfaces 18 conditional
runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-sixth wave: collapse mock boundaries into behavioral owners

The two-hundred-and-sixty-sixth wave ranks one-report files by monkeypatching, captured arguments, synthetic namespaces,
and absence of runtime work, then traces the highest-ranked candidates to their production consumers. Most mock-heavy
reports remain justified fault injection: NCCL rendezvous protects ephemeral-port rotation and bind-before-inference
ordering, checkpoint restore protects zero-meta admission and base-before-adapter initialization, and protocol round trips
preserve tensors while rejecting pickle.

Three standalone boundaries are consolidated. GLM and Kimi ModelRunner target-resolution files previously rebuilt large
model configurations even though production reads only top-level `model_type`; both also repeated the same explicit-target
branch. One compact cross-family policy now retains GLM defaults, Kimi defaults including `lm_head`, explicit targets, and
manifest targets while deleting more than one hundred lines of irrelevant fixture data.

Distributed-checkpointer process-group selection now closes the existing I/O policy rather than reporting alone. The
retained owner covers NCCL-to-Gloo selection, one-time caching, native-Gloo reuse, PP and non-PP metadata selection, custom
groups, and the actual load/save routing. Server batch-slice arithmetic similarly moves into the dispatcher policy, which
already owns distinct EP slices, CP sharing, EP-FSDP coordinates, padding, rollback, routing payloads, and completion. Its
replicated-DP mapping remains asserted, and the integrated sequence verified that the rollback environment switch is
restored before later cases.

The three surviving owner policies pass. Full Ruff, scoped formatting, compileall, decision-JSON validation, public-tree
lint, diff whitespace, global collection, and the static audit pass. Repository collection is now 432 items and the static
inventory is 428 definitions with 1,517 curated decisions across 314 Python test files. The audit surfaces 18 conditional
runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-seventh wave: remove production code that exists only for tests

The two-hundred-and-sixty-seventh wave reverses the audit direction: it inventories low-reference production helpers, then
checks whether tests are their only real consumer. This distinguishes externally callable or dynamically dispatched APIs
from convenience code that survives solely because an assertion imports it.

The simulator's `compare_kernel_variants` helper had one caller, a test that divided two literal latencies after the retained
ranker had already ordered the same rows. The wrapper and its arithmetic assertions are removed; the surviving policy still
proves the important behavior that a faster unvalidated candidate cannot displace the validated winner. A second simulator
helper, `reference_counter_total_flops`, described itself as test support but had no callers at all after earlier policy
consolidation, so the orphan and its sole `SimpleNamespace` dependency are removed.

The NVFP4 exporter likewise no longer ships a dequantizer solely so its test can grade the production quantizer with a
second implementation from the same module. The independent fake-quant policy retains exact numerical-reference coverage.
The exporter owner retains packed layout, scale shapes and dtypes, fused shared scales, BF16 islands, activation scales,
directory metadata, and requantization rejection. Low-reference sparse-delta reset, FP8 profiling, dynamic rank filtering,
teacher-store, QARL export, fused-expert cache, and Mooncake store APIs remain because they have operational, runtime,
public-package, or CLI ownership beyond assertion convenience.

Both affected policies pass. Full test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-tree lint,
diff whitespace, global collection, and the static audit pass. Repository collection remains 432 items and the static
inventory remains 428 definitions with 1,521 curated decisions across 314 Python test files. The audit surfaces 18
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-sixty-eighth wave: remove test-only GLM inspection surfaces

The two-hundred-and-sixty-eighth wave deepens the source-consumer audit by tracing low-reference methods through dynamic
dispatch, framework registration, exports, documentation, and production callers. Runner command handlers, API endpoints,
documented DataLoader extensions, adapter transactions, and dynamically selected weight-sync methods remain. Four GLM
surfaces exist only to make assertions convenient and are removed.

The exact LM-head component no longer exposes late TP-group binding when production always supplies the group to its
constructor. Its factor-view method also duplicated the BF16 casts already performed inside both real autograd functions;
the retained custom-boundary policy directly captures and checks those production bytes. Shared- and routed-expert public
factor-view wrappers likewise only cast masters before calling internal builders that the value paths already use, so the
tests now exercise those runtime-owned builders directly.

The routed expert loses a larger test-only branch: a trace dataclass, clone-heavy hook wrappers, a capture flag production
always disabled, and a diagnostic method called only by its GPU test. Instead of inspecting staged caches from that alternate
path, the policy now compares actual module forwards with zero and live LoRA factors and verifies routing-scale linearity
through production execution. Exact buffer layout, owner remapping, independent hybrid VJPs, zero-slot gradients, and mixed
owner coverage remain. IndexShare similarly drops a context-manager wrapper absent from model execution; its lifecycle
policy now drives the same `begin` and `finish_forward` calls used by the model's try/finally.

Three focused policies pass and the optional SGLang slice policy capability-skips; all affected GPU policies collect. Full
test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-tree lint, diff whitespace, global collection,
and the static audit pass. Repository collection remains 432 items and the static inventory remains 428 definitions with
1,526 curated decisions across 314 Python test files. The audit surfaces 18 conditional runtime gates, one intentional
duplicate group, and no parse errors.

## Two-hundred-and-sixty-ninth wave: join numerical probes to their runtime owners

The two-hundred-and-sixty-ninth wave scans the remaining reports for shape-only, dtype-only, finiteness-only, literal-
configuration, and duplicated family assertions. The apparent weak reports retain stronger numerical, gradient, cache,
checkpoint, or backend comparisons that a surface assertion classifier misses. Generic fused RMSNorm, explicit family
admission, and dense and MoE Qwen3.5 policies also remain separate: they own different kernel guarantees or execute
separate production implementations and call sites.

Two genuinely narrow reports move into their behavioral owners without losing assertions. OLMo-2 no longer launches two
independent two-rank Gloo subprocesses over the same tensor-parallel mesh. Its surviving end-to-end policy first compares
plain and sharded `Olmo2QKRMSNorm` against local numerical references, then applies the production TP plan and proves model
forward, vocab-sharded LM-head execution, and gradients for every trainable parameter. The standalone Q/K-norm module is
deleted, saving one subprocess launch.

Qwen3-MoE diagnostic decode likewise no longer reports the two-value FlashAttention causal flag independently. Both flag
branches now open the retained natural-cache and routing-replay cached-forward parity lifecycle. The two focused owner
policies pass. Full test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-tree lint, diff
whitespace, global collection, and the static audit pass. Repository collection is now 430 items and the static inventory
is 426 definitions with 1,530 curated decisions across 313 Python test files. The audit surfaces 18 conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-seventieth wave: replace environment snapshots with selected behavior

The two-hundred-and-seventieth wave audits exact package, accelerator, and process-state assertions before reviewing the
remaining files with multiple reports. Native-FP8 materialization no longer asserts that the worker happens to use the
literal `2.12.1+cu132` Torch wheel. The package and lock files already own that dependency pin, and a behavior regression
should remain runnable after a compatible upgrade. The worker now explicitly disables
`swap_module_params_on_conversion`, selecting the replacement path that exposed the DTensor corruption instead of merely
asserting its ambient default. Both the ordinary conversion policy and the two-rank FSDP2 materialization transaction pass.

MoE compilation now forms one block-to-model CUDA policy. The surviving report runs every available MoE backend through
AOT eager and Inductor at block and decoder scope, then continues through native and eager full-model forward and backward.
The split reports shared the same capability gate and behavior owner, so a second pytest identity added no isolation or
failure meaning. Its real CUDA matrix passes in 64.78 seconds.

The remaining apparent pairs stay separate where capability or oracle ownership differs: CPU coverage must not disappear
behind a CUDA skip; FP64 gradcheck is independent from sampled numerical parity; sparse-MLA forward and backward own
different kernels; and routing wire decode, context layout, and packing modes have distinct failure meanings. Full
test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-tree lint, diff whitespace, global
collection, and the static audit pass. Repository collection is now 429 items and the static inventory is 425 definitions
with 1,533 curated decisions across 313 Python test files. The audit surfaces 18 conditional runtime gates, one intentional
duplicate group, and no parse errors.

## Two-hundred-and-seventy-first wave: delete dormant external integration and fail admitted backends closed

The two-hundred-and-seventy-first wave audits exception-driven availability gates rather than accepting every skip as an
environment fact. The 522-line sparse-delta trainer-to-SGLang report depended on an unpinned `delta-encoding` tree and an
SGLang receiver module absent from the repository's pinned submodule. No workflow, script, or configuration supplies its
two path variables, and catch-all imports converted both absence and implementation breakage into a skip. Its fake trainer,
runner, orchestrator, HTTP endpoint, and receiver therefore never established repository coverage. The dormant report is
removed; retained policies still own XORL's packed artifacts, source capture, sorted indices, malformed-update rejection,
hashes, endpoint payloads, version forwarding, and sparse-delta transport lifecycle. Real cross-project byte compatibility
should return only as a dependency-pinned, scheduled integration.

The standalone `TokenPartial` component file is also gone. Its caller-scaled denominator, microbatch composition, raw-sum,
sequence-mean-token-sum, and empty-mask assertions now close the shared loss policy that already proves explicit reducers
match the legacy policy and importance-sampling implementations. This preserves every oracle while removing an artificial
component report.

Finally, supported backend admission fails closed consistently. MoE compilation no longer catches errors importing its
shipped capability helper or wraps an infallible Quack list append. Hard-pinned TileLang is no longer presented as an
optional sparse-MLA dependency after the explicit Hopper gate. Three DeepEP NVSHMEM path helpers likewise stop swallowing
errors after their callers have already admitted the package with `importorskip`. The shared loss policy passes, the full
MoE compiler matrix passes in 63.35 seconds, both live TileLang sparse-MLA policies pass in 25.04 seconds, and all affected
DeepEP reports collect. Repository collection is now 427 items and the static inventory is 423 definitions with 1,537
curated decisions across 311 Python test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-seventy-second wave: move private fragments into runtime owners

The two-hundred-and-seventy-second wave traces the smallest remaining single-report files into the production lifecycles
that consume their behavior. Seven files disappear without dropping an assertion. The runtime FLOPs counter's cp1-versus-
cp64 global-sequence-length invariant now closes the simulator topology, shape, and analytical-ledger policy. Distributed
loss-group forwarding, valid-token normalization, and backward scaling now follow trainer token metadata counting, while
HSDP microbatch all-reduce deferral and restoration close the explicit SP and LM-head gradient-synchronization policy.

OPD layer-cache indexing no longer tests a private fetcher in isolation. Exact selected indices, streamed layer ranges,
layer counts, and output shapes now precede the retained streaming OPD loss, gradient, cache, metric-reduction, and debug-
artifact lifecycle. ModelRunner's LoRA fragments likewise join their consumers: family defaults, explicit targets, and
manifest precedence close the adapter ownership compiler, while nonresident checkpoint promotion, failed-kill
preservation, registry cleanup, and path rejection close the optimizer, checkpoint-load, and session-registry lifecycle.

Finally, direct sync-quantization dictionary examples now execute inside receiver detection and API admission. BF16 no-op
aliases, valid FP8 defaults and normalization, module exclusion cleanup, and every malformed or unsupported form remain,
alongside receiver discovery, unsupported-marker propagation, per-call enrichment, and persisted default behavior. The
seven focused owner policies pass. Full test-tree Ruff, scoped formatting, compileall, decision-JSON validation, public-
tree lint, diff whitespace, global collection, and the static audit pass. Repository collection is now 420 items and the
static inventory is 416 definitions with 1,542 curated decisions across 304 Python test files. The audit surfaces 18
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-seventy-third wave: make feature lifecycles own their completion branches

The two-hundred-and-seventy-third wave removes four more single-feature files by moving their exact assertions to the
policies that own the rest of each lifecycle. IndexShare checkpointing now continues from producer/shared recomputation and
mode-owned close behavior into both public callers: offline trainer and server forward failures each release retained
context exactly once. The orchestrator-runner wire protocol similarly continues from payload and tensor serialization,
command construction, and pickle rejection into the rank-zero ready handshake, including normal ACK, request-before-ACK
queueing, client identity, unexpected message, and channel-failure behavior.

Packed sequence alignment now includes its distributed completion. The retained server-versus-CLI policy already owns
boundaries, int32 metadata, SP sharding, stale metadata replacement, LCM padding, and unpacking; it now also simulates an
eight-rank maximum and proves 176-token batches extend to 512 with ignored labels, masked attention, corrected cumulative
sequence lengths, and integer max lengths.

The lm-head TP plus EP report no longer launches a four-rank process only to inspect groups. The retained FSDP transaction's
DP2 by CP2 case now enables EP2, proves the exact TP and replica memberships and active EP mesh, then continues through
parameter synchronization, vocab-sharded causal-LM loss, eager global-loss parity, and full weight and hidden-gradient
parity. The three CPU owners and this strengthened four-rank transaction pass. Full test-tree Ruff, scoped formatting,
compileall, decision-JSON validation, public-tree lint, diff whitespace, global collection, and the static audit pass.
Repository collection is now 416 items and the static inventory is 412 definitions with 1,546 curated decisions across 300
Python test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-seventy-fourth wave: run server policies against canonical modules

The two-hundred-and-seventy-fourth wave removes unrealistic import machinery rather than manufacturing a count reduction.
Seven server-runner reports were loading production files under synthetic module names, creating private copies of
`ModelRunner`, `RunnerDispatcher`, `AdapterCoordinator`, `CheckpointManager`, or `LoRAAdapterManager`. Their monkeypatches
therefore targeted test-only module state and could miss package import interactions. The reports now import and patch the
canonical runtime modules while preserving all adapter coordination, checkpoint failure, LoRA roundtrip, optimizer,
session-registry, load-state, and session-operation behavior. All 11 reports pass.

The server-argument policy had the same issue at a larger scale: it imported the real launcher for override handling, then
re-executed `launcher.py` with fake API-server, orchestrator, session, QARL, and packing modules solely to obtain
`load_server_arguments`. It now uses the canonical launcher and real dependency graph for YAML admission, shipped-example
subprocess parsing, sparse-MLA propagation, and the rest of its configuration lifecycle. All four reports pass.

The owner-level scan explicitly retains the smallest adjacent reports where size is not semantic duplication: DSv4
fallback rotation, GLM4 MTP checkpoint remapping, DeepEP async-combine admission, KKT launch geometry, families-v2 norm
dispatch, BI mean, and Class-B RoPE each protect a separate numerical or fail-closed production boundary. This rewrite
leaves collection at 416 items and the static inventory at 412 definitions, with 1,549 curated decisions across 300 Python
test files.

## Two-hundred-and-seventy-fifth wave: separate public CPU policy from optional kernel machinery

The two-hundred-and-seventy-fifth wave removes a hidden implementation-detail report from the FA3-gated ring-attention
module. That report combined direct assertions on private `_get_zigzag_step_section` slices with the public
`zigzag_reorder_packed_sequence` behavior consumed by `TextSequenceShardCollator`; because the whole module imported FA3
at collection time, none of its CPU-only assertions ran in the default environment. The public single-document,
packed-document, multi-rank, identity, and invalid-length contract now closes the collator's existing SP sharding policy,
and passes without FA3. The private step-section assertions are gone. The remaining ring-attention report is solely the
real CUDA partial-output merge policy.

The skipped-runtime scan retains the FA3 numerical policies and both GLM exact SGLang joins. Those reports cross real
kernel or adapter-export, parser, and memory-pool boundaries rather than checking import availability. Their dependency
lane must remain explicit: XoRL's default profile is Torch 2.12.1 and has no `sglang-kernel`, while pinned SGLang declares
Torch 2.11.0 with `sglang-kernel==0.4.5`; loading a lazy wrapper is not an ABI smoke test. Exact SGLang-kernel execution
therefore belongs in the isolated Torch 2.11 environment, not in the default profile. Repository collection remains 416
items while the static inventory falls to 411 definitions, with 1,551 curated decisions across 300 Python test files. The
audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-seventy-sixth wave: collapse synthetic MoE module graphs into the runtime owner

The two-hundred-and-seventy-sixth wave removes the standalone EP routing-score report after moving its complete numerical
oracle into the adapter policy that owns `expert_scores`. The old report re-executed Triton and Quack source files under
test-only module names, after installing fake fused-MoE and grouped-GEMM packages in `sys.modules`. The retained policy now
patches the canonical runtime modules, proves both backend outputs and routing-score gradients against the same eager
reference, then continues through real registry admission, common signatures, FP8 rejection, optional MoE-act separation,
and adapter argument forwarding. The adjacent before-versus-after-down policy reuses only explicit CPU grouped-GEMM
doubles rather than importing helpers from another test report. Both owner policies pass.

Quack's process and cache safety policy had the same synthetic-copy smell. It source-loaded the worker protocol, ptxas
wrapper, and cache utility under private names and supplied fake `cutlass` and `tvm_ffi` modules. Those components import
through the real package in the supported environment, so timeout, truncated-frame, unique temporary output, PTX entry
selection, and non-executable cache-key checks now exercise canonical module state. The focused policy passes. Repository
collection falls to 415 items and the static inventory to 410 definitions, with 1,553 curated decisions across 300 Python
test files. The audit surfaces 18 conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-seventy-seventh wave: replace synthetic dependency state with supported runtime state

The two-hundred-and-seventy-seventh wave crosses trainer, attention, and executable-script boundaries. HSDP gradient-sync
deferral no longer replaces Torch's composable-FSDP module in `sys.modules` with a fake class. A lightweight instance of
the real Torch 2.12 `FSDPModule` API type now proves two-microbatch all-reduce deferral, last-microbatch enablement,
exception-safe restoration, and the replicate-size negative branch.

The attention registry policy likewise stops manufacturing fake `flash_attn` and `flash_attn.cute` packages and reloading
two production modules. The default environment already supplies the FA4-only state this regression protects, so the
policy now checks canonical availability, registration under the compatible FA2/FA3 names, explicit FA4 registration,
eager and native resolution, and fail-closed unavailable-flash handling against the live runtime graph.

Finally, student endpoint matching and weight-version verification now open the OPD pipeline payload and transport policy
instead of forming a second report. The standalone driver module is cached after one source load rather than re-executed
for every helper; all success, mismatch, endpoint-failure, worker, causal-shift, cache-index, and Mooncake metadata
assertions remain. The focused policies report seven passes and the expected FA3-only skip.

The remaining dependency doubles are intentionally narrower: absent `delta_encoding`, Mooncake's CUDA-bound import, and
the isolated SGLang/`sgl_kernel` ABI lane are represented only to exercise serialization, fallback, runtime-context, or
slot-combine callers. They are not accepted as compiled-kernel smoke tests. Repository collection falls to 414 items and
the static inventory to 409 definitions, with 1,557 curated decisions across 300 Python test files. The audit surfaces 18
conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-seventy-eighth wave: make shipped backends fail closed

The two-hundred-and-seventy-eighth wave reviews every remaining conditional-runtime candidate rather than treating a skip
as evidence of uselessness. Two policies were incorrectly permissive. The EP adapter report could skip the whole policy
midway when a backend was absent, and the non-gated MoE report silently tested whichever subset of Triton and native
registered. Those are shipped default-profile backends, not optional integrations. Both reports now assert their complete
registry surface and execute every expected adapter-forwarding, FP8-rejection, forward, input-gradient, and weight-gradient
branch. The focused default-runtime policies report three passes.

The full OPD pipeline also repeated its three-GPU admission check inside the test after its module-level marker had already
enforced the same rule before execution. The late branch is removed, so a skipped environment no longer creates model
artifacts before reaching a duplicate decision.

The 15 remaining candidates are retained deliberately. They cover Hopper-only FlashQLA and exact GLM52 kernels, two-rank
FSDP composition, real CUDA profiler events, the isolated SGLang/`sgl_kernel` runtime, and optional DeepGEMM bit parity.
Each guards a complete numerical, distributed, or compiled-runtime transaction; none hides an ordinary CPU policy.
Repository collection remains 414 items and the static inventory remains 409 definitions, with 1,560 curated decisions
across 300 Python test files. The audit now surfaces 15 conditional runtime gates, one intentional duplicate group, and no
parse errors.

## Two-hundred-and-seventy-ninth wave: close policy islands inside their runtime owners

The two-hundred-and-seventy-ninth wave removes a tiny QARL "training smoke" that proved ordinary AdamW changes a tiny
model's weights and logprobs, then saved and reloaded its state dictionary. None of those assertions distinguished QARL
from ordinary PyTorch behavior. QARL injection, fake-quant arithmetic, summaries, persistent quantization state, and exact
reload remain in the dense fake-quant and calibration lifecycles.

Three other standalone reports held useful behavior but no independent owner. QARL activation-quant enable, disable,
exception, non-QARL exclusion, and nested restoration now close the fake-quant owner. The Qwen3.5 families-v2 effective-
weight and dual residual-gradient wiring now closes the existing norm dispatch and site-assignment policy. LoRA BF16 base
retention, FP32 factors, trainability, dtype resolution, and generic-upcast admission now run with the FP8 and QARL model-
builder construction lifecycle. The three retained owner reports pass.

Repository collection falls to 410 items and the static inventory to 405 definitions, with 1,564 curated decisions across
296 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eightieth wave: delete test-only APIs and reunite isolated contract fragments

The two-hundred-and-eightieth wave removes two source surfaces that existed only to support tests. Canonical MoE no longer
publishes an unused sampler-plan role, launcher alias, single-ordinal accessor, or JSON/digest representation; its retained
trainer policy still proves topology validation and the real distributed collective. Adapter gradient capture likewise no
longer exposes a one-call convenience absent from production. The adapter policies now drive the actual stage, commit, and
abort transaction, including the real multi-rank fatal boundary.

Two useful but overly isolated reports join their contract owners. DeepEP's unsafe async-combine opt-in now closes the same
policy as internode topology, transport preflight, and buffer admission. GLM52 exact routed-ID, scaling, and shared
contributor forwarding now close the exact-MoE construction and global-inventory policy. Both retained owner reports pass.

Repository collection falls to 408 items and the static inventory to 403 definitions, with 1,568 curated decisions across
294 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-first wave: replace isolated units with real owner transactions

The two-hundred-and-eighty-first wave reunites four standalone reports with the runtime policies that consume them. Kimi's
local TikToken loader and safe generic fallback now close its DeepSeek registry/config lifecycle. SignSGD's core dense,
decay, missing-gradient, and sparse-rejection behavior now closes the optimizer policy; its repeated generic parameter-group
assertions are gone. Qwen3 projection unfusing no longer pretends to be a distributed report: the production model-level
transition, TP plan, checkpoint-handler state, and full layer inventory now close the `torch_parallelize` policy.

The BF16 communication lane receives the stronger change. Its stochastic rounding primitive has no production consumer
outside `BF16StochasticAllToAllReduceScatter`, while the old transaction required four GPUs and was normally skipped. One
default-runtime report now proves rounding admission, deterministic seeding, neighbor distribution, unbiased expectation,
and a real two-rank CPU/Gloo all-to-all plus FP32 accumulation against native reduce-scatter. The six focused owner reports
pass. A repeated balanced synthetic-routing fragment is also removed; the retained TopKRouter policy already owns its
complete expert-sequence, balance, uniform-weight, bias, and hash-table behavior.

Repository collection falls to 404 items and the static inventory to 399 definitions, with 1,573 curated decisions across
290 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-second wave: collapse format and kernel policy islands

The two-hundred-and-eighty-second wave removes seven standalone reports whose setup and objects were already owned by a
larger policy. NVFP4 normalization, forward quantization, straight-through gradients, and weight-disable behavior now close
the dense QARL fake-quant owner. MoE sqrtsoftplus/softmax replay regathering now closes TopKRouter's selection, scaling,
dtype, and configuration policy. DSV4's pure-Torch rotation fallback now closes the compressor that consumes it. Qwen3-MoE
layer and final-norm declarations now join the existing dense-Qwen and shared-attention RMSNorm family owner.

The batch-invariant lane loses three artificial boundaries. Families-v2 trainer reachability and its kill switch now close
the v2 norm realization/dispatch report. Families-v2 projection, scoring, invariance, backward, and rollback now close the
fused LM-head transaction. Full-reduce mean and dimensional reductions now close the global Torch-interpose policy that
already owns matmul, RMSNorm, log-softmax, and gradient admission. All nine retained owner reports pass, including their
CUDA execution paths.

Repository collection falls to 397 items and the static inventory to 392 definitions, with 1,580 curated decisions across
283 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-third wave: reunite lifecycle keys, folds, and sync configuration

The two-hundred-and-eighty-third wave removes four more standalone files spanning five reports. LoRA's zero-adapter and
nonzero permanent merge now closes the canonical fold/merged-forward owner for both linear and MoE storage; the same
cast-once contract now runs on CPU rather than sitting behind an unconditional CUDA assumption. Dataset split fingerprints
and preparation hashes now close the dataset loading, splitting, saving, and retry lifecycle that consumes those keys,
retaining determinism, sensitivity, fractional-size, tokenizer, column, and order-independence checks.

Virtual-stage `MultiOptimizer` construction, delegation, single-part fallback, invalid explicit groups, and scheduler
fanout now close the learning-rate scheduler owner; distributed-checkpoint state filtering remains with its checkpoint
owner. QARL-derived FP8 sync metadata, skip lists, handler defaults, quantized buffers, and incompatible overrides now close
the production FP8 `WeightSyncHandler` policy rather than forming a separate QARL report. All six focused owner reports pass.

Repository collection falls to 392 items and the static inventory to 387 definitions, with 1,584 curated decisions across
279 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-fourth wave: retire synthetic gates and reunite exact dispatch policy

The two-hundred-and-eighty-fourth wave removes a synthetic identity-layer checkpoint report that installed `MagicMock`
checkpoint functions and enumerated a local boolean condition. Real Nemotron-H training already proves the default
full-layer checkpoint path executes and propagates gradients, while the GLM-5 model lifecycle proves
`recompute_before_dispatch` bypasses the outer checkpoint and invokes the layer's pre-dispatch checkpoint. The mock truth
table added no independent runtime contract.

Two remaining standalone policy islands join their natural owners. Generic train-router true/false gradients were already
covered by routing-replay and real-model lifecycles; the unique DeepEP rejection now closes the TopKRouter/MoEBlock policy,
and the frozen server default closes configuration serialization. KKT's exact BK, warp, stage, safety, and off-lane launch
behavior now closes the GDN contract policy beside its existing solve-tril serving geometry. The three focused owner files
report seven passes.

Repository collection falls to 389 items and the static inventory to 384 definitions, with 1,587 curated decisions across
276 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-fifth wave: replace component smoke with initialized lifecycle owners

The two-hundred-and-eighty-fifth wave removes DSV4's isolated attention shape smoke. That report manually initialized
`torch.empty` parameters after bypassing model `post_init`, then repeated C0 and C128 forward shape, finiteness, and backward
reachability already owned by the fully initialized DSV4 model lifecycle. The only unique FP8-QAT dispatch and TP>1
admission checks now execute through that full model owner; the direct-layer sink-dtype scenario is gone because production
model casting deliberately preserves the sink in FP32.

Two small policy fragments also join real consumers. Strict LoRA-manifest count, rank, configured-target, unlisted-module,
schema, Boolean, and integer failures now close the fused-GDN injection/checkpoint lifecycle instead of a fake two-layer
attention tree. GLM4 MTP embedding, norm, and head aliases plus ignored auxiliary tail fields now close the GLM4 family
construction and checkpoint lifecycle that creates its ordinary and prequantized handlers. The three focused owner files
report four passes.

Repository collection falls to 386 items and the static inventory to 381 definitions, with 1,590 curated decisions across
273 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-sixth wave: reunite reducer and RoPE parity owners

The two-hundred-and-eighty-sixth wave first rejects a misleading mock-count heuristic. The remaining orchestrator runner,
client, and API-server reports protect different live transactions: rank-zero readiness and tensor-safe serialization,
API-engine ZMQ request handling, and public response metrics. Likewise, low apparent import counts for the model loader,
cautious decay, and Nemotron parallel plans come from package and model-owned call paths, not test-only APIs. Those
boundaries remain.

Two genuinely duplicated loss reports are removed. The standalone importance-sampling and policy-loss files each rebuilt
the shared owner's masked tensors, global `TokenPartial` denominator, full-batch call, microbatch calls, and summable-metric
loop. Microbatch composition now runs beside legacy-identity coverage in the parameterized shared loss contract for basic,
KL, TIS, and IcePop modes. The copied one-test modules are gone.

Dense-Qwen eager RoPE parity also no longer owns an isolated file. Its CUDA Q/K bitwise oracle against the serving
arithmetic now closes the existing frequency-table lifecycle, which already owns fp32 construction, lazy serving caches,
exact-model device recipes, and zero-K3 table bits. All four focused owner reports pass, including the CUDA parity path.

Repository collection falls to 383 items and the static inventory to 378 definitions, with 1,593 curated decisions across
270 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-seventh wave: remove negative spies and close the handler owner

The two-hundred-and-eighty-seventh wave removes a production method that existed only to support negative assertions.
`AdapterCoordinator.broadcast_adapter_optimizer_state` was a deprecated no-op with no runtime caller: topology-specific
optimizer shards already fail closed under rank-zero broadcast and restore through the all-ranks checkpoint path. The dead
method and seven lifecycle spies whose sole claim was that it stayed uncalled are gone. The real transactional optimizer
rejection remains in the adapter-coordinator lifecycle, which passes.

Two one-test weight-sync reports also join the production owner they exercised. Trainer-side HCA selection, physical-GPU
mapping, abort-marker cleanup, and peer-status failure gathering now close the `WeightSyncHandler` configuration and sender
selection policy. The PP NCCL named-tensor codec's empty, sender, metadata, flattened-BF16, scalar, and receiver roundtrips
close the same handler owner. Both standalone private-method modules are removed; all three handler owner reports pass.

Repository collection falls to 381 items and the static inventory to 376 definitions, with 1,595 curated decisions across
268 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-eighth wave: retire stale aliases and close complete construction

The two-hundred-and-eighty-eighth wave removes two deprecated configuration aliases that had become compatibility-only
branches. No shipped configuration or current documentation uses `ep_outside` or `moe_checkpoint_method`; the native
`ep_intranode` and `gradient_checkpointing_method` fields own those policies. The aliases, parser remapping, compatibility
inputs, and redundant simulator dimension are gone. The active `gradient_checkpointing_method="moe_act"` execution mode
remains supported.

GLM-5.2's standalone exact-attention constructor report also duplicated the admission matrix already exercised by the
complete exact-MoE constructor. Its unique 780 attention-factor names, projection classes, source FQNs, per-layer trainable
sets, and three dense roots now close the complete 1,700-factor inventory. The isolated rank/alpha, dense-component,
sparse-MLA, and all-to-all cases are removed; the retained owner and argument/simulator reports produce six focused passes.

Repository collection falls to 380 items and the static inventory to 375 definitions, with 1,597 curated decisions across
267 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-eighty-ninth wave: remove algebra stand-ins for loader and backend owners

The two-hundred-and-eighty-ninth wave removes a prequantized GNK-to-GKN report that never invoked the loader it claimed to
test. It re-derived transpose equivariance for block-FP8 and NVFP4 tensors using local helpers. The retained QLoRA expert
loader owner exercises the real `_load_experts` byte and scale transformations for both formats and multiple shapes, while
the codec owners retain their quantize/dequantize roundtrips.

The standalone GKN-format report is also gone. It rebuilt a manual MoE and repeated `ExpertWeightBuffer` conversion already
exercised through the actual DeepSeek-V3 checkpoint handler, then repeated eager/native agreement owned by the backend
parity lifecycle. Grouped-GEMM and combined QuACK/SGLang parity continue to cover the Triton path without the removed
report's optional-import bypasses. The four retained owner reports produce four focused passes.

Repository collection falls to 377 items and the static inventory to 372 definitions, with 1,599 curated decisions across
265 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-ninetieth wave: unify the Qwen3.5 norm-family contract

The two-hundred-and-ninetieth wave replaces parallel dense and MoE Qwen3.5 RMSNorm reports with one family-wide owner.
Both files independently enumerated the same copied zero-centered dispatch matrix, v2 admission cases, per-site family
propagation, and layer/final residual selection. The unified CPU contract now drives both production classes through those
cases instead of maintaining two model-specific harnesses.

The consolidation preserves the genuinely distinct boundaries: dense linear-attention GDN remains outside the ordinary
RMSNorm-family surface, both model constructors must propagate v2 to every zero-centered site, and the custom v2 plain and
residual backward paths still compare with autograd. One representative GPU lifecycle retains real-kernel module parity,
family-1 interpose bits, full MoE-layer parity, and the family-2 serving tree. The retained CPU report passes.

Repository collection falls to 376 items and the static inventory to 371 definitions, with 1,600 curated decisions across
264 Python test files. The audit still surfaces the same 15 substantive conditional runtime gates, one intentional duplicate
group, and no parse errors.

## Two-hundred-and-ninety-first wave: make the SGLang ABI boundary executable

The two-hundred-and-ninety-first wave fixes the environment contract behind the exact-kernel reports. The default XoRL
profile remains on Torch 2.12.1 and contains no `sglang-kernel`; the combined profile and installation docs now match the
pinned SGLang tree at Torch 2.11.0 instead of the stale Torch 2.9.1 instructions. A separately ignored `.venv-sglang` was
materialized from that pin without changing the default environment.

One test is intentionally added because wrapper imports were not a meaningful gate: SGLang loads its compiled extension
lazily. In optional mode the smoke skips when the wheel is absent from the default profile. With
`XORL_REQUIRE_SGL_KERNEL=1`, absence fails; the test eagerly imports `sgl_kernel`, `hash_topk`, and `LoRABatchInfo`, then
executes a real compiled RMSNorm operation. It skips in the default Torch-2.12 environment and passes in `.venv-sglang`
with Torch 2.11.0 and `sglang-kernel` 0.4.5.

Repository collection rises deliberately to 377 items and the static inventory to 372 definitions, with 1,601 curated
decisions across 265 Python test files. The audit now surfaces 16 substantive conditional runtime gates, one intentional
duplicate group, and no parse errors.

## Two-hundred-and-ninety-second wave: retire rollback-only server and kernel branches

The two-hundred-and-ninety-second wave removes compatibility assertions together with the obsolete behavior that made
them necessary. SGLang fused experts now have one weight-presentation policy: the documented `WEIGHT_MODE` values own
strided, transient, and cached layouts. The test-only `CACHE_WEIGHTS` alias and its precedence cases are gone; cache reuse,
explicit invalidation, and all three actual layouts remain covered.

Server EP dispatch likewise has one correct topology. The undocumented duplicate-batch rollback switch replicated the same
packed batch across every EP rank, causing `ep_size`-times redundant compute. Per-rank EP slices are now unconditional, while
the retained dispatcher and OPD owners still cover EP/CP slice identity, padding, routed payloads, and teacher-cache ordering.

Two P2P compatibility branches are also retired. The disabled fused QKV slicer had no matching locator in pinned SGLang and
could only construct the wrong layout for its separate Q/K/V receivers; canonical locator slicing is now the only path.
Cold-prepare cache invalidation now uses the native `cache_invalidation_mode=none` opt-out instead of a second undocumented
environment alias. The fused-expert, dispatcher, OPD, handler-layout, P2P slicing, and prepare-lifecycle owner reports pass.

Repository collection remains at 377 items and the static inventory at 372 definitions, with 1,604 curated decisions across
265 Python test files. The wave deletes compatibility-only assertion blocks inside retained transaction owners rather than
manufacturing a lower count by splitting or renaming them. The audit still surfaces 16 substantive conditional runtime
gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-ninety-third wave: delete duplicate controls and close native FP8 owners

The two-hundred-and-ninety-third wave removes configuration branches whose tests existed only to preserve a second way to
select an already-owned behavior. Optimizer cache release now uses the native
`skip_empty_cache_after_optim_step` train field instead of an undocumented environment duplicate. Weight sync likewise has
one MoE bucket override, and both manual legacy-receiver post-process switches are gone. Direct P2P writes already target
receiver-native FP8 storage; endpoint requirements remain the owner of KV-cache finalization.

Direct-EP scatter no longer offers debug-only shallow/deep locator-copy modes or their legacy boolean alias. Prepared
locators are immutable in this phase and `scatter_object_list` serializes recipient payloads, so the production default is
now the only behavior. The retained direct-EP manifest owner still proves rank filtering, dense ownership, endpoint state,
and locator identity through the real payload construction path.

The unused NeMo `fp8_cfg` translation is also retired. No shipped XoRL configuration or documentation selected it, while
`enable_fp8_training` and the native `fp8_training_*` fields already own the supported contract. Dataclass aliases,
normalization/extraction APIs, launcher remapping, and compatibility assertions are removed; the shared tombstone rejects
the retired key with an explicit migration message.

That removal makes the standalone FP8 compatibility report unnecessary. Its exhaustive external-knob rejection matrix was
already exercised through the public train and server parsers. The two real behaviors move to runtime owners: BF16 layer
islands execute through FP8 injection, and Blackwell admission executes through `build_training_model`. The focused native
FP8, argument, server, optimizer-step, handler, and P2P owners produce 23 passes.

Repository collection falls to 375 items and the static inventory to 370 definitions, with 1,608 curated decisions across
264 Python test files. The audit still surfaces 16 substantive conditional runtime gates, one intentional duplicate group,
and no parse errors.

## Two-hundred-and-ninety-fourth wave: make numerical programs structural

The two-hundred-and-ninety-fourth wave removes three process-environment rollback paths whose assertions duplicated model
configuration. `XORL_FAMILIES_V2` and `SGLANG_FAMILIES_V2` could move trainer and sampler processes onto different reduction
families even though the exact model program already owns that choice. Ordinary models now use v2, exact Qwen3.5 selects
its qualified v1 norm and LM-head program, and canonical GLM-5.2 selects v2.

The retained numerical reports no longer test environment kill switches. V1 RMSNorm and fused-LM-head owners select the
Qwen program explicitly; the v2 norm owner executes the default production dispatcher; model-builder reports prove GLM v2,
Qwen v1, and restoration to the ordinary program. The public LM-head contract now documents structural selection rather
than a coordinated environment rollback.

`XORL_MOE_ROUTING_WEIGHTS_BEFORE_DOWN` is removed for the same reason. The native
`moe_routing_weights_before_down` model/server field already resolves auto, true, and false before model construction. Its
retained CPU oracle still proves both arithmetic positions against fp64, router-score gradients, no-router-gradient behavior,
dispatch regimes, and the SGLang parity exclusion; only the redundant lazy environment override assertion is gone.

All 14 focused numerical-family, model-program, routing, and handler owners pass. Repository collection remains at 375
items and the static inventory at 370 definitions, with 1,610 curated decisions across 264 Python test files. The audit still
surfaces 16 substantive conditional runtime gates, one intentional duplicate group, and no parse errors.

## Two-hundred-and-ninety-fifth wave: prefer lifecycle owners over private input matrices

The two-hundred-and-ninety-fifth wave removes two standalone reports whose strongest behavior already had a production
lifecycle owner. QARL calibration remains exercised through `build_training_model`, including real calibration-data loading,
observer population, pre-parallelization ordering, and summary counts. Persistent calibration state is retained in the dense
fake-quant owner, so the deleted report's synthetic JSON/JSONL permutations and private bad-shape case no longer form a
separate contract.

R3 Mooncake side payloads likewise remain covered at their actual boundaries. The request processor writes references in
packed order and cleans them on both success and failure; the runner dispatcher consumes only its rank-local slice. The
deleted fake-store report repeated the same codec and metadata flow in isolation, plus narrow malformed-metadata cases that
did not add a distinct server guarantee.

DSV4 RoPE cache capacity now has one authority. `config.max_position_embeddings` owns allocation and the context-parallel
consumer still exercises the too-short-cache failure. The test/profiling-only `XORL_DSV4_ROPE_MAX_SEQ_LEN` override and its
setup assertions are removed. The tensor-collator report is also narrowed semantically: rather than hiding a large scalar,
boolean, string, dimensionality, and batch-size matrix inside one collected test, it now covers the four real pipeline forms
and their type boundaries with one representative sample each.

All 10 focused QARL, R3, DSV4, and collator owners pass. The isolated Torch-2.11 SGLang ABI smoke also passes its real CUDA
operation. Repository collection falls to 373 items and the static inventory to 368 definitions, with 1,614 curated
decisions across 262 Python test files. The audit still surfaces 16 substantive conditional runtime gates, one intentional
duplicate group, and no parse errors.

## Two-hundred-and-ninety-sixth wave: execute hidden kernel claims and remove fake contracts

The two-hundred-and-ninety-sixth wave removes an unshipped SGLang EP slot-combine experiment. The default-off sub-flag had
no configuration or documentation owner, was scoring-only, and its assertions replaced both `moe_sum_reduce` and the
distributed return path with world-size-one fakes. The retained EP path uses the qualified serving-kernel expert compute and
the stock all-to-all combine; real extension loading and the independent FP32-routing backward oracle remain its owners.

Repairing the isolated Torch-2.11 environment made another previously skipped claim executable. The GPU report asserted
bitwise gradients against stock Triton, but failed because stock Triton and the serving wrapper intentionally use different
routing-rounding programs. The wrapper preserves an FP32 routing boundary, so stock equality was not a valid oracle. That
test is deleted; the retained eager oracle proves local and EP input, routing, and weight gradients against the actual
serving program.

Server batch preparation is consolidated at its lifecycle boundary. The deleted `test_batch_utils.py` directly invoked the
non-packed sharder with a one-row batch even though production routes that shape through `TextSequenceShardCollator`.
`RunnerDispatcher` now owns a real two-row ragged teacher-state conversion and CP shard, while the packed collator reports
retain their own path. Duplicate diagnostic environment aliases are removed in favor of request parameters, and an
untested environment-only minimal dummy constructor is removed in favor of the single zero-loss padding lifecycle.

The API tensor report now covers valid rank-1 token IDs, rank-2 teacher states, and rank-3 routing payloads without promising
flat-list fallback semantics for malformed or zero-sized metadata. The isolated SGLang instructions and uv profile also pin
the Quack/CUTLASS versions required to import this XoRL trainer source. A fresh uv resolution succeeds, the real
`sgl_kernel` CUDA operation passes, and the FP32-routing plus DSV4 model, LoRA, and compressor owners pass under Torch 2.11.

The final focused default-environment run produces 9 passes and one intentional optional-kernel skip; the isolated
environment produces 5 passes. Repository collection falls to 371 items and the static inventory to 366 definitions, with
1,620 curated decisions across 261 Python test files. The audit now surfaces 15 substantive conditional runtime gates, one
intentional duplicate group, and no parse errors.

## Two-hundred-and-ninety-seventh wave: replace false-confidence mocks and assertions

The two-hundred-and-ninety-seventh wave keeps `XORL_DCP_LOAD_NO_DIST` because it is a real shared-filesystem recovery mode,
but removes two fake-loader cases that merely echoed `process_group=None` and `no_dist=True`. The GLM exact-DCP owner still
performs a real round trip through that mode. The ModelState owner now also saves and loads a real model-only DCP while
proving that a requested optimizer is not fabricated or mutated; the separate pipeline case still protects custom-group
ordering.

OPD metric finalization no longer guesses its loss family from key prefixes. Production always supplies the resolved
`loss_fn`; only a direct test omitted it. The private finalizer now requires the production signature, and its retained
aggregation, extrema, loss-group reduction, and empty-rank collective-shape policies pass with an explicit `opd_loss`.

Dataset preparation loses two false contracts. `shards` plus `preprocess_shards` was documented as invalid and absent from
shipped configurations, yet a test promised silent precedence. The input is now rejected, while preprocessing expansion
consumes `preprocess_shards` into concrete shard coordinates. The retry lifecycle drops a redundant immediate-success row;
its transient success already proves return propagation alongside backoff, exhaustion, Hub errors, and unrelated failures.

The remaining dataset assertions now test outcomes rather than activity. Merge coverage distinguishes ordered
concatenation, a whole-dataset permutation, and per-dataset permutations instead of checking only row count. Downloaded
`data_files` now honor the documented `ds_type`; JSON-string and Parquet-list cases verify both format routing and resolved
files, closing a bug the former download-count assertions could not detect.

All three focused checkpoint, data-preparation, and OPD policies pass. Full test-tree Ruff, scoped formatting, compileall,
decision-JSON validation, public-tree lint, diff whitespace, global collection, and the static audit pass. Repository
collection remains 371 items and the static inventory remains 366 definitions, with 1,626 curated decisions across 261
Python test files. The audit still surfaces 15 substantive conditional runtime gates, one intentional duplicate group, and
no parse errors.

## Two-hundred-and-ninety-eighth wave: make the isolated kernel gate fail closed

The two-hundred-and-ninety-eighth wave finishes the SGLang dependency boundary rather than treating lazy wrapper imports as
coverage. The default profile remains on Torch 2.12.1 without `sglang-kernel`; the exact SGLang lane remains isolated in
`.venv-sglang` with Torch 2.11.0 and the pinned `sglang-kernel` 0.4.5 wheel. The last stale server-training instructions no
longer install SGLang into the active default environment or advertise the obsolete Torch 2.9.1 contract.

Required smoke mode is now literal. Import-loader failures from either Python or the dynamic linker fail with the active
Torch version, and `XORL_REQUIRE_SGL_KERNEL=1` cannot turn missing CUDA into a successful skip. It must import `sgl_kernel`,
`hash_topk`, and `LoRABatchInfo` and execute the compiled RMSNorm operation. The default environment intentionally skips the
absent optional wheel; the Torch-2.11 environment passes the real CUDA operation.

Repository collection remains 371 items and the static inventory remains 366 definitions, with 1,627 curated decisions
across 261 Python test files. This wave strengthens one retained transaction without adding a new collected test.

## Two-hundred-and-ninety-ninth wave: replace primitive matrices with production transactions

The two-hundred-and-ninety-ninth wave removes both standalone block-FP8 codec reports. They enumerated dimensionality,
random scale ranges, block sizes, determinism, tiny and large constants, signs, and internal assertion failures without
entering a model operation. The retained FP8-linear owner already runs activation and tiled-weight quantize/dequantize at
block sizes 64 and 128, consumes their scale layouts in a real GEMM, and covers a non-divisible weight row count. It now
also compares both dequantized operands with their originals, preserving the useful independent numerical oracle. Five
unexported, unreferenced compatibility aliases disappear with the obsolete primitive vocabulary.

Two test-created implementation contracts are also retired. The real two-GPU sequence-parallel metric report was the only
caller using an obsolete positional order, which forced production to detect a dictionary and swap its arguments. It now
uses the same declared signature as both runtime callers and passes its NCCL partial-sum and extrema transaction. The GLM
absorbed-attention report no longer replaces projections and rotary execution merely to assert two private `einsum`
decompositions; the full-model sparse-versus-dense owner already executes both generic projection directions numerically,
while the retained exact-kv_b owner protects factor-only routing and non-materialization.

Focused GLM and FP8 owners pass, including the real CUDA GEMM and two-GPU NCCL collective. Repository collection falls to
368 items and the static inventory to 363 definitions, with 1,630 curated decisions across 259 Python test files. The audit
still surfaces 15 substantive conditional runtime gates, one intentional duplicate group, and no parse errors.

## Three-hundredth wave: move NF4 coverage into shipped QLoRA lifecycles

The three-hundredth wave removes the standalone NF4 codec report. Its flat and GKN sections enumerated shapes, dtypes,
zeros, codebook constants, three group sizes, and large random tensors without entering a shipped module. Production NF4
QLoRA admits group size 64, not that synthetic matrix. The report also embedded bandwidth benchmarks whose missed target
became a successful skip, so they could neither enforce correctness nor act as a performance gate.

The useful reconstruction oracle now lives at both real consumers. The QLoRA linear owner includes NF4 in its quantized
storage, forward, backward, and memory transaction, then compares the dequantized flat weight with its original operand.
The distributed expert owner compares every production-shaped GKN NF4 projection with its original base before running the
real Triton EP path through two optimizer steps. The flat CUDA lifecycle and direct two-GPU NCCL transaction both pass.

The same size-ordered review explicitly keeps three nearby reports. `ToTensorCollator` uniquely owns conversion and
structural preservation across flat, batched, nested, and empty pipeline forms. The QARL GPU transaction uniquely proves
that fake-quantized expert Parameters reach the real Triton grouped GEMM and receive gradients. The local Qwen3.5 and Kimi
registry reports load actual config and tokenizer artifacts, covering conversion behavior that direct tiny-model builders
do not exercise.

Repository collection falls to 367 items and the static inventory to 362 definitions, with 1,634 curated decisions across
258 Python test files. The audit still surfaces 15 substantive conditional runtime gates, one intentional duplicate group,
and no parse errors.

## Three-hundred-and-first wave: keep protocol and security transactions intact

The three-hundred-and-first wave audits the smallest server and API reports plus the remaining compatibility-labeled
branches. It intentionally removes nothing. API metric mapping, rank-zero wire serialization and readiness, scheduler state
transitions, path containment, SSRF and DNS pinning, checkpoint tenant isolation, and weight-sync receiver fencing are all
distinct runtime boundaries. Their process and socket doubles isolate external systems without replacing the payload,
serialization, queue, filesystem, or byte-layout behavior under test.

The legacy labels also describe live inputs rather than obsolete test fixtures. DRGRPO accepts both current
`old_logprobs` and rollout `logprobs`; Tinker session payloads still map into current model and optimizer controls; supported
checkpoint URI and on-disk forms remain public while unsafe pickle-backed optimizer state fails closed. Removing those
cases would silently narrow compatibility or security guarantees.

The adjacent numerical reports remain independent as well. The DSV4 compressor owner uniquely exercises
context-parallel offsets, cache capacity, C4 overlap admission, and the CPU Hadamard fallback. The batch-invariant GEMM table
is checked numerically against the pinned reduction tree and across batch buckets rather than by asserting literal table
entries. Repository collection therefore remains at 367 items and the static inventory at 362 definitions, with 1,637
curated decisions across 258 Python test files.

## Three-hundred-and-second wave: replace optimizer fakes with real transactions

The three-hundred-and-second wave removes two narrow reports from optimizer and training instrumentation. The standalone
Gram Newton-Schulz CUDA test replaced every backend operation with logging fakes and asserted only the dtypes observed by
those fakes. Its real requirement now lives in the two-GPU full-gradient Muon owner: the shipped CUDA orthogonalizer must
match an independent FP32 Newton-Schulz program exactly, then the distributed update must match the single-rank optimizer
oracle.

The per-component timer loses a disabled no-op test that manually inserted fake objects into private event-pair
dictionaries. Its `last_skipped_event_pair_count` production attribute had no runtime consumer and existed solely for that
assertion, so it is removed. Invalid CUDA event pairs remain safely ignored, while the retained real CUDA lifecycle attaches
hooks to GLM-style and Qwen-style decoder layers and records their present forward and backward phases.

The remaining optimizer reports survive semantic review. Schedule endpoints, DistSignSGD sign and collective ordering,
cautious-decay math, standard batched Newton-Schulz layout, chunked CE parity, token-voter counting, and explicit gradient
synchronization all have independent numerical or collective oracles that end-to-end liveness cannot replace. Focused
owners pass, including real CUDA timer hooks and the strengthened two-GPU Muon transaction. Repository collection falls to
365 items and the static inventory to 360 definitions, with 1,641 curated decisions across 258 Python test files.

The learning-rate trace is also brought back to the production lifecycle. It previously advanced `LambdaLR` without an
optimizer step, emitting PyTorch's skipped-first-value warning even though the trainer always steps the optimizer first.
Single and multi-optimizer traces now follow that real ordering and retain the same exact warmup, decay, and floor oracles
without warnings.

## Three-hundred-and-third wave: make artifact lifecycles own helper semantics

The three-hundred-and-third wave removes synthetic helper coverage from the quantized exporter. String size parsing and
direct-function sharding were separate setup reports even though the module CLI is the public transaction. The retained
subprocess invocation now consumes a `24B` YAML shard limit, writes and reconciles a multi-shard safetensors index, reloads
every emitted shard, and verifies the converted and preserved tensor layouts. This proves the configured string reaches the
artifact writer while deleting the private parser examples and duplicate direct sharding fixture.

Model-support aggregates lose three dominated implementation reports. GLM's bare-config constants duplicated its real
official-shaped local-config load, and its monkeypatched FP32-indexer dispatch was a subset of the canonical GLM-5.2 router
and indexer contract. MiniMax's three private expert-key aliases were already a strict subset of the centralized checkpoint
key classifier. The retained owners still cover GLM sparse/dense and Hugging Face logit parity, MiniMax forward/backward and
checkpoint/EP behavior, and Qwen2/OLMo2 Hugging Face checkpoint conversion plus numerical parity.

The focused exporter and model-support run passes all 6 collected transactions. Scoped Ruff, formatting, compileall,
decision-JSON validation, diff whitespace, global collection, and the static audit pass. Repository collection remains 365
items and the static inventory remains 360 definitions across 258 Python test files, now with 1,644 curated decisions. The
audit still surfaces 15 substantive conditional runtime gates, one intentional duplicate group, and no parse errors.

## Three-hundred-and-fourth wave: retire test-created configuration and rollback surfaces

The three-hundred-and-fourth wave follows typed configuration values into distributed execution instead of preserving a
private conversion vocabulary. FSDP prefetch inputs are booleans at the argument and model-builder boundaries, so their
private parser no longer accepts integer values or yes/no/on/off strings solely for a direct helper matrix. FSDP2 now fails
fast on non-booleans while retaining the forward default, backward inheritance, CP-folding admission, and exact neighbor
prefetch directions.

The optimizer cache field is narrowed the same way. Its lifecycle already supplies a boolean and executes both cache
policies, so the runner no longer interprets arbitrary objects or undocumented truthy strings. The session, optimizer,
P2P-async, checkpoint-broadcast, and adapter-resume reports survive review because they cross real state, transport,
artifact, collective, or failure boundaries; their doubles do not create the behavior being asserted.

MoE slot assignment loses a larger rollback-only surface. `XORL_MOE_DETERMINISTIC_SCATTER` was not exposed by shipped
arguments, examples, or documentation and existed only to restore a relaxed-atomic kernel whose within-expert order varies
with CTA scheduling. The old kernel, environment parser, false-value alias matrix, and mocked route checks are removed.
Stable sorting is now the sole production program, while the retained CUDA transaction independently proves exact stable
order, full slot coverage, per-expert cumsum regions, run invariance, and both routing integer widths.

The focused distributed, runner, and CUDA MoE run produces 6 passes. Repository collection remains at 365 items and the
static inventory remains at 360 definitions across 258 Python test files, now with 1,650 curated decisions. This wave
reduces semantic compatibility and implementation surface inside retained transaction owners rather than manufacturing a
lower collected count.

## Three-hundred-and-fifth wave: finish on authoritative runtime contracts

The final wave re-audits all fifteen conditional-runtime candidates and keeps them. They are real ABI, CUDA, distributed,
or optional-backend gates: the SGLang owner executes a compiled operation in its isolated Torch 2.11 environment; FlashQLA
compares a two-rank context-parallel program with a local reference; GLM exact kernels compose with real FSDP ownership;
and the DeepGEMM and SGLang MoE reports exercise documented production dispatch with numerical or gradient oracles. Their
conditional admission reflects unavailable hardware or dependencies, not a missing outcome.

Three remaining test-maintained compatibility surfaces are removed. Routing-weight position accepts its declared boolean
and `auto`/`true`/`false` forms without undocumented numeric-string aliases. `AdapterState.local_params` is now the sole
parameter store after deleting the deprecated `lora_params` property and test fakes that reproduced it. EP replicated-
gradient synchronization no longer falls back to the broader replicated group: the production classifier always emits
`ep_replicated_gradient_sync`, so missing authoritative metadata now fails fast instead of silently changing the reduction
domain.

The retained owners continue to exercise automatic and explicit routing selection, adapter optimizer and checkpoint
lifecycles, and real multi-rank replicated-gradient coalescing, missing-gradient materialization, nonfinite rejection, and
clipping. Repository collection remains 365 items and the static inventory remains 360 definitions across 258 Python test
files, now with 1,654 curated decisions. The audit closes with fifteen substantive conditional-runtime gates, one
intentional duplicate group, and no parse errors.

## 2026-08-19 current-main rebase

The new pull request replays the historical consolidation onto current `main` after the exact trainer-serving stack merged
in PR 57. At `26ba08b27`, current main collects 4,070 tests with no collection errors and contains 3,272 static test
definitions across 426 Python test files. The initial replay encountered 71 conflict paths, then rebasing after main
advanced through PRs 58, 61, and 62 encountered 13 additional conflict paths. Every conflict resolves to current main; the
older consolidation is applied only where it merges cleanly. Later main updates through PRs 64 and 67 merge cleanly.

Production code is explicitly out of scope for this rebase. The entire `src/` tree matches current main byte-for-byte.
PR 67 removed `src/xorl/rl` and its direct self-test upstream after confirming that its six exports had no caller; merging
main carries that deletion into the branch without adding it to this pull request's diff. Earlier ledger decisions that
removed other production helpers describe the historical audit and are superseded for this PR by TA-1655.

The downstream trace uses XoRL Client PR 14 at `a10d65a5`. Its Wordle adapter imports `xorl_client.rl`, never `xorl.rl`,
and submits `importance_sampling`, `cispo`, or `policy_loss` over the service API. The trainer resolves those names directly
to `xorl.ops.loss`, independently confirming PR 67's conclusion that the standalone Slime-style helpers were not part of
the client request, server dispatch, or executed training objective. Organization-wide code search finds no other source
consumer.

The rebased tree collects 2,012 tests with no collection errors. Its static inventory contains 1,597 definitions across
335 Python test files, 40 scanner candidates, five intentional duplicate-body groups, and no parse errors. Three groups
retain distinct GLM-5.2 and DSV4 temperature-gradient implementations plus wrappers for different distributed workers.
Two additional MoE groups are unchanged current-main conflict owners and remain under the same compatibility rule. The
ledger records 1,661 decisions and explicitly preserves all current-main conflicts and audit candidates rather than
representing them as reviewed removals.

The first rebased CPU workflow exposed four test-only compatibility errors. A dataset aggregate still asserted production
validation and private loader dispatch from the historical source cleanup, a CPU-marked GLM aggregate retained one
unconditional CUDA allocation, and two MoE owners required optional grouped-GEMM globals to pre-exist before installing
their CPU references. Those assertions and helpers now follow the unchanged current-main boundary. With CUDA hidden, the
four failed owners pass and, after merging PR 67, the complete CPU misc shard reports 528 passes, 28 skips, and 463
deselections.
