# P3 upstream manifest: production FP8 and NVFP4 QAT

## Immutable inputs

- Target base: `oss/main@577f34dbc4bd6a3298d93e53d5f0ee10dbeb6178`
- Sanitized source: public staging ref `fca0655d7753e607e54b9a8d98f6064ec42c4ddc`
- Local branch: `codex/oss-p3-low-precision-20260730`
- Worktree: clean sibling checkout for P3; run validation from its root.
- Push policy: local only; this branch was not pushed.

This is the low-precision product slice previously rebuilt for upstream review, refreshed with the later QARL HF-save fix. Shared model/MoE runtime, trainer/server integration, and weight-sync normalization remain dependencies of P1/P2 rather than duplicated here. The earlier upstream split recorded the same boundary: P3 owns FP8/QARL/quantized export, while the shared trainer and FP8 weight-sync wiring is carried by the model/runtime and server packages.

## Lineage disposition

| Internal PR | Disposition | P3 result |
|---|---|---|
| #348 | **ported** | Adds `xorl.fp8_training`, FP8 rowwise/block quantization, QARL calibration/sync primitives, the native quantized exporter, low-precision policy tests, and Hopper E2E coverage. Shared trainer/server/FP8-KV-cache wiring is intentionally consumed from P1/P2 after stacking. |
| #396 | **ported** | Adds NVFP4 weight-only fake quantization, dense and MoE QARL support, servable NVFP4 export, export round-trip tests, and calibration/config validation. Private two-node launch YAML and internal model locations are excluded. |
| #398 | **ported** | Adds W4A4 activation fake quantization, expert down-input quantization, activation override support, and focused CPU/GPU tests. The OPD streaming-forward-KL call-site is shared server-loss code and is consumed from P2 rather than duplicated in this branch. |
| #399 | **ported** | Carries final group-size validation, expert target behavior, and fused gate/up serve-scale parity in the NVFP4 implementation and tests. Internal calibration jobs are excluded. |
| #400 | **superseded** | One-off configs, cluster jobs, benchmark receipts, and environment-specific eval scripts are replaced by portable config/export tests plus `examples/server/configs/export/qwen3_8b_block_fp8_export.yaml`. No raw experiment tree is published. |
| #401 | **ported** | Carries reusable W4/W4A4 activation override behavior and its held-out-mode unit coverage. The server runner call-site remains owned by P2 to avoid conflicting copies of `model_runner.py`. |
| #402 | **superseded** | Script/config corrections are represented by the final #399/#404 implementation and fail-loud tests. Sanitized staging contains no portable remainder of the old cluster/eval scripts. |
| #404 | **ported** | Restores per-half NVFP4 gate/up scaling and locks the behavior with fake-quant/export tests. |
| #427 | **ported** | Filters `qarl_*` observer/scale buffers before HF checkpoint transforms and adds the fused-gate/up regression test. Only the two #427 hunks are carried; unrelated later changes in those shared files are excluded. |

All nine P3 lineage PRs therefore have a final disposition in this package. “Superseded” means the old experiment artifact itself is intentionally absent while its stable product contract is represented by current code/tests.

## Changed-file manifest

Production and portable example:

- `examples/server/configs/export/qwen3_8b_block_fp8_export.yaml`
- `src/xorl/cli/export_nvfp4.py`
- `src/xorl/cli/export_quantized.py`
- `src/xorl/fp8_training/{__init__,config_compat,grouped,linear,profiler,utils}.py`
- `src/xorl/models/module_utils.py`
- `src/xorl/models/transformers/qwen3/modeling_qwen3.py`
- `src/xorl/ops/quantize/{__init__,block_fp8_gkn_quantize,block_fp8_quantize,nvfp4_fake_quant}.py`
- `src/xorl/qarl/{__init__,calibration,fake_quant,moe_experts,sync}.py`

Focused validation:

- `tests/e2e/test_fp8_training.py`
- `tests/fp8_training/{test_config_compat,test_fp8_linear,test_fp8_moe}.py`
- `tests/models/{test_moe_fused_gate_up_proj,test_mtp_low_precision_policy}.py`
- `tests/ops/loss/test_fp8_lm_head_ce.py`
- `tests/ops/{test_block_fp8_gkn,test_nvfp4_fake_quant}.py`
- `tests/qarl/{test_activation_quant_override,test_calibration,test_fake_quant,test_moe_experts_w4a4_down_quant,test_nvfp4_moe_experts,test_nvfp4_moe_gpu,test_nvfp4_qarl,test_sync_quantization,test_training_smoke}.py`
- `tests/scripts/{test_export_nvfp4,test_export_quantized}.py`

## Source fidelity and reconciliation

Before this manifest was added, 33 of the 40 package paths were byte-identical to the required sanitized source commit. Seven paths are deliberate upstream reconciliations:

- `export_quantized.py` and `qarl/sync.py` keep dependency imports lazy until P2 supplies `weight_sync.quantization_config`.
- `module_utils.py` and `test_moe_fused_gate_up_proj.py` carry only #427, excluding later model/runtime changes owned by P1/P5.
- `modeling_qwen3.py` carries the low-precision hidden-state output contract but excludes later P4 RMSNorm-family changes.
- `test_fp8_lm_head_ce.py` excludes later P2 loss-temperature assertions.
- `test_block_fp8_gkn.py` retains the curated `gpu` marker so CUDA kernels are never collected as CPU tests.

## Explicit exclusions

- No `experiments/` tree, Kubernetes/Slurm manifests, raw run logs, benchmark receipts, profiler databases, or model-specific private paths.
- No user-home or shared-storage absolute paths, private hosts, internal repository names, binary shims, or coordination ledgers.
- No one-off W4A4 evaluation shell wrappers. Portable behavior is exercised through library tests and the export example.
- No duplicated P1/P2 trainer/server/weight-sync source. Those dependency branches must land or be stacked before the full runtime/E2E gate is meaningful.

## Validation record

Validation used Python 3.12.3 with this worktree's `src/` first on `PYTHONPATH`.

- Focused CPU collection initially stopped with 17 collection errors because the shared venv did not contain the base repository's `flash-linear-attention` dependency. A cached copy of the pinned FLA source loads only when the unrelated broken TileLang/TVM package is hidden. Hiding TileLang allowed all 195 selected tests to collect without an import error.
- A controlled Hopper training/export/serving parity gate was not run. Some CUDA-visible unit cases did execute during the mixed focused suite, but that is not equivalent to the required P3 runtime gate.
- Final ruff, import/export smokes, private-reference scan, and exact focused-test results are recorded below.

### Final commands and results

- `ruff check <39 changed Python files>`: **passed** (`All checks passed!`).
- `ruff format --check <39 changed Python files>`: **passed** (`39 files already formatted`).
- Focused mixed suite (FP8, QARL, quantization ops, save filtering, and both exporters), with this worktree on `PYTHONPATH`, cached pinned FLA, and incompatible TileLang hidden: **146 passed, 37 failed, 1 skipped, 11 not reached after an intentional stop at 87%; 284.74 s**.
  - Passing evidence includes all 15 FP8 config-compatibility cases, the FP8 linear suite, dense QARL calibration/fake-quant/training tests, 12 NVFP4 dense-QARL tests, 22 NVFP4 primitive tests, both MTP low-precision policy tests, all 7 QARL HF-save/fused-projection tests, and all 5 NVFP4 exporter tests.
  - The 37 failures are expected stack dependencies rather than private-source omissions: FP8/QARL MoE and Quack backend hooks are owned by P1 (the earlier split called this dependency #407); QARL sync normalization, quantized-export normalization, and handler wiring are owned by P2 (#411); FP8 LM-head call-site support is shared loss/trainer code owned by P1/P2.
  - The run was stopped on the parent packaging coordinator's speed instruction once those dependency groups were fully identified. The branch must be retested after stacking P1/P2.
- Staged-diff private/provenance scan over 41 files: **passed**. No absolute home/scratch paths, private repo/host names, cluster scheduling fields, binary shims, forbidden co-author trailers, or generated-by references were found.
- `git diff --cached --check`: **passed**.
