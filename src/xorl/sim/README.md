# XoRL Training Simulator

`xorl.sim` is the installable, CPU-only planning and calibration surface for XoRL local training. It resolves distributed
topology, calculates token and model shapes, builds analytical FLOP/activation/communication/memory ledgers, ingests
structured trainer logs, and ranks measured or extrapolated training scenarios.

The package deliberately contains no cluster launcher, scheduler configuration, storage configuration, or workload
manifest. Execution infrastructure remains outside the simulator.

## Installed Commands

After installing XoRL, the following commands are available:

| Command | Purpose |
| --- | --- |
| `xorl-sim-packs` | List, locate, or validate built-in calibration packs |
| `xorl-sim-predict` | Build a static report for one config |
| `xorl-sim-plan` | Compare micro-batch and parallelism scenarios |
| `xorl-sim-calibrate` | Run leave-one-out prediction validation |
| `xorl-sim-feasibility` | Replay measured fit and OOM boundaries |
| `xorl-sim-rank` | Rank raw and correctness-promotable measurements |
| `xorl-sim-kernels` | Rank portable kernel-variant measurements |
| `xorl-sim-validate` | Run pack sanitation and analytical golden gates |
| `xorl-sim-collect` | Summarize a portable observed-run calibration record |

## Built-In Qwen Packs

```bash
xorl-sim-packs list
xorl-sim-packs validate
xorl-sim-predict --pack qwen3_6_35b_a3b --balanced-routing
xorl-sim-plan --pack qwen3_5_397b_a17b --micro-batch-sizes 4,5
xorl-sim-feasibility --pack qwen3_235b_a22b
```

The built-in packs are:

- `qwen3_235b_a22b`: 4-node 2k-context GA observations and an OOM boundary.
- `qwen3_5_397b_a17b`: 8-node short-context throughput rows with separate raw and static-K3-promotable winners.
- `qwen3_6_35b_a3b`: 4-node 8k throughput, allocator-pressure, and failed static-K3 evidence.

Each pack contains `manifest.json`, portable YAML configs, summarized observations, limitations, and frozen golden values.
Pack validation rejects missing files, model/config mismatches, internal absolute paths, and infrastructure-specific content.

## External Calibration Packs

All APIs continue to accept a filesystem benchmark directory. A portable pack has this shape:

```text
my_model/
  manifest.json
  README.md or RESULTS.md
  configs/
    training.yaml
  results/
    measurements.json
```

The manifest schema version is currently `1`. `configs` and `results` are relative paths, `default_config` selects the
prediction baseline, and `golden` freezes the expected behavior count, topology, throughput, and analytical memory floor.
Use `xorl-sim-packs validate path/to/my_model` before relying on a new pack.

The behavior reader also accepts results roots containing resolved run directories. A directory containing
`xorl_cli.yaml` plus a sibling `node-0.log` is ingested as observed evidence. Structured `[STEP ...]`, `[STEP_PHASES ...]`,
and `[STEP_MEMORY ...]` rows provide throughput, timing, and memory calibration. OOM logs become measured failure
boundaries rather than being discarded.

## Prediction And Planning

```bash
xorl-sim-predict \
  --config path/to/xorl_config.yaml \
  --balanced-routing \
  --benchmark-dir path/to/calibration_pack \
  --output prediction.json

xorl-sim-plan \
  --config path/to/xorl_config.yaml \
  --benchmark-dir path/to/calibration_pack \
  --micro-batch-sizes 1,2,4 \
  --expert-parallel-sizes 8,16 \
  --topology-sweep auto \
  --output scenarios.json
```

Reports distinguish exact calibration, interpolation, extrapolation, observed failures, and missing support. Raw throughput,
risk-adjusted throughput, correctness promotion, memory feasibility, timing coverage, and communication scope remain
separate fields. An extrapolated candidate is never silently promoted as a measured winner.

## Analytical Core

The analytical surface includes:

- Dense and MoE FLOP ledgers matching XoRL trainer accounting.
- Hybrid full-attention/GatedDeltaNet model metadata and parameter ownership.
- Activation lower bounds with recompute, CE-logit, attention, and MoE buffer terms.
- Per-rank FSDP, TP, PP, EP, and context-parallel communication bytes and cross-node scope.
- Parameter, gradient, optimizer, unshard, calibrated residual, and peak-memory attribution.
- Observed or benchmark-derived timing ledgers and explicit support/blocker reporting.

Analytical memory remains a lower bound until a pack supplies calibrated peak or residual evidence. Timing requires measured
step or phase data; the simulator does not infer hardware timing from communication bytes alone.

## Kernel Variants

Kernel comparison consumes portable JSON measurements instead of loading files from a particular machine:

```json
[
  {
    "family": "attention",
    "variant": "candidate_a",
    "workload": "qwen35-seq4096",
    "latency_ms": 8.5,
    "correctness_status": "pass"
  }
]
```

```bash
xorl-sim-kernels measurements.json
```

Measurements must share a kernel family and workload. By default, the fastest ungated row is reported but cannot become
the selected winner.

## Validation

```bash
xorl-sim-validate --output validation.json
```

The consolidated validator checks every installed pack, frozen behavior counts, default topology, calibrated throughput,
correctness-aware rankings, analytical memory goldens, and FLOP/activation/communication coverage. It exits nonzero on any
failed check and is intended to run in CPU CI and against an installed wheel.

Built-in results are historical synthetic-data observations on H100 GPUs. They are regression fixtures and calibration
priors, not universal performance claims or substitutes for a fresh correctness gate on a changed runtime.
