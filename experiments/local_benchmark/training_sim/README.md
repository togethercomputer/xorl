# XoRL Training-Engine Simulator

This is a CPU-only first slice of the local training-engine simulator. It resolves the launch topology from a
XoRL YAML config, computes deterministic balanced-routing token shapes, estimates a sharded persistent model-state
memory floor, and parses structured trainer logs into calibration summaries.

It does not yet model activation, attention workspace, MoE kernel workspace, FSDP transient, or allocator slack
memory. Those are left as explicit `unsupported_buckets` in the JSON report until calibrated formulas are added.

## Predict From A Config

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.predict \
  --config path/to/xorl_config.yaml \
  --world-size 16 \
  --local-world-size 8 \
  --balanced-routing
```

## Add Log Calibration

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.predict \
  --config path/to/xorl_config.yaml \
  --world-size 16 \
  --local-world-size 8 \
  --balanced-routing \
  --num-experts 128 \
  --top-k 8 \
  --logs /shared/path/to/trainer-head/logs/run.log \
  --warmup-steps 3 \
  --output experiments/local_benchmark/training_sim/calibration/report.json
```

Pass `--benchmark-dir` to include empirical behavior calibrated from a recipe README and result JSON:

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.predict \
  --config path/to/benchmark/configs/xorl_cli.yaml \
  --balanced-routing \
  --benchmark-dir path/to/benchmark
```

Empirical matches are config-specific. The simulator checks both topology/workload shape and runtime knobs such as
`deepep_async_combine`, `deepep_num_sms`, `deepep_buffer_size_gb`, `enable_compile`,
`gradient_checkpointing_method`, activation offload, and prefetch count before treating a benchmark row as an exact
calibration point.

`--benchmark-dir` can also point at a results root containing resolved run directories. Any subdirectory with
`xorl_cli.yaml` is treated as a candidate calibration source. If a matching `node-0.log` is available directly beside
the config or through `startup_metrics.json`'s `startup/master_addr`, the loader parses measured `[STEP ...]` rows
with two warmup steps excluded. OOM logs become calibrated failure boundaries, and runs that report throughput before
crashing are kept as partial-failure calibration points rather than clean promotion candidates.

## Rank Benchmark Tradeoffs

Use the tradeoff ranker to compare autotune rows in a benchmark folder. It keeps the fastest raw
candidate separate from the fastest correctness-promotable candidate, so a raw-speed win is not promoted unless it
has a matching `k3_pass` gate.

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.tradeoff_ranker \
  path/to/benchmark
```

The report keeps a faster raw candidate separate from a slower promotable candidate when only the latter has a
matching correctness gate.

## Plan What-If Scenarios

Use the scenario planner when the question is not just "what already won?" but "what should we try next?". It
mutates a base config across micro-batch and parallelism choices, computes a topology and sharded model-state memory
floor for each candidate, then ranks exact calibrated matches ahead of lower-confidence extrapolations. Extrapolated
candidates are never marked promotable; they need a fresh K3 gate.

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.scenario_planner \
  --config path/to/benchmark/configs/xorl_cli.yaml \
  --benchmark-dir path/to/benchmark \
  --micro-batch-sizes 5 \
  --expert-parallel-sizes 32,64
```

The planner compares concrete parallelism tradeoffs while preserving correctness, runtime-compatibility, and
memory-feasibility caveats.

For wider topology searches, add `--topology-sweep auto`. Auto mode derives legal candidate values for EP, TP, PP,
Ulysses, and Ring from the resolved world size and model metadata, while explicit comma lists still override any
individual dimension. Exact empirical matches are conservative: an observation only matches TP/PP/Ulysses/Ring values
known from that artifact, and legacy artifacts with missing topology dimensions only exact-match the default value of
1. Non-default TP/PP/CP candidates therefore remain extrapolated unless there is a measured row for that exact
topology.

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.scenario_planner \
  --config path/to/benchmark/configs/xorl_cli.yaml \
  --benchmark-dir path/to/benchmark \
  --gradient-accumulation-steps 1,2,4,8 \
  --topology-sweep auto
```

The planner also understands markdown result tables with `tok/s tot`, `tok/step`, and `peak GB` columns, such as the
Qwen3-235B 2k-context sweep. Observed peak memory overrides the analytic floor for feasibility checks, and OOM rows
are kept as calibrated failures when their topology and pack length match a scenario. When two or more
global-batch/GA points are calibrated for the same topology, it fits a simple
`step_time = fixed_overhead + token_slope * tokens` curve for larger GA what-ifs:

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.scenario_planner \
  --config path/to/benchmark/configs/xorl_cli.yaml \
  --benchmark-dir path/to/benchmark \
  --micro-batch-sizes 1 \
  --gradient-accumulation-steps 1,2,4,8 \
  --expert-parallel-sizes 8
```

For that q235 scenario, GA2 is a calibrated measured point and GA4/GA8 are step-time-fit extrapolations. They are
useful next-run candidates, not correctness-promotable results.

Planner candidates include `calibration_scope` and `risk_flags` fields. `exact_calibrated` means an empirical row
matched the full scenario topology. `inside_measured_envelope` and `outside_measured_envelope` describe whether an
extrapolated candidate stays inside the measured micro-batch/global-batch/parallelism range for that sequence length.
Risk flags call out cases like `requires_remeasurement`, `matched_allocator_pressure_slowdown`, an
`allocator_pressure_boundary:*`, an `observed_oom_boundary:*`, `correctness_runtime_failure_after_steps`, or
`runtime_mismatch:*` when extrapolation had to fall back to a row with different runtime knobs. Treat these flags as
launch-planning constraints: they do not erase the raw score, but they mean the row needs a fresh measurement or debug
pass before it can be used as an optimum.

Scenario reports keep `best_raw` as the fastest feasible throughput hypothesis, then add `best_risk_adjusted` and
`best_next_measurement`. The risk-adjusted score penalizes extrapolation, memory pressure, missing correctness gates,
allocator-pressure slowdowns, and observed-OOM boundaries. This makes the planner useful as an optimizer loop: launch
the best next measurement when it is a hypothesis, but prefer the risk-adjusted or promotable row when choosing what is
already defensible.

For exact calibrated rows, an observed peak below the device limit remains feasible even when the configured safety
factor would reserve slightly more than the device capacity. Those rows are marked `feasible_calibrated_peak_high_pressure`.
The safety margin still gates extrapolated peaks and analytic floors.

## Validate Prediction Fidelity

Use the calibration evaluator before trusting a scenario sweep as an optimizer. It runs leave-one-out validation over
measured benchmark rows, rebuilds the held-out topology, predicts it from the remaining calibration points, and reports
actual-vs-predicted error.

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.calibration_evaluator \
  --config path/to/benchmark/configs/xorl_cli.yaml \
  --benchmark-dir path/to/benchmark
```

Treat a large holdout error as a sign that the relevant lever needs a new calibration point or a more specific
simulator feature before promotion decisions rely on it.

## Parse Logs Only

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.collect_calibration \
  /shared/path/to/trainer-head/logs/run.log \
  --warmup-steps 3 \
  --world-size 16
```

The log parser recognizes `[STEP ...]`, `[STEP_PHASES ...]`, and `[STEP_MEMORY ...]` lines emitted by
`src/xorl/trainers/trainer.py`.

## Validate Checked-In Benchmarks

```bash
cd "$(git rev-parse --show-toplevel)"
python -m experiments.local_benchmark.training_sim.validate_benchmarks \
  --benchmarks-root path/to/benchmarks \
  --model benchmark_name
```

The validator checks benchmark YAML, README target metrics, synthetic-routing render scripts, stored throughput
summaries, and static-K3 gate status when those artifacts are present.
