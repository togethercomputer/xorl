---
title: Running Tests
---

## Quick commands

```bash
# Run everything
pytest tests/

# Stop on first failure
pytest tests/ -x

# Run a single file
pytest tests/data/test_data_loader.py -v

# Run a single test
pytest tests/data/test_data_loader.py::TestMicroBatchCollator::test_micro_batch_splitting -v
```

## By category

```bash
pytest tests/data/         # Data pipeline
pytest tests/models/       # MoE and LoRA model logic
pytest tests/ops/          # Low-level ops (GPU)
pytest tests/qlora/        # QLoRA (GPU)
pytest tests/server/       # Server infrastructure
pytest tests/e2e/          # End-to-end (GPU + torchrun)
```

## By marker

```bash
pytest tests/ -m cpu          # Only CPU tests
pytest tests/ -m gpu          # Only GPU tests
pytest tests/ -m "not slow"   # Skip slow tests
pytest tests/ -m collator     # Only collator tests
pytest tests/ -m server       # Only server tests
```

## Distributed tests

Distributed tests work with plain `pytest` — tests that require multiple processes spawn torchrun internally as a subprocess (same pattern as e2e tests):

```bash
pytest tests/distributed/ -v
```

## End-to-end tests

E2E tests call `torchrun` internally per test, so they run with plain `pytest` — no wrapper needed.

Use the helper script from the repo root:

```bash
# All e2e tests
./tests/e2e/run_e2e.sh

# One model suite
./tests/e2e/run_e2e.sh qwen3_8b

# One file
./tests/e2e/run_e2e.sh qwen3_8b/test_lora.py
```

Or invoke pytest directly:

```bash
pytest tests/e2e/ -m e2e -v
pytest tests/e2e/qwen3_8b/test_pp.py -v
```

Make sure the environment is set up per the [Installation guide](/xorl/getting-started/installation).
