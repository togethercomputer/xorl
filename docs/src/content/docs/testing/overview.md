---
title: Testing Overview
---

The test suite lives in `tests/` and covers data loading, model logic, distributed training, low-level ops, quantization, server infrastructure, and end-to-end training runs.

## Layout

```
tests/
├── conftest.py              # Shared fixtures (datasets, collators)
├── data/                    # Data loading and preparation
│   ├── collators/           # Collator pipeline tests
│   └── prepare/             # Dataset hashing, packing, loading
├── distributed/             # Multi-process parallelism tests
├── models/                  # MoE routing, LoRA injection, weight loading
├── ops/                     # Low-level tensor ops, attention, quantization
├── qlora/                   # QLoRA quantization tests
├── server/
│   ├── api_server/          # HTTP API endpoints and types
│   ├── orchestrator/        # Scheduling, batching, messaging
│   ├── runner/              # Ready signal and runner communication
│   └── weight_sync/         # Pipeline-parallel weight sync
└── e2e/                     # Full training pipeline (requires GPU + torchrun)
    ├── qwen3_8b/
    └── qwen3_30b/
```

## Pytest markers

Markers are defined in `pyproject.toml` and used to select subsets of tests:

| Marker | Meaning |
|---|---|
| `cpu` | No GPU required |
| `gpu` | Requires at least one GPU |
| `distributed` | Requires `torchrun` with multiple processes |
| `e2e` | Full end-to-end test (GPU + torchrun) |
| `server` | API server, orchestrator, or runner tests |
| `collator` | Data collator tests |
| `dataloader` | DataLoader tests |
| `slow` | Long-running tests |
| `benchmark` | Performance benchmarks |
