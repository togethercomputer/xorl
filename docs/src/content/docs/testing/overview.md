---
title: Testing Overview
---

The test suite lives in `tests/` and covers data loading, model logic, distributed training, low-level ops, quantization, server infrastructure, and end-to-end training runs.

## Layout

```
tests/
├── checkpoint/              # Checkpoint process groups and EP meshes
├── data/                    # Loading, preparation, collators, and packing
├── distillation/            # Teacher-state transport and caches
├── distributed/             # Parallelism and collective contracts
├── e2e/                     # Full training pipelines (GPU + torchrun)
├── experiments/             # Simulator and experiment utilities
├── fp8_training/            # Full-weight FP8 configuration and kernels
├── models/                  # Architecture, numerical, and loading contracts
├── ops/                     # Tensor, attention, loss, MoE, and quantization ops
├── optim/                   # Optimizers and schedulers
├── qarl/                    # QARL fake-quant paths
├── qlora/                   # Quantized LoRA paths
├── scripts/                 # Export and pipeline scripts
├── server/                  # API, orchestration, runner, backend, and weight sync
├── trainers/                # Local/server trainer behavior
└── utils/                   # Shared utility tests
```

The suite changes frequently. Use `find tests -type f -name 'test_*.py' | sort` for the exact inventory at your checkout rather than relying on a static file list.

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
