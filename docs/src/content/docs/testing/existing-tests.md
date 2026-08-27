---
title: Existing Tests
---

The test suite is organized by product surface. This page intentionally lists directories and representative live tests rather than attempting to mirror every filename.

| Area | Coverage | Representative tests |
|---|---|---|
| `tests/checkpoint/` | Checkpoint process groups and EP mesh handling | `test_ep_checkpoint_mesh.py` |
| `tests/data/` | Dataset preparation, packing, collators, dataloaders | `test_data_loader.py`, `collators/test_packing_concat_collator.py` |
| `tests/distillation/` | Teacher-state storage and transport | `test_mooncake_hidden_store.py` |
| `tests/distributed/` | FSDP/TP/PP/EP/CP collectives and numerical contracts | `test_canonical_moe_contract.py`, `test_deepep_async_combine_guard.py` |
| `tests/e2e/` | Small-model end-to-end training under torchrun | `qwen3_8b/test_lora.py`, `qwen3_30b/test_server_moe.py` |
| `tests/experiments/` | Training simulator behavior | `test_training_sim.py` |
| `tests/fp8_training/` | Full-weight FP8 configuration, linears, and MoE | `test_fp8_linear.py`, `test_fp8_moe.py` |
| `tests/models/` | Registry, model loading, attention, batch invariance, architecture guards | `test_dsv4_exact_contract.py`, `test_dsv4_native_combine.py` |
| `tests/ops/` | Losses, quantization, DSV4, MoE, attention, and kernels | `test_sgl_kernel_smoke.py`, `dsv4/test_compressor.py` |
| `tests/optim/` | AdamW/Muon/DistSignSGD and scheduler behavior | `test_muon.py`, `test_distsignsgd.py` |
| `tests/qarl/` | Calibration and fake-quant paths | `test_fake_quant.py`, `test_nvfp4_moe_experts.py` |
| `tests/qlora/` | QLoRA detection, loading, adapters, and kernels | `test_detect_prequantized.py`, `test_qlora.py` |
| `tests/scripts/` | Export and OPD payload scripts | `test_export_quantized.py`, `test_opd_pipeline_payloads.py` |
| `tests/server/` | API schemas, orchestration, backends, runners, and all weight-sync transports | `api_server/test_api_types.py`, `weight_sync/` |
| `tests/trainers/` | Trainer construction and architecture-specific training guards | `test_fp8_model_builder.py`, `test_deepseek_v3_training_guards.py` |
| `tests/utils/` | Teacher caches for distillation | `test_distillation_teacher_cache.py` |

Get the current inventory:

```bash
find tests -type f -name 'test_*.py' | sort
```

Presence of a conventional test demonstrates the checked-in mechanism and its tested scope. GPU/backend availability, full-model behavior, trainer/sampler revision-pair exactness, and production qualification remain separate gates.
