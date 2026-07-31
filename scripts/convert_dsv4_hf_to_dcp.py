"""Convert a DeepSeek-V4 HF safetensors snapshot into an xorl DCP checkpoint.

When launched with ``torchrun`` this is a true distributed converter: every
rank materializes its FSDP2/EP-local shard from the HF snapshot and DCP writes
one shard per rank. A single-process CPU fallback remains for tiny unit tests
and emergency one-off conversions, but production Flash/Pro conversion should
use the distributed path.

Usage::

    torchrun --nproc_per_node=8 --nnodes=2 \\
        scripts/convert_dsv4_hf_to_dcp.py \\
        --hf-snapshot /path/to/hf-cache/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/<sha> \\
        --dcp-out /path/to/checkpoints/deepseek-v4-flash-dcp \\
        --ep-size 16

Memory footprint at peak ~ FP8-on-disk + BF16 model + per-layer expert
buffer. For Flash that's roughly ``149GB + 280GB + 12GB ≈ 440GB``; pick a
node with at least 512GB RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from xorl.checkpoint.checkpointer import DistributedCheckpointer as Checkpointer
from xorl.models.module_utils import init_empty_weights
from xorl.models.transformers.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    cast_dsv4_model_dtype,
    stream_load_hf_directory_into_model,
)
from xorl.utils.logging import get_logger
from xorl.utils.mmap_alloc import MmapTensorAllocator


logger = get_logger(__name__)


def _say(msg: str) -> None:
    """Unbuffered progress print. The xorl logger can drop messages depending
    on how the root logger is configured at script start; ``print`` with
    ``flush=True`` is the dependable way to show progress in
    ``pod logs``."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_config_to_xorl(snapshot_dir: Path) -> DeepseekV4Config:
    """Read the HF ``config.json`` and adapt it to ``DeepseekV4Config``."""
    cfg_path = snapshot_dir / "config.json"
    with cfg_path.open() as f:
        raw = json.load(f)

    class _Obj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    return DeepseekV4Config.from_hf_config(_Obj(raw))


def _maybe_truncate_config(cfg: DeepseekV4Config, num_hidden_layers: int | None) -> None:
    if num_hidden_layers is None or num_hidden_layers == cfg.num_hidden_layers:
        return
    full_n = cfg.num_hidden_layers
    cfg.num_hidden_layers = num_hidden_layers
    if cfg.compress_ratios is not None:
        cfg.compress_ratios = cfg.compress_ratios[:num_hidden_layers]
    _rank0_say(f"truncated num_hidden_layers: {full_n} -> {cfg.num_hidden_layers}")


def _init_single_rank_pg():
    """Init a single-rank gloo PG so ``dcp.save`` is callable."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _is_torchrun_multi_rank() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank0_say(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        _say(msg)


def _save_dcp(path: Path, model: torch.nn.Module) -> None:
    """Save a DCP, tolerating CPU-only single-process conversion."""
    if not torch.cuda.is_available():
        # ``Checkpointer.save`` ends with ``torch.cuda.synchronize()`` for
        # cleanup. On a CPU-only conversion process that raises even though the
        # DCP files have already been written.
        _orig_sync = torch.cuda.synchronize
        torch.cuda.synchronize = lambda *_a, **_kw: None
        try:
            Checkpointer.save(str(path), {"model": model})
        finally:
            torch.cuda.synchronize = _orig_sync
    else:
        Checkpointer.save(str(path), {"model": model})


def _barrier(local_rank: int | None = None) -> None:
    if local_rank is not None and torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _distributed_main(args: argparse.Namespace) -> int:
    """Distributed HF -> DCP conversion path used under torchrun."""
    if args.disk_backed_staging is not None:
        raise ValueError("--disk-backed-staging is only supported by the single-process fallback")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_id = None
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(backend=backend, timeout=timedelta(hours=8), device_id=device_id)
    except TypeError:
        dist.init_process_group(backend=backend, timeout=timedelta(hours=8))
    rank = dist.get_rank()
    world = dist.get_world_size()

    ep_size = args.ep_size or world
    if world % ep_size != 0:
        raise ValueError(f"WORLD_SIZE={world} must be divisible by --ep-size={ep_size}")

    from xorl.distributed.parallel_state import init_parallel_state  # noqa: PLC0415
    from xorl.distributed.torch_parallelize import build_parallelize_model  # noqa: PLC0415

    init_parallel_state(
        dp_size=world,
        dp_shard_size=world,
        ep_size=ep_size,
        dp_mode="fsdp2",
    )

    target_dtype = getattr(torch, args.target_dtype)
    if rank == 0:
        args.dcp_out.mkdir(parents=True, exist_ok=True)
    _barrier(local_rank)

    t0 = time.time()
    _rank0_say(f"distributed conversion: world={world} ep_size={ep_size} backend={backend}")
    _rank0_say(f"loading HF config from {args.hf_snapshot}")
    cfg = _hf_config_to_xorl(args.hf_snapshot)
    _maybe_truncate_config(cfg, args.num_hidden_layers)
    _rank0_say(
        f"  hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"n_routed_experts={cfg.n_routed_experts}, top_k={cfg.num_experts_per_tok}"
    )

    _rank0_say("instantiating DSv4 on meta")
    with init_empty_weights():
        model = DeepseekV4ForCausalLM(cfg, moe_implementation=args.moe_implementation)
    model._dsv4_dequantize_fp8 = not args.skip_fp8_dequant
    model._dsv4_target_dtype = target_dtype
    cast_dsv4_model_dtype(model, target_dtype)

    _rank0_say(f"loading HF snapshot with {args.load_weights_mode} distributed load")
    t_load = time.time()
    model = build_parallelize_model(
        model,
        weights_path=str(args.hf_snapshot),
        enable_full_shard=True,
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
        skip_param_upcast=True,
        init_device="meta",
        basic_modules=["DeepseekV4DecoderLayer"],
        load_weights_mode=args.load_weights_mode,
    )
    _barrier(local_rank)
    _rank0_say(f"  distributed HF load done in {time.time() - t_load:.1f}s")

    _rank0_say(f"writing distributed DCP to {args.dcp_out}")
    t_save = time.time()
    _save_dcp(args.dcp_out, model)
    _barrier(local_rank)
    if rank == 0:
        metadata = args.dcp_out / ".metadata"
        distcp_files = list(args.dcp_out.glob("*.distcp"))
        if not metadata.exists() or not distcp_files:
            raise RuntimeError(
                f"DCP write did not produce expected files in {args.dcp_out}: "
                f"metadata={metadata.exists()}, .distcp count={len(distcp_files)}"
            )
        total_gb = sum(p.stat().st_size for p in [metadata, *distcp_files]) / 1e9
        _say(f"  DCP write done in {time.time() - t_save:.1f}s ({len(distcp_files)} shards, {total_gb:.1f} GB on disk)")
        _say(f"total wall time: {time.time() - t0:.1f}s")
    dist.destroy_process_group()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hf-snapshot",
        required=True,
        type=Path,
        help="Path to the HF snapshot dir (the one that contains config.json + *.safetensors).",
    )
    parser.add_argument(
        "--dcp-out",
        required=True,
        type=Path,
        help="Output directory for the DCP checkpoint.",
    )
    parser.add_argument(
        "--target-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="dtype to dequantize FP8 weights into. Defaults to bfloat16.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any HF tensor is unmapped or any model param is unfilled.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help="Expert-parallel size for torchrun distributed conversion. Defaults to WORLD_SIZE.",
    )
    parser.add_argument(
        "--load-weights-mode",
        default="all_ranks",
        choices=["all_ranks", "broadcast"],
        help=(
            "Distributed HF load mode. all_ranks lets each EP rank read/fuse only its local experts; "
            "broadcast is mostly for dense/small checkpoints."
        ),
    )
    parser.add_argument(
        "--moe-implementation",
        default="triton",
        choices=["eager", "triton", "native", "quack"],
        help="MoE implementation used to instantiate the model. No forward is run during conversion.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Optional layer truncation for converter smokes. Omit for full model conversion.",
    )
    parser.add_argument(
        "--skip-fp8-dequant",
        action="store_true",
        help="Pass through FP8 weights without dequantizing (for debugging).",
    )
    parser.add_argument(
        "--disk-backed-staging",
        type=Path,
        default=None,
        help="When set, materialize model parameters in a sparse mmap'd file "
        "at this path instead of in RAM. Required when the model footprint exceeds "
        "host RAM (DSv4 Pro BF16 is ~3.1 TB). Use a fast shared volume "
        "(e.g. /path/to/fast-storage/dsv4_pro_staging.bin). Trade-off: disk-bound "
        "wall time vs a Flash-sized in-RAM conversion.",
    )
    parser.add_argument(
        "--disk-backed-capacity-bytes",
        type=int,
        default=4 * (1 << 40),  # 4 TiB
        help="Sparse-file capacity for ``--disk-backed-staging``. Defaults to "
        "4 TiB which comfortably fits Pro's 3.1 TB BF16 footprint plus headroom.",
    )
    args = parser.parse_args(argv)

    if _is_torchrun_multi_rank():
        return _distributed_main(args)

    target_dtype = getattr(torch, args.target_dtype)
    args.dcp_out.mkdir(parents=True, exist_ok=True)

    # Single-rank distributed init for dcp.save.
    _init_single_rank_pg()

    t0 = time.time()
    _say(f"loading HF config from {args.hf_snapshot}")
    cfg = _hf_config_to_xorl(args.hf_snapshot)
    _maybe_truncate_config(cfg, args.num_hidden_layers)
    _say(
        f"  hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"n_routed_experts={cfg.n_routed_experts}, top_k={cfg.num_experts_per_tok}"
    )

    _say("instantiating empty xorl DeepseekV4ForCausalLM on meta device")
    # eager backend works on CPU; for GPU training the backend is read from the
    # training config — this conversion only sets weights, no forward.
    with init_empty_weights():
        model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")

    # Set up the materialization allocator. In-RAM mode allocates straight
    # ``torch.empty`` on CPU; disk-backed mode bumps allocations into a sparse
    # mmap'd file. The latter is required when the BF16 model footprint exceeds
    # the host RAM available to a single conversion process.
    mmap_allocator: Optional[MmapTensorAllocator] = None
    if args.disk_backed_staging is not None:
        mmap_allocator = MmapTensorAllocator(args.disk_backed_staging, args.disk_backed_capacity_bytes)
        _say(
            f"materializing parameters on disk-backed mmap "
            f"(staging: {args.disk_backed_staging}, "
            f"capacity: {args.disk_backed_capacity_bytes / (1 << 40):.1f} TiB)"
        )
    else:
        _say(f"materializing parameters on CPU RAM (target dtype: {target_dtype})")

    def _alloc(shape, dtype: torch.dtype) -> torch.Tensor:
        if mmap_allocator is not None:
            return mmap_allocator.alloc(shape, dtype)
        return torch.empty(shape, dtype=dtype, device="cpu")

    # Walk the meta-allocated parameters and replace each with a CPU tensor at
    # the right dtype: BF16 (target_dtype) for plain Linear/Embedding weights,
    # FP32 for anything flagged ``_keep_fp32 = True``. This avoids the
    # ~1.1 TB FP32 peak that ``model().to(bf16)`` would incur on Flash dims.
    for module in model.modules():
        for name, p in list(module._parameters.items()):
            if p is None or p.device.type != "meta":
                continue
            keep_fp32 = getattr(p, "_keep_fp32", False)
            dtype = torch.float32 if keep_fp32 else target_dtype
            new_p = torch.nn.Parameter(
                _alloc(p.shape, dtype),
                requires_grad=p.requires_grad,
            )
            if keep_fp32:
                new_p._keep_fp32 = True
            module._parameters[name] = new_p
        for buf_name, b in list(module._buffers.items()):
            if b is None or b.device.type != "meta":
                continue
            module._buffers[buf_name] = _alloc(b.shape, b.dtype)
    n_params = sum(p.numel() for p in model.parameters())
    _say(f"model has {n_params / 1e9:.2f}B params (target dtype: {target_dtype})")
    if mmap_allocator is not None:
        _say(
            f"  mmap allocator used {mmap_allocator.used_bytes / (1 << 40):.2f} TiB "
            f"of {mmap_allocator.capacity_bytes / (1 << 40):.1f} TiB capacity"
        )

    _say("streaming HF safetensors -> model (one shard at a time)")
    t_xform_start = time.time()
    summary = stream_load_hf_directory_into_model(
        model,
        args.hf_snapshot,
        strict=args.strict,
        dequantize_fp8=not args.skip_fp8_dequant,
        target_dtype=target_dtype,
    )
    _say(
        f"  streaming load done in {time.time() - t_xform_start:.1f}s: "
        f"loaded={summary.loaded}, fp8_dequantized={summary.fp8_dequantized}, "
        f"ape_unhotfixed={summary.ape_unhotfixed}, experts_fused={summary.experts_fused}, "
        f"skipped_mtp={summary.skipped_mtp}, "
        f"unmapped={len(summary.unmapped)}, missing_in_model={len(summary.missing_in_model)}"
    )
    if summary.unmapped:
        _say(f"  {len(summary.unmapped)} HF tensors had no mapping; first 5: {summary.unmapped[:5]}")
    if summary.missing_in_model:
        _say(
            f"  {len(summary.missing_in_model)} mapped names not present in the xorl model; "
            f"first 5: {summary.missing_in_model[:5]}"
        )

    _say(f"writing DCP to {args.dcp_out}")
    t_save_start = time.time()
    _save_dcp(args.dcp_out, model)

    metadata = args.dcp_out / ".metadata"
    distcp_files = list(args.dcp_out.glob("*.distcp"))
    if not metadata.exists() or not distcp_files:
        raise RuntimeError(
            f"DCP write did not produce expected files in {args.dcp_out}: "
            f"metadata={metadata.exists()}, .distcp count={len(distcp_files)}"
        )
    total_gb = sum(p.stat().st_size for p in [metadata, *distcp_files]) / 1e9
    _say(f"  DCP write done in {time.time() - t_save_start:.1f}s ({total_gb:.1f} GB on disk)")

    # Best-effort: write the xorl-style ``checkpoint_metadata.json`` sidecar
    # so consumers that look for it (like the smoke-job pre-flight) find it.
    # ``_save_checkpoint_metadata`` is gated on ``dist.get_rank() == 0`` so
    # it's a no-op when the rank-0 PG is not initialized — we have one.
    try:
        from xorl.checkpoint.checkpointer import _save_checkpoint_metadata  # noqa: PLC0415

        _save_checkpoint_metadata(str(args.dcp_out), model, save_lora_only=False)
        _say("  wrote checkpoint_metadata.json sidecar")
    except Exception as e:
        _say(f"  (checkpoint_metadata.json sidecar write skipped: {e!s})")

    if mmap_allocator is not None:
        # Drop strong refs to model params/buffers so their mmap views release
        # before we close the allocator. This avoids "BufferError: cannot close
        # exported pointers exist" on the mmap.close().
        del model
        mmap_allocator.close()
        _say(f"  mmap staging file unlinked: {args.disk_backed_staging}")
    _say(f"total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
