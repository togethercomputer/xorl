"""
QLoRA utility functions.

Functions for injecting QLoRA into models and managing quantized checkpoints.
"""

import json
import logging
import os
from typing import Collection, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.nvfp4_linear import NvFP4QLoRALinear
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.nf4_linear import NF4QLoRALinear
from xorl.qlora.modules.moe_experts import QLoRAMoeExperts

logger = logging.getLogger(__name__)

# Merge source mapping for fused modules in pre-quantized checkpoints.
# When the model has a fused module (e.g., qkv_proj) but the checkpoint has
# separate projections (q_proj, k_proj, v_proj), this maps fused → source names.
_PREQUANT_MERGE_SOURCES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _get_submodule(model: nn.Module, target: str) -> Tuple[nn.Module, str]:
    """Get parent module and attribute name for a dot-separated target path."""
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_linear_modules(
    model: nn.Module,
    target_modules: List[str],
    exclude_types: tuple = (),
) -> List[str]:
    """Find all nn.Linear module paths matching target names, excluding certain types."""
    matched_paths = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if exclude_types and isinstance(module, exclude_types):
            continue
        module_name = name.split(".")[-1] if name else ""
        if module_name in target_modules:
            matched_paths.append(name)
    return matched_paths


def inject_qlora_into_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    quant_format: str = "nvfp4",
    quant_group_size: int = 16,
    target_modules: Optional[List[str]] = None,
    checkpoint_quant_format: Optional[str] = None,
    merge_qkv: bool = True,
    exclude_modules: Optional[Collection[str]] = None,
    enable_aqn: bool = False,
    aqn_alpha: float = 1.0,
) -> nn.Module:
    """
    Inject QLoRA into a model: replace linear modules with QLoRALinear (pre-quantized).

    Replaces nn.Linear with QLoRALinear which stores weights in quantized
    format (~4x weight memory reduction for nvfp4, ~2x for block_fp8).
    Only LoRA params are trainable.

    Requires a pre-quantized checkpoint. QLoRALinear modules are created without
    weight parameters — quantized data is loaded later via ``maybe_load_prequantized_qlora()``.

    Args:
        model: Model to inject QLoRA into
        r: LoRA rank
        lora_alpha: LoRA alpha for scaling
        quant_format: Quantization format: "nvfp4" or "block_fp8"
        quant_group_size: Block size for quantization (16 for nvfp4, 128 for block_fp8)
        target_modules: Module names to target. Default: fused projections (qkv_proj, gate_up_proj, ...).
        checkpoint_quant_format: Format of pre-quantized checkpoint ("nvfp4" or "block_fp8").
            Must match ``quant_format`` (cross-format conversion is not supported).
        merge_qkv: If True (default), merge q/k/v into fused qkv_proj. If False,
            each projection gets its own QLoRALinear with independent LoRA parameters.
            Requires the model's QKV projections to be unfused before calling this.
        enable_aqn: If True, add Adaptive Quantization Noise during training forward passes.
        aqn_alpha: Scale factor for AQN noise magnitude (default 1.0).
    """
    if quant_format not in ("nvfp4", "block_fp8", "nf4"):
        raise ValueError(
            f"Supported QLoRA formats: 'nvfp4', 'block_fp8', 'nf4'. Got quant_format={quant_format!r}"
        )
    if target_modules is None:
        # Pre-quantized checkpoints: target the fused module names in the model.
        # Separate HF projections (q/k/v, gate/up) are merged during loading.
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    # Ensure quant_group_size matches target format
    if quant_format == "block_fp8":
        quant_group_size = 128
    elif quant_format == "nvfp4":
        quant_group_size = 16
    elif quant_format == "nf4":
        quant_group_size = 64

    target_paths = _find_linear_modules(model, target_modules, exclude_types=(QLoRALinear,))

    # Always exclude lm_head and MoE router (gate) from QLoRA — quantizing these
    # critical modules degrades quality with minimal memory savings.
    _default_exclude = {"lm_head", "gate"}
    if exclude_modules:
        exclude_modules = _default_exclude | set(exclude_modules)
    else:
        exclude_modules = _default_exclude

    # Filter out modules excluded from quantization.
    # exclude_modules contains short names (e.g., "lm_head", "gate") — match
    # against the last component of each FQN.
    if exclude_modules:
        all_short_names = {p.split(".")[-1] for p in target_paths}
        before = len(target_paths)
        target_paths = [
            p for p in target_paths
            if p.split(".")[-1] not in exclude_modules
        ]
        excluded = before - len(target_paths)
        if excluded > 0:
            logger.info(
                f"Excluded {excluded} modules from QLoRA injection "
                f"(not quantized in checkpoint): {exclude_modules}"
            )
        # Warn about exclude_modules names that didn't match any target module
        unmatched = exclude_modules - all_short_names
        if unmatched:
            logger.warning(
                f"exclude_modules contains names not matching any target module: {unmatched}. "
                f"Available target short names: {all_short_names}"
            )

    # Check if MoE experts may be handled later (gate_proj/up_proj/down_proj targets)
    _moe_proj_names = {"gate_proj", "up_proj", "down_proj"}
    _has_moe_targets = bool(_moe_proj_names & set(target_modules))

    if not target_paths and not _has_moe_targets:
        raise ValueError(
            f"No nn.Linear modules found matching target_modules={target_modules}. "
            f"Available: {[n.split('.')[-1] for n, _ in model.named_modules() if n][:20]}..."
        )

    if target_paths:
        logger.info(
            f"Injecting QLoRA into {len(target_paths)} modules "
            f"(r={r}, alpha={lora_alpha}, {quant_format}, group_size={quant_group_size})"
        )

    for target_path in target_paths:
        parent, attr_name = _get_submodule(model, target_path)
        original_module = getattr(parent, attr_name)

        if quant_format == "nf4":
            # NF4: quantize bf16 weights on-the-fly (no pre-quantized checkpoint)
            qlora_module = NF4QLoRALinear.from_module(
                original_module, r=r, lora_alpha=lora_alpha,
                enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
            )
        else:
            # nvfp4/block_fp8: create empty shell, load pre-quantized weights later
            if quant_format == "block_fp8":
                qlora_cls = BlockFP8QLoRALinear
            else:
                qlora_cls = NvFP4QLoRALinear
            qlora_module = qlora_cls(
                in_features=original_module.in_features,
                out_features=original_module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                bias=original_module.bias is not None,
                device=original_module.weight.device,
                enable_aqn=enable_aqn,
                aqn_alpha=aqn_alpha,
            )
            qlora_module._is_prequantized = True
            qlora_module._source_quant_format = checkpoint_quant_format

            # Determine merge sources and HF source FQN
            merge_sources = _PREQUANT_MERGE_SOURCES.get(attr_name)
            qlora_module._merge_sources = merge_sources
            if merge_sources is not None:
                # Fused module (qkv_proj, gate_up_proj): source FQN is parent path
                qlora_module._source_fqn = target_path.rsplit(".", 1)[0]
            else:
                # Direct module (o_proj, down_proj): source FQN is the full path
                qlora_module._source_fqn = target_path

            # Tell checkpoint loading to silently skip weight key
            qlora_module._qlora_expected_skip_keys = {"weight"}

        setattr(parent, attr_name, qlora_module)

    logger.info(f"Replaced {len(target_paths)} modules with QLoRALinear")

    # Also inject QLoRA into MoE experts (replace module.experts, keep original MoeBlock)
    if not any(t in ("gate_proj", "up_proj", "down_proj") for t in target_modules):
        return model

    from xorl.distributed.parallel_state import get_parallel_state
    try:
        parallel_state = get_parallel_state()
        ep_size = parallel_state.ep_size if parallel_state.ep_enabled else 1
        ep_rank = parallel_state.ep_rank if parallel_state.ep_enabled else 0
    except Exception:
        ep_size = 1
        ep_rank = 0

    moe_count = 0
    for name, module in list(model.named_modules()):
        # Match MoE blocks: modules with .gate (nn.Linear) and .experts with 3D weights
        if not (hasattr(module, "gate") and hasattr(module, "experts")):
            continue
        experts = module.experts
        has_3d_weights = (
            hasattr(experts, "gate_proj") and isinstance(experts.gate_proj, nn.Parameter)
            and hasattr(experts, "up_proj") and isinstance(experts.up_proj, nn.Parameter)
            and hasattr(experts, "down_proj") and isinstance(experts.down_proj, nn.Parameter)
            and experts.gate_proj.dim() == 3
            and not isinstance(experts, QLoRAMoeExperts)
        )
        if not has_3d_weights:
            continue

        num_experts = experts.gate_proj.shape[0]
        num_local_experts = num_experts // ep_size
        expert_offset = ep_rank * num_local_experts

        qlora_experts = QLoRAMoeExperts.from_module(
            experts,
            r=r,
            lora_alpha=lora_alpha,
            quant_format=quant_format,
            quant_group_size=quant_group_size,
            num_local_experts=num_local_experts,
            expert_offset=expert_offset,
            hybrid_shared=True,
        )

        experts_fqn = name + ".experts" if not name.endswith(".experts") else name

        if quant_format == "nf4":
            is_meta = experts.gate_proj.device.type == "meta"
            if is_meta:
                # Defer: load bf16 from checkpoint after FSDP, then quantize
                qlora_experts._source_fqn = experts_fqn
            else:
                # Quantize bf16 expert weights immediately
                for proj_name, src_param in [
                    ("gate", experts.gate_proj),
                    ("up", experts.up_proj),
                    ("down", experts.down_proj),
                ]:
                    local_w = src_param[expert_offset:expert_offset + num_local_experts]
                    qlora_experts._quantize_proj(proj_name, local_w.float())
                qlora_experts._weights_loaded = True
        else:
            # nvfp4/block_fp8: source FQN and checkpoint format for deferred loading
            qlora_experts._source_fqn = experts_fqn
            qlora_experts._source_quant_format = checkpoint_quant_format

        # Replace only the experts module, keep the original MoeBlock
        module.experts = qlora_experts
        moe_count += 1

    if moe_count > 0:
        logger.info(
            f"Replaced {moe_count} MoE experts with QLoRAMoeExperts "
            f"(ep_size={ep_size}, num_local_experts={num_local_experts})"
        )

    return model


def save_qlora_checkpoint(
    model: nn.Module,
    save_path: str,
) -> str:
    """
    Save QLoRA checkpoint: quantized base weights + LoRA parameters.

    Args:
        model: Model with QLoRALinear layers
        save_path: Directory to save checkpoint
    """
    os.makedirs(save_path, exist_ok=True)

    quantized_state = {}
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, QLoRALinear):
            # Quantized base weights
            q_data = module.get_quantized_state_dict()
            for key, val in q_data.items():
                quantized_state[f"{name}.{key}"] = val.cpu().clone()
            # LoRA weights
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
            # Bias
            if module.bias is not None:
                lora_state[f"{name}.bias"] = module.bias.detach().cpu().clone()

    if quantized_state:
        save_file(quantized_state, os.path.join(save_path, "quantized_weights.safetensors"))
    if lora_state:
        save_file(lora_state, os.path.join(save_path, "lora_weights.safetensors"))

    logger.info(f"Saved QLoRA checkpoint to {save_path}")
    return save_path


def maybe_quantize_qlora(model: nn.Module) -> int:
    """Quantize bf16 ``weight`` parameters in QLoRALinear modules to packed form.

    After FSDP loads bf16 weights from checkpoint, this converts them to the
    target quantization format (NF4, etc.) and frees the bf16 storage.

    FSDP2 tracking of the ``weight`` parameter is removed *before* quantization
    so that ``reset_sharded_param`` does not crash on ``None``.

    Returns:
        Number of modules quantized.
    """
    modules_to_quantize = [
        m for m in model.modules()
        if isinstance(m, QLoRALinear) and m.weight is not None
    ]
    if not modules_to_quantize:
        return 0

    # Deregister ``weight`` from FSDP2 param groups BEFORE setting weight=None.
    # Without this, FSDP's lazy_init → reset_sharded_param tries to access
    # weight._local_tensor on the now-None parameter and crashes.
    removed = _deregister_qlora_weights_from_fsdp(model, param_names=("weight",))
    if removed > 0:
        logger.info(f"Deregistered {removed} weight params from FSDP2 before quantization")

    count = 0
    for module in modules_to_quantize:
        module.quantize_weight()
        count += 1
    logger.info(f"Quantized {count} QLoRALinear bf16 → packed (deferred)")
    return count


def maybe_load_and_quantize_moe_qlora(
    model: nn.Module,
    weights_path: str,
    load_mode: str = "broadcast",
) -> int:
    """Load bf16 MoE expert weights from checkpoint and quantize.

    For formats like NF4 that don't use pre-quantized checkpoints, this loads
    the bf16 expert weights and quantizes them on-the-fly.

    Args:
        model: Model with QLoRAMoeExperts modules.
        weights_path: Path to HF model directory with bf16 weights.
        load_mode: "broadcast" or "all_ranks".

    Returns:
        Number of MoE modules loaded.
    """
    import json
    import torch.distributed as dist

    needs_moe = any(
        isinstance(m, QLoRAMoeExperts) and not m._weights_loaded
        for m in model.modules()
    )
    if not needs_moe:
        return 0

    # Load weight map from index file
    weight_map = None
    if weights_path:
        try:
            from transformers.utils import cached_file, SAFE_WEIGHTS_INDEX_NAME
            index_path = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME)
            if index_path:
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
        except Exception:
            pass

    if weight_map is None:
        raise RuntimeError(
            f"Could not load weight index from {weights_path}. "
            "MoE expert loading requires model.safetensors.index.json."
        )

    shard_cache: dict = {}

    moe_count = 0
    for module in model.modules():
        if isinstance(module, QLoRAMoeExperts) and not module._weights_loaded:
            module.load_and_quantize_weights(
                weights_path, weight_map=weight_map, shard_cache=shard_cache,
            )
            # Materialize LoRA params from meta device to GPU
            from torch.distributed._tensor import DTensor
            for name, param in module.named_parameters():
                if param.device.type == "meta":
                    if isinstance(param, DTensor):
                        local_shape = param.to_local().shape
                        placement = param.placements
                        mesh = param.device_mesh
                        local_data = torch.zeros(
                            local_shape, dtype=param.dtype, device="cuda",
                        )
                        materialized = nn.Parameter(
                            DTensor.from_local(
                                local_data, mesh, placement, run_check=False,
                            ),
                            requires_grad=param.requires_grad,
                        )
                    else:
                        import math as _math
                        materialized = nn.Parameter(
                            torch.zeros(
                                param.shape, dtype=param.dtype, device="cuda",
                            ),
                            requires_grad=param.requires_grad,
                        )
                        parts = name.split("_")
                        if parts[-1] == "A":
                            for i in range(materialized.shape[0]):
                                nn.init.kaiming_uniform_(
                                    materialized.data[i], a=_math.sqrt(5),
                                )
                    setattr(module, name, materialized)
            moe_count += 1

    shard_cache.clear()

    if moe_count > 0:
        logger.info(
            f"Loaded and quantized {moe_count} MoE expert modules from bf16 checkpoint"
        )
    return moe_count


def _deregister_qlora_weights_from_fsdp(
    model: nn.Module,
    param_names: tuple = ("weight",),
) -> int:
    """Remove QLoRALinear parameters from all FSDP2 param groups.

    After quantization, the ``weight`` parameter storage is freed — we must remove
    it from FSDP2's tracking to prevent all-gather on empty storage.

    ``packed_weight_f32`` must ALSO be deregistered because FSDP2's mixed-precision
    policy casts it from float32 to ``param_dtype`` (e.g. bfloat16) during forward.
    This lossy cast corrupts the packed uint8 byte patterns.  After deregistration
    the parameter stays as a sharded DTensor; forward code calls ``full_tensor()``
    to all-gather in the original float32 dtype without the mixed-precision cast.

    DCP checkpoint save/load still works because the parameter remains a DTensor
    with mesh placement metadata — DCP reads that directly, not FSDP param groups.

    Args:
        model: Model containing QLoRALinear modules.
        param_names: Parameter names to deregister (default: just ``"weight"``).
    """
    param_name_set = set(param_names)

    # Collect all QLoRALinear modules for matching
    qlora_modules = set()
    for module in model.modules():
        if isinstance(module, QLoRALinear):
            qlora_modules.add(id(module))

    removed = 0
    for module in model.modules():
        state_fn = getattr(module, "_get_fsdp_state", None)
        if state_fn is None:
            continue
        try:
            fsdp_state = state_fn()
            pg = fsdp_state._fsdp_param_group
            if pg is None:
                continue
            to_remove = [
                fp for fp in pg.fsdp_params
                if hasattr(fp, "_module_info")
                and fp._module_info.param_name in param_name_set
                and id(fp._module_info.module) in qlora_modules
            ]
            for fp in to_remove:
                pg.fsdp_params.remove(fp)
                removed += 1
        except Exception:
            pass
    return removed



def detect_prequantized_nvfp4(weights_path: str) -> bool:
    """Detect whether a checkpoint contains pre-quantized NVFP4 weights (modelopt format).

    Delegates to :func:`~xorl.models.checkpoint_handlers.buffers.detect_prequantized_checkpoint`.

    Args:
        weights_path: Path to HF model directory

    Returns:
        True if the checkpoint is pre-quantized NVFP4.
    """
    from xorl.models.checkpoint_handlers.buffers import detect_prequantized_checkpoint
    return detect_prequantized_checkpoint(weights_path)


def detect_prequantized_block_fp8(weights_path: str) -> bool:
    """Detect whether a checkpoint contains pre-quantized block FP8 weights.

    Delegates to :func:`~xorl.models.checkpoint_handlers.buffers.detect_prequantized_block_fp8_checkpoint`.

    Args:
        weights_path: Path to HF model directory

    Returns:
        True if the checkpoint is pre-quantized block FP8.
    """
    from xorl.models.checkpoint_handlers.buffers import detect_prequantized_block_fp8_checkpoint
    return detect_prequantized_block_fp8_checkpoint(weights_path)


def _broadcast_shard_cache(
    needed_shards: List[str],
    weights_path: str,
) -> dict:
    """Rank 0 reads safetensors shard files and broadcasts tensors to all ranks.

    Uses NCCL broadcast via GPU for speed (~50 GB/s NVLink), with background
    prefetching of the next shard file from disk while the current one is
    being broadcast (overlaps disk I/O with network).

    Returns:
        shard_cache: dict mapping shard filename → {tensor_name: tensor (CPU)}
    """
    import torch.distributed as dist
    from concurrent.futures import ThreadPoolExecutor
    from transformers.utils import cached_file
    import safetensors.torch

    rank = dist.get_rank()
    device = torch.device("cuda")
    shard_cache = {}

    def _read_shard(shard_file):
        shard_path = cached_file(weights_path, shard_file)
        return safetensors.torch.load_file(shard_path, device="cpu")

    # Background thread for prefetching next shard from disk on rank 0
    executor = ThreadPoolExecutor(max_workers=1) if rank == 0 else None
    prefetch_future = None

    for i, shard_file in enumerate(needed_shards):
        # Rank 0: get current shard (from prefetch or direct read)
        if rank == 0:
            if prefetch_future is not None:
                shard_data = prefetch_future.result()
            else:
                shard_data = _read_shard(shard_file)
            # Prefetch next shard while broadcasting current one
            if i + 1 < len(needed_shards):
                prefetch_future = executor.submit(_read_shard, needed_shards[i + 1])
            else:
                prefetch_future = None
            meta = [(name, list(t.shape), str(t.dtype)) for name, t in shard_data.items()]
        else:
            shard_data = None
            meta = None

        # Broadcast metadata (one call per shard — list of (name, shape, dtype))
        meta = [meta]
        dist.broadcast_object_list(meta, src=0)
        meta = meta[0]

        # Broadcast each tensor via GPU (NCCL), then move to CPU
        shard_dict = {}
        for name, shape, dtype_str in meta:
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            if rank == 0:
                cpu_tensor = shard_data[name].contiguous()
                gpu_tensor = cpu_tensor.pin_memory().to(device, non_blocking=True)
            else:
                gpu_tensor = torch.empty(shape, dtype=dtype, device=device)

            dist.broadcast(gpu_tensor, src=0)
            shard_dict[name] = gpu_tensor.cpu()
            del gpu_tensor

        shard_cache[shard_file] = shard_dict
        if rank == 0:
            del shard_data

        # Free GPU memory between shards
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(
                f"Broadcast shard {i + 1}/{len(needed_shards)}: {shard_file} "
                f"({len(shard_dict)} tensors)"
            )

    if executor is not None:
        executor.shutdown(wait=False)

    return shard_cache


def maybe_load_prequantized_qlora(model: nn.Module, weights_path: str, load_mode: str = "broadcast") -> int:
    """Load pre-quantized weights into QLoRALinear modules from checkpoint.

    For pre-quantized NVFP4 checkpoints (modelopt format), this replaces the
    deferred quantization step. Loads packed weights, block scales, and global
    scales directly into QLoRALinear buffers.

    Also handles QLoRAMoeExperts (auto-detected internally via weight_map probing).

    When distributed is initialized, uses rank 0 broadcast to avoid redundant
    disk reads across ranks (~6-8× faster for large models).

    Args:
        model: Model with QLoRALinear/QLoRAMoeExperts modules
        weights_path: Path to HF model directory

    Returns:
        Number of modules loaded.
    """
    import torch.distributed as dist

    # Load weight map from index file
    weight_map = None
    try:
        from transformers.utils import cached_file, SAFE_WEIGHTS_INDEX_NAME
        index_path = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME)
        if index_path:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
    except Exception:
        pass

    if weight_map is None:
        raise RuntimeError(
            f"Could not load weight index from {weights_path}. "
            "Pre-quantized loading requires model.safetensors.index.json."
        )

    # Loading mode:
    # "broadcast" (default): rank 0 reads, broadcasts via NCCL. Best for shared/NFS filesystems.
    # "all_ranks": every rank reads from disk independently. Best for local SSDs.
    use_broadcast = (
        load_mode == "broadcast"
        and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if use_broadcast:
        needed_shards = sorted(set(weight_map.values()))
        logger.info(
            f"Broadcasting {len(needed_shards)} shard files from rank 0 "
            f"(rank={dist.get_rank()}, world={dist.get_world_size()})"
        )
        shard_cache = _broadcast_shard_cache(needed_shards, weights_path)
    else:
        shard_cache = {}

    # Load QLoRALinear modules (attention, dense MLP)
    # Skip modules already loaded inline via QLoRAWeightBuffer
    linear_count = 0
    for fqn, module in model.named_modules():
        if isinstance(module, QLoRALinear) and module._is_prequantized:
            if module._inline_loaded:
                continue  # loaded via checkpoint handler
            module.load_prequantized_weights(weight_map, shard_cache, weights_path)
            linear_count += 1

    # Load QLoRAMoeExperts (auto-detects prequant via weight_map probing)
    moe_count = 0
    for module in model.modules():
        if isinstance(module, QLoRAMoeExperts) and not module._weights_loaded:
            module.load_and_quantize_weights(weights_path, weight_map=weight_map, shard_cache=shard_cache)
            # Materialize LoRA params from meta device to GPU.
            # With EP, params are DTensors (Shard/Replicate) — must preserve placement.
            # For Replicate DTensors, all ranks must have identical values.
            # Since lora_B=0 → initial delta = A @ 0 = 0 regardless of A,
            # we initialize ALL DTensor params to zeros and let kaiming_uniform
            # only apply to non-DTensor (non-EP) params.
            from torch.distributed._tensor import DTensor
            _rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            for name, param in module.named_parameters():
                is_meta = param.device.type == "meta"
                is_dtensor = isinstance(param, DTensor)
                if _rank == 0 and moe_count == 0:
                    logger.info(
                        f"[QLoRA-EP] {name}: is_meta={is_meta}, is_dtensor={is_dtensor}, "
                        f"shape={list(param.shape)}, device={param.device}, "
                        f"type={type(param).__name__}"
                    )
                if is_meta:
                    if is_dtensor:
                        local_shape = param.to_local().shape
                        placement = param.placements
                        mesh = param.device_mesh
                        # DTensor: initialize to zeros (preserves Replicate consistency)
                        local_data = torch.zeros(
                            local_shape, dtype=param.dtype, device="cuda",
                        )
                        materialized = nn.Parameter(
                            DTensor.from_local(local_data, mesh, placement,
                                               run_check=False),
                            requires_grad=param.requires_grad,
                        )
                    else:
                        # Non-EP: standard initialization
                        materialized = nn.Parameter(
                            torch.zeros(param.shape, dtype=param.dtype, device="cuda"),
                            requires_grad=param.requires_grad,
                        )
                        parts = name.split("_")
                        if parts[-1] == "A":
                            import math
                            for i in range(materialized.shape[0]):
                                nn.init.kaiming_uniform_(materialized.data[i], a=math.sqrt(5))
                    setattr(module, name, materialized)
            moe_count += 1

    # Free shard cache
    shard_cache.clear()

    total = linear_count + moe_count
    if total > 0:
        # Deregister packed_weight_f32 from FSDP2 to prevent mixed-precision
        # bf16 cast that corrupts packed uint8 byte patterns.
        removed = _deregister_qlora_weights_from_fsdp(
            model, param_names=("packed_weight_f32",),
        )
        torch.cuda.empty_cache()
        logger.info(
            f"Loaded pre-quantized weights: {linear_count} QLoRALinear + "
            f"{moe_count} QLoRAMoeExperts modules (direct from checkpoint), "
            f"deregistered {removed} packed_weight_f32 params from FSDP2"
        )

    # Diagnostic: check first QLoRALinear dequant.
    # IMPORTANT: _dequantize_weight() may trigger FSDP2 all-gather, which requires ALL ranks
    # to participate. Run on all ranks; only rank 0 logs the results.
    _diag_rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    _diag_fqn = None
    _diag_mod = None
    for fqn, mod in model.named_modules():
        if hasattr(mod, '_dequantize_weight') and hasattr(mod, 'weight_block_scales'):
            _diag_fqn = fqn
            _diag_mod = mod
            break
    if _diag_mod is not None:
        try:
            w = _diag_mod._dequantize_weight()
            if _diag_rank == 0:
                logger.info(
                    f"[DIAG] {_diag_fqn} ({type(_diag_mod).__name__}): "
                    f"shape={list(w.shape)}, mean={w.float().mean():.6f}, "
                    f"std={w.float().std():.6f}, "
                    f"nan={w.isnan().any().item()}, inf={w.isinf().any().item()}"
                )
        except Exception as e:
            if _diag_rank == 0:
                logger.warning(f"[DIAG] {_diag_fqn}: dequant FAILED: {e}")
    elif _diag_rank == 0:
        logger.warning("[DIAG] No QLoRALinear module found in model!")

    return total


def maybe_requant_qlora(model: nn.Module, ema_decay: float = 0.1) -> int:
    """
    Merge LoRA deltas and re-quantize all QLoRA modules unconditionally.

    Call this between optimizer.step() and the next forward pass.
    Under FSDP2, parameters must not be mutated during forward — this function
    performs the merge/re-quantize outside the forward pass.

    The training loop controls when this is called (every N steps).
    EMA-tracked amax is used to inform global_scale for better fp8 precision.

    Args:
        model: Model with QLoRALinear/QLoRAMoeExperts modules.
        ema_decay: EMA decay for amax tracking (0.1 = 10% new, 90% old).

    Returns:
        Number of modules that were re-quantized.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, QLoRALinear):
            module.merge_weights(ema_decay=ema_decay)
            count += 1
        elif isinstance(module, QLoRAMoeExperts):
            module.merge_weights(ema_decay=ema_decay)
            count += 1
    if count > 0:
        logger.info(f"Re-quantized {count} QLoRA modules (LoRA merged into base)")
    return count
