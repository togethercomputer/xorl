import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import MixedPrecision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import noop_context_fn

from ..models import all_ranks_load_weights, rank0_load_and_broadcast_weights
from ..utils import logging
from ..utils.device import get_device_type
from ..utils.import_utils import is_torch_version_greater_than
from .checkpoint import CheckpointFunction
from .parallel_state import get_parallel_state


if is_torch_version_greater_than("2.4"):
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module


logger = logging.get_logger(__name__)


def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_mixed_precision: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
    """
    Applies EP (when enabled) + FSDP2 parallel strategy to the model.

    Flow:
    1. Apply EP: Expert tensors [128,H,I] -> [32,H,I] local tensors per EP rank
    2. Apply FSDP2 to expert modules: Shard expert tensors along dim-1 (hidden dim)
    3. Apply FSDP2 to regular modules: Standard dim-0 sharding
    4. Result: Expert params [32,H/fsdp_size,I], regular params use standard FSDP2
    """
    parallel_state = get_parallel_state()

    # Step 0: Get target classes to shard later
    target_classes = set((getattr(model, "_no_split_modules", []) or []) + (basic_modules or []))
    # Make a list of tuples that contains layer's name and module
    decoder_blocks: List[Tuple[str, nn.Module]] = [
        (fqn, mod) for fqn, mod in model.named_modules() if mod.__class__.__name__ in target_classes
    ]
    logger.info_rank0(f"target classes to shard: {target_classes}")

    # Step 1: Apply expert parallelism (slice expert tensors [128,H,I] -> [16,H,I])
    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        assert parallel_plan is not None, (
            "Expert parallelism needs parallel plan defined in the model! Please see xorl/models/transformers/qwen3_moe/parallel_plan.py for example."
        )
        ep_fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
        # Attach spec mapping for checkpoint load-time reconstruction
        setattr(model, "_fqn2spec_info", ep_fqn2spec_info)
        # ep_mesh does not really exist in EP parameters' device mesh.
        # EP parameters are loaded as local tensors to be later sharded by fully_shard
        ep_mesh = parallel_state.ep_fsdp_device_mesh["ep"]
        # experts_map is a dict {experts_fqn: experts_mod}
        # For example, Qwen3MoE keys: model.layers.N.mlp.experts
        experts_map = parallel_plan.get_fsdp_no_shard_info(model)

        logger.info_rank0(f"Applied EP: expert tensors sliced along expert dimension (EP mesh: {ep_mesh})")
        logger.info_rank0(f"Experts Map: {experts_map}")
    else:
        ep_fqn2spec_info = None
        ep_mesh = None
        experts_map = None

    # Extract experts module from the layer if any, then pair them
    layer_pairs = []
    for layer_fqn, layer_mod in decoder_blocks:
        if experts_map is not None:
            # extract experts module from the layer
            experts_mod = next(
                (exp_mod for exp_fqn, exp_mod in experts_map.items() if exp_fqn.startswith(layer_fqn + ".")),
                None,
            )
            layer_pairs.append((layer_fqn, layer_mod, experts_mod))
        else:
            # No experts module found in this layer
            # this is often the case for models like deepseek in which some decoder layers are dense instead of MoE
            layer_pairs.append((layer_fqn, layer_mod, None))

    logger.info_rank0(f"layer pairs: {layer_pairs}")

    # Step 2: Update fsdp2 kwargs
    fsdp_kwargs = {"mesh": parallel_state.fsdp_mesh}
    # mp_policy kwargs
    if enable_mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs["mp_policy"] = mp_policy

    if hasattr(model, "get_ignore_modules_in_mixed_precision"):
        modules_to_ignore_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()
    else:
        modules_to_ignore_in_mixed_precision = None

    if modules_to_ignore_in_mixed_precision:
        assert isinstance(modules_to_ignore_in_mixed_precision, tuple), (
            "modules_to_ignore_in_mixed_precision needs to be a tuple!"
        )
        mp_ignored_classes = modules_to_ignore_in_mixed_precision
        fsdp_kwargs_without_mp = dict(fsdp_kwargs)
        fsdp_kwargs_without_mp.pop("mp_policy", None)
    else:
        mp_ignored_classes = None
        fsdp_kwargs_without_mp = fsdp_kwargs

    # prepare ep_fsdp2 kwargs
    if parallel_state.ep_enabled:
        # Use the ep_fsdp dimension as DP mesh for experts (shards orthogonal to EP)
        ep_fsdp_mesh = parallel_state.ep_fsdp_device_mesh["ep_fsdp"]
        expert_fsdp_kwargs = dict(fsdp_kwargs)
        expert_fsdp_kwargs["mesh"] = ep_fsdp_mesh

        # Prefer dim-1 sharding for expert weights when composing with EP shard on dim-0
        def _experts_shard_placement_fn(param):
            return Shard(1)

        expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn

    # Here we have a basic assumption for the decoder layer hierarchy
    # Decoder Layer
    # | -- layers that are sharded by fully_shard(decode_layer) (e.g., Attention)
    # | -- experts layer (apply fully_shard separately in order to shard across EP groups on the same EP rank instead of sharding globally)
    # | -- layers (declared in model.modules_to_ignore_in_mixed_precision) that need to apply fully_shard separately due to different mp policy as the decoder layer
    #      (e.g., some models requires MoE TopK gate layer to have parameters in higher FP32 precision in forward).
    for layer_fqn, layer_mod, experts_mod in layer_pairs:
        # register all the FSDPModule inside this decoder layer for the convenience of manual prefetching configuration
        layer_mod._fsdp_modules = []
        # ep enabled and this layer contains the expert module
        if parallel_state.ep_enabled and experts_mod is not None:
            # shard expert
            fully_shard(experts_mod, **expert_fsdp_kwargs)
            if hasattr(experts_mod, "set_gradient_divide_factor"):
                # average EP grads across EP ranks
                experts_mod.set_gradient_divide_factor(parallel_state.ep_size)
            layer_mod._fsdp_modules.append(experts_mod)
        # shard module that needs to ignore mixed precision control
        if mp_ignored_classes:
            for sub_mod in layer_mod.modules():
                if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                    # this will also create a AllGather communication group
                    # when modules here are small (like gating), this would slightly impacts the peformance
                    # a better method might be adding them to ignored_params of fully_shard
                    # but then they will need to be initialized separately
                    fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                    layer_mod._fsdp_modules.append(sub_mod)

        # shard everything else in the decoder layer
        fully_shard(layer_mod, **fsdp_kwargs)
        layer_mod._fsdp_modules.append(layer_mod)
        logger.info_rank0(f"{layer_fqn=}, {layer_mod._fsdp_modules=}")
    # shard root model
    fully_shard(model, **fsdp_kwargs)

    # configure manual prefetching when needed
    need_manual_prefetch = parallel_state.ep_enabled or mp_ignored_classes is not None
    if need_manual_prefetch:
        blocks = [pair[1] for pair in layer_pairs]
        next_blocks = blocks[1:] + [None]
        for current_block, next_block in zip(blocks, next_blocks):
            if next_block is not None:
                prefetch_modules = next_block._fsdp_modules
                # prefetch in order of attn, gate, experts
                current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

        # configure backward prefetch
        rev_blocks = list(reversed(blocks))
        prev_blocks = rev_blocks[1:] + [None]
        for current_block, prev_block in zip(rev_blocks, prev_blocks):
            if prev_block is not None:
                prefetch_modules = prev_block._fsdp_modules
                current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))

    # Handle meta initialization for FSDP2 (fallback if pre-load not done)
    assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"

    # skip_weight_loading: Used when caller will handle weight loading separately
    # (e.g., FSDP2+LoRA where we broadcast from rank 0 after this function returns)
    if kwargs.get("skip_weight_loading"):
        logger.info_rank0("Skipping weight loading in parallelize_model_fsdp2 (caller will handle)")
    elif weights_path is None:
        model.to_empty(device="cuda")
        model.init_weights()
    else:
        from torch.distributed.tensor import distribute_tensor

        logger.info_rank0("starting to load model weights...")
        load_weights_mode = kwargs.get("load_weights_mode", "broadcast")
        if load_weights_mode == "broadcast":
            logger.info_rank0("Loading model weights from disk on rank0 then broadcasting to other ranks...")
            rank0_load_and_broadcast_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)
        else:
            logger.info_rank0("Every rank reading weights from disk independently...")
            all_ranks_load_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)

    # Register grad norm clipping method for FSDP2
    from .fsdp2 import clip_grad_norm as clip_grad_norm_fn

    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_fn, model)

    return model


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    sharding_plan: Optional[Dict[str, Any]] = None,
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.
    """
    parallel_state = get_parallel_state()

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") not in ["cuda", "npu"]:
            raise ValueError("Only FSDP training supports `init_device=cpu` or `init_device=meta`.")
        if kwargs.pop("enable_fsdp_offload", False):
            raise ValueError("Only FSDP training supports `enable_fsdp_offload`.")
    if enable_mixed_precision:
        model = model.float()


    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": use_reentrant,
                "context_fn": kwargs.pop("recompute_context_fn", noop_context_fn),
            },
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            model = parallelize_model_fsdp2(
                model=model,
                weights_path=weights_path,
                enable_full_shard=enable_full_shard,
                enable_mixed_precision=enable_mixed_precision,
                basic_modules=basic_modules,
                **kwargs,
            )
        elif parallel_state.dp_mode == "ddp":
            ddp_kwargs = {"device_ids": [parallel_state.local_rank]}
            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
                ddp_kwargs["mixed_precision"] = mixed_precision
            model = DDP(model, **ddp_kwargs)
        else:
            # dp_mode == "none": no parallelization, just use model directly
            logger.info_rank0("No data parallelism (dp_mode=none), using model directly.")

    return model
