import functools
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.fsdp import MixedPrecision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import noop_context_fn

from xorl.distributed.checkpoint import CheckpointFunction
from xorl.distributed.fsdp2 import clip_grad_norm
from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.pipeline_parallel import (
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)
from xorl.lora import LoraLinear
from xorl.models import all_ranks_load_weights, rank0_load_and_broadcast_weights
from xorl.utils import logging
from xorl.utils.device import get_device_type
from xorl.utils.import_utils import is_torch_version_greater_than


if is_torch_version_greater_than("2.4"):
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )


logger = logging.get_logger(__name__)


_TP_STYLE_MAP = {
    "colwise": ColwiseParallel if is_torch_version_greater_than("2.4") else None,
    "rowwise": RowwiseParallel if is_torch_version_greater_than("2.4") else None,
}


def _build_tp_plan(model: "nn.Module") -> Dict[str, Any]:
    """Build a PyTorch TP plan dict from model's _tp_plan and config's base_model_tp_plan."""
    plan = {}

    # Get base model plan from config (e.g., "layers.*.self_attn.q_proj": "colwise")
    config = getattr(model, "config", None)
    if config is not None:
        base_plan = getattr(config, "base_model_tp_plan", None)
        if base_plan:
            # Prefix with the base model attribute name (e.g., "model.")
            base_model_prefix = ""
            for attr_name in ["model", "transformer", "encoder"]:
                if hasattr(model, attr_name):
                    base_model_prefix = attr_name + "."
                    break
            for fqn, style_str in base_plan.items():
                full_fqn = base_model_prefix + fqn
                plan[full_fqn] = _resolve_tp_style(style_str)

    # Get top-level plan from model class (e.g., "lm_head": "colwise")
    model_plan = getattr(model, "_tp_plan", None)
    if model_plan:
        for fqn, style_str in model_plan.items():
            plan[fqn] = _resolve_tp_style(style_str)

    return plan


def _resolve_tp_style(style_str: str):
    """Convert a string TP style to a PyTorch ParallelStyle object."""
    if style_str == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style_str == "embedding":
        # Embedding: shard weight on vocab dim, replicated input/output
        # Weight [vocab, hidden] → Shard(0) [vocab/tp, hidden] per rank
        # Lookup → partial results → all-reduce → replicated output
        return RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate())
    elif style_str in _TP_STYLE_MAP:
        return _TP_STYLE_MAP[style_str]()
    else:
        raise ValueError(f"Unknown TP style: {style_str}")


def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_mixed_precision: bool = True,
    basic_modules: Optional[List[str]] = None,
    pp_enabled: bool = False,
    reshard_after_forward: Optional[bool] = None,
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
    # When skip_weight_loading is set (TP pre-loaded weights with EP-local expert params),
    # expert params are already at local shapes — skip DTensor redistribute, just annotate.
    ep_already_local = bool(kwargs.get("skip_weight_loading"))
    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        assert parallel_plan is not None, (
            "Expert parallelism needs parallel plan defined in the model! Please see xorl/models/transformers/qwen3_moe/parallel_plan.py for example."
        )
        ep_fqn2spec_info = parallel_plan.apply(
            model, parallel_state.ep_fsdp_device_mesh, already_local=ep_already_local
        )
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

    logger.debug_rank0(f"layer pairs: {layer_pairs}")

    # Step 2: Update fsdp2 kwargs
    fsdp_kwargs = {"mesh": parallel_state.fsdp_mesh}
    # reshard_after_forward: None = auto (False for PP, True/default for non-PP)
    if reshard_after_forward is not None:
        fsdp_kwargs["reshard_after_forward"] = reshard_after_forward
    elif pp_enabled:
        fsdp_kwargs["reshard_after_forward"] = False
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

        # When ep_fsdp mesh size is 1, there's no all-gather for experts.
        # Always reshard (free bf16 compute copies) after forward to save memory,
        # since the only cost is a cheap dtype cast during backward recomputation.
        if ep_fsdp_mesh.size() == 1:
            expert_fsdp_kwargs.pop("reshard_after_forward", None)

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
        if parallel_state.ep_enabled and experts_mod is not None and not getattr(experts_mod, "_skip_fsdp", False):
            # shard expert
            fully_shard(experts_mod, **expert_fsdp_kwargs)
            if hasattr(experts_mod, "set_gradient_divide_factor"):
                # average EP grads across EP ranks
                experts_mod.set_gradient_divide_factor(parallel_state.ep_size)
                # mark so the global divide-factor reset below doesn't override this
                experts_mod._is_ep_fsdp = True
            layer_mod._fsdp_modules.append(experts_mod)
        # shard module that needs to ignore mixed precision control
        if mp_ignored_classes:
            for sub_mod in layer_mod.modules():
                if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                    # this will also create a AllGather communication group
                    # when modules here are small (like gating), this would slightly impacts the performance
                    # a better method might be adding them to ignored_params of fully_shard
                    # but then they will need to be initialized separately
                    fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                    layer_mod._fsdp_modules.append(sub_mod)

        # shard everything else in the decoder layer
        # If experts_mod has _skip_fsdp, exclude its params from the parent FSDP unit
        # (each EP rank has different local expert LoRA params — they should not be all-gathered globally)
        layer_fsdp_kwargs = dict(fsdp_kwargs)
        if parallel_state.ep_enabled and experts_mod is not None and getattr(experts_mod, "_skip_fsdp", False):
            layer_fsdp_kwargs["ignored_params"] = set(experts_mod.parameters())
        fully_shard(layer_mod, **layer_fsdp_kwargs)
        layer_mod._fsdp_modules.append(layer_mod)
        logger.debug_rank0(f"{layer_fqn=}, {layer_mod._fsdp_modules=}")
    # Torchtitan optimization: group norm + lm_head into a single FSDP unit
    # with reshard_after_forward=False. When norm.forward() runs inside
    # the base model, FSDP all-gathers both norm and lm_head weights.
    # They stay gathered so external compute_loss() can access lm_head.weight
    # without a redundant all-gather.
    if not pp_enabled and fsdp_kwargs.get("reshard_after_forward", True) is not False:
        base_model = getattr(model, "model", None)
        norm_mod = getattr(base_model, "norm", None) if base_model else None
        lm_head_mod = getattr(model, "lm_head", None)
        last_modules = [m for m in [norm_mod, lm_head_mod] if m is not None]
        if last_modules:
            last_fsdp_kwargs = dict(fsdp_kwargs)
            last_fsdp_kwargs["reshard_after_forward"] = False
            fully_shard(last_modules, **last_fsdp_kwargs)

    # shard root model
    # Collect all _skip_fsdp experts params so they're also ignored by the
    # root-level fully_shard (layer-level already ignores them above, but the
    # root FSDP would re-shard them otherwise).
    root_ignored_params: set = set()
    for _, _, experts_mod in layer_pairs:
        if experts_mod is not None and getattr(experts_mod, "_skip_fsdp", False):
            root_ignored_params.update(experts_mod.parameters())
    if root_ignored_params:
        root_fsdp_kwargs = dict(fsdp_kwargs)
        root_fsdp_kwargs["ignored_params"] = root_ignored_params
        fully_shard(model, **root_fsdp_kwargs)
    else:
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

    # Disable FSDP's automatic gradient averaging — we normalize gradients manually
    # (gradient_accumulate_loss for non-PP; explicit grad.mul_(1/gvt) for PP).
    # Skip EP-sharded expert modules: they manage their own divide factor (ep_size)
    # for averaging expert gradients across EP ranks.
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule

    for module in model.modules():
        if isinstance(module, FSDPModule) and not getattr(module, "_is_ep_fsdp", False):
            module.set_gradient_divide_factor(1.0)

    # Handle meta initialization for FSDP2 (fallback if pre-load not done)
    assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"

    weight_device = get_device_type()

    # skip_weight_loading: Used when caller will handle weight loading separately
    # (e.g., FSDP2+LoRA where we broadcast from rank 0 after this function returns)
    if kwargs.get("skip_weight_loading"):
        logger.info_rank0("Skipping weight loading in parallelize_model_fsdp2 (caller will handle)")
    elif weights_path is None:
        model.to_empty(device=weight_device)
        with torch.no_grad():
            model.init_weights()
            # Re-initialize LoRA params — init_weights() only handles nn.Linear,
            # and reset_lora_parameters() was a no-op on meta tensors during __init__
            for m in model.modules():
                if hasattr(m, "reset_lora_parameters"):
                    m.reset_lora_parameters()
    else:
        from torch.distributed.tensor import distribute_tensor

        logger.info_rank0("starting to load model weights...")
        load_weights_mode = kwargs.get("load_weights_mode", "broadcast")
        if load_weights_mode == "broadcast":
            logger.info_rank0("Loading model weights from disk on rank0 then broadcasting to other ranks...")
            rank0_load_and_broadcast_weights(model, weights_path, weight_device, dtensor_factory=distribute_tensor)
        else:
            logger.info_rank0("Every rank reading weights from disk independently...")
            all_ranks_load_weights(model, weights_path, weight_device, dtensor_factory=distribute_tensor)

    # Register grad norm clipping method for FSDP2
    model.clip_grad_norm_ = types.MethodType(clip_grad_norm, model)

    return model


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    sharding_plan: Optional[Dict[str, Any]] = None,
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    enable_compile: bool = False,
    basic_modules: Optional[List[str]] = None,
    pp_schedule: Optional[str] = None,
    reshard_after_forward: Optional[bool] = None,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.

    When PP is enabled (pp_schedule is not None), returns a dict with keys:
        - "stages": list of PipelineStage objects
        - "model_parts": list of pruned models (with FSDP applied)
        - "has_first_stage": True if this rank has the first PP stage
        - "has_last_stage": True if this rank has the last PP stage
    Otherwise returns the parallelized model directly.
    """
    parallel_state = get_parallel_state()

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") not in ["cuda", "npu"]:
            raise ValueError("Only FSDP training supports `init_device=cpu` or `init_device=meta`.")
    if enable_mixed_precision and not kwargs.pop("skip_param_upcast", False):
        model = model.float()

    if pp_schedule is not None:
        # ---- Pipeline Parallelism path ----
        ps = get_parallel_state()
        pp_mesh = ps.pp_mesh
        pp_degree = ps.pp_size
        device = torch.device(f"{ps.device_type}:{ps.local_rank}")

        # 1. Get PP config from model
        if hasattr(model, "get_pp_module_config"):
            pp_config = model.get_pp_module_config()
        else:
            raise ValueError(
                "Model must implement get_pp_module_config() for pipeline parallelism. "
                "See Qwen3ForCausalLM for an example."
            )

        # 2. Generate FQN assignment per stage
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_stages=pp_degree,
            num_layers=pp_config["num_layers"],
            input_fqns=pp_config.get("input_fqns"),
            layer_prefix=pp_config.get("layer_prefix", "layers"),
            output_fqns=pp_config.get("output_fqns"),
        )

        # 3. Split model into pipeline stages
        stages, model_parts = pipeline_module_split(
            model,
            pp_mesh=pp_mesh,
            device=device,
            module_names_per_stage=module_names_per_stage,
            always_keep_fqns=pp_config.get("always_keep_fqns"),
        )
        logger.info_rank0(f"Model split into {pp_degree} PP stages")

        # Extract gradient checkpointing kwargs before the loop
        use_reentrant = kwargs.pop("enable_reentrant", False)
        recompute_context_fn = kwargs.pop("recompute_context_fn", noop_context_fn)
        recompute_modules = kwargs.pop("recompute_modules", None)
        moe_checkpoint_method = kwargs.pop("moe_checkpoint_method", None)

        # 4. Apply parallelism to each model part
        for i, model_part in enumerate(model_parts):
            # Gradient checkpointing
            if enable_gradient_checkpointing and hasattr(model_part, "gradient_checkpointing_enable"):
                if i == 0:
                    logger.info_rank0("Enable gradient checkpointing.")
                if use_reentrant:
                    torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

                gc_kwargs = {"use_reentrant": use_reentrant}
                # Skip context_fn when torch.compile is enabled -- dynamo can't trace
                # through the SkipFunctionVariable (noop_context_fn). Without context_fn,
                # checkpoint uses the same default behavior and dynamo can trace natively.
                if not enable_compile:
                    gc_kwargs["context_fn"] = recompute_context_fn if i == 0 else noop_context_fn
                if recompute_modules is not None:
                    gc_kwargs["recompute_modules"] = recompute_modules
                if moe_checkpoint_method is not None:
                    gc_kwargs["moe_checkpoint_method"] = moe_checkpoint_method
                model_part.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

            # TP (if enabled)
            if ps.tp_enabled:
                # TP + LoRA is not currently supported
                if i == 0:
                    if any(isinstance(m, LoraLinear) for m in model_part.modules()):
                        raise NotImplementedError("Tensor parallelism + LoRA is not currently supported.")

                # Unfuse fused projections (qkv_proj, gate_up_proj) for TP compatibility
                if hasattr(model_part, "unfuse_for_tp"):
                    if i == 0:
                        logger.info_rank0("Unfusing projections for tensor parallelism...")
                    model_part.unfuse_for_tp()

                tp_plan = _build_tp_plan(model_part)
                if tp_plan:
                    if i == 0:
                        logger.info_rank0(f"Apply tensor parallel to the model. Plan: {list(tp_plan.keys())}")
                    model_part = parallelize_module(
                        model_part,
                        device_mesh=ps.tp_mesh,
                        parallelize_plan=tp_plan,
                    )
                    model_parts[i] = model_part

                # With TP + meta init, we must load weights BEFORE FSDP wrapping.
                if kwargs.get("init_device") == "meta" and weights_path is not None:
                    from torch.distributed.tensor import distribute_tensor

                    if i == 0:
                        logger.info_rank0("TP enabled: loading weights before FSDP wrapping...")
                    load_weights_mode = kwargs.get("load_weights_mode", "broadcast")
                    if load_weights_mode == "broadcast":
                        rank0_load_and_broadcast_weights(
                            model_part, weights_path, get_device_type(), dtensor_factory=distribute_tensor
                        )
                    else:
                        all_ranks_load_weights(
                            model_part, weights_path, get_device_type(), dtensor_factory=distribute_tensor
                        )
                    kwargs["skip_weight_loading"] = True

            # torch.compile (if enabled)
            if enable_compile:
                target_classes = set((getattr(model_part, "_no_split_modules", []) or []) + (basic_modules or []))
                compiled_count = 0
                for fqn, mod in model_part.named_modules():
                    if mod.__class__.__name__ in target_classes:
                        parent_fqn, _, child_name = fqn.rpartition(".")
                        parent = model_part.get_submodule(parent_fqn) if parent_fqn else model_part
                        compiled_mod = torch.compile(mod)
                        setattr(parent, child_name, compiled_mod)
                        compiled_count += 1
                if i == 0:
                    logger.info_rank0(f"torch.compile applied to {compiled_count} decoder layers")

                # Enable compiled vocab-parallel cross-entropy kernels
                if hasattr(model_part, "loss_function"):
                    model_part.loss_function = functools.partial(model_part.loss_function, use_compile=True)
                    if i == 0:
                        logger.info_rank0("Enabled compiled vocab-parallel cross-entropy")

            # FSDP
            if ps.fsdp_enabled:
                if i == 0:
                    logger.info_rank0(f"Apply data parallel to the model: {ps.dp_mode}.")
                if ps.dp_mode == "fsdp2":
                    model_part = parallelize_model_fsdp2(
                        model=model_part,
                        weights_path=weights_path,
                        enable_full_shard=enable_full_shard,
                        enable_mixed_precision=enable_mixed_precision,
                        basic_modules=basic_modules,
                        pp_enabled=True,
                        reshard_after_forward=reshard_after_forward,
                        **kwargs,
                    )
                elif ps.dp_mode == "ddp":
                    ddp_kwargs = {"device_ids": [ps.local_rank]}
                    if enable_mixed_precision:
                        if i == 0:
                            logger.info_rank0("Enable mixed precision training.")
                        mixed_precision = MixedPrecision(
                            param_dtype=torch.bfloat16,
                            reduce_dtype=torch.float32,
                            buffer_dtype=torch.bfloat16,
                        )
                        ddp_kwargs["mixed_precision"] = mixed_precision
                    model_part = DDP(model_part, **ddp_kwargs)
                else:
                    if i == 0:
                        logger.info_rank0("No data parallelism (dp_mode=none), using model directly.")

                model_parts[i] = model_part

            # Update stage with wrapped model
            stages[i].submod = model_part

        # Determine which stages this rank has
        has_first_stage = any(s.stage_index == 0 for s in stages)
        has_last_stage = any(s.stage_index == pp_degree - 1 for s in stages)

        return {
            "stages": stages,
            "model_parts": model_parts,
            "has_first_stage": has_first_stage,
            "has_last_stage": has_last_stage,
        }

    # ---- Non-PP path (existing code, unchanged) ----

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        gc_kwargs = {"use_reentrant": use_reentrant}
        # Only pass context_fn when torch.compile is NOT enabled AND a custom
        # recompute_context_fn is explicitly provided. Passing context_fn
        # (even noop_context_fn) triggers SAC which uses torch.compile internally,
        # causing unexpected Inductor compilation when compile is disabled.
        recompute_fn = kwargs.pop("recompute_context_fn", None)
        if recompute_fn is not None and not enable_compile:
            gc_kwargs["context_fn"] = recompute_fn

        recompute_modules = kwargs.pop("recompute_modules", None)
        moe_checkpoint_method = kwargs.pop("moe_checkpoint_method", None)
        if recompute_modules is not None:
            gc_kwargs["recompute_modules"] = recompute_modules
        if moe_checkpoint_method is not None:
            gc_kwargs["moe_checkpoint_method"] = moe_checkpoint_method

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

        if use_reentrant:
            # Reentrant checkpointing doesn't support kwargs. Wrap the checkpoint
            # function to bundle any kwargs into the run_function via partial.
            for module in model.modules():
                if hasattr(module, "_gradient_checkpointing_func"):
                    _orig = module._gradient_checkpointing_func

                    def _make_wrapper(orig_fn):
                        def _reentrant_ckpt_with_kwargs(fn, *args, **kw):
                            if kw:
                                fn = functools.partial(fn, **kw)
                            return orig_fn(fn, *args)

                        return _reentrant_ckpt_with_kwargs

                    module._gradient_checkpointing_func = _make_wrapper(_orig)

    if parallel_state.tp_enabled:
        # TP + LoRA is not currently supported
        if any(isinstance(m, LoraLinear) for m in model.modules()):
            raise NotImplementedError("Tensor parallelism + LoRA is not currently supported.")

        # Unfuse fused projections (qkv_proj, gate_up_proj) for TP compatibility
        if hasattr(model, "unfuse_for_tp"):
            logger.info_rank0("Unfusing projections for tensor parallelism...")
            model.unfuse_for_tp()

        tp_plan = _build_tp_plan(model)
        logger.info_rank0(f"Apply tensor parallel to the model. Plan: {list(tp_plan.keys())}")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
            parallelize_plan=tp_plan,
        )

        # With TP + meta init, we must load weights BEFORE FSDP wrapping.
        # After parallelize_module, params are meta DTensors with TP placements.
        # FSDP's lazy_init can't handle meta DTensors correctly (size mismatch
        # between logical DTensor shape and local shard).
        # Load weights now so FSDP wraps materialized TP DTensors.
        if kwargs.get("init_device") == "meta" and weights_path is not None:
            from torch.distributed.tensor import distribute_tensor

            logger.info_rank0("TP enabled: loading weights before FSDP wrapping...")
            load_weights_mode = kwargs.get("load_weights_mode", "broadcast")
            if load_weights_mode == "broadcast":
                rank0_load_and_broadcast_weights(
                    model, weights_path, get_device_type(), dtensor_factory=distribute_tensor
                )
            else:
                all_ranks_load_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)
            # Mark weights as already loaded so FSDP path skips loading
            kwargs["skip_weight_loading"] = True

    if enable_compile:
        # Compile each decoder layer for torch.compile + FSDP2 compatibility.
        # Must happen BEFORE fully_shard so FSDP wraps the compiled modules.
        target_classes = set((getattr(model, "_no_split_modules", []) or []) + (basic_modules or []))
        compiled_count = 0
        for fqn, mod in model.named_modules():
            if mod.__class__.__name__ in target_classes:
                parent_fqn, _, child_name = fqn.rpartition(".")
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                compiled_mod = torch.compile(mod)
                setattr(parent, child_name, compiled_mod)
                compiled_count += 1
        logger.info_rank0(f"torch.compile applied to {compiled_count} decoder layers")

        # Enable compiled vocab-parallel cross-entropy kernels
        if hasattr(model, "loss_function"):
            model.loss_function = functools.partial(model.loss_function, use_compile=True)
            logger.info_rank0("Enabled compiled vocab-parallel cross-entropy")

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            model = parallelize_model_fsdp2(
                model=model,
                weights_path=weights_path,
                enable_full_shard=enable_full_shard,
                enable_mixed_precision=enable_mixed_precision,
                basic_modules=basic_modules,
                reshard_after_forward=reshard_after_forward,
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
