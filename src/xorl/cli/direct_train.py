import os


# Must be set before importing torch / initializing CUDA so the
# allocator picks up the setting on first use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import socket
import time
from dataclasses import asdict
from typing import Any, Dict, List

import torch.distributed as dist
from tqdm import trange

from xorl.arguments import Arguments, parse_args, save_args
from xorl.checkpoint import build_checkpointer
from xorl.data.constants import IGNORE_INDEX
from xorl.data.data_loader import DataLoaderBuilder
from xorl.data.prepare.prepare_datasets import prepare_datasets
from xorl.distributed.gradient_accumulate_loss import gradient_accumulate_loss
from xorl.distributed.offloading import build_activation_offloading_context
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.sync_padding import synchronize_micro_batch_padding
from xorl.distributed.torch_parallelize import build_parallelize_model
from xorl.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from xorl.models.layers.moe.aux_loss import global_load_balancing_loss_func
from xorl.models.module_utils import compute_loss
from xorl.optim import build_lr_scheduler, build_optimizer
from xorl.trainers.training_utils import (
    clip_gradients,
    count_valid_tokens,
    maybe_merge_lora,
    sync_sp_gradients,
)
from xorl.utils import helper
from xorl.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from xorl.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


def main():
    args = parse_args(Arguments)
    dist.init_process_group(backend=get_nccl_backend())
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)

    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)
        if args.train.use_wandb:
            import wandb

            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                tags=args.train.wandb_tags,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )
            config_file = os.path.join(args.train.output_dir, "xorl_cli.yaml")
            if os.path.exists(config_file):
                wandb.save(config_file, policy="now")

    host_payload = {
        "global_rank": args.train.global_rank,
        "local_rank": args.train.local_rank,
        "hostname": socket.gethostname(),
    }
    gathered_hosts = [None] * args.train.world_size
    dist.all_gather_object(gathered_hosts, host_payload)
    if args.train.global_rank == 0:
        unique_hostnames = sorted({item["hostname"] for item in gathered_hosts if item is not None})
        rank_to_hostname = {str(item["global_rank"]): item["hostname"] for item in gathered_hosts if item is not None}
        logger.info_rank0(
            "Host inventory:\n"
            + json.dumps(
                {
                    "master_addr": os.environ.get("MASTER_ADDR"),
                    "master_port": os.environ.get("MASTER_PORT"),
                    "node_count": len(unique_hostnames),
                    "hostnames": unique_hostnames,
                    "ranks": gathered_hosts,
                },
                indent=2,
            )
        )
        if args.train.use_wandb:
            import wandb

            wandb.config.update(
                {
                    "master_addr": os.environ.get("MASTER_ADDR"),
                    "master_port": os.environ.get("MASTER_PORT"),
                    "hostnames": unique_hostnames,
                    "rank_to_hostname": rank_to_hostname,
                },
                allow_val_change=True,
            )
            wandb.log({"startup/node_count": len(unique_hostnames)}, step=0, commit=False)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        ringattn_size=args.train.ringattn_parallel_size,
        dp_mode=args.train.data_parallel_mode,
        cp_fsdp_mode=args.train.cp_fsdp_mode,
    )

    # Initialize DTensor RNG tracker with run_state_sync=False to prevent a
    # world-group broadcast that deadlocks when PP stages run asynchronously.
    # Same approach as torchtitan (see torchtitan/distributed/utils.py:set_determinism).
    ps = get_parallel_state()
    if ps.device_mesh is not None:
        import torch.distributed.tensor._random

        torch.distributed.tensor._random.manual_seed(args.train.seed, ps.device_mesh)

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)

    # Load the datasets
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer)

    train_dataloader = DataLoaderBuilder(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        num_workers=args.data.dataloader_num_workers,
        drop_last=args.data.dataloader_drop_last,
        pin_memory=args.data.dataloader_pin_memory,
        prefetch_factor=args.data.dataloader_prefetch_factor,
        seed=args.train.seed,
        pad_to_multiple_of=args.data.pad_to_multiple_of,
    ).build()

    # Calculate train steps from dataloader length
    train_steps_per_epoch = len(train_dataloader)
    total_train_steps = train_steps_per_epoch * args.train.num_train_epochs
    if args.train.max_steps is not None:
        total_train_steps = min(total_train_steps, args.train.max_steps)
    logger.info_rank0(f"Train steps per epoch: {train_steps_per_epoch}, Total train steps: {total_train_steps}")

    # Convert save_epochs (fractional) to a step interval
    save_epoch_steps = int(args.train.save_epochs * train_steps_per_epoch) if args.train.save_epochs else 0
    if save_epoch_steps:
        logger.info_rank0(f"Save every {args.train.save_epochs} epoch(s) = every {save_epoch_steps} steps")

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        ep_dispatch=args.model.ep_dispatch,
        deepep_buffer_size_gb=args.model.deepep_buffer_size_gb,
        deepep_num_sms=args.model.deepep_num_sms,
        deepep_async_combine=args.model.deepep_async_combine,
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    # Unfuse QKV projections if merge_qkv=False so each projection is handled independently.
    if not args.model.merge_qkv:
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "unfuse_for_tp"):
                layer.self_attn.unfuse_for_tp()
        logger.info_rank0("Unfused QKV projections (merge_qkv=False)")

    # QLoRA injection: replace target nn.Linear with QLoRALinear.
    # With meta init, quantization is deferred — weights stay as nn.Parameter
    # so FSDP can load them normally. After FSDP loading, maybe_quantize_qlora()
    # converts them to uint8 buffers.
    # For pre-quantized checkpoints (NVFP4 modelopt format), quantization is
    # skipped entirely — packed weights + scales are loaded directly.
    is_prequantized = False
    checkpoint_quant_format = None
    exclude_modules = set()
    if args.lora.enable_qlora:
        from xorl.qlora import detect_prequantized_block_fp8, detect_prequantized_nvfp4, inject_qlora_into_model

        if detect_prequantized_nvfp4(args.model.model_path):
            is_prequantized = True
            checkpoint_quant_format = "nvfp4"
            logger.info_rank0("Detected pre-quantized NVFP4 checkpoint")
        elif detect_prequantized_block_fp8(args.model.model_path):
            is_prequantized = True
            checkpoint_quant_format = "block_fp8"
            logger.info_rank0("Detected pre-quantized block FP8 checkpoint")
        if args.lora.exclude_modules is not None:
            exclude_modules = set(args.lora.exclude_modules)
            logger.info_rank0(f"Using user-specified exclude_modules: {exclude_modules}")
        elif is_prequantized:
            from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules

            exclude_modules = get_prequantized_exclude_modules(args.model.model_path)
            if exclude_modules:
                logger.info_rank0(
                    f"Auto-detected {len(exclude_modules)} excluded modules from checkpoint config: {exclude_modules}"
                )
        if is_prequantized and checkpoint_quant_format != args.lora.quant_format:
            logger.info_rank0(
                f"Cross-format conversion: checkpoint={checkpoint_quant_format}, "
                f"target={args.lora.quant_format} — will dequantize and re-quantize"
            )

        inject_qlora_into_model(
            model,
            r=args.lora.lora_rank,
            lora_alpha=args.lora.lora_alpha,
            quant_format=args.lora.quant_format,
            quant_group_size=args.lora.quant_group_size,
            target_modules=args.lora.lora_target_modules,
            checkpoint_quant_format=checkpoint_quant_format,
            merge_qkv=args.model.merge_qkv,
            exclude_modules=exclude_modules,
            enable_aqn=args.lora.enable_aqn,
            aqn_alpha=args.lora.aqn_alpha,
        )
        # Store exclude_modules on model so checkpoint handler can use the
        # same set (user-specified or auto-detected) instead of re-detecting.
        if exclude_modules:
            model._qlora_exclude_modules = exclude_modules
        helper.print_device_mem_info("VRAM usage after QLoRA injection")
    elif args.lora.enable_lora:
        from xorl.lora.utils import inject_lora_into_model

        inject_lora_into_model(
            model,
            r=args.lora.lora_rank,
            lora_alpha=args.lora.lora_alpha,
            target_modules=args.lora.lora_target_modules,
        )
        helper.print_device_mem_info("VRAM usage after LoRA injection")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    build_result = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_compile=args.train.enable_compile,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        recompute_modules=args.train.recompute_modules,
        moe_checkpoint_method=args.train.moe_checkpoint_method,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        load_weights_mode=args.train.load_weights_mode,
        pp_schedule=args.train.pipeline_parallel_schedule if args.train.pipeline_parallel_size > 1 else None,
        reshard_after_forward=args.train.reshard_after_forward,
        skip_param_upcast=args.lora.enable_qlora,
    )

    # PP returns dict with stages + model_parts; otherwise returns model directly
    pp_enabled = isinstance(build_result, dict)
    pp_stages = None
    model_parts = None
    has_first_stage = False
    has_last_stage = False
    if pp_enabled:
        pp_stages = build_result["stages"]
        model_parts = build_result["model_parts"]
        has_first_stage = build_result["has_first_stage"]
        has_last_stage = build_result["has_last_stage"]
        model = model_parts[0]  # primary model for optimizer etc.
    else:
        model = build_result

    # Deferred QLoRA quantization: now that FSDP has loaded weights into the
    # weight parameters, quantize them into uint8 buffers and free the originals.
    # For pre-quantized checkpoints, skip quantization — load packed weights directly.
    if args.lora.enable_qlora:
        if is_prequantized:
            from xorl.qlora import maybe_load_prequantized_qlora

            logger.info("Starting pre-quantized NVFP4 weight loading...")
            helper.print_device_mem_info("VRAM before pre-quantized loading")
            maybe_load_prequantized_qlora(model, args.model.model_path)
            logger.info("Done pre-quantized weight loading, freezing non-LoRA params...")
        else:
            from xorl.qlora import maybe_load_and_quantize_moe_qlora, maybe_quantize_qlora
            from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp

            logger.info("Starting maybe_quantize_qlora...")
            helper.print_device_mem_info("VRAM before QLoRA quantization")
            maybe_quantize_qlora(model)
            logger.info("Done maybe_quantize_qlora, starting MoE weight loading...")
            helper.print_device_mem_info("VRAM after QLoRA linear quantization")
            # Load and quantize MoE expert weights directly from checkpoint
            # (bypasses FSDP to avoid OOM for large MoE models)
            maybe_load_and_quantize_moe_qlora(model, args.model.model_path)
            logger.info("Done MoE weight loading, deregistering packed weights...")
            # Deregister packed_weight_f32 from FSDP2 (prevent mixed-precision corruption)
            removed = _deregister_qlora_weights_from_fsdp(
                model,
                param_names=("packed_weight_f32",),
            )
            torch.cuda.empty_cache()
            if removed > 0:
                logger.info(f"Deregistered {removed} packed_weight_f32 params from FSDP2")
        # Freeze all non-LoRA parameters (embeddings, norms, lm_head, etc.)
        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad = False
        helper.print_device_mem_info("VRAM usage after QLoRA quantization")

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        optimizer_dtype=args.train.optimizer_dtype,
        optimizer_kwargs=args.train.optimizer_kwargs,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        # save model_assets before training
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        empty_cache_steps=args.train.empty_cache_steps,
        gc_enabled=args.train.enable_gradient_checkpointing,
        recompute_modules=args.train.recompute_modules,
        moe_checkpoint_method=args.train.moe_checkpoint_method,
        cp_size=args.train.ulysses_parallel_size * args.train.ringattn_parallel_size,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // train_steps_per_epoch
        start_step = global_step % train_steps_per_epoch
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    # Build PP schedule if pipeline parallelism is enabled
    pp_schedule = None
    pp_context = {}  # mutable container for per-step state used by pp_loss_fn
    if pp_enabled:
        import torch.nn.functional as F

        from xorl.distributed.pipeline_parallel import build_pipeline_schedule

        @torch.compile
        def _pp_ce_loss(pred, labels, ntokens):
            """PP loss: sum reduction, normalized by global_valid_tokens."""
            return (
                F.cross_entropy(
                    pred.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                / ntokens
            )

        def pp_loss_fn(pred, labels):
            return _pp_ce_loss(pred, labels, pp_context["global_valid_tokens"])

        pp_schedule = build_pipeline_schedule(
            stages=pp_stages,
            n_microbatches=args.train.gradient_accumulation_steps,
            loss_fn=pp_loss_fn,
            schedule_name=args.train.pipeline_parallel_schedule,
        )
        logger.info_rank0(f"PP schedule built: {args.train.pipeline_parallel_schedule}")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps_per_epoch: {train_steps_per_epoch}, "
        f"total_train_steps: {total_train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        # Compute actual steps this epoch, capped by max_steps
        steps_this_epoch = train_steps_per_epoch - start_step
        if args.train.max_steps is not None:
            steps_this_epoch = min(steps_this_epoch, args.train.max_steps - global_step)
        if steps_this_epoch <= 0:
            break

        data_loader_tqdm = trange(
            steps_this_epoch,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=start_step + steps_this_epoch,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, train_steps_per_epoch):
            if args.train.max_steps is not None and global_step >= args.train.max_steps:
                logger.info_rank0(f"Reached max_steps={args.train.max_steps}, stopping training.")
                break
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            # Synchronize padding across DP ranks to prevent load imbalance
            # Use fsdp_group (not world) when PP enabled to avoid NCCL conflicts
            sync_group = ps.fsdp_group if pp_enabled else None
            synchronize_micro_batch_padding(micro_batches, group=sync_group)

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            # compute global valid tokens across all ranks
            global_valid_tokens = count_valid_tokens(
                micro_batches,
                group=ps.fsdp_group if pp_enabled else None,
            )

            optimizer.zero_grad()

            # AQN: pre-generate noise for all QLoRA layers on side streams.
            # Runs async — overlaps with data prep below, so forward only
            # pays the cheap addcmul cost per layer.
            if args.lora.enable_aqn:
                from xorl.qlora.modules.linear import prefetch_aqn_noise

                prefetch_aqn_noise(model)

            # Routing replay stage switching for MoE checkpoint determinism.
            # Only needed with EP — without EP, expert compute has fixed output
            # shapes regardless of routing, so checkpoint recompute is safe.
            # See models/layers/moe/routing_replay.py for lifecycle docs.
            from xorl.models.layers.moe.routing_replay import RoutingReplay, set_replay_stage

            use_routing_replay = ps.ep_size > 1 and args.train.moe_recomputed

            if pp_enabled:
                # === Pipeline Parallel training path ===
                # Set global_valid_tokens for pp_loss_fn normalization
                pp_context["global_valid_tokens"] = global_valid_tokens

                for micro_batch in micro_batches:
                    environ_meter.add(micro_batch)

                # Prepare input_ids and labels tensors for PP schedule
                # PP schedule expects full batch tensors, splits into microbatches internally
                device = get_device_type()
                input_ids = torch.cat([mb["input_ids"].to(device, non_blocking=True) for mb in micro_batches], dim=0)
                labels = torch.cat([mb["labels"].to(device, non_blocking=True) for mb in micro_batches], dim=0)

                # Extract per-microbatch metadata for PP forward:
                # - position_ids: full-length (not SP-sliced) for correct per-document RoPE
                # - cu_seq_lens/max_length: flash-attention varlen kwargs for document boundaries
                # Each _pp_forward call pops one entry from the deque.
                from collections import deque

                _PP_FA_KEYS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
                pp_metadata_list = []
                for mb in micro_batches:
                    md = {}
                    if "position_ids" in mb:
                        md["position_ids"] = mb["position_ids"]
                    for key in _PP_FA_KEYS:
                        if key in mb:
                            md[key] = mb[key]
                    pp_metadata_list.append(md)
                for model_part in model_parts:
                    model_part._pp_batch_metadata = deque(pp_metadata_list)

                # Only last stage computes loss
                if has_last_stage:
                    targets = labels
                    losses = []
                else:
                    targets = None
                    losses = None

                # Routing replay: global stage = "replay_backward" so checkpoint
                # recompute uses recorded routing.  _pp_forward temporarily
                # switches to "record" during each forward call.
                if use_routing_replay:
                    set_replay_stage("replay_backward")

                # Run PP schedule (handles fwd/bwd for all microbatches)
                if has_first_stage:
                    pp_schedule.step(input_ids, target=targets, losses=losses)
                else:
                    pp_schedule.step(target=targets, losses=losses)

                if use_routing_replay:
                    set_replay_stage(None)
                    RoutingReplay.clear_all()

                # Compute loss for logging (losses already normalized by global_valid_tokens)
                if has_last_stage:
                    total_loss = torch.sum(torch.stack(losses)).item()
                    loss_tensor = torch.tensor([total_loss], device=device)
                else:
                    loss_tensor = torch.tensor([-1.0], device=device)

                # Share loss across PP stages (MAX broadcasts from last stage)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.MAX, group=ps.pp_group)
                total_loss = loss_tensor.item()

                del input_ids, labels
            else:
                # === Standard gradient accumulation path ===
                # Global stage for recompute; switched to "record" around forward.
                if use_routing_replay:
                    set_replay_stage("replay_backward")

                for micro_batch in micro_batches:
                    environ_meter.add(micro_batch)

                    micro_batch = {
                        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in micro_batch.items()
                    }

                    # Pop labels before forward (model no longer takes labels)
                    labels = micro_batch.pop("labels", None)

                    if use_routing_replay:
                        set_replay_stage("record")
                    with model_fwd_context:
                        outputs = model(**micro_batch, use_cache=False, output_hidden_states=False)

                        # Loss computation: lm_head weight stays all-gathered
                        # via reshard_after_forward=False on norm + lm_head FSDP unit
                        result = compute_loss(
                            model.lm_head,
                            outputs.last_hidden_state,
                            loss_fn_name=None,
                            loss_fn_inputs={"labels": labels},
                            loss_fn_params=None,
                            logits_to_keep=0,
                        )
                        loss = result.loss

                        # MoE aux loss from router logits (if applicable)
                        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                            aux_loss = global_load_balancing_loss_func(
                                outputs.router_logits,
                                model.num_experts,
                                model.num_experts_per_tok,
                                dp_group=ps.dp_group if ps.dp_enabled else None,
                            )
                            if aux_loss != 0:
                                loss = loss + model.router_aux_loss_coef * aux_loss.to(loss.device)

                        local_valid_tokens = (labels != IGNORE_INDEX).sum()
                        ga_loss, _ = gradient_accumulate_loss(loss, local_valid_tokens, global_valid_tokens)
                    if use_routing_replay:
                        set_replay_stage("replay_backward")

                    with model_bwd_context:
                        ga_loss.backward()

                    # NOTE: Do NOT reset backward indices here — the backward_index
                    # must increment across micro-batches (entry 0 = MB0, entry 1 = MB1, etc.)
                    # reset_all_backward() would cause MB1's recompute to replay MB0's routing.

                    loss_item = ga_loss.item()
                    total_loss += loss_item

                    # Clean up tensors to free memory
                    del micro_batch, labels, loss, outputs, ga_loss

                if use_routing_replay:
                    set_replay_stage(None)
                    RoutingReplay.clear_all()

            # Sync gradients across ring/Ulysses dims not folded into FSDP
            sync_sp_gradients(model, ps.sp_grad_sync_group)

            # Gradient clipping
            grad_norm = clip_gradients(
                model,
                args.train.max_grad_norm,
                pp_enabled=pp_enabled,
                pp_group=ps.pp_group if pp_enabled else None,
            )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Periodic LoRA merge: absorb LoRA delta into base weights
            maybe_merge_lora(
                model,
                enable_lora=args.lora.enable_lora,
                enable_qlora=args.lora.enable_qlora,
                merge_interval=args.lora.merge_lora_interval,
                global_step=global_step,
                optimizer=optimizer,
                reset_optimizer=args.lora.reset_optimizer_on_merge,
            )
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # Collect mean loss and grad_norm across data parallel group for logging.
            # For PP: total_loss is this rank's sum(losses)/global_valid_tokens where
            # global_valid_tokens already includes all DP ranks. SUM across DP gives
            # the correct global per-token loss. grad_norm is already consistent
            # across all ranks (FSDP all-reduce + PP MAX), so just take mean (no-op).
            if pp_enabled:
                total_loss = all_reduce(total_loss, op="sum", group=ps.fsdp_group)
                grad_norm = all_reduce(grad_norm, op="mean", group=ps.fsdp_group)
            else:
                total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=ps.fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            tokens_per_sec = train_metrics.get("efficiency/tokens_per_second(K)", 0) * 1e3
            data_loader_tqdm.set_postfix_str(
                f"loss={total_loss:.2f} gn={grad_norm:.2f} lr={lr:.1e} tok/s={tokens_per_sec:.0f}"
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb and global_step % args.train.wandb_log_interval == 0:
                    import wandb

                    train_metrics.update(
                        {
                            "training/loss": total_loss,
                            "training/grad_norm": grad_norm,
                            "training/lr": lr,
                            "training/epoch": epoch,
                            "training/step_time": delta_time,
                            "training/samples_seen": global_step * args.train.global_batch_size,
                        }
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            should_save = (args.train.save_steps and global_step % args.train.save_steps == 0) or (
                save_epoch_steps and global_step % save_epoch_steps == 0
            )
            if should_save:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                # Determine if we should save only LoRA params (base weights unchanged)
                is_lora_training = args.lora.enable_lora or args.lora.enable_qlora
                _save_lora_only = is_lora_training and args.lora.merge_lora_interval == 0
                Checkpointer.save(
                    args.train.save_checkpoint_path,
                    state,
                    global_steps=global_step,
                    save_lora_only=_save_lora_only,
                )

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

    synchronize()

    # Gather full model state via NCCL for HF save (all ranks must participate).
    # This is much faster than the DCP round-trip (write to disk → read back)
    # because NCCL AllGather is ~10-50 GB/s vs ~0.65 GB/s NFS.
    is_lora_training = args.lora.enable_lora or args.lora.enable_qlora
    save_peft_adapter = is_lora_training and args.lora.merge_lora_interval == 0

    hf_model_state_dict = None
    if args.train.save_hf_weights and not save_peft_adapter:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        logger.info_rank0("Gathering full model state dict for HF checkpoint via NCCL...")
        hf_model_state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))

    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()

    # save model in huggingface's format (rank 0 only)
    if args.train.global_rank == 0 and args.train.save_hf_weights:
        hf_weights_path = os.path.join(args.train.output_dir, f"global_step_{global_step}", "hf_ckpt")
        if save_peft_adapter:
            # Save PEFT adapter format (LoRA-only, base weights unchanged)
            from xorl.lora.utils import save_lora_checkpoint

            save_lora_checkpoint(
                model,
                hf_weights_path,
                base_model_name=args.model.model_path,
                target_modules=args.lora.lora_target_modules,
                r=args.lora.lora_rank,
                lora_alpha=args.lora.lora_alpha,
            )
            logger.info_rank0(f"PEFT adapter checkpoint saved at {hf_weights_path} successfully!")
        elif hf_model_state_dict is not None:
            checkpoint_handler = model.get_checkpoint_handler() if hasattr(model, "get_checkpoint_handler") else None
            save_model_weights(
                hf_weights_path, hf_model_state_dict, model_assets=model_assets, checkpoint_handler=checkpoint_handler
            )
            del hf_model_state_dict
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
