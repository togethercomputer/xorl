import os

# Must be set before importing torch / initializing CUDA so the
# allocator picks up the setting on first use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

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
from xorl.optim import build_lr_scheduler, build_optimizer
from xorl.utils import helper

from xorl.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)

from xorl.arguments import Arguments, DataArguments, ModelArguments, TrainingArguments, parse_args, save_args

from xorl.utils.dist_utils import all_reduce
from xorl.ops.loss.causallm_loss import causallm_loss_function


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

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)
    
    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        ep_size=args.train.expert_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

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
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        load_weights_mode=args.train.load_weights_mode,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        optimizer_dtype=args.train.optimizer_dtype,
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
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

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
            synchronize_micro_batch_padding(micro_batches)

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            # compute global valid tokens across all ranks
            global_valid_tokens = torch.tensor(0, device=get_device_type())
            for micro_batch in micro_batches:
                global_valid_tokens += (micro_batch["labels"] != IGNORE_INDEX).sum()

            dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM)

            
            # Handle gradient accumulation here
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                with model_fwd_context:
                    outputs = model(**micro_batch, use_cache=False, output_hidden_states=False)
                    loss = outputs.loss
                    
                    local_valid_tokens = (micro_batch["labels"] != IGNORE_INDEX).sum()
                    ga_loss, _ = gradient_accumulate_loss(loss, local_valid_tokens, global_valid_tokens)
                
                with model_bwd_context:
                    ga_loss.backward()

                loss_item = ga_loss.item()
                total_loss += loss_item

                '''
                memory pressure is too high and need to be optimized
                '''
                # # Gather debugging info from all ranks
                # local_debug_info = {
                #     "rank": args.train.global_rank,
                #     "ga_loss": loss_item,
                #     "ga_loss_sum": ga_loss_sum.item(),
                #     "loss": loss.item(),
                #     "local_valid_tokens": local_valid_tokens.item(),
                #     "global_valid_tokens": global_valid_tokens.item(),
                #     "input_ids": micro_batch["input_ids"].tolist(),
                #     "labels": labels.tolist(),
                #     "per_token_losses": per_token_losses.tolist(),
                # }
                
                # # Gather from all ranks to rank 0
                # all_ranks_debug_info = [None] * args.train.world_size
                # dist.all_gather_object(all_ranks_debug_info, local_debug_info)
                
                # step_batch_history.append(
                #     {
                #         "all_ranks_info": all_ranks_debug_info if args.train.global_rank == 0 else None,
                #     }
                # )

                # Clean up tensors to free memory
                del micro_batch, loss, outputs, ga_loss
            
            # Prefer model-provided clip_grad_norm_ (FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                logger.info_rank0(
                    "Can NOT find regitsered clip_grad_norm_ method in the model, using PyTorch default implementation.."
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            tflops_per_gpu = train_metrics.get("flops_achieved(T)", 0) / args.train.world_size
            data_loader_tqdm.set_postfix_str(f"loss={total_loss:.2f} gn={grad_norm:.2f} lr={lr:.1e} tflops={tflops_per_gpu:.1f}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            should_save = (args.train.save_steps and global_step % args.train.save_steps == 0) or \
                          (save_epoch_steps and global_step % save_epoch_steps == 0)
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
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

    synchronize()

    # Gather full model state via NCCL for HF save (all ranks must participate).
    # This is much faster than the DCP round-trip (write to disk → read back)
    # because NCCL AllGather is ~10-50 GB/s vs ~0.65 GB/s NFS.
    hf_model_state_dict = None
    if args.train.save_hf_weights:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        logger.info_rank0("Gathering full model state dict for HF checkpoint via NCCL...")
        hf_model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True)
        )

    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()

    # save model in huggingface's format (rank 0 only)
    if args.train.global_rank == 0 and hf_model_state_dict is not None:
        hf_weights_path = os.path.join(args.train.output_dir, f"global_step_{global_step}", "hf_ckpt")
        checkpoint_handler = model.get_checkpoint_handler() if hasattr(model, "get_checkpoint_handler") else None
        save_model_weights(hf_weights_path, hf_model_state_dict, model_assets=model_assets, checkpoint_handler=checkpoint_handler)
        del hf_model_state_dict
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
