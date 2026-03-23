"""
CheckpointManager - Handles Checkpoint Save/Load Operations for ModelRunner

Extracted from model_runner.py to separate checkpoint concerns:
- DCP checkpointing (save_state / load_state)
- LoRA adapter saving in PEFT-compatible format (save_adapter_state / load_adapter_state)
- Full weight extraction with Expert Parallelism support
- Safetensors export (single-writer and distributed)
- HuggingFace-compatible weight saving for inference/sampling
"""

import gc
import json
import logging
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from xorl.checkpoint import ckpt_to_state_dict
from xorl.checkpoint.checkpointer import ModelState
from xorl.distributed.parallel_state import get_parallel_state
from xorl.lora.utils import get_lora_state_dict, save_lora_checkpoint
from xorl.models import save_model_weights
from xorl.utils import helper

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint save/load operations for ModelRunner.

    Handles DCP checkpointing, LoRA adapter saving (PEFT format),
    full weight extraction with EP support, and safetensors export.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        checkpointer,
        lora_config: Dict[str, Any],
        model_config: Dict[str, Any],
        train_config: Dict[str, Any],
        rank: int,
        local_rank: int,
        adapter_manager=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.Checkpointer = checkpointer
        self.lora_config = lora_config
        self.model_config = model_config
        self.train_config = train_config
        self.rank = rank
        self.local_rank = local_rank
        self._adapter_manager = adapter_manager

        # These will be set/updated by ModelRunner
        self.global_step = 0
        self.global_forward_backward_step = 0
        self.global_rank = rank  # Default to rank; ModelRunner can override
        self.lora_target_modules = None
        self.lora_alpha_value = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lora_save_config(self):
        """Get target_modules and lora_alpha for PEFT-format saving.

        Uses cached values from model initialization, falling back to
        config reconstruction for backward compatibility.

        Returns:
            Tuple of (target_modules, lora_alpha)
        """
        target_modules = getattr(self, "lora_target_modules", None)
        lora_alpha = getattr(self, "lora_alpha_value", None)

        if target_modules is None or lora_alpha is None:
            logger.warning("lora_target_modules not set, reconstructing from config")
            explicit_target_modules = self.lora_config.get("lora_target_modules", None)
            if explicit_target_modules is not None:
                target_modules = explicit_target_modules
                lora_alpha = self.lora_config.get("lora_alpha", 16)
            else:
                target_modules = []
                if self.lora_config.get("train_attn", True):
                    target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
                if self.lora_config.get("train_mlp", True):
                    target_modules.extend(["gate_proj", "up_proj", "down_proj"])
                if self.lora_config.get("train_unembed", True):
                    target_modules.append("lm_head")
                lora_alpha = self.lora_config.get("lora_alpha", 32)

        return target_modules, lora_alpha

    # ------------------------------------------------------------------
    # Adapter save / load (multi-tenancy LoRA)
    # ------------------------------------------------------------------

    def _save_lora_weights(self, save_path: str, model_id: str) -> None:
        """
        Core LoRA saving logic: activate adapter, gather weights, write PEFT checkpoint.

        This is a collective operation — ALL ranks must call it.
        Rank 0 writes the PEFT-format checkpoint files.
        Memory is cleaned up after saving.

        Args:
            save_path: Directory to save LoRA weights to
            model_id: The adapter/session identifier to activate before saving
        """
        # Ensure adapter weights are synced to model
        if self._adapter_manager is not None:
            self._adapter_manager.switch_adapter(model_id, auto_register=True)

        # EP+FSDP2-aware LoRA weight gathering (collective operation)
        lora_state_dict = get_lora_state_dict(self.model)

        # Only rank 0 writes files
        if self.rank == 0:
            target_modules, lora_alpha = self._get_lora_save_config()
            save_lora_checkpoint(
                model=self.model,
                save_path=save_path,
                base_model_name=self.model_config.get("model_path"),
                target_modules=target_modules,
                r=self.lora_config.get("lora_rank", 32),
                lora_alpha=lora_alpha,
                moe_hybrid_shared_lora=self.lora_config.get("moe_hybrid_shared_lora", False),
                lora_state_dict=lora_state_dict,
            )

        # Cleanup
        del lora_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def save_adapter_state(
        self, model_id: str, path: Optional[str] = None, save_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Save a specific LoRA adapter's state to disk.

        Uses shared LoRA utilities for PEFT-format saving with proper EP+FSDP2
        gathering. ALL ranks must call this method (collective operations).

        Args:
            model_id: The adapter/session identifier to save
            path: Directory to save to (auto-generated if None)
            save_optimizer: Whether to save optimizer state

        Returns:
            Dict with path, model_id, step, and success status
        """
        if self._adapter_manager is None:
            raise ValueError(
                "Multi-tenancy is not enabled. save_adapter_state requires "
                "LoRA with adapter_manager enabled."
            )

        start_time = time.time()

        # Get adapter state for metadata and optimizer
        adapter_state = self._adapter_manager.get_adapter_state(model_id)

        # Use default path if not provided
        if path is None:
            path = os.path.join(self._adapter_manager.checkpoint_dir, model_id)

        # Save LoRA weights (collective operation)
        self._save_lora_weights(path, model_id)

        # Only rank 0 writes optimizer and metadata
        if self.rank == 0:
            # Save optimizer state (adapter-specific)
            if save_optimizer:
                optimizer_path = os.path.join(path, "optimizer.pt")
                torch.save(adapter_state.optimizer.state_dict(), optimizer_path)

            # Save metadata (adapter-specific)
            metadata = {
                "model_id": model_id,
                "global_step": adapter_state.global_step,
                "global_forward_backward_step": adapter_state.global_forward_backward_step,
                "lr": adapter_state.lr,
                "timestamp": time.time(),
                "save_optimizer": save_optimizer,
            }
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"Saved adapter state for model_id={model_id} to {path} "
                f"(step={adapter_state.global_step}, save_optimizer={save_optimizer})"
            )

        dist.barrier()

        return {
            "path": path,
            "model_id": model_id,
            "step": adapter_state.global_step,
            "save_time": time.time() - start_time,
            "success": True,
        }

    def load_adapter_state(
        self,
        model_id: str,
        path: Optional[str] = None,
        load_optimizer: bool = True,
        lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Load a LoRA adapter's state from disk.

        Args:
            model_id: Target adapter/session identifier to load into
            path: Directory to load from
            load_optimizer: Whether to load optimizer state
            lr: Learning rate (uses saved value if None)

        Returns:
            Dict with path, model_id, step, and success status
        """
        if self._adapter_manager is None:
            raise ValueError(
                "Multi-tenancy is not enabled. load_adapter_state requires "
                "LoRA with adapter_manager enabled."
            )

        result = self._adapter_manager.load_adapter_state(
            model_id=model_id,
            path=path,
            load_optimizer=load_optimizer,
            lr=lr,
        )
        result["success"] = True
        return result

    # ------------------------------------------------------------------
    # Weight extraction helpers
    # ------------------------------------------------------------------

    def extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract model weights from FSDP model for checkpoint-engine broadcast.

        Returns:
            Dictionary mapping parameter names to tensors (on CPU)
        """
        # For FSDP2, use the proper distributed checkpoint API to get full state dict
        state_dict_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=False,  # rank0_only mode
        )

        logger.info(f"Rank {self.rank}: [WeightSync] Calling get_model_state_dict (FSDP collective)...")
        state_dict = get_model_state_dict(self.model, options=state_dict_options)
        logger.info(f"Rank {self.rank}: [WeightSync] get_model_state_dict complete, {len(state_dict)} params")

        if self.rank == 0:
            return state_dict
        else:
            return {}

    def extract_full_weights_with_ep(self) -> Dict[str, torch.Tensor]:
        """
        Extract full model weights, handling EP (Expert Parallelism) correctly.

        For EP-enabled models, this uses ModelState which properly handles EP dimension
        restoration. The returned state_dict may contain DTensors for EP-sharded params
        which are then gathered to full tensors.

        This is a collective operation - ALL ranks must participate.

        Returns:
            Dictionary mapping parameter names to full tensors (on CPU), only on rank 0.
            Other ranks return empty dict after participating in collective ops.
        """
        ps = get_parallel_state()

        if ps.ep_enabled:
            # For EP-enabled models, use ModelState which handles EP dimension properly
            logger.debug(f"Rank {self.rank}: Using EP-aware ModelState for state_dict extraction...")

            # Create ModelState wrapper - this is a collective operation
            model_state = ModelState(self.model)
            state_dict = model_state.state_dict()

            logger.debug(f"Rank {self.rank}: Extracted {len(state_dict)} params via EP-aware ModelState")

            # Now gather DTensors to full tensors
            # DTensors represent EP-sharded parameters that need to be gathered
            result = {}
            dtensor_count = 0
            regular_count = 0

            for name, tensor in state_dict.items():
                if isinstance(tensor, DTensor):
                    # Gather DTensor to full tensor (collective operation - all ranks must call)
                    # full_tensor() gathers across the mesh and returns a regular tensor
                    full_tensor = tensor.full_tensor()

                    # Only rank 0 keeps the result on CPU to avoid OOM
                    # (8 ranks * 470GB model = 3.76TB would exceed node memory)
                    if self.rank == 0:
                        result[name] = full_tensor.cpu()

                        if dtensor_count <= 3:
                            logger.debug(
                                f"Rank {self.rank}: Gathered DTensor {name}: "
                                f"local {tensor.shape} -> full {result[name].shape}"
                            )

                    # All ranks: free GPU memory immediately
                    del full_tensor
                    dtensor_count += 1

                    # Periodic cache cleanup to keep GPU memory pressure low
                    if dtensor_count % 50 == 0:
                        torch.cuda.empty_cache()
                else:
                    # Regular tensor - only rank 0 keeps it
                    if self.rank == 0:
                        result[name] = tensor.cpu() if tensor.is_cuda else tensor
                    regular_count += 1

            # Synchronize
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            logger.debug(
                f"Rank {self.rank}: EP weight extraction complete - "
                f"gathered {dtensor_count} DTensors, {regular_count} regular tensors"
            )

            # Only rank 0 needs the full result for saving
            if self.rank == 0:
                return result
            else:
                return {}
        else:
            # Non-EP: use standard extraction
            return self.extract_model_weights()

    # ------------------------------------------------------------------
    # Filesystem / safetensors save
    # ------------------------------------------------------------------

    def save_full_weights(
        self,
        output_path: str,
        dtype: str = "bfloat16",
        base_model_path: Optional[str] = None,
        distributed_write: bool = True,
    ) -> Dict[str, Any]:
        """
        Save full model weights as safetensors with config files for SGLang loading.

        Creates a directory structure compatible with HuggingFace/SGLang:
        output_path/
            model.safetensors (or model-00001-of-00002.safetensors for sharded)
            model.safetensors.index.json (if sharded)
            config.json
            tokenizer.json
            tokenizer_config.json
            ...

        Args:
            output_path: Directory to save safetensors and config files
            dtype: Target dtype for weights (bfloat16, float16, float32)
            base_model_path: Path to base model for config files
            distributed_write: If True, distribute shard writing across nodes (faster for multi-node).
                              If False, only rank 0 writes (original behavior).

        Returns:
            Dictionary with save status, path, and number of shards
        """
        ps = get_parallel_state()

        # Use distributed writing only if we have multiple nodes and EP is enabled
        # For single-node or non-EP, fall back to rank-0-only writing
        use_distributed = distributed_write and ps.ep_enabled and ps.world_size > 1

        if use_distributed:
            return self._save_full_weights_distributed(
                output_path, dtype, base_model_path
            )
        else:
            return self._save_full_weights_single_writer(
                output_path, dtype, base_model_path
            )

    @staticmethod
    def _copy_model_configs(output_path: str, base_model_path: str) -> None:
        """Copy model config files (config.json, tokenizer.json, etc.) from base model."""
        config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]
        for config_file in config_files:
            src = os.path.join(base_model_path, config_file)
            if os.path.exists(src):
                dst = os.path.join(output_path, config_file)
                shutil.copy2(src, dst)
                logger.info(f"Copied {config_file}")

    def _save_full_weights_single_writer(
        self,
        output_path: str,
        dtype: str,
        base_model_path: Optional[str],
    ) -> Dict[str, Any]:
        """Single-writer implementation using shared save_model_weights()."""
        start_time = time.time()

        # ALL ranks must participate in weight extraction (collective operation)
        state_dict = self.extract_full_weights_with_ep()

        # Get checkpoint handler for HF-compatible weight transforms
        # (e.g., splitting gate_up_proj back into gate_proj + up_proj)
        checkpoint_handler = (
            self.model.get_checkpoint_handler()
            if hasattr(self.model, "get_checkpoint_handler")
            else None
        )

        # save_model_weights handles dtype conversion, sharding, and index.json
        save_model_weights(
            output_dir=output_path,
            state_dict=state_dict,
            global_rank=self.rank,
            save_dtype=dtype,
            checkpoint_handler=checkpoint_handler,
        )

        # Copy config files from base model (server-specific)
        if self.rank == 0 and base_model_path and os.path.isdir(base_model_path):
            self._copy_model_configs(output_path, base_model_path)

        save_time = time.time() - start_time

        dist.barrier()

        if self.rank == 0:
            logger.info(f"Saved full weights to {output_path} in {save_time:.2f}s")
            return {
                "status": "success",
                "output_path": output_path,
                "dtype": dtype,
                "save_time": save_time,
            }
        else:
            return {"status": "skipped", "reason": "non-rank-0"}

    def _save_full_weights_distributed(
        self,
        output_path: str,
        dtype: str,
        base_model_path: Optional[str],
    ) -> Dict[str, Any]:
        """
        Distributed safetensors writing - multiple nodes write shards in parallel.

        This is optimized for large EP models across multiple nodes:
        - Writer ranks (local_rank == 0 on each node) each write a subset of shards
        - All ranks participate in DTensor gathering (required for collectives)
        - Memory is distributed: each writer only holds its assigned shards
        - I/O is parallelized: 8 nodes = 8x write bandwidth

        For EP=64 across 8 nodes with 95 shards:
        - Each node writes ~12 shards
        - Expected speedup: ~6-8x for I/O phase
        """
        start_time = time.time()
        ps = get_parallel_state()

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(dtype, torch.bfloat16)
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5 GB per shard

        # Determine writer ranks: local_rank == 0 on each node
        local_rank = ps.local_rank
        is_writer = local_rank == 0
        world_size = ps.world_size
        # Assuming 8 GPUs per node (common for H100/A100 clusters)
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
        num_writers = world_size // gpus_per_node
        # Writer index: which writer am I (0 to num_writers-1)
        writer_idx = self.rank // gpus_per_node if is_writer else -1

        logger.debug(
            f"Rank {self.rank}: distributed safetensors save - "
            f"is_writer={is_writer}, writer_idx={writer_idx}, num_writers={num_writers}"
        )

        # Create output directory (all writers need to write here)
        if is_writer:
            os.makedirs(output_path, exist_ok=True)
        dist.barrier()

        # Phase 1: Get state_dict structure and compute shard assignments
        # All ranks need to participate in this to get consistent tensor ordering
        if ps.ep_enabled:
            model_state = ModelState(self.model)
            state_dict_meta = model_state.state_dict()
        else:
            state_dict_meta = {
                name: param for name, param in self.model.named_parameters()
            }

        # Compute tensor sizes and shard assignments (all ranks compute same assignment)
        tensor_infos = []  # [(name, estimated_size, is_dtensor), ...]
        for name, tensor in state_dict_meta.items():
            if isinstance(tensor, DTensor):
                # For DTensor, compute full size from local shape and mesh
                local_shape = tensor.to_local().shape
                # Get the sharding spec to compute full shape
                full_shape = tensor.shape  # DTensor.shape returns full logical shape
                numel = 1
                for dim in full_shape:
                    numel *= dim
            else:
                numel = tensor.numel()

            # Estimate size after dtype conversion
            if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                element_size = target_dtype.itemsize if hasattr(target_dtype, 'itemsize') else (
                    2 if target_dtype in (torch.float16, torch.bfloat16) else 4
                )
            else:
                element_size = tensor.element_size()

            tensor_size = numel * element_size
            tensor_infos.append((name, tensor_size, isinstance(tensor, DTensor)))

        # Compute shard assignments: which tensors go in which shard
        shard_assignments = []  # [(shard_idx, [tensor_names]), ...]
        current_shard_tensors = []
        current_shard_size = 0
        shard_idx = 0

        for name, size, _ in tensor_infos:
            if current_shard_size + size > max_shard_size and current_shard_tensors:
                shard_assignments.append((shard_idx, current_shard_tensors))
                shard_idx += 1
                current_shard_tensors = []
                current_shard_size = 0
            current_shard_tensors.append(name)
            current_shard_size += size

        if current_shard_tensors:
            shard_assignments.append((shard_idx, current_shard_tensors))

        num_shards = len(shard_assignments)
        logger.debug(f"Rank {self.rank}: computed {num_shards} shards from {len(tensor_infos)} tensors")

        # Assign shards to writers (round-robin)
        # shard_to_writer[shard_idx] = writer_idx
        shard_to_writer = {s_idx: s_idx % num_writers for s_idx, _ in shard_assignments}

        # Build reverse mapping: for this writer, which shards do I own?
        my_shards = []
        if is_writer:
            my_shards = [s_idx for s_idx, w_idx in shard_to_writer.items() if w_idx == writer_idx]
            logger.debug(f"Rank {self.rank} (writer {writer_idx}): assigned shards {my_shards}")

        # Phase 2: Extract weights with distributed ownership
        # All ranks participate in DTensor gathers, but only assigned writers keep results
        my_shard_data = {s_idx: {} for s_idx in my_shards}  # shard_idx -> {name: tensor}

        # Build tensor_name -> shard_idx mapping
        tensor_to_shard = {}
        for shard_idx, tensor_names in shard_assignments:
            for name in tensor_names:
                tensor_to_shard[name] = shard_idx

        dtensor_count = 0
        regular_count = 0

        for name, tensor in state_dict_meta.items():
            shard_idx = tensor_to_shard[name]
            shard_writer = shard_to_writer[shard_idx]
            i_should_keep = is_writer and (writer_idx == shard_writer)

            if isinstance(tensor, DTensor):
                # All ranks must call full_tensor() - it's a collective
                full_tensor = tensor.full_tensor()

                if i_should_keep:
                    # Convert dtype and move to CPU
                    cpu_tensor = full_tensor.cpu()
                    if cpu_tensor.dtype == target_dtype:
                        converted = cpu_tensor
                    elif cpu_tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        converted = cpu_tensor.to(target_dtype)
                    else:
                        converted = cpu_tensor
                    my_shard_data[shard_idx][name] = converted

                # All ranks free GPU memory
                del full_tensor
                dtensor_count += 1

                if dtensor_count % 50 == 0:
                    torch.cuda.empty_cache()
            else:
                # Regular tensor (replicated) - only assigned writer keeps it
                if i_should_keep:
                    cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor.clone()
                    if cpu_tensor.dtype == target_dtype:
                        converted = cpu_tensor
                    elif cpu_tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        converted = cpu_tensor.to(target_dtype)
                    else:
                        converted = cpu_tensor
                    my_shard_data[shard_idx][name] = converted
                regular_count += 1

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        logger.debug(
            f"Rank {self.rank}: extracted {dtensor_count} DTensors, {regular_count} regular tensors"
        )

        # Phase 3: Writers save their shards in parallel
        my_shard_results = []  # [(shard_idx, shard_name, weight_map, size), ...]

        if is_writer and my_shards:
            def _save_shard(shard_idx):
                if num_shards == 1:
                    shard_name = "model.safetensors"
                else:
                    shard_name = f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
                shard_path = os.path.join(output_path, shard_name)
                shard_data = my_shard_data[shard_idx]
                save_file(shard_data, shard_path)
                shard_weight_map = {n: shard_name for n in shard_data.keys()}
                shard_size = sum(t.numel() * t.element_size() for t in shard_data.values())
                return shard_idx, shard_name, shard_weight_map, shard_size

            # Use ThreadPoolExecutor for parallel writes within this writer
            max_workers = min(4, len(my_shards))
            if len(my_shards) == 1:
                result = _save_shard(my_shards[0])
                my_shard_results.append(result)
                logger.debug(f"Rank {self.rank}: saved {result[1]}")
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_save_shard, s_idx): s_idx for s_idx in my_shards}
                    for future in as_completed(futures):
                        result = future.result()
                        my_shard_results.append(result)
                        logger.debug(f"Rank {self.rank}: saved {result[1]}")

            # Free shard data after writing
            my_shard_data.clear()

        # Barrier to ensure all shards are written before index.json
        dist.barrier()

        # Phase 4: Gather shard metadata to rank 0 and write index.json
        if num_shards > 1:
            # Serialize shard results for gathering
            if is_writer:
                local_results_bytes = pickle.dumps(my_shard_results)
            else:
                local_results_bytes = pickle.dumps([])

            # Gather sizes first
            local_size = torch.tensor([len(local_results_bytes)], dtype=torch.long, device="cuda")
            all_sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)

            # Gather serialized results
            max_size = max(s.item() for s in all_sizes)
            local_padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
            local_padded[:len(local_results_bytes)] = torch.tensor(
                list(local_results_bytes), dtype=torch.uint8, device="cuda"
            )
            all_results_padded = [
                torch.zeros(max_size, dtype=torch.uint8, device="cuda") for _ in range(world_size)
            ]
            dist.all_gather(all_results_padded, local_padded)

            if self.rank == 0:
                # Deserialize and combine all results
                all_shard_results = []
                for i, (padded, size_tensor) in enumerate(zip(all_results_padded, all_sizes)):
                    size = size_tensor.item()
                    if size > 0:
                        result_bytes = bytes(padded[:size].cpu().tolist())
                        results = pickle.loads(result_bytes)
                        all_shard_results.extend(results)

                # Sort by shard index and build weight_map
                all_shard_results.sort(key=lambda x: x[0])
                weight_map = {}
                total_size_for_index = 0
                for shard_idx, shard_name, shard_weight_map, shard_size in all_shard_results:
                    weight_map.update(shard_weight_map)
                    total_size_for_index += shard_size

                # Write index.json
                index = {
                    "metadata": {"total_size": total_size_for_index},
                    "weight_map": weight_map,
                }
                index_path = os.path.join(output_path, "model.safetensors.index.json")
                with open(index_path, "w") as f:
                    json.dump(index, f, indent=2)
                logger.info(f"Rank 0: wrote index.json with {len(weight_map)} entries")

        # Copy config files (rank 0 only)
        if self.rank == 0 and base_model_path and os.path.isdir(base_model_path):
            self._copy_model_configs(output_path, base_model_path)

        save_time = time.time() - start_time

        dist.barrier()

        if self.rank == 0:
            logger.info(
                f"Saved full weights (distributed) to {output_path} in {save_time:.2f}s "
                f"({num_shards} shards across {num_writers} writers)"
            )
            return {
                "status": "success",
                "output_path": output_path,
                "dtype": dtype,
                "num_shards": num_shards,
                "num_writers": num_writers,
                "save_time": save_time,
            }
        else:
            return {
                "status": "participated",
                "is_writer": is_writer,
                "shards_written": len(my_shard_results) if is_writer else 0,
            }

    # ------------------------------------------------------------------
    # DCP checkpoint save / load
    # ------------------------------------------------------------------

    def save_state(
        self, checkpoint_path: str, save_optimizer: bool = True, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save model and optimizer state.

        For multi-tenancy LoRA training: Delegates to save_adapter_state since
        base model weights never change - only adapter state needs to be saved.

        For non-LoRA or single-adapter mode: Uses xorl's DCP checkpointer to
        save full model and optimizer state.

        Args:
            checkpoint_path: Path to save checkpoint
            save_optimizer: Whether to save optimizer state
            model_id: For multi-tenancy, which adapter to save (default: current adapter)

        Returns:
            Dictionary with save status
        """
        # For multi-tenancy with LoRA, save adapter state (not full model)
        # Base weights never change, so we only need adapter + optimizer + metadata
        if self._adapter_manager is not None:
            target_model_id = model_id or self._adapter_manager.current_adapter_id or "default"
            logger.info(
                f"Multi-tenancy mode: delegating save_state to save_adapter_state "
                f"(model_id={target_model_id})"
            )
            return self.save_adapter_state(
                model_id=target_model_id,
                path=checkpoint_path,
                save_optimizer=save_optimizer,
            )

        # For non-LoRA or single-adapter mode, use original DCP approach
        start_time = time.time()

        helper.empty_cache()

        # Save using distributed checkpoint format (works for both full and LoRA models)
        # For LoRA models, only LoRA parameters are trainable, so only those get saved
        state = {
            "model": self.model,
            "optimizer": self.optimizer if save_optimizer else None,
            "extra_state": {
                "global_step": self.global_step,
                "global_forward_backward_step": self.global_forward_backward_step,
                "torch_rng_state": torch.get_rng_state(),
                "lora_enabled": self.lora_config.get("enable_lora", False),
            },
        }

        self.Checkpointer.save(checkpoint_path, state, global_steps=None)

        # Cleanup after checkpoint save to release intermediate memory
        del state
        gc.collect()
        torch.cuda.empty_cache()

        dist.barrier()

        result = {
            "checkpoint_path": checkpoint_path,
            "step": self.global_step,
            "save_optimizer": save_optimizer,
            "save_time": time.time() - start_time,
            "success": True,
            "lora_enabled": self.lora_config.get("enable_lora", False),
        }

        logger.info(f"Checkpoint saved to {checkpoint_path} (lora={self.lora_config.get('enabled', False)})")
        return result

    def save_lora_only(self, lora_path: str, model_id: str = "default") -> Dict[str, Any]:
        """
        Save only LoRA adapter weights (not the full model).

        Uses xorl's shared LoRA utilities to save in PEFT-compatible format.
        Handles FSDP2 DTensor gathering and EP shard gathering via shared utils.

        This is a collective operation — ALL ranks must call it.

        Args:
            lora_path: Path to save LoRA weights
            model_id: The model_id for multi-adapter training (default: "default")

        Returns:
            Dictionary with save status
        """
        if not self.lora_config.get("enable_lora", False):
            raise ValueError("LoRA is not enabled, cannot save LoRA-only checkpoint")

        start_time = time.time()

        # Save LoRA weights (collective operation)
        self._save_lora_weights(lora_path, model_id)

        dist.barrier()

        # Get step from adapter manager if available
        if self._adapter_manager is not None:
            current_step = self._adapter_manager.get_global_step(model_id)
        else:
            current_step = self.global_step

        return {
            "lora_path": lora_path,
            "step": current_step,
            "model_id": model_id,
            "save_time": time.time() - start_time,
            "success": True,
        }

    def load_state(
        self, checkpoint_path: str, load_optimizer: bool = True, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model and optimizer state.

        For multi-tenancy LoRA training: Delegates to load_adapter_state since
        base model weights never change - only adapter state needs to be loaded.

        For non-LoRA or single-adapter mode: Uses xorl's DCP checkpointer to
        load full model and optimizer state.

        Args:
            checkpoint_path: Path to load checkpoint from
            load_optimizer: Whether to load optimizer state
            model_id: For multi-tenancy, target adapter to load into (default: current adapter)

        Returns:
            Dictionary with load status
        """
        # For multi-tenancy with LoRA, load adapter state
        # This handles weight loading + optimizer + metadata
        if self._adapter_manager is not None:
            target_model_id = model_id or self._adapter_manager.current_adapter_id or "default"
            logger.info(
                f"Multi-tenancy mode: delegating load_state to load_adapter_state "
                f"(model_id={target_model_id})"
            )
            return self.load_adapter_state(
                model_id=target_model_id,
                path=checkpoint_path,
                load_optimizer=load_optimizer,
            )

        # For non-LoRA or single-adapter mode, use original DCP approach
        start_time = time.time()

        state = {"model": self.model, "optimizer": self.optimizer if load_optimizer else None, "extra_state": {}}

        self.Checkpointer.load(checkpoint_path, state)

        # Restore state
        self.global_step = state["extra_state"].get("global_step", 0)
        self.global_forward_backward_step = state["extra_state"].get("global_forward_backward_step", 0)
        torch.set_rng_state(state["extra_state"].get("torch_rng_state", torch.get_rng_state()))

        dist.barrier()

        result = {
            "checkpoint_path": checkpoint_path,
            "step": self.global_step,
            "load_optimizer": load_optimizer,
            "load_time": time.time() - start_time,
            "success": True,
        }

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return result

    # ------------------------------------------------------------------
    # HuggingFace-compatible weight saving for inference / sampling
    # ------------------------------------------------------------------

    def save_weights_for_sampler(
        self, checkpoint_path: str, output_path: str, save_dtype: str = "bfloat16"
    ) -> Dict[str, Any]:
        """
        Save model weights in HuggingFace format for inference/sampling.

        Args:
            checkpoint_path: Path to the distributed checkpoint
            output_path: Path to save the HF-compatible weights
            save_dtype: Data type for saved weights

        Returns:
            Dictionary with save status
        """
        start_time = time.time()

        # Convert distributed checkpoint to state dict
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=checkpoint_path,
            output_dir=output_path,
            ckpt_manager="dcp",
        )

        # Save model assets if available
        model_assets = getattr(self, "model_assets", None)

        # Save in HF format
        save_model_weights(
            output_dir=output_path,
            state_dict=model_state_dict,
            global_rank=self.global_rank,
            save_dtype=save_dtype,
            model_assets=model_assets,
        )

        dist.barrier()

        result = {
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "save_dtype": save_dtype,
            "save_time": time.time() - start_time,
        }

        logger.info(f"HF-compatible weights saved to {output_path}")
        return result
