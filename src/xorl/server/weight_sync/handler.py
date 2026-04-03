"""
Weight Sync Handler — manages weight synchronization with inference endpoints.

Entry point: handle_sync_inference_weights (called on all training ranks).

Flow (nccl_broadcast, streaming per-layer):
1. Rank 0: health-check → create NCCL group → pause inference
2. For each PP stage (sequential; non-PP has only stage 0):
   a. Per FSDP module in this stage (intra-stage collective):
      - All stage ranks: unshard() → QLoRA collective ops → reshard()
      - Stage leader extracts params to bf16 buffer
   b. Stage 0: rank 0 quantizes + broadcasts to SGLang directly
   c. Stage 1+: stage leader sends bf16 buffer to rank 0 via per-module
      NCCL broadcast on pp_group (metadata via pickle, tensors via NCCL).
      Rank 0 quantizes and broadcasts to SGLang.
   d. Barrier between stages
3. Rank 0: resume inference → destroy NCCL group
4. All ranks: final barrier

With PP, stages are processed sequentially so rank 0 can stream each stage's
params to SGLang.  FSDP shard groups are independent per stage.  Inter-stage
transfer uses pp_group (one rank per stage at the same dp_shard position).
Only metadata (names/shapes) is pickled; the actual tensor data is transferred
as a flat bf16 NCCL broadcast — same speed as intra-stage all-gather.

This keeps GPU memory to ~1-2 layers (one unsharding + one broadcasting).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)

# Default bucket size for MoE expert broadcasting (256 MB)
_DEFAULT_MOE_BUCKET_BYTES = 256 * 1024 * 1024


def _prod(shape) -> int:
    """Product of a shape tuple."""
    r = 1
    for d in shape:
        r *= d
    return r


class WeightSyncHandler:
    """Handles weight synchronization between training and inference endpoints."""

    def __init__(self, rank: int, world_size: int, trainer) -> None:
        self.rank = rank
        self.world_size = world_size
        self.trainer = trainer

    # ========================================================================
    # Main entry point
    # ========================================================================

    async def handle_sync_inference_weights(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle sync inference weights request (all ranks participate).

        The ``sync_method`` field selects the transport backend.  Currently
        supported: ``"nccl_broadcast"``.  New backends (RDMA, multi-rank NCCL,
        etc.) can be added by implementing :class:`WeightTransportBackend` and
        registering in :func:`backends.create_backend`.
        """
        logger.info(f"Rank {self.rank}: [WeightSync] Starting sync_inference_weights")

        from xorl.server.protocol.operations import SyncWeightsData

        p: SyncWeightsData = command_dict.get("payload", SyncWeightsData())

        endpoints = p.endpoints
        master_address = p.master_address
        master_port = p.master_port
        group_name = p.group_name
        buffer_size_mb = p.buffer_size_mb
        sync_method = p.sync_method
        flush_cache = p.flush_cache
        pause_mode = p.pause_mode
        weight_version = p.weight_version
        quantization = p.quantization

        logger.info(
            f"Rank {self.rank}: [WeightSync] sync_method={sync_method}, "
            f"endpoints={len(endpoints)}, flush_cache={flush_cache}, "
            f"weight_version={weight_version}, quantization={quantization}"
        )

        try:
            result = self._sync_weights(
                endpoints=endpoints,
                master_address=master_address,
                master_port=master_port,
                group_name=group_name,
                buffer_size_mb=buffer_size_mb,
                sync_method=sync_method,
                flush_cache=flush_cache,
                pause_mode=pause_mode,
                weight_version=weight_version,
                quantization=quantization,
            )

            return result

        except Exception as e:
            logger.error(f"Rank {self.rank}: sync_inference_weights failed: {e}", exc_info=True)
            return {"success": False, "message": f"Weight sync failed: {str(e)}"}

    # ========================================================================
    # Streaming per-layer weight sync (backend-agnostic)
    # ========================================================================

    def _sync_weights(
        self,
        endpoints: List[Dict[str, Any]],
        master_address: str,
        master_port: int,
        group_name: str,
        buffer_size_mb: int,
        sync_method: str,
        flush_cache: bool,
        pause_mode: str,
        weight_version: Optional[str],
        quantization: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Streaming per-layer weight sync using a pluggable transport backend.

        Instead of materializing the entire model on CPU, this:
        1. Unshards one FSDP module at a time (all-gather to GPU)
        2. Clones params to a buffer on GPU
        3. Reshards to free memory
        4. Sends the buffer via the transport backend
        5. Pipelines: unshard(N+1) overlaps with transfer(N)
        """
        from torch.distributed.fsdp._fully_shard import FSDPModule

        model = self.trainer.model
        device = f"cuda:{self.trainer.local_rank}"

        # ------------------------------------------------------------------
        # Step 1: Preparation (all ranks)
        # ------------------------------------------------------------------
        logger.info(f"Rank {self.rank}: [WeightSync] Clearing GPU memory...")
        if hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
            try:
                self.trainer.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.trainer.optimizer.zero_grad()
        torch.cuda.empty_cache()

        if self.rank == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Rank {self.rank}: [WeightSync] GPU memory: {allocated:.2f}GB allocated")

        # ------------------------------------------------------------------
        # Step 2: Discover FSDP modules (all ranks, no communication)
        # ------------------------------------------------------------------
        root_module, layer_modules = self._get_fsdp_modules(model)
        total_modules = (1 if root_module else 0) + len(layer_modules)
        logger.info(
            f"Rank {self.rank}: [WeightSync] Found {total_modules} FSDP modules "
            f"(root={'yes' if root_module else 'no'}, layers={len(layer_modules)})"
        )

        # ------------------------------------------------------------------
        # Step 3: Create backend + endpoint manager, initialize on sender ranks
        # ------------------------------------------------------------------
        from xorl.server.weight_sync.backends import TransportConfig, create_backend
        from xorl.server.weight_sync.backends.base import EndpointConfig
        from xorl.server.weight_sync.endpoint_manager import EndpointManager

        transport_cfg = TransportConfig(
            endpoints=[
                EndpointConfig(
                    host=ep["host"],
                    port=ep["port"],
                    world_size=ep.get("world_size", 1),
                )
                for ep in endpoints
            ],
            master_address=master_address,
            master_port=master_port,
            group_name=group_name,
            buffer_size_mb=buffer_size_mb,
            device=device,
            training_world_size=self.world_size,
            training_rank=self.rank,
        )

        backend = create_backend(sync_method, transport_cfg)
        _is_sender = self.rank in backend.sender_ranks

        # Endpoint management lives on rank 0 (coordinator).  Future multi-rank
        # backends still designate one rank for HTTP pause/resume coordination.
        endpoint_mgr = EndpointManager(endpoints) if self.rank == 0 else None

        if self.rank == 0:
            if not endpoints:
                return {"success": False, "message": "No endpoints provided"}
            endpoint_mgr.health_check()

        # Backend init: all sender ranks participate (collective for NCCL).
        if _is_sender:
            logger.info(f"Rank {self.rank}: [WeightSync] Initializing {sync_method} backend...")
            if not backend.initialize():
                return {
                    "success": False,
                    "message": f"Failed to initialize {sync_method} backend",
                }
            logger.info(f"Rank {self.rank}: [WeightSync] Backend initialized")

        # Pause inference: coordinator only (after backend init).
        if self.rank == 0:
            logger.info(f"Rank {self.rank}: [WeightSync] Pausing inference (mode={pause_mode})...")
            pause_results, all_paused = endpoint_mgr.pause(pause_mode)
            if not all_paused:
                endpoint_mgr.resume()
                if _is_sender:
                    backend.destroy()
                return {
                    "success": False,
                    "message": f"Failed to pause inference endpoints: {pause_results}",
                }

        # ------------------------------------------------------------------
        # Step 4: Stream weights — per-module unshard + broadcast pipeline
        # ------------------------------------------------------------------
        start_time = time.perf_counter()
        total_bytes = 0
        total_params = 0
        num_buckets = 0

        # Build ordered list of FSDP modules to process
        modules_to_sync: List[Tuple[str, FSDPModule]] = []
        if root_module is not None:
            modules_to_sync.append(("(root)", root_module))
        modules_to_sync.extend(layer_modules)

        # Detect EP mode
        from xorl.distributed.parallel_state import get_parallel_state

        _ps = get_parallel_state()
        _ep_enabled = _ps.ep_enabled and _ps.ep_size > 1

        # Detect PP mode (Pipeline Parallelism)
        # With PP, each stage has an independent FSDP shard group.  We process
        # stages sequentially so that rank 0 can stream each stage's params to
        # SGLang.  For remote stages (>0), the stage leader sends each module's
        # params to rank 0 via a per-module NCCL broadcast on pp_group, keeping
        # memory bounded to one module at a time.
        _pp_enabled = _ps.pp_enabled
        _pp_rank = _ps.pp_rank if _pp_enabled else 0
        _pp_size = _ps.pp_size if _pp_enabled else 1

        try:
            for pp_stage in range(_pp_size):
                _is_my_stage = _pp_rank == pp_stage
                _is_remote = _pp_enabled and pp_stage > 0
                # Stage leader: the dp_shard_rank==0 rank within this stage
                _stage_leader = _ps.dp_shard_rank == 0 if _pp_enabled else (self.rank == 0)

                stage_modules = modules_to_sync if _is_my_stage else []

                # For remote stages: tell rank 0 how many modules to expect
                pp_grp = None
                stage_src = 0
                num_stage_modules = len(stage_modules)
                if _is_remote and _ps.dp_shard_rank == 0:
                    pp_grp = _ps.pp_group
                    stage_src = dist.get_global_rank(pp_grp, pp_stage)
                    obj = [num_stage_modules] if _is_my_stage else [None]
                    dist.broadcast_object_list(obj, src=stage_src, group=pp_grp)
                    num_stage_modules = obj[0]

                if _is_my_stage:
                    logger.info(
                        f"Rank {self.rank}: [WeightSync] Processing PP stage {pp_stage} "
                        f"({num_stage_modules} modules, remote={_is_remote})"
                    )

                for mod_idx in range(num_stage_modules):
                    is_last_overall = mod_idx == num_stage_modules - 1 and pp_stage == _pp_size - 1

                    # ── FSDP ops (only ranks owning this stage) ──────────
                    current_buffer = None
                    moe_contexts = []
                    ep_moe_contexts = []

                    if _is_my_stage:
                        mod_name, fsdp_mod = stage_modules[mod_idx]

                        fsdp_mod.unshard()

                        qlora_linear_buffer, moe_contexts = self._qlora_collective_ops(
                            fsdp_mod,
                            mod_name,
                            collect_results=_stage_leader,
                        )

                        if _ep_enabled:
                            ep_moe_contexts = self._collect_ep_moe_data(
                                fsdp_mod,
                                mod_name,
                                _ps,
                            )

                        # EP MoE prefixes to skip in extraction
                        ep_moe_prefixes = set()
                        for ctx in ep_moe_contexts:
                            p = ctx["prefix"]
                            if mod_name != "(root)":
                                if p == mod_name:
                                    ep_moe_prefixes.add("")
                                elif p.startswith(mod_name + "."):
                                    ep_moe_prefixes.add(p[len(mod_name) + 1 :])
                            else:
                                ep_moe_prefixes.add(p)

                        if _stage_leader:
                            if ep_moe_prefixes:
                                logger.info(
                                    f"Rank {self.rank}: [WeightSync] ep_moe_prefixes={ep_moe_prefixes} for {mod_name}"
                                )
                            from torch.distributed._tensor import DTensor

                            current_buffer = self._extract_params_for_sync(
                                fsdp_mod,
                                mod_name,
                                DTensor,
                                skip_moe_prefixes=ep_moe_prefixes,
                            )
                            current_buffer.extend(qlora_linear_buffer)
                        del qlora_linear_buffer

                        fsdp_mod.reshard()

                    # ── Transfer / broadcast to SGLang ───────────────────
                    if not _is_remote:
                        # Stage 0: sender rank(s) broadcast directly to SGLang
                        if _is_sender and current_buffer:
                            current_buffer = self._unfuse_for_inference(
                                current_buffer,
                                model,
                            )
                            if quantization and quantization.get("quant_method") == "fp8":
                                current_buffer = self._quantize_buffer_for_fp8(
                                    current_buffer,
                                    quantization_config=quantization,
                                )
                            logger.info(f"Rank 0: [WeightSync] Module {mod_name}: {len(current_buffer)} params")
                            b, p = self._broadcast_buffer(
                                backend,
                                current_buffer,
                                flush_cache=(flush_cache and is_last_overall and not moe_contexts),
                                weight_version=weight_version if is_last_overall and not moe_contexts else None,
                            )
                            total_bytes += b
                            total_params += p
                            num_buckets += 1
                            del current_buffer

                        # Stage 0 MoE handling (unchanged)
                        if moe_contexts or ep_moe_contexts:
                            if _ep_enabled:
                                for ctx in moe_contexts + ep_moe_contexts:
                                    b, p, n = self._gather_and_broadcast_ep_moe_experts(
                                        backend,
                                        ctx,
                                        flush_cache=(flush_cache and is_last_overall),
                                        weight_version=weight_version if is_last_overall else None,
                                        quantization=quantization,
                                        ps=_ps,
                                    )
                                    total_bytes += b
                                    total_params += p
                                    num_buckets += n
                            elif _is_sender:
                                for ctx in moe_contexts:
                                    b, p, n = self._broadcast_moe_experts_bucketed(
                                        backend,
                                        ctx,
                                        flush_cache=(flush_cache and is_last_overall),
                                        weight_version=weight_version if is_last_overall else None,
                                        quantization=quantization,
                                    )
                                    total_bytes += b
                                    total_params += p
                                    num_buckets += n
                    else:
                        # Remote stage: per-module NCCL transfer to rank 0
                        if _ps.dp_shard_rank == 0:
                            # Prepare send buffer on stage leader (bf16, not quantized)
                            send_buf = None
                            if _is_my_stage and _stage_leader:
                                send_buf = current_buffer if current_buffer else []
                                send_buf = self._unfuse_for_inference(send_buf, model)
                                # Include MoE experts in the same buffer
                                for ctx in moe_contexts:
                                    expert_items = self._compute_moe_experts_buffer(ctx)
                                    send_buf.extend(expert_items)

                            # NCCL transfer: metadata + flat bf16 tensor
                            received = self._pp_nccl_transfer_buffer(
                                send_buf if (_is_my_stage and _stage_leader) else None,
                                pp_grp,
                                stage_src,
                                device,
                            )

                            # Rank 0: quantize + broadcast to SGLang
                            if self.rank == 0 and received:
                                if quantization and quantization.get("quant_method") == "fp8":
                                    received = self._quantize_buffer_for_fp8(
                                        received,
                                        quantization_config=quantization,
                                    )
                                logger.info(
                                    f"Rank 0: [WeightSync] PP stage {pp_stage} module "
                                    f"{mod_idx}: {len(received)} params via NCCL"
                                )
                                b, p = self._broadcast_buffer(
                                    backend,
                                    received,
                                    flush_cache=(flush_cache and is_last_overall),
                                    weight_version=weight_version if is_last_overall else None,
                                )
                                total_bytes += b
                                total_params += p
                                num_buckets += 1
                                del received

                # Barrier between PP stages (all ranks)
                if _pp_enabled:
                    dist.barrier()

            transfer_time = time.perf_counter() - start_time

            # ------------------------------------------------------------------
            # Step 5: Resume inference, cleanup
            # ------------------------------------------------------------------
            if self.rank == 0:
                throughput = (total_bytes / transfer_time / (1024**3)) if transfer_time > 0 else 0
                logger.info(
                    f"Rank {self.rank}: [WeightSync] Transfer complete: "
                    f"{transfer_time:.2f}s, {throughput:.2f} GB/s, "
                    f"{total_bytes / 1e9:.2f} GB, {total_params} params, "
                    f"{num_buckets} buckets"
                )
                endpoint_mgr.resume()
            if _is_sender:
                backend.destroy()

            return {
                "success": True,
                "message": f"Synced {total_params} params to {len(endpoints)} endpoint(s)",
                "transfer_time": time.perf_counter() - start_time,
                "total_bytes": total_bytes,
                "num_parameters": total_params,
                "num_buckets": num_buckets,
                "endpoint_results": [{"host": ep["host"], "port": ep["port"], "success": True} for ep in endpoints],
            }

        except Exception:
            if endpoint_mgr is not None:
                try:
                    endpoint_mgr.resume()
                except Exception as resume_err:
                    logger.warning(f"Rank 0: [WeightSync] Failed to resume inference during cleanup: {resume_err}")
            if _is_sender:
                try:
                    backend.destroy()
                except Exception as destroy_err:
                    logger.warning(
                        f"Rank {self.rank}: [WeightSync] Failed to destroy backend during cleanup: {destroy_err}"
                    )
            raise

    # ========================================================================
    # QLoRA collective operations (all ranks must call)
    # ========================================================================

    def _qlora_collective_ops(
        self,
        fsdp_mod,
        mod_name: str,
        collect_results: Optional[bool] = None,
    ) -> Tuple[List[Tuple[str, torch.Tensor]], List[Dict[str, Any]]]:
        """QLoRA collective ops — ALL ranks must call (collective full_tensor).

        Must be called between unshard() and reshard().

        Phase 1 (QLoRALinear): Dequantize base weights + compute LoRA delta.
          Both packed_weight_f32 and lora_A/lora_B are sharded DTensors that
          require collective full_tensor() on all ranks. The stage leader merges
          and returns the bf16 weight; other ranks return empty.

        Phase 2 (QLoRAMoeExperts): Gather LoRA params only.
          LoRA params may be FSDP-managed DTensors (when EP is disabled).
          All ranks call full_tensor() collectively. The stage leader clones the
          full params for later per-expert processing. Quantized buffers are plain
          tensors — they don't need collective ops and can be accessed after
          reshard.

        Args:
            collect_results: if True, this rank collects merged weights/contexts.
                Defaults to self.rank == 0 (backward-compatible).

        Returns:
            (linear_buffer, moe_contexts):
              linear_buffer: list of (name, tensor) for QLoRALinear merged weights
              moe_contexts: list of dicts with info for per-expert processing
        """
        if collect_results is None:
            collect_results = self.rank == 0
        # Discover QLoRA modules (only those directly owned by this FSDP module,
        # not by child FSDP modules — child modules will be processed separately)
        from torch.distributed.fsdp._fully_shard import FSDPModule

        from xorl.qlora.modules.linear import QLoRALinear
        from xorl.qlora.modules.moe_experts import QLoRAMoeExperts

        child_fsdp_prefixes = set()
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, FSDPModule) and mname != "":
                child_fsdp_prefixes.add(mname + ".")

        qlora_linears = {}
        qlora_moe = {}
        for mname, mod in fsdp_mod.named_modules():
            # Skip modules that belong to child FSDP modules
            if any(mname.startswith(p) or mname + "." == p for p in child_fsdp_prefixes):
                continue
            if isinstance(mod, QLoRALinear):
                qlora_linears[mname] = mod
            elif isinstance(mod, QLoRAMoeExperts):
                qlora_moe[mname] = mod

        if not qlora_linears and not qlora_moe:
            return [], []

        linear_buffer: List[Tuple[str, torch.Tensor]] = []
        moe_contexts: List[Dict[str, Any]] = []

        # --- Phase 1: QLoRALinear (all ranks participate in collective ops) ---
        for mname, mod in qlora_linears.items():
            full_prefix = f"{mod_name}.{mname}" if mod_name != "(root)" else mname

            # All ranks: dequantize base weight (collective full_tensor on packed_weight_f32)
            # Note: if Hadamard rotation is enabled, dequantized weight is in rotated space
            base_w = mod._dequantize_weight().to(dtype=torch.bfloat16)
            # All ranks: compute delta (collective full_tensor on lora_A, lora_B)
            delta = mod.get_delta_weight().to(dtype=torch.bfloat16)

            if collect_results:
                merged = base_w + delta
                linear_buffer.append((f"{full_prefix}.weight", merged))
            del base_w, delta

        # --- Phase 2: QLoRAMoeExperts (gather lora params, defer heavy work) ---
        from xorl.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        ep_enabled = ps.ep_enabled and ps.ep_size > 1

        for mname, mod in qlora_moe.items():
            full_prefix = f"{mod_name}.{mname}" if mod_name != "(root)" else mname

            # All ranks: gather lora params (collective if DTensor)
            # EP mode: each rank keeps LOCAL lora params (no need for full_tensor,
            #   each rank merges its own local experts with local LoRA shards)
            # Non-EP: rank 0 gathers full params via full_tensor for deferred processing
            lora_params = {}
            for hf_proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("lora_A", "lora_B"):
                    key = f"{hf_proj}_{suffix}"
                    param = getattr(mod, key)
                    if ep_enabled:
                        # EP: use local shard directly (no collective needed)
                        if hasattr(param, "to_local"):
                            lora_params[key] = param.to_local().clone()
                        else:
                            lora_params[key] = param.data.clone()
                    else:
                        # Non-EP: collective full_tensor to get global params on stage leader
                        if hasattr(param, "full_tensor"):
                            full = param.full_tensor()  # collective: all ranks
                        else:
                            full = param.data
                        if collect_results:
                            lora_params[key] = full.clone()
                        del full

            ctx = {
                "module": mod,
                "prefix": full_prefix,
                "lora_params": lora_params,
            }
            if ep_enabled:
                moe_contexts.append(ctx)  # all ranks keep context
            elif collect_results:
                moe_contexts.append(ctx)  # only stage leader

        return linear_buffer, moe_contexts

    # ========================================================================
    # EP MoE data collection (all ranks, during unshard)
    # ========================================================================

    def _collect_ep_moe_data(
        self,
        fsdp_mod,
        mod_name: str,
        ps,
    ) -> List[Dict[str, Any]]:
        """Collect local EP-sharded MoE expert data during unshard phase.

        Identifies full-weight MoE modules (MoEExperts, MoEExpertsLoRA) whose
        expert params are EP-sharded DTensors. Clones local expert data for
        later EP gathering after reshard.

        QLoRAMoeExperts are handled separately by _qlora_collective_ops.

        Must be called between unshard() and reshard().
        Returns list of context dicts for _gather_and_broadcast_ep_moe_experts.
        """
        from torch.distributed._tensor import DTensor

        # Skip modules owned by child FSDP modules (processed separately)
        from torch.distributed.fsdp._fully_shard import FSDPModule

        from xorl.models.layers.moe.experts import MoEExperts
        from xorl.models.layers.moe.lora import MoEExpertsLoRA
        from xorl.qlora.modules.moe_experts import QLoRAMoeExperts

        child_fsdp_prefixes = set()
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, FSDPModule) and mname != "":
                child_fsdp_prefixes.add(mname + ".")

        contexts = []
        for mname, mod in fsdp_mod.named_modules():
            if any(mname.startswith(p) or mname + "." == p for p in child_fsdp_prefixes):
                continue
            # Skip QLoRA MoE (handled by _qlora_collective_ops)
            if isinstance(mod, QLoRAMoeExperts):
                continue
            if not isinstance(mod, (MoEExperts, MoEExpertsLoRA)):
                continue

            # Get expert params — after unshard they may be plain tensors or DTensors
            gate_up = getattr(mod, "gate_up_proj", None)
            if isinstance(gate_up, torch.nn.Parameter):
                E_local = gate_up.shape[0]
            else:
                gate = getattr(mod, "gate_proj", None)
                if gate is None or not isinstance(gate, torch.nn.Parameter):
                    continue
                E_local = gate.shape[0]

            if mname:
                full_prefix = f"{mod_name}.{mname}" if mod_name != "(root)" else mname
            else:
                full_prefix = mod_name

            # Clone local expert data for each projection.
            # With EP, each rank's module already holds only local experts [E_local, K, N].
            local_experts = {}
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                param = getattr(mod, proj_name)
                if isinstance(param.data, DTensor):
                    local = param.to_local()
                else:
                    local = param.data

                # Merge LoRA if applicable
                if isinstance(mod, MoEExpertsLoRA):
                    if proj_name in mod.lora_config.target_modules:
                        delta = mod._compute_proj_delta(proj_name)
                        if isinstance(delta, DTensor):
                            delta = delta.to_local()
                        local = local.to(torch.bfloat16) + delta.to(torch.bfloat16)
                    else:
                        local = local.to(torch.bfloat16)
                else:
                    local = local.to(torch.bfloat16)

                local_experts[proj_name] = local.clone()

            contexts.append(
                {
                    "type": "full_weight",
                    "prefix": full_prefix,
                    "local_experts": local_experts,  # {proj: [E_local, K, N]}
                    "num_local_experts": E_local,
                }
            )

        return contexts

    # ========================================================================
    # EP-aware MoE expert gathering and broadcasting (all ranks)
    # ========================================================================

    def _gather_and_broadcast_ep_moe_experts(
        self,
        backend,
        ctx: Dict[str, Any],
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
        bucket_size_bytes: int = _DEFAULT_MOE_BUCKET_BYTES,
        quantization: Optional[Dict[str, Any]] = None,
        ps=None,
    ) -> Tuple[int, int, int]:
        """EP-aware MoE expert sync using per-projection dist.gather.

        With EP=N, each rank holds E_local = E_total/N experts. For each
        projection (gate, up, down):
        1. ALL ranks: prepare local expert weights [E_local, N, K]
        2. ALL ranks: dist.gather → rank 0 gets [E_total, N, K]
        3. Rank 0: split into per-expert HF weights, quantize, broadcast

        Handles both full-weight MoE (via local_experts in ctx) and
        QLoRA MoE (via module dequant + local LoRA merge).

        Uses 3 gather calls per MoE layer (one per projection).
        Peak memory on rank 0: ~one projection's worth of all experts.

        ALL ranks must call this (collective gather).
        """
        full_prefix = ctx["prefix"]
        ep_group = ps.ep_group
        ep_size = ps.ep_size
        device = f"cuda:{self.rank % torch.cuda.device_count()}"

        is_qlora = ctx.get("type") != "full_weight"
        ep_fsdp_rank = 0
        if ps.ep_fsdp_device_mesh is not None:
            ep_fsdp_rank = ps.ep_fsdp_device_mesh.get_local_rank("ep_fsdp")

        # EP-FSDP replicas hold identical local expert shards after unshard().
        # Only one replica column needs to gather and forward those experts to
        # rank 0; the others would just duplicate the same weights.
        if ep_fsdp_rank != 0:
            if is_qlora:
                ctx["lora_params"] = None
            else:
                ctx["local_experts"] = None
            return 0, 0, 0

        if self.rank == 0:
            logger.info(
                f"Rank 0: [EP-Gather] prefix={full_prefix}, type={'qlora' if is_qlora else 'full_weight'}, "
                f"ep_size={ep_size}"
            )

        if is_qlora:
            mod = ctx["module"]
            lora_params = ctx["lora_params"]
            E_local = mod.num_local_experts
            projections = [
                ("gate", "gate_proj", mod.hidden_size, mod.intermediate_size),
                ("up", "up_proj", mod.hidden_size, mod.intermediate_size),
                ("down", "down_proj", mod.intermediate_size, mod.hidden_size),
            ]
        else:
            local_experts = ctx["local_experts"]
            E_local = ctx["num_local_experts"]
            projections = [
                ("gate", "gate_proj", None, None),
                ("up", "up_proj", None, None),
                ("down", "down_proj", None, None),
            ]

        E_total = E_local * ep_size

        total_bytes = 0
        total_params = 0
        num_buckets = 0
        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        for proj_name, hf_proj, K, N in projections:
            # Each rank: prepare local expert weights in HF format [E_local, N, K]
            if is_qlora:
                lora_A = lora_params[f"{hf_proj}_lora_A"]  # [1 or E_local, K, r]
                lora_B = lora_params[f"{hf_proj}_lora_B"]  # [E_local or 1, r, N]
                local_merged = []
                for i in range(E_local):
                    base_w = mod.dequantize_expert(proj_name, i, K, N)
                    a_idx = min(i, lora_A.shape[0] - 1)
                    b_idx = min(i, lora_B.shape[0] - 1)
                    delta = (lora_A[a_idx] @ lora_B[b_idx]) * mod.scaling
                    merged = base_w.to(torch.bfloat16) + delta.to(torch.bfloat16)
                    local_merged.append(merged.t().contiguous())  # [N, K]
                    del base_w, delta, merged
                local_stack = torch.stack(local_merged, dim=0).to(device)
                del local_merged
            else:
                # Full-weight: already cloned as [E_local, K, N] bf16
                local_data = local_experts[hf_proj]  # [E_local, K, N]
                # Transpose each expert to HF [N, K]
                local_stack = local_data.permute(0, 2, 1).contiguous().to(device)
                del local_data

            # Gather from all EP ranks → rank 0 gets [E_total, N, K]
            if self.rank == 0:
                gathered = [torch.empty_like(local_stack) for _ in range(ep_size)]
            else:
                gathered = None
            dist.gather(local_stack, gathered, dst=0, group=ep_group)
            del local_stack

            # Rank 0: split into per-expert (name, tensor) pairs and bucket
            if self.rank == 0:
                all_experts = torch.cat(gathered, dim=0)  # [E_total, N, K]
                if hf_proj == "gate_proj":
                    logger.info(
                        f"Rank 0: [EP-Gather] {full_prefix}: all_experts={list(all_experts.shape)}, "
                        f"E_local={E_local}, E_total={E_total}, bucket_len={len(bucket)}"
                    )
                del gathered
                for global_idx in range(E_total):
                    hf_name = f"{full_prefix}.{global_idx}.{hf_proj}.weight"
                    expert_weight = all_experts[global_idx]
                    weight_bytes = expert_weight.numel() * expert_weight.element_size()
                    bucket.append((hf_name, expert_weight))
                    bucket_bytes += weight_bytes

                    # Flush bucket when full
                    if bucket_bytes >= bucket_size_bytes:
                        if quantization and quantization.get("quant_method") == "fp8":
                            bucket = self._quantize_buffer_for_fp8(
                                bucket,
                                quantization_config=quantization,
                            )
                        b, p = self._broadcast_buffer(
                            backend,
                            bucket,
                            flush_cache=False,
                        )
                        total_bytes += b
                        total_params += p
                        num_buckets += 1
                        bucket = []
                        bucket_bytes = 0

                del all_experts

        # Flush remaining bucket
        if self.rank == 0 and bucket:
            if quantization and quantization.get("quant_method") == "fp8":
                bucket = self._quantize_buffer_for_fp8(
                    bucket,
                    quantization_config=quantization,
                )
            b, p = self._broadcast_buffer(
                backend,
                bucket,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
            total_bytes += b
            total_params += p
            num_buckets += 1

        # Free cloned data
        if is_qlora:
            ctx["lora_params"] = None
        else:
            ctx["local_experts"] = None

        return total_bytes, total_params, num_buckets

    # ========================================================================
    # QLoRA MoE expert bucketed broadcasting (rank 0 only, non-EP)
    # ========================================================================

    def _broadcast_moe_experts_bucketed(
        self,
        backend,
        ctx: Dict[str, Any],
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
        bucket_size_bytes: int = _DEFAULT_MOE_BUCKET_BYTES,
        quantization: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Process QLoRA MoE experts per-expert and broadcast in buckets.

        Called AFTER reshard on rank 0 only. Dequantizes one expert at a time,
        computes per-expert LoRA delta, merges to bf16, and accumulates into
        a bucket. When the bucket reaches bucket_size_bytes, it's broadcast
        via NCCL and cleared.

        This keeps peak memory to ~bucket_size instead of materializing all
        128×3 expert weights simultaneously.

        Args:
            backend: WeightTransportBackend for transferring weight buckets
            ctx: dict with 'module', 'prefix', 'lora_params'
            flush_cache: whether to flush inference KV cache on last bucket
            bucket_size_bytes: max bucket size before broadcasting

        Returns:
            (total_bytes, total_params, num_buckets)
        """
        mod = ctx["module"]
        full_prefix = ctx["prefix"]
        lora_params = ctx["lora_params"]

        total_bytes = 0
        total_params = 0
        num_buckets = 0
        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        projections = [
            ("gate", "gate_proj", mod.hidden_size, mod.intermediate_size),
            ("up", "up_proj", mod.hidden_size, mod.intermediate_size),
            ("down", "down_proj", mod.intermediate_size, mod.hidden_size),
        ]

        E = mod.num_local_experts

        for proj_name, hf_proj, K, N in projections:
            lora_A = lora_params[f"{hf_proj}_lora_A"]  # [1 or E, K, r]
            lora_B = lora_params[f"{hf_proj}_lora_B"]  # [E or 1, r, N]

            for expert_idx in range(E):
                # Dequantize single expert (plain tensor, no collective needed)
                base_w = mod.dequantize_expert(proj_name, expert_idx, K, N)  # [K, N]

                # Per-expert LoRA delta: A[i] @ B[i] * scaling → [K, N]
                a_idx = min(expert_idx, lora_A.shape[0] - 1)
                b_idx = min(expert_idx, lora_B.shape[0] - 1)
                delta = (lora_A[a_idx] @ lora_B[b_idx]) * mod.scaling  # [K, N]

                merged = base_w.to(torch.bfloat16) + delta.to(torch.bfloat16)
                del base_w, delta

                # HF format: [N, K] (transpose from GKN)
                hf_name = f"{full_prefix}.{expert_idx}.{hf_proj}.weight"
                expert_weight = merged.t().contiguous()
                del merged

                weight_bytes = expert_weight.numel() * expert_weight.element_size()
                bucket.append((hf_name, expert_weight))
                bucket_bytes += weight_bytes

                # Flush bucket when full
                if bucket_bytes >= bucket_size_bytes:
                    if quantization and quantization.get("quant_method") == "fp8":
                        bucket = self._quantize_buffer_for_fp8(
                            bucket,
                            quantization_config=quantization,
                        )
                    b, p = self._broadcast_buffer(
                        backend,
                        bucket,
                        flush_cache=False,
                    )
                    total_bytes += b
                    total_params += p
                    num_buckets += 1
                    bucket = []
                    bucket_bytes = 0

        # Flush remaining bucket
        if bucket:
            if quantization and quantization.get("quant_method") == "fp8":
                bucket = self._quantize_buffer_for_fp8(
                    bucket,
                    quantization_config=quantization,
                )
            b, p = self._broadcast_buffer(
                backend,
                bucket,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
            total_bytes += b
            total_params += p
            num_buckets += 1

        # Free cloned lora params
        del lora_params
        ctx["lora_params"] = None

        return total_bytes, total_params, num_buckets

    # ========================================================================
    # PP inter-stage NCCL transfer helpers
    # ========================================================================

    def _pp_nccl_transfer_buffer(
        self,
        send_buffer: Optional[List[Tuple[str, torch.Tensor]]],
        pp_grp,
        src_global: int,
        device: str,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """Transfer named tensor buffer from one PP stage leader to others via NCCL.

        ALL pp_group members must call this (collective broadcast).

        The source rank (src_global) provides send_buffer: a list of
        (name, bf16_tensor) pairs. Other members pass None.

        Protocol:
          1. broadcast_object_list: metadata [(name, shape), ...] — tiny, pickle OK
          2. dist.broadcast: single flat bf16 tensor — all weights concatenated

        Returns:
            For non-source ranks: list of (name, tensor) on device
            For source rank: None
        """
        is_src = self.rank == src_global

        # Step 1: broadcast metadata (names + shapes)
        if is_src:
            meta = [(n, list(t.shape)) for n, t in send_buffer] if send_buffer else []
            obj = [meta]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=src_global, group=pp_grp)
        meta = obj[0]

        if not meta:
            return [] if not is_src else None

        # Step 2: broadcast flat bf16 tensor via NCCL
        if is_src:
            flat = torch.cat([t.to(dtype=torch.bfloat16).reshape(-1) for _, t in send_buffer])
        else:
            total_el = sum(_prod(s) for _, s in meta)
            flat = torch.empty(total_el, dtype=torch.bfloat16, device=device)
        dist.broadcast(flat, src=src_global, group=pp_grp)

        if is_src:
            del flat
            return None

        # Split flat tensor back into named tensors
        result = []
        offset = 0
        for name, shape in meta:
            numel = _prod(shape)
            result.append((name, flat[offset : offset + numel].view(shape)))
            offset += numel
        return result

    def _compute_moe_experts_buffer(
        self,
        ctx: Dict[str, Any],
    ) -> List[Tuple[str, torch.Tensor]]:
        """Compute merged QLoRA MoE expert weights as bf16 GPU tensors.

        Used by PP follower stage leaders to include MoE expert weights in the
        per-module NCCL transfer buffer (bf16, NOT quantized — rank 0 quantizes
        after receiving).

        Args:
            ctx: dict with 'module', 'prefix', 'lora_params' (from _qlora_collective_ops)

        Returns:
            list of (name, bf16_gpu_tensor) pairs
        """
        mod = ctx["module"]
        full_prefix = ctx["prefix"]
        lora_params = ctx["lora_params"]
        E = mod.num_local_experts

        projections = [
            ("gate", "gate_proj", mod.hidden_size, mod.intermediate_size),
            ("up", "up_proj", mod.hidden_size, mod.intermediate_size),
            ("down", "down_proj", mod.intermediate_size, mod.hidden_size),
        ]

        items: List[Tuple[str, torch.Tensor]] = []
        for proj_name, hf_proj, K, N in projections:
            lora_A = lora_params[f"{hf_proj}_lora_A"]
            lora_B = lora_params[f"{hf_proj}_lora_B"]
            for expert_idx in range(E):
                base_w = mod.dequantize_expert(proj_name, expert_idx, K, N)
                a_idx = min(expert_idx, lora_A.shape[0] - 1)
                b_idx = min(expert_idx, lora_B.shape[0] - 1)
                delta = (lora_A[a_idx] @ lora_B[b_idx]) * mod.scaling
                merged = (base_w.to(torch.bfloat16) + delta.to(torch.bfloat16)).t().contiguous()
                del base_w, delta
                hf_name = f"{full_prefix}.{expert_idx}.{hf_proj}.weight"
                items.append((hf_name, merged))

        logger.info(f"Rank {self.rank}: [WeightSync] MoE {full_prefix}: {len(items)} experts")

        # Free cloned lora params
        del lora_params
        ctx["lora_params"] = None

        return items

    # ========================================================================
    # FP8 quantization
    # ========================================================================

    @staticmethod
    def _quantize_buffer_for_fp8(
        buffer: List[Tuple[str, torch.Tensor]],
        quantization_config: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Quantize bf16 weight tensors to FP8 with block-wise scales.

        For each weight tensor, produces two entries:
        - (name, fp8_weight): quantized weight in the specified FP8 format
        - (name_scale_inv, scale_inv): per-block scale_inv in float32

        Uses quantization_config (HF format) for:
        - fmt: "e4m3" (default) or "e5m2" → torch.float8_e4m3fn or torch.float8_e5m2
        - weight_block_size: [row_block, col_block], default [128, 128]
        - modules_to_not_convert: list of module name prefixes to skip quantization

        Non-weight params (e.g. layernorm, embedding) are passed through as-is.
        """
        if quantization_config is None:
            quantization_config = {}

        # FP8 format: e4m3 (default, higher precision) or e5m2 (wider range)
        fmt = quantization_config.get("fmt", "e4m3")
        if fmt == "e5m2":
            fp8_dtype = torch.float8_e5m2
        else:
            fp8_dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(fp8_dtype).max

        block_size_list = quantization_config.get("weight_block_size", [128, 128])
        block_size_row = block_size_list[0]
        block_size_col = block_size_list[1] if len(block_size_list) > 1 else block_size_list[0]
        modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])

        result = []
        for name, tensor in buffer:
            # Must be a 2D weight tensor to be quantized
            if not (name.endswith(".weight") and tensor.ndim == 2):
                result.append((name, tensor))
                continue

            # Check modules_to_not_convert: match if the param name starts with
            # any entry (prefix match). E.g. "lm_head" matches "lm_head.weight",
            # "model.layers.0.mlp.gate" matches "model.layers.0.mlp.gate.weight"
            if modules_to_not_convert:
                skip = any(
                    name == prefix + ".weight" or name.startswith(prefix + ".") for prefix in modules_to_not_convert
                )
                if skip:
                    result.append((name, tensor))
                    continue
            else:
                # Default skip logic when no explicit list: only quantize _proj weights
                if "_proj.weight" not in name:
                    result.append((name, tensor))
                    continue

            rows, cols = tensor.shape
            # Pad to block_size alignment if needed
            pad_rows = (block_size_row - rows % block_size_row) % block_size_row
            pad_cols = (block_size_col - cols % block_size_col) % block_size_col

            if pad_rows > 0 or pad_cols > 0:
                padded = torch.zeros(
                    rows + pad_rows,
                    cols + pad_cols,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                padded[:rows, :cols] = tensor
            else:
                padded = tensor

            # Reshape into blocks: [nr, block_size_row, nc, block_size_col]
            nr = padded.shape[0] // block_size_row
            nc = padded.shape[1] // block_size_col
            blocks = padded.reshape(nr, block_size_row, nc, block_size_col).permute(0, 2, 1, 3)
            # blocks shape: [nr, nc, block_size_row, block_size_col]

            # Compute per-block scale: max(abs(block)) / fp8_max
            block_max = blocks.abs().reshape(nr, nc, -1).max(dim=-1).values  # [nr, nc]
            scale = block_max.clamp(min=1e-12) / fp8_max  # [nr, nc]
            scale_inv = scale.to(torch.float32)

            # Quantize: divide by scale, clamp, cast to fp8
            # Expand scale for broadcasting: [nr, nc, 1, 1]
            scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)  # [nr, nc, 1, 1]
            quantized_blocks = (blocks.float() / scale_expanded).clamp(-fp8_max, fp8_max)
            quantized_blocks = quantized_blocks.to(fp8_dtype)

            # Reshape back: [nr, nc, block_size, block_size] → [padded_rows, padded_cols]
            quantized = quantized_blocks.permute(0, 2, 1, 3).reshape(padded.shape[0], padded.shape[1])

            # Remove padding
            if pad_rows > 0 or pad_cols > 0:
                quantized = quantized[:rows, :cols].contiguous()

            # scale_inv name: replace .weight with .weight_scale_inv
            scale_name = name.replace(".weight", ".weight_scale_inv")

            result.append((name, quantized))
            result.append((scale_name, scale_inv))

        return result

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _extract_params_for_sync(
        fsdp_mod,
        mod_name: str,
        DTensor,
        skip_moe_prefixes: Optional[set] = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Extract parameters from an unsharded FSDP module for sync.

        Handles LoRA by computing merged weights (base + delta) on-the-fly,
        and skips LoRA-only params (lora_A, lora_B) since inference doesn't
        have them. Also skips DTensor params (belong to child FSDP modules)
        and EP-handled MoE modules (gathered separately).

        Supports both LoraLinear (dense LoRA) and MoEExpertsLoRA (MoE LoRA).
        """
        from xorl.lora.modules.base import LoraModule
        from xorl.lora.modules.linear import LoraLinear
        from xorl.models.layers.moe.experts import MoEExperts
        from xorl.models.layers.moe.lora import MoEExpertsLoRA
        from xorl.qlora.modules.linear import QLoRALinear
        from xorl.qlora.modules.moe_experts import QLoRAMoeExperts

        buffer = []

        # Identify child FSDP module prefixes — their params will be processed
        # when that child module is unsharded separately. Parent unshard may
        # expose child params as plain tensors, so we can't rely on DTensor check.
        from torch.distributed.fsdp._fully_shard import FSDPModule

        child_fsdp_prefixes = set()
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, FSDPModule) and mname != "":
                child_fsdp_prefixes.add(mname + ".")

        # Collect LoRA modules for on-the-fly merging
        lora_modules = {}
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, LoraModule):
                lora_modules[mname] = mod

        # Track which params are LoRA A/B or QLoRA internal so we skip them
        lora_param_names = set()
        for mname, mod in lora_modules.items():
            prefix = f"{mname}." if mname else ""
            if isinstance(mod, QLoRALinear):
                lora_param_names.add(f"{prefix}lora_A")
                lora_param_names.add(f"{prefix}lora_B")
                lora_param_names.add(f"{prefix}packed_weight_f32")
            elif isinstance(mod, LoraLinear):
                lora_param_names.add(f"{prefix}lora_A")
                lora_param_names.add(f"{prefix}lora_B")
            elif isinstance(mod, (QLoRAMoeExperts, MoEExpertsLoRA)):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    lora_param_names.add(f"{prefix}{proj}_lora_A")
                    lora_param_names.add(f"{prefix}{proj}_lora_B")

        for pname, param in fsdp_mod.named_parameters():
            # Skip params owned by child FSDP modules (processed separately)
            if any(pname.startswith(p) for p in child_fsdp_prefixes):
                continue
            # Skip DTensor params (EP-sharded or other distributed tensors)
            if isinstance(param.data, DTensor):
                continue
            # Skip EP-handled MoE expert params (gathered separately)
            if skip_moe_prefixes:
                if "" in skip_moe_prefixes:
                    logger.info(f"[SKIP-EP] Skipping ALL param: {pname} (skip_moe_prefixes={''})")
                    continue  # entire FSDP module is EP MoE — skip all
                if any(pname.startswith(p + ".") or pname == p for p in skip_moe_prefixes):
                    continue
            elif pname in ("gate_proj", "up_proj", "down_proj"):
                logger.info(f"[NO-SKIP] MoE param NOT skipped: {pname}, skip_moe_prefixes={skip_moe_prefixes}")
            # Skip LoRA-only params — they'll be merged into base weight
            if pname in lora_param_names:
                continue

            full_name = f"{mod_name}.{pname}" if mod_name != "(root)" else pname

            # Check if this is a base weight with LoRA to merge
            # Case 1: LoraLinear — pname like "self_attn.q_proj.weight"
            #   parent module "self_attn.q_proj" is the LoraLinear
            parent_name = ".".join(pname.split(".")[:-1])
            param_leaf = pname.split(".")[-1]  # e.g. "weight", "gate_proj"

            if parent_name in lora_modules:
                lora_mod = lora_modules[parent_name]
                if isinstance(lora_mod, LoraLinear):
                    delta = lora_mod.get_delta_weight()
                    merged = param.data.to(dtype=torch.bfloat16) + delta.to(dtype=torch.bfloat16)
                    cloned = merged.clone()
                    buffer.append((full_name, cloned))
                    continue
                elif isinstance(lora_mod, MoEExpertsLoRA):
                    if param_leaf == "gate_up_proj":
                        merged = param.data.to(dtype=torch.bfloat16).clone()
                        half = merged.shape[2] // 2
                        if "gate_proj" in lora_mod.lora_config.target_modules:
                            gate_delta = lora_mod._compute_proj_delta("gate_proj")
                            merged[:, :, :half].add_(gate_delta.to(dtype=torch.bfloat16))
                        if "up_proj" in lora_mod.lora_config.target_modules:
                            up_delta = lora_mod._compute_proj_delta("up_proj")
                            merged[:, :, half:].add_(up_delta.to(dtype=torch.bfloat16))
                        buffer.append((full_name, merged))
                        continue
                    if param_leaf == "down_proj":
                        if "down_proj" in lora_mod.lora_config.target_modules:
                            delta = lora_mod._compute_proj_delta("down_proj")
                            merged = param.data.to(dtype=torch.bfloat16) + delta.to(dtype=torch.bfloat16)
                        else:
                            merged = param.data.to(dtype=torch.bfloat16)
                        buffer.append((full_name, merged.clone()))
                        continue
            # Check if this is a non-LoRA MoEExperts fused tensor
            _is_moe_experts = False
            if parent_name:
                # Walk up to find parent module
                parts_list = parent_name.split(".")
                parent_mod = fsdp_mod
                try:
                    for p in parts_list:
                        parent_mod = getattr(parent_mod, p)
                    if isinstance(parent_mod, (MoEExperts, MoEExpertsLoRA)):
                        if param_leaf in ("gate_up_proj", "down_proj"):
                            _is_moe_experts = True
                except (AttributeError, TypeError):
                    pass

            if _is_moe_experts:
                buffer.append((full_name, param.data.to(dtype=torch.bfloat16).clone()))
            else:
                cloned = param.data.to(dtype=torch.bfloat16).clone()
                buffer.append((full_name, cloned))

        # Handle tied weights (e.g. tie_word_embeddings: lm_head.weight == embed_tokens.weight).
        # named_parameters() deduplicates by identity, so tied weights are only yielded once.
        # We need to emit them under all names so inference engines get all expected params.
        tied_keys = getattr(fsdp_mod, "_tied_weights_keys", None)
        if tied_keys:
            buffer_names = {n for n, _ in buffer}
            for tied_name, source_name in tied_keys.items():
                full_tied = f"{mod_name}.{tied_name}" if mod_name != "(root)" else tied_name
                full_source = f"{mod_name}.{source_name}" if mod_name != "(root)" else source_name
                if full_tied not in buffer_names and full_source in buffer_names:
                    for buf_name, buf_tensor in buffer:
                        if buf_name == full_source:
                            logger.info(
                                f"Rank 0: [WeightSync] Tied weight: emitting {full_tied} (clone of {full_source})"
                            )
                            buffer.append((full_tied, buf_tensor.clone()))
                            break

        return buffer

    @staticmethod
    def _unfuse_for_inference(
        buffer: List[Tuple[str, torch.Tensor]],
        model,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Split fused projections (qkv_proj, gate_up_proj) into HF-format names.

        Handles:
        - qkv_proj → q_proj + k_proj + v_proj (split fused attention)
        - gate_up_proj → gate_proj + up_proj (split fused dense/shared MLP)
        - MoE experts: gate_up_proj/down_proj → per-expert HF gate/up/down weights
        - Qwen3.5 linear attention: remap split GatedDeltaNet params back to
          HF fused names (q_proj/k_proj/v_proj → in_proj_qkv, etc.)
        """
        from xorl.models.transformers.qwen3_5_shared import (
            has_linear_attention_layers,
            remap_linear_attention_params_for_inference,
        )

        This splits fused weights back to individual projections before sending.
        """
        config = model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        result = []
        for name, tensor in buffer:
            if ".qkv_proj." in name:
                # Split [q_size + 2*kv_size, hidden] → q, k, v
                prefix, suffix = name.rsplit(".qkv_proj.", 1)
                q = tensor[:q_size].clone()
                k = tensor[q_size : q_size + kv_size].clone()
                v = tensor[q_size + kv_size :].clone()
                result.append((f"{prefix}.q_proj.{suffix}", q))
                result.append((f"{prefix}.k_proj.{suffix}", k))
                result.append((f"{prefix}.v_proj.{suffix}", v))
            elif name.endswith(".mlp.experts.gate_up_proj"):
                prefix = name.rsplit(".gate_up_proj", 1)[0]
                half = tensor.shape[2] // 2
                gate = tensor[:, :, :half].transpose(1, 2).contiguous()
                up = tensor[:, :, half:].transpose(1, 2).contiguous()
                for expert_idx in range(tensor.shape[0]):
                    result.append((f"{prefix}.{expert_idx}.gate_proj.weight", gate[expert_idx]))
                    result.append((f"{prefix}.{expert_idx}.up_proj.weight", up[expert_idx]))
            elif name.endswith(".mlp.experts.down_proj"):
                prefix = name.rsplit(".down_proj", 1)[0]
                down = tensor.transpose(1, 2).contiguous()
                for expert_idx in range(tensor.shape[0]):
                    result.append((f"{prefix}.{expert_idx}.down_proj.weight", down[expert_idx]))
            elif ".gate_up_proj." in name:
                # Split [2*intermediate, hidden] → gate, up
                prefix, suffix = name.rsplit(".gate_up_proj.", 1)
                half = tensor.shape[0] // 2
                gate = tensor[:half].clone()
                up = tensor[half:].clone()
                result.append((f"{prefix}.gate_proj.{suffix}", gate))
                result.append((f"{prefix}.up_proj.{suffix}", up))
            else:
                result.append((name, tensor))
        return result

    def _broadcast_buffer(
        self,
        backend,
        buffer: List[Tuple[str, torch.Tensor]],
        flush_cache: bool,
        weight_version: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Broadcast a buffer of (name, tensor) pairs to inference endpoints.

        Uses the transport backend's transfer_bucket for HTTP coordination
        + data transfer.

        Returns:
            (total_bytes, num_params)
        """
        if not buffer:
            return 0, 0

        bucket_bytes = sum(t.numel() * t.element_size() for _, t in buffer)
        logger.info(f"Rank {self.rank}: [WeightSync] Broadcasting {len(buffer)} params, {bucket_bytes / 1e6:.1f} MB")

        backend.transfer_bucket(
            buffer,
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        return bucket_bytes, len(buffer)

    @staticmethod
    def _get_fsdp_modules(model) -> Tuple[Optional[Any], List[Tuple[str, Any]]]:
        """
        Discover FSDP modules in the model.

        Returns:
            (root_fsdp_module_or_none, [(fqn, fsdp_module), ...])

        Root module holds non-layer params (embed, norm, head).
        Layer modules hold per-layer params.
        """
        from torch.distributed.fsdp._fully_shard import FSDPModule

        root_module = None
        layer_modules = []

        for fqn, mod in model.named_modules():
            if not isinstance(mod, FSDPModule):
                continue
            if fqn == "":
                root_module = mod
            else:
                layer_modules.append((fqn, mod))

        return root_module, layer_modules
