"""
QLoRA wrapper for MoE expert modules (e.g., Qwen3MoeSparseExperts).

Quantizes 3D expert weights (gate_proj, up_proj, down_proj) into nvfp4
and adds hybrid shared LoRA. Dequantizes on-the-fly during forward.

All weights are stored in (G, K, N) format — [num_experts, in_features, out_features].

LoRA weight parameter names match MoEExpertsLoRA for checkpoint compatibility::

    gate_proj_lora_A: [1, hidden, r]     gate_proj_lora_B: [E, r, inter]   (hybrid shared)
    up_proj_lora_A:   [1, hidden, r]     up_proj_lora_B:   [E, r, inter]   (hybrid shared)
    down_proj_lora_A: [E, inter, r]      down_proj_lora_B: [1, r, hidden]  (hybrid shared)

For large MoE models (e.g., 235B), base weights are NOT registered as nn.Parameter
to avoid FSDP allocating them on GPU (which would OOM). Instead, weights are loaded
directly from the checkpoint and quantized on-the-fly via load_and_quantize_weights().

Supports Expert Parallelism (EP) via the backend registry: dispatch (alltoall/deepep)
→ compute (triton/quack/native) → combine.

Drop-in replacement for MoEExperts/MoEExpertsLoRA — just replace module.experts in
the original MoeBlock.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xorl.ops.quantize import (
    nvfp4_quantize_gkn,
    nvfp4_dequantize_gkn,
)
from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX
from xorl.ops.quantize import block_fp8_quantize_gkn, block_fp8_dequantize_gkn
from xorl.ops.group_gemm.kernel.lora_utils import compute_lora_scaling
from xorl.lora.modules.base import LoraModule


class QLoRAMoeExperts(LoraModule, nn.Module):
    """
    QLoRA wrapper for MoE experts with separate gate/up/down projections.

    Stores only LOCAL experts (num_local_experts = num_experts // ep_size).
    Uses (G, K, N) weight format and backend registry for compute.

    Args:
        num_local_experts: Number of local experts on this EP rank
        num_experts: Total number of experts (global)
        intermediate_size: MoE intermediate size (gate/up output dim)
        hidden_size: Model hidden size (gate/up input dim, down output dim)
        r: LoRA rank per expert
        lora_alpha: LoRA scaling factor
        quant_format: Only "nvfp4" is supported
        quant_group_size: Block size for quantization (default: 16)
        expert_offset: Starting global expert index for this rank
        moe_implementation: Backend for compute ("triton", "quack", "native", "eager")
        hybrid_shared: Use hybrid shared LoRA design
        use_rslora: Use rank-stabilized LoRA scaling
        ep_dispatch: EP dispatch strategy ("alltoall" or "deepep")
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        r: int = 16,
        lora_alpha: int = 16,
        quant_format: str = "nvfp4",
        quant_group_size: int = 16,
        act_fn: Optional[nn.Module] = None,
        expert_offset: int = 0,
        device: Optional[torch.device] = None,
        moe_implementation: str = "triton",
        hybrid_shared: bool = True,
        use_rslora: bool = False,
        ep_dispatch: str = "alltoall",
        deepep_buffer_size_gb: float = 2.0,
        deepep_num_sms: int = 20,
        deepep_async_combine: bool = False,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_size  # alias for compatibility with MoEExpertsLoRA
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = compute_lora_scaling(lora_alpha, r, use_rslora)
        self.quant_format = quant_format
        self.quant_group_size = quant_group_size
        self.act_fn = act_fn or nn.SiLU()
        self.expert_offset = expert_offset
        self.moe_implementation = moe_implementation
        self.hybrid_shared = hybrid_shared
        self.use_rslora = use_rslora

        # EP dispatch settings
        self.ep_dispatch = ep_dispatch
        self.deepep_buffer_size_gb = deepep_buffer_size_gb
        self.deepep_num_sms = deepep_num_sms
        self.deepep_async_combine = deepep_async_combine

        # Base weights are NOT nn.Parameters — loaded separately via
        # load_and_quantize_weights() to avoid FSDP GPU allocation OOM.
        self._source_fqn: Optional[str] = None  # set by inject_qlora_into_model
        self._source_quant_format: Optional[str] = None  # checkpoint format (for cross-format conversion)
        self._weights_loaded = False

        # EMA-tracked amax for re-quantization (per projection, per-expert)
        self._ema_amax: dict = {"gate": None, "up": None, "down": None}

        # Quantized weight storage — nvfp4 (populated after quantization)
        self.register_buffer("gate_packed", None)
        self.register_buffer("gate_block_scales", None)
        self.register_buffer("gate_global_scale", None)
        self.register_buffer("up_packed", None)
        self.register_buffer("up_block_scales", None)
        self.register_buffer("up_global_scale", None)
        self.register_buffer("down_packed", None)
        self.register_buffer("down_block_scales", None)
        self.register_buffer("down_global_scale", None)

        # LoRA parameters in (G, K, N) format with hybrid shared design:
        #   gate/up: lora_A shared [1, hidden, r], lora_B per-expert [E, r, inter]
        #   down: lora_A per-expert [E, inter, r], lora_B shared [1, r, hidden]
        # Created at GLOBAL shape (num_experts) — the EP parallel plan will
        # Shard(0) them to local shape.  Shared params (size=1 on dim-0) are
        # automatically replicated by the EP plan.
        shared_exp = 1 if hybrid_shared else num_experts
        self._create_lora_params(
            "gate_proj", shared_exp, num_experts, r,
            hidden_size, intermediate_size, device,
        )
        self._create_lora_params(
            "up_proj", shared_exp, num_experts, r,
            hidden_size, intermediate_size, device,
        )
        self._create_lora_params(
            "down_proj", num_experts, (1 if hybrid_shared else num_experts), r,
            intermediate_size, hidden_size, device,
        )

        self._scale_dtypes = {"gate": {}, "up": {}, "down": {}}

        # QLoRAMoeExperts is NOT compatible with fully_shard's __class__ reassignment
        # (register_buffer(..., None) pattern causes layout mismatch).
        # Mark _skip_fsdp so torch_parallelize.py skips fully_shard(experts_mod)
        # and instead uses ignored_params to exclude LoRA params from parent FSDP.
        # LoRA params start at GLOBAL shape and are EP-sharded by the parallel plan;
        # no further FSDP sharding is needed (fsdp_mesh is size-1 when ep_size == world_size).
        self._skip_fsdp = True

        # Tell checkpoint loading to silently skip these keys (base weights
        # are loaded separately via load_and_quantize_weights, not through FSDP).
        self._qlora_expected_skip_keys = {"gate_proj", "up_proj", "down_proj"}

        self.reset_lora_parameters()

    def _create_lora_params(
        self, name: str, A_experts: int, B_experts: int, r: int,
        in_features: int, out_features: int, device: Optional[torch.device] = None,
    ):
        """Create LoRA A and B parameters in (G, K, N) format.

        A: [A_experts, in_features, r]
        B: [B_experts, r, out_features]
        """
        setattr(self, f"{name}_lora_A", nn.Parameter(
            torch.empty(A_experts, in_features, r, dtype=torch.float32, device=device)
        ))
        setattr(self, f"{name}_lora_B", nn.Parameter(
            torch.empty(B_experts, r, out_features, dtype=torch.float32, device=device)
        ))

    def reset_lora_parameters(self):
        """Initialize LoRA weights: kaiming_uniform for A, zeros for B."""
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lora_A = getattr(self, f"{proj}_lora_A")
            lora_B = getattr(self, f"{proj}_lora_B")
            for i in range(lora_A.shape[0]):
                nn.init.kaiming_uniform_(lora_A.data[i], a=math.sqrt(5))
            nn.init.zeros_(lora_B.data)

    def _compute_proj_delta(self, proj_name: str) -> torch.Tensor:
        """Compute LoRA delta for one projection. Returns [E, K, N] in GKN format."""
        lora_A = getattr(self, f"{proj_name}_lora_A")  # [1 or E, in, r]
        lora_B = getattr(self, f"{proj_name}_lora_B")  # [E or 1, r, out]
        E = max(lora_A.shape[0], lora_B.shape[0])
        A = lora_A.expand(E, -1, -1)  # [E, in, r]
        B = lora_B.expand(E, -1, -1)  # [E, r, out]
        return torch.bmm(A, B) * self.scaling  # [E, in, out] = [E, K, N]

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weights and re-quantize.

        Dequantize -> add LoRA delta -> EMA-update _ema_amax -> re-quantize -> store.
        Resets LoRA parameters after merge.

        For block_fp8: skip EMA amax (per-block scales are independent, no global scale).
        """
        with torch.no_grad():
            for proj_name, K, N in [
                ("gate", self.hidden_size, self.intermediate_size),
                ("up", self.hidden_size, self.intermediate_size),
                ("down", self.intermediate_size, self.hidden_size),
            ]:
                w = self.dequantize_all_experts(proj_name, K, N)  # [E, K, N] bf16
                delta = self._compute_proj_delta(f"{proj_name}_proj").to(w.dtype)
                w_merged = w + delta
                if self.quant_format == "block_fp8":
                    # block_fp8: re-quantize with fresh per-block scales, no global amax
                    self._quantize_proj(proj_name, w_merged)
                else:
                    fresh_amax = w_merged.float().abs().amax(dim=(1, 2))  # [E]
                    if self._ema_amax[proj_name] is not None:
                        self._ema_amax[proj_name].lerp_(
                            fresh_amax.to(self._ema_amax[proj_name].device), ema_decay
                        )
                    else:
                        self._ema_amax[proj_name] = fresh_amax
                    self._quantize_proj(
                        proj_name, w_merged,
                        global_amax_per_expert=self._ema_amax[proj_name],
                    )
        self.reset_lora_parameters()

    # ------------------------------------------------------------------
    # Construction from existing module
    # ------------------------------------------------------------------

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        r: int = 16,
        lora_alpha: int = 16,
        quant_format: str = "nvfp4",
        quant_group_size: int = 16,
        num_local_experts: Optional[int] = None,
        expert_offset: int = 0,
        hybrid_shared: bool = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> "QLoRAMoeExperts":
        """Create from a Qwen3MoeSparseExperts or MoEExperts module.

        Inherits EP settings (moe_implementation, ep_dispatch, deepep_*) from source.
        Source weights are in (G,K,N) format [num_experts, in_features, out_features].
        """
        gate = module.gate_proj  # [num_experts, hidden, intermediate] (G,K,N)
        total_experts = gate.shape[0]

        if num_local_experts is None:
            num_local_experts = total_experts

        hidden_size = getattr(module, "hidden_dim", gate.shape[1])
        intermediate_size = getattr(module, "intermediate_size", gate.shape[2])
        act_fn = getattr(module, "act_fn", None)
        moe_implementation = getattr(module, "moe_implementation", "triton")
        ep_dispatch = getattr(module, "ep_dispatch", "alltoall")

        qlora = cls(
            num_local_experts=num_local_experts,
            num_experts=total_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            quant_format=quant_format,
            quant_group_size=quant_group_size,
            act_fn=act_fn,
            expert_offset=expert_offset,
            device=gate.device,
            moe_implementation=moe_implementation,
            hybrid_shared=hybrid_shared,
            use_rslora=use_rslora,
            ep_dispatch=ep_dispatch,
            deepep_buffer_size_gb=getattr(module, "deepep_buffer_size_gb", 2.0),
            deepep_num_sms=getattr(module, "deepep_num_sms", 20),
            deepep_async_combine=getattr(module, "deepep_async_combine", False),
        )

        if gate.device.type != "meta":
            # Non-meta: quantize local experts immediately (already in G,K,N)
            local_gate = gate[expert_offset:expert_offset + num_local_experts].detach()
            local_up = module.up_proj[expert_offset:expert_offset + num_local_experts].detach()
            local_down = module.down_proj[expert_offset:expert_offset + num_local_experts].detach()
            qlora._quantize_proj("gate", local_gate)
            qlora._quantize_proj("up", local_up)
            qlora._quantize_proj("down", local_down)
            # Initialize EMA amax per-expert for each projection (not needed for block_fp8)
            if quant_format != "block_fp8":
                qlora._ema_amax["gate"] = local_gate.float().abs().amax(dim=(1, 2))
                qlora._ema_amax["up"] = local_up.float().abs().amax(dim=(1, 2))
                qlora._ema_amax["down"] = local_down.float().abs().amax(dim=(1, 2))
            qlora._weights_loaded = True
        # For meta init: weights will be loaded via load_and_quantize_weights()

        return qlora

    # ------------------------------------------------------------------
    # Quantize / Dequantize helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(t: Tensor) -> Tensor:
        if t.dtype == torch.uint8:
            return t.contiguous()
        return t.view(torch.uint8).contiguous()

    def _recover_tensor(self, buf: Tensor, original_dtype: torch.dtype) -> Tensor:
        if original_dtype == torch.uint8:
            return buf
        return buf.contiguous().view(original_dtype)

    def _quantize_2d(self, w: Tensor, global_amax=None):
        """Quantize a 2D [K, N] weight in G,K,N format. Groups along K (contraction dim)."""
        if self.quant_format == "block_fp8":
            fp8_w, scales = block_fp8_quantize_gkn(w.float(), self.quant_group_size)
            return self._to_uint8(fp8_w), {
                "weight_block_scales": (self._to_uint8(scales), scales.dtype),
            }
        else:
            packed, block_scales, global_scale = nvfp4_quantize_gkn(
                w, self.quant_group_size, global_amax=global_amax
            )
            return self._to_uint8(packed), {
                "weight_block_scales": (self._to_uint8(block_scales), block_scales.dtype),
                "weight_global_scale": (self._to_uint8(global_scale), global_scale.dtype),
            }

    def _dequantize_2d(self, packed: Tensor, scales_dict: dict, K: int, N: int) -> Tensor:
        """Dequantize to [K, N] G,K,N format."""
        if self.quant_format == "block_fp8":
            fp8_w = packed.view(torch.float8_e4m3fn).reshape(K, N)
            scales = self._recover_tensor(*scales_dict["weight_block_scales"])
            return block_fp8_dequantize_gkn(fp8_w, scales, self.quant_group_size)
        else:
            block_scales = self._recover_tensor(*scales_dict["weight_block_scales"])
            global_scale = self._recover_tensor(*scales_dict["weight_global_scale"])
            return nvfp4_dequantize_gkn(packed, block_scales, global_scale, K, N, self.quant_group_size)

    def _quantize_proj(self, proj_name: str, w3d: Tensor,
                        global_amax_per_expert: Optional[Tensor] = None) -> None:
        """Quantize a 3D [num_local_experts, K, N] tensor and store as stacked buffers."""
        results = []
        for i in range(w3d.shape[0]):
            ga = global_amax_per_expert[i:i+1] if global_amax_per_expert is not None else None
            packed, scales_dict = self._quantize_2d(w3d[i], global_amax=ga)
            results.append((packed, scales_dict))

        # Stack packed weights
        setattr(self, f"{proj_name}_packed", torch.stack([r[0] for r in results]))

        # Stack scales
        first_scales = results[0][1]
        self._scale_dtypes[proj_name] = {}
        for key in first_scales:
            stacked = torch.stack([r[1][key][0] for r in results])
            dtype = first_scales[key][1]
            self._scale_dtypes[proj_name][key] = dtype
            if key == "weight_block_scales":
                setattr(self, f"{proj_name}_block_scales", stacked)
            elif key == "weight_global_scale":
                setattr(self, f"{proj_name}_global_scale", stacked)

        # For block_fp8: no global_scale buffer needed
        if self.quant_format == "block_fp8":
            setattr(self, f"{proj_name}_global_scale", None)

    def _build_scales_dict(self, proj_name: str, expert_idx: int) -> dict:
        """Build scales dict for a single expert from stacked buffers."""
        scales_dict = {}
        for key, dtype in self._scale_dtypes[proj_name].items():
            if key == "weight_block_scales":
                buf = getattr(self, f"{proj_name}_block_scales")
            elif key == "weight_global_scale":
                buf = getattr(self, f"{proj_name}_global_scale")
                if buf is None:
                    continue  # block_fp8: no global_scale
            else:
                continue
            scales_dict[key] = (buf[expert_idx], dtype)
        return scales_dict

    def dequantize_all_experts(self, proj_name: str, K: int, N: int) -> Tensor:
        """Dequantize all local experts for a projection into a 3D tensor.

        Returns:
            Tensor of shape [num_local_experts, K, N] in (G,K,N) format (detached, no grad).
        """
        packed = getattr(self, f"{proj_name}_packed")
        experts = []
        for i in range(self.num_local_experts):
            w = self._dequantize_2d(packed[i], self._build_scales_dict(proj_name, i), K, N)
            experts.append(w)
        return torch.stack(experts)

    def dequantize_expert(self, proj_name: str, expert_idx: int, K: int, N: int) -> Tensor:
        """Dequantize weight for a single LOCAL expert of a given projection.

        Returns:
            Tensor of shape [K, N] in (K,N) format.
        """
        packed = getattr(self, f"{proj_name}_packed")
        scales_dict = self._build_scales_dict(proj_name, expert_idx)
        return self._dequantize_2d(packed[expert_idx], scales_dict, K, N)

    # ------------------------------------------------------------------
    # Weight loading from checkpoint (only local experts)
    # ------------------------------------------------------------------

    def load_and_quantize_weights(self, weights_path: str, weight_map: Optional[dict] = None) -> None:
        """Load LOCAL expert weights from checkpoint and quantize directly.

        HF checkpoints store per-expert weights as individual keys:
        `model.layers.X.mlp.experts.{i}.gate_proj.weight`  — shape [out, in]
        We load only local experts, stack into 3D, transpose to (G,K,N), quantize, then free.

        Pre-quantized checkpoints (NVFP4 modelopt format) have:
        `{fqn}.{i}.{proj}.weight` — uint8 packed [out, in//2]
        `{fqn}.{i}.{proj}.weight_scale` — fp8 block_scales [out, in//block_size]
        `{fqn}.{i}.{proj}.weight_scale_2` — fp32 global_scale [1]
        These are loaded directly, transposed to GKN, with global_scale absorbed.

        Args:
            weights_path: Path to HF model (local dir or hub ID)
            weight_map: Optional pre-loaded weight_map from index.json
        """
        if self._weights_loaded or self._source_fqn is None:
            return

        import safetensors.torch
        from transformers.utils import cached_file

        # Cache of loaded shard files to avoid re-reading
        _shard_cache = {}

        def _load_tensor(ckpt_key: str) -> Tensor:
            if weight_map is not None:
                shard_file = weight_map.get(ckpt_key)
                if shard_file is None:
                    raise RuntimeError(f"Key {ckpt_key} not found in checkpoint index")
            else:
                shard_file = "model.safetensors"

            if shard_file not in _shard_cache:
                shard_path = cached_file(weights_path, shard_file)
                _shard_cache[shard_file] = safetensors.torch.load_file(shard_path, device="cpu")
            return _shard_cache[shard_file][ckpt_key]

        # Detect checkpoint format: use _source_quant_format if set, otherwise probe keys
        first_expert = self.expert_offset
        probe_nvfp4 = f"{self._source_fqn}.{first_expert}.gate_proj.weight_scale"
        probe_fp8 = f"{self._source_fqn}.{first_expert}.gate_proj.weight_scale_inv"

        source_format = self._source_quant_format
        if source_format is None:
            # Auto-detect from weight_map probing
            if weight_map is not None and probe_fp8 in weight_map:
                source_format = "block_fp8"
            elif weight_map is not None and probe_nvfp4 in weight_map:
                source_format = "nvfp4"

        if source_format == "block_fp8":
            self._load_prequantized_block_fp8_experts(_load_tensor, _shard_cache)
        elif source_format == "nvfp4":
            self._load_prequantized_experts(_load_tensor, _shard_cache)
        else:
            self._load_and_quantize_bf16_experts(_load_tensor, _shard_cache)

        # Cross-format conversion: checkpoint format differs from target
        if source_format is not None and source_format != self.quant_format:
            self._convert_prequantized_format()

        self._weights_loaded = True

    def _load_and_quantize_bf16_experts(self, _load_tensor, _shard_cache) -> None:
        """Standard path: load bf16 expert weights, transpose, quantize."""
        for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
            expert_weights = []
            for i in range(self.expert_offset, self.expert_offset + self.num_local_experts):
                ckpt_key = f"{self._source_fqn}.{i}.{hf_name}.weight"
                w = _load_tensor(ckpt_key)
                expert_weights.append(w)

            # Stack: [num_local_experts, out_features, in_features] (HF format)
            # Transpose to (G,K,N): [num_local_experts, in_features, out_features]
            w3d = torch.stack(expert_weights).cuda()
            w3d = w3d.transpose(1, 2).contiguous()
            del expert_weights
            if self.quant_format != "block_fp8":
                self._ema_amax[proj_name] = w3d.float().abs().amax(dim=(1, 2))
            self._quantize_proj(proj_name, w3d)
            del w3d
            torch.cuda.empty_cache()
            _shard_cache.clear()

    def _load_prequantized_experts(self, _load_tensor, _shard_cache) -> None:
        """Pre-quantized path: load packed/scales per expert, transpose to GKN, store directly."""
        for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
            packed_list = []
            block_scales_list = []
            amax_list = []

            for i in range(self.expert_offset, self.expert_offset + self.num_local_experts):
                fqn_prefix = f"{self._source_fqn}.{i}.{hf_name}"
                packed = _load_tensor(f"{fqn_prefix}.weight")           # [N, K//2] uint8
                block_scales = _load_tensor(f"{fqn_prefix}.weight_scale")  # [N, K//bs] fp8
                global_scale = _load_tensor(f"{fqn_prefix}.weight_scale_2")  # [1] fp32

                # Recover calibrated amax from global_scale
                calibrated_amax = global_scale.float().item() * FP4_E2M1_MAX * FP8_E4M3_MAX
                amax_list.append(calibrated_amax)

                # Transpose from HF [N, K//2] to GKN [K//2, N]
                packed_gkn = packed.T.contiguous()
                # Absorb global_scale into block_scales, transpose to GKN [K//bs, N]
                block_scales_gkn = (block_scales.float() * global_scale.float()).T.contiguous()

                packed_list.append(self._to_uint8(packed_gkn))
                block_scales_list.append(self._to_uint8(block_scales_gkn))

            # Stack per-expert: [num_local_experts, K//2, N] and [num_local_experts, K//bs, N]
            # Move to CUDA — buffers are loaded from CPU checkpoint but Triton kernels need GPU tensors
            device = torch.device("cuda")
            setattr(self, f"{proj_name}_packed", torch.stack(packed_list).to(device))
            setattr(self, f"{proj_name}_block_scales", torch.stack(block_scales_list).to(device))

            # Global scale is absorbed — set to 1.0 per expert
            global_scale_ones = self._to_uint8(
                torch.ones(self.num_local_experts, 1, dtype=torch.float32)
            )
            setattr(self, f"{proj_name}_global_scale", global_scale_ones.to(device))

            # Record scale dtypes
            self._scale_dtypes[proj_name] = {
                "weight_block_scales": torch.float32,  # absorbed, stored as fp32
                "weight_global_scale": torch.float32,
            }

            # Store per-expert EMA amax recovered from calibrated global_scale
            self._ema_amax[proj_name] = torch.tensor(amax_list, dtype=torch.float32, device=device)

            _shard_cache.clear()

    def _load_prequantized_block_fp8_experts(self, _load_tensor, _shard_cache) -> None:
        """Pre-quantized block FP8 path: load fp8 weights + scale_inv per expert.

        HF FP8 format: {fqn}.{i}.{proj}.weight (float8_e4m3fn [N, K])
                        {fqn}.{i}.{proj}.weight_scale_inv (float32 [N//128, K//128])
        Transpose from HF [N, K] to GKN [K, N] format.
        """
        for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
            packed_list = []
            block_scales_list = []

            for i in range(self.expert_offset, self.expert_offset + self.num_local_experts):
                fqn_prefix = f"{self._source_fqn}.{i}.{hf_name}"
                fp8_w = _load_tensor(f"{fqn_prefix}.weight")              # [N, K] fp8
                scales = _load_tensor(f"{fqn_prefix}.weight_scale_inv")   # [N//128, K//128] f32

                # Transpose from HF [N, K] to GKN [K, N]
                fp8_w_gkn = fp8_w.T.contiguous()
                scales_gkn = scales.float().T.contiguous()

                packed_list.append(self._to_uint8(fp8_w_gkn))
                block_scales_list.append(self._to_uint8(scales_gkn))

            device = torch.device("cuda")
            setattr(self, f"{proj_name}_packed", torch.stack(packed_list).to(device))
            setattr(self, f"{proj_name}_block_scales", torch.stack(block_scales_list).to(device))
            setattr(self, f"{proj_name}_global_scale", None)

            self._scale_dtypes[proj_name] = {
                "weight_block_scales": torch.float32,
            }
            # block_fp8: no EMA amax needed
            _shard_cache.clear()

    def _convert_prequantized_format(self) -> None:
        """Convert loaded prequantized weights from source to target format.

        Dequantizes all experts per-projection from source format, then
        re-quantizes in the target format. Used when the checkpoint format
        differs from the desired training format.
        """
        source_format = self._source_quant_format
        source_gs = 128 if source_format == "block_fp8" else 16
        target_format = self.quant_format
        target_gs = self.quant_group_size

        for proj_name, K, N in [
            ("gate", self.hidden_size, self.intermediate_size),
            ("up", self.hidden_size, self.intermediate_size),
            ("down", self.intermediate_size, self.hidden_size),
        ]:
            # Dequantize in source format
            self.quant_format = source_format
            self.quant_group_size = source_gs
            w3d = self.dequantize_all_experts(proj_name, K, N)

            # Re-quantize in target format
            self.quant_format = target_format
            self.quant_group_size = target_gs
            if target_format != "block_fp8":
                self._ema_amax[proj_name] = w3d.float().abs().amax(dim=(1, 2))
            else:
                self._ema_amax[proj_name] = None
            self._quantize_proj(proj_name, w3d)

        # Ensure target format is set
        self.quant_format = target_format
        self.quant_group_size = target_gs

    # ------------------------------------------------------------------
    # Properties — expose dequantized base weights in (G,K,N) format
    # ------------------------------------------------------------------

    @property
    def gate_proj(self) -> Tensor:
        """Dequantize all local gate_proj weights. [num_local_experts, hidden, intermediate]"""
        return self.dequantize_all_experts("gate", self.hidden_size, self.intermediate_size)

    @property
    def up_proj(self) -> Tensor:
        """Dequantize all local up_proj weights. [num_local_experts, hidden, intermediate]"""
        return self.dequantize_all_experts("up", self.hidden_size, self.intermediate_size)

    @property
    def down_proj(self) -> Tensor:
        """Dequantize all local down_proj weights. [num_local_experts, intermediate, hidden]"""
        return self.dequantize_all_experts("down", self.intermediate_size, self.hidden_size)

    # ------------------------------------------------------------------
    # Forward — unified with backend registry
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        routing_weights: Tensor = None,
        selected_experts: Tensor = None,
        expert_idx: int = None,
    ) -> Tensor:
        """Forward pass with LoRA via backend registry.

        For **eager**: called per-expert with ``expert_idx``.
        For **triton/quack/native**: checks EP first, falls back to local path.
        """
        if self.moe_implementation == "eager":
            assert expert_idx is not None
            return self._eager_lora_forward(hidden_states, expert_idx)

        # Check EP — use unified dispatch/compute/combine path
        from xorl.distributed.parallel_state import get_parallel_state
        parallel_state = get_parallel_state()

        if parallel_state.ep_enabled:
            return self._ep_forward(
                hidden_states, routing_weights, selected_experts, parallel_state
            )

        # Local path — registry-based
        from xorl.models.layers.moe.backend import MOE_EXPERT_BACKENDS_LORA

        compute_dtype = hidden_states.dtype
        fn = MOE_EXPERT_BACKENDS_LORA[self.moe_implementation]
        return fn(
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            gate_proj=self.gate_proj.to(compute_dtype),
            up_proj=self.up_proj.to(compute_dtype),
            down_proj=self.down_proj.to(compute_dtype),
            gate_proj_lora_A=self.gate_proj_lora_A,
            gate_proj_lora_B=self.gate_proj_lora_B,
            up_proj_lora_A=self.up_proj_lora_A,
            up_proj_lora_B=self.up_proj_lora_B,
            down_proj_lora_A=self.down_proj_lora_A,
            down_proj_lora_B=self.down_proj_lora_B,
            scaling=self.scaling,
        )

    def _ep_forward(
        self,
        hidden_states: Tensor,
        routing_weights: Tensor,
        selected_experts: Tensor,
        parallel_state,
    ) -> Tensor:
        """Unified EP forward with LoRA: dispatch -> compute -> combine.

        Uses the same dispatch/combine as MoEExperts._ep_forward() but routes
        to the LoRA-aware EP compute registry.
        """
        from xorl.models.layers.moe.backend import EP_DISPATCH, EP_COMBINE, EP_EXPERT_COMPUTE_LORA

        if self.moe_implementation not in EP_EXPERT_COMPUTE_LORA:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"EP with LoRA. Available: {list(EP_EXPERT_COMPUTE_LORA.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. "
                f"Available: {list(EP_DISPATCH.keys())}"
            )

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]
        compute_fn = EP_EXPERT_COMPUTE_LORA[self.moe_implementation]

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(
            hidden_states, routing_weights, selected_experts, parallel_state
        )
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)

        # Step 2: Expert computation with dequantized base + LoRA
        compute_dtype = permute_tokens.dtype
        expert_output = compute_fn(
            permute_tokens, cumsum,
            self.gate_proj.to(compute_dtype),
            self.up_proj.to(compute_dtype),
            self.down_proj.to(compute_dtype),
            self.gate_proj_lora_A, self.gate_proj_lora_B,
            self.up_proj_lora_A, self.up_proj_lora_B,
            self.down_proj_lora_A, self.down_proj_lora_B,
            self.scaling,
        )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(
            expert_output, ctx, dispatch_kwargs, parallel_state
        )
        return combine_fn(**combine_kwargs)

    def _build_dispatch_kwargs(self, hidden_states, routing_weights, selected_experts, parallel_state):
        """Build dispatch kwargs based on ep_dispatch strategy."""
        kwargs = dict(
            hidden_states=hidden_states,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=self.num_experts,
        )
        if self.ep_dispatch == "alltoall":
            kwargs["ep_group"] = parallel_state.ep_group
        elif self.ep_dispatch == "deepep":
            from xorl.distributed.moe.deepep import get_default_buffer
            kwargs["buffer"] = get_default_buffer(
                ep_group=parallel_state.ep_group,
                buffer_size_gb=self.deepep_buffer_size_gb,
                num_sms=self.deepep_num_sms,
            )
            kwargs["num_local_experts"] = self.num_local_experts
        return kwargs

    def _build_combine_kwargs(self, expert_output, ctx, dispatch_kwargs, parallel_state):
        """Build combine kwargs based on ep_dispatch strategy."""
        if self.ep_dispatch == "alltoall":
            return dict(expert_output=expert_output, ctx=ctx, ep_group=parallel_state.ep_group)
        elif self.ep_dispatch == "deepep":
            return dict(
                buffer=dispatch_kwargs["buffer"],
                expert_output=expert_output,
                ctx=ctx,
                async_combine=self.deepep_async_combine,
            )

    def _eager_lora_forward(self, hidden_states: Tensor, expert_idx: int) -> Tensor:
        """Per-expert LoRA forward (eager mode).

        All weights in (G, K, N) format — direct matmul, no transpose.
        expert_idx is a GLOBAL index. Converted to local index via expert_offset.
        Returns zeros for non-local experts (EP case).
        """
        # Convert global expert index to local
        local_idx = expert_idx - self.expert_offset
        if local_idx < 0 or local_idx >= self.num_local_experts:
            return torch.zeros_like(hidden_states[:, :self.hidden_size])

        compute_dtype = hidden_states.dtype

        # gate_proj: x @ W (no transpose with G,K,N)
        gate_w = self.dequantize_expert("gate", local_idx, self.hidden_size, self.intermediate_size)
        gate_out = torch.matmul(hidden_states, gate_w.to(compute_dtype))

        # gate LoRA: (x @ A) @ B * scaling — hybrid shared via min()
        A = self.gate_proj_lora_A[min(local_idx, self.gate_proj_lora_A.shape[0] - 1)].to(compute_dtype)
        B = self.gate_proj_lora_B[local_idx].to(compute_dtype)
        gate_out = gate_out + torch.matmul(torch.matmul(hidden_states, A), B) * self.scaling

        # up_proj: x @ W
        up_w = self.dequantize_expert("up", local_idx, self.hidden_size, self.intermediate_size)
        up_out = torch.matmul(hidden_states, up_w.to(compute_dtype))

        # up LoRA
        A = self.up_proj_lora_A[min(local_idx, self.up_proj_lora_A.shape[0] - 1)].to(compute_dtype)
        B = self.up_proj_lora_B[local_idx].to(compute_dtype)
        up_out = up_out + torch.matmul(torch.matmul(hidden_states, A), B) * self.scaling

        # Activation
        out = self.act_fn(gate_out) * up_out

        # down_proj: h @ W
        down_w = self.dequantize_expert("down", local_idx, self.intermediate_size, self.hidden_size)
        down_out = torch.matmul(out, down_w.to(compute_dtype))

        # down LoRA
        A = self.down_proj_lora_A[local_idx].to(compute_dtype)
        B = self.down_proj_lora_B[min(local_idx, self.down_proj_lora_B.shape[0] - 1)].to(compute_dtype)
        down_out = down_out + torch.matmul(torch.matmul(out, A), B) * self.scaling

        return down_out

    def extra_repr(self) -> str:
        return (
            f"num_local_experts={self.num_local_experts}, "
            f"num_experts={self.num_experts}, "
            f"expert_offset={self.expert_offset}, "
            f"intermediate_size={self.intermediate_size}, "
            f"hidden_size={self.hidden_size}, "
            f"r={self.r}, quant_format={self.quant_format}, "
            f"moe_implementation={self.moe_implementation}, "
            f"hybrid_shared={self.hybrid_shared}, "
            f"ep_dispatch={self.ep_dispatch}"
        )
