"""
QLoRA wrapper for MoE expert modules with format-specific subclasses.

Base class QLoRAMoeExperts holds shared logic (LoRA, forward, EP, weight loading).
Subclasses define format-specific scale buffers, quantization, and dequantization:

- NvFP4QLoRAMoeExperts: 4-bit (nvfp4) — 9 buffers (packed + block_scales + global_scale per proj)
- BlockFP8QLoRAMoeExperts: 8-bit (block_fp8) — 6 buffers (packed + block_scales per proj, no global_scale)

All weights are stored in (G, K, N) format -- [num_experts, in_features, out_features].

LoRA weight parameter names match MoEExpertsLoRA for checkpoint compatibility::

    gate_proj_lora_A: [1, hidden, r]     gate_proj_lora_B: [E, r, inter]   (hybrid shared)
    up_proj_lora_A:   [1, hidden, r]     up_proj_lora_B:   [E, r, inter]   (hybrid shared)
    down_proj_lora_A: [E, inter, r]      down_proj_lora_B: [1, r, hidden]  (hybrid shared)

Supports Expert Parallelism (EP) via the backend registry: dispatch (alltoall/deepep)
-> compute (triton/quack/native) -> combine.

Drop-in replacement for MoEExperts/MoEExpertsLoRA -- just replace module.experts in
the original MoeBlock.
"""

import math
from typing import Optional

import safetensors.torch
import torch
import torch.nn as nn
from torch import Tensor
from transformers.utils import cached_file

from xorl.lora.modules.base import LoraModule
from xorl.ops.group_gemm.kernel.lora_utils import compute_lora_scaling
from xorl.ops.quantize import (
    block_fp8_dequantize_gkn,
    block_fp8_quantize_gkn,
    nf4_dequantize_gkn,
    nf4_quantize_gkn,
    nvfp4_dequantize_gkn,
    nvfp4_quantize_gkn,
)
from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX


class QLoRAMoeExperts(LoraModule, nn.Module):
    """
    Base QLoRA wrapper for MoE experts with separate gate/up/down projections.

    Do not instantiate directly -- use NvFP4QLoRAMoeExperts or BlockFP8QLoRAMoeExperts.

    Stores only LOCAL experts (num_local_experts = num_experts // ep_size).
    Uses (G, K, N) weight format and backend registry for compute.
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

        # Base weights are NOT nn.Parameters -- loaded separately via
        # load_and_quantize_weights() to avoid FSDP GPU allocation OOM.
        self._source_fqn: Optional[str] = None  # set by inject_qlora_into_model
        self._source_quant_format: Optional[str] = None  # checkpoint format (set by inject_qlora_into_model)
        self._weights_loaded = False

        # EMA-tracked amax for re-quantization (per projection, per-expert)
        self._ema_amax: dict = {"gate": None, "up": None, "down": None}

        # LoRA parameters in (G, K, N) format with hybrid shared design:
        #   gate/up: lora_A shared [1, hidden, r], lora_B per-expert [E, r, inter]
        #   down: lora_A per-expert [E, inter, r], lora_B shared [1, r, hidden]
        # Created at GLOBAL shape (num_experts) -- the EP parallel plan will
        # Shard(0) them to local shape.  Shared params (size=1 on dim-0) are
        # automatically replicated by the EP plan.
        shared_exp = 1 if hybrid_shared else num_experts
        self._create_lora_params(
            "gate_proj",
            shared_exp,
            num_experts,
            r,
            hidden_size,
            intermediate_size,
            device,
        )
        self._create_lora_params(
            "up_proj",
            shared_exp,
            num_experts,
            r,
            hidden_size,
            intermediate_size,
            device,
        )
        self._create_lora_params(
            "down_proj",
            num_experts,
            (1 if hybrid_shared else num_experts),
            r,
            intermediate_size,
            hidden_size,
            device,
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

        # NOTE: Subclasses MUST register quantized weight buffers,
        # then call self.reset_lora_parameters()

    def _create_lora_params(
        self,
        name: str,
        A_experts: int,
        B_experts: int,
        r: int,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
    ):
        """Create LoRA A and B parameters in (G, K, N) format.

        A: [A_experts, in_features, r]
        B: [B_experts, r, out_features]
        """
        setattr(
            self,
            f"{name}_lora_A",
            nn.Parameter(torch.empty(A_experts, in_features, r, dtype=torch.float32, device=device)),
        )
        setattr(
            self,
            f"{name}_lora_B",
            nn.Parameter(torch.empty(B_experts, r, out_features, dtype=torch.float32, device=device)),
        )

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

    # ------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # ------------------------------------------------------------------

    def _quantize_2d(self, w: Tensor, global_amax=None):
        """Quantize a 2D [K, N] weight. Returns (packed_uint8, scales_dict)."""
        raise NotImplementedError

    def _dequantize_2d(self, packed: Tensor, scales_dict: dict, K: int, N: int) -> Tensor:
        """Dequantize to [K, N] G,K,N format."""
        raise NotImplementedError

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weights and re-quantize."""
        raise NotImplementedError

    def _load_experts(self, _load_tensor, _shard_cache) -> None:
        """Load pre-quantized expert weights from checkpoint. Subclass must implement."""
        raise NotImplementedError

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

        # Pick the right subclass if called on base
        if cls is QLoRAMoeExperts:
            if quant_format == "block_fp8":
                subcls = BlockFP8QLoRAMoeExperts
            elif quant_format == "nf4":
                subcls = NF4QLoRAMoeExperts
            else:
                subcls = NvFP4QLoRAMoeExperts
        else:
            subcls = cls

        qlora = subcls(
            num_local_experts=num_local_experts,
            num_experts=total_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            r=r,
            lora_alpha=lora_alpha,
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

        # Weights will be loaded via load_and_quantize_weights()
        return qlora

    # ------------------------------------------------------------------
    # Shared quantize / dequantize helpers
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

    def _quantize_proj(self, proj_name: str, w3d: Tensor, global_amax_per_expert: Optional[Tensor] = None) -> None:
        """Quantize a 3D [num_local_experts, K, N] tensor and store as stacked buffers."""
        results = []
        for i in range(w3d.shape[0]):
            ga = global_amax_per_expert[i : i + 1] if global_amax_per_expert is not None else None
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
            elif key == "weight_scales":
                setattr(self, f"{proj_name}_scales", stacked)

    def _build_scales_dict(self, proj_name: str, expert_idx: int) -> dict:
        """Build scales dict for a single expert from stacked buffers."""
        scales_dict = {}
        for key, dtype in self._scale_dtypes[proj_name].items():
            if key == "weight_block_scales":
                buf = getattr(self, f"{proj_name}_block_scales")
            elif key == "weight_global_scale":
                buf = getattr(self, f"{proj_name}_global_scale", None)
                if buf is None:
                    continue
            elif key == "weight_scales":
                buf = getattr(self, f"{proj_name}_scales")
            else:
                continue
            scales_dict[key] = (buf[expert_idx], dtype)
        return scales_dict

    @torch.compiler.disable
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

    def load_and_quantize_weights(
        self, weights_path: str, weight_map: Optional[dict] = None, shard_cache: Optional[dict] = None
    ) -> None:
        """Load LOCAL expert weights from checkpoint and quantize directly.

        Each subclass loads its own format via _load_experts().
        Cross-format loading is not supported.

        Args:
            weights_path: Path to HF model (local dir or hub ID)
            weight_map: Optional pre-loaded weight_map from index.json
            shard_cache: Optional shared dict of loaded shard files
        """
        if self._weights_loaded or self._source_fqn is None:
            return

        _shard_cache = shard_cache if shard_cache is not None else {}

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

        self._load_experts(_load_tensor, _shard_cache)
        self._weights_loaded = True

    # ------------------------------------------------------------------
    # Properties -- expose dequantized base weights in (G,K,N) format
    # ------------------------------------------------------------------

    @property
    def gate_proj(self) -> Tensor:
        """Dequantize all local gate_proj weights. [num_local_experts, hidden, intermediate]"""
        return self.dequantize_all_experts("gate", self.hidden_size, self.intermediate_size)

    @property
    def gate_up_proj(self) -> Tensor:
        """Dequantize all local fused gate_up_proj weights. [num_local_experts, hidden, 2 * intermediate]"""
        return torch.cat([self.gate_proj, self.up_proj], dim=2)

    @property
    def up_proj(self) -> Tensor:
        """Dequantize all local up_proj weights. [num_local_experts, hidden, intermediate]"""
        return self.dequantize_all_experts("up", self.hidden_size, self.intermediate_size)

    @property
    def down_proj(self) -> Tensor:
        """Dequantize all local down_proj weights. [num_local_experts, intermediate, hidden]"""
        return self.dequantize_all_experts("down", self.intermediate_size, self.hidden_size)

    # ------------------------------------------------------------------
    # Forward -- unified with backend registry
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

        # Check EP -- use unified dispatch/compute/combine path
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()

        if parallel_state.ep_enabled:
            return self._ep_forward(hidden_states, routing_weights, selected_experts, parallel_state)

        # Local path -- registry-based
        from xorl.models.layers.moe.backend import MOE_EXPERT_BACKENDS_LORA  # noqa: PLC0415

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

    @torch.compiler.disable
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
        from xorl.models.layers.moe.backend import EP_COMBINE, EP_DISPATCH, EP_EXPERT_COMPUTE_LORA  # noqa: PLC0415

        if self.moe_implementation not in EP_EXPERT_COMPUTE_LORA:
            raise ValueError(
                f"moe_implementation={self.moe_implementation!r} does not support "
                f"EP with LoRA. Available: {list(EP_EXPERT_COMPUTE_LORA.keys())}"
            )
        if self.ep_dispatch not in EP_DISPATCH:
            raise ValueError(
                f"ep_dispatch={self.ep_dispatch!r} is not available. Available: {list(EP_DISPATCH.keys())}"
            )

        dispatch_fn = EP_DISPATCH[self.ep_dispatch]
        combine_fn = EP_COMBINE[self.ep_dispatch]
        compute_fn = EP_EXPERT_COMPUTE_LORA[self.moe_implementation]

        # Step 1: Dispatch tokens to expert-owning ranks
        dispatch_kwargs = self._build_dispatch_kwargs(hidden_states, routing_weights, selected_experts, parallel_state)
        permute_tokens, cumsum, ctx = dispatch_fn(**dispatch_kwargs)

        # Step 2: Expert computation with dequantized base + LoRA
        compute_dtype = permute_tokens.dtype
        expert_output = compute_fn(
            permute_tokens,
            cumsum,
            self.gate_proj.to(compute_dtype),
            self.up_proj.to(compute_dtype),
            self.down_proj.to(compute_dtype),
            self.gate_proj_lora_A,
            self.gate_proj_lora_B,
            self.up_proj_lora_A,
            self.up_proj_lora_B,
            self.down_proj_lora_A,
            self.down_proj_lora_B,
            self.scaling,
        )

        # Step 3: Combine expert outputs back to original ranks
        combine_kwargs = self._build_combine_kwargs(expert_output, ctx, dispatch_kwargs, parallel_state)
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
            from xorl.distributed.moe.deepep import get_default_buffer  # noqa: PLC0415

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

        All weights in (G, K, N) format -- direct matmul, no transpose.
        expert_idx is a GLOBAL index. Converted to local index via expert_offset.
        Returns zeros for non-local experts (EP case).
        """
        # Convert global expert index to local
        local_idx = expert_idx - self.expert_offset
        if local_idx < 0 or local_idx >= self.num_local_experts:
            return torch.zeros_like(hidden_states[:, : self.hidden_size])

        compute_dtype = hidden_states.dtype

        # gate_proj: x @ W (no transpose with G,K,N)
        gate_w = self.dequantize_expert("gate", local_idx, self.hidden_size, self.intermediate_size)
        gate_out = torch.matmul(hidden_states, gate_w.to(compute_dtype))

        # gate LoRA: (x @ A) @ B * scaling -- hybrid shared via min()
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


class NvFP4QLoRAMoeExperts(QLoRAMoeExperts):
    """
    NVFP4 QLoRA MoE experts: 4-bit quantized base weights + trainable LoRA.

    Registers 9 buffers per instance: {gate,up,down}_{packed,block_scales,global_scale}.
    EMA-tracked _ema_amax informs global_scale for re-quantization.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        r: int = 16,
        lora_alpha: int = 16,
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
        super().__init__(
            num_local_experts,
            num_experts,
            intermediate_size,
            hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            quant_format="nvfp4",
            quant_group_size=16,
            act_fn=act_fn,
            expert_offset=expert_offset,
            device=device,
            moe_implementation=moe_implementation,
            hybrid_shared=hybrid_shared,
            use_rslora=use_rslora,
            ep_dispatch=ep_dispatch,
            deepep_buffer_size_gb=deepep_buffer_size_gb,
            deepep_num_sms=deepep_num_sms,
            deepep_async_combine=deepep_async_combine,
        )
        # Quantized weight storage: 9 buffers (packed + block_scales + global_scale per projection)
        self.register_buffer("gate_packed", None)
        self.register_buffer("gate_block_scales", None)
        self.register_buffer("gate_global_scale", None)
        self.register_buffer("up_packed", None)
        self.register_buffer("up_block_scales", None)
        self.register_buffer("up_global_scale", None)
        self.register_buffer("down_packed", None)
        self.register_buffer("down_block_scales", None)
        self.register_buffer("down_global_scale", None)

        self.reset_lora_parameters()

    def _load_experts(self, _load_tensor, _shard_cache) -> None:
        """Load expert weights from checkpoint, quantizing bf16 to NVFP4 if needed.

        Supports two checkpoint formats:
        - Pre-quantized NVFP4 (modelopt): weight [N, K//2] uint8 + weight_scale [N, K//BS] fp8
          + weight_scale_2 scalar f32. GPU-batch strategy avoids per-expert CPU transpose.
        - BF16 (standard HF): weight [N, K] bf16. Quantized to NVFP4 on GPU per expert.

        HF format  : weight [N, K//2] uint8, weight_scale [N, K//BS] fp8
        GKN format : weight [E, K//2, N] uint8, block_scales [E, K//BS, N] f32
        """
        device = torch.device("cuda")
        E = self.num_local_experts
        expert_range = range(self.expert_offset, self.expert_offset + E)

        # Detect checkpoint format: try loading weight_scale for the first expert.
        # If not found in weight_map, it's a plain bf16 checkpoint.
        first_scale_key = f"{self._source_fqn}.{self.expert_offset}.gate_proj.weight_scale"
        try:
            _load_tensor(first_scale_key)
            is_prequantized = True
        except RuntimeError:
            is_prequantized = False

        for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
            amax_list: list = []

            if is_prequantized:
                packed_hf_list: list = []  # [N, K//2] uint8 per expert
                bs_u8_list: list = []  # [N, K//BS] uint8 (fp8 bits reinterpreted) per expert
                gs_list: list = []  # scalar f32 per expert

                for i in expert_range:
                    fqn_prefix = f"{self._source_fqn}.{i}.{hf_name}"
                    packed = _load_tensor(f"{fqn_prefix}.weight")  # [N, K//2] uint8
                    bs = _load_tensor(f"{fqn_prefix}.weight_scale")  # [N, K//BS] fp8
                    gs = _load_tensor(f"{fqn_prefix}.weight_scale_2")  # scalar f32

                    gs_val = gs.float().item()
                    packed_hf_list.append(packed)
                    # Reinterpret fp8 bits as uint8 (both 1-byte, zero-copy on CPU)
                    bs_u8_list.append(bs.view(torch.uint8))
                    gs_list.append(gs_val)
                    amax_list.append(gs_val * FP4_E2M1_MAX * FP8_E4M3_MAX)

                # Stack in HF layout — no per-expert transpose on CPU
                packed_hf = torch.stack(packed_hf_list)  # [E, N, K//2] uint8
                bs_u8_hf = torch.stack(bs_u8_list)  # [E, N, K//BS] uint8 (fp8 bits)
                gs_cpu = torch.tensor(gs_list, dtype=torch.float32)  # [E]

                # Bulk H2D copy
                packed_gpu = packed_hf.to(device)  # [E, N, K//2] uint8
                bs_u8_gpu = bs_u8_hf.to(device)  # [E, N, K//BS] uint8
                gs_gpu = gs_cpu.to(device)  # [E] f32

                # GPU: reinterpret uint8 → fp8, upcast to f32, absorb global scale
                bs_f32 = bs_u8_gpu.view(torch.float8_e4m3fn).float()  # [E, N, K//BS] f32
                bs_absorbed = bs_f32 * gs_gpu[:, None, None]  # [E, N, K//BS] f32

                # GPU permute HF [E, N, *] → GKN [E, *, N]
                packed_gkn = packed_gpu.permute(0, 2, 1).contiguous()  # [E, K//2, N] uint8
                bs_gkn = bs_absorbed.permute(0, 2, 1).contiguous()  # [E, K//BS, N] f32

            else:
                # BF16 checkpoint: load weight, move to GPU, quantize per expert.
                packed_gkn_list: list = []
                bs_gkn_list: list = []

                for i in expert_range:
                    fqn_prefix = f"{self._source_fqn}.{i}.{hf_name}"
                    w_cpu = _load_tensor(f"{fqn_prefix}.weight")  # [N, K] bf16/f32 on CPU
                    w_gkn = w_cpu.to(device=device, dtype=torch.bfloat16).T.contiguous()  # [K, N]
                    del w_cpu

                    amax = w_gkn.float().abs().max().reshape(1)
                    amax_list.append(amax.item())

                    packed, block_scales, global_scale = nvfp4_quantize_gkn(
                        w_gkn, self.quant_group_size, global_amax=amax
                    )
                    del w_gkn
                    # Absorb global_scale into block_scales so global_scale = 1.0
                    bs_absorbed = (block_scales.float() * global_scale.float()).contiguous()
                    packed_gkn_list.append(self._to_uint8(packed))
                    bs_gkn_list.append(self._to_uint8(bs_absorbed))
                    _shard_cache.clear()

                packed_gkn = torch.stack(packed_gkn_list)  # [E, K//2, N] uint8
                bs_gkn = torch.stack(bs_gkn_list)  # [E, K//BS, N] f32 (in uint8)

            setattr(self, f"{proj_name}_packed", self._to_uint8(packed_gkn))
            setattr(self, f"{proj_name}_block_scales", self._to_uint8(bs_gkn))
            setattr(
                self, f"{proj_name}_global_scale", self._to_uint8(torch.ones(E, 1, dtype=torch.float32, device=device))
            )
            self._scale_dtypes[proj_name] = {
                "weight_block_scales": torch.float32,
                "weight_global_scale": torch.float32,
            }
            self._ema_amax[proj_name] = torch.tensor(amax_list, dtype=torch.float32, device=device)

            if is_prequantized:
                _shard_cache.clear()

    def _quantize_2d(self, w: Tensor, global_amax=None):
        """Quantize a 2D [K, N] weight using NVFP4. Groups along K (contraction dim)."""
        packed, block_scales, global_scale = nvfp4_quantize_gkn(w, self.quant_group_size, global_amax=global_amax)
        return self._to_uint8(packed), {
            "weight_block_scales": (self._to_uint8(block_scales), block_scales.dtype),
            "weight_global_scale": (self._to_uint8(global_scale), global_scale.dtype),
        }

    def _dequantize_2d(self, packed: Tensor, scales_dict: dict, K: int, N: int) -> Tensor:
        """Dequantize to [K, N] G,K,N format using NVFP4."""
        block_scales = self._recover_tensor(*scales_dict["weight_block_scales"])
        global_scale = self._recover_tensor(*scales_dict["weight_global_scale"])
        return nvfp4_dequantize_gkn(packed, block_scales, global_scale, K, N, self.quant_group_size)

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weights, EMA-update _ema_amax, re-quantize."""
        with torch.no_grad():
            for proj_name, K, N in [
                ("gate", self.hidden_size, self.intermediate_size),
                ("up", self.hidden_size, self.intermediate_size),
                ("down", self.intermediate_size, self.hidden_size),
            ]:
                w = self.dequantize_all_experts(proj_name, K, N)  # [E, K, N] bf16
                delta = self._compute_proj_delta(f"{proj_name}_proj").to(w.dtype)
                w_merged = w + delta
                fresh_amax = w_merged.float().abs().amax(dim=(1, 2))  # [E]
                if self._ema_amax[proj_name] is not None:
                    self._ema_amax[proj_name].lerp_(fresh_amax.to(self._ema_amax[proj_name].device), ema_decay)
                else:
                    self._ema_amax[proj_name] = fresh_amax
                self._quantize_proj(
                    proj_name,
                    w_merged,
                    global_amax_per_expert=self._ema_amax[proj_name],
                )
        self.reset_lora_parameters()


class BlockFP8QLoRAMoeExperts(QLoRAMoeExperts):
    """
    Block FP8 QLoRA MoE experts: float8_e4m3fn quantized base weights + trainable LoRA.

    Registers 6 buffers per instance: {gate,up,down}_{packed,block_scales}.
    No global_scale, no EMA amax.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        r: int = 16,
        lora_alpha: int = 16,
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
        super().__init__(
            num_local_experts,
            num_experts,
            intermediate_size,
            hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            quant_format="block_fp8",
            quant_group_size=128,
            act_fn=act_fn,
            expert_offset=expert_offset,
            device=device,
            moe_implementation=moe_implementation,
            hybrid_shared=hybrid_shared,
            use_rslora=use_rslora,
            ep_dispatch=ep_dispatch,
            deepep_buffer_size_gb=deepep_buffer_size_gb,
            deepep_num_sms=deepep_num_sms,
            deepep_async_combine=deepep_async_combine,
        )
        # Quantized weight storage: 6 buffers (packed + block_scales per projection, no global_scale)
        self.register_buffer("gate_packed", None)
        self.register_buffer("gate_block_scales", None)
        self.register_buffer("up_packed", None)
        self.register_buffer("up_block_scales", None)
        self.register_buffer("down_packed", None)
        self.register_buffer("down_block_scales", None)

        self.reset_lora_parameters()

    def _load_experts(self, _load_tensor, _shard_cache) -> None:
        """Load pre-quantized block FP8 expert weights from checkpoint.

        GPU-transpose strategy: collect raw HF tensors without per-expert CPU
        transposition, stack into batches, bulk H2D copy, then permute on GPU
        (~3× faster for large expert banks).

        HF format  : weight [N, K] fp8, weight_scale_inv [N//BS, K//BS] bf16/f32
        GKN format : weight [E, K, N] uint8, block_scales [E, K//BS, N//BS] f32
        """
        device = torch.device("cuda")
        E = self.num_local_experts
        expert_range = range(self.expert_offset, self.expert_offset + E)

        for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
            fp8_u8_list: list = []  # [N, K] uint8 (fp8 bits reinterpreted) per expert
            scales_list: list = []  # [N//BS, K//BS] f32 per expert

            for i in expert_range:
                fqn_prefix = f"{self._source_fqn}.{i}.{hf_name}"
                fp8_w = _load_tensor(f"{fqn_prefix}.weight")  # [N, K] fp8
                scales = _load_tensor(f"{fqn_prefix}.weight_scale_inv")  # [N//BS, K//BS] bf16/f32

                # Reinterpret fp8 bits as uint8 (both 1-byte, zero-copy on CPU)
                fp8_u8_list.append(fp8_w.view(torch.uint8))
                scales_list.append(scales.float())

            # Stack in HF layout — no per-expert transpose on CPU
            fp8_u8_hf = torch.stack(fp8_u8_list)  # [E, N, K] uint8
            scales_hf = torch.stack(scales_list)  # [E, N//BS, K//BS] f32

            # Bulk H2D copy
            fp8_u8_gpu = fp8_u8_hf.to(device)  # [E, N, K] uint8
            scales_gpu = scales_hf.to(device)  # [E, N//BS, K//BS] f32

            # GPU permute HF [E, N, *] → GKN [E, *, N]
            fp8_gkn = fp8_u8_gpu.permute(0, 2, 1).contiguous()  # [E, K, N] uint8
            scales_gkn = scales_gpu.permute(0, 2, 1).contiguous()  # [E, K//BS, N//BS] f32

            setattr(self, f"{proj_name}_packed", self._to_uint8(fp8_gkn))
            setattr(self, f"{proj_name}_block_scales", self._to_uint8(scales_gkn))
            self._scale_dtypes[proj_name] = {
                "weight_block_scales": torch.float32,
            }

            _shard_cache.clear()

    def _quantize_2d(self, w: Tensor, global_amax=None):
        """Quantize a 2D [K, N] weight using block FP8. Groups along K (contraction dim)."""
        fp8_w, scales = block_fp8_quantize_gkn(w.float(), self.quant_group_size)
        return self._to_uint8(fp8_w), {
            "weight_block_scales": (self._to_uint8(scales), scales.dtype),
        }

    def _dequantize_2d(self, packed: Tensor, scales_dict: dict, K: int, N: int) -> Tensor:
        """Dequantize to [K, N] G,K,N format using block FP8."""
        fp8_w = packed.view(torch.float8_e4m3fn).reshape(K, N)
        scales = self._recover_tensor(*scales_dict["weight_block_scales"])
        return block_fp8_dequantize_gkn(fp8_w, scales, self.quant_group_size)

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weights, re-quantize with fresh per-block scales."""
        with torch.no_grad():
            for proj_name, K, N in [
                ("gate", self.hidden_size, self.intermediate_size),
                ("up", self.hidden_size, self.intermediate_size),
                ("down", self.intermediate_size, self.hidden_size),
            ]:
                w = self.dequantize_all_experts(proj_name, K, N)
                delta = self._compute_proj_delta(f"{proj_name}_proj").to(w.dtype)
                w_merged = w + delta
                self._quantize_proj(proj_name, w_merged)
        self.reset_lora_parameters()


class NF4QLoRAMoeExperts(QLoRAMoeExperts):
    """
    NF4 QLoRA MoE experts: 4-bit quantized base weights + trainable LoRA.

    NF4 uses a non-uniform 16-level codebook optimized for normally distributed
    weights. Simpler scale structure: one float32 absmax per group (no global_scale).
    Registers 6 buffers: {gate,up,down}_{packed,scales}.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        r: int = 16,
        lora_alpha: int = 16,
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
        super().__init__(
            num_local_experts,
            num_experts,
            intermediate_size,
            hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            quant_format="nf4",
            quant_group_size=64,
            act_fn=act_fn,
            expert_offset=expert_offset,
            device=device,
            moe_implementation=moe_implementation,
            hybrid_shared=hybrid_shared,
            use_rslora=use_rslora,
            ep_dispatch=ep_dispatch,
            deepep_buffer_size_gb=deepep_buffer_size_gb,
            deepep_num_sms=deepep_num_sms,
            deepep_async_combine=deepep_async_combine,
        )
        # Quantized weight storage: 6 buffers (packed + scales per projection)
        self.register_buffer("gate_packed", None)
        self.register_buffer("gate_scales", None)
        self.register_buffer("up_packed", None)
        self.register_buffer("up_scales", None)
        self.register_buffer("down_packed", None)
        self.register_buffer("down_scales", None)

        self.reset_lora_parameters()

    def _load_experts(self, _load_tensor, _shard_cache) -> None:
        """Load bf16 expert weights from checkpoint and quantize to NF4.

        Loads per-expert 2D bf16 weights, transposes from HF [out, in] to GKN
        [in, out] format, then quantizes with nf4_quantize_gkn.
        """
        for proj_name, hf_name, K, N in [
            ("gate", "gate_proj", self.hidden_size, self.intermediate_size),
            ("up", "up_proj", self.hidden_size, self.intermediate_size),
            ("down", "down_proj", self.intermediate_size, self.hidden_size),
        ]:
            expert_weights = []
            for i in range(self.expert_offset, self.expert_offset + self.num_local_experts):
                ckpt_key = f"{self._source_fqn}.{i}.{hf_name}.weight"
                w_hf = _load_tensor(ckpt_key)  # [out_features, in_features]
                w_gkn = w_hf.T.contiguous().float()  # [K, N] = [in_features, out_features]
                expert_weights.append(w_gkn)

            # Stack and move to GPU before quantization (Triton kernels require CUDA tensors)
            w3d = torch.stack(expert_weights).cuda()  # [num_local_experts, K, N]
            self._quantize_proj(proj_name, w3d)

            _shard_cache.clear()

    def _quantize_2d(self, w: Tensor, global_amax=None):
        """Quantize a 2D [K, N] weight using NF4. Groups along K (contraction dim)."""
        packed, scales = nf4_quantize_gkn(w, self.quant_group_size)
        return self._to_uint8(packed), {
            "weight_scales": (self._to_uint8(scales), scales.dtype),
        }

    def _dequantize_2d(self, packed: Tensor, scales_dict: dict, K: int, N: int) -> Tensor:
        """Dequantize to [K, N] G,K,N format using NF4."""
        scales = self._recover_tensor(*scales_dict["weight_scales"])
        return nf4_dequantize_gkn(packed, scales, K, N, self.quant_group_size)

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weights, re-quantize with fresh per-group scales."""
        with torch.no_grad():
            for proj_name, K, N in [
                ("gate", self.hidden_size, self.intermediate_size),
                ("up", self.hidden_size, self.intermediate_size),
                ("down", self.intermediate_size, self.hidden_size),
            ]:
                w = self.dequantize_all_experts(proj_name, K, N)
                delta = self._compute_proj_delta(f"{proj_name}_proj").to(w.dtype)
                w_merged = w + delta
                self._quantize_proj(proj_name, w_merged)
        self.reset_lora_parameters()
