"""
QLoRA Linear layer implementation.

Stores base weights in quantized format (nvfp4 or block_fp8) for memory savings.
Dequantizes on-the-fly during forward. Only LoRA parameters are trainable.

Supports:
- Dynamic quantization from bf16 weights with amax tracking
- Loading pre-quantized weights (skip quantization)
- Periodic re-quantization: merge LoRA delta into base, re-quantize with fresh scales

Memory comparison per weight element:
    bf16:      2 bytes
    nvfp4:     0.5 byte (4x savings) + small overhead for scales
    block_fp8: 1 byte (2x savings) + small overhead for 2D block scales
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xorl.ops.quantize import (
    nvfp4_quantize,
    nvfp4_dequantize,
)
from xorl.ops.quantize import block_fp8_quantize_gkn, block_fp8_dequantize_gkn
from xorl.lora.modules.base import LoraModule


class QLoRALinear(LoraModule, nn.Module):
    """
    QLoRA Linear: quantized base weights + trainable LoRA.

    Base weights are stored in quantized format (nvfp4 or block_fp8) as uint8 buffers.
    During forward, weights are dequantized to compute dtype, then F.linear + LoRA.
    Only LoRA parameters (lora_A, lora_B) are trainable in fp32.

    Supports two initialization modes:
    - from_module(): quantize bf16 weights from an nn.Linear
    - from_quantized(): load pre-quantized packed weights directly

    Re-quantization (optional, step-based):
    - Called from training loop every N steps via maybe_requant_qlora()
    - Merges LoRA delta into base, EMA-updates _ema_amax, re-quantizes
    - _ema_amax informs global_scale for better fp8 block_scale precision (nvfp4 only)
    - block_fp8 uses per-block scales only, no EMA amax needed

    Args:
        in_features: Size of input features
        out_features: Size of output features
        r: LoRA rank
        lora_alpha: LoRA alpha for scaling
        quant_format: "nvfp4" or "block_fp8"
        quant_group_size: Block size for quantization (16 for nvfp4, 128 for block_fp8)
        bias: Whether to include bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        quant_format: str = "nvfp4",
        quant_group_size: int = 16,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format = quant_format
        self.quant_group_size = quant_group_size

        # LoRA parameters (trainable, fp32)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=torch.float32, device=device))

        # Bias (optional, frozen)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Base weight: kept as a frozen nn.Parameter when deferred quantization is used
        # (meta init). FSDP loads weights into this, then quantize_weight() converts it
        # into packed_weight_f32. Set to None when quantized.
        self.register_parameter("weight", None)

        # Quantized weight storage: float32 parameter for FSDP2 sharding.
        # Pack 4 uint8 bytes into 1 float32 element via .view(torch.float32).
        # FSDP2 naturally shards this → N× memory reduction vs replicated buffers.
        # DCP naturally saves/loads it → correct resumability.
        self.register_parameter("packed_weight_f32", None)

        # Scales: kept as replicated buffers (tiny — e.g., [4, 2] for 512×256 block_fp8).
        self.register_buffer("weight_block_scales", None)
        self.register_buffer("weight_global_scale", None)

        # EMA-tracked amax for re-quantization (plain attribute, not parameter — FSDP2 ignores it)
        self._ema_amax: Optional[Tensor] = None

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    # Metadata for pre-quantized loading (set by inject_qlora_into_model)
    _is_prequantized: bool = False
    _merge_sources: Optional[Tuple[str, ...]] = None  # e.g. ("q_proj", "k_proj", "v_proj")
    _source_fqn: Optional[str] = None  # full HF FQN prefix for this module
    _source_quant_format: Optional[str] = None  # checkpoint format (for cross-format conversion)

    # ------------------------------------------------------------------
    # Construction: from bf16 module or from pre-quantized weights
    # ------------------------------------------------------------------

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        r: int = 16,
        lora_alpha: int = 16,
        quant_format: str = "nvfp4",
        quant_group_size: int = 16,
        **kwargs,
    ) -> "QLoRALinear":
        """
        Create QLoRALinear from an existing nn.Linear.

        If the weight is bf16/fp16/fp32, it is dynamically quantized.
        """
        assert isinstance(module, nn.Linear), f"Expected nn.Linear, got {type(module)}"

        qlora = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            r=r,
            lora_alpha=lora_alpha,
            quant_format=quant_format,
            quant_group_size=quant_group_size,
            bias=module.bias is not None,
            device=module.weight.device,
        )

        w = module.weight.detach()
        if w.device.type == "meta":
            # Deferred quantization: keep weight as a parameter so FSDP can load into it.
            # Call quantize_weight() after FSDP loads the real weights.
            qlora.weight = nn.Parameter(w, requires_grad=False)
            # Create float32 placeholder for FSDP2 to shard.
            # packed_weight_f32 packs 4 uint8 bytes into 1 float32 element.
            if quant_format == "block_fp8":
                # block_fp8: 1 byte per element → K/4 float32 elements per row
                f32_shape = (module.out_features, module.in_features // 4)
            else:
                # nvfp4: 0.5 byte per element → K/8 float32 elements per row
                f32_shape = (module.out_features, module.in_features // 8)
            qlora.packed_weight_f32 = nn.Parameter(
                torch.empty(f32_shape, dtype=torch.float32, device="meta"),
                requires_grad=False,
            )
        elif w.dtype in (torch.bfloat16, torch.float16, torch.float32):
            if quant_format != "block_fp8":
                qlora._ema_amax = w.float().abs().max().reshape(1)
            qlora._quantize_and_store(w)
        else:
            raise ValueError(
                f"Unexpected weight dtype {w.dtype}. "
                f"Use from_quantized() for pre-quantized weights."
            )

        if module.bias is not None:
            with torch.no_grad():
                qlora.bias.copy_(module.bias)

        return qlora

    @classmethod
    def from_quantized(
        cls,
        packed_weight: Tensor,
        weight_scales: Optional[Tensor] = None,
        weight_block_scales: Optional[Tensor] = None,
        weight_global_scale: Optional[Tensor] = None,
        in_features: int = 0,
        out_features: int = 0,
        r: int = 16,
        lora_alpha: int = 16,
        quant_format: str = "nvfp4",
        quant_group_size: int = 16,
        bias: bool = False,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "QLoRALinear":
        """
        Create QLoRALinear from pre-quantized weights (skip quantization).

        Use this when loading from a quantized checkpoint (own format or HF).
        """
        qlora = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            quant_format=quant_format,
            quant_group_size=quant_group_size,
            bias=bias,
            device=device,
        )

        # Store packed weight as float32 parameter (pack 4 uint8 → 1 float32)
        uint8_data = cls._to_uint8(packed_weight)
        f32_data = uint8_data.contiguous().view(torch.float32)
        qlora.packed_weight_f32 = nn.Parameter(f32_data, requires_grad=False)

        # Store scales with dtype tracking (nvfp4: block_scales + global_scale)
        qlora._scale_dtypes = {}
        if weight_block_scales is not None:
            qlora._scale_dtypes["weight_block_scales"] = weight_block_scales.dtype
            qlora.weight_block_scales = cls._to_uint8(weight_block_scales)
        if weight_global_scale is not None:
            qlora._scale_dtypes["weight_global_scale"] = weight_global_scale.dtype
            qlora.weight_global_scale = cls._to_uint8(weight_global_scale)

        return qlora

    # ------------------------------------------------------------------
    # Pre-quantized weight loading (NVFP4 modelopt format)
    # ------------------------------------------------------------------

    def load_prequantized_weights(
        self,
        weight_map: dict,
        shard_cache: dict,
        weights_path: str,
    ) -> None:
        """Load pre-quantized weights from checkpoint (NVFP4 or block_fp8).

        For merged modules (qkv_proj, gate_up_proj), loads separate source projections
        and concatenates along dim=0.

        NVFP4: absorbs each source's global_scale into its block_scales (fp32).
        Block FP8: loads weight (fp8) + weight_scale_inv (f32 scales) directly.

        QLoRALinear uses NK format (out_features, in_features) — same as HF nn.Linear.

        Args:
            weight_map: Checkpoint key -> shard filename mapping
            shard_cache: Shared dict of loaded shard files (mutated for caching)
            weights_path: Path to HF model directory
        """
        import safetensors.torch
        from transformers.utils import cached_file

        def _load_tensor(ckpt_key: str) -> Tensor:
            shard_file = weight_map.get(ckpt_key)
            if shard_file is None:
                raise RuntimeError(f"Key {ckpt_key} not found in checkpoint index")
            if shard_file not in shard_cache:
                shard_path = cached_file(weights_path, shard_file)
                shard_cache[shard_file] = safetensors.torch.load_file(shard_path, device="cpu")
            return shard_cache[shard_file][ckpt_key]

        source_format = self._source_quant_format or self.quant_format

        if source_format == "block_fp8":
            self._load_prequantized_block_fp8(_load_tensor)
        else:
            self._load_prequantized_nvfp4(_load_tensor)

        # Cross-format conversion: source checkpoint format differs from target
        if source_format != self.quant_format:
            self._convert_prequantized_format(source_format)

    def _load_prequantized_block_fp8(self, _load_tensor) -> None:
        """Load pre-quantized block FP8 weights (HF format).

        HF FP8 format: {module}.weight (float8_e4m3fn) + {module}.weight_scale_inv (float32)
        Scale convention: dequant = fp8_weight * weight_scale_inv
        """
        if self._merge_sources is not None:
            packed_parts = []
            scales_parts = []
            for src_proj in self._merge_sources:
                src_fqn = f"{self._source_fqn}.{src_proj}"
                fp8_w = _load_tensor(f"{src_fqn}.weight")
                scales = _load_tensor(f"{src_fqn}.weight_scale_inv")
                packed_parts.append(fp8_w)
                scales_parts.append(scales)
            merged_packed = torch.cat(packed_parts, dim=0)
            merged_scales = torch.cat(scales_parts, dim=0)
        else:
            fqn = self._source_fqn
            merged_packed = _load_tensor(f"{fqn}.weight")
            merged_scales = _load_tensor(f"{fqn}.weight_scale_inv")

        device = self.lora_A.device if self.lora_A.device.type != "meta" else torch.device("cuda")

        uint8_data = self._to_uint8(merged_packed).to(device)
        self._write_packed_weight(uint8_data)
        self._scale_dtypes = {
            "weight_block_scales": torch.float32,
        }
        self.weight_block_scales = self._to_uint8(merged_scales.float()).to(device)
        self.weight_global_scale = None
        # block_fp8: no EMA amax needed
        self._ema_amax = None

    def _load_prequantized_nvfp4(self, _load_tensor) -> None:
        """Load pre-quantized NVFP4 weights from a modelopt checkpoint."""
        if self._merge_sources is not None:
            packed_parts = []
            block_scales_parts = []
            amax_values = []
            for src_proj in self._merge_sources:
                src_fqn = f"{self._source_fqn}.{src_proj}"
                packed = _load_tensor(f"{src_fqn}.weight")
                block_scales = _load_tensor(f"{src_fqn}.weight_scale")
                global_scale = _load_tensor(f"{src_fqn}.weight_scale_2")

                calibrated_amax = global_scale.float().item() * 6.0 * 448.0
                amax_values.append(calibrated_amax)

                effective_block_scales = block_scales.float() * global_scale.float()
                packed_parts.append(packed)
                block_scales_parts.append(effective_block_scales)

            merged_packed = torch.cat(packed_parts, dim=0)
            merged_block_scales = torch.cat(block_scales_parts, dim=0)
            merged_global_scale = torch.tensor([1.0], dtype=torch.float32)
            max_amax = max(amax_values)
        else:
            fqn = self._source_fqn
            merged_packed = _load_tensor(f"{fqn}.weight")
            block_scales = _load_tensor(f"{fqn}.weight_scale")
            global_scale = _load_tensor(f"{fqn}.weight_scale_2")
            max_amax = global_scale.float().item() * 6.0 * 448.0
            merged_block_scales = block_scales.float() * global_scale.float()
            merged_global_scale = torch.tensor([1.0], dtype=torch.float32)

        device = self.lora_A.device if self.lora_A.device.type != "meta" else torch.device("cuda")

        uint8_data = self._to_uint8(merged_packed).to(device)
        self._write_packed_weight(uint8_data)
        self._scale_dtypes = {
            "weight_block_scales": torch.float32,
            "weight_global_scale": torch.float32,
        }
        self.weight_block_scales = self._to_uint8(merged_block_scales).to(device)
        self.weight_global_scale = self._to_uint8(merged_global_scale).to(device)
        self._ema_amax = torch.tensor([max_amax], dtype=torch.float32, device=device)

    def _convert_prequantized_format(self, source_format: str) -> None:
        """Convert loaded prequantized weights from source to target format.

        Dequantizes from source format to bf16, then re-quantizes in target format.
        Used when the checkpoint format differs from the desired training format.
        """
        target_format = self.quant_format
        target_gs = self.quant_group_size

        # Dequantize using source format settings
        self.quant_format = source_format
        self.quant_group_size = 128 if source_format == "block_fp8" else 16
        w = self._dequantize_weight()

        # Re-quantize using target format
        self.quant_format = target_format
        self.quant_group_size = target_gs
        if target_format != "block_fp8":
            self._ema_amax = w.float().abs().max().reshape(1).to(w.device)
        else:
            self._ema_amax = None
        self._quantize_and_store(w)

    # ------------------------------------------------------------------
    # Quantize / dequantize
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(t: Tensor) -> Tensor:
        """Convert tensor to uint8 view for FSDP2-safe buffer storage.

        All quantized tensors (packed weights AND scales) are stored as uint8.
        This prevents FSDP2 mixed precision from casting floating-point scales
        (float32, float16, float8_e4m3fn) to param_dtype (bf16), which would
        corrupt precision or data. The original dtype is recorded in _scale_dtypes
        and recovered during dequantize.
        """
        if t.dtype == torch.uint8:
            return t.contiguous()
        return t.view(torch.uint8).contiguous()

    def _recover_tensor(self, buf: Tensor, original_dtype: torch.dtype) -> Tensor:
        """Recover original dtype from uint8-stored buffer via view."""
        if original_dtype == torch.uint8:
            return buf
        return buf.contiguous().view(original_dtype)

    def _quantize_and_store(self, w: Tensor, global_amax: Optional[Tensor] = None) -> None:
        """Quantize a full-precision weight tensor and store as float32 parameter + uint8 scale buffers."""
        if self.quant_format == "block_fp8":
            fp8_w, scales = block_fp8_quantize_gkn(w.float(), self.quant_group_size)
            uint8_data = self._to_uint8(fp8_w)
            self._write_packed_weight(uint8_data)
            self._scale_dtypes = {
                "weight_block_scales": scales.dtype,
            }
            self.weight_block_scales = self._to_uint8(scales)
            self.weight_global_scale = None
        else:
            packed, block_scales, global_scale = nvfp4_quantize(
                w, self.quant_group_size, global_amax=global_amax
            )
            uint8_data = self._to_uint8(packed)
            self._write_packed_weight(uint8_data)
            self._scale_dtypes = {
                "weight_block_scales": block_scales.dtype,
                "weight_global_scale": global_scale.dtype,
            }
            self.weight_block_scales = self._to_uint8(block_scales)
            self.weight_global_scale = self._to_uint8(global_scale)

    def _write_packed_weight(self, uint8_data: Tensor) -> None:
        """Write uint8 quantized data into the float32 parameter.

        Packs 4 uint8 bytes into 1 float32 element for FSDP2 sharding.
        Handles plain tensors, meta placeholders, and DTensors (under FSDP2).

        Note: nvfp4_quantize returns 1D flat packed data while block_fp8 returns
        2D. We reshape to match the parameter's expected shape before writing.
        """
        f32_data = uint8_data.contiguous().view(torch.float32)
        if self.packed_weight_f32 is None or self.packed_weight_f32.device.type == "meta":
            # First write or replacing meta placeholder.
            # Reshape to 2D (M, K//4 or K//8) matching the meta placeholder shape.
            if self.packed_weight_f32 is not None and self.packed_weight_f32.device.type == "meta":
                target_shape = self.packed_weight_f32.shape
                f32_data = f32_data.reshape(target_shape)
            self.packed_weight_f32 = nn.Parameter(f32_data, requires_grad=False)
        elif hasattr(self.packed_weight_f32, '_local_tensor'):
            # DTensor (under FSDP2): write to local shard.
            # f32_data is FULL tensor; extract the local shard portion.
            local = self.packed_weight_f32._local_tensor
            # Reshape f32_data to match DTensor's logical (full) shape.
            # nvfp4_quantize returns 1D flat packed data; block_fp8 returns 2D.
            full_shape = self.packed_weight_f32.shape  # DTensor logical shape
            f32_data = f32_data.reshape(full_shape)
            # Determine which shard this rank owns via placement offset
            from torch.distributed._tensor import Shard as _Shard
            placements = self.packed_weight_f32.placements
            shard_dim = 0
            for p in placements:
                if isinstance(p, _Shard):
                    shard_dim = p.dim
                    break
            # For FSDP2, sharding is typically on dim 0
            mesh = self.packed_weight_f32.device_mesh
            local_rank = mesh.get_local_rank()
            total_chunks = mesh.size()
            shard_size = f32_data.shape[shard_dim] // total_chunks
            start = local_rank * shard_size
            end = start + shard_size
            if shard_dim == 0:
                local.copy_(f32_data[start:end])
            else:
                local.copy_(f32_data[:, start:end])
        elif self.packed_weight_f32.shape != f32_data.shape:
            # Shape changed (e.g., cross-format conversion: nvfp4 ↔ block_fp8).
            # Try to reshape first, fall back to re-create if numel differs.
            if f32_data.numel() == self.packed_weight_f32.numel():
                f32_data = f32_data.reshape(self.packed_weight_f32.shape)
                self.packed_weight_f32.data.copy_(f32_data)
            else:
                self.packed_weight_f32 = nn.Parameter(f32_data, requires_grad=False)
        else:
            # Reshape to match (nvfp4 1D → 2D if needed)
            f32_data = f32_data.reshape(self.packed_weight_f32.shape)
            self.packed_weight_f32.data.copy_(f32_data)

    def _read_packed_weight_uint8(self) -> Tensor:
        """Read the packed weight as uint8 data (unpack float32 → uint8)."""
        f32 = self.packed_weight_f32.data
        if hasattr(f32, '_local_tensor'):
            # DTensor: use local shard (during forward, FSDP2 has all-gathered)
            f32 = f32._local_tensor
        return f32.contiguous().view(torch.uint8)

    def _dequantize_weight(self) -> Tensor:
        """Dequantize packed weight to full precision for matmul."""
        M, K = self.out_features, self.in_features
        uint8_data = self._read_packed_weight_uint8()
        if self.quant_format == "block_fp8":
            fp8_w = uint8_data.view(torch.float8_e4m3fn).reshape(M, K)
            scales = self._recover_tensor(
                self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
            )
            return block_fp8_dequantize_gkn(fp8_w, scales, self.quant_group_size)
        else:
            block_scales = self._recover_tensor(
                self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
            )
            global_scale = self._recover_tensor(
                self.weight_global_scale, self._scale_dtypes["weight_global_scale"]
            )
            w = nvfp4_dequantize(
                uint8_data, block_scales,
                global_scale, M * K, self.quant_group_size
            )
            return w.reshape(M, K)

    # ------------------------------------------------------------------
    # Deferred quantization (for meta init with FSDP)
    # ------------------------------------------------------------------

    def quantize_weight(self) -> None:
        """Quantize the deferred `weight` parameter into packed_weight_f32.

        Called after FSDP loads real weights into the `weight` parameter.
        For FSDP2 DTensors, all-gathers the full weight first.
        After quantization, the weight data is zeroed to free memory
        (can't delete the parameter because FSDP2 still tracks it).
        """
        if self.weight is None:
            return  # No deferred weight
        # Check if already quantized: packed_weight_f32 exists and has real data
        # (not just a meta placeholder from from_module)
        if (self.packed_weight_f32 is not None
                and self.packed_weight_f32.device.type != "meta"
                and self.weight_block_scales is not None):
            return  # Already quantized
        with torch.no_grad():
            w = self._to_regular_tensor(self.weight)
            if self.quant_format != "block_fp8":
                self._ema_amax = w.float().abs().max().reshape(1).to(w.device)
            self._quantize_and_store(w)
            del w
            # Free weight storage. Can't delete the parameter because FSDP2
            # tracks it, but we can shrink its underlying storage to zero bytes.
            if hasattr(self.weight, "_local_tensor"):
                # DTensor (FSDP2 sharded): shrink the local shard's storage
                self.weight._local_tensor.data.untyped_storage().resize_(0)
            else:
                self.weight.data.untyped_storage().resize_(0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize base weight
        w = self._dequantize_weight().to(x.dtype)

        result = F.linear(x, w, self.bias)

        # LoRA path (fp32 for numerical stability)
        x_lora = x.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(x_lora, self.lora_A), self.lora_B)
        lora_out = lora_out * self.scaling

        return result + lora_out.to(result.dtype)

    # ------------------------------------------------------------------
    # LoRA merge and re-quantization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_regular_tensor(t: Tensor) -> Tensor:
        """Convert DTensor (from FSDP2) to regular Tensor via all-gather if needed."""
        if hasattr(t, "full_tensor"):
            return t.full_tensor()
        return t

    def get_delta_weight(self) -> Tensor:
        """Compute LoRA delta: lora_B @ lora_A * scaling.

        Handles FSDP2 DTensors by gathering full parameters first.
        """
        lora_A = self._to_regular_tensor(self.lora_A)
        lora_B = self._to_regular_tensor(self.lora_B)
        return (lora_B @ lora_A) * self.scaling

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """
        Merge LoRA into base weight and re-quantize.

        Dequantize → add LoRA delta → EMA-update _ema_amax → re-quantize → store.
        Resets LoRA parameters after merge.

        For block_fp8: skip EMA amax (per-block scales are independent, no global scale).
        """
        with torch.no_grad():
            w = self._dequantize_weight()
            delta = self.get_delta_weight().to(w.dtype)
            w_merged = w + delta
            if self.quant_format == "block_fp8":
                # block_fp8: re-quantize with fresh per-block scales, no global amax
                self._quantize_and_store(w_merged)
            else:
                fresh_amax = w_merged.float().abs().max().reshape(1)
                if self._ema_amax is not None:
                    self._ema_amax.lerp_(fresh_amax.to(self._ema_amax.device), ema_decay)
                else:
                    self._ema_amax = fresh_amax
                self._quantize_and_store(w_merged, global_amax=self._ema_amax)
            # Reset LoRA: kaiming for A, zeros for B.
            # delta = B @ A = 0 (since B=0), but A≠0 so B receives gradients.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.zero_()

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def get_quantized_state_dict(self) -> Dict[str, Tensor]:
        """Return the quantized weight tensors for checkpoint saving."""
        state = {"packed_weight_f32": self.packed_weight_f32}
        if self.weight_block_scales is not None:
            state["weight_block_scales"] = self.weight_block_scales
        if self.weight_global_scale is not None:
            state["weight_global_scale"] = self.weight_global_scale
        return state

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, r={self.r}, lora_alpha={self.lora_alpha}, "
            f"quant_format={self.quant_format}, quant_group_size={self.quant_group_size}"
        )
