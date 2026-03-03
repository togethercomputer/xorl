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
from xorl.ops.quantize.hadamard import generate_hadamard_signs, hadamard_rotate
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
        enable_hadamard: bool = False,
        hadamard_block_size: int = 256,
        stochastic_rounding: bool = False,
        clip_ratio: Optional[float] = 1.0,
        enable_aqn: bool = False,
        aqn_alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format = quant_format
        self.quant_group_size = quant_group_size

        # Error reduction flags
        self.enable_hadamard = enable_hadamard
        self.hadamard_block_size = hadamard_block_size
        self.stochastic_rounding = stochastic_rounding
        # clip_ratio=None means auto-search on first quantize; 1.0 means no clipping
        self._clip_ratio = clip_ratio
        # AQN: Adaptive Quantization Noise (training-time regularization).
        # Noise is pre-generated via prefetch_aqn_noise() before forward;
        # falls back to inline generation if not prefetched.
        self.enable_aqn = enable_aqn
        self.aqn_alpha = aqn_alpha
        self._aqn_step_cache: Optional[Tensor] = None
        self._aqn_stream: Optional[torch.cuda.Stream] = None
        self._aqn_noise_buf: Optional[Tensor] = None
        self._aqn_noise_ready: bool = False

        # Hadamard rotation signs (persistent buffer for checkpoint save/load)
        if enable_hadamard:
            signs = generate_hadamard_signs(in_features, hadamard_block_size, device=device)
            self.register_buffer("_hadamard_signs", signs)
        else:
            self.register_buffer("_hadamard_signs", None)

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
        enable_hadamard: bool = False,
        hadamard_block_size: int = 256,
        stochastic_rounding: bool = False,
        clip_ratio: Optional[float] = 1.0,
        enable_aqn: bool = False,
        aqn_alpha: float = 1.0,
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
            enable_hadamard=enable_hadamard,
            hadamard_block_size=hadamard_block_size,
            stochastic_rounding=stochastic_rounding,
            clip_ratio=clip_ratio,
            enable_aqn=enable_aqn,
            aqn_alpha=aqn_alpha,
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

        # Re-quantize with error reduction if hadamard or auto-clip is enabled.
        # Pre-quantized weights were quantized WITHOUT rotation — we must
        # dequant → rotate → re-quantize so forward(Rx) @ (RW)^T = x @ W^T.
        needs_requant = self.enable_hadamard or self._clip_ratio is None
        if needs_requant:
            self._apply_error_reduction_requant()

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
        self._aqn_step_cache = None

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
        self._aqn_step_cache = None

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

    def _apply_error_reduction_requant(self) -> None:
        """Re-quantize pre-quantized weights with error reduction techniques.

        Pre-quantized checkpoints store weights in unrotated space.  When
        hadamard rotation or auto-clip is enabled, we must dequant → rotate →
        re-quantize so that forward(Rx) @ (RW)^T = x @ W^T.
        """
        w = self._dequantize_weight()  # unrotated bf16
        if self.quant_format != "block_fp8":
            self._ema_amax = w.float().abs().max().reshape(1).to(w.device)
        # _quantize_and_store applies hadamard rotation + clip search + stochastic rounding
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

    def _quantize_and_store(self, w: Tensor, global_amax: Optional[Tensor] = None,
                            already_rotated: bool = False) -> None:
        """Quantize a full-precision weight tensor and store as float32 parameter + uint8 scale buffers.

        Applies error reduction techniques if enabled:
        - Hadamard rotation: rotates weight before quantizing to spread outliers
        - MSE-optimal clipping: auto-searches clip_ratio on first call
        - Stochastic rounding: unbiased rounding in FP4/FP8 encoding

        Args:
            w: Weight tensor (unrotated or already rotated if already_rotated=True).
            global_amax: Optional EMA-tracked amax for nvfp4.
            already_rotated: If True, skip hadamard rotation (weight already in rotated space,
                e.g. from merge_weights where dequant returns rotated weight).
        """
        # Hadamard rotation: rotate weight before quantizing (skip if already rotated)
        if self.enable_hadamard and self._hadamard_signs is not None and not already_rotated:
            w = hadamard_rotate(w, self._hadamard_signs.to(w.device), self.hadamard_block_size)

        # MSE-optimal clip_ratio: auto-search on first quantize call
        clip_ratio = self._clip_ratio
        if clip_ratio is None:
            clip_ratio = self._search_optimal_clip_ratio(w)
            self._clip_ratio = clip_ratio

        if self.quant_format == "block_fp8":
            fp8_w, scales = block_fp8_quantize_gkn(
                w.float(), self.quant_group_size,
                clip_ratio=clip_ratio,
                stochastic_rounding=self.stochastic_rounding,
            )
            uint8_data = self._to_uint8(fp8_w)
            self._write_packed_weight(uint8_data)
            self._scale_dtypes = {
                "weight_block_scales": scales.dtype,
            }
            self.weight_block_scales = self._to_uint8(scales)
            self.weight_global_scale = None
        else:
            packed, block_scales, global_scale = nvfp4_quantize(
                w, self.quant_group_size, global_amax=global_amax,
                clip_ratio=clip_ratio,
                stochastic_rounding=self.stochastic_rounding,
            )
            uint8_data = self._to_uint8(packed)
            self._write_packed_weight(uint8_data)
            self._scale_dtypes = {
                "weight_block_scales": block_scales.dtype,
                "weight_global_scale": global_scale.dtype,
            }
            self.weight_block_scales = self._to_uint8(block_scales)
            self.weight_global_scale = self._to_uint8(global_scale)

        # Invalidate AQN step cache (scales changed)
        self._aqn_step_cache = None

    def _search_optimal_clip_ratio(self, w: Tensor) -> float:
        """Grid-search clip_ratio that minimizes quantize-dequantize MSE.

        Tries a range of clip ratios and returns the one with lowest reconstruction
        error. This is a lightweight GPTQ-style calibration using existing amax.

        Args:
            w: Weight tensor to calibrate against.

        Returns:
            Optimal clip_ratio (float between 0.7 and 1.0).
        """
        best_ratio = 1.0
        best_mse = float('inf')
        w_f32 = w.float()

        for alpha in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
            if self.quant_format == "block_fp8":
                fp8_w, scales = block_fp8_quantize_gkn(
                    w_f32, self.quant_group_size, clip_ratio=alpha,
                )
                w_recon = block_fp8_dequantize_gkn(fp8_w, scales, self.quant_group_size)
            else:
                packed, block_scales, global_scale = nvfp4_quantize(
                    w_f32, self.quant_group_size, clip_ratio=alpha,
                )
                M, K = w.shape
                w_recon = nvfp4_dequantize(
                    packed, block_scales, global_scale, M * K, self.quant_group_size,
                ).reshape(M, K)
            mse = (w_f32 - w_recon.float()).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_ratio = alpha

        return best_ratio

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
            # DTensor (FSDP2 sharded, deregistered from param group to avoid
            # mixed-precision bf16 cast that corrupts packed bytes).
            # All-gather in original float32 dtype.
            f32 = f32.full_tensor()
        elif f32.dtype != torch.float32:
            # FSDP2 already unsharded and cast to param_dtype (e.g., bfloat16).
            # The float32→bf16 conversion is lossy and corrupts the packed
            # uint8 byte patterns.  This indicates packed_weight_f32 was not
            # deregistered from FSDP before forward — raise early so the
            # corruption doesn't silently produce wrong results.
            raise RuntimeError(
                f"QLoRA packed_weight_f32 has dtype {f32.dtype} (expected float32). "
                f"FSDP2 mixed precision likely cast it, corrupting packed weight data. "
                f"Ensure packed_weight_f32 is deregistered from FSDP param groups."
            )
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
    # AQN: Adaptive Quantization Noise
    # ------------------------------------------------------------------

    def _compute_aqn_step(self) -> Tensor:
        """Compute per-element quantization step size from block scales.

        nvfp4:     0.5 * block_scale * global_scale  (FP4 min linear step)
        block_fp8: 0.125 * block_scale               (FP8 E4M3 ULP at unit)
        """
        M, K = self.out_features, self.in_features
        bs = self.quant_group_size

        if self.quant_format == "block_fp8":
            scales = self._recover_tensor(
                self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
            ).float()
            step = scales.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)[:M, :K]
            return (0.125 * step).contiguous()
        else:
            block_scales = self._recover_tensor(
                self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
            ).float()
            global_scale = self._recover_tensor(
                self.weight_global_scale, self._scale_dtypes["weight_global_scale"]
            ).float()
            effective = block_scales * global_scale
            step = effective.reshape(M, K // bs).repeat_interleave(bs, dim=1)
            return (0.5 * step).contiguous()

    def _aqn_start_noise(self, device: torch.device, dtype: torch.dtype) -> None:
        """Generate noise on a side CUDA stream (async, returns immediately)."""
        M, K = self.out_features, self.in_features
        if self._aqn_stream is None:
            self._aqn_stream = torch.cuda.Stream(device=device)
        if (self._aqn_noise_buf is None
                or self._aqn_noise_buf.shape != (M, K)
                or self._aqn_noise_buf.dtype != dtype
                or self._aqn_noise_buf.device != device):
            self._aqn_noise_buf = torch.empty(M, K, device=device, dtype=dtype)
        self._aqn_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._aqn_stream):
            self._aqn_noise_buf.normal_()
        self._aqn_noise_ready = True

    def _aqn_apply_noise(self, w_deq: Tensor) -> Tensor:
        """Sync noise stream and apply: w + alpha * noise * step (fused addcmul)."""
        torch.cuda.current_stream(w_deq.device).wait_stream(self._aqn_stream)
        if self._aqn_step_cache is None:
            self._aqn_step_cache = self._compute_aqn_step()
        step = self._aqn_step_cache.to(dtype=w_deq.dtype, device=w_deq.device)
        self._aqn_noise_ready = False
        return torch.addcmul(w_deq, self._aqn_noise_buf, step, value=self.aqn_alpha)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        # AQN: fallback if prefetch_aqn_noise() was not called before forward
        if self.enable_aqn and self.training and not self._aqn_noise_ready:
            self._aqn_start_noise(x.device, x.dtype)

        # Dequantize base weight (stays in rotated space if hadamard enabled)
        w = self._dequantize_weight().to(x.dtype)

        # Hadamard rotation: rotate input to match rotated weight space
        if self.enable_hadamard and self._hadamard_signs is not None:
            x = hadamard_rotate(x, self._hadamard_signs, self.hadamard_block_size)

        # AQN: apply pre-generated noise
        if self.enable_aqn and self.training:
            w = self._aqn_apply_noise(w)

        result = F.linear(x, w, self.bias)

        # LoRA path (fp32 for numerical stability)
        # LoRA learns in rotated space (correct by design: delta is in same basis)
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

        Note: dequantized weight is already in rotated space (if hadamard enabled),
        and LoRA delta was learned in rotated space, so w_merged is already rotated.
        Pass already_rotated=True to skip double-rotation.
        """
        with torch.no_grad():
            w = self._dequantize_weight()
            delta = self.get_delta_weight().to(w.dtype)
            w_merged = w + delta
            # Weight is already in rotated space (dequant returns rotated weight,
            # LoRA delta learned in rotated space)
            already_rotated = self.enable_hadamard
            if self.quant_format == "block_fp8":
                # block_fp8: re-quantize with fresh per-block scales, no global amax
                self._quantize_and_store(w_merged, already_rotated=already_rotated)
            else:
                fresh_amax = w_merged.float().abs().max().reshape(1)
                if self._ema_amax is not None:
                    self._ema_amax.lerp_(fresh_amax.to(self._ema_amax.device), ema_decay)
                else:
                    self._ema_amax = fresh_amax
                self._quantize_and_store(w_merged, global_amax=self._ema_amax,
                                         already_rotated=already_rotated)
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
        parts = [
            f"in_features={self.in_features}, out_features={self.out_features}",
            f"bias={self.bias is not None}, r={self.r}, lora_alpha={self.lora_alpha}",
            f"quant_format={self.quant_format}, quant_group_size={self.quant_group_size}",
        ]
        if self.enable_hadamard:
            parts.append(f"hadamard=True(bs={self.hadamard_block_size})")
        if self.stochastic_rounding:
            parts.append("stochastic_rounding=True")
        if self._clip_ratio is not None and self._clip_ratio != 1.0:
            parts.append(f"clip_ratio={self._clip_ratio:.2f}")
        if self.enable_aqn:
            parts.append(f"aqn=True(alpha={self.aqn_alpha})")
        return ", ".join(parts)


def prefetch_aqn_noise(
    model: nn.Module,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Pre-generate AQN noise for all QLoRALinear layers on side CUDA streams.

    Call before ``model.forward()`` each training step.  Noise generation
    runs asynchronously; during forward each layer only pays the cost of a
    single fused ``addcmul`` (~0.09 ms/layer vs ~0.47 ms/layer without
    prefetch on H100).

    If not called, forward falls back to inline generation (still correct,
    just slower).

    Returns:
        Number of layers prefetched.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, QLoRALinear) and module.enable_aqn and module.training:
            dev = device or module.lora_A.device
            if dev.type == "meta":
                dev = torch.device("cuda")
            module._aqn_start_noise(dev, dtype)
            count += 1
    return count
