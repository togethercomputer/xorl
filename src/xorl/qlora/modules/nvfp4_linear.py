"""NvFP4 QLoRA Linear: 4-bit (nvfp4) quantized base weights + trainable LoRA."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from xorl.ops.quantize import nvfp4_quantize, nvfp4_dequantize
from xorl.qlora.modules.linear import QLoRALinear


class NvFP4QLoRALinear(QLoRALinear):
    """
    NVFP4 QLoRA Linear: 4-bit quantized base weights + trainable LoRA.

    Stores packed uint8 data (2 fp4 values per byte) + fp8 block_scales + f32 global_scale.
    EMA-tracked _ema_amax informs global_scale for re-quantization.

    Memory: 0.5 byte/element (4x savings vs bf16) + small scale overhead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        bias: bool = False,
        device: Optional[torch.device] = None,
        enable_aqn: bool = False,
        aqn_alpha: float = 1.0,
    ):
        super().__init__(
            in_features, out_features, r=r, lora_alpha=lora_alpha,
            quant_format="nvfp4", quant_group_size=16,
            bias=bias, device=device, enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
        )
        # nvfp4: 2 fp4 values per byte -> in_features // 2 bytes -> in_features // 8 float32 elements
        pw_cols = in_features // 8
        self.packed_weight_f32 = nn.Parameter(
            torch.empty(out_features, pw_cols, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        bs_shape = (out_features, (in_features // self.quant_group_size) * 4)
        self.register_buffer("weight_block_scales", torch.empty(*bs_shape, dtype=torch.uint8, device=device))
        self.register_buffer("weight_global_scale", torch.empty(4, dtype=torch.uint8, device=device))

        self._scale_dtypes = {
            "weight_block_scales": torch.float32,
            "weight_global_scale": torch.float32,
        }

        self.reset_lora_parameters()

    @classmethod
    def from_module(cls, module: nn.Module, r: int = 16, lora_alpha: int = 16,
                    enable_aqn: bool = False, aqn_alpha: float = 1.0, **kwargs) -> "NvFP4QLoRALinear":
        """Create from a bf16 nn.Linear by quantizing its weight to nvfp4."""
        qlora = cls(
            in_features=module.in_features, out_features=module.out_features,
            r=r, lora_alpha=lora_alpha,
            bias=module.bias is not None, device=module.weight.device,
            enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
        )
        w = module.weight.detach()
        qlora._ema_amax = w.float().abs().max().reshape(1).to(w.device)
        qlora._quantize_and_store(w, global_amax=qlora._ema_amax)
        return qlora

    def _quantize_and_store(self, w: Tensor, global_amax: Optional[Tensor] = None) -> None:
        packed, block_scales, global_scale = nvfp4_quantize(
            w, self.quant_group_size, global_amax=global_amax,
        )
        uint8_data = self._to_uint8(packed)
        self._write_packed_weight(uint8_data)
        self._scale_dtypes = {
            "weight_block_scales": block_scales.dtype,
            "weight_global_scale": global_scale.dtype,
        }
        self.weight_block_scales = self._to_uint8(block_scales)
        self.weight_global_scale = self._to_uint8(global_scale)
        self._aqn_step_cache = None

    @torch.compiler.disable
    def _dequantize_weight(self) -> Tensor:
        M, K = self.out_features, self.in_features
        uint8_data = self._read_packed_weight_uint8()
        block_scales = self._recover_tensor(
            self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
        )
        global_scale = self._recover_tensor(
            self.weight_global_scale, self._scale_dtypes["weight_global_scale"]
        )
        w = nvfp4_dequantize(uint8_data, block_scales, global_scale, M * K, self.quant_group_size)
        return w.reshape(M, K)

    @torch.compiler.disable
    def _compute_aqn_step(self) -> Tensor:
        """nvfp4: 0.5 * block_scale * global_scale (FP4 min linear step)."""
        M, K = self.out_features, self.in_features
        bs = self.quant_group_size
        block_scales = self._recover_tensor(
            self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
        ).float()
        global_scale = self._recover_tensor(
            self.weight_global_scale, self._scale_dtypes["weight_global_scale"]
        ).float()
        effective = block_scales * global_scale
        step = effective.reshape(M, K // bs).repeat_interleave(bs, dim=1)
        return (0.5 * step).contiguous()

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weight, EMA-update _ema_amax, re-quantize."""
        with torch.no_grad():
            w = self._dequantize_weight()
            delta = self.get_delta_weight().to(w.dtype)
            w_merged = w + delta
            fresh_amax = w_merged.float().abs().max().reshape(1)
            if self._ema_amax is not None:
                self._ema_amax.lerp_(fresh_amax.to(self._ema_amax.device), ema_decay)
            else:
                self._ema_amax = fresh_amax
            self._quantize_and_store(w_merged, global_amax=self._ema_amax)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.zero_()

    def _load_prequantized(self, _load_tensor) -> None:
        """Load pre-quantized NVFP4 weights from a modelopt checkpoint."""
        if self._merge_sources is not None:
            packed_parts, block_scales_parts, amax_values = [], [], []
            for src_proj in self._merge_sources:
                src_fqn = f"{self._source_fqn}.{src_proj}"
                packed = _load_tensor(f"{src_fqn}.weight")
                block_scales = _load_tensor(f"{src_fqn}.weight_scale")
                global_scale = _load_tensor(f"{src_fqn}.weight_scale_2")
                amax_values.append(global_scale.float().item() * 6.0 * 448.0)
                block_scales_parts.append(block_scales.float() * global_scale.float())
                packed_parts.append(packed)
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
        self._write_packed_weight(self._to_uint8(merged_packed).to(device))
        self._scale_dtypes = {
            "weight_block_scales": torch.float32,
            "weight_global_scale": torch.float32,
        }
        self.weight_block_scales = self._to_uint8(merged_block_scales).to(device)
        self.weight_global_scale = self._to_uint8(merged_global_scale).to(device)
        self._ema_amax = torch.tensor([max_amax], dtype=torch.float32, device=device)
        self._aqn_step_cache = None
