"""NF4 QLoRA Linear: 4-bit NormalFloat quantized base weights + trainable LoRA."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from xorl.ops.quantize import nf4_quantize, nf4_dequantize
from xorl.ops.quantize.nf4_codec import NF4_MIN_STEP
from xorl.qlora.modules.linear import QLoRALinear


class NF4QLoRALinear(QLoRALinear):
    """
    NF4 QLoRA Linear: 4-bit NormalFloat quantized base weights + trainable LoRA.

    NF4 uses a non-uniform 16-level codebook optimized for normally distributed
    weights. Simpler scale structure than NVFP4: one float32 absmax per group
    (no two-level global_scale + block_scale system).

    No pre-quantized checkpoint needed — quantizes bf16 weights on the fly.

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
            quant_format="nf4", quant_group_size=64,
            bias=bias, device=device, enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
        )
        # nf4: 2 codes per byte -> in_features // 2 bytes -> in_features // 8 float32 elements
        pw_cols = in_features // 8
        self.packed_weight_f32 = nn.Parameter(
            torch.empty(out_features, pw_cols, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        # Per-group float32 scales stored as uint8 for FSDP2 safety
        ns = (in_features // self.quant_group_size) * 4  # 4 uint8 per float32
        self.register_buffer("weight_scales", torch.empty(out_features, ns, dtype=torch.uint8, device=device))

        self._scale_dtypes = {"weight_scales": torch.float32}

        self.reset_lora_parameters()

    @classmethod
    def from_module(cls, module: nn.Module, r: int = 16, lora_alpha: int = 16,
                    enable_aqn: bool = False, aqn_alpha: float = 1.0, **kwargs) -> "NF4QLoRALinear":
        """Create from a bf16 nn.Linear by quantizing its weight to NF4.

        On meta device: defers quantization — keeps ``weight`` as a parameter
        for FSDP to load bf16 into, then call :meth:`quantize_weight` after.
        On real device: quantizes immediately and discards ``weight``.
        """
        qlora = cls(
            in_features=module.in_features, out_features=module.out_features,
            r=r, lora_alpha=lora_alpha,
            bias=module.bias is not None, device=module.weight.device,
            enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
        )
        if module.weight.device.type == "meta":
            qlora.weight = nn.Parameter(module.weight.detach(), requires_grad=False)
        else:
            qlora._quantize_and_store(module.weight.detach())
        return qlora

    def _quantize_and_store(self, w: Tensor, **kwargs) -> None:
        packed, scales = nf4_quantize(w, self.quant_group_size)
        self._write_packed_weight(self._to_uint8(packed))
        self._scale_dtypes = {"weight_scales": scales.dtype}
        self.weight_scales = self._to_uint8(scales)
        self._aqn_step_cache = None

    @torch.compiler.disable
    def _dequantize_weight(self) -> Tensor:
        M, K = self.out_features, self.in_features
        uint8_data = self._read_packed_weight_uint8()
        scales = self._recover_tensor(self.weight_scales, self._scale_dtypes["weight_scales"])
        w = nf4_dequantize(uint8_data, scales, M * K, self.quant_group_size)
        return w.reshape(M, K)

    @torch.compiler.disable
    def _compute_aqn_step(self) -> Tensor:
        """nf4: 0.5 * NF4_MIN_STEP * per_group_scale (minimum quantization resolution)."""
        M, K = self.out_features, self.in_features
        gs = self.quant_group_size
        scales = self._recover_tensor(
            self.weight_scales, self._scale_dtypes["weight_scales"]
        ).float()
        step = scales.reshape(M, K // gs).repeat_interleave(gs, dim=1)
        return (0.5 * NF4_MIN_STEP * step).contiguous()

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weight, re-quantize with fresh per-group scales."""
        with torch.no_grad():
            w = self._dequantize_weight()
            delta = self.get_delta_weight().to(w.dtype)
            w_merged = w + delta
            self._quantize_and_store(w_merged)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.zero_()

    def _load_prequantized(self, _load_tensor) -> None:
        """NF4 pre-quantized loading not supported (no standard checkpoint format)."""
        raise NotImplementedError(
            "NF4 does not support pre-quantized checkpoint loading. "
            "Use from_module() to quantize bf16 weights to NF4 on the fly."
        )
