"""Block FP8 QLoRA Linear: float8_e4m3fn quantized base weights + trainable LoRA."""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from xorl.ops.quantize import block_fp8_quantize_gkn, block_fp8_dequantize_gkn
from xorl.qlora.modules.linear import QLoRALinear


class BlockFP8QLoRALinear(QLoRALinear):
    """
    Block FP8 QLoRA Linear: float8_e4m3fn quantized base weights + trainable LoRA.

    Stores fp8 weight + f32 per-block scales. No global_scale, no EMA amax.

    Memory: 1 byte/element (2x savings vs bf16) + small scale overhead.
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
            quant_format="block_fp8", quant_group_size=128,
            bias=bias, device=device, enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
        )
        # block_fp8: 1 fp8 byte per element -> in_features bytes -> in_features // 4 float32 elements
        pw_cols = in_features // 4
        self.packed_weight_f32 = nn.Parameter(
            torch.empty(out_features, pw_cols, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        bs_shape = (out_features // 128, (in_features // 128) * 4)
        self.register_buffer("weight_block_scales", torch.empty(*bs_shape, dtype=torch.uint8, device=device))

        self._scale_dtypes = {"weight_block_scales": torch.float32}

        self.reset_lora_parameters()

    @classmethod
    def from_module(cls, module: nn.Module, r: int = 16, lora_alpha: int = 16,
                    enable_aqn: bool = False, aqn_alpha: float = 1.0, **kwargs) -> "BlockFP8QLoRALinear":
        """Create from a bf16 nn.Linear by quantizing its weight to block_fp8."""
        qlora = cls(
            in_features=module.in_features, out_features=module.out_features,
            r=r, lora_alpha=lora_alpha,
            bias=module.bias is not None, device=module.weight.device,
            enable_aqn=enable_aqn, aqn_alpha=aqn_alpha,
        )
        qlora._quantize_and_store(module.weight.detach())
        return qlora

    def _quantize_and_store(self, w: Tensor, global_amax: Optional[Tensor] = None) -> None:
        fp8_w, scales = block_fp8_quantize_gkn(w.float(), self.quant_group_size)
        uint8_data = self._to_uint8(fp8_w)
        self._write_packed_weight(uint8_data)
        self._scale_dtypes = {"weight_block_scales": scales.dtype}
        self.weight_block_scales = self._to_uint8(scales)
        self._aqn_step_cache = None

    @torch.compiler.disable
    def _dequantize_weight(self) -> Tensor:
        M, K = self.out_features, self.in_features
        uint8_data = self._read_packed_weight_uint8()
        fp8_w = uint8_data.view(torch.float8_e4m3fn).reshape(M, K)
        scales = self._recover_tensor(
            self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
        )
        return block_fp8_dequantize_gkn(fp8_w, scales, self.quant_group_size)

    @torch.compiler.disable
    def _compute_aqn_step(self) -> Tensor:
        """block_fp8: 0.125 * block_scale (FP8 E4M3 ULP at unit)."""
        M, K = self.out_features, self.in_features
        bs = self.quant_group_size
        scales = self._recover_tensor(
            self.weight_block_scales, self._scale_dtypes["weight_block_scales"]
        ).float()
        step = scales.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)[:M, :K]
        return (0.125 * step).contiguous()

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        """Merge LoRA into base weight, re-quantize with fresh per-block scales."""
        with torch.no_grad():
            w = self._dequantize_weight()
            delta = self.get_delta_weight().to(w.dtype)
            w_merged = w + delta
            self._quantize_and_store(w_merged)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.zero_()

    def _load_prequantized(self, _load_tensor) -> None:
        """Load pre-quantized block FP8 weights (HF format).

        HF FP8 format: {module}.weight (float8_e4m3fn) + {module}.weight_scale_inv (float32).
        """
        if self._merge_sources is not None:
            packed_parts, scales_parts = [], []
            for src_proj in self._merge_sources:
                src_fqn = f"{self._source_fqn}.{src_proj}"
                packed_parts.append(_load_tensor(f"{src_fqn}.weight"))
                scales_parts.append(_load_tensor(f"{src_fqn}.weight_scale_inv"))
            merged_packed = torch.cat(packed_parts, dim=0)
            merged_scales = torch.cat(scales_parts, dim=0)
        else:
            fqn = self._source_fqn
            merged_packed = _load_tensor(f"{fqn}.weight")
            merged_scales = _load_tensor(f"{fqn}.weight_scale_inv")

        device = self.lora_A.device if self.lora_A.device.type != "meta" else torch.device("cuda")
        self._write_packed_weight(self._to_uint8(merged_packed).to(device))
        self._scale_dtypes = {"weight_block_scales": torch.float32}
        self.weight_block_scales = self._to_uint8(merged_scales.float()).to(device)
        self._ema_amax = None
        self._aqn_step_cache = None
