"""
QLoRA Linear base class.

QLoRALinear holds shared logic: LoRA parameters, forward pass, AQN, packed
weight I/O, and FSDP2-safe buffer storage.

Format-specific subclasses live in separate modules:
- NvFP4QLoRALinear  → nvfp4_linear.py
- BlockFP8QLoRALinear → block_fp8_linear.py
- NF4QLoRALinear    → nf4_linear.py
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xorl.lora.modules.base import LoraModule


class QLoRALinear(LoraModule, nn.Module):
    """
    Base QLoRA Linear: quantized base weights + trainable LoRA.

    Do not instantiate directly — use NvFP4QLoRALinear, BlockFP8QLoRALinear,
    or NF4QLoRALinear.

    Base weights are stored in quantized format as uint8 buffers.
    During forward, weights are dequantized to compute dtype, then F.linear + LoRA.
    Only LoRA parameters (lora_A, lora_B) are trainable in fp32.
    """

    # Metadata for pre-quantized loading (set by inject_qlora_into_model)
    _is_prequantized: bool = False
    _inline_loaded: bool = False  # True when loaded via checkpoint handler (no deferred I/O needed)
    _merge_sources: Optional[Tuple[str, ...]] = None  # e.g. ("q_proj", "k_proj", "v_proj")
    _source_fqn: Optional[str] = None  # full HF FQN prefix for this module
    _source_quant_format: Optional[str] = None  # checkpoint format (must match quant_format)

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
        enable_aqn: bool = False,
        aqn_alpha: float = 1.0,
    ):
        if type(self) is QLoRALinear:
            raise TypeError(
                "QLoRALinear is abstract. Use NvFP4QLoRALinear, BlockFP8QLoRALinear, or NF4QLoRALinear."
            )
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format = quant_format
        self.quant_group_size = quant_group_size

        # AQN: Adaptive Quantization Noise (training-time regularization).
        # Noise is pre-generated via prefetch_aqn_noise() before forward;
        # falls back to inline generation if not prefetched.
        self.enable_aqn = enable_aqn
        self.aqn_alpha = aqn_alpha
        self._aqn_step_cache: Optional[Tensor] = None
        self._aqn_stream: Optional[torch.cuda.Stream] = None
        self._aqn_noise_buf: Optional[Tensor] = None
        self._aqn_noise_ready: bool = False

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

        # EMA-tracked amax for re-quantization (plain attribute, not parameter — FSDP2 ignores it)
        self._ema_amax: Optional[Tensor] = None

        # NOTE: Subclasses MUST register packed_weight_f32 and scale buffers,
        # then call self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    # ------------------------------------------------------------------
    # Construction: dispatch to subclass based on quant_format
    # ------------------------------------------------------------------

    @classmethod
    def from_module(cls, module: nn.Module, r: int = 16, lora_alpha: int = 16, **kwargs) -> "QLoRALinear":
        """Create QLoRALinear from a bf16 nn.Linear by quantizing its weight.

        When called on the base class, dispatches to the appropriate subclass
        based on ``quant_format`` kwarg (default: "nvfp4").
        """
        if cls is QLoRALinear:
            from xorl.qlora.modules.nvfp4_linear import NvFP4QLoRALinear
            from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
            from xorl.qlora.modules.nf4_linear import NF4QLoRALinear
            fmt = kwargs.pop("quant_format", "nvfp4")
            kwargs.pop("quant_group_size", None)  # subclass sets this
            if fmt == "block_fp8":
                target_cls = BlockFP8QLoRALinear
            elif fmt == "nf4":
                target_cls = NF4QLoRALinear
            else:
                target_cls = NvFP4QLoRALinear
            return target_cls.from_module(module, r=r, lora_alpha=lora_alpha, **kwargs)
        raise NotImplementedError(f"{cls.__name__} must implement from_module")

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
        """Create QLoRALinear from pre-quantized weights (skip quantization)."""
        if cls is QLoRALinear:
            from xorl.qlora.modules.nvfp4_linear import NvFP4QLoRALinear
            from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
            from xorl.qlora.modules.nf4_linear import NF4QLoRALinear
            if quant_format == "block_fp8":
                cls = BlockFP8QLoRALinear
            elif quant_format == "nf4":
                cls = NF4QLoRALinear
            else:
                cls = NvFP4QLoRALinear

        qlora = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=bias,
            device=device,
        )

        uint8_data = cls._to_uint8(packed_weight)
        f32_data = uint8_data.contiguous().view(torch.float32)
        qlora.packed_weight_f32 = nn.Parameter(f32_data, requires_grad=False)

        qlora._scale_dtypes = {}
        if weight_block_scales is not None:
            qlora._scale_dtypes["weight_block_scales"] = weight_block_scales.dtype
            qlora.weight_block_scales = cls._to_uint8(weight_block_scales)
        if weight_global_scale is not None:
            qlora._scale_dtypes["weight_global_scale"] = weight_global_scale.dtype
            qlora.weight_global_scale = cls._to_uint8(weight_global_scale)

        return qlora

    # ------------------------------------------------------------------
    # Shared utility methods
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

    def _write_packed_weight(self, uint8_data: Tensor) -> None:
        """Write uint8 quantized data into the float32 parameter.

        Packs 4 uint8 bytes into 1 float32 element for FSDP2 sharding.
        Handles plain tensors, meta placeholders, and DTensors (under FSDP2).
        """
        f32_data = uint8_data.contiguous().view(torch.float32)
        if self.packed_weight_f32 is None or self.packed_weight_f32.device.type == "meta":
            if self.packed_weight_f32 is not None and self.packed_weight_f32.device.type == "meta":
                target_shape = self.packed_weight_f32.shape
                f32_data = f32_data.reshape(target_shape)
            self.packed_weight_f32 = nn.Parameter(f32_data, requires_grad=False)
        elif hasattr(self.packed_weight_f32, '_local_tensor'):
            local = self.packed_weight_f32._local_tensor
            full_shape = self.packed_weight_f32.shape
            f32_data = f32_data.reshape(full_shape)
            from torch.distributed._tensor import Shard as _Shard
            placements = self.packed_weight_f32.placements
            shard_dim = 0
            for p in placements:
                if isinstance(p, _Shard):
                    shard_dim = p.dim
                    break
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
            if f32_data.numel() == self.packed_weight_f32.numel():
                f32_data = f32_data.reshape(self.packed_weight_f32.shape)
                self.packed_weight_f32.data.copy_(f32_data)
            else:
                self.packed_weight_f32 = nn.Parameter(f32_data, requires_grad=False)
        else:
            f32_data = f32_data.reshape(self.packed_weight_f32.shape)
            self.packed_weight_f32.data.copy_(f32_data)

    @torch.compiler.disable
    def _read_packed_weight_uint8(self) -> Tensor:
        """Read the packed weight as uint8 data (unpack float32 -> uint8)."""
        f32 = self.packed_weight_f32.data
        if hasattr(f32, '_local_tensor'):
            f32 = f32.full_tensor()
        elif f32.dtype != torch.float32:
            raise RuntimeError(
                f"QLoRA packed_weight_f32 has dtype {f32.dtype} (expected float32). "
                f"FSDP2 mixed precision likely cast it, corrupting packed weight data. "
                f"Ensure packed_weight_f32 is deregistered from FSDP param groups."
            )
        return f32.contiguous().view(torch.uint8)

    # ------------------------------------------------------------------
    # Pre-quantized weight loading
    # ------------------------------------------------------------------

    def load_prequantized_weights(self, weight_map: dict, shard_cache: dict, weights_path: str) -> None:
        """Load pre-quantized weights from checkpoint."""
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

        self._load_prequantized(_load_tensor)

    def _load_prequantized(self, _load_tensor) -> None:
        """Load pre-quantized weights. Subclasses must implement."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # ------------------------------------------------------------------

    def quantize_weight(self) -> None:
        """Quantize the bf16 weight parameter into packed form (deferred quantization)."""
        if self.weight is None:
            return
        w = self.weight.data
        from torch.distributed._tensor import DTensor
        if isinstance(w, DTensor):
            w = w.full_tensor()
        self._quantize_and_store(w)
        self.weight = None

    def _quantize_and_store(self, w: Tensor, global_amax: Optional[Tensor] = None) -> None:
        raise NotImplementedError

    @torch.compiler.disable
    def _dequantize_weight(self) -> Tensor:
        raise NotImplementedError

    @torch.compiler.disable
    def _compute_aqn_step(self) -> Tensor:
        raise NotImplementedError

    def merge_weights(self, ema_decay: float = 0.1) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # AQN: Adaptive Quantization Noise
    # ------------------------------------------------------------------

    @torch.compiler.disable
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

    @torch.compiler.disable
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
        if self.enable_aqn and self.training and not self._aqn_noise_ready:
            self._aqn_start_noise(x.device, x.dtype)

        w = self._dequantize_weight().to(x.dtype)

        if self.enable_aqn and self.training:
            w = self._aqn_apply_noise(w)

        result = F.linear(x, w, self.bias)

        x_lora = x.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling

        return result + lora_out.to(result.dtype)

    # ------------------------------------------------------------------
    # LoRA merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_regular_tensor(t: Tensor) -> Tensor:
        if hasattr(t, "full_tensor"):
            return t.full_tensor()
        return t

    def get_delta_weight(self) -> Tensor:
        lora_A = self._to_regular_tensor(self.lora_A)
        lora_B = self._to_regular_tensor(self.lora_B)
        return (lora_B @ lora_A) * self.scaling

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def get_quantized_state_dict(self) -> Dict[str, Tensor]:
        state = {"packed_weight_f32": self.packed_weight_f32}
        if self.weight_block_scales is not None:
            state["weight_block_scales"] = self.weight_block_scales
        if getattr(self, "weight_global_scale", None) is not None:
            state["weight_global_scale"] = self.weight_global_scale
        return state

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}, out_features={self.out_features}",
            f"bias={self.bias is not None}, r={self.r}, lora_alpha={self.lora_alpha}",
            f"quant_format={self.quant_format}, quant_group_size={self.quant_group_size}",
        ]
        if self.enable_aqn:
            parts.append(f"aqn=True(alpha={self.aqn_alpha})")
        return ", ".join(parts)


def prefetch_aqn_noise(
    model: nn.Module,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Pre-generate AQN noise for all QLoRALinear/QLoRAMoeExperts on side CUDA streams.

    Call before ``model.forward()`` each training step. Noise generation
    runs asynchronously; during forward each layer only pays the cost of a
    single fused ``addcmul`` (~0.09 ms/layer vs ~0.47 ms/layer without
    prefetch on H100).

    If not called, forward falls back to inline generation (still correct,
    just slower).

    Returns:
        Number of modules prefetched.
    """
    from xorl.qlora.modules.moe_experts import QLoRAMoeExperts

    count = 0
    for module in model.modules():
        if isinstance(module, QLoRALinear) and module.enable_aqn and module.training:
            dev = device or module.lora_A.device
            if dev.type == "meta":
                dev = torch.device("cuda")
            module._aqn_start_noise(dev, dtype)
            count += 1
        elif isinstance(module, QLoRAMoeExperts) and module.enable_aqn and module.training:
            dev = device or next(module.parameters()).device
            if dev.type == "meta":
                dev = torch.device("cuda")
            module._aqn_start_noise(dev, dtype)
            count += 1
    return count
