"""Checkpoint handler for GPT-OSS models.

GPT-OSS original checkpoints use non-standard key names. Expert weights are
stored in MXFP4 quantized format (``.blocks`` + ``.scales`` pairs) and need
dequantization during loading.  Non-expert weights (attention, norms,
embeddings) are BF16 and just need key renaming.

Load transforms:
    1. Rename keys → xorl internal parameter names
    2. Buffer MXFP4 ``.blocks``/``.scales`` pairs, dequantize to BF16
    3. Transpose expert weights from [E, out, in] → [E, in, out]
    4. Deinterleave HF gate/up tensors when needed
    5. Slice expert dimension for EP when ep_size > 1

Save transforms:
    1. Preserve xorl's normalized parameter names so saved checkpoints remain
       directly reloadable by xorl.
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ...checkpoint_handlers.base import CheckpointHandler


# ============================================================================
# MXFP4 dequantization (CPU-compatible, no Triton dependency)
# ============================================================================

# FP4 E2M1 lookup table: 4-bit code → float32 value
# Codes 0-7 are positive, 8-15 are negative (sign bit = bit 3)
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _mxfp4_dequantize_cpu(
    blocks: Tensor,
    scales: Tensor,
) -> Tensor:
    """Dequantize MXFP4 packed expert weights to bfloat16.

    Args:
        blocks: uint8 tensor of shape [..., num_groups, 16] (packed FP4 pairs)
        scales: uint8 tensor of shape [..., num_groups] (E8M0 exponents)

    Returns:
        bfloat16 tensor of shape [..., num_groups * 32] (32 values per group)
    """
    # Unpack two FP4 values per byte: low nibble = even index, high nibble = odd
    lo = (blocks & 0x0F).to(torch.int64)
    hi = (blocks >> 4).to(torch.int64)

    # Lookup float values: lo/hi each [..., num_groups, 16]
    lut = _FP4_E2M1_LUT.to(blocks.device)
    val_lo = lut[lo]  # [..., G, 16]
    val_hi = lut[hi]  # [..., G, 16]

    # Interleave: even positions = lo, odd positions = hi → [..., G, 32]
    interleaved = torch.stack([val_lo, val_hi], dim=-1)  # [..., G, 16, 2]
    interleaved = interleaved.reshape(*blocks.shape[:-1], 32)  # [..., G, 32]

    # E8M0 scale: 2^(scale - 127)
    scale_f32 = torch.pow(2.0, scales.to(torch.float32) - 127.0)  # [..., G]

    # Broadcast multiply: [..., G, 1] * [..., G, 32]
    result = interleaved * scale_f32.unsqueeze(-1)

    # Flatten groups: [..., G, 32] → [..., G * 32]
    result = result.reshape(*scales.shape[:-1], -1)

    return result.to(torch.bfloat16)


# ============================================================================
# Key mapping: original checkpoint → xorl internal
# ============================================================================

_LOAD_KEY_MAP = [
    # Embeddings
    (re.compile(r"^embedding\.weight$"), "model.embed_tokens.weight", False),
    (re.compile(r"^unembedding\.weight$"), "lm_head.weight", False),
    # Final norm
    (re.compile(r"^norm\.scale$"), "model.norm.weight", False),
    # Attention norm
    (re.compile(r"^block\.(\d+)\.attn\.norm\.scale$"), r"model.layers.\1.input_layernorm.weight", False),
    # Attention output
    (re.compile(r"^block\.(\d+)\.attn\.out\.(weight|bias)$"), r"model.layers.\1.self_attn.o_proj.\2", False),
    # Attention sinks
    (re.compile(r"^block\.(\d+)\.attn\.sinks$"), r"model.layers.\1.self_attn.sinks", False),
    # MLP norm
    (re.compile(r"^block\.(\d+)\.mlp\.norm\.scale$"), r"model.layers.\1.post_attention_layernorm.weight", False),
    # Router gate
    (re.compile(r"^block\.(\d+)\.mlp\.gate\.(weight|bias)$"), r"model.layers.\1.mlp.gate.\2", False),
    # Expert biases (BF16, no transpose needed)
    (re.compile(r"^block\.(\d+)\.mlp\.mlp1_bias$"), r"model.layers.\1.mlp.experts.gate_up_bias", False),
    (re.compile(r"^block\.(\d+)\.mlp\.mlp2_bias$"), r"model.layers.\1.mlp.experts.down_bias", False),
]

# Pattern for MXFP4 expert weight keys (original checkpoint format)
_MXFP4_PATTERN = re.compile(r"^block\.(\d+)\.mlp\.(mlp[12])_weight\.(blocks|scales)$")

# Internal names for expert weights after dequant
_EXPERT_INTERNAL_NAMES = {
    "mlp1": "mlp.experts.gate_up_proj",
    "mlp2": "mlp.experts.down_proj",
}


# ============================================================================
# HF checkpoint format support
# ============================================================================

# HF format key renames → xorl internal
_HF_LOAD_KEY_MAP = [
    # Router → gate
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.router\.(weight|bias)$"), r"model.layers.\1.mlp.gate.\2", False),
    # Expert biases (gate_up_proj_bias → gate_up_bias, drop _proj)
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj_bias$"),
        r"model.layers.\1.mlp.experts.gate_up_bias",
        False,
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj_bias$"),
        r"model.layers.\1.mlp.experts.down_bias",
        False,
    ),
]

# HF format MXFP4: gate_up_proj_blocks/_scales, down_proj_blocks/_scales
_HF_MXFP4_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj_(blocks|scales)$")

_HF_EXPERT_INTERNAL_NAMES = {
    "gate_up": "mlp.experts.gate_up_proj",
    "down": "mlp.experts.down_proj",
}

# HF format separate q/k/v projections that need fusing
_HF_QKV_PATTERN = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$")

# Original and xorl-local fused qkv projections
_ORIGINAL_FUSED_QKV_PATTERN = re.compile(r"^block\.(\d+)\.attn\.qkv\.(weight|bias)$")
_INTERNAL_FUSED_QKV_PATTERN = re.compile(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")

# Keys holding stacked expert tensors (need EP slicing)
_EXPERT_KEY_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|gate_up_bias|down_proj|down_bias)$")


def _detect_checkpoint_format(checkpoint_keys: Optional[set[str]]) -> str:
    """Best-effort source format detection for ambiguous GPT-OSS keys."""
    if not checkpoint_keys:
        return "xorl"

    if any(
        key.startswith("block.") or key in {"embedding.weight", "unembedding.weight", "norm.scale"}
        for key in checkpoint_keys
    ):
        return "original"

    if any(
        ".mlp.gate." in key
        or ".self_attn.qkv_proj." in key
        or key.endswith(".mlp.experts.gate_up_bias")
        or key.endswith(".mlp.experts.down_bias")
        for key in checkpoint_keys
    ):
        return "xorl"

    if any(
        ".mlp.router." in key
        or key.endswith(".mlp.experts.gate_up_proj_bias")
        or key.endswith(".mlp.experts.down_proj_bias")
        for key in checkpoint_keys
    ):
        return "hf"

    return "xorl"


def _remap_key(key: str, key_map, tensor: Tensor) -> Optional[Tuple[str, Tensor]]:
    """Apply the first matching pattern from *key_map*."""
    for pattern, replacement, transpose in key_map:
        new_key, n = pattern.subn(replacement, key)
        if n > 0:
            if transpose and tensor.ndim == 3:
                tensor = tensor.transpose(1, 2).contiguous()
            return new_key, tensor
    return None


def _deinterleave_gate_up(tensor: Tensor) -> Tensor:
    """Convert interleaved ``[g0,u0,g1,u1,...]`` to ``[g0,g1,...|u0,u1,...]`` in the last dim.

    The original GPT-OSS checkpoint stores gate and up values interleaved.
    The standard xorl MoE layout concatenates gate then up.
    """
    gate = tensor[..., 0::2]
    up = tensor[..., 1::2]
    return torch.cat([gate, up], dim=-1).contiguous()


def _split_fused_qkv(
    tensor: Tensor,
    *,
    layer_idx: str,
    wb: str,
    q_dim: int,
    kv_dim: int,
) -> List[Tuple[str, Tensor]]:
    """Split fused qkv tensor into q/k/v tensors under xorl/HF-style names."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    q = tensor[:q_dim].contiguous()
    k = tensor[q_dim : q_dim + kv_dim].contiguous()
    v = tensor[q_dim + kv_dim : q_dim + (2 * kv_dim)].contiguous()
    return [
        (f"{prefix}.q_proj.{wb}", q),
        (f"{prefix}.k_proj.{wb}", k),
        (f"{prefix}.v_proj.{wb}", v),
    ]


class GptOssCheckpointHandler(CheckpointHandler):
    """Checkpoint handler for GPT-OSS models.

    Handles original GPT-OSS checkpoints, HF-style GPT-OSS checkpoints, and
    xorl-local normalized checkpoints.

    Args:
        num_experts: Total number of experts.
        num_attention_heads: Number of query heads (used to split fused qkv).
        num_key_value_heads: Number of key/value heads (used to split fused qkv).
        head_dim: Attention head dimension.
        ep_rank: This rank's expert-parallel index (default 0).
        ep_size: Total number of expert-parallel ranks (default 1).
        checkpoint_keys: Optional full checkpoint key set for source-format detection.
        skip_qkv_merge: When True, emit separate q/k/v tensors instead of fused
            ``qkv_proj`` tensors (used after ``merge_qkv=False`` / TP unfuse).
    """

    def __init__(
        self,
        num_experts: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        ep_rank: int = 0,
        ep_size: int = 1,
        checkpoint_keys: Optional[set[str]] = None,
        skip_qkv_merge: bool = False,
    ):
        self._num_experts = num_experts
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._local_num_experts = num_experts // ep_size
        self._expert_start = ep_rank * self._local_num_experts
        self._expert_end = self._expert_start + self._local_num_experts
        self._q_dim = num_attention_heads * head_dim
        self._kv_dim = num_key_value_heads * head_dim
        self._skip_qkv_merge = skip_qkv_merge
        self._checkpoint_format = _detect_checkpoint_format(checkpoint_keys)

        # Buffer for MXFP4 pairs: {(layer_idx, mlp_name): {"blocks": ..., "scales": ...}}
        self._mxfp4_buffer: Dict[Tuple[str, str], Dict[str, Tensor]] = {}
        # Buffer for HF separate q/k/v → fused qkv: {(layer_idx, "weight"|"bias"): {"q": ..., "k": ..., "v": ...}}
        self._qkv_buffer: Dict[Tuple[str, str], Dict[str, Tensor]] = {}

    def _slice_expert_tensor_for_ep(self, key: str, tensor: Tensor) -> Tensor:
        """Slice stacked expert tensor along dim 0 for expert parallelism."""
        if self._ep_size <= 1:
            return tensor
        if _EXPERT_KEY_PATTERN.match(key):
            return tensor[self._expert_start : self._expert_end].contiguous()
        return tensor

    def _handle_mxfp4_dequant(
        self,
        layer_idx: str,
        mlp_name: str,
        part: str,
        tensor: Tensor,
        internal_names: dict,
    ) -> List[Tuple[str, Tensor]]:
        """Buffer MXFP4 blocks/scales and dequantize when both arrive."""
        buf_key = (layer_idx, mlp_name)

        if buf_key not in self._mxfp4_buffer:
            self._mxfp4_buffer[buf_key] = {}
        self._mxfp4_buffer[buf_key][part] = tensor

        if len(self._mxfp4_buffer[buf_key]) < 2:
            return []

        parts = self._mxfp4_buffer.pop(buf_key)
        weight_bf16 = _mxfp4_dequantize_cpu(parts["blocks"], parts["scales"])
        # [E, out_dim, in_dim] → [E, in_dim, out_dim]
        weight_bf16 = weight_bf16.transpose(1, 2).contiguous()

        # Deinterleave gate_up from [g0,u0,g1,u1,...] to [g0,g1,...|u0,u1,...].
        if mlp_name in ("mlp1", "gate_up"):
            weight_bf16 = _deinterleave_gate_up(weight_bf16)

        internal_name = f"model.layers.{layer_idx}.{internal_names[mlp_name]}"
        weight_bf16 = self._slice_expert_tensor_for_ep(internal_name, weight_bf16)
        return [(internal_name, weight_bf16)]

    def on_load_weight(self, key: str, tensor: Tensor) -> List[Tuple[str, Tensor]]:
        # --- Original format: MXFP4 expert weights ---
        m = _MXFP4_PATTERN.match(key)
        if m is not None:
            return self._handle_mxfp4_dequant(
                m.group(1),
                m.group(2),
                m.group(3),
                tensor,
                _EXPERT_INTERNAL_NAMES,
            )

        # --- HF format: MXFP4 expert weights ---
        m = _HF_MXFP4_PATTERN.match(key)
        if m is not None:
            return self._handle_mxfp4_dequant(
                m.group(1),
                m.group(2),
                m.group(3),
                tensor,
                _HF_EXPERT_INTERNAL_NAMES,
            )

        # --- Original format: fused qkv ---
        m = _ORIGINAL_FUSED_QKV_PATTERN.match(key)
        if m is not None:
            layer_idx, wb = m.group(1), m.group(2)
            if self._skip_qkv_merge:
                return _split_fused_qkv(tensor, layer_idx=layer_idx, wb=wb, q_dim=self._q_dim, kv_dim=self._kv_dim)
            internal_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.{wb}"
            return [(internal_name, tensor)]

        # --- xorl-local format: fused qkv ---
        m = _INTERNAL_FUSED_QKV_PATTERN.match(key)
        if m is not None:
            layer_idx, wb = m.group(1), m.group(2)
            if self._skip_qkv_merge:
                return _split_fused_qkv(tensor, layer_idx=layer_idx, wb=wb, q_dim=self._q_dim, kv_dim=self._kv_dim)
            return [(key, tensor)]

        # --- HF/xorl-unfused format: separate q/k/v ---
        m = _HF_QKV_PATTERN.match(key)
        if m is not None:
            if self._skip_qkv_merge:
                return [(key, tensor)]

            layer_idx, qkv, wb = m.group(1), m.group(2), m.group(3)
            buf_key = (layer_idx, wb)

            if buf_key not in self._qkv_buffer:
                self._qkv_buffer[buf_key] = {}
            self._qkv_buffer[buf_key][qkv] = tensor

            if len(self._qkv_buffer[buf_key]) < 3:
                return []

            parts = self._qkv_buffer.pop(buf_key)
            fused = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            internal_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.{wb}"
            return [(internal_name, fused)]

        # --- xorl/HF stacked expert tensors ---
        if _EXPERT_KEY_PATTERN.match(key):
            if key.endswith("gate_up_proj") and self._checkpoint_format == "hf":
                tensor = _deinterleave_gate_up(tensor)
            tensor = self._slice_expert_tensor_for_ep(key, tensor)
            return [(key, tensor)]

        # --- HF format: simple renames (router→gate, expert biases) ---
        result = _remap_key(key, _HF_LOAD_KEY_MAP, tensor)
        if result is not None:
            new_key, tensor = result
            if "gate_up_bias" in new_key:
                tensor = _deinterleave_gate_up(tensor)
            tensor = self._slice_expert_tensor_for_ep(new_key, tensor)
            return [(new_key, tensor)]

        # --- Original format: standard key remapping ---
        result = _remap_key(key, _LOAD_KEY_MAP, tensor)
        if result is not None:
            new_key, tensor = result
            if "gate_up_bias" in new_key:
                tensor = _deinterleave_gate_up(tensor)
            tensor = self._slice_expert_tensor_for_ep(new_key, tensor)
            return [(new_key, tensor)]

        # Pass-through (HF keys that already match xorl internal names)
        return [(key, tensor)]

    def on_load_complete(self) -> List[Tuple[str, Tensor]]:
        if self._mxfp4_buffer:
            incomplete = [f"layer {l} {m}" for (l, m) in self._mxfp4_buffer.keys()]
            warnings.warn(f"Incomplete MXFP4 pairs at load completion: {incomplete}")
            self._mxfp4_buffer.clear()
        if self._qkv_buffer:
            incomplete = [f"layer {l} qkv {wb}" for (l, wb) in self._qkv_buffer.keys()]
            warnings.warn(f"Incomplete QKV groups at load completion: {incomplete}")
            self._qkv_buffer.clear()
        return []
