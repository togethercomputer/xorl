"""HF -> xorl state-dict loader for DeepSeek-V4.

Single-host batch loader (the streaming :class:`CheckpointHandler` infra is
not used here — Phase 5 V0 targets single-host LoRA training where the full
state-dict fits in CPU memory during load).

Three transforms beyond the name-renaming map:

1. **FP8 block dequantization.** Every Linear weight on disk is FP8-E4M3
   with a paired ``.scale`` tensor in UE8M0; block size is ``[128, 128]``.
   We dequantize to BF16 at load time so the rest of the stack can ignore
   the quantization.

2. **APE hotfix undo (C4 layers only).** miles applies a permutation to
   the ``compressor.ape`` parameter when exporting to HF/SGLang format
   (matches SGLang's runtime kernel layout). We invert it on load so the
   xorl compressor sees the natural layout.

3. **Per-expert fusion.** HF stores routed experts as
   ``layers.{N}.ffn.experts.{eid}.{w1,w2,w3}.weight``; xorl's ``MoEExperts``
   uses fused stacked tensors ``[num_experts, hidden, 2*intermediate]``
   (gate+up packed) and ``[num_experts, intermediate, hidden]`` (down).

Note: ``mtp.*`` (multi-token-prediction) tensors are silently skipped — V0
does not implement MTP. Same for ``.scale`` tensors after their paired
``.weight`` has been consumed.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from xorl.models.checkpoint_handlers.base import CheckpointHandler


_LAYER_RE = re.compile(r"layers\.(\d+)\.(.+)")


@dataclass
class LoadSummary:
    loaded: int = 0
    fp8_dequantized: int = 0
    ape_unhotfixed: int = 0
    experts_fused: int = 0
    skipped_mtp: int = 0
    unmapped: list[str] = field(default_factory=list)
    missing_in_model: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


# ``compressor.ape`` for compress_ratio==4 (the DSA-indexer C4 path) is
# stored in HF as ``[4, 2*head_dim]`` and view-reshaped to ``[8, head_dim]``
# during the un-hotfix. The factor 4 is fixed by the C4 path and not by any
# variable; the named constant is purely for readability.
_C4_APE_HEAD_GROUPS = 4


def _undo_ape_hotfix(param: torch.Tensor, *, expected_head_dim: int | None = None) -> torch.Tensor:
    """Inverse of miles ``_apply_ape_hotfix_mirror``.

    HF on disk stores the ``compressor.ape`` of C4 layers in SGLang kernel
    layout (a row/column permutation). The natural layout (xorl + the
    Megatron training-time format) is recovered by:

        view (4, 2D) -> (8, D), split halves along dim 0, cat on dim -1.

    Pass ``expected_head_dim`` when the caller knows the model's
    ``compressor.head_dim``; that asserts the inferred D matches and
    catches future config changes that would otherwise produce silently
    wrong-shape output.
    """
    assert param.dim() == 2 and param.shape[0] == _C4_APE_HEAD_GROUPS, (
        f"_undo_ape_hotfix expects ({_C4_APE_HEAD_GROUPS}, 2D); got {tuple(param.shape)}"
    )
    assert param.shape[-1] % 2 == 0, f"_undo_ape_hotfix expects even last-dim 2D; got {tuple(param.shape)}"
    head_dim = param.shape[-1] // 2
    if expected_head_dim is not None and head_dim != expected_head_dim:
        raise AssertionError(
            f"_undo_ape_hotfix shape mismatch: inferred head_dim={head_dim} "
            f"from param.shape={tuple(param.shape)} but expected {expected_head_dim}"
        )
    eight = param.reshape(2 * _C4_APE_HEAD_GROUPS, head_dim)
    a = eight[:_C4_APE_HEAD_GROUPS]
    b = eight[_C4_APE_HEAD_GROUPS:]
    return torch.cat([a, b], dim=-1).contiguous()


# OCP MXFP4 E2M1 lookup table (4 bits -> float).
# Bit pattern (sign | exp[2] | mantissa[1]) -> value:
#   0000 +0    0001 +0.5  0010 +1   0011 +1.5  0100 +2   0101 +3   0110 +4   0111 +6
#   1000 -0    1001 -0.5  1010 -1   1011 -1.5  1100 -2   1101 -3   1110 -4   1111 -6
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequantize_mxfp4_packed_int8(
    packed_int8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 2-D MXFP4-packed weight given its UE8M0 per-block scale.

    Storage convention (per OCP MX spec): each ``int8`` byte packs two
    consecutive FP4-E2M1 values — low nibble (bits 0-3) is the even-index
    element, high nibble (bits 4-7) is the odd-index element. Block size
    along the inner dim is typically 32.

    Args:
        packed_int8: ``[out, in_packed]`` ``int8``, where the logical inner
            dim is ``in = 2 * in_packed``.
        scale_e8m0: ``[out, in / block_size]`` ``float8_e8m0fnu`` (or any
            float dtype that ``.float()`` decodes losslessly).
        block_size: number of FP4 elements per scale group along the inner
            dim. DSv4 routed experts use 32.
        out_dtype: dtype of the dequantized output.

    Returns:
        ``[out, 2 * in_packed]`` ``out_dtype``.
    """
    assert packed_int8.dim() == 2, f"expected 2-D packed weight, got {packed_int8.shape}"
    out_dim, in_packed = packed_int8.shape
    in_dim = 2 * in_packed

    u8 = packed_int8.view(torch.uint8).to(torch.long)
    lo = u8 & 0x0F
    hi = (u8 >> 4) & 0x0F

    lut = _FP4_E2M1_LUT.to(packed_int8.device)
    fp_lo = lut[lo]  # [out, in_packed]
    fp_hi = lut[hi]  # [out, in_packed]

    full = torch.empty(out_dim, in_dim, dtype=torch.float32, device=packed_int8.device)
    full[:, 0::2] = fp_lo
    full[:, 1::2] = fp_hi

    s_full = scale_e8m0.float().repeat_interleave(block_size, dim=-1)[:, :in_dim]
    return (full * s_full).to(out_dtype)


def _dequantize_fp8_block(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 2-D block-FP8-E4M3 weight given its UE8M0 scale.

    Args:
        weight_fp8: ``[out, in]`` ``float8_e4m3fn``.
        scale: ``[ceil(out/B0), ceil(in/B1)]`` ``float8_e8m0fnu`` (or a
            float dtype that ``.float()`` converts losslessly).
        block_size: ``(B0, B1)``. For DSv4 this is ``(128, 128)``.
        out_dtype: dtype of the dequantized output (default ``bfloat16``).

    Returns:
        ``[out, in]`` ``out_dtype`` tensor where each element is
        ``weight[i, j] * scale[i // B0, j // B1]``.
    """
    assert weight_fp8.dim() == 2, f"expected 2-D weight, got {weight_fp8.shape}"
    out_dim, in_dim = weight_fp8.shape
    B0, B1 = block_size
    # PyTorch's ``.float()`` on float8_e4m3fn / float8_e8m0fnu does the
    # IEEE-style decode; promoting both to fp32 first is safe and matches
    # SGLang's ``per_block_cast_to_fp8`` reference.
    w_fp32 = weight_fp8.float()
    s_fp32 = scale.float()
    # Broadcast the scale up to the full ``[out, in]`` grid via repeat-along
    # block dims. ``repeat_interleave`` keeps memory contiguous for the
    # subsequent multiply.
    s_full = s_fp32.repeat_interleave(B0, dim=0)[:out_dim].repeat_interleave(B1, dim=1)[:, :in_dim]
    return (w_fp32 * s_full).to(out_dtype)


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------


# Static (non-layer) tensors. ``None`` skip-marks tensors not present in
# the V0 xorl model (the MTP head + its sub-tree).
_TOP_LEVEL = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_fn": "model.hc_head_fn",
    "hc_head_base": "model.hc_head_base",
    "hc_head_scale": "model.hc_head_scale",
}

# Per-attention sub-tree. The HF name suffix after ``layers.{N}.attn.``
# maps to an xorl name suffix after ``model.layers.{N}.self_attn.``.
_ATTN_SUFFIX = {
    "wq_a.weight": "wq_a.weight",
    "q_norm.weight": "q_norm.weight",
    "wq_b.weight": "wq_b.weight",
    "wkv.weight": "wkv.weight",
    "kv_norm.weight": "kv_norm.weight",
    "wo_a.weight": "wo_a.weight",
    "wo_b.weight": "wo_b.weight",
    "attn_sink": "attn_sink",
    # Compressor (C128 + C4 layers).
    "compressor.ape": "compressor.ape",
    "compressor.wkv.weight": "compressor.wkv.weight",
    "compressor.wgate.weight": "compressor.wgate.weight",
    "compressor.norm.weight": "compressor.norm.weight",
    # Indexer (C4 layers only).
    "indexer.wq_b.weight": "indexer.linear_wq_b.weight",
    "indexer.weights_proj.weight": "indexer.linear_weights_proj.weight",
    "indexer.compressor.ape": "indexer.compressor.ape",
    "indexer.compressor.wkv.weight": "indexer.compressor.wkv.weight",
    "indexer.compressor.wgate.weight": "indexer.compressor.wgate.weight",
    "indexer.compressor.norm.weight": "indexer.compressor.norm.weight",
}

# Per-FFN sub-tree (non-expert tensors). Expert tensors are handled by the
# fusion path because they need stacking + w1/w3 packing.
_FFN_NONEXPERT_SUFFIX = {
    "gate.weight": "gate.weight",
    "gate.bias": "gate.e_score_correction_bias",  # noaux_tc bias
    "gate.tid2eid": "tid2eid",  # hash table (sits on the block, not on the gate)
    "shared_experts.w1.weight": "shared_experts.gate_proj.weight",
    "shared_experts.w2.weight": "shared_experts.down_proj.weight",
    "shared_experts.w3.weight": "shared_experts.up_proj.weight",
}

# Per-layer top-level params (HC + norms).
_LAYER_TOPLEVEL = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "hc_attn_fn": "hc_attn_fn",
    "hc_attn_base": "hc_attn_base",
    "hc_attn_scale": "hc_attn_scale",
    "hc_ffn_fn": "hc_ffn_fn",
    "hc_ffn_base": "hc_ffn_base",
    "hc_ffn_scale": "hc_ffn_scale",
}


_EXPERT_RE = re.compile(r"ffn\.experts\.(\d+)\.(w[123])\.weight")


def _hf_to_xorl_name(hf_name: str) -> Optional[str]:
    """Map a non-expert HF name to its xorl counterpart.

    Returns ``None`` for tensors we deliberately skip (``mtp.*``,
    ``.scale`` tensors that are paired with a ``.weight`` and consumed via
    dequant, and per-expert ``ffn.experts.*`` tensors that go through the
    fusion path).
    """
    if hf_name.startswith("mtp."):
        return None  # MTP head — not implemented in V0.
    if hf_name in _TOP_LEVEL:
        return _TOP_LEVEL[hf_name]

    m = _LAYER_RE.match(hf_name)
    if m is None:
        return None
    layer_idx, rest = m.groups()
    layer_prefix = f"model.layers.{layer_idx}"

    if rest in _LAYER_TOPLEVEL:
        return f"{layer_prefix}.{_LAYER_TOPLEVEL[rest]}"
    if rest.startswith("attn."):
        suffix = rest[len("attn.") :]
        if suffix in _ATTN_SUFFIX:
            return f"{layer_prefix}.self_attn.{_ATTN_SUFFIX[suffix]}"
        return None
    if rest.startswith("ffn."):
        suffix = rest[len("ffn.") :]
        if suffix in _FFN_NONEXPERT_SUFFIX:
            xorl_suffix = _FFN_NONEXPERT_SUFFIX[suffix]
            # ``tid2eid`` lives on the block itself, not on the gate.
            if xorl_suffix == "tid2eid":
                return f"{layer_prefix}.mlp.tid2eid"
            return f"{layer_prefix}.mlp.{xorl_suffix}"
        return None
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_mxfp4_packed(t: torch.Tensor, *, has_scale: bool) -> bool:
    """A tensor is MXFP4-packed iff it is ``int8`` AND ships with a paired
    ``.scale`` sidecar. The dtype check alone is too loose — any future
    ``int8`` weight (e.g. an integer index buffer) would otherwise be
    silently dequantized into garbage. Callers must locate the paired
    scale in the source state dict and pass ``has_scale``."""
    return t.dtype == torch.int8 and has_scale


def _is_compress_ratio_4(layer_idx: int, compress_ratios: Optional[list[int]]) -> bool:
    if compress_ratios is None:
        return False
    if 0 <= layer_idx < len(compress_ratios):
        return compress_ratios[layer_idx] == 4
    return False


def _fillable_state(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], set[str]]:
    """Return load targets, excluding non-persistent buffers.

    ``named_buffers()`` includes non-persistent runtime caches such as
    ``freqs_cis``. Those buffers are derived from config, not checkpoints, so
    strict loading must not require them to appear in HF/DCP state.
    """
    model_params = dict(model.named_parameters())
    model_buffers: dict[str, torch.Tensor] = {}
    for module_prefix, module in model.named_modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for name, buf in module._buffers.items():
            if buf is None or name in non_persistent:
                continue
            full_name = f"{module_prefix}.{name}" if module_prefix else name
            model_buffers[full_name] = buf
    fillable = set(model_params) | set(model_buffers)
    return model_params, model_buffers, fillable


def load_hf_state_dict_into_model(
    model: torch.nn.Module,
    hf_state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = False,
    dequantize_fp8: bool = True,
    target_dtype: torch.dtype = torch.bfloat16,
) -> LoadSummary:
    """Load an HF DeepseekV4 state-dict into an xorl ``DeepseekV4ForCausalLM``.

    Args:
        model: an xorl ``DeepseekV4ForCausalLM`` (or compatible) instance.
        hf_state_dict: full HF state-dict, name -> tensor. Including ``.scale``
            entries when the disk format is FP8.
        strict: when True, raises if any expected model parameter wasn't filled,
            or if any HF tensor was unmapped (other than the deliberate skips
            ``mtp.*``).
        dequantize_fp8: when True (default), block-FP8 weights are dequantized
            to ``target_dtype`` using their paired ``.scale``. When False, FP8
            weights are passed through (the caller had better know what to do).
        target_dtype: dtype to cast dequantized FP8 weights into. Defaults to
            ``bfloat16``; the fp32 ``attn_sink`` / ``hc_*`` / compressor params
            are passed through untouched.

    Returns:
        :class:`LoadSummary` with counts + lists of unmapped / missing names.
    """
    config = model.config
    compress_ratios = config.compress_ratios
    # Compressor head_dim drives the APE hotfix shape; pass it down so
    # ``_undo_ape_hotfix`` can sanity-check the inferred reshape against
    # the model's actual config rather than trusting tensor shape alone.
    # The two compressors live at different head_dims:
    #   .self_attn.compressor.ape          -> config.head_dim
    #   .self_attn.indexer.compressor.ape  -> config.index_head_dim
    attn_compressor_head_dim = config.head_dim
    indexer_compressor_head_dim = getattr(config, "index_head_dim", config.head_dim)

    summary = LoadSummary()

    # First pass: collect per-layer expert tensors so we can fuse them.
    # ``expert_buf[layer_idx][("w1"|"w3"|"w2")][expert_idx] = tensor``.
    expert_buf: dict[int, dict[str, dict[int, torch.Tensor]]] = {}

    # Build a set of model parameter names so we can validate.
    model_params, model_buffers, fillable = _fillable_state(model)
    filled: set[str] = set()

    # Walk HF state-dict.
    for hf_name, tensor in hf_state_dict.items():
        # Skip ``.scale`` entries — they're consumed when their ``.weight`` is.
        if hf_name.endswith(".scale"):
            continue

        # MTP skip.
        if hf_name.startswith("mtp."):
            summary.skipped_mtp += 1
            continue

        # Per-expert routed weights -> deferred fusion.
        m = _LAYER_RE.match(hf_name)
        if m is not None:
            layer_idx_str, rest = m.groups()
            layer_idx = int(layer_idx_str)
            em = _EXPERT_RE.match(rest)
            if em is not None:
                expert_idx = int(em.group(1))
                w_name = em.group(2)  # w1 | w2 | w3
                # Dequantize on the spot if needed.
                t = tensor
                if dequantize_fp8:
                    scale_key = hf_name.replace(".weight", ".scale")
                    if _is_fp8(t) and scale_key in hf_state_dict:
                        t = _dequantize_fp8_block(t, hf_state_dict[scale_key], out_dtype=target_dtype)
                        summary.fp8_dequantized += 1
                    elif _is_mxfp4_packed(t, has_scale=scale_key in hf_state_dict):
                        # DSv4 routed experts: each int8 byte packs 2 FP4 values
                        # along the inner dim; the scale is per-32-block UE8M0.
                        t = _dequantize_mxfp4_packed_int8(
                            t,
                            hf_state_dict[scale_key],
                            block_size=32,
                            out_dtype=target_dtype,
                        )
                        summary.fp8_dequantized += 1
                expert_buf.setdefault(layer_idx, {}).setdefault(w_name, {})[expert_idx] = t
                continue

        xorl_name = _hf_to_xorl_name(hf_name)
        if xorl_name is None:
            summary.unmapped.append(hf_name)
            continue

        if xorl_name not in fillable:
            summary.missing_in_model.append(xorl_name)
            continue

        t = tensor
        if dequantize_fp8 and _is_fp8(t):
            scale_key = hf_name.replace(".weight", ".scale")
            if scale_key in hf_state_dict:
                t = _dequantize_fp8_block(t, hf_state_dict[scale_key], out_dtype=target_dtype)
                summary.fp8_dequantized += 1

        # APE hotfix undo for C4 layers (compressor.ape and indexer.compressor.ape).
        if (
            xorl_name.endswith(".compressor.ape")
            and m is not None
            and _is_compress_ratio_4(int(m.group(1)), compress_ratios)
        ):
            expected_hd = (
                indexer_compressor_head_dim
                if xorl_name.endswith(".indexer.compressor.ape")
                else attn_compressor_head_dim
            )
            t = _undo_ape_hotfix(t, expected_head_dim=expected_hd)
            summary.ape_unhotfixed += 1

        _copy_into(model_params, model_buffers, xorl_name, t)
        filled.add(xorl_name)
        summary.loaded += 1

    # Second pass: fuse the expert tensors and copy into model.
    # Pop entries instead of iterating so each layer's per-expert tensors
    # become unreachable as soon as we've consumed them — this caps the
    # second-pass peak at one layer's worth of experts (~12 GB for Flash)
    # plus the in-flight fused tensor, instead of carrying all 43 layers.
    import gc as _gc  # noqa: PLC0415

    for layer_idx in sorted(expert_buf.keys()):
        ws = expert_buf.pop(layer_idx)
        gate_up = _fuse_gate_up(ws.get("w1", {}), ws.get("w3", {}), config.n_routed_experts)
        down = _fuse_down(ws.get("w2", {}), config.n_routed_experts)
        ws.clear()
        if gate_up is not None:
            xorl_name = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
            if xorl_name in fillable:
                _copy_into(model_params, model_buffers, xorl_name, gate_up.to(target_dtype))
                filled.add(xorl_name)
                summary.experts_fused += 1
            del gate_up
        if down is not None:
            xorl_name = f"model.layers.{layer_idx}.mlp.experts.down_proj"
            if xorl_name in fillable:
                _copy_into(model_params, model_buffers, xorl_name, down.to(target_dtype))
                filled.add(xorl_name)
            del down
        # Force a sweep every few layers — Python's reference-cycle GC is
        # cheap relative to a 12 GB allocation drop.
        if layer_idx % 4 == 0:
            _gc.collect()

    # Strict-mode checks.
    if strict:
        unfilled = sorted(fillable - filled)
        if unfilled:
            raise RuntimeError(f"strict=True: {len(unfilled)} model params unfilled, e.g. {unfilled[:5]}")
        # Unmapped HF entries that aren't deliberate skips.
        if summary.unmapped:
            raise RuntimeError(
                f"strict=True: {len(summary.unmapped)} HF tensors had no xorl mapping, e.g. {summary.unmapped[:5]}"
            )

    return summary


def _copy_into(model_params, model_buffers, name: str, tensor: torch.Tensor):
    """Copy ``tensor`` into the model param/buffer registered as ``name``.

    Validates shape and casts dtype to match the destination so callers don't
    have to track dtype boundaries (fp32 ``attn_sink`` / ``hc_*`` vs bf16
    Linear weights vs int32 ``tid2eid``).
    """
    target = model_params.get(name)
    if target is None:
        target = model_buffers[name]
    if target.shape != tensor.shape:
        raise RuntimeError(f"shape mismatch for {name}: model expects {tuple(target.shape)}, got {tuple(tensor.shape)}")
    target.data.copy_(tensor.to(target.dtype))


def _fuse_gate_up(
    w1_per_expert: dict[int, torch.Tensor],
    w3_per_expert: dict[int, torch.Tensor],
    num_experts: int,
) -> Optional[torch.Tensor]:
    """Stack per-expert ``w1`` (gate) and ``w3`` (up) into a ``[E, H, 2I]`` tensor.

    HF stores each expert's ``w1`` / ``w3`` as ``[I, H]`` (Linear weight =
    ``[out, in]``); xorl's ``MoEExperts`` stores ``gate_up_proj`` as
    ``[E, H, 2I]`` (input-major, gate then up packed along the output axis).
    """
    if not w1_per_expert and not w3_per_expert:
        return None
    if len(w1_per_expert) != num_experts or len(w3_per_expert) != num_experts:
        raise RuntimeError(
            f"expert fusion got {len(w1_per_expert)} w1 and {len(w3_per_expert)} w3 "
            f"tensors but expected {num_experts} of each"
        )
    fused_per_expert = []
    for e in range(num_experts):
        # Each is [I, H] in HF (Linear weight). Transpose to [H, I] then pack.
        w1 = w1_per_expert[e].t().contiguous()  # [H, I] gate
        w3 = w3_per_expert[e].t().contiguous()  # [H, I] up
        fused_per_expert.append(torch.cat([w1, w3], dim=-1))  # [H, 2I]
    return torch.stack(fused_per_expert, dim=0)  # [E, H, 2I]


def _fuse_down(
    w2_per_expert: dict[int, torch.Tensor],
    num_experts: int,
) -> Optional[torch.Tensor]:
    """Stack per-expert ``w2`` (down) into a ``[E, I, H]`` tensor."""
    if not w2_per_expert:
        return None
    if len(w2_per_expert) != num_experts:
        raise RuntimeError(f"expert fusion got {len(w2_per_expert)} w2 tensors but expected {num_experts}")
    rows = []
    for e in range(num_experts):
        # HF [H, I] (Linear weight = [out, in] = [hidden, intermediate]).
        rows.append(w2_per_expert[e].t().contiguous())  # [I, H]
    return torch.stack(rows, dim=0)  # [E, I, H]


class DeepseekV4CheckpointHandler(CheckpointHandler):
    """Streaming HF -> xorl transform for the generic distributed loader.

    This mirrors :func:`stream_load_hf_directory_into_model`, but exposes the
    transforms through the standard ``CheckpointHandler`` interface so
    ``all_ranks_load_weights`` can materialize an FSDP2/EP-sharded DSv4 model
    directly from HF safetensors and then save a true multi-rank DCP.
    """

    def __init__(
        self,
        config,
        *,
        checkpoint_keys: set[str],
        ep_rank: int = 0,
        ep_size: int = 1,
        dequantize_fp8: bool = True,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.compress_ratios = config.compress_ratios
        self.attn_compressor_head_dim = config.head_dim
        self.indexer_compressor_head_dim = getattr(config, "index_head_dim", config.head_dim)
        self.checkpoint_keys = checkpoint_keys
        self.dequantize_fp8 = dequantize_fp8
        self.target_dtype = target_dtype
        self.summary = LoadSummary()

        self.ep_rank = ep_rank
        self.ep_size = ep_size
        if config.n_routed_experts % ep_size != 0:
            raise ValueError(f"n_routed_experts={config.n_routed_experts} must be divisible by ep_size={ep_size}")
        self.local_num_experts = config.n_routed_experts // ep_size
        self.expert_start = ep_rank * self.local_num_experts
        self.expert_end = self.expert_start + self.local_num_experts

        # ``pending[weight_name] = {"weight": w, "scale": s}`` for paired
        # FP8/MXFP4 tensors that may arrive in either order or in different
        # safetensors shards.
        self.pending: dict[str, dict[str, torch.Tensor]] = {}
        self.expert_buf: dict[int, dict[str, dict[int, torch.Tensor]]] = {}

    def _scale_name_for(self, weight_name: str) -> str | None:
        if weight_name.endswith(".weight"):
            scale_name = weight_name[: -len(".weight")] + ".scale"
            if scale_name in self.checkpoint_keys:
                return scale_name
        return None

    def _layer_index(self, key: str) -> int | None:
        weight_name = key[: -len(".scale")] + ".weight" if key.endswith(".scale") else key
        m = _LAYER_RE.match(weight_name)
        if m is None:
            return None
        return int(m.group(1))

    def _is_layer_out_of_range(self, key: str) -> bool:
        layer_idx = self._layer_index(key)
        return layer_idx is not None and layer_idx >= self.config.num_hidden_layers

    def _maybe_dequant(self, weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        if not self.dequantize_fp8 or scale is None:
            return weight
        if _is_fp8(weight):
            self.summary.fp8_dequantized += 1
            return _dequantize_fp8_block(weight, scale, out_dtype=self.target_dtype)
        if _is_mxfp4_packed(weight, has_scale=True):
            self.summary.fp8_dequantized += 1
            return _dequantize_mxfp4_packed_int8(weight, scale, block_size=32, out_dtype=self.target_dtype)
        return weight

    def _expert_local_index(self, expert_idx: int) -> int | None:
        if expert_idx < self.expert_start or expert_idx >= self.expert_end:
            return None
        return expert_idx - self.expert_start

    def _try_fuse_layer(self, layer_idx: int) -> list[tuple[str, torch.Tensor]]:
        ws = self.expert_buf.get(layer_idx)
        if ws is None:
            return []
        expected = self.local_num_experts
        if not (
            len(ws.get("w1", {})) == expected
            and len(ws.get("w2", {})) == expected
            and len(ws.get("w3", {})) == expected
        ):
            return []

        ws = self.expert_buf.pop(layer_idx)
        gate_up = _fuse_gate_up(ws.get("w1", {}), ws.get("w3", {}), expected)
        down = _fuse_down(ws.get("w2", {}), expected)
        ws.clear()

        results: list[tuple[str, torch.Tensor]] = []
        if gate_up is not None:
            results.append((f"model.layers.{layer_idx}.mlp.experts.gate_up_proj", gate_up.to(self.target_dtype)))
            self.summary.experts_fused += 1
        if down is not None:
            results.append((f"model.layers.{layer_idx}.mlp.experts.down_proj", down.to(self.target_dtype)))
        return results

    def _process(self, weight_name: str) -> list[tuple[str, torch.Tensor]]:
        slot = self.pending.get(weight_name)
        if slot is None:
            return []
        needed_scale = self._scale_name_for(weight_name) is not None
        if needed_scale and "scale" not in slot:
            return []
        if "weight" not in slot:
            return []

        weight = slot["weight"]
        scale = slot.get("scale")
        del self.pending[weight_name]

        m = _LAYER_RE.match(weight_name)
        if m is not None:
            layer_idx = int(m.group(1))
            em = _EXPERT_RE.match(m.group(2))
            if em is not None:
                local_expert_idx = self._expert_local_index(int(em.group(1)))
                if local_expert_idx is None:
                    return []
                w_name = em.group(2)  # w1 | w2 | w3
                t = self._maybe_dequant(weight, scale)
                self.expert_buf.setdefault(layer_idx, {}).setdefault(w_name, {})[local_expert_idx] = t
                return self._try_fuse_layer(layer_idx)

        xorl_name = _hf_to_xorl_name(weight_name)
        if xorl_name is None:
            if weight_name.startswith("mtp."):
                if not weight_name.endswith(".scale"):
                    self.summary.skipped_mtp += 1
            else:
                self.summary.unmapped.append(weight_name)
            return []

        t = self._maybe_dequant(weight, scale)
        if (
            xorl_name.endswith(".compressor.ape")
            and m is not None
            and _is_compress_ratio_4(int(m.group(1)), self.compress_ratios)
        ):
            expected_hd = (
                self.indexer_compressor_head_dim
                if xorl_name.endswith(".indexer.compressor.ape")
                else self.attn_compressor_head_dim
            )
            t = _undo_ape_hotfix(t, expected_head_dim=expected_hd)
            self.summary.ape_unhotfixed += 1

        self.summary.loaded += 1
        return [(xorl_name, t)]

    def on_load_weight(self, key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        if self._is_layer_out_of_range(key):
            return []
        if key.startswith("mtp."):
            if not key.endswith(".scale"):
                self.summary.skipped_mtp += 1
            return []
        if key.endswith(".scale"):
            weight_name = key[: -len(".scale")] + ".weight"
            self.pending.setdefault(weight_name, {})["scale"] = tensor
            return self._process(weight_name)

        self.pending.setdefault(key, {})["weight"] = tensor
        return self._process(key)

    def on_skip_weight(self, key: str) -> list[tuple[str, torch.Tensor]]:
        # The handler fuses once all local experts are present; skipped
        # out-of-range EP experts do not need placeholder entries.
        if key.startswith("mtp.") and not key.endswith(".scale"):
            self.summary.skipped_mtp += 1
        return []

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        has_layer_filter = any(self._is_layer_out_of_range(key) for key in self.checkpoint_keys)
        has_mtp = any(key.startswith("mtp.") for key in self.checkpoint_keys)
        if self.ep_size <= 1 and not has_layer_filter and not has_mtp:
            return None

        def _should_skip(key: str) -> bool:
            if key.startswith("mtp."):
                return True
            if self._is_layer_out_of_range(key):
                return True
            if self.ep_size <= 1:
                return False
            weight_name = key[: -len(".scale")] + ".weight" if key.endswith(".scale") else key
            m = _LAYER_RE.match(weight_name)
            if m is None:
                return False
            em = _EXPERT_RE.match(m.group(2))
            if em is None:
                return False
            expert_idx = int(em.group(1))
            return expert_idx < self.expert_start or expert_idx >= self.expert_end

        return _should_skip

    def on_load_complete(self) -> list[tuple[str, torch.Tensor]]:
        if self.pending:
            warnings.warn(f"Incomplete DSv4 pending HF weight/scale pairs after loading: {sorted(self.pending)[:8]}")
        pending_experts = {
            layer_idx: {proj: sorted(experts) for proj, experts in by_proj.items()}
            for layer_idx, by_proj in self.expert_buf.items()
        }
        if pending_experts:
            warnings.warn(f"Incomplete DSv4 expert weights after loading: {pending_experts}")
        return []


def stream_load_hf_directory_into_model(
    model: torch.nn.Module,
    hf_dir,
    *,
    strict: bool = False,
    dequantize_fp8: bool = True,
    target_dtype: torch.dtype = torch.bfloat16,
    progress: bool = True,
) -> LoadSummary:
    """Memory-efficient HF safetensors directory -> xorl model loader.

    Walks shards in order via ``safetensors.torch.load_file``, dequantizes
    paired ``.weight + .scale`` tensors as they arrive, and frees each
    shard before moving to the next. Per-layer expert buffers are fused
    and dropped as soon as a layer is complete (256 experts × {w1, w2, w3}).

    Peak memory ≈ ``model_bf16 + 1 active shard + 1 layer's expert buf``,
    typically well under 350 GiB for Flash — vs ~1 TiB+ for the
    full-state-dict path.

    Differs from :func:`load_hf_state_dict_into_model` only in input shape:
    this one takes a directory containing ``model.safetensors.index.json``
    + N safetensors shards. Identical transforms (FP8/FP4 dequant, APE
    hotfix, expert fusion, name remap).
    """
    import gc  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from safetensors.torch import load_file  # noqa: PLC0415

    hf_dir = Path(hf_dir)
    config = model.config
    compress_ratios = config.compress_ratios
    attn_compressor_head_dim = config.head_dim
    indexer_compressor_head_dim = getattr(config, "index_head_dim", config.head_dim)
    summary = LoadSummary()

    with (hf_dir / "model.safetensors.index.json").open() as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    # ``scale_for[weight_name] = scale_name`` only if both keys are in the
    # index. Non-quantized tensors map to ``None`` (consumed without scale).
    scale_for: dict[str, str | None] = {}
    all_keys = set(weight_map)
    for k in all_keys:
        if k.endswith(".scale"):
            continue
        if k.endswith(".weight"):
            s = k[: -len(".weight")] + ".scale"
            scale_for[k] = s if s in all_keys else None
        else:
            scale_for[k] = None  # buffer / non-weight tensor (e.g. attn_sink)

    # Group tensor names by their containing shard.
    files_to_tensors: dict[str, list[str]] = {}
    for name, fname in weight_map.items():
        files_to_tensors.setdefault(fname, []).append(name)

    model_params, model_buffers, fillable = _fillable_state(model)
    filled: set[str] = set()

    # Tensors waiting for their pair across a shard boundary.
    # ``pending[weight_name] = {"weight": w, "scale": s}`` (one or both keys).
    pending: dict[str, dict[str, torch.Tensor]] = {}

    # Per-layer expert tensors -> fused tensor.
    expert_buf: dict[int, dict[str, dict[int, torch.Tensor]]] = {}

    def _maybe_dequant(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        """Apply FP8 / MXFP4 dequant when scale is present and dequantize_fp8 is on."""
        if not dequantize_fp8 or scale is None:
            return weight
        if _is_fp8(weight):
            summary.fp8_dequantized += 1
            return _dequantize_fp8_block(weight, scale, out_dtype=target_dtype)
        if _is_mxfp4_packed(weight, has_scale=True):
            summary.fp8_dequantized += 1
            return _dequantize_mxfp4_packed_int8(weight, scale, block_size=32, out_dtype=target_dtype)
        return weight

    def _try_fuse_layer(layer_idx: int) -> None:
        """If layer's experts are all gathered, fuse + assign + free."""
        ws = expert_buf.get(layer_idx)
        if ws is None:
            return
        n = config.n_routed_experts
        if not (len(ws.get("w1", {})) == n and len(ws.get("w2", {})) == n and len(ws.get("w3", {})) == n):
            return
        ws = expert_buf.pop(layer_idx)
        gate_up = _fuse_gate_up(ws.get("w1", {}), ws.get("w3", {}), n)
        down = _fuse_down(ws.get("w2", {}), n)
        ws.clear()
        if gate_up is not None:
            xorl_name = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
            if xorl_name in fillable:
                _copy_into(model_params, model_buffers, xorl_name, gate_up.to(target_dtype))
                filled.add(xorl_name)
                summary.experts_fused += 1
            del gate_up
        if down is not None:
            xorl_name = f"model.layers.{layer_idx}.mlp.experts.down_proj"
            if xorl_name in fillable:
                _copy_into(model_params, model_buffers, xorl_name, down.to(target_dtype))
                filled.add(xorl_name)
            del down

    def _process(weight_name: str) -> None:
        """Try to consume a fully-arrived tensor (weight + matching scale, if any)."""
        slot = pending.get(weight_name)
        if slot is None:
            return
        needed_scale = scale_for.get(weight_name) is not None
        if needed_scale and "scale" not in slot:
            return
        if "weight" not in slot:
            return
        weight = slot["weight"]
        scale = slot.get("scale")
        del pending[weight_name]

        m = _LAYER_RE.match(weight_name)
        if m is not None:
            layer_idx = int(m.group(1))
            em = _EXPERT_RE.match(m.group(2))
            if em is not None:
                expert_idx = int(em.group(1))
                w_name = em.group(2)  # w1 | w2 | w3
                t = _maybe_dequant(weight, scale)
                expert_buf.setdefault(layer_idx, {}).setdefault(w_name, {})[expert_idx] = t
                # Fuse opportunistically — also done after each shard.
                _try_fuse_layer(layer_idx)
                return

        xorl_name = _hf_to_xorl_name(weight_name)
        if xorl_name is None:
            summary.unmapped.append(weight_name)
            return
        if xorl_name not in fillable:
            summary.missing_in_model.append(xorl_name)
            return

        t = _maybe_dequant(weight, scale)

        if (
            xorl_name.endswith(".compressor.ape")
            and m is not None
            and _is_compress_ratio_4(int(m.group(1)), compress_ratios)
        ):
            expected_hd = (
                indexer_compressor_head_dim
                if xorl_name.endswith(".indexer.compressor.ape")
                else attn_compressor_head_dim
            )
            t = _undo_ape_hotfix(t, expected_head_dim=expected_hd)
            summary.ape_unhotfixed += 1

        _copy_into(model_params, model_buffers, xorl_name, t)
        filled.add(xorl_name)
        summary.loaded += 1

    sorted_files = sorted(files_to_tensors.keys())
    for shard_idx, fname in enumerate(sorted_files):
        if progress:
            print(f"[{shard_idx + 1}/{len(sorted_files)}] streaming {fname}", flush=True)
        shard = load_file(str(hf_dir / fname))
        for name, tensor in shard.items():
            if name.startswith("mtp."):
                # Don't double-count ``mtp.*.scale`` sidecars against the
                # weight count — ``skipped_mtp`` is meant to track distinct
                # logical tensors, not raw safetensors entries.
                if not name.endswith(".scale"):
                    summary.skipped_mtp += 1
                continue
            if name.endswith(".scale"):
                weight_name = name[: -len(".scale")] + ".weight"
                pending.setdefault(weight_name, {})["scale"] = tensor
                _process(weight_name)
            else:
                pending.setdefault(name, {})["weight"] = tensor
                _process(name)
        del shard
        # Sweep for any newly-complete layer (e.g. last expert arrived in this shard).
        for li in list(expert_buf.keys()):
            _try_fuse_layer(li)
        gc.collect()

    if strict:
        unfilled = sorted(fillable - filled)
        if unfilled:
            raise RuntimeError(f"strict=True: {len(unfilled)} model params unfilled, e.g. {unfilled[:5]}")
        if summary.unmapped:
            raise RuntimeError(
                f"strict=True: {len(summary.unmapped)} HF tensors had no xorl mapping, e.g. {summary.unmapped[:5]}"
            )

    return summary


__all__ = [
    "DeepseekV4CheckpointHandler",
    "LoadSummary",
    "load_hf_state_dict_into_model",
    "stream_load_hf_directory_into_model",
]
# Internal helpers (``_dequantize_fp8_block``, ``_dequantize_mxfp4_packed_int8``,
# ``_undo_ape_hotfix``, ``_hf_to_xorl_name``) are exercised by tests via
# explicit module-path imports rather than re-exported here. Their underscore
# prefix denotes private contract — leave them out of ``__all__``.
