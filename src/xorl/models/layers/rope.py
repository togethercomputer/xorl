"""Local copy of Rotary Position Embedding utilities.

Copied from ``transformers.modeling_rope_utils`` to avoid breakage across
transformers major versions.  In particular, the ``"default"`` rope type was
removed in transformers >= 5.0 -- this local copy keeps it available.
"""

import logging
import math
import warnings
from functools import wraps
from typing import Optional

import torch
import torch.nn as nn

try:
    from flash_attn.layers.rotary import apply_rotary_emb as _flash_apply_rotary_emb
except ImportError:
    _flash_apply_rotary_emb = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dynamic_rope_update decorator
# ---------------------------------------------------------------------------

def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).
    """

    def longrope_frequency_update(self, position_ids, device, layer_type=None):
        """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
        seq_len = torch.max(position_ids) + 1

        if layer_type is None:
            rope_type = self.rope_type
            original_inv_freq = self.original_inv_freq
            prefix = ""
            original_max_position_embeddings = self.config.rope_parameters["original_max_position_embeddings"]
        else:
            rope_type = self.rope_type[layer_type]
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"
            original_max_position_embeddings = self.config.rope_parameters[layer_type][
                "original_max_position_embeddings"
            ]

        if seq_len > original_max_position_embeddings:
            if not hasattr(self, f"{layer_type}_long_inv_freq"):
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                long_inv_freq, _ = rope_init_fn(
                    self.config,
                    device,
                    seq_len=original_max_position_embeddings + 1,
                    layer_type=layer_type,
                )
            self.register_buffer(f"{prefix}inv_freq", long_inv_freq, persistent=False)
            setattr(self, f"{prefix}long_inv_freq", long_inv_freq)
        else:
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)

    def dynamic_frequency_update(self, position_ids, device, layer_type=None):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if layer_type is None:
            rope_type = self.rope_type
            max_seq_len_cached = self.max_seq_len_cached
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            rope_type = self.rope_type[layer_type]
            max_seq_len_cached = getattr(self, f"{layer_type}_max_seq_len_cached", self.max_seq_len_cached)
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        if seq_len > max_seq_len_cached:  # growth
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(
                self.config,
                device,
                seq_len=seq_len,
                layer_type=layer_type,
            )
            # TODO joao: may break with compilation
            self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_max_seq_len_cached", seq_len)

        if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            setattr(self, f"{layer_type}_max_seq_len_cached", self.original_max_seq_len)

    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
        kwargs = {"layer_type": layer_type} if layer_type is not None else {}
        if "dynamic" in rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device, **kwargs)
        elif rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device, **kwargs)
        return rope_forward(self, x, position_ids, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# RoPE parameter computation functions
# ---------------------------------------------------------------------------

def _compute_default_rope_parameters(
    config=None,
    device=None,
    seq_len=None,
    **kwargs,
):
    """Standard RoPE inverse-frequency computation (no scaling).

    This is the ``"default"`` rope type that was present in transformers < 5.0.
    It reads ``rope_theta`` from either ``config.rope_theta`` or
    ``config.rope_scaling["rope_theta"]`` / ``config.rope_parameters`` for
    backwards compatibility.
    """
    # Try the new standardized rope_parameters first, fall back to legacy attrs
    rope_parameters_dict = getattr(config, "rope_parameters", None) or {}
    if rope_parameters_dict:
        base = rope_parameters_dict.get("rope_theta", None)
    else:
        base = None

    if base is None:
        if hasattr(config, "rope_theta") and config.rope_theta is not None:
            base = config.rope_theta
        elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            base = config.rope_scaling.get("rope_theta", 10000.0)
        else:
            base = 10000.0

    partial_rotary_factor = rope_parameters_dict.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 1.0),
    )
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    attention_factor = 1.0
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config=None,
    device=None,
    seq_len=None,
    layer_type=None,
):
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    """
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict["factor"]

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config=None,
    device=None,
    seq_len=None,
    layer_type=None,
):
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    """
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = rope_parameters_dict["factor"]
    attention_factor = 1.0

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = config.max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(config.max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, config.max_position_embeddings)

    base = base * ((factor * seq_len / config.max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config=None,
    device=None,
    seq_len=None,
    layer_type=None,
):
    """
    Computes the inverse frequencies with YaRN scaling. Please refer to the
    original paper: https://arxiv.org/abs/2309.00071
    """
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: DeekSeek-V3 (and potentially other models) have `original_max_position_embeddings` field
    # containing the pretrained value. They use the ratio between `max_position_embeddings` and this value
    # to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_parameters_dict.get("beta_fast") or 32
    beta_slow = rope_parameters_dict.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config=None,
    device=None,
    seq_len=None,
    layer_type=None,
):
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    original implementation: https://github.com/microsoft/LongRoPE
    """
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    long_factor = rope_parameters_dict["long_factor"]
    short_factor = rope_parameters_dict["short_factor"]
    factor = rope_parameters_dict.get("factor")
    attention_factor = rope_parameters_dict.get("attention_factor")
    original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if factor is None:
        factor = config.max_position_embeddings / original_max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config=None,
    device=None,
    seq_len=None,
    layer_type=None,
):
    """
    Computes the inverse frequencies for llama 3.1.
    """
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    factor = rope_parameters_dict["factor"]  # `8` in the original implementation
    low_freq_factor = rope_parameters_dict["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = rope_parameters_dict["high_freq_factor"]  # `4` in the original implementation
    old_context_len = rope_parameters_dict["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


# ---------------------------------------------------------------------------
# ROPE_INIT_FUNCTIONS registry
# ---------------------------------------------------------------------------

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}


# ---------------------------------------------------------------------------
# RotaryEmbedding module
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding layer.

    Computes cos/sin position embeddings from inverse frequencies. Supports all
    rope types registered in ``ROPE_INIT_FUNCTIONS`` (default, linear, dynamic,
    yarn, longrope, llama3). Dynamic frequency updates are handled by the
    ``@dynamic_rope_update`` decorator.
    """

    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# RoPE application helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _naive_apply_rotary_pos_emb(q, k, cos, sin):
    """Naive RoPE application (pure PyTorch, no fused kernel).

    All tensors use [B, S, H, D] layout. cos/sin are [B, S, D].
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Uses flash_attn's fused CUDA kernel when available for better performance
    and numerical precision in reduced-precision dtypes. Falls back to a naive
    PyTorch implementation otherwise.

    All tensors use [B, S, H, D] layout (flash attention native format).

    Args:
        q: The query tensor of shape [batch, seq_len, heads, head_dim].
        k: The key tensor of shape [batch, seq_len, heads, head_dim].
        cos: The cosine part from RotaryEmbedding, shape [batch, seq_len, head_dim].
        sin: The sine part from RotaryEmbedding, shape [batch, seq_len, head_dim].
    """
    if _flash_apply_rotary_emb is not None and q.is_cuda:
        # flash_attn expects x: [B, S, H, D], cos/sin: [S, D//2]
        # Our cos/sin are [B, S, D] with doubled freqs — take first batch, first half
        half_dim = cos.shape[-1] // 2
        cos_half = cos[0, :, :half_dim]
        sin_half = sin[0, :, :half_dim]
        q_embed = _flash_apply_rotary_emb(q, cos_half, sin_half)
        k_embed = _flash_apply_rotary_emb(k, cos_half, sin_half)
        return q_embed, k_embed

    return _naive_apply_rotary_pos_emb(q, k, cos, sin)


# ---------------------------------------------------------------------------
# Deprecated helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def rope_config_validation(config, ignore_keys=None):
    """
    Deprecated function. Calls config.standardize_rope_params() and
    config.validate_rope() directly.
    """
    warnings.warn(
        "`rope_config_validation` is deprecated. "
        "Call config.standardize_rope_params() and config.validate_rope() instead.",
        FutureWarning,
    )
    config.standardize_rope_params()
    config.validate_rope(ignore_keys=ignore_keys)


__all__ = [
    "ROPE_INIT_FUNCTIONS",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "dynamic_rope_update",
    "rope_config_validation",
    "rotate_half",
]
