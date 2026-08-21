"""Local copy of Rotary Position Embedding utilities.

Copied from ``transformers.modeling_rope_utils`` to avoid breakage across
transformers major versions.  In particular, the ``"default"`` rope type was
removed in transformers >= 5.0 -- this local copy keeps it available.
"""

import logging
import math
import os
import warnings
from functools import wraps

import torch
import torch.nn as nn

from xorl.models.exact_contract import glm52_exact_forward_enabled


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
            self._set_inv_freq_fp32(long_inv_freq)
        else:
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            self._set_inv_freq_fp32(original_inv_freq)

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
            self._set_inv_freq_fp32(inv_freq)

        if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            setattr(self, f"{layer_type}_max_seq_len_cached", self.original_max_seq_len)
            self._set_inv_freq_fp32(original_inv_freq)

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

    if base is None and hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        base = config.rope_scaling.get("rope_theta", None)
    if base is None and hasattr(config, "rope_theta") and config.rope_theta is not None:
        base = config.rope_theta
    if base is None:
        base = 10000.0

    partial_rotary_factor = rope_parameters_dict.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 1.0),
    )
    if partial_rotary_factor is None:
        partial_rotary_factor = 1.0
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
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
        # The loader's model-wide `.to(torch_dtype)` downcasts the inv_freq buffer, and a
        # bf16 frequency table corrupts cos/sin from position 1. Keep a cast-immune fp32 CPU-computed
        # table (a plain attribute, invisible to `nn.Module._apply`) and read that in forward(). CPU
        # provenance is what serving builds its cache from, so the zero-K3 contract keeps holding.
        # Unconditional and registry-wide: the buffer is cast in every bf16-built lane, not only the
        # rope_native ones, and every rope type reads it.
        self._inv_freq_fp32: torch.Tensor | None = None
        self._set_inv_freq_fp32(self._cpu_fp32_inv_freq())
        self._sglang_default_cache = None
        self._use_sglang_default_cache = bool(getattr(config, "_rope_native", False) and self.rope_type == "default")
        self._fp32_single_round = bool(getattr(config, "_rope_fp32_single_round", False) or glm52_exact_forward_enabled(config))

    def _cpu_fp32_inv_freq(self) -> torch.Tensor:
        """Frequency table computed on CPU in fp32 — the provenance serving's cos/sin cache is built with."""
        with torch.device("cpu"):
            inv_freq, _ = self.rope_init_fn(self.config, "cpu")
        return inv_freq.float()

    def _set_inv_freq_fp32(self, inv_freq: torch.Tensor) -> None:
        self._inv_freq_fp32 = inv_freq.float()

    def _resolve_inv_freq(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq_fp32 is None:
            return self.inv_freq
        if self._inv_freq_fp32.device != device:
            self._inv_freq_fp32 = self._inv_freq_fp32.to(device)
        return self._inv_freq_fp32

    def _default_rope_base_and_dim(self) -> tuple[float, int]:
        rope_parameters_dict = getattr(self.config, "rope_parameters", None) or {}
        base = rope_parameters_dict.get("rope_theta") if rope_parameters_dict else None
        if base is None and getattr(self.config, "rope_scaling", None) is not None:
            base = self.config.rope_scaling.get("rope_theta")
        if base is None:
            base = getattr(self.config, "rope_theta", None)
        if base is None:
            base = 10000.0

        partial_rotary_factor = rope_parameters_dict.get(
            "partial_rotary_factor",
            getattr(self.config, "partial_rotary_factor", 1.0),
        )
        if partial_rotary_factor is None:
            partial_rotary_factor = 1.0
        head_dim = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )
        return float(base), int(head_dim * partial_rotary_factor)

    def _build_sglang_default_cache(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Table provenance is architecture-specific, matching each family's
        # certified serving program. GLM-5.2 computes inverse frequencies on
        # CPU in fp32, transfers them to the execution device, then builds the
        # position outer product and cos/sin on that device. The Qwen3.5-family
        # program evaluates the whole table on CPU and moves the finished fp32
        # values (host/device transfers are bit-exact). The split is
        # load-bearing: CUDA cos/sin differ from CPU libm by up to one fp32
        # ulp, and a table built on the wrong device seeds a position-onset
        # trainer/sampler logprob mismatch.
        base, dim = self._default_rope_base_and_dim()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if glm52_exact_forward_enabled(self.config):
            inv_freq = inv_freq.to(device=device)
            positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            return torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device=device)

    def _ensure_sglang_default_cache(self, needed_max_pos: int, device: torch.device) -> None:
        if (
            self._sglang_default_cache is None
            or self._sglang_default_cache.device != device
            or needed_max_pos >= self._sglang_default_cache.shape[0]
        ):
            # SGLang materializes the configured default-cache capacity at
            # construction.  Match both its execution device and table shape;
            # this also keeps the admitted run out of the cache-growth path.
            cache_len = max(self.max_seq_len_cached, needed_max_pos + 1)
            self._sglang_default_cache = self._build_sglang_default_cache(cache_len, device)

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if self._use_sglang_default_cache:
            needed_max_pos = int(position_ids.max().item())
            self._ensure_sglang_default_cache(needed_max_pos, x.device)
            flat_positions = position_ids.reshape(-1).to(device=x.device, dtype=torch.long)
            cos_sin = self._sglang_default_cache.index_select(0, flat_positions)
            cos_half, sin_half = cos_sin.chunk(2, dim=-1)
            cos = torch.cat((cos_half, cos_half), dim=-1).view(*position_ids.shape, -1)
            sin = torch.cat((sin_half, sin_half), dim=-1).view(*position_ids.shape, -1)
            out_dtype = torch.float32 if self._fp32_single_round else x.dtype
            return cos.to(device=x.device, dtype=out_dtype), sin.to(device=x.device, dtype=out_dtype)

        inv_freq = self._resolve_inv_freq(x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # Class B feeds cos/sin straight into an fp32 kernel cache; a bf16 round here
        # would put the result back in Class A.
        out_dtype = torch.float32 if self._fp32_single_round else x.dtype
        return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)


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
    Handles partial rotary automatically when cos/sin dim < head_dim.
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]
    if q.shape[-1] > rotary_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
        k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


_rope_native = False


def set_rope_native(enabled: bool):
    """Set whether to use naive RoPE instead of flash_attn fused kernel."""
    global _rope_native
    _rope_native = enabled


# ---------------------------------------------------------------------------
# Class-B RoPE numerics: a compiled fp32 chain aligned with SGLang's stock fused
# CUDA kernel, with an explicit adjoint
# ---------------------------------------------------------------------------
#
# Two RoPE numerics classes exist across the trainer/sampler pair:
#   Class A -- per-op bf16 rounding (8 rounding points). ``_naive_apply_rotary_pos_emb``,
#              SGLang eager ``apply_rotary_emb`` and the ``bi_fused_native`` triton
#              kernel all agree bitwise here.
#   Class B -- one fp32 chain with a single final round. SGLang's ``torch.compile``
#              RoPE path and its stock fused CUDA kernel agree bitwise here.
#
# Class B needs fp32 cos/sin: the kernel reads an fp32 ``[max_pos, rotary_dim]`` cache
# (cos in the first half of each row, sin in the second). Rounding cos/sin to bf16
# first would land back in Class A, so the Class-B lane keeps the table in fp32 all
# the way to the kernel.

_rope_fp32_single_round = (
    os.environ.get("XORL_ROPE_FP32_SINGLE_ROUND", os.environ.get("XORL_ROPE_CLASS_B", "")) == "1"
)


def set_rope_fp32_single_round(enabled: bool) -> None:
    """Select compiled fp32-chain numerics aligned with SGLang's stock fused CUDA kernel."""
    global _rope_fp32_single_round
    _rope_fp32_single_round = enabled


def rope_fp32_single_round_enabled() -> bool:
    return _rope_fp32_single_round


def stock_fused_apply_rotary_pos_emb(q, k, cos, sin, *, interleaved: bool = False, doubled: bool = True):
    """Class-B RoPE application backed by the compiled expression in ``xorl.ops.exact.rope_fp32_single_round``."""
    from xorl.ops.exact.rope_fp32_single_round import single_round_apply_rotary_pos_emb  # noqa: PLC0415

    return single_round_apply_rotary_pos_emb(q, k, cos, sin, interleaved=interleaved, doubled=doubled)


def apply_rotary_pos_emb(q, k, cos, sin, *, force_native: bool = False):
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
    if _rope_fp32_single_round and q.is_cuda:
        return stock_fused_apply_rotary_pos_emb(q, k, cos, sin)

    if _flash_apply_rotary_emb is not None and q.is_cuda and not (_rope_native or force_native):
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
    if ignore_keys is None:
        config.validate_rope()
        return

    try:
        config.validate_rope(ignore_keys=ignore_keys)
    except TypeError as exc:
        if "unexpected keyword argument 'ignore_keys'" not in str(exc):
            raise
        # transformers>=5.5 removed the ignore_keys kwarg from validate_rope().
        config.validate_rope()


__all__ = [
    "ROPE_INIT_FUNCTIONS",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "dynamic_rope_update",
    "rope_fp32_single_round_enabled",
    "rope_config_validation",
    "rotate_half",
    "set_rope_fp32_single_round",
    "stock_fused_apply_rotary_pos_emb",
]
