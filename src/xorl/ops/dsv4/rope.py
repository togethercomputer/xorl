"""YaRN-aware RoPE for DeepSeek-V4.

Adapted from miles ``miles_plugins/models/deepseek_v4/ops/rope.py``: drops the
Megatron ``TransformerConfig`` import and reads YaRN parameters
(``original_max_position_embeddings``, ``factor``, ``beta_fast``, ``beta_slow``)
from xorl's ``DeepseekV4Config.rope_parameters`` dict instead of top-level
config fields.
"""

import math
import os
from functools import lru_cache

import torch


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precompute the complex rotary frequencies for RoPE, with optional YaRN smoothing.

    When ``original_seq_len > 0``, applies YaRN factor rescaling interpolated by a
    linear ramp between ``beta_fast`` and ``beta_slow``. Otherwise the base
    frequencies are used verbatim.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE in-place to the last dim of ``x``.

    ``x`` has shape ``[..., dim]`` where ``dim`` is even; the last-dim pairs are
    treated as complex numbers multiplied by ``freqs_cis``. When ``inverse=True``
    the conjugate rotation is applied (used for the indexer's inverse rope).
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def wrapped_precompute_freqs_cis(config, rope_head_dim: int, base: float, yarn_disabled: bool = False):
    """Cached YaRN-aware freqs_cis for a single layer.

    Reads ``factor`` / ``beta_fast`` / ``beta_slow`` / ``original_max_position_embeddings``
    from ``config.rope_parameters`` (xorl convention). ``yarn_disabled=True``
    forces ``original_seq_len=0``, which makes ``precompute_freqs_cis`` skip the
    YaRN correction-range interpolation — used by the 0415 ckpt for pure-window
    (compress_ratio==0) layers.
    """
    rope_params = config.rope_parameters or {}
    original_max_pos = int(rope_params.get("original_max_position_embeddings", 65536))
    factor = float(rope_params.get("factor", 1.0))
    beta_fast = float(rope_params.get("beta_fast", 32.0))
    beta_slow = float(rope_params.get("beta_slow", 1.0))

    # Full-weight V4 configs advertise their extended YaRN context in
    # max_position_embeddings. Keep the env override for tests/profiling runs
    # that intentionally want a smaller cache than the model maximum.
    max_seq_len = int(
        os.environ.get("XORL_DSV4_ROPE_MAX_SEQ_LEN", getattr(config, "max_position_embeddings", original_max_pos))
    )
    original_seq_len = 0 if yarn_disabled else original_max_pos

    return precompute_freqs_cis(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
