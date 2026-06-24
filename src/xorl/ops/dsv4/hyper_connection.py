"""HyperConnection mixer for DeepSeek-V4.

Adapted from miles ``miles_plugins/models/deepseek_v4/ops/hyper_connection.py``:

* ``MegatronModule`` → ``nn.Module``.
* Config fields renamed: ``dsv4_hc_mult`` → ``hc_mult``, ``dsv4_hc_sinkhorn_iters``
  → ``hc_sinkhorn_iters``, ``dsv4_hc_eps`` → ``hc_eps``, ``layernorm_epsilon`` →
  ``rms_norm_eps``.

Routing math (``hc_pre_raw`` / ``hc_post_raw`` / ``hc_head_raw``) is kept
verbatim and runs under ``torch.no_grad`` per the V4 contract — gradients flow
only through the residual streams, not through the mixer.
"""

import os

import einops
import torch
import torch.nn.functional as F
from torch import Tensor


# ``hc_split_sinkhorn`` is imported lazily inside ``hc_pre_raw`` because it
# requires tilelang. A pure-torch fallback (``_hc_split_sinkhorn_torch``)
# is selected when tilelang is unavailable or when
# ``XORL_DSV4_SINKHORN_IMPL=torch`` is set; the math matches the kernel
# bit-for-bit modulo floating-point reduction order.

_HYPER_CONNECTION_MIXER_NO_GRAD = True
_DEFAULT_HC_CHUNK_TOKENS = 1024


def _hc_chunk_tokens() -> int:
    value = os.environ.get("XORL_DSV4_HC_CHUNK_TOKENS")
    if value is None:
        return _DEFAULT_HC_CHUNK_TOKENS
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"XORL_DSV4_HC_CHUNK_TOKENS must be an int, got {value!r}") from exc
    return max(parsed, 0)


def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Pure-torch reference for ``hc_split_sinkhorn``.

    ``mixes``: ``[B, S, (2 + hc_mult) * hc_mult]`` fp32.
    ``hc_scale``: ``[3]`` fp32 — scales for (pre, post, comb) blocks.
    ``hc_base``: ``[(2 + hc_mult) * hc_mult]`` fp32 — additive bias.

    Returns ``(pre [B, S, hc_mult], post [B, S, hc_mult], comb [B, S, hc_mult, hc_mult])``.
    """
    hc = hc_mult
    pre_in = mixes[..., :hc]
    post_in = mixes[..., hc : 2 * hc]
    comb_in = mixes[..., 2 * hc : 2 * hc + hc * hc].reshape(*mixes.shape[:-1], hc, hc)

    base_pre = hc_base[:hc]
    base_post = hc_base[hc : 2 * hc]
    base_comb = hc_base[2 * hc : 2 * hc + hc * hc].reshape(hc, hc)

    pre = torch.sigmoid(pre_in * hc_scale[0] + base_pre) + eps
    post = 2.0 * torch.sigmoid(post_in * hc_scale[1] + base_post)

    comb = comb_in * hc_scale[2] + base_comb

    # First sinkhorn iter: softmax over rows, then column-normalize.
    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / (comb.sum(dim=-1, keepdim=True))
    comb = comb + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # Remaining iterations: alternating row + column normalize.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def _resolve_sinkhorn_impl(device: torch.device | None = None):
    """Pick tilelang or torch fallback based on device + env + availability.

    The tilelang ``hc_split_sinkhorn`` kernel is CUDA-only; for CPU
    inputs we always use the torch fallback regardless of env. Env
    overrides for explicit testing:

    - ``XORL_DSV4_SINKHORN_IMPL=tilelang`` — force tilelang (errors on CPU)
    - ``XORL_DSV4_SINKHORN_IMPL=torch`` — force torch fallback
    - ``XORL_DSV4_SINKHORN_IMPL=auto`` (default) — tilelang on CUDA when
      available, torch otherwise.
    """
    impl = os.environ.get("XORL_DSV4_SINKHORN_IMPL", "auto")
    if impl not in {"auto", "tilelang", "torch"}:
        raise ValueError(f"XORL_DSV4_SINKHORN_IMPL must be auto/tilelang/torch, got {impl!r}")
    if impl == "torch":
        return _hc_split_sinkhorn_torch
    if device is not None and device.type != "cuda":
        # tilelang kernel is CUDA-only; CPU has no choice.
        return _hc_split_sinkhorn_torch
    try:
        from .kernel.sinkhorn import hc_split_sinkhorn  # noqa: PLC0415

        return hc_split_sinkhorn
    except ImportError:
        if impl == "tilelang":
            raise
        return _hc_split_sinkhorn_torch


class DeepSeekV4HyperConnectionUtil:
    """Per-layer pre/post + per-block expand/head helpers."""

    def __init__(self, config):
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps

    def _compute_mixes(
        self,
        x_flat: Tensor,
        hc_fn_fp32: Tensor,
    ) -> Tensor:
        """Compute HC mixer logits without materializing full fp32 activations."""
        mix_size = hc_fn_fp32.shape[0]
        flat_2d = x_flat.reshape(-1, x_flat.shape[-1])
        mixes = torch.empty((*x_flat.shape[:-1], mix_size), device=x_flat.device, dtype=torch.float32)
        mixes_2d = mixes.reshape(-1, mix_size)

        chunk_tokens = _hc_chunk_tokens()
        if chunk_tokens <= 0:
            chunk_tokens = flat_2d.shape[0]

        for start in range(0, flat_2d.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, flat_2d.shape[0])
            x_chunk = flat_2d[start:end].float()
            x_sq_mean = x_chunk.square().mean(-1, keepdim=True)
            rsqrt = torch.rsqrt(x_sq_mean + self.norm_eps)
            linear_out = F.linear(x_chunk, hc_fn_fp32)
            mixes_2d[start:end].copy_(linear_out.mul_(rsqrt))

        return mixes

    def hc_pre_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2)

        sinkhorn = _resolve_sinkhorn_impl(x.device)

        # HC params are nominally fp32; promote in case the smoke / a
        # downstream caller cast the model to bf16 uniformly.
        hc_fn_fp32 = hc_fn.float()
        hc_scale_fp32 = hc_scale.float()
        hc_base_fp32 = hc_base.float()

        assert _HYPER_CONNECTION_MIXER_NO_GRAD
        with torch.no_grad():
            mixes = self._compute_mixes(x_flat, hc_fn_fp32)
            pre, post, comb = sinkhorn(
                mixes, hc_scale_fp32, hc_base_fp32, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            )
            assert not pre.requires_grad
            assert not post.requires_grad
            assert not comb.requires_grad

        pre_expanded = pre.unsqueeze(-1)
        x_viewed = x_flat.view(shape)
        y = torch.sum(pre_expanded * x_viewed, dim=2)
        return y.to(dtype), post, comb

    def hc_post_raw(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        post_expanded = post.unsqueeze(-1)
        x_expanded = x.unsqueeze(-2)
        term1 = post_expanded * x_expanded
        comb_expanded = comb.unsqueeze(-1)
        residual_expanded = residual.unsqueeze(-2)
        term2 = torch.sum(comb_expanded * residual_expanded, dim=2)
        y = term1 + term2
        return y.type_as(x)

    def hc_head_raw(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        # The HC params are nominally fp32 (loaded as fp32 from the
        # checkpoint and marked ``_keep_fp32 = True``); the compute below
        # promotes to fp32 internally regardless of input dtype, so the
        # bf16-cast smoke path works fine. Keep a debug check rather than
        # a hard assert.
        if not (hc_fn.dtype == hc_scale.dtype == hc_base.dtype):
            raise RuntimeError(
                f"HC params have mismatched dtypes: hc_fn={hc_fn.dtype}, "
                f"hc_scale={hc_scale.dtype}, hc_base={hc_base.dtype}"
            )

        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2)

        # Promote HC params to fp32 in case the model was cast to bf16.
        hc_fn_fp32 = hc_fn.float()
        hc_scale_fp32 = hc_scale.float()
        hc_base_fp32 = hc_base.float()

        assert _HYPER_CONNECTION_MIXER_NO_GRAD
        with torch.no_grad():
            mixes = self._compute_mixes(x_flat, hc_fn_fp32)
            scaled = mixes * hc_scale_fp32 + hc_base_fp32
            pre = torch.sigmoid(scaled) + self.hc_eps
            assert not pre.requires_grad

        pre_expanded = pre.unsqueeze(-1)
        x_viewed = x_flat.view(shape)
        y = torch.sum(pre_expanded * x_viewed, dim=2)
        return y.to(dtype)

    def layer_pre(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pre-sublayer mixer.

        Args:
            hidden_states: ``[batch, seqlen, hc_mult, hidden]`` (BSHD-style).
        Returns:
            ``(out [B, S, hidden], post [B, S, hc], comb [B, S, hc, hc])``.
        """
        # The HC params are nominally fp32 (loaded as fp32 from the
        # checkpoint and marked ``_keep_fp32 = True``); the compute below
        # promotes to fp32 internally regardless of input dtype, so the
        # bf16-cast smoke path works fine. Keep a debug check rather than
        # a hard assert.
        if not (hc_fn.dtype == hc_scale.dtype == hc_base.dtype):
            raise RuntimeError(
                f"HC params have mismatched dtypes: hc_fn={hc_fn.dtype}, "
                f"hc_scale={hc_scale.dtype}, hc_base={hc_base.dtype}"
            )

        return self.hc_pre_raw(x=hidden_states, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)

    def layer_post(
        self,
        output_with_bias: Tensor | tuple[Tensor, Tensor | None],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        """Post-sublayer mixer.

        Args:
            output_with_bias: ``[batch, seqlen, hidden]`` (the sublayer output)
                or a ``(tensor, bias=None)`` tuple.
            residual: ``[batch, seqlen, hc_mult, hidden]``.
            post / comb: as returned by ``layer_pre``.
        Returns:
            ``[batch, seqlen, hc_mult, hidden]``.
        """
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            assert bias is None
        else:
            out = output_with_bias
        assert isinstance(out, torch.Tensor)

        return self.hc_post_raw(x=out, residual=residual, post=post, comb=comb)

    def block_expand(self, hidden_states: Tensor) -> Tensor:
        """Expand a 3-D ``[B, S, hidden]`` state into 4-D ``[B, S, hc_mult, hidden]``."""
        return einops.repeat(hidden_states, "b s d -> b s hc d", hc=self.hc_mult)

    def block_head(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> Tensor:
        """Collapse the ``hc_mult`` streams back to a single ``[B, S, hidden]`` state."""
        return self.hc_head_raw(x=hidden_states, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base)
