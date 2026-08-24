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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ExactMhcReplaySegment:
    """One serving-sized MHC launch projected onto this rank's local rows.

    ``launch_rows`` preserves the serving launch geometry (and therefore its
    TF32 split count). ``source_rows`` names compact trainer rows, while
    ``launch_positions`` places those rows at their positions in the serving
    launch. Rows owned by other CP ranks are zero placeholders; MHC pre-norm
    has no cross-token arithmetic, so they cannot affect selected local rows.
    """

    launch_rows: int
    source_rows: tuple[int, ...]
    launch_positions: tuple[int, ...]


def _exact_mhc_pre_norm_forward(
    residual: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    norm_weight: Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the literal SGLang SM90 MHC-pre plus RMSNorm value path."""

    if not residual.is_cuda or residual.dtype != torch.bfloat16:
        raise RuntimeError("DSV4 exact MHC pre requires CUDA BF16 residuals")
    if residual.shape[-2:] != (4, 4096):
        raise RuntimeError(
            "DSV4 exact MHC pre admits only the official (hc_mult=4, hidden=4096) geometry, "
            f"got {tuple(residual.shape[-2:])}"
        )
    if hc_fn.dtype != torch.float32 or hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise RuntimeError("DSV4 exact MHC routing parameters must remain FP32")

    from sglang.kernels.ops.layernorm.mhc import (  # noqa: PLC0415
        _compute_num_split_for_mhc_pre,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (  # noqa: PLC0415
        tf32_hc_prenorm_gemm,
    )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.contiguous().view(-1, 4, 4096)
    num_tokens = residual_flat.shape[0]
    n_splits = _compute_num_split_for_mhc_pre(num_tokens, 4 * 4096)
    gemm_out_mul = torch.empty(n_splits, num_tokens, 24, dtype=torch.float32, device=residual.device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=residual.device)
    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, 4 * 4096),
        hc_fn.contiguous(),
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    post = torch.empty(num_tokens, 4, dtype=torch.float32, device=residual.device)
    comb = torch.empty(num_tokens, 16, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(num_tokens, 4096, dtype=torch.bfloat16, device=residual.device)
    norm_weight_bf16 = norm_weight.bfloat16().contiguous()
    mhc_pre_big_fuse_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale.contiguous(),
        hc_base.contiguous(),
        residual_flat,
        post,
        comb,
        layer_input,
        norm_weight_bf16,
        4096,
        rms_eps,
        hc_eps,
        hc_eps,
        2.0,
        sinkhorn_iters,
        rms_eps,
        n_splits,
        4,
        24,
    )
    return (
        layer_input.view(*outer_shape, 4096),
        post.view(*outer_shape, 4),
        comb.view(*outer_shape, 4, 4),
    )


def _exact_mhc_post_forward(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Run the literal SGLang SM90 MHC-post value path."""

    if not residual.is_cuda or residual.dtype != torch.bfloat16:
        raise RuntimeError("DSV4 exact MHC post requires CUDA BF16 residuals")
    if residual.shape[-2:] != (4, 4096) or x.shape[-1] != 4096:
        raise RuntimeError("DSV4 exact MHC post admits only official DSV4-Flash geometry")
    from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang  # noqa: PLC0415

    outer_shape = residual.shape[:-2]
    x_flat = x.contiguous().view(-1, 4096)
    residual_flat = residual.contiguous().view(-1, 4, 4096)
    post_flat = post.contiguous().view(-1, 4)
    comb_flat = comb.contiguous().view(-1, 4, 4)
    out = torch.empty_like(residual_flat)
    mhc_post_tilelang(comb_flat, residual_flat, post_flat, x_flat, out, 4, 4096)
    return out.view(*outer_shape, 4, 4096)


class _ExactMhcPreNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, hc_fn, hc_scale, hc_base, norm_weight, rms_eps, hc_eps, sinkhorn_iters):
        layer_input, post, comb = _exact_mhc_pre_norm_forward(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
        )
        ctx.save_for_backward(residual, hc_fn, hc_scale, hc_base, norm_weight)
        ctx.rms_eps = rms_eps
        ctx.hc_eps = hc_eps
        return layer_input, post, comb

    @staticmethod
    def backward(ctx, grad_layer_input, grad_post, grad_comb):
        residual, hc_fn, hc_scale, hc_base, norm_weight = ctx.saved_tensors
        with torch.enable_grad():
            surrogate = residual.detach().requires_grad_(True)
            flat = surrogate.flatten(-2).float()
            rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + ctx.rms_eps)
            mixes = F.linear(flat, hc_fn) * rsqrt
            pre = (torch.sigmoid(mixes[..., :4] * hc_scale[0] + hc_base[:4]) + ctx.hc_eps).detach()
            mixed = (pre.unsqueeze(-1) * surrogate.float()).sum(dim=-2).to(torch.bfloat16)
            mixed_fp32 = mixed.float()
            normed = mixed_fp32 * torch.rsqrt(mixed_fp32.square().mean(-1, keepdim=True) + ctx.rms_eps)
            normed = (normed * norm_weight.float()).to(torch.bfloat16)
            grad_residual = torch.autograd.grad(
                normed,
                surrogate,
                grad_layer_input,
                retain_graph=False,
                create_graph=False,
            )[0]
        return grad_residual, None, None, None, None, None, None, None


class _ExactMhcPost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, post, comb):
        out = _exact_mhc_post_forward(x, residual, post, comb)
        ctx.save_for_backward(post, comb)
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        post, comb = ctx.saved_tensors
        grad = grad_output.float()
        grad_x = (grad * post.float().unsqueeze(-1)).sum(dim=-2)
        grad_residual = torch.einsum("...jh,...ij->...ih", grad, comb.float())
        return grad_x.to(ctx.x_dtype), grad_residual.to(ctx.residual_dtype), None, None


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

    def layer_pre_norm_exact(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
        norm_weight: Tensor,
        serving_segments: tuple[int | ExactMhcReplaySegment, ...] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if serving_segments is not None:
            if hidden_states.ndim != 4:
                raise ValueError(
                    "DSV4 serving-segment MHC replay requires [B, S, hc, H] residuals, "
                    f"got {tuple(hidden_states.shape)}"
                )
            compute_rows = hidden_states.shape[1]
            layer_inputs = []
            posts = []
            combs = []
            source_order: list[int] = []
            start = 0
            for segment in serving_segments:
                if isinstance(segment, int):
                    if segment <= 0:
                        raise ValueError(f"DSV4 serving MHC segments must be positive, got {serving_segments}")
                    end = start + segment
                    source_rows = tuple(range(start, end))
                    launch_positions = tuple(range(segment))
                    launch_residual = hidden_states[:, start:end]
                    start = end
                else:
                    if segment.launch_rows <= 0:
                        raise ValueError(f"DSV4 serving MHC launch rows must be positive, got {segment}")
                    source_rows = segment.source_rows
                    launch_positions = segment.launch_positions
                    if not source_rows or len(source_rows) != len(launch_positions):
                        raise ValueError(
                            "DSV4 CP serving MHC segments require equally sized nonempty source and launch rows, "
                            f"got {segment}"
                        )
                    if len(set(source_rows)) != len(source_rows) or any(
                        row < 0 or row >= compute_rows for row in source_rows
                    ):
                        raise ValueError(f"DSV4 CP serving MHC source rows are invalid: {segment}")
                    if len(set(launch_positions)) != len(launch_positions) or any(
                        row < 0 or row >= segment.launch_rows for row in launch_positions
                    ):
                        raise ValueError(f"DSV4 CP serving MHC launch positions are invalid: {segment}")
                    source_index = torch.tensor(source_rows, dtype=torch.long, device=hidden_states.device)
                    launch_index = torch.tensor(launch_positions, dtype=torch.long, device=hidden_states.device)
                    local_residual = hidden_states.index_select(1, source_index)
                    launch_residual = hidden_states.new_zeros(
                        hidden_states.shape[0],
                        segment.launch_rows,
                        *hidden_states.shape[2:],
                    ).index_copy(1, launch_index, local_residual)

                layer_input, post, comb = _ExactMhcPreNorm.apply(
                    launch_residual,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    norm_weight,
                    self.norm_eps,
                    self.hc_eps,
                    self.hc_sinkhorn_iters,
                )
                if not isinstance(segment, int):
                    launch_index = torch.tensor(
                        launch_positions,
                        dtype=torch.long,
                        device=hidden_states.device,
                    )
                    layer_input = layer_input.index_select(1, launch_index)
                    post = post.index_select(1, launch_index)
                    comb = comb.index_select(1, launch_index)
                layer_inputs.append(layer_input)
                posts.append(post)
                combs.append(comb)
                source_order.extend(source_rows)

            if sorted(source_order) != list(range(compute_rows)):
                raise ValueError(
                    "DSV4 serving MHC segments must cover the compute rows exactly once: "
                    f"source_rows={source_order} rows={compute_rows}"
                )
            layer_input = torch.cat(layer_inputs, dim=1)
            post = torch.cat(posts, dim=1)
            comb = torch.cat(combs, dim=1)
            if source_order != list(range(compute_rows)):
                restore_order = torch.tensor(
                    sorted(range(compute_rows), key=source_order.__getitem__),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                layer_input = layer_input.index_select(1, restore_order)
                post = post.index_select(1, restore_order)
                comb = comb.index_select(1, restore_order)
            return (
                layer_input,
                post,
                comb,
            )
        return _ExactMhcPreNorm.apply(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            self.norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )

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

    def layer_post_exact(
        self,
        output_with_bias: Tensor | tuple[Tensor, Tensor | None],
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ) -> Tensor:
        if isinstance(output_with_bias, tuple):
            out, bias = output_with_bias
            if bias is not None:
                raise RuntimeError("DSV4 exact MHC post does not admit a separate bias")
        else:
            out = output_with_bias
        return _ExactMhcPost.apply(out, residual, post, comb)

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
