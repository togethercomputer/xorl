from __future__ import annotations

# The K3 GDN trainer contract (P5), vendored from xorl-sglang.
#
# Under XORL_BI_GDN=1 the two remaining GDN cross-engine terms that live on the
# trainer side are contracted by running serving's exact kernels with real
# gradients (closed-form backward — gradients do not enter the forward K3):
#
#   - gating: the trainer's torch-composed
#     ``g = -exp(A_log) * softplus(a + dt_bias)`` differs from serving's
#     ``fused_gdn_gating`` triton kernel by 1 fp32 ULP on ~40% of elements
#     (tl.log(1+tl.exp) vs torch softplus). The serving kernel is vendored
#     byte-identically below.
#   - gated RMSNorm: the trainer's fp32 torch composition differs from
#     serving's ``layernorm_gated`` triton kernel on a rare bf16-ULP tail
#     (~1e-5 of elements; reduction-order + rsqrt + multiply-association).
#     The serving forward kernel is vendored byte-identically below
#     (per-row numerics are invariant to ROWS_PER_BLOCK, which only tiles
#     rows; serving pins 4 under piecewise cuda graphs and so do we).
#
# Kernel bodies are byte-identical to xorl-sglang @ the PR base
# (python/sglang/srt/layers/attention/fla/{fused_gdn_gating,layernorm_gated}.py);
# only imports/drivers are adapted. The companion serving-side flag is
# SGLANG_BI_GDN_PREFILL (scan composition adoption); enable both for the
# Qwen3.5 GDN parity lane, alongside XORL_BI_TRUNK_LINEAR for the trunk.
#
# Portions adapted from flash-linear-attention (Tri Dao layernorm_gated),
# MIT/BSD licenses per upstream.
import os

import torch
import triton
import triton.language as tl


def is_gdn_contract_enabled() -> bool:
    return os.environ.get("XORL_BI_GDN", "").lower() in {"1", "true", "yes", "on"}


# --- vendored from sglang fla/fused_gdn_gating.py -----------------------------


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # P5 fp32-beta convention (paired with serving's fused_gdn_gating):
    # beta stays fp32; the old bf16 round disagreed with decode's in-kernel
    # fp32 gating on every token.
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask)


def _fused_gdn_gating_fwd(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serving's fused_gdn_gating driver over flattened [rows, num_heads]."""
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


class _BIFusedGDNGating(torch.autograd.Function):
    """Serving-exact GDN gating forward; closed-form backward.

    Forward bits match serving's ``fused_gdn_gating``:
    ``g = -exp(A_log) * softplus(a + dt_bias)`` (fp32),
    ``beta = sigmoid(b)`` (fp32 — the fp32-beta convention, paired with the
    serving-side change). Backward uses the analytic derivatives of the
    reference composition.
    """

    @staticmethod
    def forward(ctx, A_log, a, b, dt_bias):
        rows = a.shape[:-1].numel()
        num_heads = a.shape[-1]
        a2 = a.reshape(rows, num_heads).contiguous()
        b2 = b.reshape(rows, num_heads).contiguous()
        g, beta = _fused_gdn_gating_fwd(A_log, a2, b2, dt_bias)
        g = g.view(*a.shape[:-1], num_heads)
        beta = beta.view(*a.shape[:-1], num_heads)
        ctx.save_for_backward(A_log, a, b, dt_bias, g)
        return g, beta

    @staticmethod
    def backward(ctx, dg, dbeta):
        A_log, a, b, dt_bias, g = ctx.saved_tensors
        x = a.float() + dt_bias.float()
        neg_exp_a_log = -A_log.float().exp()
        # d softplus(x) / dx == sigmoid(x); the kernel's threshold branch
        # (softplus(x) = x for x > 20) has derivative ~1 == sigmoid(x) there.
        dsp = torch.sigmoid(x)
        da = dg.float() * neg_exp_a_log * dsp
        ddt_bias = da.reshape(-1, da.shape[-1]).sum(0)
        # dg/dA_log = -exp(A_log) * softplus(x) == g
        dA_log = (dg.float() * g).reshape(-1, g.shape[-1]).sum(0)
        sig_b = torch.sigmoid(b.float())
        db = dbeta.float() * sig_b * (1.0 - sig_b)
        return (
            dA_log.to(A_log.dtype),
            da.to(a.dtype),
            db.to(b.dtype),
            ddt_bias.to(dt_bias.dtype),
        )


def bi_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable, serving-bit-exact GDN gating ``(g, beta)`` (both fp32,
    shaped like ``a``)."""
    return _BIFusedGDNGating.apply(A_log, a, b, dt_bias)


# --- vendored from sglang fla/layernorm_gated.py ------------------------------

# Serving tiles rows in blocks of MAX_ROWS_PER_BLOCK under piecewise cuda
# graphs (the production default); per-row numerics are invariant to the row
# tiling, and the contract gate verifies it.
MAX_ROWS_PER_BLOCK = 4


@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)


def _bi_rms_norm_gated_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    eps: float,
    activation: str = "swish",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serving's ``_layer_norm_fwd`` for the GDN gated-RMSNorm contract config
    (is_rms_norm, norm_before_gate, no bias, single group). Returns (y, rstd)."""
    M, N = x.shape
    assert x.stride(-1) == 1 and z.stride(-1) == 1
    assert z.shape == (M, N)
    assert weight.shape == (N,)
    out = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    rows_per_block = MAX_ROWS_PER_BLOCK
    grid = (triton.cdiv(M, rows_per_block), 1)
    _layer_norm_fwd_1pass_kernel[grid](
        x,
        out,
        weight,
        None,
        z,
        None,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0),
        M,
        N,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=False,
        HAS_Z=True,
        NORM_BEFORE_GATE=True,
        IS_RMS_NORM=True,
        num_warps=num_warps,
        ACTIVATION=activation,
    )
    return out, rstd


class _BIRMSNormGated(torch.autograd.Function):
    """Serving-exact gated RMSNorm forward; closed-form backward.

    Forward bits match serving's ``rms_norm_gated`` (fla/layernorm_gated) for
    the GDN contract config: ``y = (x * rstd * w) * silu(z)`` with the kernel's
    fp32 intermediates. Backward reuses the kernel-saved rstd.
    """

    @staticmethod
    def forward(ctx, x, weight, z, eps, activation):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        z2 = z.reshape(-1, shape[-1]).contiguous()
        w = weight.contiguous()
        y, rstd = _bi_rms_norm_gated_fwd(x2, w, z2, eps, activation=activation)
        ctx.save_for_backward(x2, w, z2, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, z2, rstd = ctx.saved_tensors
        n_cols = x2.shape[-1]
        dy2 = dy.reshape(-1, n_cols).float()
        xf = x2.float()
        zf = z2.float()
        wf = w.float()
        r = rstd[:, None]
        n = xf * r
        sig_z = torch.sigmoid(zf)
        silu_z = zf * sig_z
        yw = n * wf
        dz = dy2 * yw * (sig_z * (1.0 + zf * (1.0 - sig_z)))
        d_yw = dy2 * silu_z
        dw = (d_yw * n).sum(0)
        dn = d_yw * wf
        dx = r * (dn - n * (dn * n).mean(dim=-1, keepdim=True))
        return (
            dx.to(x2.dtype).view(ctx.shape),
            dw.to(w.dtype),
            dz.to(z2.dtype).view(ctx.shape),
            None,
            None,
        )


def bi_rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    eps: float,
    activation: str = "swish",
) -> torch.Tensor:
    """Differentiable, serving-bit-exact gated RMSNorm (norm-before-gate)."""
    if activation not in {"swish", "silu", "sigmoid"}:
        raise ValueError(f"Unsupported activation: {activation}")
    return _BIRMSNormGated.apply(x, weight, z, eps, activation)
