from __future__ import annotations

import torch
import torch.nn.functional as F


def _apply_activation(y: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in {"silu", "swish"}:
        return F.silu(y)
    if activation in {None, "identity"}:
        return y
    raise ValueError(f"Unsupported activation: {activation}")


def causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal depthwise 1D convolution over the sequence dim, with activation applied after.

    Args:
        x: input of shape ``[batch, seq_len, channels]``.
        weight: depthwise kernel of shape ``[channels, 1, kernel_size]`` (``nn.Conv1d`` layout)
            or ``[channels, kernel_size]``.
        bias: optional per-channel bias of shape ``[channels]``.
        activation: ``"silu"``/``"swish"``, or ``None``/``"identity"`` for no activation.
        seq_idx: optional ``[batch, seq_len]`` integer sequence index for packed varlen rows
            (``mamba_ssm`` ``causal_conv1d`` semantics): taps that would read a position with a
            different ``seq_idx`` are zeroed, so the convolution never bleeds across packed
            sequence boundaries.

    Returns:
        Tensor of shape ``[batch, seq_len, channels]``.
    """
    if weight.dim() == 2:
        weight = weight.unsqueeze(1)
    seq_len = x.shape[1]
    if seq_idx is None:
        y = F.conv1d(x.transpose(1, 2), weight, bias, padding=weight.shape[-1] - 1, groups=weight.shape[0])
        y = y[..., :seq_len].transpose(1, 2)
        return _apply_activation(y, activation)

    if seq_idx.shape != x.shape[:2]:
        raise ValueError(f"Expected seq_idx of shape {tuple(x.shape[:2])}, got {tuple(seq_idx.shape)}.")
    # Unrolled causal conv: y[t] = sum_d w[K-1-d] * x[t-d], keeping only taps whose source
    # position belongs to the same packed sequence as t.
    taps = weight.squeeze(1)  # [channels, kernel_size]
    kernel_size = taps.shape[-1]
    y = x * taps[:, -1]
    for offset in range(1, kernel_size):
        shifted = F.pad(x, (0, 0, offset, 0))[:, :seq_len]
        same_seq = F.pad(seq_idx, (offset, 0), value=-1)[:, :seq_len] == seq_idx
        y = y + shifted * same_seq.unsqueeze(-1).to(x.dtype) * taps[:, kernel_size - 1 - offset]
    if bias is not None:
        y = y + bias
    return _apply_activation(y, activation)
