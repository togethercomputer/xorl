from __future__ import annotations

# Chunked SSD (Mamba2 / state-space duality) scan.
#
# The torch path is adapted from transformers' NemotronHMamba2Mixer.torch_forward
# (segsum-based chunked SSD), restructured around einsums, with one correctness
# fix: the inter-chunk state recurrence in the transformers 5.5.3 nemotron_h
# torch fallback mis-indexes the chunk decay matrix (the original mamba2 code's
# `decay_chunk.transpose(1, 3)` was lost when the code was restructured), so its
# output drifts from the true SSD recurrence for seq_len > chunk_size. This
# implementation matches the sequential SSD recurrence — and therefore the
# mamba_ssm `mamba_chunk_scan_combined` kernel semantics — exactly; see
# tests/ops/test_ssm_mamba2.py.
import os

import torch
import torch.nn.functional as F


try:  # pragma: no cover - exercised only when mamba_ssm is installed
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

# Set XORL_SSD_FORCE_TORCH=1 to disable the mamba_ssm fast path even when it is importable.
_FORCE_TORCH_ENV = "XORL_SSD_FORCE_TORCH"


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: out[..., i, j] = sum(input[..., j+1:i+1]) for i >= j, -inf above the diagonal."""
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    return tensor_segsum.masked_fill(~mask, -torch.inf)


def _pad_seq(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Zero-pad `pad_size` positions on the seq_len dim (dim=1) of a 3D or 4D tensor."""
    if pad_size == 0:
        return input_tensor
    pad_shape = (0, 0, 0, pad_size) if input_tensor.dim() == 3 else (0, 0, 0, 0, 0, pad_size)
    return F.pad(input_tensor, pad_shape)


def _into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """Pad the seq_len dim to a multiple of chunk_size and split it into [num_chunks, chunk_size]."""
    input_tensor = _pad_seq(input_tensor, pad_size)
    if input_tensor.dim() == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3])


def _ssd_chunked_torch(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    chunk_size: int,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = x.shape
    n_groups = B.shape[2]

    x = x.float()
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    # Broadcast B/C groups to heads (GQA-style sharing).
    B = B.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    C = C.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    d_residual = D[..., None].float() * _pad_seq(x, pad_size) if D is not None else None

    # Discretize x and A.
    x = x * dt[..., None]
    a = A * dt  # [batch, seq_len, heads], negative

    # Rearrange into chunks: x [b,c,l,h,p], a [b,c,l,h], B/C [b,c,l,h,n].
    x, a, B, C = (_into_chunks(t, pad_size, chunk_size) for t in (x, a, B, C))
    a = a.permute(0, 3, 1, 2)  # [batch, heads, chunks, chunk_size]
    a_cumsum = torch.cumsum(a, dim=-1)

    # Packed-varlen boundary masks. seq_idx is non-decreasing along the row, so "same
    # sequence" between two positions implies every position in between is too; cutting
    # the L matrix, the carried-state contributions and the state->output terms at
    # index mismatches is exactly a state reset at each boundary (mamba_ssm semantics).
    if seq_idx is not None:
        seq_idx = seq_idx.long()
        if pad_size > 0:  # pad the tail with the last sequence's index (outputs there are dropped)
            seq_idx = torch.cat([seq_idx, seq_idx[:, -1:].expand(-1, pad_size)], dim=1)
        seq_idx = seq_idx.reshape(batch_size, -1, chunk_size)  # [b,c,l]
        chunk_first = seq_idx[:, :, 0]  # [b,c]
        chunk_last = seq_idx[:, :, -1]  # [b,c]

    # 1. Intra-chunk output (diagonal blocks): causal attention-like weights.
    decay = torch.exp(segment_sum(a))  # [b,h,c,l,s], l = target pos, s = source pos
    if seq_idx is not None:
        decay = decay * (seq_idx[:, :, :, None] == seq_idx[:, :, None, :]).unsqueeze(1).to(decay.dtype)
    attn = torch.einsum("bclhn,bcshn,bhcls->bclsh", C, B, decay)
    y_diag = torch.einsum("bclsh,bcshp->bclhp", attn, x)

    # 2. State at the end of each chunk (assuming zero initial state).
    decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)  # [b,h,c,l]
    if seq_idx is not None:
        # Only positions in the chunk's final sequence feed the carried-out state.
        decay_states = decay_states * (seq_idx == chunk_last[:, :, None]).unsqueeze(1).to(decay_states.dtype)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, x)

    # 3. Inter-chunk recurrence on the chunk-boundary states.
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(a_cumsum[:, :, :, -1], (1, 0))))  # [b,h,z,c], z >= c
    if seq_idx is not None:
        # decay_chunk[z, c] carries the end-of-chunk-(c-1) state into chunk z; allow it only
        # when the sequence runs unbroken from the end of chunk c-1 to the start of chunk z
        # (equivalent, by monotonicity, to their sequence indices matching). c=0 is the zero
        # initial state, masked harmlessly.
        first_ext = torch.cat([chunk_first, chunk_first[:, -1:]], dim=1)  # [b, z]
        last_prev = F.pad(chunk_last, (1, 0), value=-1)  # [b, c], entry c -> chunk c-1
        decay_chunk = decay_chunk * (first_ext[:, :, None] == last_prev[:, None, :]).unsqueeze(1).to(decay_chunk.dtype)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]  # state entering each chunk

    # 4. State -> output conversion per chunk (off-diagonal blocks).
    state_decay_out = torch.exp(a_cumsum)  # [b,h,c,l]
    if seq_idx is not None:
        # The carried-in state only reaches positions before the first boundary in the chunk.
        state_decay_out = state_decay_out * (seq_idx == chunk_first[:, :, None]).unsqueeze(1).to(state_decay_out.dtype)
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    y = (y_diag + y_off).reshape(batch_size, -1, num_heads, head_dim)
    if d_residual is not None:
        y = y + d_residual
    if pad_size > 0:
        y = y[:, :seq_len]
    return y


def ssd_chunked(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    *,
    chunk_size: int = 128,
    seq_idx: torch.Tensor | None = None,
    use_kernel: bool | None = None,
) -> torch.Tensor:
    """Chunked SSD scan: y_t = C_t @ h_t + D * x_t with h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t.

    Trainable (pure-torch by default); state math runs in fp32 regardless of input dtype.
    seq_len does not need to be divisible by chunk_size (the tail is zero-padded internally).

    Args:
        x: input per head, ``[batch, seq_len, num_heads, head_dim]``.
        dt: discretization step (post-softplus/clamp, positive), ``[batch, seq_len, num_heads]``.
        A: per-head decay rate (negative, e.g. ``-exp(A_log)``), ``[num_heads]``.
        B: input projection per group, ``[batch, seq_len, n_groups, state_size]``; ``n_groups``
            must divide ``num_heads``.
        C: output projection per group, ``[batch, seq_len, n_groups, state_size]``.
        D: optional per-head skip connection, ``[num_heads]``.
        chunk_size: scan chunk length.
        seq_idx: optional ``[batch, seq_len]`` non-decreasing integer sequence index for packed
            varlen rows (``mamba_ssm`` semantics): the SSM state is reset at every index change,
            so no state propagates across packed sequence boundaries.
        use_kernel: ``True`` forces the ``mamba_ssm`` ``mamba_chunk_scan_combined`` fast path
            (raises if not importable), ``False`` forces the torch path, ``None`` (default)
            auto-selects the kernel when importable, on CUDA, and not disabled via
            ``XORL_SSD_FORCE_TORCH=1``.

    Returns:
        y: ``[batch, seq_len, num_heads, head_dim]`` in float32.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x of shape [batch, seq_len, num_heads, head_dim], got {tuple(x.shape)}.")
    batch_size, seq_len, num_heads, _ = x.shape
    if dt.shape != (batch_size, seq_len, num_heads):
        raise ValueError(f"Expected dt of shape {(batch_size, seq_len, num_heads)}, got {tuple(dt.shape)}.")
    if B.dim() != 4 or C.shape != B.shape or B.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            f"Expected B and C of shape [batch, seq_len, n_groups, state_size], got {tuple(B.shape)} "
            f"and {tuple(C.shape)}."
        )
    if num_heads % B.shape[2] != 0:
        raise ValueError(f"n_groups={B.shape[2]} must divide num_heads={num_heads}.")
    if seq_idx is not None and seq_idx.shape != (batch_size, seq_len):
        raise ValueError(f"Expected seq_idx of shape {(batch_size, seq_len)}, got {tuple(seq_idx.shape)}.")

    if use_kernel is None:
        use_kernel = (
            mamba_chunk_scan_combined is not None and x.is_cuda and os.environ.get(_FORCE_TORCH_ENV, "0") != "1"
        )
    elif use_kernel and mamba_chunk_scan_combined is None:
        raise RuntimeError("ssd_chunked(use_kernel=True) requires mamba_ssm, which is not installed.")

    if use_kernel:
        y = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            z=None,
            dt_bias=None,
            seq_idx=seq_idx.to(torch.int32).contiguous() if seq_idx is not None else None,
            dt_softplus=False,
        )
        return y.float()
    return _ssd_chunked_torch(x, dt, A, B, C, D, chunk_size, seq_idx=seq_idx)
