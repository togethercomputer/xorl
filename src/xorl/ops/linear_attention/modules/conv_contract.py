"""GDN conv contract: trainer adopts serving's causal_conv1d prefill kernel.

The measured dominant per-element GDN term is the short conv: xorl's three torch
depthwise ``ShortConvolution``s vs serving's packed ``causal_conv1d`` (~43% element
mismatch, max ~1.6e-2 on real Qwen3.5 layer-0 weights). This wrapper closes it:

- forward: the vendored serving Triton varlen kernel on the packed qkv view,
  invoked exactly as ``gdn_backend.forward_extend`` does (channel-last ``[dim, T]``
  transpose view, fresh zero window states, ``has_initial_state=False``). The
  packed weight is a cat of the three split parameters — a pure memcpy, so split
  checkpoints stay canonical. The kernel is deterministic, layout-invariant and
  packed-vs-split bitwise-invariant (measured), and bitwise-stable across the
  trainer/serving venv pair (triton 3.6.0/torch 2.10 == 3.5.1/torch 2.9.1).
- backward: closed-form torch depthwise recompute on the saved raw inputs
  (the pre-contract trainer composition), so trainability is preserved without a
  handwritten kernel; grads match torch autograd of the depthwise composition.

Decode composability (window-state adapter, for the decode PR): serving decode's
``causal_conv1d_update`` window cache ``[slots, dim, width-1]`` holds the last
``width-1`` RAW post-projection inputs per channel; this prefill kernel writes the
same windows into ``conv_states`` (measured bitwise-equal across kernels). A
recompute-decode lane can therefore rebuild serving windows from raw-projection
buffers, and the trainer-side ``ShortConvolution`` cache (last ``width`` raw
inputs) maps onto it as ``serving_window = trainer_state[..., 1:]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.ops.causal_conv1d_triton import causal_conv1d_fn
from xorl.ops.linear_attention.ops.cp import FLACPContext, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd


def _pack_conv_weight(*weights: torch.Tensor) -> torch.Tensor:
    """Split ``nn.Conv1d`` ``[C, 1, W]`` weights -> serving packed ``[dim, W]`` view (memcpy, no rounding)."""
    return torch.cat([w.reshape(w.shape[0], w.shape[-1]) for w in weights], dim=0)


def _depthwise_recompute(
    x_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    activation: str | None,
    seq_lens: list[int],
    prefix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch depthwise composition used for backward (per-sequence causal conv + silu)."""
    width = weight_packed.shape[-1]
    weight = weight_packed.unsqueeze(1)
    outputs = []
    start = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        seq = x_packed[start : start + seq_len]
        prefix_len = 0
        if seq_idx == 0 and prefix is not None:
            prefix_len = prefix.shape[0]
            seq = torch.cat((prefix, seq), dim=0)
        seq_t = seq.transpose(0, 1).unsqueeze(0)
        y = F.conv1d(seq_t, weight, None, padding=width - 1, groups=weight.shape[0])[..., : seq.shape[0]]
        outputs.append(y[..., prefix_len:].squeeze(0).transpose(0, 1))
        start += seq_len
    y = torch.cat(outputs, dim=0)
    if activation in {"silu", "swish"}:
        return F.silu(y)
    if activation in {None, "identity"}:
        return y
    raise ValueError(f"Unsupported activation: {activation}")


def _prepare_cp_prefix(
    x_packed: torch.Tensor,
    cp_context: FLACPContext,
    prefix_width: int,
) -> tuple[torch.Tensor, int]:
    if cp_context.group is None or cp_context.cu_seqlens is None:
        raise ValueError("Exact Qwen3.5 CP convolution requires a fully initialized cp_context.")
    if cp_context.is_first_rank is None or cp_context.pre_num_conv_tokens is None:
        raise ValueError("Exact Qwen3.5 CP convolution requires packed-sequence boundary metadata.")
    if cp_context.conv1d_kernel_size not in {None, prefix_width + 1}:
        raise ValueError(
            "Exact Qwen3.5 CP convolution kernel size does not match cp_context: "
            f"kernel={prefix_width + 1}, context={cp_context.conv1d_kernel_size}."
        )
    if prefix_width > 0 and x_packed.shape[0] < prefix_width:
        raise ValueError(
            "Exact Qwen3.5 CP convolution requires each Ulysses shard to contain at least "
            f"kernel_size - 1 tokens, got local_tokens={x_packed.shape[0]} and prefix_width={prefix_width}."
        )

    tails = x_packed.new_zeros(prefix_width, x_packed.shape[-1])
    if prefix_width == 0:
        return tails, 0
    tails[-prefix_width:] = x_packed[-prefix_width:]
    previous_tails = conv_cp_send_recv_fwd(tails.contiguous(), cp_context.group)
    valid_len = 0 if cp_context.is_first_rank else min(prefix_width, cp_context.pre_num_conv_tokens)
    prefix = torch.zeros_like(tails)
    if valid_len > 0:
        prefix[-valid_len:] = previous_tails[-valid_len:]
    return prefix, valid_len


class _CausalConv1dContract(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_packed: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
        weight_v: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: tuple[int, ...],
        activation: str | None,
        cp_context: FLACPContext | None,
    ) -> torch.Tensor:
        weight_packed = _pack_conv_weight(weight_q, weight_k, weight_v)
        num_seqs = len(seq_lens)
        device = x_packed.device
        prefix_width = weight_packed.shape[-1] - 1
        if cp_context is None:
            prefix = x_packed.new_empty(0, x_packed.shape[-1])
            prefix_valid_len = 0
        else:
            prefix, prefix_valid_len = _prepare_cp_prefix(x_packed, cp_context, prefix_width)
        # Serving invocation (gdn_backend.forward_extend): channel-last [dim, T]
        # view. Under CP, the first local sequence resumes from the raw projection
        # tail exchanged from the preceding Ulysses rank; every local document
        # boundary still starts from a zero window. conv_states is scratch after
        # the kernel consumes that initial prefix.
        conv_states = torch.zeros(num_seqs, x_packed.shape[1], prefix_width, device=device, dtype=x_packed.dtype)
        has_initial_state = torch.zeros(num_seqs, device=device, dtype=torch.bool)
        if prefix_valid_len > 0:
            conv_states[0].copy_(prefix.transpose(0, 1))
            has_initial_state[0] = True
        out = causal_conv1d_fn(
            x_packed.transpose(0, 1),
            weight_packed,
            None,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=list(seq_lens),
            cache_indices=torch.arange(num_seqs, device=device, dtype=torch.int32),
            has_initial_state=has_initial_state,
            activation=activation,
        ).transpose(0, 1)
        ctx.save_for_backward(x_packed, weight_q, weight_k, weight_v, prefix)
        ctx.seq_lens = seq_lens
        ctx.activation = activation
        ctx.cp_context = cp_context.copy_for_backward() if cp_context is not None else None
        ctx.prefix_valid_len = prefix_valid_len
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_packed, weight_q, weight_k, weight_v, prefix = ctx.saved_tensors
        with torch.enable_grad():
            x_leaf = x_packed.detach().requires_grad_(True)
            w_leaves = [w.detach().requires_grad_(True) for w in (weight_q, weight_k, weight_v)]
            prefix_leaf = prefix.detach().requires_grad_(True) if ctx.cp_context is not None else None
            inputs = [x_leaf, *w_leaves]
            if prefix_leaf is not None:
                inputs.append(prefix_leaf)
            y = _depthwise_recompute(
                x_leaf,
                _pack_conv_weight(*w_leaves),
                ctx.activation,
                list(ctx.seq_lens),
                prefix=prefix_leaf,
            )
            grads = torch.autograd.grad(y, inputs, grad_output)

        dx = grads[0]
        if ctx.cp_context is not None:
            dprefix = grads[-1]
            if ctx.prefix_valid_len < prefix.shape[0]:
                dprefix = dprefix.clone()
                invalid_len = prefix.shape[0] - ctx.prefix_valid_len
                dprefix[:invalid_len].zero_()
            recv_dprefix = conv_cp_send_recv_bwd(dprefix.contiguous(), ctx.cp_context.group)
            tail_len = min(dx.shape[0], prefix.shape[0])
            if tail_len > 0:
                dx[-tail_len:].add_(recv_dprefix[-tail_len:])
        return dx, grads[1], grads[2], grads[3], None, None, None, None


def causal_conv1d_qkv_contract(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    q_conv: torch.nn.Module,
    k_conv: torch.nn.Module,
    v_conv: torch.nn.Module,
    cu_seqlens: torch.Tensor | None = None,
    cp_context: FLACPContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serving-bit short conv over the packed (q|k|v) projection outputs.

    Args:
        q_input/k_input/v_input: raw projection outputs ``[B, T, dim_*]``; packed
            varlen inputs use ``B == 1`` plus ``cu_seqlens``, batched inputs use
            ``cu_seqlens=None`` (each row is one sequence, as in serving).
        q_conv/k_conv/v_conv: the split ``ShortConvolution`` modules (weights stay
            split for checkpoint/weight-sync compatibility; packing is a
            forward-time view).
        cu_seqlens: optional ``[num_seqs + 1]`` boundaries for packed inputs.
        cp_context: optional Ulysses metadata. The contract exchanges the raw
            ``kernel_size - 1`` projection tail and feeds it to serving's own
            ``has_initial_state`` path for the first continued local sequence.

    Returns:
        The three activated conv outputs, shaped like the inputs.
    """
    convs = (q_conv, k_conv, v_conv)
    activation = q_conv.activation
    for conv in convs:
        if conv.bias is not None:
            raise NotImplementedError("Exact Qwen3.5 GDN does not support convolution bias")
        if conv.activation != activation:
            raise ValueError("Exact Qwen3.5 GDN requires one activation across q/k/v convolutions")

    batch_size, seq_len = q_input.shape[0], q_input.shape[1]
    x_packed = torch.cat((q_input, k_input, v_input), dim=-1)
    if cu_seqlens is None:
        seq_lens = (seq_len,) * batch_size
        boundaries = torch.arange(0, (batch_size + 1) * seq_len, seq_len, device=q_input.device, dtype=torch.int32)
    else:
        if batch_size != 1:
            raise ValueError("Packed varlen conv contract expects batch size 1.")
        starts = cu_seqlens.tolist()
        seq_lens = tuple(end - start for start, end in zip(starts[:-1], starts[1:], strict=False))
        boundaries = cu_seqlens.to(device=q_input.device, dtype=torch.int32)

    out = _CausalConv1dContract.apply(
        x_packed.reshape(-1, x_packed.shape[-1]),
        q_conv.weight,
        k_conv.weight,
        v_conv.weight,
        boundaries,
        seq_lens,
        activation,
        cp_context,
    ).view_as(x_packed)
    return out.split((q_input.shape[-1], k_input.shape[-1], v_input.shape[-1]), dim=-1)
