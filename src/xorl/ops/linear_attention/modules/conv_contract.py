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

import logging

import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.ops.causal_conv1d_triton import causal_conv1d_fn
from xorl.ops.linear_attention.ops.cp.comm import conv_cp_send_recv_bwd, conv_cp_send_recv_fwd


logger = logging.getLogger("xorl.gdn_cp_conv")
_cp_engagement_logged = False


def _pack_conv_weight(*weights: torch.Tensor) -> torch.Tensor:
    """Split ``nn.Conv1d`` ``[C, 1, W]`` weights -> serving packed ``[dim, W]`` view (memcpy, no rounding)."""
    return torch.cat([w.reshape(w.shape[0], w.shape[-1]) for w in weights], dim=0)


def _depthwise_recompute(
    x_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    activation: str | None,
    seq_lens: list[int],
) -> torch.Tensor:
    """Torch depthwise composition used for backward (per-sequence causal conv + silu)."""
    width = weight_packed.shape[-1]
    weight = weight_packed.unsqueeze(1)
    outputs = []
    start = 0
    for seq_len in seq_lens:
        seq = x_packed[start : start + seq_len].transpose(0, 1).unsqueeze(0)
        y = F.conv1d(seq, weight, None, padding=width - 1, groups=weight.shape[0])[..., :seq_len]
        outputs.append(y.squeeze(0).transpose(0, 1))
        start += seq_len
    y = torch.cat(outputs, dim=0)
    if activation in {"silu", "swish"}:
        return F.silu(y)
    if activation in {None, "identity"}:
        return y
    raise ValueError(f"Unsupported activation: {activation}")


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
    ) -> torch.Tensor:
        weight_packed = _pack_conv_weight(weight_q, weight_k, weight_v)
        num_seqs = len(seq_lens)
        device = x_packed.device
        # Serving invocation (gdn_backend.forward_extend): channel-last [dim, T]
        # view, fresh zero windows, no initial state. conv_states is scratch here;
        # the kernel writes final windows into it, which prefill discards.
        conv_states = torch.zeros(
            num_seqs, x_packed.shape[1], weight_packed.shape[-1] - 1, device=device, dtype=x_packed.dtype
        )
        out = causal_conv1d_fn(
            x_packed.transpose(0, 1),
            weight_packed,
            None,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=list(seq_lens),
            cache_indices=torch.arange(num_seqs, device=device, dtype=torch.int32),
            has_initial_state=torch.zeros(num_seqs, device=device, dtype=torch.bool),
            activation=activation,
        ).transpose(0, 1)
        ctx.save_for_backward(x_packed, weight_q, weight_k, weight_v)
        ctx.seq_lens = seq_lens
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_packed, weight_q, weight_k, weight_v = ctx.saved_tensors
        with torch.enable_grad():
            x_leaf = x_packed.detach().requires_grad_(True)
            w_leaves = [w.detach().requires_grad_(True) for w in (weight_q, weight_k, weight_v)]
            y = _depthwise_recompute(
                x_leaf,
                _pack_conv_weight(*w_leaves),
                ctx.activation,
                list(ctx.seq_lens),
            )
            grads = torch.autograd.grad(y, [x_leaf, *w_leaves], grad_output)
        return grads[0], grads[1], grads[2], grads[3], None, None, None


class _CausalConv1dContractCP(torch.autograd.Function):
    """CP variant of the conv contract: SAME serving kernel, with the first
    local sequence's window seeded by the (width-1)-token RAW halo from the
    previous rank. Operands cross the wire, never results: the halo rows are
    conv INPUTS; every output byte is computed by this rank inside the
    contracted kernel.

    Backward recomputes the torch depthwise composition WITH the prefix and
    ships the prefix gradient back to the previous rank's tail rows (the
    reverse halo), so trainability is preserved without a handwritten kernel.
    """

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
        halo: torch.Tensor,          # [width-1, dim] raw rows from prev rank (zeros if fresh)
        first_seq_continues: bool,
        cp_group,
    ) -> torch.Tensor:
        weight_packed = _pack_conv_weight(weight_q, weight_k, weight_v)
        width = weight_packed.shape[-1]
        num_seqs = len(seq_lens)
        device = x_packed.device
        conv_states = torch.zeros(
            num_seqs, x_packed.shape[1], width - 1, device=device, dtype=x_packed.dtype
        )
        has_init = torch.zeros(num_seqs, device=device, dtype=torch.bool)
        if first_seq_continues:
            # halo rows are chronological (oldest first); the kernel window
            # axis is likewise oldest-first (causal_conv1d_triton.py:134-140).
            conv_states[0] = halo.transpose(0, 1)
            has_init[0] = True
        out = causal_conv1d_fn(
            x_packed.transpose(0, 1),
            weight_packed,
            None,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=list(seq_lens),
            cache_indices=torch.arange(num_seqs, device=device, dtype=torch.int32),
            has_initial_state=has_init,
            activation=activation,
        ).transpose(0, 1)
        ctx.save_for_backward(x_packed, weight_q, weight_k, weight_v, halo)
        ctx.seq_lens = seq_lens
        ctx.activation = activation
        ctx.first_seq_continues = first_seq_continues
        ctx.cp_group = cp_group
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_packed, weight_q, weight_k, weight_v, halo = ctx.saved_tensors
        prefix_width = halo.shape[0]
        with torch.enable_grad():
            x_leaf = x_packed.detach().requires_grad_(True)
            w_leaves = [w.detach().requires_grad_(True) for w in (weight_q, weight_k, weight_v)]
            halo_leaf = halo.detach().requires_grad_(True)
            if ctx.first_seq_continues:
                # Recompute the ACTUAL forward composition: first sequence
                # convolved with the prefix prepended, prefix outputs dropped.
                first_len = ctx.seq_lens[0]
                x_ext = torch.cat([halo_leaf, x_leaf[:first_len]], dim=0)
                y_first = _depthwise_recompute(
                    x_ext, _pack_conv_weight(*w_leaves), ctx.activation, [prefix_width + first_len]
                )[prefix_width:]
                y_rest = (
                    _depthwise_recompute(
                        x_leaf[first_len:], _pack_conv_weight(*w_leaves), ctx.activation,
                        list(ctx.seq_lens[1:]),
                    )
                    if len(ctx.seq_lens) > 1
                    else x_leaf.new_zeros(0, x_leaf.shape[1])
                )
                y = torch.cat([y_first, y_rest], dim=0)
            else:
                y = _depthwise_recompute(
                    x_leaf, _pack_conv_weight(*w_leaves), ctx.activation, list(ctx.seq_lens)
                )
            grads = torch.autograd.grad(y, [x_leaf, *w_leaves, halo_leaf], grad_output,
                                        allow_unused=True)
        dx, dwq, dwk, dwv, dhalo = grads
        # Reverse halo: my prefix gradient belongs to the PREVIOUS rank's last
        # rows; symmetrically I receive my successor's and add it to my tail.
        # Every rank participates in the collective (zeros when unused).
        send = dhalo if dhalo is not None else torch.zeros_like(halo)
        recv = conv_cp_send_recv_bwd(send.to(x_packed.dtype).contiguous(), ctx.cp_group)
        tail = min(dx.shape[0], recv.shape[0])
        if tail > 0:
            dx[-tail:] = dx[-tail:] + recv[-tail:]
        return dx, dwq, dwk, dwv, None, None, None, None, None, None


def causal_conv1d_qkv_contract(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    q_conv: torch.nn.Module,
    k_conv: torch.nn.Module,
    v_conv: torch.nn.Module,
    cu_seqlens: torch.Tensor | None = None,
    cp_context=None,
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

    if cp_context is not None:
        from xorl.ops.linear_attention.ops.cp.context import FLACPContext  # noqa: PLC0415

        if not isinstance(cp_context, FLACPContext):
            raise TypeError(
                f"Exact GDN CP conv requires a FLACPContext, got {type(cp_context).__name__}. "
                "Fail closed: partial/duck-typed contexts cannot carry the alignment metadata.",
            )
        if cu_seqlens is None:
            raise ValueError("Exact GDN CP conv requires the LOCAL cu_seqlens from cp_context.")
        if cp_context.group is None:
            raise ValueError("Exact GDN CP conv requires an initialized cp_context group.")
        width = q_conv.weight.shape[-1]
        prefix_width = width - 1
        x_flat = x_packed.reshape(-1, x_packed.shape[-1])
        if x_flat.shape[0] < prefix_width:
            raise RuntimeError(
                "Exact GDN CP conv: local shard shorter than the conv window — "
                "the C1 collator contract (shard length multiple of 64) is violated",
            )
        pre_tokens = int(cp_context.pre_num_conv_tokens or 0)
        first_seq_continues = pre_tokens > 0
        if 0 < pre_tokens < prefix_width:
            raise RuntimeError(
                f"Exact GDN CP conv: crossing document has only {pre_tokens} upstream "
                f"tokens (< {prefix_width}) — impossible under the C1/C2 alignment "
                "contract; the collator is misconfigured. Fail closed.",
            )
        # Halo exchange of RAW operand rows (all ranks participate; the wire
        # never carries conv outputs).
        tails = x_flat[-prefix_width:].detach().contiguous()
        halo = conv_cp_send_recv_fwd(tails, cp_context.group)
        global _cp_engagement_logged
        if not _cp_engagement_logged:
            logger.info(
                "gdn-cp conv halo engaged: width %d, first_seq_continues=%s, dim %d",
                width, first_seq_continues, x_flat.shape[-1],
            )
            _cp_engagement_logged = True
        out = _CausalConv1dContractCP.apply(
            x_flat,
            q_conv.weight,
            k_conv.weight,
            v_conv.weight,
            boundaries,
            seq_lens,
            activation,
            halo,
            first_seq_continues,
            cp_context.group,
        ).view_as(x_packed)
        return out.split((q_input.shape[-1], k_input.shape[-1], v_input.shape[-1]), dim=-1)

    out = _CausalConv1dContract.apply(
        x_packed.reshape(-1, x_packed.shape[-1]),
        q_conv.weight,
        k_conv.weight,
        v_conv.weight,
        boundaries,
        seq_lens,
        activation,
    ).view_as(x_packed)
    return out.split((q_input.shape[-1], k_input.shape[-1], v_input.shape[-1]), dim=-1)
