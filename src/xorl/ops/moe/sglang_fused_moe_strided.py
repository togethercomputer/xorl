"""Strided-weight variant of SGLang's fused-MoE serving orchestration.

VENDORED from ``sglang.srt.layers.moe.fused_moe_triton.fused_moe.fused_experts_impl``
(merged tip 424f72d3d) — the mirror direction of the ``trainer_group_gemm`` vendor
that carried xorl kernels into sglang.

Why: xorl stores expert weights in GKN layout (``gate_up_proj [E, H, 2I]``
gate-first, ``down_proj [E, I, H]``); SGLang's serving layout ``w13 [E, 2I, H]`` /
``w2 [E, H, I]`` is exactly the ``transpose(1, 2)`` VIEW of those tensors — same
elements, swapped last-two strides. The stock ``fused_experts_impl`` asserts
``w1.is_contiguous()`` / ``w2.is_contiguous()``, forcing either a per-forward
transpose copy (~4.9 ms per q30 layer — the bulk of the parity-mode training tax)
or a serving-layout weight cache (~1.1 GiB per q30 layer, ~54 GiB model-wide).
Every downstream consumer (``invoke_fused_moe_kernel`` -> ``fused_moe_kernel``)
already takes explicit strides from the tensor, so the copy is pure tax.

This copy differs from upstream ONLY at sites marked ``XORL-EDIT``:

1. the w1/w2 contiguity asserts also accept the transpose-view of a contiguous
   GKN tensor (K-major serving shape);
2. the down-GEMM TMA descriptor path additionally requires ``w2.is_contiguous()``
   (TMA needs an innermost-contiguous tensor; the path is already dead under the
   deterministic pin xorl runs with — ``down_config`` is None — but is guarded
   so a non-pinned caller cannot silently read garbage through a descriptor).

Everything else — config selection, ``moe_align_block_size`` blocking, both
triton GEMM launches, activation, top-k combine — executes through the SAME
sglang functions imported live from the tree on PYTHONPATH, so the accumulation
tree is the serving tree by construction. Strides only change how tiles are
ADDRESSED, never which elements land in which tile or the K-loop order.
Layout behavior is covered by
``tests/models/test_moe_sglang_fused_experts.py``; release qualification also
requires the frozen-layer replay described in the public contract note.
"""

from __future__ import annotations

import functools
from typing import List, Optional

import torch


_STRIDED_FALLBACK_HINT = "set XORL_MOE_SGLANG_FUSED_EXPERTS=0 to use XoRL's stock expert forward"


@functools.lru_cache(maxsize=1)
def _fm():
    """The live sglang fused_moe module (lazy: xorl must import without sglang)."""
    try:
        from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fused_moe_module  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "XORL_MOE_SGLANG_FUSED_EXPERTS=1 could not import "
            "sglang.srt.layers.moe.fused_moe_triton.fused_moe from the sglang tree on PYTHONPATH; "
            f"fix the sglang install/PYTHONPATH or {_STRIDED_FALLBACK_HINT}"
        ) from exc
    return fused_moe_module


def serving_layout_or_gkn_view(w: torch.Tensor) -> bool:
    """True for a serving-contiguous ``[E, N, K]`` tensor or the ``transpose(1, 2)``
    view of a contiguous GKN ``[E, K, N]`` tensor (the only two layouts whose
    element order matches what the strided kernel launch addresses)."""
    return w.is_contiguous() or w.transpose(1, 2).is_contiguous()


def fused_experts_impl_strided(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
):
    """Verbatim ``fused_experts_impl`` body (upstream 424f72d3d) with the
    ``XORL-EDIT`` sites documented in the module docstring. All helpers resolve
    through the live sglang module so kernel behavior stays version-locked."""
    # XORL-EDIT(0): upstream module globals, bound locally so the body below
    # stays textually identical to upstream. Bound inside a try so an upstream
    # restructure fails loud with the mode name and the fallback, not a bare
    # AttributeError from deep inside a forward.
    fm = _fm()
    try:
        tl = fm.tl
        F = fm.F  # noqa: N806
        padding_size = fm.padding_size
        _use_aiter = fm._use_aiter
        _is_cuda = fm._is_cuda
        _is_hip = fm._is_hip
        _is_xpu = fm._is_xpu
        _has_vllm_ops = fm._has_vllm_ops
        get_config_dtype_str = fm.get_config_dtype_str
        try_get_optimal_moe_config = fm.try_get_optimal_moe_config
        _down_moe_use_tma = fm._down_moe_use_tma
        moe_align_block_size = fm.moe_align_block_size
        invoke_fused_moe_kernel = fm.invoke_fused_moe_kernel
        act_and_mul_triton = fm.act_and_mul_triton
        moe_sum_reduce_triton = fm.moe_sum_reduce_triton
        moe_sum_reduce_torch_compile = fm.moe_sum_reduce_torch_compile
        _swiglu_silu_clamp_mul = fm._swiglu_silu_clamp_mul
        _swiglu_gpt_oss_sigmoid_alpha = fm._swiglu_gpt_oss_sigmoid_alpha
        get_global_server_args = fm.get_global_server_args
    except AttributeError as exc:
        raise RuntimeError(
            "XORL_MOE_SGLANG_FUSED_EXPERTS=1 could not bind the vendored "
            f"orchestration to the sglang tree on PYTHONPATH ({exc}); the upstream fused_moe module changed — "
            f"re-vendor xorl/ops/moe/sglang_fused_moe_strided.py against it, or {_STRIDED_FALLBACK_HINT}"
        ) from exc
    silu_and_mul = getattr(fm, "silu_and_mul", None)
    gelu_and_mul = getattr(fm, "gelu_and_mul", None)
    moe_sum_reduce = getattr(fm, "moe_sum_reduce", None)
    moe_sum = getattr(fm, "moe_sum", None)
    vllm_ops = getattr(fm, "vllm_ops", None)

    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    # XORL-EDIT(1): the triton launches take explicit strides from the tensors,
    # so a GKN transpose-view is as valid as the serving-contiguous layout.
    assert serving_layout_or_gkn_view(w1), "Expert weights1 must be contiguous or a GKN transpose-view"
    assert serving_layout_or_gkn_view(w2), "Expert weights2 must be contiguous or a GKN transpose-view"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024  # noqa: N806
    M = min(num_tokens, CHUNK_SIZE)  # noqa: N806
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )

    config, (down_config, max_block_m) = get_config_func(M)
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
        # XORL-EDIT(2): TMA descriptors require innermost-contiguous weights.
        and w2.is_contiguous()
    )
    topk = topk_ids.shape[1]
    max_padded_tokens = min(M * topk, E + 1) * (max_block_m - 1) if down_moe_use_tma else 0
    total_tokens = M * topk + max_padded_tokens
    cache = torch.empty(
        total_tokens * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            config, (down_config, _) = get_config_func(tokens_in_chunk)
            down_moe_use_tma = (
                _down_moe_use_tma()
                and down_config is not None
                and down_config.pop("USE_TMA", False)
                # XORL-EDIT(2): TMA descriptors require innermost-contiguous weights.
                and w2.is_contiguous()
            )
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        padded_tokens = min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1) if down_moe_use_tma else 0
        total_tokens = tokens_in_chunk * topk + padded_tokens
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        use_fused_moe_sum_all_reduce = (
            get_global_server_args().enable_fused_moe_sum_all_reduce
            and (not no_combine)
            and (curr_topk_ids.shape[1] > 2)
            and (not use_int8_w8a16)
            and (not use_int4_w4a16)
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            b1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            c_sorted=down_moe_use_tma,
            filter_expert=filter_expert,
        )

        # Activation function with multiplication
        if activation == "silu" and is_gated:
            # - gemm1_alpha != None: GPT-OSS-style swiglu(alpha, limit)
            # - gemm1_alpha == None and gemm1_limit != None: silu+clamp+mul(limit-only)
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(intermediate_cache1.view(-1, N), gemm1_limit)
            elif _is_cuda or _is_hip or _is_xpu:
                if not filter_expert:
                    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        curr_topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
                else:
                    # Fallback: native PyTorch silu_and_mul
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
        elif activation == "gelu" and is_gated:
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                if not filter_expert:
                    gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        curr_topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.gelu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
                else:
                    # Fallback: native PyTorch gelu_and_mul
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
        # Activation function without multiplication
        elif activation == "silu" and not is_gated:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu" and not is_gated:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == "relu2" and not is_gated:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

        out_slice = None
        if use_fused_moe_sum_all_reduce:
            out_slice = out_hidden_states[begin_chunk_idx:end_chunk_idx]
            out_slice.zero_()

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                out_slice
                if use_fused_moe_sum_all_reduce
                else (
                    intermediate_cache3
                    if not no_combine and topk_ids.shape[1] != 1
                    else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
                )
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
            fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
            router_topk=curr_topk_ids.shape[1],
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if use_fused_moe_sum_all_reduce:
                if routed_scaling_factor is None:
                    routed_scaling_factor = 1.0
                if routed_scaling_factor != 1.0:
                    assert out_slice is not None
                    out_slice.mul_(routed_scaling_factor)
            elif topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if not get_global_server_args().enable_deterministic_inference and tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )

        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        elif _is_xpu:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
                routed_scaling_factor,
            )
        else:
            if _has_vllm_ops:
                vllm_ops.moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # Fallback: use triton moe_sum_reduce when vllm is not available
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )

    return out_hidden_states
