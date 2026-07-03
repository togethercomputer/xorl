"""MiniMax Sparse Attention adapter.

The MiniMax MSA package targets SM100. Imports are intentionally lazy so
ordinary CPU tests and unsupported GPU hosts fail at the call site with a
clear error instead of breaking xorl model registration.
"""

from __future__ import annotations

import torch


def _sequence_lens_from_kwargs(
    query: torch.Tensor,
    key: torch.Tensor,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    batch, q_len = query.shape[:2]
    k_len = key.shape[1]
    if cu_seq_lens_q is None:
        qo_lens = torch.full((batch,), q_len, dtype=torch.int32, device=query.device)
    else:
        qo_lens = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).to(device=query.device, dtype=torch.int32)
    if cu_seq_lens_k is None:
        kv_lens = torch.full((batch,), k_len, dtype=torch.int32, device=query.device)
    else:
        kv_lens = (cu_seq_lens_k[1:] - cu_seq_lens_k[:-1]).to(device=query.device, dtype=torch.int32)
    return qo_lens, kv_lens, int(qo_lens.numel())


def _to_paged_kv(x: torch.Tensor, lengths: torch.Tensor, page_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    pages = []
    indices = []
    next_page = 0
    for batch_idx, length_tensor in enumerate(lengths.detach().cpu()):
        length = int(length_tensor.item())
        seq = x[batch_idx, :length]
        pad = (-length) % page_size
        if pad:
            seq = torch.nn.functional.pad(seq, (0, 0, 0, 0, 0, pad))
        seq_pages = seq.view(-1, page_size, x.shape[2], x.shape[3]).permute(0, 2, 1, 3).contiguous()
        pages.append(seq_pages)
        indices.extend(range(next_page, next_page + seq_pages.shape[0]))
        next_page += seq_pages.shape[0]
    return torch.cat(pages, dim=0), torch.tensor(indices, dtype=torch.int32, device=x.device)


def minimax_msa_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    *,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    scaling: float,
    topk_blocks: int,
    block_size: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    check_input_valid: bool = True,
) -> torch.Tensor:
    if query.device.type != "cuda":
        raise RuntimeError("MiniMax MSA attention requires CUDA/SM100; use attn_implementation='eager' on CPU.")
    capability = torch.cuda.get_device_capability(query.device)
    if capability[0] < 10:
        raise RuntimeError(
            "MiniMax MSA attention requires SM100-compatible hardware; "
            f"got compute capability {capability[0]}.{capability[1]}."
        )
    if query.requires_grad or key.requires_grad or value.requires_grad:
        raise RuntimeError(
            "MiniMax MSA backward is not available through fmha_sm100 in this xorl path yet; "
            "use attn_implementation='eager' for training/backward validation."
        )
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        raise RuntimeError("MiniMax MSA requires q/k/v head_dim=128.")
    qo_lens, kv_lens, batch = _sequence_lens_from_kwargs(query, key, cu_seq_lens_q, cu_seq_lens_k)
    if batch != 1:
        raise RuntimeError("MiniMax MSA adapter currently supports one packed sequence per call.")
    if topk_blocks != 16:
        raise RuntimeError(f"MiniMax MSA sparse_topk_select requires topk_blocks=16, got {topk_blocks}.")

    try:
        from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MiniMax MSA package fmha_sm100 is not importable; install /home/apanda/MSA.") from exc

    q = query.reshape(-1, query.shape[2], query.shape[3]).contiguous()
    iq = index_query.reshape(-1, index_query.shape[2], index_query.shape[3]).contiguous()
    k_pages, kv_indices = _to_paged_kv(key, kv_lens, block_size)
    v_pages, _ = _to_paged_kv(value, kv_lens, block_size)
    ik_pages, _ = _to_paged_kv(index_key, kv_lens, block_size)

    proxy_plan = fmha_sm100_plan(
        qo_lens,
        kv_lens,
        iq.shape[1],
        num_kv_heads=ik_pages.shape[1],
        page_size=block_size,
        output_maxscore=True,
        causal=True,
    )
    _, max_score = fmha_sm100(
        iq,
        ik_pages,
        ik_pages,
        proxy_plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        sm_scale=index_query.shape[-1] ** -0.5,
        check_input_valid=check_input_valid,
    )
    num_valid_pages = int((int(kv_lens[0].item()) + block_size - 1) // block_size)
    kv_block_indexes = sparse_topk_select(
        max_score.contiguous(),
        topk_blocks,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
    )

    sparse_plan = fmha_sm100_plan(
        qo_lens,
        kv_lens,
        q.shape[1],
        num_kv_heads=k_pages.shape[1],
        page_size=block_size,
        kv_block_num=topk_blocks,
        causal=True,
    )
    out, _ = fmha_sm100(
        q,
        k_pages,
        v_pages,
        sparse_plan,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        sm_scale=scaling,
        check_input_valid=check_input_valid,
    )
    return out.reshape(query.shape[0], query.shape[1], query.shape[2], value.shape[-1])


__all__ = ["minimax_msa_attention_forward"]
