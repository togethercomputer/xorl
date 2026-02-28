"""Native PyTorch MoE expert backend — ``torch._grouped_mm`` (cuBLAS/CUTLASS).

Aligned with torchtitan's implementation:

- Inner grouped GEMM function compiled with ``torch.compile(fullgraph=True)``
- Token permutation uses vectorized torch ops (no Python for-loops)

Uses ``torch._grouped_mm`` which dispatches to cuBLAS/CUTLASS grouped GEMM
kernels — no custom Triton code required.

The public entry point (``native_expert_forward``) accepts the same
``(hidden_states, routing_weights, selected_experts, ...)`` interface as
the Triton/quack backends. Token reordering, expert grouping, and
alignment padding are handled internally.
"""

import torch
import torch.nn.functional as F

# Alignment for token groups: 8 for bf16/fp16.
_TOKEN_GROUP_ALIGN = 8

# Needed for torch.compile to handle dynamic shapes in grouped_mm offsets
# (same setting used by torchtitan for token-choice MoE).
torch._dynamo.config.capture_scalar_outputs = True

# Allow enough compiled variants for different configs × grad modes.
# In production the model config is fixed so only 2-3 variants are needed
# (grad enabled, grad disabled, maybe different batch sizes).
torch._dynamo.config.cache_size_limit = max(
    torch._dynamo.config.cache_size_limit, 64
)


# ---------------------------------------------------------------------------
# Padding helpers (vectorized — no Python for-loops, no .tolist())
# ---------------------------------------------------------------------------

def _compute_pad_indices(
    counts: torch.Tensor,
    padded_counts: torch.Tensor,
    num_experts: int,
    total_sorted: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute index mapping from sorted layout to padded layout.

    For each position ``i`` in the sorted token array, returns the
    corresponding destination index in the padded array. Uses
    ``torch.searchsorted`` to avoid Python loops and host-device sync.

    Args:
        counts: Real token counts per expert, ``(num_experts,)`` int32.
        padded_counts: Aligned token counts per expert, ``(num_experts,)`` int32.
        num_experts: Number of experts.
        total_sorted: Total number of sorted tokens (``N * top_k``).
        device: Target device.

    Returns:
        ``pad_dst`` tensor of shape ``(total_sorted,)`` int64 — index into
        the padded array for each sorted position.
    """
    sorted_offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.int64)
    sorted_offsets[1:] = counts.cumsum(0).to(torch.int64)

    padded_offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.int64)
    padded_offsets[1:] = padded_counts.cumsum(0).to(torch.int64)

    arange = torch.arange(total_sorted, device=device)
    expert_of_pos = torch.searchsorted(sorted_offsets[1:], arange, right=True)
    local_pos = arange - sorted_offsets[expert_of_pos]

    return (padded_offsets[expert_of_pos] + local_pos).to(torch.int64)


def _pad_to_alignment(
    x_sorted: torch.Tensor,
    counts: torch.Tensor,
    num_experts: int,
) -> tuple:
    """Pad each expert's token group to alignment boundary.

    Args:
        x_sorted: Tokens sorted by expert, ``(total_tokens, dim)``.
        counts: Tokens per expert, ``(num_experts,)`` int.
        num_experts: Number of experts.

    Returns:
        ``(x_padded, padded_counts)`` where ``padded_counts`` are aligned.
    """
    padded_counts = torch.clamp_min(counts, _TOKEN_GROUP_ALIGN)
    padded_counts = (
        (padded_counts + _TOKEN_GROUP_ALIGN - 1) // _TOKEN_GROUP_ALIGN * _TOKEN_GROUP_ALIGN
    ).to(torch.int32)

    total_sorted = x_sorted.shape[0]
    total_padded = int(padded_counts.sum().item())
    dim = x_sorted.shape[-1]

    pad_dst = _compute_pad_indices(counts, padded_counts, num_experts, total_sorted, x_sorted.device)

    x_padded = x_sorted.new_zeros(total_padded, dim)
    x_padded[pad_dst] = x_sorted

    return x_padded, padded_counts


def _unpad(
    out_padded: torch.Tensor,
    counts: torch.Tensor,
    padded_counts: torch.Tensor,
    num_experts: int,
    total_real: int,
) -> torch.Tensor:
    """Strip padding from expert output back to real token count."""
    pad_dst = _compute_pad_indices(counts, padded_counts, num_experts, total_real, out_padded.device)
    return out_padded[pad_dst]


# ---------------------------------------------------------------------------
# Core grouped-mm expert forward (compiled like torchtitan)
# ---------------------------------------------------------------------------

def _run_experts_grouped_mm(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    x: torch.Tensor,
    padded_counts: torch.Tensor,
) -> torch.Tensor:
    """Run MoE experts using ``torch._grouped_mm``.

    Compiled with ``torch.compile(fullgraph=True)`` for operator fusion.

    Weight shapes in (G, K, N) format::

        gate_proj: [num_experts, hidden_dim, intermediate_size]
        up_proj:   [num_experts, hidden_dim, intermediate_size]
        down_proj: [num_experts, intermediate_size, hidden_dim]
    """
    offsets = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    compute_dtype = torch.bfloat16

    # gate: x @ gate_proj -> (tokens, intermediate)
    gate_out = F.silu(
        torch._grouped_mm(
            x.to(compute_dtype),
            gate_proj.to(compute_dtype),
            offs=offsets,
        )
    )

    # up: x @ up_proj -> (tokens, intermediate)
    up_out = torch._grouped_mm(
        x.to(compute_dtype),
        up_proj.to(compute_dtype),
        offs=offsets,
    )

    # SwiGLU: silu(gate) * up
    h = gate_out * up_out

    # down: h @ down_proj -> (tokens, hidden)
    out = torch._grouped_mm(
        h,
        down_proj.to(compute_dtype),
        offs=offsets,
    ).to(x.dtype)

    return out


# Compile the inner GEMM function (like torchtitan).
_run_experts_compiled = torch.compile(_run_experts_grouped_mm, fullgraph=True)


# ---------------------------------------------------------------------------
# Public entry point (same interface as triton/quack backends)
# ---------------------------------------------------------------------------

def native_expert_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    **kwargs,
) -> torch.Tensor:
    """Forward pass using native PyTorch ``torch._grouped_mm``.

    Accepts the same interface as the Triton/quack backends — token
    reordering and expert grouping are handled internally.

    Args:
        hidden_states: Input tensor ``(num_tokens, hidden_dim)``.
        routing_weights: Routing weights ``(num_tokens, top_k)``.
        selected_experts: Selected expert indices ``(num_tokens, top_k)``.
        gate_proj: Gate projection weights ``[num_experts, hidden, intermediate]``.
        up_proj: Up projection weights ``[num_experts, hidden, intermediate]``.
        down_proj: Down projection weights ``[num_experts, intermediate, hidden]``.
        num_experts: Total number of experts.
        **kwargs: Extra arguments (ignored).

    Returns:
        Output tensor ``(num_tokens, hidden_dim)``.
    """
    num_tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    device = hidden_states.device

    # 1. Flatten top-k assignments
    flat_experts = selected_experts.view(-1)          # (N * top_k,)
    flat_weights = routing_weights.view(-1)            # (N * top_k,)

    # 2. Sort by expert (stable preserves token order within each expert)
    sorted_order = torch.argsort(flat_experts, stable=True)
    token_ids = sorted_order // top_k                  # original token index

    sorted_hidden = hidden_states[token_ids]           # (N * top_k, dim)
    sorted_weights = flat_weights[sorted_order]        # (N * top_k,)

    # 3. Histogram — how many tokens per expert
    num_tokens_per_expert = torch.histc(
        flat_experts.float(), bins=num_experts, min=0, max=num_experts - 1,
    ).to(torch.int32)

    # 4. Compute padded counts for grouped_mm alignment
    padded_counts = torch.clamp_min(num_tokens_per_expert, _TOKEN_GROUP_ALIGN)
    padded_counts = (
        (padded_counts + _TOKEN_GROUP_ALIGN - 1) // _TOKEN_GROUP_ALIGN * _TOKEN_GROUP_ALIGN
    ).to(torch.int32)

    # 5. Compute pad index mapping (once — reused for pad + unpad)
    total_sorted = num_tokens * top_k
    pad_dst = _compute_pad_indices(
        num_tokens_per_expert, padded_counts, num_experts, total_sorted, device
    )

    # 6. Scatter sorted tokens into padded layout
    total_padded = int(padded_counts.sum().item())
    sorted_hidden_padded = sorted_hidden.new_zeros(total_padded, hidden_dim)
    sorted_hidden_padded[pad_dst] = sorted_hidden

    # 7. Compiled grouped GEMM
    expert_out_padded = _run_experts_compiled(
        gate_proj, up_proj, down_proj, sorted_hidden_padded, padded_counts
    )

    # 8. Gather from padded layout (reuse pad_dst) + apply routing weights
    expert_out = expert_out_padded[pad_dst] * sorted_weights.unsqueeze(-1)

    # 9. Scatter-add back to original token positions
    output = hidden_states.new_zeros(num_tokens, hidden_dim)
    output.index_add_(0, token_ids, expert_out)

    return output


# ---------------------------------------------------------------------------
# Shared LoRA helpers (extracted from MoEExpertsLoRA)
# ---------------------------------------------------------------------------

_RANK_ALIGN = 8  # bf16: 8 * 2 bytes = 16 bytes


def expand_shared_lora_weight(tensor: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Expand shared (shape[0]==1) LoRA weight to per-expert for grouped_mm."""
    if tensor.shape[0] == 1 and num_experts > 1:
        return tensor.expand(num_experts, -1, -1)
    return tensor


def align_lora_rank_for_grouped_mm(
    lora_A: torch.Tensor, lora_B: torch.Tensor
) -> tuple:
    """Pad LoRA rank dimension so ``torch._grouped_mm`` stride is 16-byte aligned.

    With (G, K, N) format:
        lora_A: ``[E, in_features, r]``
        lora_B: ``[E, r, out_features]``

    For bf16 the inner dim of the second matmul (r) must be a multiple of 8.
    """
    r = lora_A.shape[2]
    if r % _RANK_ALIGN == 0:
        return lora_A, lora_B
    pad = (r + _RANK_ALIGN - 1) // _RANK_ALIGN * _RANK_ALIGN - r
    lora_A = F.pad(lora_A, (0, pad))
    lora_B = F.pad(lora_B, (0, 0, 0, pad))
    return lora_A, lora_B


def _cumsum_to_counts(cumsum: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Convert cumulative sum to per-expert counts."""
    counts = torch.zeros(num_experts, device=cumsum.device, dtype=torch.int32)
    cumsum_i32 = cumsum.to(torch.int32)
    counts[0] = cumsum_i32[0]
    counts[1:] = cumsum_i32[1:] - cumsum_i32[:-1]
    return counts


# ---------------------------------------------------------------------------
# Core grouped-mm with LoRA (NOT compiled — LoRA expand/align not compile-friendly)
# ---------------------------------------------------------------------------

def _run_grouped_mm_with_lora(
    x: torch.Tensor,
    padded_counts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_proj_lora_A: torch.Tensor,
    gate_proj_lora_B: torch.Tensor,
    up_proj_lora_A: torch.Tensor,
    up_proj_lora_B: torch.Tensor,
    down_proj_lora_A: torch.Tensor,
    down_proj_lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Run MoE experts with LoRA using ``torch._grouped_mm``.

    All weights in (G, K, N) format.
    """
    offsets = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    compute_dtype = torch.bfloat16
    x_bf16 = x.to(compute_dtype)
    num_experts = gate_proj.shape[0]

    # --- Gate projection with LoRA ---
    gate_out = torch._grouped_mm(x_bf16, gate_proj.to(compute_dtype), offs=offsets)
    gate_lora_A = expand_shared_lora_weight(gate_proj_lora_A, num_experts).to(compute_dtype)
    gate_lora_B = gate_proj_lora_B.to(compute_dtype)
    gate_lora_A, gate_lora_B = align_lora_rank_for_grouped_mm(gate_lora_A, gate_lora_B)
    gate_lora_out = torch._grouped_mm(
        torch._grouped_mm(x_bf16, gate_lora_A, offs=offsets),
        gate_lora_B, offs=offsets,
    )
    gate_out = F.silu(gate_out + gate_lora_out * scaling)

    # --- Up projection with LoRA ---
    up_out = torch._grouped_mm(x_bf16, up_proj.to(compute_dtype), offs=offsets)
    up_lora_A = expand_shared_lora_weight(up_proj_lora_A, num_experts).to(compute_dtype)
    up_lora_B = up_proj_lora_B.to(compute_dtype)
    up_lora_A, up_lora_B = align_lora_rank_for_grouped_mm(up_lora_A, up_lora_B)
    up_lora_out = torch._grouped_mm(
        torch._grouped_mm(x_bf16, up_lora_A, offs=offsets),
        up_lora_B, offs=offsets,
    )
    up_out = up_out + up_lora_out * scaling

    # SwiGLU
    h = gate_out * up_out

    # --- Down projection with LoRA ---
    out = torch._grouped_mm(h, down_proj.to(compute_dtype), offs=offsets)
    down_lora_A = down_proj_lora_A.to(compute_dtype)
    down_lora_B = expand_shared_lora_weight(down_proj_lora_B, num_experts).to(compute_dtype)
    down_lora_A, down_lora_B = align_lora_rank_for_grouped_mm(down_lora_A, down_lora_B)
    down_lora_out = torch._grouped_mm(
        torch._grouped_mm(h, down_lora_A, offs=offsets),
        down_lora_B, offs=offsets,
    )
    out = out + down_lora_out * scaling

    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Public entry point: local native LoRA forward
# ---------------------------------------------------------------------------

def native_expert_lora_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    gate_proj_lora_A: torch.Tensor,
    gate_proj_lora_B: torch.Tensor,
    up_proj_lora_A: torch.Tensor,
    up_proj_lora_B: torch.Tensor,
    down_proj_lora_A: torch.Tensor,
    down_proj_lora_B: torch.Tensor,
    scaling: float,
    **kwargs,
) -> torch.Tensor:
    """Native LoRA forward using ``torch._grouped_mm`` (local single-GPU).

    Same token reordering logic as ``native_expert_forward`` but routes
    through the LoRA-aware grouped GEMM path.
    """
    num_tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    device = hidden_states.device

    # 1. Flatten top-k assignments
    flat_experts = selected_experts.view(-1)
    flat_weights = routing_weights.view(-1)

    # 2. Sort by expert
    sorted_order = torch.argsort(flat_experts, stable=True)
    token_ids = sorted_order // top_k
    sorted_hidden = hidden_states[token_ids]
    sorted_weights = flat_weights[sorted_order]

    # 3. Histogram
    num_tokens_per_expert = torch.histc(
        flat_experts.float(), bins=num_experts, min=0, max=num_experts - 1,
    ).to(torch.int32)

    # 4. Pad for grouped_mm alignment
    sorted_hidden_padded, padded_counts = _pad_to_alignment(
        sorted_hidden, num_tokens_per_expert, num_experts
    )

    # 5. Grouped GEMM with LoRA
    expert_out_padded = _run_grouped_mm_with_lora(
        sorted_hidden_padded, padded_counts,
        gate_proj, up_proj, down_proj,
        gate_proj_lora_A, gate_proj_lora_B,
        up_proj_lora_A, up_proj_lora_B,
        down_proj_lora_A, down_proj_lora_B,
        scaling,
    )

    # 6. Unpad + apply routing weights
    total_sorted = num_tokens * top_k
    expert_out = _unpad(
        expert_out_padded, num_tokens_per_expert, padded_counts, num_experts, total_sorted
    )
    expert_out = expert_out * sorted_weights.unsqueeze(-1)

    # 7. Scatter-add back
    output = hidden_states.new_zeros(num_tokens, hidden_dim)
    output.index_add_(0, token_ids, expert_out)

    return output


# ---------------------------------------------------------------------------
# EP compute function (same interface as EPGroupGemm.apply / QuackEPGroupGemm.apply)
# ---------------------------------------------------------------------------

def native_ep_compute(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """EP expert compute using ``torch._grouped_mm``.

    Same interface as ``EPGroupGemm.apply()`` and ``QuackEPGroupGemm.apply()``.
    Tokens have already been dispatched via all-to-all; this only handles
    the expert MLP computation.

    Args:
        permute_tokens: Dispatched tokens ``[total_local_tokens, hidden_dim]``.
        cumsum: Cumulative sum of tokens per local expert ``[num_local_experts]``.
        gate_proj: ``[num_local_experts, hidden_dim, intermediate_size]``.
        up_proj: ``[num_local_experts, hidden_dim, intermediate_size]``.
        down_proj: ``[num_local_experts, intermediate_size, hidden_dim]``.

    Returns:
        Expert outputs ``[total_local_tokens, hidden_dim]``.
    """
    if permute_tokens.shape[0] == 0:
        return permute_tokens

    num_local_experts = gate_proj.shape[0]
    counts = _cumsum_to_counts(cumsum, num_local_experts)

    # Pad for alignment and run compiled grouped GEMM
    padded_tokens, padded_counts = _pad_to_alignment(permute_tokens, counts, num_local_experts)
    out_padded = _run_experts_compiled(gate_proj, up_proj, down_proj, padded_tokens, padded_counts)

    # Unpad back to real token counts
    return _unpad(out_padded, counts, padded_counts, num_local_experts, permute_tokens.shape[0])


# ---------------------------------------------------------------------------
# EP compute with LoRA (same interface as EPGroupGemmWithLoRA.apply)
# ---------------------------------------------------------------------------

def native_ep_compute_lora(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_proj_lora_A: torch.Tensor,
    gate_proj_lora_B: torch.Tensor,
    up_proj_lora_A: torch.Tensor,
    up_proj_lora_B: torch.Tensor,
    down_proj_lora_A: torch.Tensor,
    down_proj_lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """EP expert compute with LoRA using ``torch._grouped_mm``.

    Same interface as ``EPGroupGemmWithLoRA.apply()``.
    """
    if permute_tokens.shape[0] == 0:
        return permute_tokens

    num_local_experts = gate_proj.shape[0]
    counts = _cumsum_to_counts(cumsum, num_local_experts)

    padded_tokens, padded_counts = _pad_to_alignment(permute_tokens, counts, num_local_experts)
    out_padded = _run_grouped_mm_with_lora(
        padded_tokens, padded_counts,
        gate_proj, up_proj, down_proj,
        gate_proj_lora_A, gate_proj_lora_B,
        up_proj_lora_A, up_proj_lora_B,
        down_proj_lora_A, down_proj_lora_B,
        scaling,
    )

    return _unpad(out_padded, counts, padded_counts, num_local_experts, permute_tokens.shape[0])
