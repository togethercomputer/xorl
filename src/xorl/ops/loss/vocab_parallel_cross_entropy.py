"""
Vocab-parallel cross-entropy for tensor parallelism.

When the lm_head weight is column-sharded across TP ranks (each rank holds
vocab_size/tp_size rows), we compute cross-entropy without gathering full
logits. This saves memory proportional to tp_size.

Algorithm (per-token):
    1. local_logits = hidden_states @ local_weight.T   # [BT, V/tp]
    2. local_max = max(local_logits)
       global_max = all_reduce(local_max, MAX)         # numerical stability
    3. shifted = local_logits - global_max
    4. local_sumexp = sum(exp(shifted))
       global_sumexp = all_reduce(local_sumexp, SUM)
    5. log_normalizer = log(global_sumexp)
    6. For each token, only ONE rank holds the target's logit.
       Use masked gather + all_reduce(SUM) to get the target logit.
    7. nll = -(target_logit - log_normalizer)

Backward: d(CE)/d(logits) = softmax(logits) - one_hot(target)
          Softmax is recomputed from saved global_max/global_sumexp (tiny [BT,1]
          tensors) to avoid saving the full [BT, V/tp] softmax tensor.

Uses functional collectives (torch.distributed._functional_collectives) so
that torch.compile can trace through the all-reduce calls.
"""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


def _forward_kernel(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group,
    vocab_offset: int,
    local_vocab_size: int,
    ignore_index: int,
):
    """Forward computation kernel — extracted so torch.compile can trace it.

    funcol.all_reduce is a proper torch op, so the compiler can build a graph
    that includes the NCCL collectives and fuse the surrounding elementwise ops.
    """
    # 1. Compute local logits: [BT, V/tp]
    local_logits = (hidden_states @ weight.t()).float()

    # 2. Distributed max for numerical stability
    local_max = local_logits.max(dim=-1, keepdim=True).values
    global_max = funcol.all_reduce(local_max, reduceOp=c10d.ReduceOp.MAX.name, group=tp_group)

    # 3. Shifted logits and distributed sum-exp
    shifted = local_logits - global_max
    exp_shifted = shifted.exp()
    local_sumexp = exp_shifted.sum(dim=-1, keepdim=True)
    global_sumexp = funcol.all_reduce(local_sumexp, reduceOp=c10d.ReduceOp.SUM.name, group=tp_group)
    log_normalizer = global_sumexp.log()

    # 4. Gather target logit from the rank that owns it
    valid_mask = labels != ignore_index
    local_target = labels - vocab_offset
    target_in_range = (local_target >= 0) & (local_target < local_vocab_size) & valid_mask
    safe_local_target = local_target.clamp(0, local_vocab_size - 1)

    target_logit = shifted.gather(1, safe_local_target.unsqueeze(1)).squeeze(1)
    target_logit = target_logit * target_in_range.float()
    target_logit = funcol.all_reduce(target_logit, reduceOp=c10d.ReduceOp.SUM.name, group=tp_group)

    # 5. NLL = -(shifted_target - log_normalizer)
    per_token_ce = -(target_logit - log_normalizer.squeeze(1))
    per_token_ce = per_token_ce * valid_mask.float()

    # Return global_max and global_sumexp (tiny [BT,1]) instead of softmax_local
    # (huge [BT, V/tp]). Softmax is recomputed in backward from these + logits.
    return per_token_ce, global_max, global_sumexp, target_in_range, safe_local_target, valid_mask


def _forward_from_logits_kernel(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group,
    vocab_offset: int,
    local_vocab_size: int,
    ignore_index: int,
):
    """Forward kernel for callers that already computed local vocab logits."""

    local_logits = local_logits.float()
    local_max = local_logits.max(dim=-1, keepdim=True).values
    global_max = funcol.all_reduce(local_max, reduceOp=c10d.ReduceOp.MAX.name, group=tp_group)

    shifted = local_logits - global_max
    exp_shifted = shifted.exp()
    local_sumexp = exp_shifted.sum(dim=-1, keepdim=True)
    global_sumexp = funcol.all_reduce(local_sumexp, reduceOp=c10d.ReduceOp.SUM.name, group=tp_group)
    log_normalizer = global_sumexp.log()

    valid_mask = labels != ignore_index
    local_target = labels - vocab_offset
    target_in_range = (local_target >= 0) & (local_target < local_vocab_size) & valid_mask
    safe_local_target = local_target.clamp(0, local_vocab_size - 1)

    target_logit = shifted.gather(1, safe_local_target.unsqueeze(1)).squeeze(1)
    target_logit = target_logit * target_in_range.float()
    target_logit = funcol.all_reduce(target_logit, reduceOp=c10d.ReduceOp.SUM.name, group=tp_group)

    per_token_ce = -(target_logit - log_normalizer.squeeze(1))
    per_token_ce = per_token_ce * valid_mask.float()
    return per_token_ce, global_max, global_sumexp, target_in_range, safe_local_target, valid_mask


def _backward_kernel(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    global_max: torch.Tensor,
    global_sumexp: torch.Tensor,
    target_in_range: torch.Tensor,
    safe_local_target: torch.Tensor,
    valid_mask: torch.Tensor,
    tp_group,
    weight_requires_grad: bool,
):
    """Backward computation kernel — extracted so torch.compile can trace it.

    Recomputes softmax from hidden_states @ weight.t() + saved global_max/sumexp
    to avoid saving the full [BT, V/tp] softmax tensor in forward.
    Uses funcol.all_reduce (compile-friendly) for the grad_hidden all-reduce.
    """
    # Recompute softmax_local from saved statistics (1 extra matmul, saves ~V/tp×BT×4 bytes)
    local_logits = (hidden_states @ weight.t()).float()
    shifted = local_logits - global_max
    softmax_local = shifted.exp() / global_sumexp  # [BT, V/tp]

    # d(CE)/d(logits) = softmax - one_hot(target)
    grad_logits = softmax_local  # [BT, V/tp], no clone needed — freshly computed

    # Subtract 1 at the target position (only for tokens this rank owns)
    grad_logits.scatter_add_(
        1,
        safe_local_target.unsqueeze(1),
        -target_in_range.float().unsqueeze(1),
    )

    # Zero out ignored tokens
    grad_logits = grad_logits * valid_mask.float().unsqueeze(1)

    # Scale by upstream gradient: [BT, 1]
    grad_logits = grad_logits * grad_output.unsqueeze(1)

    # d(CE)/d(hidden_states) = sum_over_ranks(grad_logits_local @ weight_local)
    # Each rank has local shard: grad_logits [BT, V/tp] @ weight [V/tp, H] -> [BT, H]
    # funcol.all_reduce SUM to get the full gradient (compile-friendly)
    grad_hidden = grad_logits.to(weight.dtype) @ weight  # [BT, V/tp] @ [V/tp, H]
    grad_hidden = funcol.all_reduce(grad_hidden, reduceOp=c10d.ReduceOp.SUM.name, group=tp_group)

    # d(CE)/d(weight) = grad_logits.T @ hidden_states  [V/tp, H]
    grad_weight = None
    if weight_requires_grad:
        grad_weight = grad_logits.to(hidden_states.dtype).t() @ hidden_states  # [V/tp, BT] @ [BT, H]

    return grad_hidden, grad_weight


def _backward_from_logits_kernel(
    grad_output: torch.Tensor,
    local_logits: torch.Tensor,
    global_max: torch.Tensor,
    global_sumexp: torch.Tensor,
    target_in_range: torch.Tensor,
    safe_local_target: torch.Tensor,
    valid_mask: torch.Tensor,
):
    shifted = local_logits.float() - global_max
    grad_logits = shifted.exp() / global_sumexp
    grad_logits.scatter_add_(
        1,
        safe_local_target.unsqueeze(1),
        -target_in_range.float().unsqueeze(1),
    )
    grad_logits = grad_logits * valid_mask.float().unsqueeze(1)
    grad_logits = grad_logits * grad_output.unsqueeze(1)
    return grad_logits.to(local_logits.dtype)


# Lazy-init compiled kernels (one per process)
_compiled_forward_kernel = None
_compiled_backward_kernel = None
_compiled_forward_from_logits_kernel = None
_compiled_backward_from_logits_kernel = None


def _get_vocab_shard_offset(local_vocab_size: int, tp_group: dist.ProcessGroup, device: torch.device) -> int:
    """Return this rank's global vocab row offset from actual local shard sizes."""
    tp_rank = dist.get_rank(tp_group)
    tp_world_size = dist.get_world_size(tp_group)
    if tp_world_size == 1:
        return 0

    local_size = torch.tensor([local_vocab_size], device=device, dtype=torch.long)
    all_sizes = [torch.empty_like(local_size) for _ in range(tp_world_size)]
    dist.all_gather(all_sizes, local_size, group=tp_group)

    if tp_rank == 0:
        return 0
    return int(torch.cat(all_sizes[:tp_rank]).sum().item())


def _get_compiled_forward_kernel():
    global _compiled_forward_kernel
    if _compiled_forward_kernel is None:
        _compiled_forward_kernel = torch.compile(_forward_kernel)
    return _compiled_forward_kernel


def _get_compiled_backward_kernel():
    global _compiled_backward_kernel
    if _compiled_backward_kernel is None:
        _compiled_backward_kernel = torch.compile(_backward_kernel)
    return _compiled_backward_kernel


def _get_compiled_forward_from_logits_kernel():
    global _compiled_forward_from_logits_kernel
    if _compiled_forward_from_logits_kernel is None:
        _compiled_forward_from_logits_kernel = torch.compile(_forward_from_logits_kernel)
    return _compiled_forward_from_logits_kernel


def _get_compiled_backward_from_logits_kernel():
    global _compiled_backward_from_logits_kernel
    if _compiled_backward_from_logits_kernel is None:
        _compiled_backward_from_logits_kernel = torch.compile(_backward_from_logits_kernel)
    return _compiled_backward_from_logits_kernel


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Custom autograd for vocab-parallel cross-entropy.

    Forward and backward call kernel functions (eager or compiled) that use
    funcol.all_reduce (compile-friendly).

    Memory optimization: saves global_max + global_sumexp ([BT,1] each) instead
    of softmax_local ([BT, V/tp]). Softmax is recomputed in backward from
    hidden_states @ weight.t(), trading one extra matmul for massive memory savings.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        tp_group,
        vocab_offset: int,
        local_vocab_size: int,
        ignore_index: int,
        use_compile: bool = False,
    ) -> torch.Tensor:
        kernel = _get_compiled_forward_kernel() if use_compile else _forward_kernel
        per_token_ce, global_max, global_sumexp, target_in_range, safe_local_target, valid_mask = kernel(
            hidden_states,
            weight,
            labels,
            tp_group,
            vocab_offset,
            local_vocab_size,
            ignore_index,
        )

        # Save tiny [BT,1] statistics + inputs; NOT the huge [BT, V/tp] softmax
        ctx.save_for_backward(hidden_states, weight, global_max, global_sumexp)
        ctx.target_in_range = target_in_range
        ctx.safe_local_target = safe_local_target
        ctx.valid_mask = valid_mask
        ctx.tp_group = tp_group
        ctx.use_compile = use_compile

        return per_token_ce

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, weight, global_max, global_sumexp = ctx.saved_tensors

        kernel = _get_compiled_backward_kernel() if ctx.use_compile else _backward_kernel
        grad_hidden, grad_weight = kernel(
            grad_output,
            hidden_states,
            weight,
            global_max,
            global_sumexp,
            ctx.target_in_range,
            ctx.safe_local_target,
            ctx.valid_mask,
            ctx.tp_group,
            weight.requires_grad,
        )

        return grad_hidden, grad_weight, None, None, None, None, None, None


class _AllReduceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group) -> torch.Tensor:
        ctx.tp_group = tp_group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = funcol.all_reduce(grad_output, reduceOp=c10d.ReduceOp.SUM.name, group=ctx.tp_group)
        return grad_input, None


class _VocabParallelCrossEntropyFromLogits(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group,
        vocab_offset: int,
        local_vocab_size: int,
        ignore_index: int,
        use_compile: bool = False,
    ) -> torch.Tensor:
        kernel = _get_compiled_forward_from_logits_kernel() if use_compile else _forward_from_logits_kernel
        per_token_ce, global_max, global_sumexp, target_in_range, safe_local_target, valid_mask = kernel(
            local_logits,
            labels,
            tp_group,
            vocab_offset,
            local_vocab_size,
            ignore_index,
        )

        ctx.save_for_backward(local_logits, global_max, global_sumexp)
        ctx.target_in_range = target_in_range
        ctx.safe_local_target = safe_local_target
        ctx.valid_mask = valid_mask
        ctx.use_compile = use_compile
        return per_token_ce

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_logits, global_max, global_sumexp = ctx.saved_tensors
        kernel = _get_compiled_backward_from_logits_kernel() if ctx.use_compile else _backward_from_logits_kernel
        grad_logits = kernel(
            grad_output,
            local_logits,
            global_max,
            global_sumexp,
            ctx.target_in_range,
            ctx.safe_local_target,
            ctx.valid_mask,
        )
        return grad_logits, None, None, None, None, None, None


def _to_local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _call_lm_head_local(lm_head: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if hasattr(lm_head, "forward_with_weight") and hasattr(lm_head, "weight"):
        local_weight = _to_local_tensor(lm_head.weight)
        local_bias = _to_local_tensor(getattr(lm_head, "bias", None))
        return lm_head.forward_with_weight(hidden_states, local_weight, local_bias)
    logits = lm_head(hidden_states)
    return logits.to_local() if hasattr(logits, "to_local") else logits


def _vocab_parallel_cross_entropy_from_logits_chunk(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    vocab_offset: int,
    local_vocab_size: int,
    ignore_index: int,
    use_compile: bool,
) -> torch.Tensor:
    return _VocabParallelCrossEntropyFromLogits.apply(
        local_logits,
        labels,
        tp_group,
        vocab_offset,
        local_vocab_size,
        ignore_index,
        use_compile,
    )


def vocab_parallel_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    ignore_index: int = -100,
    num_chunks: int = 8,
    use_compile: bool = False,
) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss with vocab-parallel logits.

    Args:
        hidden_states: [BT, H] flattened hidden states (replicated across TP ranks)
        weight: [V/tp, H] local shard of lm_head weight (each rank has different rows)
        labels: [BT] target token indices (replicated across TP ranks)
        tp_group: TP process group for all-reduce communication
        ignore_index: label value to ignore (default: -100)
        num_chunks: Number of chunks to split BT into (default: 8)
        use_compile: If True, torch.compile the forward/backward kernels (default: False)

    Returns:
        per_token_ce: [BT] per-token cross-entropy loss (replicated across TP ranks)
    """
    local_vocab_size = weight.shape[0]
    vocab_offset = _get_vocab_shard_offset(local_vocab_size, tp_group, weight.device)

    # Chunked execution to reduce peak memory (logits are [chunk, V/tp])
    BT = hidden_states.shape[0]
    chunk_size = (BT + num_chunks - 1) // num_chunks
    ce_chunks = []

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        ce_chunks.append(
            _VocabParallelCrossEntropy.apply(
                hidden_states[start:end],
                weight,
                labels[start:end],
                tp_group,
                vocab_offset,
                local_vocab_size,
                ignore_index,
                use_compile,
            )
        )

    return torch.cat(ce_chunks, dim=0)


def vocab_parallel_cross_entropy_with_lm_head(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    ignore_index: int = -100,
    num_chunks: int = 8,
    use_compile: bool = False,
) -> torch.Tensor:
    """Vocab-parallel CE that obtains local logits by calling ``lm_head``.

    This keeps FP8 ``lm_head`` training on the module path while preserving the
    raw-weight vocab-parallel CE contract: the hidden-state gradient is summed
    across TP ranks after the local head backward contribution is computed.
    """

    BT = hidden_states.shape[0]
    if BT == 0:
        return hidden_states.new_empty((0,), dtype=torch.float32)

    chunk_count = max(1, int(num_chunks))
    chunk_size = (BT + chunk_count - 1) // chunk_count
    ce_chunks = []
    vocab_offset = None
    local_vocab_size = None

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        hidden_chunk = _AllReduceGrad.apply(hidden_states[start:end], tp_group)
        local_logits = _call_lm_head_local(lm_head, hidden_chunk)
        local_logits = local_logits.to_local() if hasattr(local_logits, "to_local") else local_logits
        if local_vocab_size is None:
            local_vocab_size = local_logits.shape[-1]
            vocab_offset = _get_vocab_shard_offset(local_vocab_size, tp_group, local_logits.device)
        ce_chunks.append(
            _vocab_parallel_cross_entropy_from_logits_chunk(
                local_logits,
                labels[start:end],
                tp_group,
                int(vocab_offset),
                int(local_vocab_size),
                ignore_index,
                use_compile,
            )
        )

    return torch.cat(ce_chunks, dim=0)
