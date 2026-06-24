"""Tests for MoE load-balancing loss: micro-batch and global-batch (buffer) scopes."""

import pytest
import torch
import torch.nn.functional as F

from xorl.models.layers.moe.aux_loss import LoadBalancingBuffer, global_load_balancing_loss_func


pytestmark = [pytest.mark.cpu]


def _logits(num_tokens: int, num_experts: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, num_experts, generator=gen)


def _reference_frequency(logits: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Full-batch ``f_i`` = mean over all tokens of the one-hot top-k selection."""
    routing_weights = F.softmax(logits, dim=-1)
    _, selected = torch.topk(routing_weights, top_k, dim=-1)
    return F.one_hot(selected, num_experts).float().mean(dim=0)


def test_empty_inputs_return_zero():
    assert global_load_balancing_loss_func(None, num_experts=8, top_k=2) == 0
    assert global_load_balancing_loss_func((), num_experts=8, top_k=2) == 0
    assert global_load_balancing_loss_func((None,), num_experts=8, top_k=2) == 0


def test_single_microbatch_buffer_matches_micro_batch():
    """With a single micro-batch and no DP, global-batch == micro-batch."""
    logits = _logits(32, 8, seed=0)
    micro = global_load_balancing_loss_func((logits,), num_experts=8, top_k=2)
    glob = global_load_balancing_loss_func((logits,), num_experts=8, top_k=2, buffer=LoadBalancingBuffer())
    assert torch.allclose(micro, glob)


def test_buffer_accumulates_global_frequency():
    """After the GA window, ``f_i`` from the buffer equals the full-batch frequency."""
    num_experts, top_k = 8, 2
    full = _logits(64, num_experts, seed=1)
    chunks = [full[:16], full[16:40], full[40:]]

    ref_f = _reference_frequency(full, num_experts, top_k)

    buffer = LoadBalancingBuffer()
    f_running = None
    for chunk in chunks:
        global_load_balancing_loss_func((chunk,), num_experts=num_experts, top_k=top_k, buffer=buffer)
        f_running = buffer.counts / buffer.denom

    assert torch.allclose(f_running, ref_f, atol=1e-6)


def test_buffer_running_frequency_tracks_prefix():
    """Mid-window ``f_i`` reflects only the micro-batches accumulated so far."""
    num_experts, top_k = 4, 1
    full = _logits(40, num_experts, seed=2)
    first = full[:10]

    buffer = LoadBalancingBuffer()
    global_load_balancing_loss_func((first,), num_experts=num_experts, top_k=top_k, buffer=buffer)
    # After only the first chunk, the buffer frequency equals that chunk's frequency.
    assert torch.allclose(buffer.counts / buffer.denom, _reference_frequency(first, num_experts, top_k), atol=1e-6)


def test_buffer_loss_is_differentiable_via_router_prob():
    """Gradients flow through P_i (router prob); f_i from the buffer is detached."""
    logits = _logits(32, 8, seed=3).requires_grad_(True)
    loss = global_load_balancing_loss_func((logits,), num_experts=8, top_k=2, buffer=LoadBalancingBuffer())
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    # The buffer must not retain an autograd graph across micro-batches.
    buffer = LoadBalancingBuffer()
    global_load_balancing_loss_func((logits.detach(),), num_experts=8, top_k=2, buffer=buffer)
    assert not buffer.counts.requires_grad


def test_global_differs_from_summed_micro_for_skewed_chunks():
    """When chunks are skewed, global-batch f_i differs from per-chunk f_i."""
    num_experts, top_k = 4, 1
    # Chunk A routes everything to expert 0; chunk B to expert 1.
    chunk_a = torch.tensor([[10.0, 0.0, 0.0, 0.0]] * 8)
    chunk_b = torch.tensor([[0.0, 10.0, 0.0, 0.0]] * 8)

    # Per-micro-batch f_i is a one-hot (fully imbalanced within each chunk).
    f_micro_a = _reference_frequency(chunk_a, num_experts, top_k)
    assert torch.allclose(f_micro_a, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Global f_i over both chunks is balanced across experts 0 and 1.
    buffer = LoadBalancingBuffer()
    for chunk in (chunk_a, chunk_b):
        global_load_balancing_loss_func((chunk,), num_experts=num_experts, top_k=top_k, buffer=buffer)
    f_global = buffer.counts / buffer.denom
    assert torch.allclose(f_global, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))


def test_masked_single_microbatch_buffer_matches_micro_batch():
    """Buffer path with an attention mask matches micro-batch for a single micro-batch."""
    num_experts, top_k = 4, 2
    batch_size, seq_len = 2, 8
    logits = _logits(batch_size * seq_len, num_experts, seed=4)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 5:] = 0  # pad some positions

    micro = global_load_balancing_loss_func(
        (logits,), num_experts=num_experts, top_k=top_k, attention_mask=attention_mask
    )
    glob = global_load_balancing_loss_func(
        (logits,), num_experts=num_experts, top_k=top_k, attention_mask=attention_mask, buffer=LoadBalancingBuffer()
    )
    assert torch.allclose(micro, glob)
