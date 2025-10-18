"""Shared attention utilities."""

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand key/value heads from (batch, num_kv_heads, seqlen, head_dim)
    to (batch, num_attn_heads, seqlen, head_dim).

    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
