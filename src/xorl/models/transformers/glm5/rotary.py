"""GLM-5 rotary embedding helpers."""

from __future__ import annotations

import torch


def glm5_rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def glm5_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if interleaved:
        # RotaryEmbedding emits halved layout [c0..cN, c0..cN], while
        # interleaved rotate_half pairs dimensions as (0, 1), (2, 3), ...
        # Expand the angles to [c0, c0, c1, c1, ...] before applying RoPE.
        half = cos.shape[-1] // 2
        cos = cos[..., :half].repeat_interleave(2, dim=-1)
        sin = sin[..., :half].repeat_interleave(2, dim=-1)

    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (glm5_rotate_half(q_rot, interleaved=interleaved) * sin)
    k_embed = (k_rot * cos) + (glm5_rotate_half(k_rot, interleaved=interleaved) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


__all__ = ["glm5_apply_rotary_pos_emb", "glm5_rotate_half"]
