import pytest
import torch

from xorl.models.layers.rope import rotate_half
from xorl.models.transformers.qwen3_5_shared import qwen3_5_apply_rotary_pos_emb


pytestmark = pytest.mark.cpu


def _build_halved_cos_sin(batch: int, seq: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mimic `RotaryEmbedding.forward`: halved layout [c0..c_{d/2-1}, c0..c_{d/2-1}]."""
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().expand(batch, -1, -1).contiguous()
    sin = emb.sin().expand(batch, -1, -1).contiguous()
    return cos, sin


def _hf_reference_interleaved(
    q: torch.Tensor, k: torch.Tensor, cos_halved: torch.Tensor, sin_halved: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-style reference for interleaved-layout q/k with halved cos/sin.

    Reshape interleaved q/k to halved, apply standard non-interleaved rotation,
    then reshape back. Equivalent to `qwen3_5_apply_rotary_pos_emb(interleaved=True)`.
    """

    def to_halved(x: torch.Tensor) -> torch.Tensor:
        b, s, h, d = x.shape
        return x.view(b, s, h, d // 2, 2).transpose(-1, -2).reshape(b, s, h, d)

    def to_interleaved(x: torch.Tensor) -> torch.Tensor:
        b, s, h, d = x.shape
        return x.view(b, s, h, 2, d // 2).transpose(-1, -2).reshape(b, s, h, d)

    q_h = to_halved(q)
    k_h = to_halved(k)
    cos = cos_halved.unsqueeze(2)
    sin = sin_halved.unsqueeze(2)
    q_embed_h = q_h * cos + rotate_half(q_h) * sin
    k_embed_h = k_h * cos + rotate_half(k_h) * sin
    return to_interleaved(q_embed_h), to_interleaved(k_embed_h)


def test_interleaved_matches_hf_reference():
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 2, 5, 3, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_ours, k_ours = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=True)
    q_ref, k_ref = _hf_reference_interleaved(q, k, cos, sin)

    torch.testing.assert_close(q_ours, q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_ours, k_ref, atol=1e-6, rtol=1e-6)


def test_interleaved_pairwise_rotation_d8():
    """Hand-worked d=8 sanity: pair i must be rotated by angle θ_i, not θ_{i+1}."""
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 1, 1, 1, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.zeros_like(q)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_out, _ = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=True)
    # cos_unique[t=0] = [1,1,1,1] and sin_unique[t=0] = [0,0,0,0], so rotation
    # at position 0 is the identity.
    torch.testing.assert_close(q_out, q, atol=1e-6, rtol=1e-6)


def test_non_interleaved_unchanged():
    """Non-interleaved path must be unaffected by the fix."""
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 2, 4, 3, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_out, k_out = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=False)

    cos_u = cos.unsqueeze(2)
    sin_u = sin.unsqueeze(2)
    expected_q = q * cos_u + rotate_half(q) * sin_u
    expected_k = k * cos_u + rotate_half(k) * sin_u
    torch.testing.assert_close(q_out, expected_q, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_out, expected_k, atol=1e-6, rtol=1e-6)
