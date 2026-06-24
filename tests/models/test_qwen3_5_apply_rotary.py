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


def _hf_reference_standard(
    q: torch.Tensor, k: torch.Tensor, cos_halved: torch.Tensor, sin_halved: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF/SGLang Qwen3.5 reference: standard half-rotate on q/k features."""

    cos = cos_halved.unsqueeze(2)
    sin = sin_halved.unsqueeze(2)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def test_mrope_interleaved_uses_hf_standard_half_rotate():
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 2, 5, 3, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_ours, k_ours = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=True)
    q_ref, k_ref = _hf_reference_standard(q, k, cos, sin)

    torch.testing.assert_close(q_ours, q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_ours, k_ref, atol=1e-6, rtol=1e-6)


def test_mrope_interleaved_flag_does_not_pairwise_rotate_features():
    """The flag affects MRoPE cos/sin construction, not the q/k feature layout."""
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 1, 2, 1, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn_like(q)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_interleaved, k_interleaved = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=True)
    q_standard, k_standard = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=False)

    torch.testing.assert_close(q_interleaved, q_standard, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_interleaved, k_standard, atol=1e-6, rtol=1e-6)


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
