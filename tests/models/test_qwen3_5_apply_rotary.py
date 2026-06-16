import ast
import inspect

import pytest
import torch

from xorl.models.layers.rope import rotate_half
from xorl.models.transformers.qwen3_5 import modeling_qwen3_5
from xorl.models.transformers.qwen3_5_moe import modeling_qwen3_5_moe
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


def _hf_reference_half_rotate(
    q: torch.Tensor, k: torch.Tensor, cos_halved: torch.Tensor, sin_halved: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF/SGLang Qwen3.5 reference: standard half-rotate on q/k features."""
    cos = cos_halved.unsqueeze(2)
    sin = sin_halved.unsqueeze(2)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def _hf_reference_pairwise(
    q: torch.Tensor, k: torch.Tensor, cos_halved: torch.Tensor, sin_halved: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise-interleaved reference (DSv3 MLA decoupled-RoPE convention).

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


def test_default_matches_hf_half_rotate():
    """Default (interleaved=False) is the Qwen3.5/3.6 + HF/SGLang convention."""
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 2, 4, 3, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_ours, k_ours = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin)
    q_ref, k_ref = _hf_reference_half_rotate(q, k, cos, sin)

    torch.testing.assert_close(q_ours, q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_ours, k_ref, atol=1e-6, rtol=1e-6)


def test_interleaved_matches_pairwise_reference():
    """interleaved=True is the DSv3 MLA decoupled-RoPE pairwise convention."""
    torch.manual_seed(0)
    batch, seq, num_heads, head_dim = 2, 5, 3, 8
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32)
    cos, sin = _build_halved_cos_sin(batch, seq, head_dim)

    q_ours, k_ours = qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, interleaved=True)
    q_ref, k_ref = _hf_reference_pairwise(q, k, cos, sin)

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


@pytest.mark.parametrize("modeling_module", [modeling_qwen3_5, modeling_qwen3_5_moe])
def test_qwen35_modeling_does_not_pass_interleaved_to_rotary(modeling_module):
    """Regression: Qwen3.5/3.6 attention must NOT pass `interleaved=` into
    `qwen3_5_apply_rotary_pos_emb`. q/k features use the standard half-rotate
    convention (HF/SGLang); `mrope_interleaved` controls T/H/W frequency mixing
    in cos/sin construction upstream, not the q/k rotation convention. Plumbing
    it in would silently switch q/k to pairwise rotation for any HF Qwen3.5/3.6
    config that ships with `mrope_interleaved=true`.
    """
    tree = ast.parse(inspect.getsource(modeling_module))
    offenders: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "qwen3_5_apply_rotary_pos_emb"
            and any(kw.arg == "interleaved" for kw in node.keywords)
        ):
            offenders.append(node.lineno)
    assert not offenders, (
        f"{modeling_module.__name__} calls `qwen3_5_apply_rotary_pos_emb(..., interleaved=...)` "
        f"at line(s) {offenders}; this must not be plumbed from `mrope_interleaved`."
    )
