import pytest
import torch

from xorl.models.layers.rope import rotate_half
from xorl.models.transformers import qwen3_5_shared
from xorl.models.transformers.qwen3_5 import modeling_qwen3_5
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5_moe import modeling_qwen3_5_moe
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
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


def _assert_interleaved_matches_pairwise_reference():
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


def _assert_class_b_rotary_admission_policy(monkeypatch):
    sentinel = (object(), object())
    calls = []

    class _CudaInput:
        is_cuda = True
        dtype = torch.bfloat16
        ndim = 4
        shape = (1, 2, 1, 4)

    def _stock(q, k, cos, sin, *, interleaved):
        calls.append((q, k, cos, sin, interleaved))
        return sentinel

    monkeypatch.setattr(qwen3_5_shared, "stock_fused_apply_rotary_pos_emb", _stock)
    q, k = (_CudaInput() for _ in range(2))
    cos = torch.zeros((1, 2, 4), dtype=torch.float32)
    sin = torch.zeros_like(cos)

    assert qwen3_5_shared.qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, class_b=True) is sentinel
    assert calls == [(q, k, cos, sin, False)]

    _assert_class_b_fails_loudly_outside_cuda_contract()


def _assert_class_b_fails_loudly_outside_cuda_contract():
    cos = torch.zeros((1, 2, 4), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    for dtype in (torch.bfloat16, torch.float32):
        q = torch.zeros((1, 2, 1, 4), dtype=dtype)
        k = torch.zeros_like(q)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            qwen3_5_apply_rotary_pos_emb(q, k, cos, sin, class_b=True)


def _assert_qwen35_attention_keeps_half_rotate_when_mrope_is_interleaved(attention_type, config_type):
    """mRoPE interleaves frequencies, not the q/k feature rotation convention."""
    torch.manual_seed(23)
    config = config_type(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        mrope_interleaved=True,
    )
    attention = attention_type(config, layer_idx=0)
    hidden = torch.randn(1, 3, 8)
    cos, sin = _build_halved_cos_sin(batch=1, seq=3, head_dim=4)

    query, key, _ = attention._project_qkv(hidden, (cos, sin))

    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query_pre_rope, _ = torch.chunk(
        attention.q_proj(hidden).view(*input_shape, -1, attention.head_dim * 2),
        2,
        dim=-1,
    )
    query_pre_rope = attention.q_norm(query_pre_rope.view(hidden_shape))
    key_pre_rope = attention.k_norm(attention.k_proj(hidden).view(hidden_shape))
    expected_query, expected_key = _hf_reference_half_rotate(query_pre_rope, key_pre_rope, cos, sin)
    pairwise_query, pairwise_key = _hf_reference_pairwise(query_pre_rope, key_pre_rope, cos, sin)

    assert config.mrope_interleaved is True
    torch.testing.assert_close(query, expected_query, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(key, expected_key, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(query, pairwise_query)
    assert not torch.allclose(key, pairwise_key)


def test_qwen35_rotary_numerics_admission_and_attention_policy(monkeypatch):
    _assert_interleaved_matches_pairwise_reference()
    _assert_class_b_rotary_admission_policy(monkeypatch)

    for attention_type, config_type in (
        (modeling_qwen3_5.Qwen3_5Attention, Qwen3_5Config),
        (modeling_qwen3_5_moe.Qwen3_5MoeAttention, Qwen3_5MoeConfig),
    ):
        _assert_qwen35_attention_keeps_half_rotate_when_mrope_is_interleaved(attention_type, config_type)

    _assert_qwen35_exact_attention_casts_post_rope_qk_to_bf16_policy()


def _assert_qwen35_exact_attention_casts_post_rope_qk_to_bf16(attention_type, config_type):
    config = config_type(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        layer_types=["full_attention"],
    )
    config._attention_cast_bf16 = True
    attention = attention_type(config, layer_idx=0)

    hidden = torch.randn(1, 3, 8)
    cos, sin = _build_halved_cos_sin(batch=1, seq=3, head_dim=4)
    query, key, value = attention._project_qkv(hidden, (cos, sin))

    assert query.dtype is torch.bfloat16
    assert key.dtype is torch.bfloat16
    assert value.dtype is torch.float32


def _assert_qwen35_exact_attention_casts_post_rope_qk_to_bf16_policy():
    for attention_type, config_type in (
        (modeling_qwen3_5.Qwen3_5Attention, Qwen3_5Config),
        (modeling_qwen3_5_moe.Qwen3_5MoeAttention, Qwen3_5MoeConfig),
    ):
        _assert_qwen35_exact_attention_casts_post_rope_qk_to_bf16(attention_type, config_type)
