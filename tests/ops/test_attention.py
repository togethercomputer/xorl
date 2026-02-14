"""Tests for attention backend functions."""

import pytest
import torch
from unittest.mock import Mock, patch

from xorl.models.layers.attention.utils import repeat_kv
from xorl.models.layers.attention.backend.eager import eager_attention_forward

try:
    from xorl.models.layers.attention.backend.flash_attention import flash_attention_forward
    _FLASH_ATTN_IMPORT_ERROR = None
except ImportError as exc:
    flash_attention_forward = None
    _FLASH_ATTN_IMPORT_ERROR = exc


pytestmark = pytest.mark.cpu


class TestRepeatKV:
    """Test suite for repeat_kv function."""

    def test_no_repeat_when_n_rep_is_1(self):
        """Test that tensor is unchanged when n_rep=1."""
        batch, num_heads, seqlen, head_dim = 2, 4, 10, 64
        hidden_states = torch.randn(batch, num_heads, seqlen, head_dim)

        result = repeat_kv(hidden_states, n_rep=1)

        assert result.shape == hidden_states.shape
        assert torch.allclose(result, hidden_states)

    def test_repeat_kv_doubles_heads(self):
        """Test that repeat_kv correctly doubles the number of heads."""
        batch, num_heads, seqlen, head_dim = 2, 4, 10, 64
        hidden_states = torch.randn(batch, num_heads, seqlen, head_dim)

        result = repeat_kv(hidden_states, n_rep=2)

        expected_shape = (batch, num_heads * 2, seqlen, head_dim)
        assert result.shape == expected_shape

    def test_repeat_kv_triple_heads(self):
        """Test repeat_kv with n_rep=3."""
        batch, num_heads, seqlen, head_dim = 2, 4, 10, 64
        hidden_states = torch.randn(batch, num_heads, seqlen, head_dim)

        result = repeat_kv(hidden_states, n_rep=3)

        expected_shape = (batch, num_heads * 3, seqlen, head_dim)
        assert result.shape == expected_shape

    def test_repeat_kv_values_are_replicated(self):
        """Test that values are properly replicated across heads."""
        batch, num_heads, seqlen, head_dim = 1, 2, 3, 4
        hidden_states = torch.arange(batch * num_heads * seqlen * head_dim).reshape(
            batch, num_heads, seqlen, head_dim
        ).float()

        result = repeat_kv(hidden_states, n_rep=2)

        # First original head should match its repetition
        assert torch.allclose(result[:, 0], result[:, 1])
        # Second original head should match its repetition
        assert torch.allclose(result[:, 2], result[:, 3])
        # But first and second original heads should differ
        assert not torch.allclose(result[:, 0], result[:, 2])

    @pytest.mark.gpu
    def test_repeat_kv_on_gpu(self):
        """Test repeat_kv works on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch, num_heads, seqlen, head_dim = 2, 4, 10, 64
        hidden_states = torch.randn(batch, num_heads, seqlen, head_dim).cuda()

        result = repeat_kv(hidden_states, n_rep=2)

        assert result.device.type == "cuda"
        assert result.shape == (batch, num_heads * 2, seqlen, head_dim)


class TestFlashAttentionForward:
    """Test suite for flash_attention_forward function.

    The new flash_attention_forward is a pure attention function (no SP logic).
    It calls flash_attn_func/flash_attn_varlen_func directly.
    """

    @pytest.fixture(autouse=True)
    def _skip_when_flash_attention_unavailable(self):
        if _FLASH_ATTN_IMPORT_ERROR is not None:
            pytest.skip(f"flash attention backend unavailable: {_FLASH_ATTN_IMPORT_ERROR}")

    def test_output_attentions_warning(self):
        """Test that output_attentions=True triggers a warning."""
        module = Mock()
        module.is_causal = True

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_func') as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            with patch('xorl.models.layers.attention.backend.flash_attention.logger') as mock_logger:
                flash_attention_forward(
                    module, query, key, value,
                    attention_mask=None,
                    output_attentions=True,
                )
                assert mock_logger.warning_once.called

    def test_head_mask_warning(self):
        """Test that head_mask parameter triggers a warning."""
        module = Mock()
        module.is_causal = True

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_func') as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            with patch('xorl.models.layers.attention.backend.flash_attention.logger') as mock_logger:
                flash_attention_forward(
                    module, query, key, value,
                    attention_mask=None,
                    head_mask=torch.ones(num_heads),
                )
                assert mock_logger.warning_once.called

    def test_is_causal_removed_from_kwargs(self):
        """Test that is_causal kwarg is popped and module.is_causal is used."""
        module = Mock()
        module.is_causal = False

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_func') as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            flash_attention_forward(
                module, query, key, value,
                attention_mask=None,
                is_causal=True,  # Should be popped; module.is_causal (False) used instead
            )

            # flash_attn_func receives causal=module.is_causal (False)
            assert mock_fa.call_args[1]['causal'] == False

    def test_returns_none_attn_weights(self):
        """Test that flash attention returns None for attention weights."""
        module = Mock()
        module.is_causal = True

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_func') as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            result, attn_weights = flash_attention_forward(
                module, query, key, value,
                attention_mask=None,
            )

            assert result.shape == (batch, seqlen, num_heads, head_dim)
            assert attn_weights is None

    def test_scaling_passed_as_softmax_scale(self):
        """Test that scaling is passed as softmax_scale to flash_attn."""
        module = Mock()
        module.is_causal = True

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_func') as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            flash_attention_forward(
                module, query, key, value,
                attention_mask=None,
                scaling=0.125,
            )

            assert mock_fa.call_args[1]['softmax_scale'] == 0.125

    def test_sliding_window_converted_to_window_size(self):
        """Test that sliding_window int is converted to window_size tuple."""
        module = Mock()
        module.is_causal = True

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_func') as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            flash_attention_forward(
                module, query, key, value,
                attention_mask=None,
                sliding_window=128,
            )

            # For causal=True: window_size = (128, 0)
            assert mock_fa.call_args[1]['window_size'] == (128, 0)

    def test_varlen_path_with_cu_seqlens(self):
        """Test that cu_seqlens kwargs trigger the varlen path."""
        module = Mock()
        module.is_causal = True

        # Packed sequence: batch=1, total_tokens in seq dim
        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)

        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with patch('xorl.models.layers.attention.backend.flash_attention.flash_attn_varlen_func') as mock_varlen:
            mock_varlen.return_value = torch.zeros(total_tokens, num_heads, head_dim)

            result, _ = flash_attention_forward(
                module, query, key, value,
                attention_mask=None,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=16,
                max_length_k=16,
            )

            # Varlen func should have been called (not regular func)
            assert mock_varlen.called
            # cu_seqlens should be converted to int32
            called_cu_q = mock_varlen.call_args[1]['cu_seqlens_q']
            assert called_cu_q.dtype == torch.int32
            # Output should have batch dim restored
            assert result.shape == (1, total_tokens, num_heads, head_dim)

    @pytest.mark.gpu
    def test_flash_attention_on_gpu(self):
        """Test flash attention on GPU (requires flash-attn installed)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from flash_attn import flash_attn_func  # noqa: F401
        except ImportError:
            pytest.skip("flash-attn not installed")

        module = Mock()
        module.is_causal = True

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.float16).cuda()
        key = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.float16).cuda()
        value = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.float16).cuda()

        result, _ = flash_attention_forward(
            module, query, key, value,
            attention_mask=None,
        )

        assert result.device.type == "cuda"
        assert result.shape == (batch, seqlen, num_heads, head_dim)


class TestEagerAttentionForward:
    """Regression tests for eager attention head handling."""

    def test_runtime_kv_repeat_handles_ulysses_head_layout(self):
        # Simulate Ulysses-sync local tensors:
        # local Q heads = 4, local KV heads = 1.
        # Global config can still carry num_key_value_groups=8.
        # Eager attention should derive repeat=4 from tensor shapes (not 8).
        module = Mock()
        module.num_key_value_groups = 8
        module.training = False

        batch, seq, q_heads, kv_heads, head_dim = 1, 8, 4, 1, 16
        query = torch.randn(batch, seq, q_heads, head_dim)
        key = torch.randn(batch, seq, kv_heads, head_dim)
        value = torch.randn(batch, seq, kv_heads, head_dim)

        attn_output, attn_weights = eager_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            scaling=head_dim**-0.5,
            dropout=0.0,
        )

        assert attn_output.shape == (batch, seq, q_heads, head_dim)
        assert attn_weights.shape == (batch, q_heads, seq, seq)

    def test_runtime_kv_repeat_raises_on_invalid_head_layout(self):
        module = Mock()
        module.num_key_value_groups = 8
        module.training = False

        # q_heads is not divisible by kv_heads -> invalid GQA layout
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 4, 2, 8)
        value = torch.randn(1, 4, 2, 8)

        with pytest.raises(RuntimeError, match="query_heads=3 is not divisible by kv_heads=2"):
            eager_attention_forward(
                module=module,
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                scaling=8**-0.5,
                dropout=0.0,
            )
