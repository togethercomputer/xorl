"""Tests for xorl.ops.attention module."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from xorl.ops.attention import repeat_kv, flash_attention_forward


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
    """Test suite for flash_attention_forward function."""

    def test_output_attentions_warning(self):
        """Test that output_attentions=True triggers a warning."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            mock_ps.return_value.sp_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                with patch('xorl.ops.attention.logger') as mock_logger:
                    flash_attention_forward(
                        module,
                        query,
                        key,
                        value,
                        attention_mask=None,
                        output_attentions=True,
                    )
                    
                    # Verify warning was logged
                    assert mock_logger.warning_once.called
    
    def test_head_mask_warning(self):
        """Test that head_mask parameter triggers a warning."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                with patch('xorl.ops.attention.logger') as mock_logger:
                    flash_attention_forward(
                        module,
                        query,
                        key,
                        value,
                        attention_mask=None,
                        head_mask=torch.ones(num_heads),
                    )
                    
                    assert mock_logger.warning_once.called
    
    def test_query_key_value_transposition(self):
        """Test that query, key, value are transposed correctly."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                result, _ = flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                )
                
                # Verify that transposition happened
                called_query = mock_fa.call_args[0][0]
                # After transpose(1, 2), shape should be (batch, seqlen, num_heads, head_dim)
                assert called_query.shape == (batch, seqlen, num_heads, head_dim)
    
    def test_is_causal_removed_from_kwargs(self):
        """Test that is_causal is removed from kwargs and uses module.is_causal instead."""
        module = Mock()
        module.is_causal = False  # Module's is_causal
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    is_causal=True,  # This kwarg should be popped and ignored
                )
                
                # Verify the module's is_causal was used (False), not the kwarg (True)
                assert mock_fa.call_args[1]['is_causal'] == False
    
    def test_position_ids_3d_handling(self):
        """Test that 3D position_ids are reduced to 2D."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        position_ids = torch.randn(3, batch, seqlen)  # 3D position_ids
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    position_ids=position_ids,
                )
                
                # position_ids should be reduced to 2D
                called_pos_ids = mock_fa.call_args[1].get('position_ids')
                if called_pos_ids is not None:
                    assert called_pos_ids.ndim == 2
    
    def test_without_ulysses(self):
        """Test flash attention without Ulysses parallelism."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                expected_output = torch.zeros(batch, seqlen, num_heads, head_dim)
                mock_fa.return_value = expected_output
                
                result, attn_weights = flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                )
                
                assert result.shape == expected_output.shape
                assert attn_weights is None
    
    def test_ulysses_with_kv_repeat(self):
        """Test flash attention with Ulysses when KV heads need repeating."""
        module = Mock()
        module.is_causal = True
        
        batch, q_heads, kv_heads, seqlen, head_dim = 1, 8, 2, 16, 64
        query = torch.randn(batch, q_heads, seqlen, head_dim)
        key = torch.randn(batch, kv_heads, seqlen, head_dim)
        value = torch.randn(batch, kv_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = True
            mock_ps.return_value.ulysses_size = 4  # ulysses_size > kv_heads
            mock_ps.return_value.ulysses_group = Mock()
            
            with patch('xorl.ops.attention.gather_seq_scatter_heads') as mock_gather:
                mock_gather.return_value = torch.randn(seqlen, q_heads // 4, head_dim)
                
                with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                    mock_fa.return_value = torch.zeros(batch, seqlen, q_heads // 4, head_dim)
                    
                    with patch('xorl.ops.attention.gather_heads_scatter_seq') as mock_gather_back:
                        mock_gather_back.return_value = torch.zeros(seqlen, q_heads, head_dim)
                        
                        result, _ = flash_attention_forward(
                            module,
                            query,
                            key,
                            value,
                            attention_mask=None,
                        )
                        
                        # Verify KV repeat was called
                        assert mock_gather.call_count >= 3  # q, k, v
    
    def test_skip_ulysses_flag(self):
        """Test that skip_ulysses flag bypasses Ulysses parallelism."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = True
            mock_ps.return_value.ulysses_group = Mock()
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                with patch('xorl.ops.attention.gather_seq_scatter_heads') as mock_gather:
                    flash_attention_forward(
                        module,
                        query,
                        key,
                        value,
                        attention_mask=None,
                        skip_ulysses=True,
                    )
                    
                    # gather_seq_scatter_heads should not be called
                    assert not mock_gather.called
    
    def test_with_scaling_parameter(self):
        """Test flash attention with custom scaling factor."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        scaling = 0.125
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    scaling=scaling,
                )
                
                # Verify scaling was passed
                assert mock_fa.call_args[1]['softmax_scale'] == scaling
    
    def test_with_sliding_window(self):
        """Test flash attention with sliding window attention."""
        module = Mock()
        module.is_causal = True
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        query = torch.randn(batch, num_heads, seqlen, head_dim)
        key = torch.randn(batch, num_heads, seqlen, head_dim)
        value = torch.randn(batch, num_heads, seqlen, head_dim)
        sliding_window = 128
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            with patch('xorl.ops.attention._flash_attention_forward') as mock_fa:
                mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)
                
                flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    sliding_window=sliding_window,
                )
                
                assert mock_fa.call_args[1]['sliding_window'] == sliding_window
    
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
        
        batch, num_heads, seqlen, head_dim = 2, 8, 16, 64
        # FlashAttention only supports fp16 and bf16
        query = torch.randn(batch, num_heads, seqlen, head_dim, dtype=torch.float16).cuda()
        key = torch.randn(batch, num_heads, seqlen, head_dim, dtype=torch.float16).cuda()
        value = torch.randn(batch, num_heads, seqlen, head_dim, dtype=torch.float16).cuda()
        
        with patch('xorl.ops.attention.get_parallel_state') as mock_ps:
            mock_ps.return_value.ulysses_enabled = False
            
            result, _ = flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
            )
            
            assert result.device.type == "cuda"
            assert result.shape == (batch, seqlen, num_heads, head_dim)

