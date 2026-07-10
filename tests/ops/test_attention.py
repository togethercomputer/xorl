"""Tests for attention backend functions."""

import importlib
import sys
import types
from unittest.mock import Mock, patch

import pytest
import torch

import xorl.models.layers.attention.backend as backend_module
import xorl.models.layers.attention.backend.flash_attention as flash_module
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS, is_flash_attention
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.attention.utils import repeat_kv


try:
    from xorl.models.layers.attention.backend.flash_attention import FA3_AVAILABLE, flash_attention_forward

    _FLASH_ATTN_IMPORT_ERROR = None
except ImportError as exc:
    FA3_AVAILABLE = False
    flash_attention_forward = None
    _FLASH_ATTN_IMPORT_ERROR = exc


pytestmark = pytest.mark.cpu


class TestAttentionBackendRegistry:
    def test_flash_attention_2_registry_and_mask_detection_are_consistent(self):
        assert is_flash_attention("flash_attention_2") == ("flash_attention_2" in ATTENTION_FUNCTIONS)
        if "flash_attention_2" in ATTENTION_FUNCTIONS:
            assert ATTENTION_FUNCTIONS["flash_attention_2"] is ATTENTION_FUNCTIONS["flash_attention_3"]

    def test_flash_attention_4_can_register_without_flash_attention_3_dependency(self, monkeypatch):
        fake_flash_attn = types.ModuleType("flash_attn")
        fake_flash_attn.__path__ = []
        fake_cute = types.ModuleType("flash_attn.cute")
        fake_cute.flash_attn_func = Mock()
        fake_cute.flash_attn_varlen_func = Mock()

        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
        monkeypatch.setitem(sys.modules, "flash_attn", fake_flash_attn)
        monkeypatch.setitem(sys.modules, "flash_attn.cute", fake_cute)

        try:
            reloaded_flash = importlib.reload(flash_module)
            reloaded_backend = importlib.reload(backend_module)

            assert not reloaded_flash.FA3_AVAILABLE
            assert reloaded_flash.FA4_AVAILABLE
            # flash_attention_3 must stay registered in FA4-only environments:
            # flash_attention_forward falls through to FA4 internally, and an
            # unregistered key would silently dispatch to eager attention.
            assert "flash_attention_3" in reloaded_backend.ATTENTION_FUNCTIONS
            assert "flash_attention_4" in reloaded_backend.ATTENTION_FUNCTIONS
        finally:
            monkeypatch.undo()
            importlib.reload(flash_module)
            importlib.reload(backend_module)


class TestRepeatKV:
    """Test suite for repeat_kv function."""

    def test_repeat_kv_shapes_values_and_gpu(self):
        """repeat_kv: identity for n_rep=1, correct shapes for 2x/3x, value replication, GPU support."""
        batch, num_heads, seqlen, head_dim = 2, 4, 10, 64
        hidden_states = torch.randn(batch, num_heads, seqlen, head_dim)

        # n_rep=1: unchanged
        result1 = repeat_kv(hidden_states, n_rep=1)
        assert result1.shape == hidden_states.shape
        assert torch.allclose(result1, hidden_states)

        # n_rep=2: doubled heads
        result2 = repeat_kv(hidden_states, n_rep=2)
        assert result2.shape == (batch, num_heads * 2, seqlen, head_dim)

        # n_rep=3: tripled heads
        result3 = repeat_kv(hidden_states, n_rep=3)
        assert result3.shape == (batch, num_heads * 3, seqlen, head_dim)

        # Value replication correctness
        small = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).float()
        rep = repeat_kv(small, n_rep=2)
        assert torch.allclose(rep[:, 0], rep[:, 1])
        assert torch.allclose(rep[:, 2], rep[:, 3])
        assert not torch.allclose(rep[:, 0], rep[:, 2])

        # GPU
        if torch.cuda.is_available():
            gpu_states = torch.randn(batch, num_heads, seqlen, head_dim).cuda()
            gpu_result = repeat_kv(gpu_states, n_rep=2)
            assert gpu_result.device.type == "cuda"
            assert gpu_result.shape == (batch, num_heads * 2, seqlen, head_dim)


class TestFlashAttentionForward:
    """Test suite for flash_attention_forward function."""

    @pytest.fixture(autouse=True)
    def _skip_when_flash_attention_unavailable(self):
        if _FLASH_ATTN_IMPORT_ERROR is not None:
            pytest.skip(f"flash attention backend unavailable: {_FLASH_ATTN_IMPORT_ERROR}")
        if not FA3_AVAILABLE:
            pytest.skip("flash attention 3 backend unavailable")

    def test_flash_attention_api_behavior(self):
        """Warnings, is_causal handling, return values, scaling, sliding window."""
        module = Mock()
        module.is_causal = True
        module.config = Mock()
        module.config._flash_attention_deterministic = False

        batch, seqlen, num_heads, head_dim = 2, 16, 8, 64
        query = torch.randn(batch, seqlen, num_heads, head_dim)
        key = torch.randn(batch, seqlen, num_heads, head_dim)
        value = torch.randn(batch, seqlen, num_heads, head_dim)

        with patch("xorl.models.layers.attention.backend.flash_attention.flash_attn_func") as mock_fa:
            mock_fa.return_value = torch.zeros(batch, seqlen, num_heads, head_dim)

            # output_attentions warning
            with patch("xorl.models.layers.attention.backend.flash_attention.logger") as mock_logger:
                flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    output_attentions=True,
                )
                assert mock_logger.warning_once.called

            # head_mask warning
            with patch("xorl.models.layers.attention.backend.flash_attention.logger") as mock_logger:
                flash_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask=None,
                    head_mask=torch.ones(num_heads),
                )
                assert mock_logger.warning_once.called

            # is_causal kwarg popped, module.is_causal used
            module.is_causal = False
            flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
                is_causal=True,
            )
            assert mock_fa.call_args[1]["causal"] is False

            # Returns None attention weights
            module.is_causal = True
            result, attn_weights = flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
            )
            assert result.shape == (batch, seqlen, num_heads, head_dim)
            assert attn_weights is None

            # Scaling passed as softmax_scale
            flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
                scaling=0.125,
            )
            assert mock_fa.call_args[1]["softmax_scale"] == 0.125

            # Sliding window -> window_size tuple
            flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
                sliding_window=128,
            )
            assert mock_fa.call_args[1]["window_size"] == (128, 0)

            # Configured deterministic backward flag is forwarded to FA3.
            module.config._flash_attention_deterministic = True
            flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
            )
            assert mock_fa.call_args[1]["deterministic"] is True

    def test_varlen_path_with_cu_seqlens(self):
        """cu_seqlens kwargs trigger the varlen path."""
        module = Mock()
        module.is_causal = True
        module.config = Mock()
        module.config._flash_attention_deterministic = True

        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with patch("xorl.models.layers.attention.backend.flash_attention.flash_attn_varlen_func") as mock_varlen:
            mock_varlen.return_value = torch.zeros(total_tokens, num_heads, head_dim)
            result, _ = flash_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=None,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=16,
                max_length_k=16,
            )
            assert mock_varlen.called
            assert mock_varlen.call_args[1]["cu_seqlens_q"].dtype == torch.int32
            assert mock_varlen.call_args[1]["deterministic"] is True
            assert result.shape == (1, total_tokens, num_heads, head_dim)


class TestEagerAttentionForward:
    """Regression tests for eager attention head handling."""

    def test_eager_attention_head_layout(self):
        """Ulysses-sync head layout handling and invalid head layout error."""
        module = Mock()
        module.num_key_value_groups = 8
        module.training = False

        # Valid: local Q=4, KV=1 -> repeat=4
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

        # Invalid: q_heads not divisible by kv_heads
        with pytest.raises(RuntimeError, match="query_heads=3 is not divisible by kv_heads=2"):
            eager_attention_forward(
                module=module,
                query=torch.randn(1, 4, 3, 8),
                key=torch.randn(1, 4, 2, 8),
                value=torch.randn(1, 4, 2, 8),
                attention_mask=None,
                scaling=8**-0.5,
                dropout=0.0,
            )


class TestSglKernelParityPath:
    """XORL_FLASH_ATTN_SGL_KERNEL routes packed attention through sgl_kernel FA3."""

    @pytest.fixture(autouse=True)
    def _skip_when_flash_attention_unavailable(self):
        if _FLASH_ATTN_IMPORT_ERROR is not None:
            pytest.skip(f"flash attention backend unavailable: {_FLASH_ATTN_IMPORT_ERROR}")

    @staticmethod
    def _module():
        module = Mock()
        module.is_causal = True
        module.config = Mock()
        module.config._flash_attention_deterministic = False
        return module

    def test_varlen_path_routes_through_sgl_kernel(self):
        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with (
            patch.object(flash_module, "XORL_FLASH_ATTN_SGL_KERNEL", True),
            patch.object(flash_module, "sgl_flash_attn_with_kvcache") as mock_sgl,
        ):
            mock_sgl.return_value = torch.zeros(total_tokens, num_heads, head_dim)
            result, _ = flash_attention_forward(
                self._module(),
                query,
                key,
                value,
                attention_mask=None,
                scaling=0.125,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=16,
                max_length_k=16,
            )
        assert mock_sgl.called
        kwargs = mock_sgl.call_args[1]
        assert kwargs["num_splits"] == 1
        assert kwargs["cu_seqlens_q"].dtype == torch.int32
        # page-size-1 KV cache: one page per token
        assert kwargs["k_cache"].shape == (total_tokens, 1, num_heads, head_dim)
        assert kwargs["page_table"].shape == (2, 16)
        assert kwargs["cache_seqlens"].tolist() == [16, 16]
        assert kwargs["softmax_scale"] == 0.125
        assert kwargs["causal"] is True
        assert result.shape == (1, total_tokens, num_heads, head_dim)

    def test_single_sequence_batched_path_synthesizes_cu_seqlens(self):
        seqlen, num_heads, head_dim = 16, 8, 64
        query = torch.randn(1, seqlen, num_heads, head_dim)
        key = torch.randn(1, seqlen, num_heads, head_dim)
        value = torch.randn(1, seqlen, num_heads, head_dim)

        with (
            patch.object(flash_module, "XORL_FLASH_ATTN_SGL_KERNEL", True),
            patch.object(flash_module, "sgl_flash_attn_with_kvcache") as mock_sgl,
        ):
            mock_sgl.return_value = torch.zeros(seqlen, num_heads, head_dim)
            result, _ = flash_attention_forward(
                self._module(),
                query,
                key,
                value,
                attention_mask=None,
            )
        assert mock_sgl.called
        kwargs = mock_sgl.call_args[1]
        assert kwargs["cu_seqlens_q"].tolist() == [0, seqlen]
        assert kwargs["max_seqlen_q"] == seqlen
        assert result.shape == (1, seqlen, num_heads, head_dim)

    def test_flag_off_keeps_default_varlen_path(self):
        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with (
            patch.object(flash_module, "XORL_FLASH_ATTN_SGL_KERNEL", False),
            patch.object(flash_module, "FA3_AVAILABLE", True),
            patch.object(flash_module, "sgl_flash_attn_with_kvcache") as mock_sgl,
            patch.object(flash_module, "_should_use_fa4", return_value=False),
            patch.object(flash_module, "flash_attn_varlen_func") as mock_varlen,
        ):
            mock_varlen.return_value = torch.zeros(total_tokens, num_heads, head_dim)
            flash_attention_forward(
                self._module(),
                query,
                key,
                value,
                attention_mask=None,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=16,
                max_length_k=16,
            )
        assert not mock_sgl.called
        assert mock_varlen.called

    def test_rejects_cross_attention_cu_seqlens(self):
        num_heads, head_dim = 8, 64
        q = torch.randn(32, num_heads, head_dim)
        k = torch.randn(48, num_heads, head_dim)
        v = torch.randn(48, num_heads, head_dim)
        with (
            patch.object(flash_module, "sgl_flash_attn_with_kvcache", Mock()),
            pytest.raises(ValueError, match="self-attention"),
        ):
            flash_module._flash_attn_varlen_with_sgl_kernel(
                q,
                k,
                v,
                torch.tensor([0, 32], dtype=torch.int32),
                torch.tensor([0, 48], dtype=torch.int32),
                32,
                48,
                None,
                True,
                (-1, -1),
                None,
            )
