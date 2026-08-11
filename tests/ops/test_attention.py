"""Tests for attention backend functions."""

from unittest.mock import Mock, patch

import pytest
import torch

import xorl.models.layers.attention.backend as backend_module
import xorl.models.layers.attention.backend.flash_attention as flash_module
from xorl.models.layers.attention.backend.eager import eager_attention_forward


try:
    from xorl.models.layers.attention.backend.flash_attention import FA3_AVAILABLE, flash_attention_forward

    _FLASH_ATTN_IMPORT_ERROR = None
except ImportError as exc:
    FA3_AVAILABLE = False
    flash_attention_forward = None
    _FLASH_ATTN_IMPORT_ERROR = exc


pytestmark = pytest.mark.cpu


class TestAttentionBackendRegistry:
    """flash_attention_forward picks FA3/FA4 internally, so registration must not
    depend on FA3 alone: an unregistered key silently dispatches to eager."""

    def test_attention_backend_registry_and_resolution_policy(self, monkeypatch):
        assert flash_module.FA3_AVAILABLE or flash_module.FA4_AVAILABLE
        assert backend_module.ATTENTION_FUNCTIONS["flash_attention_2"] is flash_module.flash_attention_forward
        assert backend_module.ATTENTION_FUNCTIONS["flash_attention_3"] is flash_module.flash_attention_forward
        assert backend_module.is_flash_attention("flash_attention_2")
        assert backend_module.is_flash_attention("flash_attention_3")
        if flash_module.FA4_AVAILABLE:
            assert "flash_attention_4" in backend_module.ATTENTION_FUNCTIONS
            assert backend_module.is_flash_attention("flash_attention_4")

        self._assert_resolution_keeps_non_flash_fallback_and_rejects_unavailable_flash(monkeypatch)
        TestEagerAttentionForward()._assert_eager_attention_head_layout()

    def _assert_resolution_keeps_non_flash_fallback_and_rejects_unavailable_flash(self, monkeypatch):
        assert backend_module.get_attention_fn("eager") is eager_attention_forward
        assert backend_module.get_attention_fn("native") is backend_module.ATTENTION_FUNCTIONS["native"]
        # A non-flash implementation has no attention contract to drop, so it
        # may still default to eager; only the flash family raises.
        assert backend_module.get_attention_fn("flex_attention") is eager_attention_forward

        monkeypatch.delitem(backend_module.ATTENTION_FUNCTIONS, "flash_attention_3", raising=False)
        with pytest.raises(ImportError, match="flash_attention_3"):
            backend_module.get_attention_fn("flash_attention_3")


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
            assert not mock_fa.call_args[1]["causal"]

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

        self._assert_varlen_path_with_cu_seqlens()

    def _assert_varlen_path_with_cu_seqlens(self):
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

    def _assert_eager_attention_head_layout(self):
        """Ulysses-sync head layout handling and invalid head layout error."""
        module = Mock()
        module.num_key_value_groups = 8
        module.training = False

        # Valid: local Q=4, KV=1 -> repeat=4. Compare the complete attention
        # transaction against an independent repeat_interleave reference so
        # incorrect KV values cannot hide behind shape-only assertions.
        torch.manual_seed(0)
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
        query_heads = query.transpose(1, 2)
        key_heads = key.transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
        value_heads = value.transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
        expected_weights = torch.softmax(
            torch.matmul(query_heads, key_heads.transpose(2, 3)) * head_dim**-0.5,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        expected_output = torch.matmul(expected_weights, value_heads).transpose(1, 2).contiguous()
        torch.testing.assert_close(attn_weights, expected_weights)
        torch.testing.assert_close(attn_output, expected_output)

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


class TestPageSize1KVCacheParityPaths:
    """The env-gated routes must reach the serving entry point with num_splits=1."""

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

    def test_sgl_page_size1_kv_cache_policy(self):
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

        self._assert_single_sequence_synthesizes_cu_seqlens()
        self._assert_rejects_cross_attention_cu_seqlens()
        self._assert_alternate_flash_attention_path_selection_policy()

    def _assert_single_sequence_synthesizes_cu_seqlens(self):
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

    def _assert_alternate_flash_attention_path_selection_policy(self):
        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with (
            patch.object(flash_module, "XORL_FLASH_ATTN_SGL_KERNEL", False),
            patch.object(flash_module, "XORL_FLASH_ATTN_PAGED_KVCACHE", True),
            patch.object(flash_module, "FA3_AVAILABLE", True),
            patch.object(flash_module, "_should_use_fa4", return_value=False),
            patch.object(flash_module, "flash_attn_with_kvcache") as mock_kvcache,
        ):
            mock_kvcache.return_value = torch.zeros(total_tokens, num_heads, head_dim)
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
        assert mock_kvcache.called
        kwargs = mock_kvcache.call_args[1]
        assert kwargs["num_splits"] == 1
        assert kwargs["k_cache"].shape == (total_tokens, 1, num_heads, head_dim)

        self._assert_flags_off_keep_default_varlen_path()
        self._assert_fa4_path_pins_num_splits_and_forwards_scale()

    def _assert_flags_off_keep_default_varlen_path(self):
        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with (
            patch.object(flash_module, "XORL_FLASH_ATTN_SGL_KERNEL", False),
            patch.object(flash_module, "XORL_FLASH_ATTN_PAGED_KVCACHE", False),
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

    def _assert_fa4_path_pins_num_splits_and_forwards_scale(self):
        total_tokens, num_heads, head_dim = 32, 8, 64
        query = torch.randn(1, total_tokens, num_heads, head_dim)
        key = torch.randn(1, total_tokens, num_heads, head_dim)
        value = torch.randn(1, total_tokens, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int64)

        with (
            patch.object(flash_module, "XORL_FLASH_ATTN_SGL_KERNEL", False),
            patch.object(flash_module, "FA4_AVAILABLE", True),
            patch.object(flash_module, "_should_use_fa4", return_value=True),
            patch.object(flash_module, "fa4_flash_attn_varlen_func") as mock_fa4,
        ):
            mock_fa4.return_value = (torch.zeros(total_tokens, num_heads, head_dim), None)
            flash_attention_forward(
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
        assert mock_fa4.called
        kwargs = mock_fa4.call_args[1]
        assert kwargs["num_splits"] == 1
        assert kwargs["softmax_scale"] == 0.125
        assert kwargs["causal"] is True

    def _assert_rejects_cross_attention_cu_seqlens(self):
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
