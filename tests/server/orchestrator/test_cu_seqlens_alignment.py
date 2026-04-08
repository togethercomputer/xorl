"""
Tests to verify that the server path produces identical cu_seq_lens
as the CLI/train.py path (via PackingConcatCollator).

The server path goes through:
  packer._finalize_packed_batch -> _convert_batch_to_tensors -> model

The CLI path goes through:
  PackingConcatCollator -> model

Both must produce cu_seq_lens with:
  1. Correct boundary indices (where position_ids == 0)
  2. dtype int32 (required by flash_attn)
  3. Correct max_length values
"""

import math
from unittest.mock import Mock, patch

import pytest
import torch

from xorl.data.collators.packing_concat_collator import (
    PackingConcatCollator,
)
from xorl.data.collators.sequence_shard_collator import TextSequenceShardCollator
from xorl.server.backend import DummyBackend
from xorl.server.orchestrator.packing import SequentialPacker, unpack_per_token_outputs
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids


pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_path_cu_seqlens(datum_list, max_seq_len=4096, pad_to_multiple_of=1):
    """Simulate the server path: pack -> finalize -> convert_to_tensors."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=pad_to_multiple_of)
    batches = packer.pack(datum_list, max_seq_len=max_seq_len, request_id="test")
    assert len(batches) >= 1

    results = []
    for batch in batches:
        converted = {}
        int32_fields = {"cu_seq_lens_q", "cu_seq_lens_k"}
        float_fields = {"logprobs", "advantages", "old_logprobs"}
        for key, value in batch.items():
            if isinstance(value, list):
                try:
                    if key in float_fields:
                        dtype = torch.float
                    elif key in int32_fields:
                        dtype = torch.int32
                    else:
                        dtype = torch.long
                    converted[key] = torch.tensor(value, dtype=dtype)
                except (ValueError, TypeError):
                    converted[key] = value
            else:
                converted[key] = value
        results.append(converted)
    return results


def _cli_path_cu_seqlens(features, pad_to_multiple_of=1):
    """Simulate the CLI path: PackingConcatCollator."""
    collator = PackingConcatCollator(pad_to_multiple_of=pad_to_multiple_of)
    return collator(features)


def _make_datum(input_ids, labels=None):
    """Create a datum dict in xorl_client API format (already shifted)."""
    if labels is None:
        labels = [t + 1 for t in input_ids]
    return {"input_ids": input_ids, "target_tokens": labels}


def _make_feature(input_ids, position_ids):
    """Create a feature dict for PackingConcatCollator (tensor format)."""
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor([t + 1 for t in input_ids], dtype=torch.long),
        "position_ids": torch.tensor(position_ids, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCuSeqlensAlignmentAndDtype:
    """Verify server/CLI cu_seq_lens match, dtype, and edge cases."""

    @patch("xorl.server.orchestrator.packing.get_parallel_state")
    @patch("xorl.data.collators.packing_concat_collator.get_parallel_state")
    def test_server_cli_match_dtype_and_edge_cases(self, mock_cli_ps, mock_server_ps):
        """Test value match for 2/3 sequences, int32 dtype, single sample, SP enabled, and many short seqs."""
        mock_ps = Mock(cp_enabled=False, cp_size=1, cp_rank=0, ringattn_size=1)
        mock_cli_ps.return_value = mock_ps
        mock_server_ps.return_value = mock_ps

        # --- Two sequences match ---
        seq1, seq2 = [10, 20, 30], [40, 50, 60, 70]
        server_batch = _server_path_cu_seqlens([_make_datum(seq1), _make_datum(seq2)])[0]
        cli_batch = _cli_path_cu_seqlens(
            [
                _make_feature(seq1, list(range(len(seq1)))),
                _make_feature(seq2, list(range(len(seq2)))),
            ]
        )
        assert torch.equal(server_batch["cu_seq_lens_q"], cli_batch["cu_seq_lens_q"])
        assert torch.equal(server_batch["cu_seq_lens_k"], cli_batch["cu_seq_lens_k"])
        assert server_batch["max_length_q"] == cli_batch["max_length_q"]

        # --- Three sequences match ---
        seq1, seq2, seq3 = [1, 2], [3, 4, 5, 6, 7], [8, 9, 10]
        server_batch = _server_path_cu_seqlens([_make_datum(s) for s in [seq1, seq2, seq3]])[0]
        cli_batch = _cli_path_cu_seqlens([_make_feature(s, list(range(len(s)))) for s in [seq1, seq2, seq3]])
        expected = torch.tensor([0, 2, 7, 10], dtype=torch.int32)
        assert torch.equal(server_batch["cu_seq_lens_q"], expected)
        assert torch.equal(cli_batch["cu_seq_lens_q"], expected)
        assert server_batch["max_length_q"] == 5

        # --- int32 dtype ---
        batch = _server_path_cu_seqlens([_make_datum([1, 2, 3]), _make_datum([4, 5])])[0]
        assert batch["cu_seq_lens_q"].dtype == torch.int32
        assert batch["cu_seq_lens_k"].dtype == torch.int32
        assert cli_batch["cu_seq_lens_q"].dtype == torch.int32

        position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
        (cu_q, cu_k), _ = prepare_fa_kwargs_from_position_ids(position_ids)
        assert cu_q.dtype == torch.int32

        # --- Single sample: no cu_seq_lens ---
        batch = _server_path_cu_seqlens([_make_datum([1, 2, 3, 4, 5])])[0]
        assert "cu_seq_lens_q" not in batch

        # --- Many short sequences ---
        seqs = [[i] for i in range(10)]
        server_batch = _server_path_cu_seqlens([_make_datum(s) for s in seqs])[0]
        cli_batch = _cli_path_cu_seqlens([_make_feature(s, [0]) for s in seqs])
        expected = torch.tensor(list(range(11)), dtype=torch.int32)
        assert torch.equal(server_batch["cu_seq_lens_q"], expected)
        assert torch.equal(cli_batch["cu_seq_lens_q"], expected)

    @patch("xorl.server.orchestrator.packing.get_parallel_state")
    def test_sp_enabled_skips_cu_seqlens(self, mock_ps):
        """When SP is enabled, packer should NOT compute cu_seq_lens."""
        mock_ps.return_value = Mock(cp_enabled=True, cp_size=2, cp_rank=0, ringattn_size=1)
        batch = _server_path_cu_seqlens([_make_datum([1, 2, 3]), _make_datum([4, 5])])[0]
        assert "cu_seq_lens_q" not in batch


class TestOriginalPositionIdsAndPadding:
    """Verify _original_position_ids preservation, stale cu_seq_lens, and padding alignment."""

    def test_sequence_shard_collator_preservation_and_stale_overwrite(self):
        """Test _original_position_ids preservation, no-pad, and stale cu_seq_lens overwrite."""

        # Preservation (SP size 2, 6 tokens)
        with (
            patch("xorl.data.collators.sequence_shard_collator.get_parallel_state") as mock_ps,
            patch("xorl.data.collators.packing_concat_collator.get_parallel_state") as mock_ps2,
        ):
            mock = Mock(cp_size=2, cp_rank=0, cp_enabled=True, ringattn_size=1)
            mock_ps.return_value = mock
            mock_ps2.return_value = mock
            collator = TextSequenceShardCollator()
            batch = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
                "labels": torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
            }
            original_pos = batch["position_ids"].clone()
            result = collator(batch)
            assert "_original_position_ids" in result
            assert torch.equal(result["_original_position_ids"], original_pos)
            assert result["input_ids"].shape[-1] == 3
            assert result["position_ids"].shape[-1] >= 6

        # Not padded (SP size 4, 5 tokens -> pad to 8, but _original keeps 5)
        with (
            patch("xorl.data.collators.sequence_shard_collator.get_parallel_state") as mock_ps,
            patch("xorl.data.collators.packing_concat_collator.get_parallel_state") as mock_ps2,
        ):
            mock = Mock(cp_size=4, cp_rank=0, cp_enabled=True, ringattn_size=1)
            mock_ps.return_value = mock
            mock_ps2.return_value = mock
            collator = TextSequenceShardCollator()
            batch = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
                "labels": torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long),
            }
            result = collator(batch)
            assert result["_original_position_ids"].shape[-1] == 5
            assert result["position_ids"].shape[-1] == 8

        # Stale cu_seq_lens overwritten
        with (
            patch("xorl.data.collators.sequence_shard_collator.get_parallel_state") as mock_ps,
            patch("xorl.data.collators.packing_concat_collator.get_parallel_state") as mock_ps2,
        ):
            mock = Mock(cp_size=4, cp_rank=0, cp_enabled=True, ringattn_size=1)
            mock_ps.return_value = mock
            mock_ps2.return_value = mock
            collator = TextSequenceShardCollator()
            batch = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
                "labels": torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long),
                "cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32),
                "cu_seq_lens_k": torch.tensor([0, 3, 5], dtype=torch.int32),
                "max_length_q": 3,
                "max_length_k": 3,
            }
            result = collator(batch)
            assert result["cu_seq_lens_q"][-1].item() == 8
            assert result["cu_seq_lens_k"][-1].item() == 8

        # cu_seq_lens match with padding
        pad = 128
        with (
            patch("xorl.server.orchestrator.packing.get_parallel_state") as mock_server_ps,
            patch("xorl.data.collators.packing_concat_collator.get_parallel_state") as mock_cli_ps,
        ):
            mock_ps = Mock(cp_enabled=False, cp_size=1, cp_rank=0, ringattn_size=1)
            mock_server_ps.return_value = mock_ps
            mock_cli_ps.return_value = mock_ps
            seq1, seq2 = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10]
            server_batch = _server_path_cu_seqlens([_make_datum(seq1), _make_datum(seq2)], pad_to_multiple_of=pad)[0]
            cli_batch = _cli_path_cu_seqlens(
                [
                    _make_feature(seq1, list(range(len(seq1)))),
                    _make_feature(seq2, list(range(len(seq2)))),
                ],
                pad_to_multiple_of=pad,
            )
            assert server_batch["input_ids"].shape[-1] == pad
            assert torch.equal(server_batch["cu_seq_lens_q"], cli_batch["cu_seq_lens_q"])
            assert server_batch["cu_seq_lens_q"][-1].item() == pad


class TestPadToMultipleWithSpSize:
    """Verify padding accounts for both pad_to_multiple_of and cp_size."""

    @patch("xorl.server.orchestrator.packing.get_parallel_state")
    def test_lcm_padding_and_collator_divisibility(self, mock_ps):
        """Test lcm-based padding, RequestProcessor computation, and collator output divisibility."""
        # cp_size=3, base=128 -> lcm=384
        mock_ps.return_value = Mock(cp_enabled=True, cp_size=3, cp_rank=0, ringattn_size=1)
        effective_pad = math.lcm(128, 3)
        assert effective_pad == 384
        packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=effective_pad)
        batches = packer.pack([_make_datum([1, 2, 3]), _make_datum([4, 5])], max_seq_len=4096, request_id="test")
        seq_len = len(batches[0]["input_ids"][0])
        assert seq_len % 384 == 0 and seq_len % 3 == 0 and seq_len % 128 == 0

        # cp_size=8 -> lcm=128; cp_size=1 -> lcm=128
        for sp, expected in [(8, 128), (1, 128)]:
            mock_ps.return_value = Mock(cp_enabled=sp > 1, cp_size=sp, cp_rank=0, ringattn_size=1)
            packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=math.lcm(128, sp))
            batches = packer.pack([_make_datum([1, 2, 3]), _make_datum([4, 5])], max_seq_len=4096, request_id="test")
            assert len(batches[0]["input_ids"][0]) == expected

        # RequestProcessor computes lcm correctly

        backend = DummyBackend()
        assert RequestProcessor(backend=backend, pad_to_multiple_of=128, cp_size=3).pad_to_multiple_of == 384
        assert RequestProcessor(backend=backend, pad_to_multiple_of=128, cp_size=8).pad_to_multiple_of == 128
        assert RequestProcessor(backend=backend, pad_to_multiple_of=128, cp_size=6).pad_to_multiple_of == 384

        # After TextSequenceShardCollator, lengths divisible by cp_size

        cp_size = 3
        mock_ps.return_value = Mock(cp_enabled=True, cp_size=cp_size, cp_rank=0, ringattn_size=1)
        packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=math.lcm(128, cp_size))
        batches = packer.pack(
            [_make_datum(list(range(50))), _make_datum(list(range(30)))], max_seq_len=4096, request_id="test"
        )
        batch = batches[0]
        pre_len = len(batch["input_ids"][0])
        assert pre_len % cp_size == 0

        with (
            patch("xorl.data.collators.sequence_shard_collator.get_parallel_state") as mock_sp,
            patch("xorl.data.collators.packing_concat_collator.get_parallel_state") as mock_sp2,
        ):
            m = Mock(cp_size=cp_size, cp_rank=0, cp_enabled=True, ringattn_size=1)
            mock_sp.return_value = m
            mock_sp2.return_value = m
            collator = TextSequenceShardCollator()
            result = collator(
                {
                    "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
                    "labels": torch.tensor(batch["labels"], dtype=torch.long),
                    "position_ids": torch.tensor(batch["position_ids"], dtype=torch.long),
                }
            )
            assert result["position_ids"].shape[-1] == pre_len
            assert result["input_ids"].shape[-1] == pre_len // cp_size
            assert result["cu_seq_lens_q"][-1].item() == pre_len


class TestUnpackingWithPadding:
    """Verify that padding doesn't create spurious samples during unpacking."""

    @patch("xorl.server.orchestrator.packing.get_parallel_state")
    def test_padding_boundary_handling(self, mock_ps):
        """Padding creates extra boundary; no padding has correct count."""

        mock_ps.return_value = Mock(cp_enabled=False, cp_size=1, cp_rank=0, ringattn_size=1)

        datum_list = [_make_datum([10, 20, 30]), _make_datum([40, 50, 60, 70])]

        # With padding: extra boundary
        packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=128)
        batch = packer.pack(datum_list, max_seq_len=4096, request_id="test")[0]
        assert batch["num_samples"] == 2 and len(batch["position_ids"][0]) == 128
        pos_ids = torch.tensor(batch["position_ids"][0], dtype=torch.long)
        unpacked = unpack_per_token_outputs(torch.randn(127), pos_ids)
        assert len(unpacked) == 3 and len(unpacked[: batch["num_samples"]]) == 2

        # Without padding: correct count
        packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
        batch = packer.pack(datum_list, max_seq_len=4096, request_id="test")[0]
        assert batch["num_samples"] == 2 and len(batch["position_ids"][0]) == 7
        pos_ids = torch.tensor(batch["position_ids"][0], dtype=torch.long)
        unpacked = unpack_per_token_outputs(torch.randn(6), pos_ids)
        assert len(unpacked) == 2
