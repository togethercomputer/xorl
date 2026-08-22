"""Packing of value-model (critic) per-token fields: returns / old_values."""

import pytest

from xorl.data.constants import IGNORE_INDEX
from xorl.server.orchestrator.packing import SequentialPacker


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_returns_pack_in_shifted_client_format():
    """xorl_client format (pre-shifted target_tokens): returns concatenate as-is."""
    data = [
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [2, 3, 4],
            "weights": [0.0, 1.0, 1.0],
            "returns": [0.0, 0.7, 0.9],
            "old_values": [0.0, 0.6, 0.8],
        },
        {
            "input_ids": [5, 6],
            "target_tokens": [6, 7],
            "weights": [1.0, 1.0],
            "returns": [0.5, 0.4],
            "old_values": [0.5, 0.3],
        },
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(data, max_seq_len=100, request_id="value")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["returns"] == [[0.0, 0.7, 0.9, 0.5, 0.4]]
    assert batch["old_values"] == [[0.0, 0.6, 0.8, 0.5, 0.3]]
    # weights=0 masks; returns=0.0 must NOT mask (unlike advantages).
    assert batch["target_tokens"] == [[IGNORE_INDEX, 3, 4, 6, 7]]


def test_returns_shift_with_hf_style_labels():
    """HF format (unshifted labels): returns/old_values are target-aligned and
    shift with labels[1:]."""
    data = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [10, 20, 30, 40],
            "returns": [0.1, 0.2, 0.3, 0.4],
            "old_values": [1.1, 1.2, 1.3, 1.4],
        }
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(data, max_seq_len=100, request_id="value-hf")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"] == [[1, 2, 3]]
    assert batch["labels"] == [[20, 30, 40]]
    assert batch["returns"] == [[0.2, 0.3, 0.4]]
    assert batch["old_values"] == [[1.2, 1.3, 1.4]]


def test_zero_returns_do_not_mask_labels():
    """A legitimate return of exactly 0.0 keeps its token in the loss."""
    data = [
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [2, 3, 4],
            "weights": [1.0, 1.0, 1.0],
            "returns": [0.0, 0.0, 0.0],
        }
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(data, max_seq_len=100, request_id="value-zero")

    batch = batches[0]
    assert batch["target_tokens"] == [[2, 3, 4]]
    assert IGNORE_INDEX not in batch["target_tokens"][0]
