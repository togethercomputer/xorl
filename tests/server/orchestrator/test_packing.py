"""Tests for server orchestrator packing: SequentialPacker, pack_samples, unpack_per_token_outputs."""

import numpy as np
import pytest
import torch


pytestmark = [pytest.mark.cpu, pytest.mark.server]

from xorl.server.orchestrator.packing import (
    Packer,
    SequentialPacker,
    pack_samples,
    unpack_per_token_outputs,
    validate_micro_batches,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_data():
    return [
        {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]},
        {"input_ids": [10, 20], "labels": [20, 30]},
        {"input_ids": [100, 200, 300], "labels": [200, 300, 400]},
    ]


@pytest.fixture
def mixed_length_data():
    return [
        {"input_ids": [1] * 10, "labels": [1] * 10},
        {"input_ids": [2] * 50, "labels": [2] * 50},
        {"input_ids": [3] * 5, "labels": [3] * 5},
        {"input_ids": [4] * 30, "labels": [4] * 30},
        {"input_ids": [5] * 15, "labels": [5] * 15},
    ]


# ============================================================================
# Core packing
# ============================================================================


def test_packing_enabled(simple_data):
    """Packing ON: samples concatenated into single sequence with correct shifting."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(simple_data, max_seq_len=100, request_id="test-001")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["request_id"] == "test-001"
    assert batch["batch_id"] == 0
    assert batch["num_samples"] == 3

    # After shifting: (4-1) + (2-1) + (3-1) = 6 tokens
    assert batch["input_ids"] == [[1, 2, 3, 10, 100, 200]]
    assert batch["labels"] == [[3, 4, 5, 30, 300, 400]]
    assert batch["position_ids"] == [[0, 1, 2, 0, 0, 1]]


def test_packing_exceeds_capacity(simple_data):
    """Samples overflow one batch -> split into multiple batches."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(simple_data, max_seq_len=5, request_id="test-002")

    assert len(batches) == 2
    assert batches[0]["input_ids"][0] == [1, 2, 3]
    assert batches[0]["num_samples"] == 1
    assert batches[1]["input_ids"][0] == [10, 100, 200]
    assert batches[1]["num_samples"] == 2
    assert batches[1]["position_ids"][0] == [0, 0, 1]


def test_packing_disabled(simple_data):
    """Packing OFF: one batch per sample."""
    packer = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(simple_data, max_seq_len=1000, request_id="test-003")

    assert len(batches) == 3
    for i, batch in enumerate(batches):
        assert batch["batch_id"] == i
        assert batch["request_id"] == "test-003"
        assert len(batch["input_ids"]) == 1


def test_position_ids_and_labels():
    """Auto-generated position_ids and label creation (IGNORE_INDEX when missing)."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)

    # Without labels -> IGNORE_INDEX (-100)
    no_labels = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}, {"input_ids": [6, 7, 8, 9]}]
    batches = packer.pack(no_labels, max_seq_len=100)
    assert batches[0]["labels"][0] == [-100] * 6
    assert batches[0]["position_ids"][0] == [0, 1, 0, 0, 1, 2]
    assert batches[0]["num_samples"] == 3

    # With custom position_ids (ignored in packed mode, auto-generated)
    with_pos = [
        {"input_ids": [1, 2, 3], "position_ids": [0, 1, 2], "labels": [2, 3, 4]},
        {"input_ids": [10, 20], "position_ids": [0, 1], "labels": [20, 30]},
    ]
    batches2 = packer.pack(with_pos, max_seq_len=100)
    assert batches2[0]["position_ids"][0] == [0, 1, 0]


def test_edge_cases():
    """Empty list, single sample, oversized samples, missing input_ids."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)

    # Empty
    assert len(packer.pack([], max_seq_len=100)) == 0

    # Single sample
    batches = packer.pack([{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}], max_seq_len=100)
    assert len(batches) == 1 and batches[0]["input_ids"][0] == [1, 2]

    # Oversized samples skipped, valid ones packed
    data = [
        {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
        {"input_ids": [1] * 100, "labels": [1] * 100},
        {"input_ids": [4, 5], "labels": [5, 6]},
    ]
    batches = packer.pack(data, max_seq_len=10)
    assert batches[0]["num_samples"] == 2

    # All oversized -> ValueError
    with pytest.raises(ValueError, match="All 2 samples were skipped"):
        packer.pack([{"input_ids": [1] * 100}, {"input_ids": [2] * 200}], max_seq_len=10)

    # Missing input_ids -> skipped
    data2 = [
        {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
        {"labels": [5, 6, 7]},
        {"input_ids": [4, 5], "labels": [5, 6]},
    ]
    batches = packer.pack(data2, max_seq_len=100)
    assert batches[0]["num_samples"] == 2


def test_mixed_length_and_capacity(mixed_length_data):
    """Mixed lengths, exact fit, off-by-one, max_seq_len invariant."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)

    # Mixed lengths
    batches = packer.pack(mixed_length_data, max_seq_len=60)
    assert len(batches) == 2
    assert batches[0]["num_samples"] == 2
    assert len(batches[0]["input_ids"][0]) == 58  # (10-1) + (50-1)
    assert batches[1]["num_samples"] == 3
    assert len(batches[1]["input_ids"][0]) == 47  # (5-1) + (30-1) + (15-1)

    # No batch exceeds max_seq_len
    for batch in batches:
        assert len(batch["input_ids"][0]) <= 60

    # Exact fit: 5+5=10
    batches = packer.pack([{"input_ids": [1] * 5}, {"input_ids": [2] * 5}], max_seq_len=10)
    assert len(batches) == 1 and batches[0]["num_samples"] == 2

    # Off-by-one: 5+6=11 > 10
    batches = packer.pack([{"input_ids": [1] * 5}, {"input_ids": [2] * 6}], max_seq_len=10)
    assert len(batches) == 2


def test_validate_micro_batches():
    """Validation: valid batches pass, missing field / empty / length mismatch fail."""
    valid = [
        {
            "input_ids": [[1, 2, 3], [4, 5]],
            "labels": [[2, 3, 4], [5, 6]],
            "position_ids": [[0, 1, 2], [0, 1]],
            "request_id": "test",
            "batch_id": 0,
        }
    ]
    assert validate_micro_batches(valid) is True

    # Missing request_id
    assert (
        validate_micro_batches(
            [
                {
                    "input_ids": [[1, 2, 3]],
                    "labels": [[2, 3, 4]],
                    "position_ids": [[0, 1, 2]],
                    "batch_id": 0,
                }
            ]
        )
        is False
    )

    # Empty input_ids
    assert (
        validate_micro_batches(
            [
                {
                    "input_ids": [],
                    "labels": [],
                    "position_ids": [],
                    "request_id": "t",
                    "batch_id": 0,
                }
            ]
        )
        is False
    )

    # Length mismatch (labels vs input_ids)
    assert (
        validate_micro_batches(
            [
                {
                    "input_ids": [[1, 2, 3], [4, 5]],
                    "labels": [[2, 3, 4]],
                    "position_ids": [[0, 1, 2], [0, 1]],
                    "request_id": "t",
                    "batch_id": 0,
                }
            ]
        )
        is False
    )

    # Position_ids length mismatch
    assert (
        validate_micro_batches(
            [
                {
                    "input_ids": [[1, 2, 3]],
                    "labels": [[2, 3, 4]],
                    "position_ids": [[0, 1]],
                    "request_id": "t",
                    "batch_id": 0,
                }
            ]
        )
        is False
    )


def test_pack_samples_function(simple_data):
    """pack_samples convenience function: with/without packing, default params."""
    batches = pack_samples(
        simple_data, max_seq_len=10, enable_packing=True, request_id="func-test", pad_to_multiple_of=1
    )
    assert len(batches) == 1 and batches[0]["request_id"] == "func-test"

    batches_no_pack = pack_samples(simple_data, enable_packing=False, pad_to_multiple_of=1)
    assert len(batches_no_pack) == 3


def test_packer_abstract_and_custom():
    """Packer ABC cannot be instantiated; custom subclass works."""
    with pytest.raises(TypeError):
        Packer()

    class CustomPacker(Packer):
        def pack(self, datum_list, max_seq_len, request_id=""):
            return [
                {
                    "input_ids": [d["input_ids"]],
                    "labels": [d.get("labels", [])],
                    "position_ids": [list(range(len(d["input_ids"])))],
                    "request_id": request_id,
                    "batch_id": i,
                }
                for i, d in enumerate(datum_list)
                if "input_ids" in d
            ]

    cp = CustomPacker()
    assert cp.get_name() == "CustomPacker"
    batches = cp.pack([{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}], max_seq_len=100)
    assert len(batches) == 2


def test_batch_structure_and_invariants(simple_data):
    """Batch keys, types, length consistency, packing efficiency."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(simple_data, max_seq_len=5)

    required_keys = {"input_ids", "labels", "position_ids", "request_id", "batch_id", "num_samples", "_shifted"}
    optional_keys = {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}

    for batch in batches:
        batch_keys = set(batch.keys())
        assert required_keys <= batch_keys
        assert batch_keys <= required_keys | optional_keys
        assert isinstance(batch["input_ids"], list) and isinstance(batch["batch_id"], int)
        # Length consistency
        assert len(batch["input_ids"]) == 1
        assert len(batch["input_ids"][0]) == len(batch["labels"][0]) == len(batch["position_ids"][0])

    # Packing produces fewer batches than no-packing
    packer_off = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    data20 = [{"input_ids": list(range(i % 10 + 1))} for i in range(20)]
    packed = packer.pack(data20, max_seq_len=30)
    unpacked = packer_off.pack(data20, max_seq_len=30)
    assert len(packed) < len(unpacked) == 20

    # Total samples preserved
    assert sum(b["num_samples"] for b in packed) == 20


def test_large_batch_and_numpy():
    """Large batch: all samples accounted for, valid structure; numpy arrays converted."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)

    # Large batch
    data = [{"input_ids": list(range(i % 20 + 1)), "labels": list(range(i % 20 + 1))} for i in range(100)]
    batches = packer.pack(data, max_seq_len=50, request_id="large")
    assert len(batches) > 1
    assert sum(b["num_samples"] for b in batches) == 100
    assert validate_micro_batches(batches) is True

    # Numpy arrays converted to lists
    np_data = [
        {"input_ids": np.array([1, 2, 3]), "labels": np.array([2, 3, 4])},
        {"input_ids": np.array([4, 5]), "labels": np.array([5, 6])},
    ]
    batches = packer.pack(np_data, max_seq_len=100)
    assert isinstance(batches[0]["input_ids"][0], list)


# ============================================================================
# Unpack per-token outputs
# ============================================================================


def test_unpack_per_token_outputs():
    """Unpack: no-shift, shift, single/multi sample, 2D tensors, lists, min-length."""
    # No-shift: output length == position_ids length
    pos = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    out = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    result = unpack_per_token_outputs(out, pos)
    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result[1] == pytest.approx([0.6, 0.7, 0.8])

    # Shift: output has (total - num_samples) tokens
    out_shifted = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = unpack_per_token_outputs(out_shifted, pos)
    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2, 0.3, 0.4])  # 5-1=4
    assert result[1] == pytest.approx([0.5, 0.6])  # 3-1=2

    # Single sample
    result = unpack_per_token_outputs(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0, 1, 2, 3, 4]))
    assert len(result) == 1 and result[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])

    # Three samples shifted
    pos3 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4])
    out3 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    result = unpack_per_token_outputs(out3, pos3)
    assert len(result) == 3
    assert len(result[0]) == 3 and len(result[1]) == 2 and len(result[2]) == 4

    # 2D tensors
    result = unpack_per_token_outputs(
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]]),
    )
    assert len(result) == 2

    # Lists
    result = unpack_per_token_outputs([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 2, 3, 4, 0, 1, 2])
    assert len(result) == 2

    # Minimum-length sample (2 tokens -> 1 after shift)
    result = unpack_per_token_outputs(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0, 1, 0, 1, 2]))
    assert len(result) == 2
    assert len(result[0]) == 1 and len(result[1]) == 2


# ============================================================================
# Full pipeline roundtrip
# ============================================================================


def test_full_pipeline_roundtrip():
    """Pack -> simulate forward -> unpack: token counts preserved, single and multi-batch."""
    data = [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [2, 3, 4, 5, 6], "weights": [0, 0, 1, 1, 1]},
        {"input_ids": [10, 20, 30], "labels": [20, 30, 40], "weights": [0, 1, 1]},
        {"input_ids": [100, 200], "labels": [200, 300], "weights": [1, 1]},
    ]
    total_shifted = sum(len(d["input_ids"]) for d in data) - len(data)  # 10 - 3 = 7

    # Single batch
    batches = pack_samples(data, max_seq_len=100, enable_packing=True, request_id="rt", pad_to_multiple_of=1)
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch["input_ids"][0]) == total_shifted
    assert batch["position_ids"][0] == [0, 1, 2, 3, 0, 1, 0]

    # Simulate forward + unpack
    pos_tensor = torch.tensor([batch["position_ids"][0]])
    logprobs = torch.randn(1, total_shifted)
    result = unpack_per_token_outputs(logprobs, pos_tensor)
    assert len(result) == 3
    assert [len(r) for r in result] == [4, 2, 1]
    assert sum(len(r) for r in result) == total_shifted

    # Multi-batch roundtrip
    big_data = [
        {"input_ids": list(range(50)), "labels": list(range(50))},
        {"input_ids": list(range(40)), "labels": list(range(40))},
        {"input_ids": list(range(30)), "labels": list(range(30))},
    ]
    total_shifted_big = sum(len(d["input_ids"]) for d in big_data) - len(big_data)
    batches = pack_samples(big_data, max_seq_len=60, enable_packing=True, request_id="rt2", pad_to_multiple_of=1)
    assert len(batches) == 3

    all_unpacked = []
    for b in batches:
        out = torch.randn(1, len(b["input_ids"][0]))
        all_unpacked.extend(unpack_per_token_outputs(out, torch.tensor([b["position_ids"][0]])))
    assert sum(len(r) for r in all_unpacked) == total_shifted_big

    # Shift mode detection: no-shift vs shift
    pos = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 0, 1]])
    no_shift = unpack_per_token_outputs(torch.randn(1, 10), pos)
    assert [len(r) for r in no_shift] == [5, 3, 2]
    shifted = unpack_per_token_outputs(torch.randn(1, 7), pos)
    assert [len(r) for r in shifted] == [4, 2, 1]
