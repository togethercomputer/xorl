"""Tests for server orchestrator packing: SequentialPacker, pack_samples, unpack_per_token_outputs."""

import numpy as np
import pytest
import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.server.orchestrator.packing import (
    SequentialPacker,
    _resolve_teacher_cache_base,
    pack_samples,
    unpack_per_token_outputs,
    validate_micro_batches,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


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


def test_packing_capacity_and_batching_policy(simple_data, mixed_length_data):
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
    assert batch["_r3_sample_lengths"] == [3, 1, 2]

    _assert_packing_exceeds_capacity(simple_data)
    _assert_mixed_length_and_capacity(mixed_length_data)
    _assert_packing_edge_and_validation_policy()
    _assert_full_pipeline_roundtrip_and_generated_metadata()


def _assert_packed_token_metadata_policy():
    """OPD teacher ids, cache refs, and weights survive packed dispatch."""
    data = [
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [2, 3, 4],
            "teacher_ids": [0, 0, 0],
            "teacher_cache_indices": [10, 11, 12],
            "teacher_weights": [1.0, 1.0, 0.5],
        },
        {
            "input_ids": [5, 6],
            "target_tokens": [6, 7],
            "teacher_id": 2,
            "teacher_cache_indices": [20, 21],
            "teacher_weight": 0.25,
        },
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(data, max_seq_len=100, request_id="opd")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["labels"] == [[2, 3, 4, 6, 7]]
    assert batch["teacher_ids"] == [[0, 0, 0, 2, 2]]
    assert batch["teacher_cache_indices"] == [[10, 11, 12, 20, 21]]
    assert batch["teacher_weights"] == [[1.0, 1.0, 0.5, 0.25, 0.25]]

    _assert_opd_metadata_shifts_with_hf_labels()
    _assert_teacher_hidden_states_pad_as_vectors()
    _assert_target_tokens_and_rl_fields_pad_correctly()
    _assert_oprd_global_and_local_teacher_cache_views()
    _assert_teacher_cache_base_schema_and_legacy_fallback()
    _assert_nested_rl_target_tokens_pad_with_ignore_index()


def _assert_opd_metadata_shifts_with_hf_labels():
    """OPD per-token fields stay aligned when packing shifts HF-style labels."""
    data = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [10, 20, 30, 40],
            "teacher_ids": [0, 0, 1, 1],
            "teacher_cache_indices": [100, 101, 102, 103],
            "teacher_weights": [1.0, 0.5, 0.25, 0.125],
            "teacher_hidden_states": [
                [1.0, 1.5],
                [2.0, 2.5],
                [3.0, 3.5],
                [4.0, 4.5],
            ],
        }
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(data, max_seq_len=100, request_id="opd-hf")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"] == [[1, 2, 3]]
    assert batch["labels"] == [[20, 30, 40]]
    assert batch["teacher_ids"] == [[0, 0, 1]]
    assert batch["teacher_cache_indices"] == [[100, 101, 102]]
    assert batch["teacher_weights"] == [[1.0, 0.5, 0.25]]
    assert batch["teacher_hidden_states"] == [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]]


def _assert_teacher_hidden_states_pad_as_vectors():
    data = [
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [2, 3, 4],
            "teacher_hidden_states": [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]],
        }
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=4)
    batches = packer.pack(data, max_seq_len=100, request_id="opd-pad")

    assert len(batches) == 1
    assert batches[0]["teacher_hidden_states"] == [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [0.0, 0.0]]]


def _assert_target_tokens_and_rl_fields_pad_correctly():
    data = [
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [2, 3, 4],
            "logprobs": [-0.1, -0.2, -0.3],
            "advantages": [1.0, 0.0, 1.0],
        }
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=4)
    batches = packer.pack(data, max_seq_len=100, request_id="rl-pad")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["labels"] == [[2, -100, 4, -100]]
    assert batch["target_tokens"] == [[2, -100, 4, -100]]
    assert batch["logprobs"] == [[-0.1, -0.2, -0.3, 0]]
    assert batch["advantages"] == [[1.0, 0.0, 1.0, 0]]


def _assert_oprd_global_and_local_teacher_cache_views():
    data = [
        {
            "input_ids": [11, 12, 13],
            "target_tokens": [12, 13, 14],
            "teacher_input_ids": [101, 102, 103, 104],
            "teacher_kept_indices": [1, 2, 3],
            "teacher_cache_indices": [50, 51, 52],
            "teacher_cache_base": [50],
        },
        {
            "input_ids": [21, 22],
            "target_tokens": [22, 23],
            "teacher_input_ids": [201, 202, 203],
            "teacher_kept_indices": [0, 2],
            "teacher_cache_indices": [90, 91],
            "teacher_cache_base": torch.tensor([90]),
        },
    ]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)

    batches = packer.pack(data, max_seq_len=100, request_id="oprd")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["teacher_input_ids"] == [101, 102, 103, 104, 201, 202, 203]
    assert batch["teacher_kept_indices"] == [[1, 2, 3, 4, 6]]
    assert batch["teacher_position_ids"] == [0, 1, 2, 3, 0, 1, 2]
    assert batch["teacher_cache_indices"] == [[50, 51, 52, 90, 91]]
    assert batch["teacher_cache_local_indices"] == [[0, 1, 2, 3, 4]]


def _assert_teacher_cache_base_schema_and_legacy_fallback():
    assert _resolve_teacher_cache_base([17], [17, 18]) == 17
    assert _resolve_teacher_cache_base(torch.tensor([23]), [23, 24]) == 23
    assert _resolve_teacher_cache_base(None, [5, 7]) == 5
    assert _resolve_teacher_cache_base([], []) == 0


def _assert_packing_exceeds_capacity(simple_data):
    """Samples overflow one batch -> split into multiple batches."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(simple_data, max_seq_len=5, request_id="test-002")

    assert len(batches) == 2
    assert batches[0]["input_ids"][0] == [1, 2, 3]
    assert batches[0]["num_samples"] == 1
    assert batches[1]["input_ids"][0] == [10, 100, 200]
    assert batches[1]["num_samples"] == 2
    assert batches[1]["position_ids"][0] == [0, 0, 1]


def test_packing_disabled_policy(simple_data, monkeypatch):
    """Packing OFF: one batch per sample, but HF-format datums are still shifted."""
    packer = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    batches = packer.pack(simple_data, max_seq_len=1000, request_id="test-003")

    assert len(batches) == 3
    for i, batch in enumerate(batches):
        assert batch["batch_id"] == i
        assert batch["request_id"] == "test-003"
        assert len(batch["input_ids"]) == 1
        assert len(batch["input_ids"][0]) == len(batch["labels"][0]) == len(batch["position_ids"][0])

    assert batches[0]["input_ids"] == [[1, 2, 3]]
    assert batches[0]["labels"] == [[3, 4, 5]]
    assert batches[0]["position_ids"] == [[0, 1, 2]]
    assert batches[1]["input_ids"] == [[10]]
    assert batches[1]["labels"] == [[30]]
    assert batches[1]["position_ids"] == [[0]]
    assert batches[2]["input_ids"] == [[100, 200]]
    assert batches[2]["labels"] == [[300, 400]]
    assert batches[2]["position_ids"] == [[0, 1]]

    _assert_disabled_packing_preserves_shifted_target_tokens()
    _assert_disabled_packing_warns_on_hf_shift(monkeypatch)
    _assert_disabled_packing_preserves_explicit_target_tokens()
    _assert_disabled_packing_applies_loss_masks_to_targets()


def _assert_disabled_packing_preserves_shifted_target_tokens():
    """Packing OFF should leave already-shifted xorl_client-format datums unchanged."""
    packer = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    datum = {
        "model_input": {"input_ids": [11, 22, 33]},
        "loss_fn_inputs": {
            "target_tokens": [22, 33, 44],
            "logprobs": [0.0, 0.0, 0.0],
            "advantages": [1.0, 1.0, 1.0],
        },
    }

    batches = packer.pack([datum], max_seq_len=1000, request_id="test-shifted")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"] == [[11, 22, 33]]
    assert batch["labels"] == [[22, 33, 44]]
    assert batch["target_tokens"] == [[22, 33, 44]]
    assert batch["logprobs"] == [[0.0, 0.0, 0.0]]
    assert batch["advantages"] == [[1.0, 1.0, 1.0]]


def _assert_disabled_packing_warns_on_hf_shift(monkeypatch):
    """HF labels should warn when shifted in the non-packed path."""
    packer = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    warnings = []

    def _capture_warning(message, *args, **_kwargs):
        warnings.append(message % args)

    monkeypatch.setattr("xorl.server.orchestrator.packing.logger.warning", _capture_warning)
    datum = {
        "input_ids": [1, 2, 3],
        "labels": [10, 20, 30],
    }

    batches = packer.pack([datum], max_seq_len=1000, request_id="test-shift-warning")

    batch = batches[0]
    assert batch["input_ids"] == [[1, 2]]
    assert batch["labels"] == [[20, 30]]
    assert any("treating it as HF-format data" in warning for warning in warnings)


def _assert_disabled_packing_preserves_explicit_target_tokens():
    """Preserved target_tokens should not be replaced by labels during non-packed processing."""
    packer = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    datum = {
        "input_ids": [1, 2, 3],
        "labels": [10, 20, 30],
        "target_tokens": [101, 102, 103],
    }

    batches = packer.pack([datum], max_seq_len=1000, request_id="test-target-preserve")

    batch = batches[0]
    assert batch["input_ids"] == [[1, 2, 3]]
    assert batch["labels"] == [[10, 20, 30]]
    assert batch["target_tokens"] == [[101, 102, 103]]


def _assert_disabled_packing_applies_loss_masks_to_targets():
    packer = SequentialPacker(enable_packing=False, log_stats=False, pad_to_multiple_of=1)
    datum = {
        "input_ids": [1, 2, 3],
        "labels": [10, 20, 30],
        "target_tokens": [101, 102, 103],
        "advantages": [1.0, 0.0, 1.0],
    }

    batches = packer.pack([datum], max_seq_len=1000, request_id="test-target-mask")

    batch = batches[0]
    assert batch["labels"] == [[10, -100, 30]]
    assert batch["target_tokens"] == [[101, -100, 103]]


def _assert_nested_rl_target_tokens_pad_with_ignore_index():
    """Packed RL datums must not count padding as valid DR-GRPO target tokens."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=8)
    data = [
        {
            "model_input": {"input_ids": [11, 22, 33]},
            "loss_fn_inputs": {
                "target_tokens": [22, 33, 44],
                "logprobs": [-0.1, -0.2, -0.3],
                "advantages": [1.0, 1.0, 1.0],
            },
        },
        {
            "model_input": {"input_ids": [55, 66]},
            "loss_fn_inputs": {
                "target_tokens": [66, 77],
                "logprobs": [-0.4, -0.5],
                "advantages": [-1.0, -1.0],
            },
        },
    ]

    batches = packer.pack(data, max_seq_len=1000, request_id="test-target-pad")

    assert len(batches) == 1
    batch = batches[0]
    assert batch["target_tokens"] == [[22, 33, 44, 66, 77, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["labels"] == [[22, 33, 44, 66, 77, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["logprobs"] == [[-0.1, -0.2, -0.3, -0.4, -0.5, 0, 0, 0]]
    assert batch["advantages"] == [[1.0, 1.0, 1.0, -1.0, -1.0, 0, 0, 0]]


def _assert_packing_edge_and_validation_policy():
    """Empty list, single sample, oversized samples (skip mode), missing input_ids."""
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)

    # Empty
    assert len(packer.pack([], max_seq_len=100)) == 0

    # Single sample
    batches = packer.pack([{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}], max_seq_len=100)
    assert len(batches) == 1 and batches[0]["input_ids"][0] == [1, 2]

    # All oversized -> ValueError (skip mode: nothing survives). Mixed valid and
    # oversized input is owned by the packing-strategy admission contract.
    skip_packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1, on_oversized="skip")
    with pytest.raises(ValueError, match="All 2 samples were skipped"):
        skip_packer.pack([{"input_ids": [1] * 100}, {"input_ids": [2] * 200}], max_seq_len=10)

    # Missing input_ids -> skipped (independent of on_oversized)
    data2 = [
        {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
        {"labels": [5, 6, 7]},
        {"input_ids": [4, 5], "labels": [5, 6]},
    ]
    batches = packer.pack(data2, max_seq_len=100)
    assert batches[0]["num_samples"] == 2

    _assert_numpy_inputs_are_converted_to_lists()
    _assert_validate_micro_batches()


def _assert_mixed_length_and_capacity(mixed_length_data):
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


def _assert_validate_micro_batches():
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


def _assert_numpy_inputs_are_converted_to_lists():
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    np_data = [
        {"input_ids": np.array([1, 2, 3]), "labels": np.array([2, 3, 4])},
        {"input_ids": np.array([4, 5]), "labels": np.array([5, 6])},
    ]
    batches = packer.pack(np_data, max_seq_len=100)
    assert isinstance(batches[0]["input_ids"][0], list)


# ============================================================================
# Unpack per-token outputs
# ============================================================================


def _assert_unpack_per_token_outputs_policy():
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


def _assert_full_pipeline_roundtrip_and_generated_metadata():
    """Pack, generate metadata, simulate forward, and unpack sample boundaries."""
    _assert_packed_token_metadata_policy()
    _assert_unpack_per_token_outputs_policy()

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

    # Missing labels are generated as ignored targets while positions still
    # reset at every packed-document boundary.
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1)
    no_labels = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}, {"input_ids": [6, 7, 8, 9]}]
    generated = packer.pack(no_labels, max_seq_len=100)[0]
    assert generated["labels"][0] == [IGNORE_INDEX] * 6
    assert generated["position_ids"][0] == [0, 1, 0, 0, 1, 2]
    assert generated["num_samples"] == 3

    # Caller-provided positions do not override packed-document boundaries.
    with_pos = [
        {"input_ids": [1, 2, 3], "position_ids": [0, 1, 2], "labels": [2, 3, 4]},
        {"input_ids": [10, 20], "position_ids": [0, 1], "labels": [20, 30]},
    ]
    assert packer.pack(with_pos, max_seq_len=100)[0]["position_ids"][0] == [0, 1, 0]
