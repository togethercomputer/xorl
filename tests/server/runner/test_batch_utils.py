from unittest.mock import Mock, patch

import pytest
import torch

from xorl.server.runner.utils.batch_utils import convert_batch_to_tensors, simple_sequence_shard


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_convert_batch_to_tensors_preserves_drgrpo_logprob_floats():
    converted = convert_batch_to_tensors(
        {
            "old_logprobs": [[-1.25, -2.5]],
            "ref_logprobs": [[-1.75, -3.125]],
        }
    )

    assert converted["old_logprobs"].dtype == torch.float32
    assert converted["ref_logprobs"].dtype == torch.float32
    torch.testing.assert_close(converted["old_logprobs"], torch.tensor([[-1.25, -2.5]]))
    torch.testing.assert_close(converted["ref_logprobs"], torch.tensor([[-1.75, -3.125]]))


def test_convert_batch_to_tensors_uses_float_identity_padding_for_temperatures():
    converted = convert_batch_to_tensors({"logprob_temperatures": [[0.7, 0.7], [1.3]]})

    assert converted["logprob_temperatures"].dtype == torch.float32
    torch.testing.assert_close(
        converted["logprob_temperatures"],
        torch.tensor([[0.7, 0.7], [1.3, 1.0]]),
    )


def test_convert_batch_to_tensors_uses_sampling_transform_identity_padding():
    converted = convert_batch_to_tensors(
        {
            "logprob_top_ks": [[8, 7], [4]],
            "logprob_top_ps": [[0.9, 0.8], [0.7]],
            "logprob_min_ps": [[0.1, 0.2], [0.3]],
        }
    )
    assert torch.equal(converted["logprob_top_ks"], torch.tensor([[8, 7], [4, 1 << 30]]))
    torch.testing.assert_close(converted["logprob_top_ps"], torch.tensor([[0.9, 0.8], [0.7, 1.0]]))
    torch.testing.assert_close(converted["logprob_min_ps"], torch.tensor([[0.1, 0.2], [0.3, 0.0]]))


def test_convert_batch_to_tensors_preserves_teacher_hidden_state_floats():
    batch = {
        "teacher_hidden_states": [
            [[0.25, -1.75], [2.5, 3.125]],
        ],
    }

    converted = convert_batch_to_tensors(batch)

    assert converted["teacher_hidden_states"].dtype == torch.float32
    assert converted["teacher_hidden_states"].shape == (1, 2, 2)
    torch.testing.assert_close(
        converted["teacher_hidden_states"],
        torch.tensor([[[0.25, -1.75], [2.5, 3.125]]], dtype=torch.float32),
    )


def test_convert_batch_to_tensors_pads_ragged_teacher_hidden_states():
    batch = {
        "teacher_hidden_states": [
            [[0.25, 0.5]],
            [[1.25, 1.5], [2.25, 2.5]],
        ],
    }

    converted = convert_batch_to_tensors(batch)

    assert converted["teacher_hidden_states"].shape == (2, 2, 2)
    torch.testing.assert_close(
        converted["teacher_hidden_states"],
        torch.tensor(
            [
                [[0.25, 0.5], [0.0, 0.0]],
                [[1.25, 1.5], [2.25, 2.5]],
            ],
            dtype=torch.float32,
        ),
    )


@patch("xorl.server.runner.utils.batch_utils.get_parallel_state")
def test_simple_sequence_shard_slices_teacher_hidden_states_on_sequence_dim(mock_parallel_state):
    mock_parallel_state.return_value = Mock(cp_size=2, cp_rank=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "teacher_hidden_states": torch.tensor([[[0.25, 0.5], [1.25, 1.5], [2.25, 2.5]]]),
    }

    sharded = simple_sequence_shard(batch)

    assert sharded["teacher_hidden_states"].shape == (1, 2, 2)
    torch.testing.assert_close(
        sharded["teacher_hidden_states"],
        torch.tensor([[[2.25, 2.5], [0.0, 0.0]]]),
    )


@patch("xorl.server.runner.utils.batch_utils.get_parallel_state")
def test_simple_sequence_shard_identity_pads_temperatures(mock_parallel_state):
    mock_parallel_state.return_value = Mock(cp_size=2, cp_rank=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "logprob_temperatures": torch.tensor([[0.7, 0.9, 1.3]]),
    }

    sharded = simple_sequence_shard(batch)

    torch.testing.assert_close(
        sharded["logprob_temperatures"],
        torch.tensor([[1.3, 1.0]]),
    )


@patch("xorl.server.runner.utils.batch_utils.get_parallel_state")
def test_simple_sequence_shard_keeps_batched_temperatures_contiguous(mock_parallel_state):
    mock_parallel_state.return_value = Mock(cp_size=2, cp_rank=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "labels": torch.tensor([[2, 3, 4, -100], [6, 7, 8, -100]]),
        "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "logprob_temperatures": torch.tensor(
            [[0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4]],
            dtype=torch.float32,
        ),
    }

    sharded = simple_sequence_shard(batch)

    assert sharded["logprob_temperatures"].is_contiguous()
    torch.testing.assert_close(
        sharded["logprob_temperatures"],
        torch.tensor([[0.9, 1.0], [1.3, 1.4]], dtype=torch.float32),
    )
