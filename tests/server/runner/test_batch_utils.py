from unittest.mock import Mock, patch

import pytest
import torch

from xorl.server.runner.utils.batch_utils import convert_batch_to_tensors, simple_sequence_shard


pytestmark = [pytest.mark.cpu, pytest.mark.server]


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
