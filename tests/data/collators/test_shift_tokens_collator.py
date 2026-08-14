import pytest

from xorl.data.collators import ShiftTokensCollator


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


def test_logprob_temperatures_shift_with_causal_targets():
    sample = {
        "input_ids": [1, 2, 3, 4],
        "labels": [1, 2, 3, 4],
        "logprob_temperatures": [1.0, 0.7, 0.8, 0.9],
    }

    shifted = ShiftTokensCollator()([sample])[0]

    assert shifted["input_ids"] == [1, 2, 3]
    assert shifted["labels"] == [2, 3, 4]
    assert shifted["logprob_temperatures"] == [0.7, 0.8, 0.9]
