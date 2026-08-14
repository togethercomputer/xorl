import pytest

from xorl.data.collators import ShiftTokensCollator


pytestmark = [pytest.mark.cpu, pytest.mark.collator]


def test_logprob_temperatures_shift_with_causal_targets():
    sample = {
        "input_ids": [1, 2, 3, 4],
        "labels": [1, 2, 3, 4],
        "logprob_temperatures": [1.0, 0.7, 0.8, 0.9],
        "logprob_top_ks": [1 << 30, 8, 7, 6],
        "logprob_top_ps": [1.0, 0.9, 0.8, 0.7],
        "logprob_min_ps": [0.0, 0.1, 0.2, 0.3],
    }

    shifted = ShiftTokensCollator()([sample])[0]

    assert shifted["input_ids"] == [1, 2, 3]
    assert shifted["labels"] == [2, 3, 4]
    assert shifted["logprob_temperatures"] == [0.7, 0.8, 0.9]
    assert shifted["logprob_top_ks"] == [8, 7, 6]
    assert shifted["logprob_top_ps"] == [0.9, 0.8, 0.7]
    assert shifted["logprob_min_ps"] == [0.1, 0.2, 0.3]
