"""Session-spec normalization for value-model sessions: frozen patterns, dropout."""

import pytest

from xorl.server.session_spec import normalize_lora_runtime_config


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _normalize(raw):
    return normalize_lora_runtime_config(raw, default_rank=32, default_alpha=32, max_lora_rank=64)


def test_frozen_module_patterns_normalize_sorted_and_deduped():
    config = _normalize({"rank": 8, "frozen_module_patterns": ["v_proj", "q_proj", "q_proj"]})
    assert config["frozen_module_patterns"] == ["q_proj", "v_proj"]
    assert config["lora_rank"] == 8


def test_frozen_module_patterns_omitted_keeps_legacy_spec_shape():
    """Absent (or empty) patterns must not appear in the spec at all, so
    existing session hashes and checkpoints stay byte-identical."""
    assert set(_normalize({"rank": 8})) == {"lora_rank", "lora_alpha"}
    assert set(_normalize({"rank": 8, "frozen_module_patterns": []})) == {"lora_rank", "lora_alpha"}


@pytest.mark.parametrize("bad", ["q_proj", [1, 2], [""], [None]])
def test_frozen_module_patterns_reject_bad_types(bad):
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize({"frozen_module_patterns": bad})


def test_sdk_default_dropout_is_accepted_as_noop():
    """The tinker-style SDK always sends dropout in its LoRA payload; 0.0 is
    exactly the server behavior and must not be rejected."""
    config = _normalize({"rank": 8, "alpha": 16, "dropout": 0.0})
    assert config == {"lora_rank": 8, "lora_alpha": 16}


def test_nonzero_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout is not supported"):
        _normalize({"dropout": 0.1})


def test_other_overrides_still_rejected():
    with pytest.raises(ValueError, match="only override rank, alpha, and frozen_module_patterns"):
        _normalize({"target_modules": ["q_proj"]})
