"""Configuration guards for the certified Class-B RoPE lane."""

import pytest

from xorl.models.layers.rope import rope_class_b_enabled, set_rope_class_b
from xorl.trainers.model_builder import build_training_model


pytestmark = pytest.mark.cpu


def test_class_b_requires_serving_table_provenance():
    with pytest.raises(ValueError, match="rope_class_b=True requires rope_native=True"):
        build_training_model(
            config_path="unused",
            weights_path="unused",
            rope_class_b=True,
            rope_native=False,
        )


def test_class_b_selector_can_be_reset():
    set_rope_class_b(True)
    assert rope_class_b_enabled()
    set_rope_class_b(False)
    assert not rope_class_b_enabled()
