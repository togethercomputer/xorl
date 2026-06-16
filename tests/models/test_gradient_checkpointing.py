"""Tests for `GradientCheckpointingLayer` + `gradient_checkpointing_enable` contract."""

from unittest.mock import MagicMock

import pytest
import torch

from xorl.models.base import XorlPreTrainedModel
from xorl.models.module_utils import (
    DEFAULT_GRADIENT_CHECKPOINTING_METHOD,
    GradientCheckpointingLayer,
    MoEGradientCheckpointingLayer,
)


pytestmark = [pytest.mark.cpu]


class _IdentityCheckpointLayer(GradientCheckpointingLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _StubMoECheckpointLayer(MoEGradientCheckpointingLayer):
    """MoE stub for attribute-level assertions — forward is not exercised."""


@pytest.fixture
def model() -> XorlPreTrainedModel:
    m = XorlPreTrainedModel(config=None)
    m.layer = _IdentityCheckpointLayer()
    return m


@pytest.mark.parametrize(
    "layer_cls",
    [GradientCheckpointingLayer, MoEGradientCheckpointingLayer],
)
def test_class_default_method_is_the_recompute_default(layer_cls):
    assert layer_cls._gradient_checkpointing_method == DEFAULT_GRADIENT_CHECKPOINTING_METHOD


@pytest.mark.parametrize(
    "layer_cls",
    [_IdentityCheckpointLayer, _StubMoECheckpointLayer],
)
def test_gradient_checkpointing_enable_default(layer_cls):
    model = XorlPreTrainedModel(config=None)
    model.layer = layer_cls()
    model.gradient_checkpointing_enable()

    assert model.layer.gradient_checkpointing is True
    assert model.layer._gradient_checkpointing_method == DEFAULT_GRADIENT_CHECKPOINTING_METHOD


@pytest.mark.parametrize(
    "method",
    ["recompute_full_layer", "recompute_before_dispatch", "no_recompute"],
)
def test_enable_propagates_method_kwarg_to_every_checkpointed_layer(method):
    model = XorlPreTrainedModel(config=None)
    model.layer = _IdentityCheckpointLayer()
    model.moe_layer = _StubMoECheckpointLayer()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"gradient_checkpointing_method": method})

    assert model.layer._gradient_checkpointing_method == method
    assert model.moe_layer._gradient_checkpointing_method == method


@pytest.mark.parametrize(
    "training, flag_enabled, expect_checkpoint",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_outer_gate_fires_iff_training_and_flag(model, training, flag_enabled, expect_checkpoint):
    model.gradient_checkpointing_enable()
    model.train(training)
    model.layer.gradient_checkpointing = flag_enabled

    spy = MagicMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    model.layer._gradient_checkpointing_func = spy

    x = torch.zeros(2, 3)
    out = model.layer(x)

    assert torch.equal(out, x)
    assert spy.called is expect_checkpoint
