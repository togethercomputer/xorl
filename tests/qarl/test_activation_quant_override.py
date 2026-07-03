"""CPU tests for the W4/W4A4 dual-regime activation-quant override contextmanager.

``qarl_activation_quant_override(model, enabled)`` must set
``qarl_quantize_activation = enabled`` on EVERY QARLLinear AND QARLMoEExperts in the
model, then restore each module's prior per-module value on exit (including on an
exception). This is the toggle that makes a W4-trained and a W4A4-trained model directly
comparable at eval time without touching training dynamics.
"""

import pytest
import torch.nn as nn

from xorl.models.layers.moe.experts import MoEExperts
from xorl.qarl import QARLLinear, qarl_activation_quant_override
from xorl.qarl.moe_experts import QARLMoEExperts, convert_moe_experts_to_qarl


pytestmark = pytest.mark.cpu


class TinyW4A4Model(nn.Module):
    """A model that mixes QARL modules (toggled) with plain modules (must be untouched)."""

    def __init__(self):
        super().__init__()
        # Two QARLLinears with DIFFERENT starting activation-quant flags, so a correct
        # restore must put each one back to its OWN prior value (not a single shared one).
        self.qlin_off = QARLLinear(4, 4, quantize_activation=False, quant_format="nvfp4")
        self.qlin_on = QARLLinear(4, 4, quantize_activation=True, quant_format="nvfp4")
        # A class-swapped MoE experts module (NVFP4 weight-only by default: act-quant OFF).
        experts = MoEExperts(num_experts=2, hidden_dim=4, intermediate_size=8, moe_implementation="eager")
        self.experts = convert_moe_experts_to_qarl(experts, quantize_weight=True, quantize_activation=False)
        # A plain Linear that is NOT a QARL module — must never be touched.
        self.plain = nn.Linear(4, 4)


def _qarl_modules(model):
    return [m for m in model.modules() if isinstance(m, (QARLLinear, QARLMoEExperts))]


def test_override_sets_all_qarl_modules_and_restores():
    model = TinyW4A4Model()
    mods = _qarl_modules(model)
    # Sanity: we have exactly the three QARL modules (2 linears + 1 experts).
    assert len(mods) == 3
    assert isinstance(model.experts, QARLMoEExperts)
    prior = {id(m): m.qarl_quantize_activation for m in mods}
    assert prior[id(model.qlin_off)] is False
    assert prior[id(model.qlin_on)] is True
    assert prior[id(model.experts)] is False

    # enabled=True -> every QARL module's activation quant is ON inside the block.
    with qarl_activation_quant_override(model, enabled=True):
        assert all(m.qarl_quantize_activation is True for m in mods)

    # ...and each module is restored to its OWN prior value on exit.
    assert model.qlin_off.qarl_quantize_activation is False
    assert model.qlin_on.qarl_quantize_activation is True
    assert model.experts.qarl_quantize_activation is False


def test_override_disable_sets_all_off_and_restores():
    model = TinyW4A4Model()
    mods = _qarl_modules(model)
    with qarl_activation_quant_override(model, enabled=False):
        assert all(m.qarl_quantize_activation is False for m in mods)
    # Restored to their distinct originals.
    assert model.qlin_off.qarl_quantize_activation is False
    assert model.qlin_on.qarl_quantize_activation is True
    assert model.experts.qarl_quantize_activation is False


def test_override_restores_on_exception():
    model = TinyW4A4Model()
    with pytest.raises(RuntimeError):
        with qarl_activation_quant_override(model, enabled=True):
            assert model.qlin_off.qarl_quantize_activation is True
            assert model.experts.qarl_quantize_activation is True
            raise RuntimeError("boom")
    # The finally block must still restore each module's prior value.
    assert model.qlin_off.qarl_quantize_activation is False
    assert model.qlin_on.qarl_quantize_activation is True
    assert model.experts.qarl_quantize_activation is False


def test_override_does_not_touch_non_qarl_modules():
    model = TinyW4A4Model()
    # A plain nn.Linear has no qarl_quantize_activation attribute; the override must not
    # add one (it only touches QARLLinear / QARLMoEExperts instances).
    with qarl_activation_quant_override(model, enabled=True):
        assert not hasattr(model.plain, "qarl_quantize_activation")
    assert not hasattr(model.plain, "qarl_quantize_activation")


def test_override_is_reentrant_nested_restores_inner_first():
    # Nesting (outer True, inner False) must restore the inner block to the outer's
    # value, not the original — proving the restore is per-context, not global.
    model = TinyW4A4Model()
    with qarl_activation_quant_override(model, enabled=True):
        assert model.qlin_off.qarl_quantize_activation is True
        with qarl_activation_quant_override(model, enabled=False):
            assert model.qlin_off.qarl_quantize_activation is False
        # Inner exit restores to the OUTER block's value (True), not the original (False).
        assert model.qlin_off.qarl_quantize_activation is True
    # Outer exit restores the true original.
    assert model.qlin_off.qarl_quantize_activation is False
