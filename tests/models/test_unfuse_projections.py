"""Unfusing ``qkv_proj`` / ``gate_up_proj``: checkpoint handlers and the MoE shared expert.

The load-time contract when a model is unfused is narrow: the checkpoint handler stops
merging the projections the model no longer fuses, and keeps doing everything else. The
dense handlers used to express this by returning no handler at all, which on Qwen3.5 also
switched off the GatedDeltaNet remapping.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.models.layers.moe import MoEBlock
from xorl.models.transformers.qwen3.checkpoint_handler import Qwen3CheckpointHandler
from xorl.models.transformers.qwen3_5 import parallelize as qwen3_5_parallelize
from xorl.models.transformers.qwen3_5.checkpoint_handler import Qwen3_5CheckpointHandler
from xorl.models.transformers.qwen3_5_moe import parallelize as qwen3_5_moe_parallelize


pytestmark = [pytest.mark.cpu]


GATE_KEY = "model.layers.0.mlp.gate_proj.weight"
UP_KEY = "model.layers.0.mlp.up_proj.weight"
Q_KEY = "model.layers.0.self_attn.q_proj.weight"
K_KEY = "model.layers.0.self_attn.k_proj.weight"
V_KEY = "model.layers.0.self_attn.v_proj.weight"

LINEAR_KEY_DIM = 6
LINEAR_VALUE_DIM = 10


def _qwen3_handler(**kwargs) -> Qwen3CheckpointHandler:
    return Qwen3CheckpointHandler(num_attention_heads=4, num_key_value_heads=2, head_dim=8, **kwargs)


def _qwen3_5_handler(**kwargs) -> Qwen3_5CheckpointHandler:
    return Qwen3_5CheckpointHandler(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_dim=LINEAR_KEY_DIM,
        linear_value_dim=LINEAR_VALUE_DIM,
        **kwargs,
    )


def _assert_passthrough(result, key, tensor) -> None:
    [(name, passed)] = result
    assert name == key
    assert passed is tensor


class TestQwen3HandlerSkipFlags:
    def test_merges_gate_and_up_by_default(self):
        handler = _qwen3_handler()

        assert handler.on_load_weight(GATE_KEY, torch.ones(3, 2)) == []
        [(name, merged)] = handler.on_load_weight(UP_KEY, torch.zeros(3, 2))

        assert name == "model.layers.0.mlp.gate_up_proj.weight"
        assert merged.shape == (6, 2)

    def test_merges_qkv_by_default(self):
        handler = _qwen3_handler()

        assert handler.on_load_weight(Q_KEY, torch.ones(32, 2)) == []
        assert handler.on_load_weight(K_KEY, torch.ones(16, 2)) == []
        [(name, _merged)] = handler.on_load_weight(V_KEY, torch.ones(16, 2))

        assert name == "model.layers.0.self_attn.qkv_proj.weight"

    def test_gate_and_up_pass_through_when_skipped(self):
        handler = _qwen3_handler(skip_gate_up_merge=True)
        gate = torch.ones(3, 2)

        _assert_passthrough(handler.on_load_weight(GATE_KEY, gate), GATE_KEY, gate)

    def test_qkv_passes_through_when_skipped(self):
        handler = _qwen3_handler(skip_qkv_merge=True)
        q = torch.ones(32, 2)

        _assert_passthrough(handler.on_load_weight(Q_KEY, q), Q_KEY, q)

    def test_skipping_one_merge_leaves_the_other_active(self):
        """The flags are independent: an architecture can fuse one and not the other."""
        handler = _qwen3_handler(skip_qkv_merge=True)
        q = torch.ones(32, 2)

        _assert_passthrough(handler.on_load_weight(Q_KEY, q), Q_KEY, q)
        assert handler.on_load_weight(GATE_KEY, torch.ones(3, 2)) == []
        [(name, _merged)] = handler.on_load_weight(UP_KEY, torch.zeros(3, 2))
        assert name == "model.layers.0.mlp.gate_up_proj.weight"

    def test_no_pending_warning_when_merges_are_skipped(self):
        """A skipped merge must not report the keys it deliberately never buffered."""
        handler = _qwen3_handler(skip_qkv_merge=True, skip_gate_up_merge=True)
        handler.on_load_weight(GATE_KEY, torch.ones(3, 2))
        handler.on_load_weight(Q_KEY, torch.ones(32, 2))

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            handler.on_load_complete()

        assert [w for w in recorded if "Incomplete" in str(w.message)] == []


class TestQwen3_5HandlerKeepsLinearAttentionMapping:
    """The regression the granular flags exist to prevent.

    ``Qwen3_5CheckpointHandler`` also splits the GatedDeltaNet ``in_proj_qkv`` packing,
    which has nothing to do with how the MLP is stored. Skipping the gate/up merge must
    not take that with it.
    """

    def test_in_proj_qkv_still_splits_when_gate_up_merge_is_skipped(self):
        handler = _qwen3_5_handler(skip_gate_up_merge=True)
        rows = 2 * LINEAR_KEY_DIM + LINEAR_VALUE_DIM
        tensor = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)

        result = handler.on_load_weight("model.layers.0.linear_attn.in_proj_qkv.weight", tensor)

        assert [name for name, _ in result] == [
            "model.layers.0.linear_attn.q_proj.weight",
            "model.layers.0.linear_attn.k_proj.weight",
            "model.layers.0.linear_attn.v_proj.weight",
        ]
        (_, q), (_, k), (_, v) = result
        assert torch.equal(q, tensor[:LINEAR_KEY_DIM])
        assert torch.equal(k, tensor[LINEAR_KEY_DIM : 2 * LINEAR_KEY_DIM])
        assert torch.equal(v, tensor[2 * LINEAR_KEY_DIM :])

    def test_gate_and_up_pass_through_when_skipped(self):
        handler = _qwen3_5_handler(skip_gate_up_merge=True)
        gate = torch.ones(3, 2)

        _assert_passthrough(handler.on_load_weight(GATE_KEY, gate), GATE_KEY, gate)

    def test_gate_and_up_still_merge_by_default(self):
        handler = _qwen3_5_handler()

        assert handler.on_load_weight(GATE_KEY, torch.ones(3, 2)) == []
        [(name, _merged)] = handler.on_load_weight(UP_KEY, torch.zeros(3, 2))

        assert name == "model.layers.0.mlp.gate_up_proj.weight"


class _StubFusedMLP(nn.Module):
    """Stands in for ``Qwen3_5MoeMLP`` / ``Qwen3_5MLP``: fused until unfused."""

    def __init__(self, hidden: int = 4, intermediate: int = 4):
        super().__init__()
        self.hidden = hidden
        self.intermediate = intermediate
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def unfuse_for_tp(self):
        self.gate_proj = nn.Linear(self.hidden, self.intermediate, bias=False)
        self.up_proj = nn.Linear(self.hidden, self.intermediate, bias=False)
        del self.gate_up_proj

    @property
    def is_unfused(self) -> bool:
        return not hasattr(self, "gate_up_proj")


def _moe_block(shared_expert) -> MoEBlock:
    """A real ``MoEBlock`` instance for ``isinstance`` without its config-driven init.

    ``Qwen3_5MoeSparseMoeBlock.__init__`` needs a full model config and a MoE backend;
    only the ``isinstance`` dispatch and the ``shared_expert`` attribute matter here.
    """
    block = MoEBlock.__new__(MoEBlock)
    nn.Module.__init__(block)
    block.experts = nn.Linear(4, 4, bias=False)
    block.shared_expert = shared_expert
    return block


class _Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.self_attn = None
        self.mlp = mlp


class _Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.config = SimpleNamespace(base_model_tp_plan=None)


class TestSharedExpertUnfusing:
    def test_moe_block_shared_expert_unfuses(self):
        shared = _StubFusedMLP()
        model = _Model([_Layer(_moe_block(shared))])

        qwen3_5_moe_parallelize.unfuse_for_tp(model)

        assert shared.is_unfused
        assert model._unfused_for_tp is True

    def test_routed_experts_are_left_fused(self):
        """Routed expert weights are EP-sharded, not TP-sharded — they stay as they are."""
        block = _moe_block(_StubFusedMLP())
        model = _Model([_Layer(block)])

        qwen3_5_moe_parallelize.unfuse_for_tp(model)

        assert isinstance(block.experts, nn.Linear)

    def test_dense_layers_still_unfuse(self):
        dense = _StubFusedMLP()
        model = _Model([_Layer(dense)])

        qwen3_5_moe_parallelize.unfuse_for_tp(model)

        assert dense.is_unfused

    def test_mixed_stack_unfuses_both_dense_and_shared(self):
        shared = _StubFusedMLP()
        dense = _StubFusedMLP()
        model = _Model([_Layer(_moe_block(shared)), _Layer(dense)])

        qwen3_5_moe_parallelize.unfuse_for_tp(model)

        assert shared.is_unfused
        assert dense.is_unfused


class TestDenseQwen3_5Unfusing:
    def test_layers_without_self_attn_are_tolerated(self):
        """Qwen3.5's linear-attention layers carry no ``self_attn``."""
        dense = _StubFusedMLP()
        model = _Model([_Layer(dense)])

        qwen3_5_parallelize.unfuse_for_tp(model)

        assert dense.is_unfused
        assert model._unfused_for_tp is True
