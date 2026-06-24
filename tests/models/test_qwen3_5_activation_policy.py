from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import xorl.models.transformers.qwen3_5.modeling_qwen3_5 as qwen3_5_dense
import xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_moe
import xorl.models.transformers.qwen3_moe.modeling_qwen3_moe as qwen3_moe


pytestmark = pytest.mark.cpu

_MISSING = object()


def _config(activation_native=_MISSING):
    config = SimpleNamespace(hidden_size=4, intermediate_size=6, hidden_act="silu")
    if activation_native is not _MISSING:
        config._activation_native = activation_native
    return config


def _moe_config(activation_native=_MISSING):
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=6,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=6,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        train_router=False,
    )
    if activation_native is not _MISSING:
        config._activation_native = activation_native
    return config


def _fill_mlp_weights(mlp: torch.nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(mlp.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param)
            param.copy_(values / (param.numel() + 3 + idx) - 0.5)


def _manual_swiglu(mlp: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x, mlp.gate_up_proj.weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.linear(mlp.act_fn(gate) * up, mlp.down_proj.weight)


@pytest.mark.parametrize(
    ("module", "mlp_cls"),
    [
        (qwen3_5_dense, qwen3_5_dense.Qwen3_5MLP),
        (qwen3_5_moe, qwen3_5_moe.Qwen3_5MoeMLP),
    ],
)
def test_qwen35_mlp_activation_native_uses_eager_swiglu(monkeypatch, module, mlp_cls):
    def fail_fused_silu(_x):
        raise AssertionError("fused_silu_and_mul should not run when _activation_native=True")

    monkeypatch.setattr(module, "fused_silu_and_mul", fail_fused_silu)

    mlp = mlp_cls(_config(activation_native=True))
    _fill_mlp_weights(mlp)
    x = torch.randn(2, 3, 4)

    assert not mlp._use_fused_silu
    torch.testing.assert_close(mlp(x), _manual_swiglu(mlp, x))


@pytest.mark.parametrize(
    ("activation_native", "expected_fused"),
    [
        (_MISSING, True),
        (False, True),
        (True, False),
    ],
)
@pytest.mark.parametrize(
    "mlp_cls",
    [
        qwen3_5_dense.Qwen3_5MLP,
        qwen3_5_moe.Qwen3_5MoeMLP,
    ],
)
def test_qwen35_mlp_fused_silu_policy_matches_activation_native(mlp_cls, activation_native, expected_fused):
    mlp = mlp_cls(_config(activation_native=activation_native))

    assert mlp._use_fused_silu is expected_fused


@pytest.mark.parametrize(
    ("block_cls", "kwargs"),
    [
        (qwen3_5_moe.Qwen3_5MoeSparseMoeBlock, {"moe_implementation": "quack"}),
        (qwen3_moe.Qwen3MoeSparseQuackMoeBlock, {}),
    ],
)
def test_qwen_moe_activation_native_reaches_routed_experts(block_cls, kwargs):
    block = block_cls(_moe_config(activation_native=True), **kwargs)

    assert block.experts.activation_native is True


@pytest.mark.parametrize(
    ("block_cls", "kwargs"),
    [
        (qwen3_5_moe.Qwen3_5MoeSparseMoeBlock, {"moe_implementation": "quack"}),
        (qwen3_moe.Qwen3MoeSparseQuackMoeBlock, {}),
    ],
)
def test_qwen_moe_routed_experts_keep_fast_activation_by_default(block_cls, kwargs):
    block = block_cls(_moe_config(), **kwargs)

    assert block.experts.activation_native is False
