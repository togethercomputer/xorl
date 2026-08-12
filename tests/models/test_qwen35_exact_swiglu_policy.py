from types import SimpleNamespace

from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP


def _config(*, exact: bool, activation_native: bool) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        hidden_act="silu",
        _activation_native=activation_native,
        _qwen35_exact_contract=exact,
    )


def test_exact_qwen35_dense_and_moe_mlp_select_one_round_swiglu():
    config = _config(exact=True, activation_native=True)

    for mlp in (Qwen3_5MLP(config), Qwen3_5MoeMLP(config)):
        assert mlp._use_fused_silu
        assert mlp._exact_one_round


def test_nonexact_qwen35_preserves_native_override():
    config = _config(exact=False, activation_native=True)

    for mlp in (Qwen3_5MLP(config), Qwen3_5MoeMLP(config)):
        assert not mlp._use_fused_silu
        assert not mlp._exact_one_round


def test_nonexact_fused_qwen35_keeps_two_round_dispatch():
    config = _config(exact=False, activation_native=False)

    for mlp in (Qwen3_5MLP(config), Qwen3_5MoeMLP(config)):
        assert mlp._use_fused_silu
        assert not mlp._exact_one_round
