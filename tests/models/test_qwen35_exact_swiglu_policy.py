from types import SimpleNamespace

from xorl.models.transformers.glm5.modeling_glm5 import Glm5MLP
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


def test_exact_qwen35_dense_and_shared_mlp_select_fp32_swiglu():
    config = _config(exact=True, activation_native=True)

    assert Qwen3_5MLP(config)._use_fused_silu
    assert Qwen3_5MoeMLP(config)._use_fused_silu


def test_exact_glm52_dense_and_shared_mlp_select_fp32_swiglu():
    config = SimpleNamespace(
        hidden_size=6144,
        intermediate_size=12288,
        hidden_act="silu",
        _activation_native=False,
    )

    assert Glm5MLP(config)._use_fused_silu
def test_nonexact_qwen35_preserves_native_override():
    config = _config(exact=False, activation_native=True)

    assert not Qwen3_5MLP(config)._use_fused_silu
    assert not Qwen3_5MoeMLP(config)._use_fused_silu
