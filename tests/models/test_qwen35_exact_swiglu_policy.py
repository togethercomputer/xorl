from types import SimpleNamespace

from xorl.models.exact_contract import exact_gdn_cp_alignment_required, resolve_exact_contract_family
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


def test_exact_qwen35_dense_and_moe_mlp_select_one_round_swiglu():
    config = _config(exact=True, activation_native=True)

    for mlp in (Qwen3_5MLP(config), Qwen3_5MoeMLP(config)):
        assert mlp._use_fused_silu
        assert mlp._exact_one_round


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

    for mlp in (Qwen3_5MLP(config), Qwen3_5MoeMLP(config)):
        assert not mlp._use_fused_silu
        assert not mlp._exact_one_round


def test_nonexact_fused_qwen35_keeps_two_round_dispatch():
    config = _config(exact=False, activation_native=False)

    for mlp in (Qwen3_5MLP(config), Qwen3_5MoeMLP(config)):
        assert mlp._use_fused_silu
        assert not mlp._exact_one_round


def test_neutral_one_round_stamp_is_the_primary_key():
    """Resolution-time ``_exact_one_round_swiglu`` wins over the legacy flag."""

    stamped_on = _config(exact=False, activation_native=False)
    stamped_on._exact_one_round_swiglu = True
    stamped_off = _config(exact=True, activation_native=False)
    stamped_off._exact_one_round_swiglu = False

    for mlp_cls in (Qwen3_5MLP, Qwen3_5MoeMLP):
        assert mlp_cls(stamped_on)._exact_one_round
        assert not mlp_cls(stamped_off)._exact_one_round


def test_unstamped_configs_fall_back_to_the_legacy_flag():
    for mlp_cls in (Qwen3_5MLP, Qwen3_5MoeMLP):
        assert mlp_cls(_config(exact=True, activation_native=False))._exact_one_round
        assert not mlp_cls(_config(exact=False, activation_native=False))._exact_one_round


def test_resolve_exact_contract_family_classification():
    assert resolve_exact_contract_family(None) is None
    assert resolve_exact_contract_family(SimpleNamespace()) is None
    assert resolve_exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True)) == "qwen3_5_dense"
    assert (
        resolve_exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True, model_type="qwen3_5_moe"))
        == "qwen3_5_moe"
    )
    assert resolve_exact_contract_family(SimpleNamespace(_qwen35_exact_contract=True, num_experts=256)) == "qwen3_5_moe"
    assert resolve_exact_contract_family(SimpleNamespace(_glm52_exact_contract=True)) == "glm52"


def test_exact_gdn_alignment_is_derived_from_resolved_qwen_layers():
    assert exact_gdn_cp_alignment_required(
        SimpleNamespace(
            _qwen35_exact_contract=True,
            layer_types=["full_attention", "linear_attention"],
        )
    )
    assert exact_gdn_cp_alignment_required(
        SimpleNamespace(
            _qwen35_exact_contract=True,
            text_config=SimpleNamespace(layer_types=["linear_attention"]),
        )
    )
    assert not exact_gdn_cp_alignment_required(
        SimpleNamespace(
            _qwen35_exact_contract=True,
            layer_types=["full_attention"],
        )
    )
    assert not exact_gdn_cp_alignment_required(
        SimpleNamespace(
            _qwen35_exact_contract=False,
            layer_types=["linear_attention"],
        )
    )
