"""Structural gates for generic expert QLoRA ownership and target selection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.lora.expert_adapter_contract import ExpertAdapterFactorOwnership
from xorl.models.layers.moe.backend import DEEPEP_AVAILABLE, expert_adapter_backend_contract
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.layers.moe.lora import MoEExpertsLoRA, MoELoRAConfig, inject_lora_into_experts
from xorl.qlora.modules.moe_experts import NF4QLoRAMoeExperts, QLoRAMoeExperts
from xorl.qlora.utils import inject_qlora_into_model


pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("backend", ("eager", "triton", "native", "quack"))
def test_registered_backends_declare_local_ep_and_zero_token_execution(backend):
    contract = expert_adapter_backend_contract(backend)
    assert contract.supports_local
    assert contract.supports_ep
    assert "alltoall" in contract.supported_dispatch_methods
    assert ("deepep" in contract.supported_dispatch_methods) is DEEPEP_AVAILABLE
    assert contract.gradient_reduction_domain.value == "ep_sum"
    assert contract.zero_token_gradient_behavior.value == "structural_zero"


def test_inactive_optional_backend_capabilities_do_not_change_plan_identity():
    module = NF4QLoRAMoeExperts(
        num_local_experts=2,
        num_experts=2,
        intermediate_size=64,
        hidden_size=64,
        r=2,
        lora_alpha=2,
        device=torch.device("cpu"),
        moe_implementation="triton",
    )
    contract = module.expert_adapter_gradient_contract
    expanded_backend = replace(
        contract.backend,
        supported_dispatch_methods=(*contract.backend.supported_dispatch_methods, "future_optional_dispatch"),
    )

    assert replace(contract, backend=expanded_backend).config_guard_fields() == contract.config_guard_fields()


def test_quantized_expert_contract_preserves_requested_projection_subset():
    module = NF4QLoRAMoeExperts(
        num_local_experts=2,
        num_experts=4,
        intermediate_size=8,
        hidden_size=8,
        r=2,
        lora_alpha=2,
        device=torch.device("cpu"),
        moe_implementation="triton",
        target_modules=["down_proj"],
    )

    assert set(dict(module.named_parameters())) == {"down_proj_lora_A", "down_proj_lora_B"}
    assert {name for name, _buffer in module.named_buffers() if "lora" in name} == {
        "gate_proj_lora_A",
        "gate_proj_lora_B",
        "up_proj_lora_A",
        "up_proj_lora_B",
    }
    assert not (
        {"gate_proj_lora_A", "gate_proj_lora_B", "up_proj_lora_A", "up_proj_lora_B"} & module.state_dict().keys()
    )
    assert module.gate_proj_lora_A.shape[0] == 1
    assert module.gate_proj_lora_B.shape[0] == module.num_local_experts
    assert module.up_proj_lora_A.shape[0] == 1
    assert module.up_proj_lora_B.shape[0] == module.num_local_experts
    contract = module.expert_adapter_gradient_contract
    assert contract.projection_roles == ("down_proj",)
    assert dict(contract.factor_ownership) == {
        "down_proj_lora_A": ExpertAdapterFactorOwnership.OWNER_SHARDED,
        "down_proj_lora_B": ExpertAdapterFactorOwnership.EP_REPLICATED,
    }
    assert {name: domain.value for name, domain in module._ep_gradient_reduction_by_parameter.items()} == {
        "down_proj_lora_A": "none",
        "down_proj_lora_B": "ep_sum",
    }


def test_unquantized_expert_contract_omits_structural_zero_factors_from_checkpoints():
    module = MoEExpertsLoRA(
        num_experts=4,
        hidden_dim=8,
        intermediate_size=8,
        lora_config=MoELoRAConfig(r=2, lora_alpha=2, target_modules=["down_proj"]),
    )

    assert set(dict(module.named_parameters())) == {"gate_up_proj", "down_proj", "down_proj_lora_A", "down_proj_lora_B"}
    assert not (
        {"gate_proj_lora_A", "gate_proj_lora_B", "up_proj_lora_A", "up_proj_lora_B"} & module.state_dict().keys()
    )


def test_quantized_eager_contract_is_local_only_until_ep_is_certified():
    module = NF4QLoRAMoeExperts(
        num_local_experts=2,
        num_experts=2,
        intermediate_size=64,
        hidden_size=64,
        r=2,
        lora_alpha=2,
        device=torch.device("cpu"),
        moe_implementation="eager",
    )

    contract = module.expert_adapter_gradient_contract
    assert contract.backend.supports_local
    assert not contract.backend.supports_ep
    assert contract.backend.supported_dispatch_methods == ()


def test_generic_quantized_contract_rejects_non_silu_semantics():
    module = NF4QLoRAMoeExperts(
        num_local_experts=2,
        num_experts=2,
        intermediate_size=64,
        hidden_size=64,
        r=2,
        lora_alpha=2,
        device=torch.device("cpu"),
        act_fn=nn.ReLU(),
        hidden_act="relu",
    )

    with pytest.raises(ValueError, match="cannot preserve"):
        _ = module.expert_adapter_gradient_contract


def test_quantized_contract_accepts_xorl_standard_silu_source_module():
    source = MoEExperts(
        num_experts=2,
        hidden_dim=8,
        intermediate_size=8,
        moe_implementation="triton",
    )

    wrapped = QLoRAMoeExperts.from_module(
        source,
        r=2,
        lora_alpha=2,
        quant_format="nvfp4",
        target_modules=["gate_proj", "down_proj"],
    )

    assert wrapped.hidden_act == "silu"
    assert wrapped.expert_adapter_gradient_contract.projection_roles == ("gate_proj", "down_proj")


@pytest.mark.parametrize("unsupported_semantics", ("non_gated", "clamped", "bias", "non_silu"))
def test_generic_unquantized_wrapper_rejects_unpreserved_expert_semantics(unsupported_semantics):
    if unsupported_semantics == "non_gated":
        source = MoEExperts(
            num_experts=2,
            hidden_dim=8,
            intermediate_size=8,
            hidden_act="relu2",
            moe_implementation="native",
            gated=False,
        )
    else:
        source = MoEExperts(
            num_experts=2,
            hidden_dim=8,
            intermediate_size=8,
            moe_implementation="native",
        )
        if unsupported_semantics == "clamped":
            source.hidden_act = "clamped_swiglu"
            source.swiglu_limit = 7.0
        elif unsupported_semantics == "bias":
            source.gate_up_bias = nn.Parameter(torch.zeros(2, 16))
        else:
            source.hidden_act = "gelu_tanh"

    with pytest.raises(NotImplementedError, match="cannot preserve"):
        MoEExpertsLoRA.from_module(source, r=2, lora_alpha=2)


def test_generic_unquantized_wrapper_preserves_standard_silu_with_native_activation_flag():
    source = MoEExperts(
        num_experts=2,
        hidden_dim=8,
        intermediate_size=8,
        moe_implementation="native",
        activation_native=True,
    )

    wrapped = MoEExpertsLoRA.from_module(source, r=2, lora_alpha=2)

    assert wrapped.hidden_act == "silu"
    assert wrapped.moe_implementation == "native"


def test_injection_passes_exact_expert_target_subset(monkeypatch):
    class _Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Parameter(torch.empty(4, 8, 8, device="meta"), requires_grad=False)
            self.up_proj = nn.Parameter(torch.empty(4, 8, 8, device="meta"), requires_grad=False)
            self.down_proj = nn.Parameter(torch.empty(4, 8, 8, device="meta"), requires_grad=False)
            self.hidden_dim = 8
            self.intermediate_size = 8
            self.moe_implementation = "triton"
            self.ep_dispatch = "alltoall"
            self.gated = True
            self.hidden_act = "silu"
            self.swiglu_limit = 0.0
            self.gate_up_bias = None
            self.down_bias = None

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4, bias=False, device="meta")
            self.experts = _Experts()

    model = nn.Module()
    model.block = _Block()
    monkeypatch.setattr(
        "xorl.qlora.utils.get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=False, ep_size=1, ep_rank=0),
    )

    inject_qlora_into_model(
        model,
        r=2,
        lora_alpha=2,
        quant_format="nf4",
        quant_group_size=64,
        target_modules=["down_proj"],
    )

    assert isinstance(model.block.experts, QLoRAMoeExperts)
    assert model.block.experts.target_modules == ("down_proj",)
    assert set(dict(model.block.experts.named_parameters())) == {"down_proj_lora_A", "down_proj_lora_B"}


def test_glm4_generic_qlora_construction_and_checkpoint_buffer_are_model_independent(monkeypatch):
    from xorl.models.transformers.glm4_moe import modeling_glm4_moe  # noqa: PLC0415
    from xorl.models.transformers.glm4_moe.configuration_glm4_moe import Glm4MoeConfig  # noqa: PLC0415
    from xorl.models.transformers.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM  # noqa: PLC0415

    config = Glm4MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        moe_intermediate_size=8,
        num_experts_per_tok=2,
        n_shared_experts=0,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        _moe_implementation="triton",
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    config._ep_dispatch = "alltoall"
    model = Glm4MoeForCausalLM(config)

    inject_qlora_into_model(
        model,
        r=2,
        lora_alpha=2,
        quant_format="nvfp4",
        quant_group_size=16,
        target_modules=["gate_proj", "down_proj"],
    )

    experts = model.model.layers[0].mlp.experts
    assert experts.target_modules == ("gate_proj", "down_proj")
    assert experts.expert_adapter_gradient_contract.quantized_base_format == "nvfp4"
    monkeypatch.setattr(modeling_glm4_moe, "detect_prequantized_checkpoint", lambda _path: True)
    handler = model.get_checkpoint_handler(
        checkpoint_keys=set(),
        weights_path="synthetic-prequantized-checkpoint",
        ep_rank=0,
        ep_size=1,
        is_broadcast=False,
    )
    assert handler._qlora_expert_buffer is not None


def test_qwen3_moe_quack_nf4_construction_restores_exact_expert_targets(monkeypatch):
    from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig  # noqa: PLC0415
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM  # noqa: PLC0415

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        _moe_implementation="quack",
        hidden_act="silu",
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    config._ep_dispatch = "alltoall"
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    monkeypatch.setattr(
        "xorl.qlora.utils.get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=False, ep_size=1, ep_rank=0),
    )

    inject_qlora_into_model(
        model,
        r=2,
        lora_alpha=2,
        quant_format="nf4",
        quant_group_size=64,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )

    experts = model.model.layers[0].mlp.experts
    assert isinstance(experts, QLoRAMoeExperts)
    assert experts.moe_implementation == "quack"
    assert experts.target_modules == ("gate_proj", "up_proj", "down_proj")
    assert experts.quant_group_size == 64
    assert set(dict(experts.named_parameters())) == {
        f"{projection}_lora_{suffix}" for projection in ("gate_proj", "up_proj", "down_proj") for suffix in ("A", "B")
    }
    assert experts._source_fqn == "model.layers.0.mlp.experts"


def test_qwen35_moe_quack_lora_construction_preserves_expert_semantics():
    from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeConfig,
    )
    from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeSparseMoeBlock,
    )

    config = Qwen3_5MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        _moe_implementation="quack",
    )
    with torch.device("meta"):
        block = Qwen3_5MoeSparseMoeBlock(config, moe_implementation="quack", layer_idx=0)

    inject_lora_into_experts(
        block,
        r=2,
        lora_alpha=2,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        hybrid_shared=False,
    )

    experts = block.experts
    assert isinstance(experts, MoEExpertsLoRA)
    assert experts.moe_implementation == "quack"
    assert experts.hidden_act == "silu"
    assert experts.lora_config.hybrid_shared is False
    assert experts.expert_adapter_gradient_contract.projection_roles == (
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_quantized_expert_construction_rejects_ignored_group_size():
    source = MoEExperts(
        num_experts=2,
        hidden_dim=64,
        intermediate_size=64,
        moe_implementation="triton",
    )

    with pytest.raises(ValueError, match="requires quant_group_size=64"):
        QLoRAMoeExperts.from_module(
            source,
            r=2,
            lora_alpha=2,
            quant_format="nf4",
            quant_group_size=32,
        )


@pytest.mark.parametrize("family", ("minimax", "gpt_oss", "nemotron"))
def test_model_specific_expert_semantics_remain_explicitly_fail_closed(family):
    if family == "minimax":
        from xorl.models.transformers.minimax_m3.configuration_minimax_m3 import (  # noqa: PLC0415
            MiniMaxM3Config,
        )
        from xorl.models.transformers.minimax_m3.modeling_minimax_m3 import (  # noqa: PLC0415
            MiniMaxM3SparseMoeBlock,
        )

        config = MiniMaxM3Config(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            dense_intermediate_size=16,
            intermediate_size=8,
            shared_intermediate_size=8,
            num_local_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=0,
            moe_layer_freq=[1],
            _moe_implementation="native",
        )
        block = MiniMaxM3SparseMoeBlock(config, moe_implementation="native")
    elif family == "gpt_oss":
        from xorl.models.transformers.gpt_oss.configuration_gpt_oss import GptOssConfig  # noqa: PLC0415
        from xorl.models.transformers.gpt_oss.modeling_gpt_oss import GptOssMoEBlock  # noqa: PLC0415

        config = GptOssConfig(
            vocab_size=32,
            hidden_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            num_experts=2,
            num_experts_per_tok=1,
            _moe_implementation="native",
        )
        block = GptOssMoEBlock(config, moe_implementation="native")
    else:
        from xorl.models.transformers.nemotron_h.configuration_nemotron_h import (  # noqa: PLC0415
            NemotronHConfig,
        )
        from xorl.models.transformers.nemotron_h.modeling_nemotron_h import NemotronHMoE  # noqa: PLC0415

        config = NemotronHConfig(
            vocab_size=32,
            hidden_size=8,
            layers_block_type=["moe"],
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=16,
            n_routed_experts=2,
            n_shared_experts=0,
            moe_intermediate_size=8,
            moe_shared_expert_intermediate_size=8,
            num_experts_per_tok=1,
            _moe_implementation="native",
        )
        block = NemotronHMoE(config)

    with pytest.raises(NotImplementedError, match="cannot preserve"):
        inject_lora_into_experts(block, r=2, lora_alpha=2)
    with pytest.raises(NotImplementedError, match="cannot preserve"):
        QLoRAMoeExperts.from_module(
            block.experts,
            r=2,
            lora_alpha=2,
            quant_format="nf4",
            quant_group_size=64,
        )


@pytest.mark.parametrize(
    "targets",
    ([], ["router"], ["gate_proj", "router"], ["gate_proj", "gate_proj"]),
)
def test_quantized_expert_targets_fail_closed(targets):
    with pytest.raises(ValueError, match="target_modules"):
        NF4QLoRAMoeExperts(
            num_local_experts=2,
            num_experts=4,
            intermediate_size=8,
            hidden_size=8,
            r=2,
            lora_alpha=2,
            device=torch.device("cpu"),
            target_modules=targets,
        )


@pytest.mark.parametrize("targets", ([], ["router"], ["gate_proj", "gate_proj"]))
def test_unquantized_expert_targets_fail_closed(targets):
    with pytest.raises(ValueError, match="target_modules"):
        MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=8,
            intermediate_size=8,
            lora_config=MoELoRAConfig(target_modules=targets),
        )
