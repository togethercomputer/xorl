"""One-GPU training-mechanics gate for the Qwen3.5/3.6 LoRA projection topology."""

from __future__ import annotations

import pytest
import torch

from tests.e2e.e2e_utils import skip_if_gpu_count_less_than
from xorl.lora.modules.base import LoraModule
from xorl.lora.utils import (
    freeze_base_parameters,
    inject_lora_into_model,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from xorl.server.weight_sync.handler import WeightSyncHandler


pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.slow]

_RANK = 16
_TARGETS = ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _target_manifest() -> dict:
    expected_modules = []
    for projection in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"):
        expected_modules.append(
            {
                "pattern": f"model.layers.*.linear_attn.{projection}",
                "count": 1,
                "rank": _RANK,
            }
        )
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        expected_modules.append(
            {
                "pattern": f"model.layers.*.self_attn.{projection}",
                "count": 1,
                "rank": _RANK,
            }
        )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        expected_modules.append(
            {
                "pattern": f"model.layers.*.mlp.shared_expert.{projection}",
                "count": 2,
                "rank": _RANK,
            }
        )
    return {
        "schema_version": 1,
        "target_modules": _TARGETS,
        "expected_modules": expected_modules,
        "allow_unlisted": False,
    }


def _config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        decoder_sparse_step=1,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        _attn_implementation="eager",
        _moe_implementation="eager",
        pad_token_id=0,
    )


def _build_model(device: torch.device) -> Qwen3_5MoeForCausalLM:
    torch.manual_seed(1234)
    model = Qwen3_5MoeForCausalLM(_config())
    inject_lora_into_model(model, r=_RANK, lora_alpha=_RANK, target_manifest=_target_manifest())
    freeze_base_parameters(model)
    for module in model.modules():
        if isinstance(module, LoraModule):
            module.exact_merged_forward = True
    return model.to(device=device, dtype=torch.bfloat16)


def _is_objective_factor(name: str) -> bool:
    if ".mlp.shared_expert." in name:
        return True
    return ".linear_attn." in name and any(
        f".{projection}.lora_" in name for projection in ("q_proj", "k_proj", "v_proj", "g_proj")
    )


@skip_if_gpu_count_less_than(1)
def test_qwen35_lora_projection_topology_two_step_update_sync_and_reload(tmp_path, monkeypatch) -> None:
    """Run GDN + shared expert through two updates, sync folding, and reload."""
    monkeypatch.setenv("XORL_GDN_BACKEND", "fla")
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS", "0")
    device = torch.device("cuda", 0)
    model = _build_model(device).train()
    base_checkpoint = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items() if "lora_" not in name
    }
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    objective = {name: parameter for name, parameter in trainable.items() if _is_objective_factor(name)}
    assert len(objective) == 20

    parameters = dict(model.named_parameters())
    frozen_names = (
        "model.layers.0.linear_attn.q_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
    )
    frozen_before = {name: parameters[name].detach().clone() for name in frozen_names}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
    input_ids = torch.randint(1, model.config.vocab_size, (2, 16), device=device)

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, use_cache=False).last_hidden_state.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        if step == 1:
            missing = [name for name, parameter in objective.items() if parameter.grad is None]
            zero = [
                name
                for name, parameter in objective.items()
                if parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
            ]
            assert not missing
            assert not zero
        optimizer.step()

    assert len(optimizer.state) == len(trainable)
    assert all(torch.equal(parameters[name].detach(), frozen_before[name]) for name in frozen_names)

    class _FakeDTensor:
        pass

    layer = model.model.layers[0]
    sync = dict(
        WeightSyncHandler._extract_params_for_sync(
            layer,
            "model.layers.0",
            _FakeDTensor,
            skip_moe_prefixes={"mlp.experts"},
        )
    )
    gdn = layer.linear_attn
    shared = layer.mlp.shared_expert
    assert torch.equal(sync["model.layers.0.linear_attn.q_proj.weight"], gdn.q_proj._merged_weight())
    gate, up = shared._gate_up_weights_for_forward()
    assert torch.equal(
        sync["model.layers.0.mlp.shared_expert.gate_up_proj.weight"],
        torch.cat((gate, up), dim=0),
    )
    assert torch.equal(
        sync["model.layers.0.mlp.shared_expert.down_proj.weight"],
        shared.down_proj._merged_weight(),
    )

    save_lora_checkpoint(
        model,
        str(tmp_path),
        base_model_name="tiny-qwen35-hybrid",
        r=_RANK,
        lora_alpha=_RANK,
        preserve_lora_dtype=True,
    )
    restored = _build_model(device).eval()
    incompatible = restored.load_state_dict(base_checkpoint, strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys and all("lora_" in name for name in incompatible.missing_keys)
    load_lora_checkpoint(restored, str(tmp_path), strict=True)
    source_state = {name: value.detach() for name, value in model.state_dict().items() if _is_objective_factor(name)}
    restored_state = {
        name: value.detach() for name, value in restored.state_dict().items() if _is_objective_factor(name)
    }
    assert source_state.keys() == restored_state.keys()
    assert all(torch.equal(source_state[name], restored_state[name]) for name in source_state)

    model.eval()
    with torch.no_grad():
        source_output = model(input_ids=input_ids, use_cache=False).last_hidden_state
        restored_output = restored(input_ids=input_ids, use_cache=False).last_hidden_state
    assert torch.equal(source_output, restored_output)
