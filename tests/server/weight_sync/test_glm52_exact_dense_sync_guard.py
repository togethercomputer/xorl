import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import nn

from xorl.lora.utils import get_lora_state_dict, save_lora_checkpoint
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.exact_lm_head_qlora import (
    Glm52ExactTP16LmHeadLoraLinear,
    Glm52ExactTP16LmHeadSelectedLogprob,
    glm52_lm_head_shard,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.server.protocol.operations import SyncWeightsData
from xorl.server.weight_sync.handler import WeightSyncHandler


def _module() -> Glm52ExactTP1DenseMLP:
    return Glm52ExactTP1DenseMLP(8, 128, device=torch.device("cpu"))


def _ordinary_exact_projection() -> Glm52ExactTP1BlockFP8QLoRALinear:
    return Glm52ExactTP1BlockFP8QLoRALinear(128, 128, device=torch.device("cpu"))


def _exact_lm_head_model() -> nn.Module:
    model = nn.Module()
    lm_head = Glm52ExactTP16LmHeadLoraLinear(8, 16, r=1, lora_alpha=1, device=torch.device("cpu"))
    shard = glm52_lm_head_shard(0)
    lm_head._glm52_exact_selected_logprob = Glm52ExactTP16LmHeadSelectedLogprob(
        tp_rank=shard.tp_rank,
        vocab_start=shard.vocab_start,
        vocab_end=shard.vocab_end,
        padded_vocab_start=shard.padded_vocab_start,
        padded_vocab_end=shard.padded_vocab_end,
    )
    model.lm_head = lm_head
    return model


def _reject_if_called(name: str):
    def reject(*args, **kwargs):
        del args, kwargs
        pytest.fail(f"{name} must not run before the exact active-LoRA sync preflight")

    return reject


def _assert_factor_only_sync_guards(module: nn.Module, prefix: str) -> None:
    handler = WeightSyncHandler.__new__(WeightSyncHandler)
    handler.rank = 0
    handler.trainer = SimpleNamespace(model=module, adapter_manager=object())

    with pytest.raises(RuntimeError, match="factor-only adapter synchronization"):
        handler._prepare_lora_adapter_for_sync("policy")
    with pytest.raises(RuntimeError, match="cannot enter QLoRA merged-weight collectives"):
        handler._qlora_collective_ops(module, prefix, collect_results=True)
    with pytest.raises(RuntimeError, match="cannot be extracted by legacy merged-weight sync"):
        WeightSyncHandler._extract_params_for_sync(module, prefix, object)


def test_exact_factor_only_sync_and_publication_policy(monkeypatch, tmp_path) -> None:
    for module, prefix in (
        (_module(), "model.layers.0.mlp"),
        (_ordinary_exact_projection(), "model.layers.0.self_attn.q_a_proj"),
    ):
        _assert_factor_only_sync_guards(module, prefix)

    payloads = (
        (SyncWeightsData(), "factor-only adapter publication"),
        (
            SyncWeightsData(
                sync_method="sparse_delta",
                sparse_delta_paths=["exact-lm-head-delta.packed"],
            ),
            "including prepacked sparse-delta sync",
        ),
    )
    for payload, error in payloads:
        handler = WeightSyncHandler.__new__(WeightSyncHandler)
        handler.rank = 0
        handler.trainer = SimpleNamespace(model=_exact_lm_head_model(), adapter_manager=None)
        monkeypatch.setattr(handler, "_prepare_lora_adapter_for_sync", _reject_if_called("adapter preparation"))
        monkeypatch.setattr(handler, "_sync_weights", _reject_if_called("streaming weight sync"))
        monkeypatch.setattr(handler, "_sync_sparse_delta_paths", _reject_if_called("sparse-delta sync"))

        with pytest.raises(RuntimeError, match=error):
            asyncio.run(handler.handle_sync_inference_weights({"payload": payload}))

    _assert_exact_lm_head_publication_preserves_separate_factor_bytes(tmp_path)


def _assert_exact_lm_head_publication_preserves_separate_factor_bytes(tmp_path) -> None:
    model = _exact_lm_head_model()
    with torch.no_grad():
        model.lm_head.lora_A.copy_(torch.arange(model.lm_head.lora_A.numel()).reshape_as(model.lm_head.lora_A))
        model.lm_head.lora_B.copy_(torch.arange(model.lm_head.lora_B.numel()).reshape_as(model.lm_head.lora_B))

    logical_factors = get_lora_state_dict(model)
    assert set(logical_factors) == {"lm_head.lora_A", "lm_head.lora_B"}

    checkpoint = tmp_path / "adapter"
    save_lora_checkpoint(
        model=model,
        save_path=str(checkpoint),
        target_modules=["lm_head"],
        r=1,
        lora_alpha=1,
        lora_state_dict=logical_factors,
    )

    published = load_safetensors_file(str(checkpoint / "adapter_model.safetensors"))
    assert set(published) == {
        "base_model.model.lm_head.lora_embedding_A",
        "base_model.model.lm_head.lora_embedding_B",
    }
    assert torch.equal(
        published["base_model.model.lm_head.lora_embedding_A"].view(torch.uint8),
        logical_factors["lm_head.lora_A"].to(torch.bfloat16).view(torch.uint8),
    )
    assert torch.equal(
        published["base_model.model.lm_head.lora_embedding_B"].view(torch.uint8),
        logical_factors["lm_head.lora_B"].to(torch.bfloat16).view(torch.uint8),
    )
    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text())
    assert adapter_config["target_modules"] == ["lm_head"]
    assert adapter_config["r"] == adapter_config["lora_alpha"] == 1
