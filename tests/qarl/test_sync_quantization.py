import asyncio
from types import SimpleNamespace

import pytest
import torch.nn as nn

from xorl.qarl import inject_qarl_into_model, qarl_sync_quantization_config
from xorl.server.protocol.operations import SyncWeightsData
from xorl.server.weight_sync.handler import WeightSyncHandler


pytestmark = pytest.mark.cpu


class TinyQARLSyncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.embed_tokens = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_qarl_sync_quantization_config_quantizes_only_qarl_modules():
    model = TinyQARLSyncModel()
    inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [2, 2]},
        target_modules=["proj"],
    )

    config = qarl_sync_quantization_config(model)

    assert config is not None
    assert config["quant_method"] == "fp8"
    assert config["weight_block_size"] == [2, 2]
    assert set(config["modules_to_not_convert"]) == {"embed_tokens", "lm_head"}
    assert config["xorl_qarl_sync"] == {
        "enabled": True,
        "folded_modules": ["proj"],
        "source": "qarl_fake_quant",
    }

    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [
            ("embed_tokens.weight", model.embed_tokens.weight.detach()),
            ("proj.weight", model.proj.weight.detach()),
            ("lm_head.weight", model.lm_head.weight.detach()),
        ],
        quantization_config=config,
    )

    names = [name for name, _tensor in out]
    assert "embed_tokens.weight" in names
    assert "lm_head.weight" in names
    assert "proj.weight" in names
    assert "proj.weight_scale_inv" in names
    assert "embed_tokens.weight_scale_inv" not in names
    assert "lm_head.weight_scale_inv" not in names


def test_qarl_sync_quantization_rejects_mismatched_explicit_block_size():
    model = TinyQARLSyncModel()
    inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [2, 2]},
        target_modules=["proj"],
    )

    with pytest.raises(ValueError, match="must match wrapped modules"):
        qarl_sync_quantization_config(model, {"quant_method": "fp8", "weight_block_size": [4, 4]})


def test_weight_sync_handler_derives_qarl_sync_quantization(monkeypatch):
    model = TinyQARLSyncModel()
    inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [2, 2]},
        target_modules=["proj"],
    )
    trainer = SimpleNamespace(model=model, train_config={})
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)
    captured = {}

    def fake_sync_weights(**kwargs):
        captured["quantization"] = kwargs["quantization"]
        return {"success": True}

    monkeypatch.setattr(handler, "_sync_weights", fake_sync_weights)

    result = asyncio.run(handler.handle_sync_inference_weights({"payload": SyncWeightsData(quantization=None)}))

    assert result["success"] is True
    assert captured["quantization"]["quant_method"] == "fp8"
    assert captured["quantization"]["weight_block_size"] == [2, 2]
    assert set(captured["quantization"]["modules_to_not_convert"]) == {"embed_tokens", "lm_head"}


def test_weight_sync_handler_reports_bad_qarl_sync_quantization(monkeypatch):
    model = TinyQARLSyncModel()
    inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [2, 2]},
        target_modules=["proj"],
    )
    trainer = SimpleNamespace(model=model, train_config={})
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)

    result = asyncio.run(
        handler.handle_sync_inference_weights(
            {"payload": SyncWeightsData(quantization={"quant_method": "fp8", "weight_block_size": [4, 4]})}
        )
    )

    assert result["success"] is False
    assert "Failed to resolve QARL sync quantization" in result["message"]
