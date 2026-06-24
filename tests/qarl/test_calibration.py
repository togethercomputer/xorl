import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.qarl import QARLLinear, calibrate_qarl_model, inject_qarl_into_model, load_qarl_calibration_batches


pytestmark = pytest.mark.cpu


class TinyCalibrationModel(nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = F.silu(self.proj(self.embed_tokens(input_ids)))
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.lm_head(hidden)


def test_load_qarl_calibration_batches_from_json_truncates_and_limits(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
                    {"input_ids": [4, 3, 2, 1], "attention_mask": [1, 1, 1, 0]},
                ]
            }
        ),
        encoding="utf-8",
    )

    batches = load_qarl_calibration_batches(path, calibration_size=1, sequence_length=3)

    assert len(batches) == 1
    assert batches[0]["input_ids"].shape == (1, 3)
    assert batches[0]["input_ids"].tolist() == [[1, 2, 3]]
    assert batches[0]["attention_mask"].tolist() == [[1, 1, 1]]


def test_calibrate_qarl_model_populates_persistent_metadata(tmp_path):
    path = tmp_path / "calib.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"input_ids": [1, 2, 3, 4]}),
                json.dumps({"input_ids": [4, 3, 2, 1]}),
            ]
        ),
        encoding="utf-8",
    )
    model = TinyCalibrationModel()
    inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [4, 4]},
        target_modules=["proj", "lm_head"],
    )

    summary = calibrate_qarl_model(model, path, calibration_size=2, sequence_length=3)

    assert summary["calibration_batches"] == 2
    assert summary["calibration_samples"] == 2
    assert summary["linear_count"] == 2
    assert summary["forward_counts"]["proj"] == 2
    assert model.proj.qarl_input_amax.item() > 0
    assert model.proj.qarl_weight_scale_inv.shape == (2, 2)
    assert model.proj.qarl_weight_scale_inv.abs().max().item() > 0

    restored = TinyCalibrationModel()
    inject_qarl_into_model(
        restored,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [4, 4]},
        target_modules=["proj", "lm_head"],
    )
    restored.load_state_dict(model.state_dict())
    assert isinstance(restored.proj, QARLLinear)
    torch.testing.assert_close(restored.proj.qarl_weight_scale_inv, model.proj.qarl_weight_scale_inv)


def test_load_qarl_calibration_batches_rejects_bad_shape(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"samples": [{"input_ids": [[[1, 2], [3, 4]]]}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="1D or 2D token ids"):
        load_qarl_calibration_batches(path)
