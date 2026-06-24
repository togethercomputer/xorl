from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.qarl import QARLLinear, inject_qarl_into_model, summarize_qarl_model


pytestmark = pytest.mark.cpu


class TinyQARLLogprobModel(nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.ModuleDict({"proj": nn.Linear(hidden_size, hidden_size)})])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        hidden = F.silu(self.model.layers[0]["proj"](hidden))
        return self.lm_head(hidden)


def _target_logprobs(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids)
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def test_tiny_dense_qarl_training_smoke_changes_logprobs_and_restores_state(tmp_path):
    torch.manual_seed(17)
    model = TinyQARLLogprobModel()
    changed = inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [4, 4]},
        target_modules=["proj", "lm_head"],
    )
    assert changed == 2
    assert isinstance(model.model.layers[0]["proj"], QARLLinear)
    assert isinstance(model.lm_head, QARLLinear)

    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    initial_logprobs = _target_logprobs(model, input_ids, labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    losses: list[float] = []
    for _step in range(16):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    final_logprobs = _target_logprobs(model, input_ids, labels)
    assert losses[-1] < losses[0]
    assert torch.isfinite(final_logprobs).all()
    assert (final_logprobs - initial_logprobs).abs().max().item() > 1e-3

    summary = summarize_qarl_model(model)
    assert summary["enabled"] is True
    assert summary["linear_count"] == 2
    assert summary["forward_counts"]["model.layers.0.proj"] > 0
    assert summary["weight_scale_inv_shapes"]["model.layers.0.proj"] == (2, 2)

    checkpoint_path = tmp_path / "qarl_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    restored = TinyQARLLogprobModel()
    inject_qarl_into_model(
        restored,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [4, 4]},
        target_modules=["proj", "lm_head"],
    )
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    restored_logprobs = _target_logprobs(restored, input_ids, labels)
    torch.testing.assert_close(restored_logprobs, final_logprobs)
    torch.testing.assert_close(
        restored.model.layers[0]["proj"].qarl_weight_scale_inv,
        model.model.layers[0]["proj"].qarl_weight_scale_inv,
    )
    assert restored.model.layers[0]["proj"].qarl_forward_count.item() >= model.model.layers[0][
        "proj"
    ].qarl_forward_count.item()
