"""Server-mode engagement regression for XORL_BI_TRUNK_LINEAR (XORL-244).

#467 as merged read the env only in the offline Trainer class, so server-mode RL
training (ModelRunner -> build_training_model) silently never wrapped — the flag
was a no-op in every server-RL run. These tests pin the build_training_model
wire: with the env set, trunk linears must be wrapped BEFORE parallelization
(FSDP2 shards the wrapped modules), and without it nothing is touched.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.ops.batch_invariant_ops import is_trunk_linear_contract_enabled, set_trunk_linear_contract
from xorl.trainers.model_builder import build_training_model


pytestmark = [pytest.mark.cpu]


class TinyTrunkModel(nn.Module):
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q_proj": nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                    "o_proj": nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                    "gate_proj": nn.Linear(16, 32, bias=False, dtype=torch.bfloat16),
                    "up_proj": nn.Linear(16, 32, bias=False, dtype=torch.bfloat16),
                    "down_proj": nn.Linear(32, 16, bias=False, dtype=torch.bfloat16),
                }
            )
            for _ in range(2)
        )
        self.lm_head = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def _reset_contract_state():
    yield
    set_trunk_linear_contract(False)


def _build(monkeypatch, captured):
    def fake_parallelize(model, **_kwargs):
        captured["wrapped_at_parallelize"] = sum(
            1 for _, m in model.named_modules() if getattr(m, "_xorl_bi_trunk_wrapped", False)
        )
        return model

    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **_kwargs: TinyTrunkModel())
    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    return build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )


def test_server_mode_build_wraps_trunk_linears_with_env(monkeypatch):
    monkeypatch.setenv("XORL_BI_TRUNK_LINEAR", "1")
    captured = {}
    result = _build(monkeypatch, captured)

    assert captured["wrapped_at_parallelize"] == 10, "wrap must land before parallelization (pre-FSDP2)"
    assert not getattr(result.model.lm_head, "_xorl_bi_trunk_wrapped", False)
    assert is_trunk_linear_contract_enabled()


def test_server_mode_build_does_not_wrap_without_env(monkeypatch):
    monkeypatch.delenv("XORL_BI_TRUNK_LINEAR", raising=False)
    captured = {}
    _build(monkeypatch, captured)

    assert captured["wrapped_at_parallelize"] == 0
    assert not is_trunk_linear_contract_enabled()
