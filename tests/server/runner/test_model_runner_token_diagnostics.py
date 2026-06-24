"""Tests for ModelRunner._compute_token_diagnostics."""

import pytest
import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.server.runner.model_runner import ModelRunner


@pytest.fixture
def small_inputs():
    """4-token sequence, hidden=8, vocab=16, with two valid positions."""
    torch.manual_seed(0)
    hidden_states = torch.randn(1, 4, 8)
    weight = torch.randn(16, 8)
    labels = torch.tensor([[IGNORE_INDEX, 5, IGNORE_INDEX, 9]])
    return hidden_states, weight, labels


def test_compute_token_diagnostics_basic_shapes(small_inputs):
    """All fields match valid token count and topk."""
    hidden_states, weight, labels = small_inputs
    out = ModelRunner._compute_token_diagnostics(hidden_states, weight, labels, topk=4)

    assert out is not None
    # Two valid positions in the labels: indices 1 and 3.
    assert out["valid_positions"] == [1, 3]
    assert out["target_ids"] == [5, 9]
    assert len(out["target_logprobs"]) == 2
    assert len(out["target_ranks"]) == 2
    assert len(out["topk_ids"]) == 2
    assert len(out["topk_logprobs"]) == 2
    assert len(out["topk_ids"][0]) == 4
    assert len(out["topk_logprobs"][0]) == 4


def test_compute_token_diagnostics_target_consistent_with_topk(small_inputs):
    """A target at rank=1 must have logprob == topk[0]; otherwise topk[0] strictly greater."""
    hidden_states, weight, labels = small_inputs
    out = ModelRunner._compute_token_diagnostics(hidden_states, weight, labels, topk=8)
    for rank, target_lp, top_lp in zip(out["target_ranks"], out["target_logprobs"], out["topk_logprobs"]):
        if rank == 1:
            assert target_lp == pytest.approx(top_lp[0])
        else:
            assert top_lp[0] > target_lp


def test_compute_token_diagnostics_topk_zero_returns_none(small_inputs):
    hidden_states, weight, labels = small_inputs
    assert ModelRunner._compute_token_diagnostics(hidden_states, weight, labels, topk=0) is None


def test_compute_token_diagnostics_labels_none_returns_none():
    hs = torch.zeros(1, 4, 8)
    w = torch.zeros(16, 8)
    assert ModelRunner._compute_token_diagnostics(hs, w, None, topk=4) is None


def test_compute_token_diagnostics_all_ignore_returns_empty_lists():
    """All-IGNORE labels → dict with all empty lists (not None)."""
    hs = torch.zeros(1, 4, 8)
    w = torch.zeros(16, 8)
    labels = torch.full((1, 4), IGNORE_INDEX, dtype=torch.long)
    out = ModelRunner._compute_token_diagnostics(hs, w, labels, topk=4)
    assert out == {
        "valid_positions": [],
        "target_ids": [],
        "target_logprobs": [],
        "target_ranks": [],
        "topk_ids": [],
        "topk_logprobs": [],
    }


def test_compute_token_diagnostics_topk_clamped_to_vocab():
    """Requested topk > vocab is clamped to vocab without error."""
    hs = torch.randn(1, 2, 4)
    w = torch.randn(3, 4)  # vocab=3
    labels = torch.tensor([[0, 1]])
    out = ModelRunner._compute_token_diagnostics(hs, w, labels, topk=100)
    assert len(out["topk_ids"][0]) == 3


def test_compute_token_diagnostics_reports_loss_logprob_delta():
    hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    labels = torch.tensor([[0, 2]])

    logits = hidden_states.reshape(-1, 2) @ weight.t()
    target_logprobs = torch.log_softmax(logits, dim=-1)[torch.arange(2), labels.reshape(-1)].view_as(labels)

    out = ModelRunner._compute_token_diagnostics(
        hidden_states,
        weight,
        labels,
        topk=2,
        per_token_logprobs=target_logprobs,
    )

    assert out["loss_logprobs"] == pytest.approx(target_logprobs.reshape(-1).tolist())
    assert out["loss_logprob_deltas"] == pytest.approx([0.0, 0.0])
    assert out["loss_logprob_max_abs_delta"] == pytest.approx(0.0)


def test_compute_token_diagnostics_reports_raw_weight_reference_for_lm_head_module():
    class ScaledHead(torch.nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ (self.weight * 2.0).t()

    hidden_states = torch.tensor([[[1.0, 0.0]]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    labels = torch.tensor([[0]])

    out = ModelRunner._compute_token_diagnostics(
        hidden_states,
        weight,
        labels,
        topk=2,
        lm_head=ScaledHead(weight),
        include_weight_reference=True,
    )

    reference = torch.log_softmax(hidden_states.reshape(-1, 2).float() @ weight.float().t(), dim=-1)[0, 0].item()
    assert out["reference_target_logprobs"] == pytest.approx([reference])
    assert out["reference_target_ranks"] == [1]
    assert abs(out["reference_logprob_deltas"][0]) > 0.0
    assert out["reference_logprob_max_abs_delta"] == pytest.approx(abs(out["reference_logprob_deltas"][0]))


def test_compute_token_diagnostics_reports_hidden_state_summaries():
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]])
    weight = torch.eye(3)
    labels = torch.tensor([[IGNORE_INDEX, 1]])
    all_hidden_states = (
        hidden_states,
        hidden_states + 1.0,
    )

    out = ModelRunner._compute_token_diagnostics(
        hidden_states,
        weight,
        labels,
        topk=2,
        all_hidden_states=all_hidden_states,
        hidden_sample_count=2,
    )

    summary = out["hidden_state_summaries"][0]
    assert summary["layer_count"] == 2
    assert summary["sample_indices"] == [0, 2]
    assert len(summary["layers"]) == 2
    assert summary["layers"][0]["index"] == 0
    assert summary["layers"][0]["mean"] == pytest.approx(6.0)
    assert summary["layers"][0]["rms"] == pytest.approx(torch.sqrt(torch.tensor((16.0 + 36.0 + 64.0) / 3)).item())
    assert summary["layers"][0]["sample_values"] == pytest.approx([4.0, 8.0])
    assert summary["layers"][1]["mean"] == pytest.approx(7.0)


def test_compute_token_diagnostics_uses_explicit_hidden_sample_indices():
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]])
    weight = torch.eye(3)
    labels = torch.tensor([[IGNORE_INDEX, 1]])

    out = ModelRunner._compute_token_diagnostics(
        hidden_states,
        weight,
        labels,
        topk=2,
        all_hidden_states=(hidden_states,),
        hidden_sample_count=1,
        hidden_sample_indices="1-2",
    )

    summary = out["hidden_state_summaries"][0]
    assert summary["sample_indices"] == [1, 2]
    assert summary["layers"][0]["sample_values"] == pytest.approx([6.0, 8.0])


def test_compute_token_diagnostics_hidden_sample_indices_all_for_components():
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]])
    weight = torch.eye(3)
    labels = torch.tensor([[IGNORE_INDEX, 1]])

    out = ModelRunner._compute_token_diagnostics(
        hidden_states,
        weight,
        labels,
        topk=2,
        hidden_components=[{"layer": 34, "name": "layer_input", "order": 0, "tensor": hidden_states}],
        hidden_sample_count=1,
        hidden_sample_indices="all",
    )

    summary = out["hidden_component_summaries"][0]
    assert summary["sample_indices"] == [0, 1, 2]
    assert summary["components"][0]["sample_values"] == pytest.approx([4.0, 6.0, 8.0])


def test_compute_token_diagnostics_rejects_invalid_hidden_sample_indices():
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0]]])
    weight = torch.eye(3)
    labels = torch.tensor([[1]])

    with pytest.raises(ValueError, match="out-of-range hidden dims"):
        ModelRunner._compute_token_diagnostics(
            hidden_states,
            weight,
            labels,
            topk=2,
            all_hidden_states=(hidden_states,),
            hidden_sample_indices="3",
        )


def test_compute_token_diagnostics_reports_hidden_component_summaries():
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]])
    weight = torch.eye(3)
    labels = torch.tensor([[IGNORE_INDEX, 1]])
    hidden_components = [
        {
            "layer": 34,
            "name": "mlp",
            "order": 5,
            "tensor": hidden_states + 2.0,
        },
        {
            "layer": 34,
            "name": "layer_input",
            "order": 0,
            "tensor": hidden_states,
        },
    ]

    out = ModelRunner._compute_token_diagnostics(
        hidden_states,
        weight,
        labels,
        topk=2,
        hidden_components=hidden_components,
        hidden_sample_count=2,
    )

    summary = out["hidden_component_summaries"][0]
    assert summary["component_count"] == 2
    assert summary["sample_indices"] == [0, 2]
    assert [component["name"] for component in summary["components"]] == ["layer_input", "mlp"]
    assert summary["components"][0]["layer"] == 34
    assert summary["components"][0]["rms"] == pytest.approx(torch.sqrt(torch.tensor((16.0 + 36.0 + 64.0) / 3)).item())
    assert summary["components"][0]["sample_values"] == pytest.approx([4.0, 8.0])
    assert summary["components"][1]["mean"] == pytest.approx(8.0)


def test_write_hidden_component_tensor_dump_writes_ranked_pt(tmp_path):
    output_prefix = tmp_path / "component-dump"
    labels = torch.tensor([[IGNORE_INDEX, 7]])
    saved_path = ModelRunner._write_hidden_component_tensor_dump(
        output_prefix,
        [
            {"layer": 2, "name": "mlp", "order": 10, "tensor": torch.full((1, 2, 3), 2.0)},
            {"layer": 2, "name": "layer_input", "order": 0, "tensor": torch.ones(1, 2, 3)},
        ],
        labels=labels,
        metadata={"diagnostic_hidden_component_layers": [2]},
    )

    assert saved_path == str(output_prefix) + ".rank0.pt"
    payload = torch.load(saved_path, map_location="cpu", weights_only=True)
    assert payload["__metadata__"]["rank"] == 0
    assert payload["__metadata__"]["component_count"] == 2
    assert payload["__metadata__"]["diagnostic_hidden_component_layers"] == [2]
    torch.testing.assert_close(payload["labels"], labels)
    torch.testing.assert_close(payload["model.layers.2.layer_input"], torch.ones(1, 2, 3))
    torch.testing.assert_close(payload["model.layers.2.mlp"], torch.full((1, 2, 3), 2.0))


def test_hidden_component_hooks_capture_residual_and_mlp_equation_terms():
    class DummyInputNorm(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1.0

    class DummyAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2.0 + 0.25, None, None

    class DummyPostAttentionNorm(torch.nn.Module):
        def forward(self, hidden_states, residual=None, prenorm=False):
            assert prenorm
            assert residual is not None
            return hidden_states - 0.5, residual + hidden_states

    class DummyMlp(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 0.25 + 3.0

    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = DummyInputNorm()
            self.linear_attn = DummyAttention()
            self.post_attention_layernorm = DummyPostAttentionNorm()
            self.mlp = DummyMlp()

        def forward(self, hidden_states):
            residual = hidden_states
            input_norm = self.input_layernorm(hidden_states)
            attention, _, _ = self.linear_attn(input_norm)
            post_attention_norm, residual = self.post_attention_layernorm(attention, residual=residual, prenorm=True)
            mlp = self.mlp(post_attention_norm)
            return (residual + mlp,)

    class DummyInner(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layers = torch.nn.ModuleList([layer])

    class DummyOuter(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.model = DummyInner(layer)

    layer = DummyLayer()
    runner = object.__new__(ModelRunner)
    runner.model = DummyOuter(layer)

    captures, handles = runner._install_hidden_component_hooks([0])
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    try:
        layer_output = layer(hidden_states)[0]
    finally:
        for handle in handles:
            handle.remove()

    input_norm = hidden_states + 1.0
    attention = input_norm * 2.0 + 0.25
    post_attention_norm = attention - 0.5
    post_attention_residual = hidden_states + attention
    mlp = post_attention_norm * 0.25 + 3.0
    expected_layer_output = post_attention_residual + mlp

    by_name = {capture["name"]: capture["tensor"] for capture in captures}
    assert set(by_name) == {
        "layer_input",
        "input_norm",
        "attention",
        "post_attention_norm",
        "post_attention_residual",
        "mlp",
        "layer_output",
    }
    torch.testing.assert_close(layer_output, expected_layer_output)
    torch.testing.assert_close(by_name["layer_input"], hidden_states)
    torch.testing.assert_close(by_name["input_norm"], input_norm)
    torch.testing.assert_close(by_name["attention"], attention)
    torch.testing.assert_close(by_name["post_attention_norm"], post_attention_norm)
    torch.testing.assert_close(by_name["post_attention_residual"], post_attention_residual)
    torch.testing.assert_close(by_name["mlp"], mlp)
    torch.testing.assert_close(by_name["layer_output"], expected_layer_output)


def test_hidden_component_hooks_capture_shared_expert_split():
    class DummySharedExpert(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 2.0

    class DummySharedExpertGate(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states.new_zeros((*hidden_states.shape[:-1], 1))

    class DummyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_expert = DummySharedExpert()
            self.shared_expert_gate = DummySharedExpertGate()

        def _shared_expert(self, hidden_states):
            gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            return self.shared_expert(hidden_states) * gate

        def forward(self, hidden_states):
            return hidden_states + 10.0 + self._shared_expert(hidden_states)

    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = DummyMlp()

        def forward(self, hidden_states):
            return (self.mlp(hidden_states),)

    class DummyInner(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layers = torch.nn.ModuleList([layer])

    class DummyOuter(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.model = DummyInner(layer)

    layer = DummyLayer()
    runner = object.__new__(ModelRunner)
    runner.model = DummyOuter(layer)

    captures, handles = runner._install_hidden_component_hooks([0])
    hidden_states = torch.ones(1, 2, 3)
    try:
        layer(hidden_states)
    finally:
        for handle in handles:
            handle.remove()

    by_name = {capture["name"]: capture["tensor"] for capture in captures}
    assert set(by_name) == {
        "layer_input",
        "shared_expert_input",
        "shared_expert_gate_value",
        "shared_expert",
        "shared_expert_weighted",
        "experts",
        "mlp",
        "layer_output",
    }
    torch.testing.assert_close(by_name["shared_expert_input"], torch.full((1, 2, 3), 1.0))
    torch.testing.assert_close(by_name["shared_expert_gate_value"], torch.full((1, 2, 3), 0.5))
    torch.testing.assert_close(by_name["shared_expert"], torch.full((1, 2, 3), 3.0))
    torch.testing.assert_close(by_name["shared_expert_weighted"], torch.full((1, 2, 3), 1.5))
    torch.testing.assert_close(by_name["experts"], torch.full((1, 2, 3), 11.0))
    torch.testing.assert_close(by_name["mlp"], torch.full((1, 2, 3), 12.5))
    assert "_shared_expert" not in layer.mlp.__dict__
