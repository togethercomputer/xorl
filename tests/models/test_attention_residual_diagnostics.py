from types import SimpleNamespace

import torch
import torch.nn.functional as F

from xorl.models.layers.attention.multi_head_attention import MultiHeadAttention
from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.normalization import RMSNorm, native_rms_norm
from xorl.models.module_utils import MoEGradientCheckpointingLayer
from xorl.ops.linear_attention.layers.gated_deltanet import GatedDeltaNet


def _attention_config():
    return SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        attention_dropout=0.0,
        attention_bias=False,
        use_qk_norm=False,
        sliding_window=None,
        _attn_implementation="eager",
    )


def test_diagnostic_o_proj_split_sum_fp32_matches_materialized_bf16_partials(monkeypatch):
    attention = MultiHeadAttention(_attention_config(), layer_idx=0)
    with torch.no_grad():
        attention.o_proj.weight.copy_(
            torch.tensor(
                [
                    [0.125, -0.25, 0.375, 0.5],
                    [-0.5, 0.25, -0.125, 0.75],
                    [0.625, -0.375, 0.5, -0.25],
                    [0.25, 0.5, -0.625, 0.125],
                ],
                dtype=attention.o_proj.weight.dtype,
            )
        )
    attention = attention.to(torch.bfloat16)

    hidden_states = torch.tensor([[0.75, -1.25, 0.5, 1.5]], dtype=torch.bfloat16)
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT", "2")
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_SUM_FP32", "1")

    actual = attention._project_output_linear(hidden_states)

    weight = attention.o_proj.weight
    expected = (
        F.linear(hidden_states[:, :2], weight[:, :2]).to(torch.float32)
        + F.linear(hidden_states[:, 2:], weight[:, 2:]).to(torch.float32)
    ).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_diagnostic_o_proj_split_can_keep_fp32_sum(monkeypatch):
    attention = MultiHeadAttention(_attention_config(), layer_idx=0)
    attention = attention.to(torch.bfloat16)

    hidden_states = torch.tensor([[0.75, -1.25, 0.5, 1.5]], dtype=torch.bfloat16)
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT", "2")
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_SUM_FP32", "1")
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_KEEP_FP32", "1")

    actual = attention._project_output_linear(hidden_states)

    weight = attention.o_proj.weight
    expected = F.linear(hidden_states[:, :2], weight[:, :2]).to(torch.float32) + F.linear(
        hidden_states[:, 2:], weight[:, 2:]
    ).to(torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_diagnostic_o_proj_split_can_carry_partials(monkeypatch):
    attention = MultiHeadAttention(_attention_config(), layer_idx=0)
    attention = attention.to(torch.bfloat16)

    hidden_states = torch.tensor([[0.75, -1.25, 0.5, 1.5]], dtype=torch.bfloat16)
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT", "2")
    monkeypatch.setenv("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_CARRY_PARTIALS", "1")

    actual = attention._project_output_linear(hidden_states)
    partials = getattr(actual, "_xorl_o_proj_tp_partials", None)

    assert partials is not None
    assert len(partials) == 2
    weight = attention.o_proj.weight
    expected_partials = (
        F.linear(hidden_states[:, :2], weight[:, :2]),
        F.linear(hidden_states[:, 2:], weight[:, 2:]),
    )
    torch.testing.assert_close(partials[0], expected_partials[0])
    torch.testing.assert_close(partials[1], expected_partials[1])
    torch.testing.assert_close(actual, expected_partials[0] + expected_partials[1])


def test_diagnostic_attention_eager_candidate_matches_manual_causal_attention():
    attention = MultiHeadAttention(_attention_config(), layer_idx=1)
    q = torch.tensor(
        [
            [
                [[0.25, -0.5], [0.75, 0.125]],
                [[-0.25, 1.0], [0.5, -0.75]],
                [[0.375, 0.625], [-0.125, 0.875]],
            ]
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [
                [[0.5, 0.25], [-0.375, 0.125]],
                [[-0.5, 0.75], [0.25, 0.625]],
                [[0.125, -0.25], [0.875, -0.5]],
            ]
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [
                [[1.0, -0.5], [0.25, 0.75]],
                [[-0.25, 0.5], [1.25, -0.75]],
                [[0.75, 0.125], [-0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    actual = attention._compute_diagnostic_attention_eager_candidate(q, k, v, attention_mask=None)

    expected_heads = []
    for head in range(q.shape[2]):
        qh = q[0, :, head, :]
        kh = k[0, :, head, :]
        vh = v[0, :, head, :]
        scores = qh @ kh.T * attention.scaling
        scores = scores.masked_fill(~torch.ones(3, 3, dtype=torch.bool).tril(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        expected_heads.append(weights @ vh)
    expected = torch.stack(expected_heads, dim=1).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_diagnostic_attention_eager_candidate_respects_layer_filter(monkeypatch):
    attention = MultiHeadAttention(_attention_config(), layer_idx=3)
    monkeypatch.setenv("XORL_DIAGNOSTIC_ATTENTION_EAGER_CANDIDATE", "1")
    monkeypatch.setenv("XORL_DIAGNOSTIC_ATTENTION_EAGER_CANDIDATE_LAYERS", "2")
    assert attention._diagnostic_attention_eager_candidate_enabled() is False

    monkeypatch.setenv("XORL_DIAGNOSTIC_ATTENTION_EAGER_CANDIDATE_LAYERS", "3")
    assert attention._diagnostic_attention_eager_candidate_enabled() is True

    monkeypatch.setenv("XORL_DIAGNOSTIC_ATTENTION_EAGER", "1")
    monkeypatch.setenv("XORL_DIAGNOSTIC_ATTENTION_EAGER_LAYERS", "4")
    assert attention._diagnostic_attention_eager_replace_enabled() is False

    monkeypatch.setenv("XORL_DIAGNOSTIC_ATTENTION_EAGER_LAYERS", "3")
    assert attention._diagnostic_attention_eager_replace_enabled() is True


def test_gdn_diagnostic_o_proj_split_matches_materialized_bf16_partials(monkeypatch):
    layer = GatedDeltaNet(
        hidden_size=4,
        expand_v=1,
        head_dim=2,
        num_heads=2,
        num_v_heads=2,
        layer_idx=3,
    )
    with torch.no_grad():
        layer.o_proj.weight.copy_(
            torch.tensor(
                [
                    [0.125, -0.25, 0.375, 0.5],
                    [-0.5, 0.25, -0.125, 0.75],
                    [0.625, -0.375, 0.5, -0.25],
                    [0.25, 0.5, -0.625, 0.125],
                ],
                dtype=layer.o_proj.weight.dtype,
            )
        )
    layer = layer.to(torch.bfloat16)

    hidden_states = torch.tensor([[[0.75, -1.25, 0.5, 1.5]]], dtype=torch.bfloat16)
    monkeypatch.setenv("XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT", "2")
    monkeypatch.setenv("XORL_GDN_DIAGNOSTIC_O_PROJ_TP_SPLIT_LAYERS", "3")

    actual = layer._project_output_linear(hidden_states)

    weight = layer.o_proj.weight
    expected = F.linear(hidden_states[..., :2], weight[:, :2]) + F.linear(hidden_states[..., 2:], weight[:, 2:])
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_gdn_diagnostic_tp_split_respects_layer_filter(monkeypatch):
    layer = GatedDeltaNet(
        hidden_size=4,
        expand_v=1,
        head_dim=2,
        num_heads=2,
        num_v_heads=2,
        layer_idx=3,
    )
    monkeypatch.setenv("XORL_GDN_DIAGNOSTIC_TP_SPLIT", "2")
    monkeypatch.setenv("XORL_GDN_DIAGNOSTIC_TP_SPLIT_LAYERS", "4")
    assert layer._diagnostic_tp_split_count() == 0

    monkeypatch.setenv("XORL_GDN_DIAGNOSTIC_TP_SPLIT_LAYERS", "3")
    assert layer._diagnostic_tp_split_count() == 2


def test_diagnostic_residual_add_cast_bf16_casts_before_rmsnorm(monkeypatch):
    norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))

    hidden_states = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
    residual = torch.tensor([[1.0, 0.5, -0.25, 0.125]], dtype=torch.bfloat16)
    monkeypatch.setenv("XORL_DIAGNOSTIC_RESIDUAL_ADD_CAST_BF16", "1")

    actual, actual_residual = norm(hidden_states, residual=residual, prenorm=True)

    expected_residual = (hidden_states + residual).to(torch.bfloat16)
    expected = native_rms_norm(expected_residual, norm.weight, norm.variance_epsilon)
    assert actual_residual.dtype == torch.bfloat16
    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(actual, expected)


def test_moe_checkpoint_wrapper_captures_final_residual_add_boundary():
    class AddTwoMlp(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 2.0

    class DummyMoELayer(MoEGradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.mlp = AddTwoMlp()

        def _pre_mlp_forward(self, hidden_states, **kwargs):
            return hidden_states + 1.0, hidden_states

    layer = DummyMoELayer()
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    output = layer._moe_forward(hidden_states)[0]

    torch.testing.assert_close(captures["final_residual_input"], hidden_states)
    torch.testing.assert_close(captures["final_mlp_output"], hidden_states + 3.0)
    torch.testing.assert_close(captures["final_residual_output"], hidden_states * 2.0 + 3.0)
    torch.testing.assert_close(output, captures["final_residual_output"])


def test_moe_checkpoint_wrapper_can_delay_final_residual_add_boundary():
    class AddTwoMlp(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 2.0

    class DummyMoELayer(MoEGradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.mlp = AddTwoMlp()
            self._delay_moe_residual_output = True

        def _pre_mlp_forward(self, hidden_states, **kwargs):
            return hidden_states + 1.0, hidden_states

    layer = DummyMoELayer()
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    output = layer._moe_forward(hidden_states)[0]

    assert isinstance(output, tuple)
    delayed_hidden_states, delayed_residual = output
    torch.testing.assert_close(captures["final_residual_input"], hidden_states)
    torch.testing.assert_close(captures["final_mlp_output"], hidden_states + 3.0)
    torch.testing.assert_close(captures["final_residual_output"], hidden_states * 2.0 + 3.0)
    torch.testing.assert_close(delayed_hidden_states, hidden_states + 3.0)
    torch.testing.assert_close(delayed_residual, hidden_states)


def test_moe_checkpoint_wrapper_can_override_layer_output():
    class AddTwoMlp(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 2.0

    class DummyMoELayer(MoEGradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.mlp = AddTwoMlp()

        def _pre_mlp_forward(self, hidden_states, **kwargs):
            return hidden_states + 1.0, hidden_states

    layer = DummyMoELayer()
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    override = torch.full_like(hidden_states, 42.0)
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())
    layer._diagnostic_layer_output_override = lambda tensor: override.to(dtype=tensor.dtype, device=tensor.device)

    output = layer._moe_forward(hidden_states)[0]

    torch.testing.assert_close(captures["final_residual_output"], hidden_states * 2.0 + 3.0)
    torch.testing.assert_close(captures["layer_output_override"], override)
    torch.testing.assert_close(output, override)


def test_moe_checkpoint_wrapper_can_override_internal_components():
    class AddTwoMlp(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 2.0

    class DummyMoELayer(MoEGradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.mlp = AddTwoMlp()

        def _pre_mlp_forward(self, hidden_states, **kwargs):
            return hidden_states + 1.0, hidden_states

    layer = DummyMoELayer()
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    moe_input_override = torch.full_like(hidden_states, 10.0)
    residual_override = torch.full_like(hidden_states, 30.0)
    mlp_output_override = torch.full_like(hidden_states, 5.0)
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())
    layer._diagnostic_component_overrides = {
        "moe_input": lambda tensor: moe_input_override.to(dtype=tensor.dtype, device=tensor.device),
        "final_residual_input": lambda tensor: residual_override.to(dtype=tensor.dtype, device=tensor.device),
        "final_mlp_output": lambda tensor: mlp_output_override.to(dtype=tensor.dtype, device=tensor.device),
    }

    output = layer._moe_forward(hidden_states)[0]

    torch.testing.assert_close(captures["moe_input_override"], moe_input_override)
    torch.testing.assert_close(captures["final_residual_input"], hidden_states)
    torch.testing.assert_close(captures["final_residual_input_override"], residual_override)
    torch.testing.assert_close(captures["final_mlp_output"], moe_input_override + 2.0)
    torch.testing.assert_close(captures["final_mlp_output_override"], mlp_output_override)
    torch.testing.assert_close(captures["final_residual_output"], residual_override + mlp_output_override)
    torch.testing.assert_close(output, residual_override + mlp_output_override)


def test_moe_forward_experts_only_captures_flat_input_and_output():
    class AddRoutingWeightExperts(torch.nn.Module):
        def forward(self, hidden_states, routing_weights, selected_experts):
            del selected_experts
            return hidden_states + routing_weights[:, :1]

    block = MoEBlock.__new__(MoEBlock)
    torch.nn.Module.__init__(block)
    block.experts = AddRoutingWeightExperts()
    captures = {}
    block._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    routing_weights = torch.tensor([[0.5], [1.5], [2.5]])
    selected_experts = torch.zeros(3, 1, dtype=torch.long)

    output = block.forward_experts_only(hidden_states, routing_weights, selected_experts)

    flat_input = hidden_states.reshape(3, 4)
    expected_flat_output = flat_input + routing_weights
    torch.testing.assert_close(captures["moe_input"], flat_input)
    torch.testing.assert_close(captures["moe_experts_output"], expected_flat_output)
    torch.testing.assert_close(output, expected_flat_output.reshape_as(hidden_states))


def test_moe_forward_experts_only_can_override_expert_output():
    class AddRoutingWeightExperts(torch.nn.Module):
        def forward(self, hidden_states, routing_weights, selected_experts):
            del selected_experts
            return hidden_states + routing_weights[:, :1]

    block = MoEBlock.__new__(MoEBlock)
    torch.nn.Module.__init__(block)
    block.experts = AddRoutingWeightExperts()
    captures = {}
    block._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    routing_weights = torch.tensor([[0.5], [1.5], [2.5]])
    selected_experts = torch.zeros(3, 1, dtype=torch.long)
    override = torch.full((3, 4), 7.0)
    block._diagnostic_component_overrides = {
        "moe_experts_output": lambda tensor: override.to(dtype=tensor.dtype, device=tensor.device)
    }

    output = block.forward_experts_only(hidden_states, routing_weights, selected_experts)

    flat_input = hidden_states.reshape(3, 4)
    expected_flat_output = flat_input + routing_weights
    torch.testing.assert_close(captures["moe_experts_output"], expected_flat_output)
    torch.testing.assert_close(captures["moe_experts_output_override"], override)
    torch.testing.assert_close(output, override.reshape_as(hidden_states))
