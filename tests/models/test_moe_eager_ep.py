import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xorl.models.layers.moe import experts as experts_mod
from xorl.models.layers.moe.backend.eager import eager_ep_compute_lora
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.layers.moe.lora import MoEExpertsLoRA, MoELoRAConfig
from xorl.models.layers.moe.moe_block import MoEBlock


pytestmark = [pytest.mark.cpu]


def test_moe_block_eager_uses_ep_dispatch_path_when_ep_enabled(monkeypatch):
    moe = MoEBlock(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        intermediate_size=16,
        moe_implementation="eager",
    )

    expert_forward = MagicMock(return_value=torch.ones(6, 8))
    monkeypatch.setattr(moe.experts, "forward", expert_forward)
    monkeypatch.setattr(moe, "_eager_forward", MagicMock(side_effect=AssertionError("local eager path used")))

    with patch(
        "xorl.distributed.parallel_state.get_parallel_state",
        return_value=SimpleNamespace(ep_enabled=True),
    ):
        output, router_logits = moe(torch.randn(2, 3, 8))

    assert output.shape == (2, 3, 8)
    assert router_logits.shape == (6, 4)
    expert_forward.assert_called_once()


def test_lora_experts_eager_ep_uses_dispatch_not_global_expert_index():
    experts = MoEExpertsLoRA(
        num_experts=4,
        num_local_experts=2,
        hidden_dim=8,
        intermediate_size=16,
        moe_implementation="eager",
        lora_config=MoELoRAConfig(r=2, lora_alpha=4),
    )
    experts.ep_dispatch = "alltoall"

    compute_output = torch.randn(3, 8)
    ctx = SimpleNamespace(expert_scores=torch.ones(3))
    mock_dispatch = MagicMock(return_value=(torch.randn(3, 8), torch.tensor([1, 3]), ctx))
    mock_compute = MagicMock(return_value=compute_output)
    mock_combine = MagicMock(side_effect=lambda **kwargs: kwargs["expert_output"])

    with (
        patch.dict("xorl.models.layers.moe.lora.EP_DISPATCH", {"alltoall": mock_dispatch}),
        patch.dict("xorl.models.layers.moe.lora.EP_COMBINE", {"alltoall": mock_combine}),
        patch.dict("xorl.models.layers.moe.lora.EP_EXPERT_COMPUTE_LORA", {"eager": mock_compute}),
        patch(
            "xorl.distributed.parallel_state.get_parallel_state",
            return_value=SimpleNamespace(ep_enabled=True, ep_group=object()),
        ),
    ):
        selected_experts = torch.tensor([[0, 2], [1, 3], [2, 3]])
        output = experts(
            torch.randn(3, 8),
            torch.ones(3, 2),
            selected_experts,
        )

    torch.testing.assert_close(output, compute_output)
    mock_dispatch.assert_called_once()
    mock_compute.assert_called_once()
    mock_combine.assert_called_once()


def test_eager_ep_compute_lora_uses_local_expert_ids_after_dispatch():
    experts = MoEExpertsLoRA(
        num_experts=4,
        num_local_experts=2,
        hidden_dim=4,
        intermediate_size=6,
        moe_implementation="eager",
        lora_config=MoELoRAConfig(r=2, lora_alpha=4),
    )
    for param in experts.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.05)

    tokens = torch.randn(3, 4)
    cumsum = torch.tensor([1, 3], dtype=torch.int32)

    actual = eager_ep_compute_lora(
        tokens,
        cumsum,
        experts.gate_proj,
        experts.up_proj,
        experts.down_proj,
        experts.gate_proj_lora_A,
        experts.gate_proj_lora_B,
        experts.up_proj_lora_A,
        experts.up_proj_lora_B,
        experts.down_proj_lora_A,
        experts.down_proj_lora_B,
        experts.scaling,
    )
    expected = torch.cat(
        [
            experts._eager_lora_forward(tokens[:1], expert_idx=0),
            experts._eager_lora_forward(tokens[1:], expert_idx=1),
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)


def test_quack_ep_warmup_uses_hidden_width_for_down_dgrad(monkeypatch):
    experts = MoEExperts(
        num_experts=2,
        hidden_dim=6,
        intermediate_size=2,
        moe_implementation="quack",
    )
    experts.ep_dispatch = "alltoall"
    monkeypatch.setattr(type(experts), "_kernel_warmed_up_quack", False, raising=False)
    quack_kernel = importlib.import_module("xorl.ops.group_gemm.kernel.quack")

    nk_calls = []

    def fake_group_gemm_same_nk(a, b, cumsum_M, max_M, transpose_b=False, **kwargs):
        del cumsum_M, max_M, kwargs
        nk_calls.append((tuple(a.shape), tuple(b.shape), transpose_b))
        out_features = b.shape[1] if transpose_b else b.shape[2]
        return torch.zeros(a.shape[0], out_features, dtype=a.dtype, device=a.device)

    def fake_group_gemm_same_mn(a, b, c, cumsum_K, max_K, **kwargs):
        del a, b, cumsum_K, max_K, kwargs
        c.zero_()

    monkeypatch.setattr(quack_kernel, "quack_group_gemm_same_nk", fake_group_gemm_same_nk)
    monkeypatch.setattr(quack_kernel, "quack_group_gemm_same_mn", fake_group_gemm_same_mn)

    compute_output = torch.randn(4, 6)
    ctx = SimpleNamespace(expert_scores=torch.ones(4))
    mock_dispatch = MagicMock(return_value=(torch.randn(4, 6), torch.tensor([2, 4]), ctx))
    mock_compute = MagicMock(return_value=compute_output)
    mock_combine = MagicMock(side_effect=lambda **kwargs: kwargs["expert_output"])

    with (
        patch.dict("xorl.models.layers.moe.experts.EP_DISPATCH", {"alltoall": mock_dispatch}),
        patch.dict("xorl.models.layers.moe.experts.EP_COMBINE", {"alltoall": mock_combine}),
        patch.dict("xorl.models.layers.moe.experts.EP_EXPERT_COMPUTE", {"quack": mock_compute}),
    ):
        output = type(experts)._ep_forward.__wrapped__(
            experts,
            torch.randn(2, 6),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            parallel_state=SimpleNamespace(ep_group=object()),
        )

    torch.testing.assert_close(output, compute_output)
    assert nk_calls[2] == ((4, 6), (2, 2, 6), True)


def test_quack_ep_fp8_alltoall_threads_backend_and_warmup(monkeypatch):
    experts = MoEExperts(
        num_experts=2,
        hidden_dim=6,
        intermediate_size=2,
        moe_implementation="quack",
    )
    experts.ep_dispatch = "alltoall"
    experts.fp8_training_enabled = True
    experts.fp8_training_grouped_backend = "triton_grouped"
    experts.fp8_training_block_size = 64
    monkeypatch.setattr(type(experts), "_kernel_warmed_up_quack_fp8", False, raising=False)

    fp8_grouped = importlib.import_module("xorl.fp8_training.grouped")
    nk_calls = []
    mn_calls = []

    def fake_fp8_group_gemm_same_nk(**kwargs):
        nk_calls.append(kwargs)
        a = kwargs["a"]
        b = kwargs["b"]
        out_features = b.shape[1] if kwargs.get("transpose_b", False) else b.shape[2]
        return torch.zeros(a.shape[0], out_features, dtype=a.dtype, device=a.device)

    def fake_fp8_group_gemm_same_mn(**kwargs):
        mn_calls.append(kwargs)
        kwargs["c"].zero_()

    monkeypatch.setattr(fp8_grouped, "fp8_group_gemm_same_nk", fake_fp8_group_gemm_same_nk)
    monkeypatch.setattr(fp8_grouped, "fp8_group_gemm_same_mn", fake_fp8_group_gemm_same_mn)

    compute_output = torch.randn(4, 6)
    ctx = SimpleNamespace(expert_scores=torch.ones(4))
    mock_dispatch = MagicMock(return_value=(torch.randn(4, 6), torch.tensor([2, 4]), ctx))
    mock_compute = MagicMock(return_value=compute_output)
    mock_combine = MagicMock(side_effect=lambda **kwargs: kwargs["expert_output"])

    with (
        patch.dict("xorl.models.layers.moe.experts.EP_DISPATCH", {"alltoall": mock_dispatch}),
        patch.dict("xorl.models.layers.moe.experts.EP_COMBINE", {"alltoall": mock_combine}),
        patch.dict("xorl.models.layers.moe.experts.EP_EXPERT_COMPUTE", {"quack": mock_compute}),
    ):
        output = type(experts)._ep_forward.__wrapped__(
            experts,
            torch.randn(2, 6),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            parallel_state=SimpleNamespace(ep_group=object()),
        )

    torch.testing.assert_close(output, compute_output)
    assert experts.last_forward_used_fp8 is True
    mock_dispatch.assert_called_once()
    mock_compute.assert_called_once()
    mock_combine.assert_called_once()

    assert len(nk_calls) == 3
    assert len(mn_calls) == 1
    assert all(call["backend"] == "triton_grouped" for call in [*nk_calls, *mn_calls])
    assert all(call["block_size"] == 64 for call in [*nk_calls, *mn_calls])
    assert nk_calls[1]["transpose_b"] is True
    assert nk_calls[2]["transpose_b"] is True
    assert mn_calls[0]["transpose_a"] is True

    compute_kwargs = mock_compute.call_args.kwargs
    assert compute_kwargs["fp8_compute"] is True
    assert compute_kwargs["fp8_grouped_backend"] == "triton_grouped"
    assert compute_kwargs["fp8_block_size"] == 64


def test_quack_deepep_ep_uses_chunked_compute_combine(monkeypatch):
    experts = MoEExperts(
        num_experts=2,
        hidden_dim=6,
        intermediate_size=2,
        moe_implementation="quack",
    )
    experts.ep_dispatch = "deepep"
    monkeypatch.setattr(type(experts), "_kernel_warmed_up_quack_bf16", True, raising=False)

    buffer = object()
    permute_tokens = torch.randn(4, 6)
    cumsum = torch.tensor([2, 4], dtype=torch.int32)
    ctx = SimpleNamespace(
        permuted_scores=torch.ones(4),
        permuted_indices=torch.arange(4),
        num_recv_tokens=4,
        handle=object(),
        dtype=permute_tokens.dtype,
        hidden_dim=permute_tokens.shape[1],
    )
    dispatch = MagicMock(side_effect=AssertionError("generic DeepEP dispatch should be bypassed"))
    no_permute_dispatch = MagicMock(return_value=(permute_tokens, cumsum, ctx))
    compute = MagicMock(side_effect=AssertionError("generic Quack EP compute should be bypassed"))
    combine = MagicMock(side_effect=AssertionError("generic DeepEP combine should be bypassed"))
    expected = torch.randn(2, 6)

    from xorl.ops.moe.quack import QuackEPDeepEPNoPermute  # noqa: PLC0415

    apply = MagicMock(return_value=expected)
    monkeypatch.setattr(QuackEPDeepEPNoPermute, "apply", apply)

    with (
        patch.dict("xorl.models.layers.moe.experts.EP_DISPATCH", {"deepep": dispatch}),
        patch.dict("xorl.models.layers.moe.experts.EP_COMBINE", {"deepep": combine}),
        patch.dict("xorl.models.layers.moe.experts.EP_EXPERT_COMPUTE", {"quack": compute}),
        patch("xorl.distributed.moe.deepep.get_default_buffer", return_value=buffer),
        patch("xorl.distributed.moe.deepep.token_pre_dispatch_no_permute", no_permute_dispatch),
    ):
        output = type(experts)._ep_forward.__wrapped__(
            experts,
            torch.randn(2, 6),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            parallel_state=SimpleNamespace(ep_group=object()),
        )

    torch.testing.assert_close(output, expected)
    assert experts.last_forward_used_fp8 is False
    dispatch.assert_not_called()
    no_permute_dispatch.assert_called_once()
    compute.assert_not_called()
    combine.assert_not_called()
    apply.assert_called_once()
    args = apply.call_args.args
    assert args[0] is permute_tokens
    assert args[1] is cumsum
    assert args[5] is ctx.permuted_scores
    assert args[6] is buffer
    assert args[7] is ctx
    assert args[10] is False  # activation_native
    assert args[11] is False
    assert args[12] == "triton_grouped"
    assert args[13] == 128


def test_quack_deepep_force_generic_uses_dispatch_compute_combine(monkeypatch):
    experts = MoEExperts(
        num_experts=2,
        hidden_dim=6,
        intermediate_size=2,
        moe_implementation="quack",
    )
    experts.ep_dispatch = "deepep"
    monkeypatch.setattr(type(experts), "_kernel_warmed_up_quack_bf16", True, raising=False)
    monkeypatch.setattr(experts_mod, "_FORCE_QUACK_DEEPEP_GENERIC", True)

    buffer = object()
    dispatch_output = torch.randn(4, 6)
    cumsum = torch.tensor([2, 4], dtype=torch.int32)
    ctx = SimpleNamespace(permuted_scores=torch.ones(4))
    compute_output = torch.randn(4, 6)
    expected = torch.randn(2, 6)

    dispatch = MagicMock(return_value=(dispatch_output, cumsum, ctx))
    no_permute_dispatch = MagicMock(side_effect=AssertionError("no-permute DeepEP dispatch should be bypassed"))
    compute = MagicMock(return_value=compute_output)
    combine = MagicMock(return_value=expected)

    with (
        patch.dict("xorl.models.layers.moe.experts.EP_DISPATCH", {"deepep": dispatch}),
        patch.dict("xorl.models.layers.moe.experts.EP_COMBINE", {"deepep": combine}),
        patch.dict("xorl.models.layers.moe.experts.EP_EXPERT_COMPUTE", {"quack": compute}),
        patch("xorl.distributed.moe.deepep.get_default_buffer", return_value=buffer),
        patch("xorl.distributed.moe.deepep.token_pre_dispatch_no_permute", no_permute_dispatch),
    ):
        output = type(experts)._ep_forward.__wrapped__(
            experts,
            torch.randn(2, 6),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            parallel_state=SimpleNamespace(ep_group=object()),
        )

    torch.testing.assert_close(output, expected)
    no_permute_dispatch.assert_not_called()
    dispatch.assert_called_once()
    compute.assert_called_once()
    combine.assert_called_once()
    assert compute.call_args.args[0] is dispatch_output
    assert combine.call_args.kwargs["expert_output"] is compute_output


def test_deepep_parity_diagnostic_logs_dispatch_and_layout(monkeypatch, capsys):
    experts = MoEExperts(
        num_experts=4,
        hidden_dim=6,
        intermediate_size=2,
        moe_implementation="quack",
    )
    experts.ep_dispatch = "deepep"
    monkeypatch.setattr(type(experts), "_kernel_warmed_up_quack_bf16", True, raising=False)
    monkeypatch.setattr(experts_mod, "_FORCE_QUACK_DEEPEP_GENERIC", True)
    experts_mod._DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS.clear()
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC", "1")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS", "all")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS", "1")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", "3")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK", "2")

    class FakeEpGroup:
        def size(self):
            return 2

    buffer = object()
    dispatch_output = torch.randn(4, 6)
    cumsum = torch.tensor([1, 4], dtype=torch.int32)
    ctx = SimpleNamespace(
        permuted_scores=torch.tensor([0.5, 0.25, 0.125, 0.125]),
        permuted_indices=torch.tensor([0, 1, 1, 2]),
        num_recv_tokens=3,
        num_valid=4,
        handle=object(),
        dtype=dispatch_output.dtype,
        hidden_dim=dispatch_output.shape[1],
    )
    compute_output = torch.randn(4, 6)
    expected = torch.randn(2, 6)

    dispatch = MagicMock(return_value=(dispatch_output, cumsum, ctx))
    compute = MagicMock(return_value=compute_output)
    combine = MagicMock(return_value=expected)

    with (
        patch.dict("xorl.models.layers.moe.experts.EP_DISPATCH", {"deepep": dispatch}),
        patch.dict("xorl.models.layers.moe.experts.EP_COMBINE", {"deepep": combine}),
        patch.dict("xorl.models.layers.moe.experts.EP_EXPERT_COMPUTE", {"quack": compute}),
        patch("xorl.distributed.moe.deepep.get_default_buffer", return_value=buffer),
    ):
        output = type(experts)._ep_forward.__wrapped__(
            experts,
            torch.randn(2, 6),
            torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
            torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            parallel_state=SimpleNamespace(ep_group=FakeEpGroup()),
        )

    torch.testing.assert_close(output, expected)
    captured = capsys.readouterr().out
    lines = [line for line in captured.splitlines() if line.startswith("[DEEPEP PARITY] ")]
    assert [json.loads(line.removeprefix("[DEEPEP PARITY] "))["phase"] for line in lines] == [
        "post_dispatch",
        "post_compute",
        "post_combine",
    ]
    post_combine = json.loads(lines[-1].removeprefix("[DEEPEP PARITY] "))
    assert post_combine["tag"] == "xorl_deepep_parity_diagnostic"
    assert post_combine["ep_size"] == 2
    assert post_combine["num_experts"] == 4
    assert post_combine["num_local_experts"] == 4
    assert post_combine["expected_num_local_experts"] == 2
    assert post_combine["local_expert_global_range"] == [0, 4]
    assert post_combine["expected_local_expert_global_range"] == [0, 2]
    assert post_combine["num_local_experts_matches_expected"] is False
    assert post_combine["cumsum_length"] == 2
    assert post_combine["cumsum_length_matches_num_local_experts"] is False
    assert post_combine["cumsum_length_matches_expected_num_local_experts"] is True
    assert post_combine["selected_experts"]["top_global_experts"][0]["count"] == 1
    assert post_combine["cumsum"]["last"] == 4
    assert post_combine["dispatch_ctx"]["num_recv_tokens"] == 3
    assert post_combine["expert_output"]["shape"] == [4, 6]
    assert post_combine["result"]["shape"] == [2, 6]


def test_quack_deepep_ep_threads_fp8_compute_to_chunked_path(monkeypatch):
    experts = MoEExperts(
        num_experts=2,
        hidden_dim=6,
        intermediate_size=2,
        moe_implementation="quack",
    )
    experts.ep_dispatch = "deepep"
    experts.fp8_training_enabled = True
    experts.fp8_training_grouped_backend = "triton_grouped"
    experts.fp8_training_block_size = 64
    monkeypatch.setattr(type(experts), "_kernel_warmed_up_quack_fp8", True, raising=False)

    buffer = object()
    permute_tokens = torch.randn(4, 6)
    cumsum = torch.tensor([2, 4], dtype=torch.int32)
    ctx = SimpleNamespace(
        permuted_scores=torch.ones(4),
        permuted_indices=torch.arange(4),
        num_recv_tokens=4,
        handle=object(),
        dtype=permute_tokens.dtype,
        hidden_dim=permute_tokens.shape[1],
    )
    dispatch = MagicMock(side_effect=AssertionError("generic DeepEP dispatch should be bypassed"))
    no_permute_dispatch = MagicMock(return_value=(permute_tokens, cumsum, ctx))
    compute = MagicMock(side_effect=AssertionError("generic Quack EP compute should be bypassed"))
    combine = MagicMock(side_effect=AssertionError("generic DeepEP combine should be bypassed"))
    expected = torch.randn(2, 6)

    from xorl.ops.moe.quack import QuackEPDeepEPNoPermute  # noqa: PLC0415

    apply = MagicMock(return_value=expected)
    monkeypatch.setattr(QuackEPDeepEPNoPermute, "apply", apply)

    with (
        patch.dict("xorl.models.layers.moe.experts.EP_DISPATCH", {"deepep": dispatch}),
        patch.dict("xorl.models.layers.moe.experts.EP_COMBINE", {"deepep": combine}),
        patch.dict("xorl.models.layers.moe.experts.EP_EXPERT_COMPUTE", {"quack": compute}),
        patch("xorl.distributed.moe.deepep.get_default_buffer", return_value=buffer),
        patch("xorl.distributed.moe.deepep.token_pre_dispatch_no_permute", no_permute_dispatch),
    ):
        output = type(experts)._ep_forward.__wrapped__(
            experts,
            torch.randn(2, 6),
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.long),
            parallel_state=SimpleNamespace(ep_group=object()),
        )

    torch.testing.assert_close(output, expected)
    assert experts.last_forward_used_fp8 is True
    dispatch.assert_not_called()
    compute.assert_not_called()
    combine.assert_not_called()
    args = apply.call_args.args
    assert args[0] is permute_tokens
    assert args[1] is cumsum
    assert args[5] is ctx.permuted_scores
    assert args[6] is buffer
    assert args[7] is ctx
    assert args[10] is False  # activation_native
    assert args[11] is True
    assert args[12] == "triton_grouped"
    assert args[13] == 64
    assert args[14] is None
    assert args[15] is None
