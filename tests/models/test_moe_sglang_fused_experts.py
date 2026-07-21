"""Contract tests for the shared SGLang MoE expert forward."""

from types import SimpleNamespace

import pytest
import torch

from xorl.models.layers.moe import experts as experts_mod
from xorl.models.layers.moe.moe_block import MoEBlock


pytestmark = pytest.mark.cpu

FLAG = "XORL_MOE_SGLANG_FUSED_EXPERTS"
E, K, H, I, M = 8, 2, 32, 24, 16


def _block(backend: str = "eager") -> MoEBlock:
    torch.manual_seed(0)
    block = MoEBlock(H, E, K, I, moe_implementation=backend)
    with torch.no_grad():
        block.experts.gate_up_proj.normal_(std=0.1)
        block.experts.down_proj.normal_(std=0.1)
    return block.to(torch.bfloat16)


@pytest.fixture
def routed_inputs():
    torch.manual_seed(1)
    hidden = torch.randn(M, H, dtype=torch.bfloat16)
    weights = torch.rand(M, K, dtype=torch.bfloat16)
    experts = torch.stack([torch.randperm(E)[:K] for _ in range(M)])
    return hidden, weights, experts


def test_flag_off_preserves_default_forward(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    block = _block("eager")
    hidden = torch.randn(1, M, H, dtype=torch.bfloat16)
    expected, _ = block(hidden)
    monkeypatch.setattr(
        block.experts,
        "sglang_fused_experts_forward",
        lambda *args: pytest.fail("serving kernel engaged with contract disabled"),
    )
    actual, _ = block(hidden)
    assert torch.equal(actual, expected)


def test_flag_dispatches_forward_and_experts_only(monkeypatch, routed_inputs):
    monkeypatch.setenv(FLAG, "1")
    block = _block("eager")
    hidden, weights, selected = routed_inputs
    calls = []

    def fake(x, w, ids):
        calls.append((x.shape, w.shape, ids.shape))
        return torch.zeros_like(x)

    monkeypatch.setattr(block.experts, "sglang_fused_experts_forward", fake)
    out, _ = block(hidden.view(1, M, H))
    replay = block.forward_experts_only(hidden.view(1, M, H), weights, selected)
    assert out.shape == replay.shape == (1, M, H)
    assert calls == [((M, H), (M, K), (M, K)), ((M, H), (M, K), (M, K))]


def test_kernel_receives_zero_copy_gkn_views_and_fp32_routes(monkeypatch, routed_inputs):
    block = _block()
    block.requires_grad_(False)
    hidden, weights, selected = routed_inputs
    seen = {}

    def fake(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen.update(w13=w13, w2=w2, weights=topk_weights, ids=topk_ids, kwargs=kwargs)
        return hidden.clone()

    monkeypatch.setattr(type(block.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake))
    monkeypatch.setattr(type(block.experts), "_sglang_fused_experts_config_logged", True, raising=False)
    out = block.experts.sglang_fused_experts_forward(hidden, weights, selected)

    assert torch.equal(out, hidden)
    assert seen["w13"].shape == (E, 2 * I, H)
    assert seen["w2"].shape == (E, H, I)
    assert seen["w13"].data_ptr() == block.experts.gate_up_proj.data_ptr()
    assert seen["w2"].data_ptr() == block.experts.down_proj.data_ptr()
    assert not seen["w13"].is_contiguous() and not seen["w2"].is_contiguous()
    assert seen["weights"].dtype == torch.float32
    assert seen["kwargs"]["no_combine"] is False
    assert seen["kwargs"]["apply_router_weight_on_input"] is False


def test_training_forward_uses_explicit_backward_wrapper(monkeypatch, routed_inputs):
    block = _block()
    hidden, weights, selected = routed_inputs
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    called = {}

    monkeypatch.setattr(type(block.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: object()))

    def fake_apply(*args):
        called["args"] = args
        return torch.zeros_like(args[0])

    monkeypatch.setattr(experts_mod._SglangFusedExpertsTrainFunction, "apply", fake_apply)
    block.experts.sglang_fused_experts_forward(hidden, weights, selected)
    assert called["args"][0].data_ptr() == hidden.data_ptr()
    assert called["args"][1].data_ptr() == weights.data_ptr()
    assert called["args"][3] is block.experts.gate_up_proj
    assert called["args"][4] is block.experts.down_proj


def test_unsupported_experts_and_topology_fail_loud(monkeypatch, routed_inputs):
    monkeypatch.setenv(FLAG, "1")
    block = _block()
    hidden, weights, selected = routed_inputs
    block.experts.down_bias = torch.zeros(E, H)
    with pytest.raises(NotImplementedError, match="expert biases"):
        block.experts.sglang_fused_experts_forward(hidden, weights, selected)

    block.experts.down_bias = None
    import xorl.distributed.parallel_state as parallel_state  # noqa: PLC0415

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: SimpleNamespace(ep_size=8, tp_size=1))
    with pytest.raises(NotImplementedError, match="EP1/TP1"):
        block.forward_experts_only(hidden.view(1, M, H), weights, selected)


def test_missing_stack_error_names_contract(monkeypatch, routed_inputs):
    block = _block()
    hidden, weights, selected = routed_inputs

    def missing():
        raise ImportError("missing")

    monkeypatch.setattr(type(block.experts), "_load_sglang_fused_experts_impl", staticmethod(missing))
    with pytest.raises(ImportError, match=FLAG):
        block.experts.sglang_fused_experts_forward(hidden, weights, selected)
