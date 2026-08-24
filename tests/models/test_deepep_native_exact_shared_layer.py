from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xorl.distributed.moe.deepep_native_exact import (
    DeepEPNativeExactError,
    NativeDeepEPGeometry,
    canonicalize_native_routing_metadata,
    native_dispatch_runner_combine,
    native_exact_router_topk,
    native_zero_row_runner_routes,
    reduce_native_runner_routes_to_bf16,
)
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.layers.moe.lora import MoEExpertsLoRA, MoELoRAConfig
from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseTritonMoeBlock,
)


class _FakeBuffer:
    def __init__(self):
        self.hidden_bytes = None

    def init_buffer(self, *, hidden_bytes):
        self.hidden_bytes = hidden_bytes


def _local_experts():
    experts = MoEExperts(
        num_experts=4,
        hidden_dim=4,
        intermediate_size=2,
        hidden_act="silu",
        moe_implementation="triton",
    )
    # Model parallelization leaves only the contiguous local expert slice.
    experts.gate_up_proj = nn.Parameter(torch.empty(2, 4, 4, dtype=torch.bfloat16))
    experts.down_proj = nn.Parameter(torch.empty(2, 2, 4, dtype=torch.bfloat16))
    experts.ep_dispatch = "deepep"
    experts.deepep_native_exact = True
    return experts


def test_qwen3_native_adapter_selects_structural_fp32_exact_router():
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=2,
        hidden_act="silu",
        norm_topk_prob=True,
        train_router=False,
        _activation_native=False,
        _ep_dispatch="deepep",
        _deepep_native_exact=True,
        _lora_serving_mode="separate",
    )

    block = Qwen3MoeSparseTritonMoeBlock(config)

    assert block._exact_batch_invariant_router is True
    assert block.router._exact_batch_invariant is True
    assert block.router._exact_weights_fp32 is True
    assert block.deepep_native_exact is True
    assert block.experts.lora_serving_mode == "separate"


def test_shared_layer_owns_real_dispatch_runner_and_fold(monkeypatch):
    experts = _local_experts()
    buffer = _FakeBuffer()
    dispatch_ctx = SimpleNamespace(num_recv_tokens=2, hidden_dim=4)
    recv_hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).contiguous()
    recv_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    recv_weights = torch.tensor([[0.75, 0.0], [0.5, 0.25]], dtype=torch.float32)
    folded = torch.full((3, 4), 7.0, dtype=torch.bfloat16)
    calls = {}

    import xorl.distributed.moe.deepep as deepep
    import xorl.distributed.moe.deepep_native_exact as native_exact

    monkeypatch.setattr(deepep, "get_default_buffer", lambda **kwargs: buffer)

    def fake_dispatch(**kwargs):
        calls["dispatch"] = kwargs
        return recv_hidden, recv_ids, recv_weights, dispatch_ctx

    monkeypatch.setattr(deepep, "token_pre_dispatch_native", fake_dispatch)
    monkeypatch.setattr(
        native_exact,
        "resolve_native_deepep_geometry",
        lambda group, hidden: NativeDeepEPGeometry(ep_size=2, ep_rank=1, hidden_size=hidden),
    )

    def fake_runner(hidden, weights, ids, *, local_expert_ids):
        calls["runner"] = (hidden, weights, ids, local_expert_ids)
        return hidden.clone()

    monkeypatch.setattr(experts, "sglang_fused_experts_forward", fake_runner)

    def fake_fold(recv_output, **kwargs):
        calls["fold"] = (recv_output, kwargs)
        return folded

    monkeypatch.setattr(native_exact, "native_receive_combine_and_fold", fake_fold)

    hidden = torch.zeros((3, 4), dtype=torch.bfloat16)
    routing = torch.full((3, 2), 0.5, dtype=torch.float32)
    selected = torch.tensor([[0, 2], [1, 3], [0, 1]], dtype=torch.int64)
    parallel = SimpleNamespace(ep_group=object())

    result = experts._deepep_native_exact_forward(hidden, routing, selected, parallel)

    assert torch.equal(result, folded)
    assert buffer.hidden_bytes == 4 * 2  # original H * sizeof(BF16); repeated once per rank
    assert torch.equal(calls["dispatch"]["hidden_states"], hidden)
    assert calls["runner"][0] is recv_hidden
    assert calls["runner"][1] is recv_weights
    assert calls["runner"][2].dtype is torch.int32
    assert torch.equal(calls["runner"][2].to(torch.int64), recv_ids)
    assert calls["runner"][3] is True
    assert calls["fold"][1]["num_local_experts"] == 2


def test_shared_layer_rejects_superseded_route_cube(monkeypatch):
    buffer = _FakeBuffer()
    dispatch_ctx = SimpleNamespace(num_recv_tokens=2, hidden_dim=2)
    recv_hidden = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2).contiguous()
    recv_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    recv_weights = torch.tensor([[0.75, 123.0], [0.5, 0.25]], dtype=torch.float32)
    routes = torch.tensor(
        [[[2.0, 4.0], [99.0, 99.0]], [[3.0, 5.0], [7.0, 11.0]]],
        dtype=torch.bfloat16,
    )
    import xorl.distributed.moe.deepep as deepep
    import xorl.distributed.moe.deepep_native_exact as native_exact

    monkeypatch.setattr(deepep, "get_default_buffer", lambda **_kwargs: buffer)
    monkeypatch.setattr(
        deepep,
        "token_pre_dispatch_native",
        lambda **_kwargs: (recv_hidden, recv_ids, recv_weights, dispatch_ctx),
    )
    monkeypatch.setattr(
        native_exact,
        "resolve_native_deepep_geometry",
        lambda _group, hidden: NativeDeepEPGeometry(ep_size=2, ep_rank=0, hidden_size=hidden),
    )

    with pytest.raises(RuntimeError, match="no_combine=False runner"):
        native_dispatch_runner_combine(
            torch.zeros((2, 2), dtype=torch.bfloat16),
            torch.full((2, 2), 0.5, dtype=torch.float32),
            torch.tensor([[0, 2], [1, 3]], dtype=torch.int64),
            ep_group=object(),
            num_experts=4,
            num_local_experts=2,
            buffer_size_gb=1.0,
            num_sms=8,
            runner=lambda hidden, weights, ids: routes,
        )


def test_shared_layer_rejects_trainable_router_metadata():
    experts = _local_experts()
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    routing = torch.ones((1, 1), dtype=torch.float32, requires_grad=True)
    selected = torch.zeros((1, 1), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="frozen router"):
        experts._deepep_native_exact_forward(
            hidden,
            routing,
            selected,
            SimpleNamespace(ep_group=object()),
        )


def test_shared_runner_reduces_bf16_routes_with_fp32_metadata():
    routes = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [99.0, 99.0], [7.0, 8.0]],
        ],
        dtype=torch.bfloat16,
    )
    ids = torch.tensor([[0, 1, -1], [1, -1, 0]], dtype=torch.int32)
    weights = torch.tensor(
        [[0.25, 0.5, 123.0], [0.75, 123.0, 0.125]],
        dtype=torch.float32,
    )

    leaf = reduce_native_runner_routes_to_bf16(routes, ids, weights)
    expected = (
        torch.where(
            (ids >= 0).unsqueeze(-1),
            routes.to(torch.float32) * weights.unsqueeze(-1),
            torch.zeros((), dtype=torch.float32),
        )
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    assert leaf.dtype is torch.bfloat16
    assert leaf.is_contiguous()
    assert torch.equal(leaf, expected)


def test_shared_runner_empty_receive_avoids_kernel_reduction():
    hidden = torch.empty((0, 4), dtype=torch.bfloat16)
    ids = torch.empty((0, 3), dtype=torch.int32)
    weights = torch.empty((0, 3), dtype=torch.float32)

    routes = native_zero_row_runner_routes(hidden, ids)
    leaf = reduce_native_runner_routes_to_bf16(routes, ids, weights)

    assert routes.shape == (0, 3, 4)
    assert leaf.shape == hidden.shape
    assert leaf.dtype is torch.bfloat16


def test_shared_native_router_builds_fp32_metadata_with_fixed_order_renorm():
    logits = torch.tensor([[1.0, 3.0, 2.0, -1.0]], dtype=torch.float32)

    weights, ids = native_exact_router_topk(logits, top_k=2, renormalize=True)

    assert weights.dtype is torch.float32
    assert ids.dtype is torch.int64
    assert ids.tolist() == [[1, 2]]
    scores = torch.softmax(logits, dim=1).gather(1, ids)
    expected = (scores / (scores[:, 0] + scores[:, 1]).unsqueeze(-1)).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(weights, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shared_native_routing_metadata_preserves_available_information(dtype):
    source = torch.tensor([[0.1234567, 0.8765433]], dtype=dtype)

    metadata = canonicalize_native_routing_metadata(source)

    assert metadata.dtype is torch.float32
    assert torch.equal(metadata, source.to(torch.float32))


def test_native_route_rejects_forced_routing(monkeypatch):
    block = Qwen3MoeSparseTritonMoeBlock(
        SimpleNamespace(
            hidden_size=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=2,
            hidden_act="silu",
            norm_topk_prob=True,
            train_router=False,
            _activation_native=False,
            _ep_dispatch="deepep",
            _deepep_native_exact=True,
        )
    ).to(torch.bfloat16)
    block._diagnostic_forced_selected_experts = torch.zeros((1, 2), dtype=torch.int64)
    monkeypatch.setattr(block, "_bi_router_logits", lambda _hidden: torch.zeros((1, 4), dtype=torch.float32))

    with pytest.raises(RuntimeError, match="forced routing is forbidden"):
        block.route(torch.zeros((1, 4), dtype=torch.bfloat16))


def test_parent_native_marker_survives_wrapped_expert_attribute_loss(monkeypatch):
    block = Qwen3MoeSparseTritonMoeBlock(
        SimpleNamespace(
            hidden_size=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=2,
            hidden_act="silu",
            norm_topk_prob=True,
            train_router=False,
            _activation_native=False,
            _ep_dispatch="deepep",
            _deepep_native_exact=True,
        )
    ).to(torch.bfloat16)
    block.experts.deepep_native_exact = False
    monkeypatch.setattr(
        block,
        "_bi_router_logits",
        lambda _hidden: torch.tensor([[1.0, 3.0, 2.0, -1.0]], dtype=torch.float32),
    )

    weights, ids, _ = block.route(torch.zeros((1, 4), dtype=torch.bfloat16))

    assert weights.dtype is torch.float32
    assert ids.tolist() == [[1, 2]]


def test_shared_runner_reduction_rejects_non_bf16_routes():
    with pytest.raises(DeepEPNativeExactError, match="routes must be BF16"):
        reduce_native_runner_routes_to_bf16(
            torch.empty((1, 1, 4), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.float32),
        )


def test_lora_adapter_delegates_context_to_shared_native_program(monkeypatch):
    experts = MoEExpertsLoRA(
        num_experts=4,
        num_local_experts=2,
        hidden_dim=4,
        intermediate_size=2,
        moe_implementation="triton",
        lora_config=MoELoRAConfig(r=2, lora_alpha=2),
    )
    experts.ep_dispatch = "deepep"
    experts.deepep_native_exact = True
    experts.lora_serving_mode = "separate"
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    routing = torch.ones((1, 1), dtype=torch.float32)
    selected = torch.zeros((1, 1), dtype=torch.int64)
    expected = torch.full_like(hidden, 3)
    calls = {}

    import xorl.distributed.moe.deepep_native_exact as native_exact

    def fake_program(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(native_exact, "native_dispatch_runner_combine", fake_program)

    result = experts._ep_forward(
        hidden,
        routing,
        selected,
        SimpleNamespace(ep_group=object()),
    )

    assert result is expected
    assert calls["kwargs"]["num_experts"] == 4
    assert calls["kwargs"]["num_local_experts"] == 2
    assert calls["kwargs"]["runner"].__self__ is experts
