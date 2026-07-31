"""Guard tests for the native-EP ordered combine lane (XORL_MOE_SGLANG_EP_COMBINE=native).

The numerical bar lives in the distributed step-0 gate
(``experiments/k3_tests/moe_ep_native_combine_gate.py``: lane output bitwise vs
a captured EP8 serving all-reduce, grad-replay bitwise per rank). Here: flag
parsing, the exclusivity/topology guards, and the collective-free failure modes.
"""

import pytest
import torch

from xorl.models.layers.moe.ep_native_combine import (
    gather_ids_for_ep_combine,
    gather_tokens_for_ep_combine,
    max_rows_for_ep_combine,
    moe_sglang_ep_combine_native_enabled,
)


pytestmark = [pytest.mark.cpu]

FLAG = "XORL_MOE_SGLANG_EP_COMBINE"


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not moe_sglang_ep_combine_native_enabled()
    for off in ("0", "false", "off", "no", ""):
        monkeypatch.setenv(FLAG, off)
        assert not moe_sglang_ep_combine_native_enabled()
    monkeypatch.setenv(FLAG, "native")
    assert moe_sglang_ep_combine_native_enabled()
    monkeypatch.setenv(FLAG, "tree")
    with pytest.raises(ValueError, match="native"):
        moe_sglang_ep_combine_native_enabled()


def _qwen_block():
    from transformers import PretrainedConfig  # noqa: PLC0415

    from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock  # noqa: PLC0415

    cfg = PretrainedConfig(
        hidden_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        hidden_act="silu",
        norm_topk_prob=True,
        shared_expert_intermediate_size=24,
        train_router=False,
    )
    return Qwen3_5MoeSparseMoeBlock(cfg, moe_implementation="eager", layer_idx=0).to(torch.bfloat16)


def test_native_requires_trainer_ep(monkeypatch):
    monkeypatch.setenv(FLAG, "native")
    blk = _qwen_block()
    x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="trainer EP"):
        blk(x)


def test_native_excludes_sim(monkeypatch):
    monkeypatch.setenv(FLAG, "native")
    monkeypatch.setenv("XORL_MOE_SGLANG_EP_COMBINE_SIM", "8")

    from xorl.distributed import parallel_state as ps_mod  # noqa: PLC0415

    class _FakePS:
        ep_enabled = True
        ep_size = 8
        ep_rank = 0
        ep_group = None

    monkeypatch.setattr(ps_mod, "get_parallel_state", lambda: _FakePS())
    blk = _qwen_block()
    x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="exclusive"):
        blk(x)


def test_native_routed_partial_enters_through_module_call(monkeypatch):
    """The EP serving-kernel lane must run inside FSDP's pre-forward hooks."""
    blk = _qwen_block()
    hidden = torch.randn(4, 32, dtype=torch.bfloat16)
    routing = torch.randn(4, 2, dtype=torch.float32)
    local_ids = torch.zeros(4, 2, dtype=torch.int32)
    calls = []

    def pre_forward(_module, _args, _kwargs):
        calls.append("pre_forward")

    def routed_partial(got_hidden, got_routing, got_ids):
        calls.append("routed_partial")
        assert got_hidden is hidden
        assert got_routing is routing
        assert got_ids is local_ids
        return torch.zeros_like(hidden)

    blk.experts.register_forward_pre_hook(pre_forward, with_kwargs=True)
    monkeypatch.setattr(blk.experts, "sglang_ep_native_routed_partial", routed_partial)

    result = blk.experts(hidden, routing, sglang_ep_native_local_ids=local_ids)

    assert calls == ["pre_forward", "routed_partial"]
    assert torch.equal(result, torch.zeros_like(hidden))


def test_variable_row_token_gather_unpads_backward(monkeypatch):
    """The live River packer gives EP ranks unequal T; collectives must not."""
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    monkeypatch.setattr(combine.dist, "get_world_size", lambda _group: 2)

    def fake_gather(out, local, group=None):
        del group
        assert local.shape == (3, 2)
        out[:3].copy_(local)
        out[3:].copy_(local + 10)

    def fake_reduce_scatter(out, grad, op=None, group=None):
        del op, group
        out.copy_(grad[:3] + grad[3:])

    monkeypatch.setattr(combine.dist, "all_gather_into_tensor", fake_gather)
    monkeypatch.setattr(combine.dist, "reduce_scatter_tensor", fake_reduce_scatter)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    gathered = gather_tokens_for_ep_combine(x, group=None, padded_rows=3)
    assert gathered.shape == (6, 2)
    assert torch.equal(gathered[2], torch.zeros(2))
    gathered.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_variable_row_id_gather_uses_invalid_padding(monkeypatch):
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    monkeypatch.setattr(combine.dist, "get_world_size", lambda _group: 2)

    def fake_gather(out, local, group=None):
        del group
        assert torch.equal(local, torch.tensor([[4, 5], [-1, -1], [-1, -1]]))
        out[:3].copy_(local)
        out[3:].copy_(local)

    monkeypatch.setattr(combine.dist, "all_gather_into_tensor", fake_gather)
    gathered = gather_ids_for_ep_combine(torch.tensor([[4, 5]]), group=None, padded_rows=3)
    assert gathered.shape == (6, 2)
    assert torch.equal(gathered[1:3], torch.full((2, 2), -1))


def test_max_rows_for_ep_combine(monkeypatch):
    import xorl.models.layers.moe.ep_native_combine as combine  # noqa: PLC0415

    def fake_max(rows, op=None, group=None):
        del op, group
        rows.fill_(8192)

    monkeypatch.setattr(combine.dist, "all_reduce", fake_max)
    assert max_rows_for_ep_combine(6016, torch.device("cpu"), group=None) == 8192
