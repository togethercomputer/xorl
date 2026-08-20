"""Tests for the XORL_MOE_SGLANG_FUSED_EXPERTS K3-parity mode under Expert Parallelism.

Validates the EP extension of the serving-kernel opt-in:
1. Dispatch guards: DeepEP raises NotImplementedError (serving's reduction chain has
   not been reproduced on its combine); FP8 EP compute raises.
2. Flag off: the EP forward keeps the stock compute path (serving wrapper never
   engages) and dispatched scores are still cast to the token dtype.
3. Flag on (world-1 gloo EP): the per-rank compute presents post-dispatch pair rows
   to ``fused_experts_impl`` as topk=1 — local expert ids rebuilt from ``cumsum``,
   fp32 pair weights preserved exactly (no bf16 double-rounding), local weight
   slices in serving layout per the weight mode (zero-copy strided views by
   default; transient/cached transpose-copies as escape hatches), no post-hoc
   multiply (kernel call is faked; numerical parity is covered separately).
4. Trainable path: grad-requiring inputs route through
   ``_SglangFusedExpertsEPTrainFunction`` (no-grad keeps the plain kernel call),
   training guards raise, and empty ranks flow exact-zero weight grads. Numerical
   backward correctness is owned by the independent FP32-routing oracle report.
"""

import builtins

import pytest
import torch
import torch.distributed as dist

from xorl.models.layers.moe.experts import MoEExperts


pytestmark = [pytest.mark.cpu]


NUM_EXPERTS, TOP_K, HID, INTER, TOKENS = 8, 2, 32, 24, 16
FLAG = "XORL_MOE_SGLANG_FUSED_EXPERTS"


def _experts(moe_implementation: str = "eager") -> MoEExperts:
    torch.manual_seed(0)
    experts = MoEExperts(
        num_experts=NUM_EXPERTS,
        hidden_dim=HID,
        intermediate_size=INTER,
        hidden_act="silu",
        moe_implementation=moe_implementation,
    )
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=0.5)
        experts.down_proj.normal_(std=0.5)
    return experts.to(torch.bfloat16)


class _FakeParallelState:
    def __init__(self, ep_group=None):
        self.ep_enabled = True
        self.ep_group = ep_group


@pytest.fixture()
def routed_inputs():
    torch.manual_seed(1)
    x = torch.randn(TOKENS, HID, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(TOKENS)])
    # fp32 weights with values that are NOT bf16-representable, to detect rounding.
    w = torch.rand(TOKENS, TOP_K, dtype=torch.float32) + 1.0 / 3.0
    assert not torch.equal(w, w.to(torch.bfloat16).float())
    return x, w, ids


@pytest.fixture()
def world1_ep_group(monkeypatch):
    """Single-process gloo group standing in for the EP group.

    ``all_to_all`` short-circuits at world size 1; gloo's ``_allgather_base``
    rejects the 2D output buffer ``preprocess()`` uses, so patch it with the
    exact world-1 equivalent (copy).
    """
    if dist.is_initialized():  # pragma: no cover - defensive
        dist.destroy_process_group()
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29537", rank=0, world_size=1)

    def _world1_all_gather_into_tensor(output, input, group=None, async_op=False):
        output.copy_(input.reshape(output.shape))

    monkeypatch.setattr(dist, "all_gather_into_tensor", _world1_all_gather_into_tensor)
    yield dist.group.WORLD
    dist.destroy_process_group()


def _assert_deepep_exclusion_asserts_no_unverifiable_mechanism(routed_inputs, monkeypatch):
    """The reason states what is known; the wheel ships no CUDA source to check against."""
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    experts.ep_dispatch = "deepep"
    x, w, ids = routed_inputs
    with pytest.raises(NotImplementedError) as exc:
        experts._ep_forward(x, w, ids, _FakeParallelState())
    message = str(exc.value)
    assert "alltoall" in message
    assert "order and rounding schedule differ" in message
    assert "No mechanism is asserted" in message
    # The retired explanation did not distinguish the paths it gated: the
    # alltoall transport carries bf16 per-rank partials too.
    assert "bf16 cast of the per-rank partials" not in message


def _assert_fp8_ep_compute_raises_with_flag(routed_inputs, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    experts.fp8_training_enabled = True
    x, w, ids = routed_inputs
    with pytest.raises(NotImplementedError, match="FP8"):
        experts._ep_forward(x, w, ids, _FakeParallelState())


def _assert_flag_off_ep_forward_keeps_stock_path(routed_inputs, monkeypatch, world1_ep_group):
    from xorl.models.layers.moe.backend import EP_EXPERT_COMPUTE  # noqa: PLC0415

    monkeypatch.delenv(FLAG, raising=False)
    experts = _experts("eager")
    x, w, ids = routed_inputs
    called = {}

    def spy(*args, **kwargs):
        raise AssertionError("serving-kernel EP compute must not engage with the flag off")

    def stock_compute(permute_tokens, cumsum, *args, **kwargs):
        called["stock"] = True
        return permute_tokens.clone()

    monkeypatch.setattr(experts, "sglang_fused_experts_ep_compute", spy)
    monkeypatch.setitem(EP_EXPERT_COMPUTE, "eager", stock_compute)
    out = experts._ep_forward(x, w.to(torch.bfloat16), ids, _FakeParallelState(world1_ep_group))
    assert called.get("stock"), "stock EP compute must run with the flag off"
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


def _assert_flag_off_scores_cast_to_token_dtype(routed_inputs, monkeypatch, world1_ep_group):
    from xorl.distributed.moe.alltoall import alltoall_pre_dispatch  # noqa: PLC0415

    x, w, ids = routed_inputs
    monkeypatch.delenv(FLAG, raising=False)
    _, _, ctx = alltoall_pre_dispatch(x, w, ids, NUM_EXPERTS, world1_ep_group)
    assert ctx.expert_scores.dtype == torch.bfloat16

    monkeypatch.setenv(FLAG, "1")
    _, _, ctx = alltoall_pre_dispatch(x, w, ids, NUM_EXPERTS, world1_ep_group)
    assert ctx.expert_scores.dtype == torch.float32


def _assert_flag_on_ep_compute_topk1_presentation(routed_inputs, monkeypatch, world1_ep_group):
    """The wrapper must hand fused_experts_impl the exact EP topk=1 contract."""
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    x, w, ids = routed_inputs
    seen = {}

    def fake_impl(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen["hidden"] = hidden
        seen["w13"] = w13
        seen["w2"] = w2
        seen["topk_weights"] = topk_weights
        seen["topk_ids"] = topk_ids
        seen["kwargs"] = kwargs
        return hidden.clone()

    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    monkeypatch.setattr(type(experts), "_sglang_fused_experts_ep_config_logged", True, raising=False)
    out = experts._ep_forward(x, w, ids, _FakeParallelState(world1_ep_group))
    assert out.shape == x.shape

    num_pairs = TOKENS * TOP_K
    assert seen["hidden"].shape == (num_pairs, HID)
    # topk=1 presentation: one local expert id per pair row, ids int32 built from
    # the per-expert cumsum (world-1: local ids == sorted global selections).
    assert seen["topk_ids"].shape == (num_pairs, 1)
    assert seen["topk_ids"].dtype == torch.int32
    assert torch.equal(seen["topk_ids"].flatten().long(), torch.sort(ids.flatten()).values)
    # fp32 pair weights, exact (no bf16 round-trip anywhere in the dispatch).
    assert seen["topk_weights"].shape == (num_pairs, 1)
    assert seen["topk_weights"].dtype == torch.float32
    assert torch.equal(
        torch.sort(seen["topk_weights"].flatten()).values,
        torch.sort(w.flatten()).values,
    )
    # local weight slices in SGLang layout (world-1: full weights), gate first.
    assert seen["w13"].shape == (NUM_EXPERTS, 2 * INTER, HID)
    assert seen["w2"].shape == (NUM_EXPERTS, HID, INTER)
    assert torch.equal(seen["w13"][:, :INTER, :], experts.gate_up_proj[:, :, :INTER].transpose(1, 2))
    assert torch.equal(seen["w2"], experts.down_proj.transpose(1, 2))
    # the kernel applies the routing weight on the fp32 accumulator; no post-hoc
    # multiply and no second combine inside the kernel call.
    assert seen["kwargs"]["apply_router_weight_on_input"] is False
    assert seen["kwargs"]["no_combine"] is False
    assert seen["kwargs"]["inplace"] is False
    assert seen["kwargs"]["filter_expert"] is False


def _assert_empty_rank_short_circuits_kernel(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")

    def fake_impl(*args, **kwargs):
        raise AssertionError("kernel must not launch on an empty rank")

    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    permute_tokens = torch.empty(0, HID, dtype=torch.bfloat16)
    cumsum = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
    scores = torch.empty(0, dtype=torch.float32)
    out = experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert out.shape == (0, HID)


def _assert_ep_compute_guards(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    permute_tokens = torch.randn(4, HID, dtype=torch.bfloat16)
    cumsum = torch.tensor([1, 2, 2, 3, 3, 4, 4, 4], dtype=torch.int64)
    scores = torch.rand(4, dtype=torch.float32)

    with pytest.raises(ValueError):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, None)

    experts.gated = False
    with pytest.raises(NotImplementedError):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    experts.gated = True

    experts.down_bias = torch.zeros(NUM_EXPERTS, HID, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    experts.down_bias = None

    experts.hidden_act = "relu2"
    with pytest.raises(NotImplementedError):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    experts.hidden_act = "silu"


def _assert_missing_sglang_raises_import_error_naming_flag(monkeypatch):
    real_import = builtins.__import__

    def import_without_sglang(name, *args, **kwargs):
        if name == "sglang" or name.startswith("sglang."):
            raise ModuleNotFoundError("simulated missing sglang", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sglang)
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    permute_tokens = torch.randn(4, HID, dtype=torch.bfloat16)
    cumsum = torch.full((NUM_EXPERTS,), 4, dtype=torch.int64)
    cumsum[0] = 4
    scores = torch.rand(4, dtype=torch.float32)
    with pytest.raises(ImportError, match=FLAG):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)


def _dispatched_inputs(requires_grad: bool = False):
    """Post-dispatch pair rows for a direct sglang_fused_experts_ep_compute call."""
    torch.manual_seed(2)
    rows = 12
    permute_tokens = torch.randn(rows, HID, dtype=torch.bfloat16, requires_grad=requires_grad)
    counts = torch.tensor([2, 1, 0, 3, 2, 1, 2, 1], dtype=torch.int64)
    assert int(counts.sum()) == rows and counts.numel() == NUM_EXPERTS
    cumsum = torch.cumsum(counts, dim=0)
    scores = torch.rand(rows, dtype=torch.float32, requires_grad=requires_grad)
    return permute_tokens, cumsum, scores


def _assert_trainable_ep_dispatch_uses_autograd_function(monkeypatch):
    """When gradients are required, the EP compute must route through the
    autograd Function; the no-grad path must keep using the plain kernel call."""
    from xorl.models.layers.moe import experts as experts_mod  # noqa: PLC0415

    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    called = {}

    def fake_apply(*args):
        called["train"] = True
        return torch.zeros_like(args[0])

    def fake_kernel_call(permute_tokens, *args, **kwargs):
        called["plain"] = True
        return torch.zeros_like(permute_tokens)

    monkeypatch.setattr(experts_mod._SglangFusedExpertsEPTrainFunction, "apply", staticmethod(fake_apply))
    monkeypatch.setattr(experts_mod, "_sglang_fused_experts_ep_kernel_call", fake_kernel_call)
    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: lambda *a, **k: None))
    permute_tokens, cumsum, scores = _dispatched_inputs()

    experts.gate_up_proj.requires_grad_(True)
    experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert called == {"train": True}

    called.clear()
    experts.gate_up_proj.requires_grad_(False)
    experts.down_proj.requires_grad_(False)
    experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert called == {"plain": True}

    called.clear()
    experts.gate_up_proj.requires_grad_(True)
    with torch.no_grad():
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert called == {"plain": True}


def _assert_trainable_ep_guards(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: lambda *a, **k: None))
    experts.gate_up_proj.requires_grad_(True)
    permute_tokens, cumsum, scores = _dispatched_inputs()

    experts.gate_up_bias = torch.zeros(NUM_EXPERTS, 2 * INTER, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="gate_up_bias"):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    experts.gate_up_bias = None

    experts.hidden_act = "gelu"
    with pytest.raises(NotImplementedError, match="EP training supports"):
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    experts.hidden_act = "silu"


def _assert_trainable_ep_empty_rank_flows_zero_weight_grads(monkeypatch):
    """An empty rank must skip the kernel but still flow exact-zero weight grads
    (keeps FSDP grad reduction uniform across ranks)."""
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")

    def fake_impl(*args, **kwargs):
        raise AssertionError("kernel must not launch on an empty rank")

    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    experts.gate_up_proj.requires_grad_(True)
    experts.down_proj.requires_grad_(True)
    permute_tokens = torch.empty(0, HID, dtype=torch.bfloat16, requires_grad=True)
    cumsum = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
    scores = torch.empty(0, dtype=torch.float32, requires_grad=True)

    out = experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert out.shape == (0, HID)
    assert out.requires_grad
    out.sum().backward()
    assert experts.gate_up_proj.grad is not None and not experts.gate_up_proj.grad.any()
    assert experts.down_proj.grad is not None and not experts.down_proj.grad.any()
    assert permute_tokens.grad is not None and permute_tokens.grad.shape == permute_tokens.shape
    assert scores.grad is not None and scores.grad.shape == scores.shape


def _assert_ep_strided_mode_passes_zero_copy_weight_views(monkeypatch):
    """Default (strided) mode must hand the EP kernel transpose-VIEWS of the local
    GKN weight slices (same storage, non-contiguous, serving element order) and
    never populate the weight cache."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", raising=False)
    experts = _experts("eager")
    seen = {}

    def fake_impl(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen["w13"] = w13
        seen["w2"] = w2
        return hidden.clone()

    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    monkeypatch.setattr(type(experts), "_sglang_fused_experts_ep_config_logged", True, raising=False)
    permute_tokens, cumsum, scores = _dispatched_inputs()
    with torch.no_grad():
        out = experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)

    assert out.shape == permute_tokens.shape
    assert not seen["w13"].is_contiguous() and not seen["w2"].is_contiguous()
    assert seen["w13"].data_ptr() == experts.gate_up_proj.data_ptr()
    assert seen["w2"].data_ptr() == experts.down_proj.data_ptr()
    assert torch.equal(seen["w13"], experts.gate_up_proj.transpose(1, 2))
    assert torch.equal(seen["w2"], experts.down_proj.transpose(1, 2))
    assert getattr(experts, "_sglang_fused_weight_cache", None) in (None, {})


def _assert_ep_weight_cache_reuses_and_invalidates(monkeypatch):
    """Escape hatches: transient mode makes fresh contiguous transpose-copies per
    call; cached mode reuses the same transposed tensors across scoring AND
    trainable calls (one cache per module, shared with the local path), drops
    them on explicit invalidation, and re-materializes after in-place updates."""
    monkeypatch.setenv(FLAG, "1")
    experts = _experts("eager")
    seen = []

    def fake_impl(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen.append((w13, w2))
        return hidden.clone()

    monkeypatch.setattr(type(experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    monkeypatch.setattr(type(experts), "_sglang_fused_experts_ep_config_logged", True, raising=False)
    permute_tokens, cumsum, scores = _dispatched_inputs()

    # transient -> fresh contiguous copies each call
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "transient")
    with torch.no_grad():
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert seen[0][0] is not seen[1][0]
    assert seen[0][0].is_contiguous() and seen[0][1].is_contiguous()

    # cached -> same transposed tensors reused across scoring and trainable calls
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "cached")
    seen.clear()
    with torch.no_grad():
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    experts.gate_up_proj.requires_grad_(True)
    experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)  # trainable path
    experts.gate_up_proj.requires_grad_(False)
    assert seen[0][0] is seen[1][0] and seen[0][1] is seen[1][1]
    assert torch.equal(seen[0][0], experts.gate_up_proj.transpose(1, 2))

    # in-place parameter update bumps _version -> cache re-materializes
    with torch.no_grad():
        experts.gate_up_proj.add_(1.0)
    seen.clear()
    with torch.no_grad():
        experts.sglang_fused_experts_ep_compute(permute_tokens, cumsum, scores)
    assert torch.equal(seen[0][0], experts.gate_up_proj.transpose(1, 2))

    # explicit invalidation drops entries (same hook as the local path)
    experts.invalidate_sglang_fused_weight_cache()
    assert experts._sglang_fused_weight_cache == {}


def test_sglang_ep_policy(routed_inputs, monkeypatch, world1_ep_group):
    with monkeypatch.context() as case_patch:
        _assert_flag_on_ep_compute_topk1_presentation(routed_inputs, case_patch, world1_ep_group)
    with monkeypatch.context() as case_patch:
        _assert_deepep_exclusion_asserts_no_unverifiable_mechanism(routed_inputs, case_patch)
    with monkeypatch.context() as case_patch:
        _assert_fp8_ep_compute_raises_with_flag(routed_inputs, case_patch)
    with monkeypatch.context() as case_patch:
        _assert_missing_sglang_raises_import_error_naming_flag(case_patch)
    with monkeypatch.context() as case_patch:
        _assert_flag_off_ep_forward_keeps_stock_path(routed_inputs, case_patch, world1_ep_group)
    with monkeypatch.context() as case_patch:
        _assert_flag_off_scores_cast_to_token_dtype(routed_inputs, case_patch, world1_ep_group)
    with monkeypatch.context() as case_patch:
        _assert_empty_rank_short_circuits_kernel(case_patch)
    with monkeypatch.context() as case_patch:
        _assert_ep_compute_guards(case_patch)
    with monkeypatch.context() as trainable_patch:
        _assert_sglang_ep_trainable_dispatch_contract(trainable_patch)
    with monkeypatch.context() as weight_patch:
        _assert_sglang_ep_weight_presentation_modes(weight_patch)


def _assert_sglang_ep_trainable_dispatch_contract(monkeypatch):
    _assert_trainable_ep_dispatch_uses_autograd_function(monkeypatch)
    monkeypatch.undo()
    _assert_trainable_ep_guards(monkeypatch)
    monkeypatch.undo()
    _assert_trainable_ep_empty_rank_flows_zero_weight_grads(monkeypatch)


def _assert_sglang_ep_weight_presentation_modes(monkeypatch):
    _assert_ep_strided_mode_passes_zero_copy_weight_views(monkeypatch)
    monkeypatch.undo()
    _assert_ep_weight_cache_reuses_and_invalidates(monkeypatch)
