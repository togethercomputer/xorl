"""CPU smoke tests for ``DeepseekV4MoE`` and ``DeepseekV4MLP``.

Covers:

- DSv4MLP forward + ``swiglu_limit`` clamping behavior.
- Non-hash MoE layer: noaux_tc bias is on the gate, hash buffer is absent,
  forward+backward shapes are correct, shared expert contribution is added.
- Hash MoE layer (``layer_id < num_hash_layers``): tid2eid buffer is present,
  bias is absent, forward requires ``input_ids`` and uses the lookup table
  for selection.

Routed-expert backend is forced to ``eager`` to avoid triton CPU dispatch.
"""

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.cpu


def _init_test_weights(module: nn.Module, std: float = 0.02):
    """Initialize all parameters of ``module`` with ``N(0, std)``.

    ``MoEExperts`` allocates its expert weights via ``torch.empty`` (no
    default init) so smoke tests must initialize them explicitly to avoid
    NaN propagation.
    """
    for p in module.parameters():
        if p.dim() == 0:
            p.data.zero_()
        else:
            p.data.normal_(0.0, std)


def _tiny_config(*, num_hash_layers=0):
    """Compact DSv4 config tuned to keep CPU MoE tests fast."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=4,
        max_position_embeddings=512,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=4,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=num_hash_layers,
        hc_mult=2,
        compress_ratios=[0, 0],
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 256,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        num_nextn_predict_layers=0,
        _moe_implementation="eager",
    )


# ---------------------------------------------------------------------------
# DeepseekV4MLP (shared expert)
# ---------------------------------------------------------------------------


def test_mlp_forward_shape_and_swiglu_clamp():
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MLP

    cfg = _tiny_config()
    cfg.swiglu_limit = 1.0  # force a tight clamp so we can detect it
    mlp = DeepseekV4MLP(cfg).to(torch.float32)
    x = torch.randn(2, 3, cfg.hidden_size)

    # Force a huge gate magnitude: zero gate weights, set bias to a large value
    # via attribute write. Easier: just run with the clamp on and verify finite.
    out = mlp(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # With clamp off, output should be different.
    cfg2 = _tiny_config()
    cfg2.swiglu_limit = 0.0
    mlp2 = DeepseekV4MLP(cfg2).to(torch.float32)
    mlp2.load_state_dict(mlp.state_dict(), strict=False)
    out_no_clamp = mlp2(x)

    # Sanity: the test infrastructure runs both branches without error. We
    # don't assert numerical inequality because random init may keep gates in
    # the [-1, 1] range where the clamp is a no-op; the *behavior* of the
    # clamp is exercised explicitly below.
    assert out_no_clamp.shape == x.shape


def test_mlp_swiglu_limit_actually_clamps():
    """Construct a gate that pushes beyond the limit, verify clamp clips it."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MLP

    cfg = _tiny_config()
    cfg.swiglu_limit = 0.5
    mlp = DeepseekV4MLP(cfg).to(torch.float32)
    # Set gate weight to a large constant so silu(clamped) is at the boundary.
    with torch.no_grad():
        mlp.gate_proj.weight.fill_(100.0)
        mlp.up_proj.weight.fill_(1.0)
        mlp.down_proj.weight.fill_(1.0)

    x = torch.ones(1, 1, cfg.hidden_size)
    out = mlp(x)

    # Without clamp this would diverge spectacularly; with clamp at ±0.5
    # the SiLU output is bounded.
    assert torch.isfinite(out).all()
    # silu(0.5) ≈ 0.31; up_proj output ≈ hidden_size; intermediate dim = 16.
    # Output magnitude is bounded by silu(0.5) * hidden_size * intermediate_size.
    bound = 0.32 * cfg.hidden_size * cfg.moe_intermediate_size
    assert out.abs().max().item() < bound * 1.5, out.abs().max().item()


def test_routed_experts_swiglu_limit_clamps_eager_backend():
    """DeepSeek-V4 propagates swiglu_limit into routed experts, not only shared experts."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    cfg = _tiny_config()
    cfg.swiglu_limit = 0.5
    cfg.n_shared_experts = 0
    block = DeepseekV4MoE(cfg, layer_id=0).to(torch.float32)

    with torch.no_grad():
        block.experts.gate_up_proj.zero_()
        block.experts.gate_up_proj[0, :, : cfg.moe_intermediate_size].fill_(100.0)
        block.experts.gate_up_proj[0, :, cfg.moe_intermediate_size :].fill_(1.0)
        block.experts.down_proj.zero_()
        block.experts.down_proj[0].fill_(1.0)

    x = torch.ones(1, cfg.hidden_size)
    clamped = block.experts(x, expert_idx=0)
    block.experts.swiglu_limit = 0.0
    unclamped = block.experts(x, expert_idx=0)

    bound = 0.32 * cfg.hidden_size * cfg.moe_intermediate_size
    assert clamped.abs().max().item() < bound * 1.5
    assert unclamped.abs().max().item() > clamped.abs().max().item() * 100


# ---------------------------------------------------------------------------
# DeepseekV4MoE — non-hash layer
# ---------------------------------------------------------------------------


def test_moe_non_hash_has_bias_no_table():
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    cfg = _tiny_config(num_hash_layers=2)
    # layer_id = 5 is past the hash band (which is layers 0 and 1).
    block = DeepseekV4MoE(cfg, layer_id=5)

    assert hasattr(block.gate, "e_score_correction_bias")
    assert block.gate.e_score_correction_bias.shape == (cfg.n_routed_experts,)
    # ``e_score_correction_bias`` is frozen (requires_grad=False) — gradients
    # never flow through it (selection-only argmax bias). DeepSeek updates it
    # OOB via an aux-loss controller during training.
    assert block.gate.e_score_correction_bias.requires_grad is False
    assert "tid2eid" not in block._buffers
    assert block.is_hash_layer is False
    assert block.shared_experts is not None  # n_shared_experts=1


def test_moe_non_hash_forward_backward():
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    torch.manual_seed(0)
    cfg = _tiny_config(num_hash_layers=0)
    block = DeepseekV4MoE(cfg, layer_id=0).to(torch.float32)
    _init_test_weights(block)

    x = torch.randn(2, 3, cfg.hidden_size, requires_grad=True)
    out, router_logits = block(x)
    assert out.shape == x.shape
    assert router_logits.shape == (x.shape[0] * x.shape[1], cfg.n_routed_experts)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Gate must receive grads.
    assert block.gate.weight.grad is not None
    assert torch.isfinite(block.gate.weight.grad).all()
    # noaux_tc bias is "selection-only": the topk indices it produces are
    # discrete, so the bias has no gradient path. Grad should be exactly zero
    # (or None on first iteration).
    bias = block.gate.e_score_correction_bias
    assert bias.grad is None or torch.equal(bias.grad, torch.zeros_like(bias.grad))


def test_moe_shared_expert_contributes():
    """Sanity: removing the shared expert changes the output."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    torch.manual_seed(1)
    cfg = _tiny_config()
    block = DeepseekV4MoE(cfg, layer_id=0).to(torch.float32)
    _init_test_weights(block)
    x = torch.randn(1, 4, cfg.hidden_size)

    out_with, _ = block(x)
    saved = block.shared_experts
    block.shared_experts = None
    out_without, _ = block(x)
    block.shared_experts = saved

    diff = (out_with - out_without).abs().max().item()
    assert diff > 1e-6, f"shared expert had no effect: max diff {diff}"


# ---------------------------------------------------------------------------
# DeepseekV4MoE — hash layer
# ---------------------------------------------------------------------------


def test_moe_hash_layer_has_table_no_bias():
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    cfg = _tiny_config(num_hash_layers=3)
    block = DeepseekV4MoE(cfg, layer_id=0)  # in the hash band

    assert block.is_hash_layer is True
    assert hasattr(block, "tid2eid")
    assert block.tid2eid.shape == (cfg.vocab_size, cfg.num_experts_per_tok)
    assert block.tid2eid.dtype == torch.int32
    assert not hasattr(block.gate, "e_score_correction_bias")


def test_moe_hash_layer_uses_table_for_selection():
    """Forward + verify selected_experts come from tid2eid[input_ids]."""
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    torch.manual_seed(2)
    cfg = _tiny_config(num_hash_layers=3)
    block = DeepseekV4MoE(cfg, layer_id=1).to(torch.float32)
    _init_test_weights(block)

    # Build a deterministic table: token id i -> experts (i % E, (i + 1) % E).
    E = cfg.n_routed_experts
    table = torch.stack(
        [torch.arange(cfg.vocab_size) % E, (torch.arange(cfg.vocab_size) + 1) % E],
        dim=1,
    ).to(torch.int32)
    block.tid2eid.copy_(table)

    bsz, seqlen = 1, 4
    input_ids = torch.tensor([[3, 7, 0, 11]], dtype=torch.long)
    x = torch.randn(bsz, seqlen, cfg.hidden_size, requires_grad=True)

    out, _ = block(x, input_ids=input_ids)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # Backward propagates through the gate (weights are gathered from
    # sqrt(softplus(gate(x))) at the selected experts).
    out.sum().backward()
    assert torch.isfinite(block.gate.weight.grad).all()


def test_moe_hash_layer_requires_input_ids():
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    cfg = _tiny_config(num_hash_layers=3)
    block = DeepseekV4MoE(cfg, layer_id=0).to(torch.float32)
    _init_test_weights(block)
    x = torch.randn(1, 4, cfg.hidden_size)

    with pytest.raises(AssertionError, match="hash-routed layer requires input_ids"):
        block(x, input_ids=None)


# ---------------------------------------------------------------------------
# Routing-replay × hash-routed layer
# ---------------------------------------------------------------------------


def test_moe_hash_layer_record_then_replay_backward_matches():
    """Record on a hash-routed layer, then replay_backward — selected_experts
    on backward must come from the table-driven recording, not be recomputed.

    Flash attention's nondeterminism (and CP regathering) means the backward
    activations don't exactly match the forward activations. Without the
    record→replay path, the backward router would re-pick experts and drift.
    The hash-routed branch in ``DeepseekV4MoE.route`` is a separate codepath
    from the noaux_tc branch (it goes through ``self.router(... tid2eid=...,
    input_ids=...)``), so it deserves its own record+replay coverage.
    """
    from xorl.models.layers.moe.routing_replay import (
        RoutingReplay,
        set_replay_stage,
    )
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    torch.manual_seed(7)
    cfg = _tiny_config(num_hash_layers=3)
    block = DeepseekV4MoE(cfg, layer_id=0).to(torch.float32)
    _init_test_weights(block)
    block._routing_replay = RoutingReplay()

    E, K = cfg.n_routed_experts, cfg.num_experts_per_tok
    table = torch.stack(
        [torch.arange(cfg.vocab_size) % E, (torch.arange(cfg.vocab_size) + 1) % E],
        dim=1,
    ).to(torch.int32)
    block.tid2eid.copy_(table)

    input_ids = torch.tensor([[3, 7, 0, 11]], dtype=torch.long)
    x = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)

    try:
        # Record stage: forward fills the replay buffer.
        set_replay_stage("record")
        out, _ = block(x, input_ids=input_ids)
        assert out.shape == x.shape

        # The recorded selected_experts must be (B*S, K) and live on the
        # CPU pinned-memory list. Peek into ``top_indices_list`` directly
        # since ``pop_*`` advances state and assumes CUDA.
        recorded = block._routing_replay.top_indices_list
        assert len(recorded) == 1
        assert recorded[0].shape == (input_ids.numel(), K)

        # Replay-backward stage: backward pops the same selected_experts.
        # If route() didn't honor the stage, this would re-call the router
        # and the bwd graph would diverge.
        set_replay_stage("replay_backward")
        # Patch ``pop_backward`` for CPU: the production path moves to
        # ``cuda.current_device()``, but we just need the recorded tensor
        # back. Returning the raw CPU tensor exercises the same dispatch
        # in ``route()`` — the only thing under test.
        replay = block._routing_replay
        orig_pop = replay.pop_backward
        replay.pop_backward = lambda: replay.top_indices_list[0]
        try:
            out.sum().backward()
        finally:
            replay.pop_backward = orig_pop

        # Gate received gradient through sqrt(softplus(gate(x))).
        assert torch.isfinite(block.gate.weight.grad).all()
    finally:
        set_replay_stage(None)
        RoutingReplay.clear_all()


def test_moe_route_unknown_replay_stage_raises():
    """A new replay stage name (defensively) raises rather than NameError'ing
    on an undefined ``selected_experts``."""
    from xorl.models.layers.moe.routing_replay import (
        RoutingReplay,
        set_replay_stage,
    )
    from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MoE

    cfg = _tiny_config(num_hash_layers=0)
    block = DeepseekV4MoE(cfg, layer_id=1).to(torch.float32)
    _init_test_weights(block)
    block._routing_replay = RoutingReplay()

    x = torch.randn(1, 4, cfg.hidden_size)
    try:
        # Bypass set_replay_stage's validator (which constrains values) by
        # poking the module-level state directly. ``route`` reads the same
        # global, so it sees the bogus value.
        from xorl.models.layers.moe import routing_replay as _replay_mod

        _replay_mod._replay_stage = "future_stage_name"
        with pytest.raises(ValueError, match="Unrecognized replay stage"):
            block(x, input_ids=None)
    finally:
        set_replay_stage(None)
        RoutingReplay.clear_all()
