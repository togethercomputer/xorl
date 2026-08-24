"""Reduced-depth full-model backward gate for GLM-5.2 full-param training.

Component, EP, and FSDP2 tests do not by themselves prove that gradients
traverse every frozen module between trainable surfaces.  This test runs a
real full-depth forward and backward on one GPU at reduced GLM-5.2 geometry
with a scoped trainable set.

Trainable scope mirrored from production (scoped admission): dense MLP
composites (layer 0), one in-scope routed-expert bank (layer 1), ALL
routers (layers 1, 2).  Frozen trunk on the gradient path: attention
projections (q_a/q_b/kv_a/o as ``NativeBlockFP8Linear``; kv_b through the
absorbed dequant-einsum), shared experts (``NativeBlockFP8Linear`` with
block-aligned ranges), the out-of-scope frozen expert bank (layer 2), all
norms, the frozen BF16 LM head.

Asserts, on one loss backward:
1. every admitted FP32 master receives a finite, nonzero gradient — the
   layer-0 dense masters specifically prove the gradient traversed layers
   1-2's ENTIRE frozen trunk;
2. no frozen parameter receives any gradient;
3. no frozen parameter byte and no quantized-cache byte changes — frozen
   means frozen: the dgrad mechanism must never write masters or caches.

Reduced-scale seam (documented fidelity boundary): the canonical MoE
dispatch requires the EP16/CP16 topology
(``_canonical_ep_forward``), and the generic eager MoE path cannot call
the native banks at all, so this gate substitutes a SINGLE-CONTRIBUTOR
dispatch that calls the model's real ``_canonical_routed_local_partial``
and ``_canonical_shared_local_partial`` seams — every bank / shared-expert
/ projection class boundary (where the backward mechanisms live) is the
production one.  The 16-rank ordered combine itself stays gated by
tests/distributed/test_glm52_fullparam_ep16_combine.py; attention uses the
differentiable torch sparse-MLA reference (the flashmla envelope is
official-geometry-only; its trainable backward is gated separately).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.models.test_glm52_fullparam_admission import (
    _hopper_or_skip,
    _seed_native_bytes,
    _tiny_config,
)


def _gate_config():
    """3-layer reduced geometry: dense + in-scope sparse + out-of-scope sparse."""

    config = _tiny_config()
    config.num_hidden_layers = 3
    config.mlp_layer_types = ["dense", "sparse", "sparse"]
    config.indexer_types = ["full", "shared", "shared"]
    # Keep the routed scaling out of the reduced seam: the generic trainer
    # route pre-multiplies routing weights while the canonical bank call
    # applies the factor inside the kernel; 1.0 makes both identity so the
    # reduced path cannot double-apply it.
    config.routed_scaling_factor = 1.0
    # Production uses sparse MLA. flashmla's envelope is official-geometry-only;
    # the torch reference is the differentiable reduced-scale stand-in.
    config._sparse_mla_enabled = True
    config._sparse_mla_backend = "torch"
    quantization = dict(config.quantization_config)
    exclusions = [entry for entry in quantization["modules_to_not_convert"] if not entry.startswith("model.layers.")]
    exclusions.extend(f"model.layers.{layer}.self_attn.indexers_proj" for layer in range(3))
    quantization["modules_to_not_convert"] = exclusions
    config.quantization_config = quantization
    return config


def _single_contributor_experts_with_shared(
    self,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    absolute_positions: torch.Tensor | None = None,
    backward_layer_dependency: torch.Tensor | None = None,
):
    """EP1 projection of the canonical dispatch through the REAL partial seams."""

    del absolute_positions, backward_layer_dependency
    batch_size, seq_len, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    rows = flat.shape[0]
    routing = routing_weights.reshape(rows, -1).float().contiguous()
    # Single contributor owns the full bank: global ids ARE the local slots.
    local_ids = selected_experts.reshape(rows, -1).to(torch.int32).contiguous()
    routed = self._canonical_routed_local_partial(flat, routing, selected_experts, local_ids)
    shared = self._canonical_shared_local_partial(flat, contributor_ordinal=0, contributor_count=1)
    return (routed + shared).to(torch.bfloat16).reshape(batch_size, seq_len, hidden_dim)


def _byte_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone every frozen parameter and every buffer (byte caches included)."""

    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            snapshot[f"param:{name}"] = parameter.detach().clone()
    for name, buffer in model.named_buffers():
        snapshot[f"buffer:{name}"] = buffer.detach().clone()
    return snapshot


def _run_full_depth_backward_gate(monkeypatch: pytest.MonkeyPatch, *, exact_program: bool) -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        install_glm52_fullparam_components,
    )
    from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM, Glm5MoEBlock

    config = _gate_config()
    if exact_program:
        # The production resolver's full-param flag selects the exact forward — the
        # structural BI router contract, canonical serving grouped top-k in
        # route(), Class-B RoPE, and the exact indexer selector — at reduced
        # geometry.  The fused serving-norm mode rides along.
        config._glm52_fullparam_training = True
        import xorl.models.layers.normalization as normalization

        # RMSNorm modules capture the mode at __init__; monkeypatch restores
        # the module global at teardown.
        monkeypatch.setattr(normalization, "_RMSNORM_MODE", "sglang_fused")
    torch.manual_seed(9173)
    model = Glm5ForCausalLM(config).to(torch.bfloat16).to(device)
    if exact_program:
        # The exact indexer SELECTOR's kernels (flashinfer layernorm, fused
        # BF16 projection) are official-geometry-bound and reject reduced
        # shapes; the frozen selector runs entirely under no_grad, so
        # it is outside the backward under test.  Pin it to the reduced-scale
        # legacy path — a documented fidelity boundary of this gate, like
        # flashmla and the distributed combine.
        from xorl.models.transformers.glm5.modeling_glm5 import Glm5Attention

        for module in model.modules():
            if isinstance(module, Glm5Attention) and module.indexer is not None:
                monkeypatch.setattr(module.indexer, "selector_version", "legacy_torch_or_tilelang")
    _seed_native_bytes(model)
    report = install_glm52_fullparam_components(
        model, config, trainable_expert_layers=(1,), _skip_geometry_validation=True
    )
    assert report.dense_mlp_layers == (0,)
    assert report.routed_expert_layers == (1,)
    assert report.router_layers == (1, 2)

    monkeypatch.setattr(
        Glm5MoEBlock,
        "forward_experts_with_shared",
        _single_contributor_experts_with_shared,
    )

    model.train()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(trainable) == {
        "model.layers.0.mlp.gate_up_proj.weight_master",
        "model.layers.0.mlp.down_proj.weight_master",
        "model.layers.1.mlp.experts.gate_up_weight_master",
        "model.layers.1.mlp.experts.down_weight_master",
        "model.layers.1.mlp.gate.full_param.weight_master",
        "model.layers.2.mlp.gate.full_param.weight_master",
    }
    frozen = {name: parameter for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen, "reduced model lost its frozen trunk"

    before = _byte_snapshot(model)

    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    outputs = model(input_ids=input_ids, index_share_mode="training_with_backward")
    logits = model.lm_head(outputs.last_hidden_state).float()
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    assert loss.requires_grad, "loss is detached from the trainable masters"
    loss.backward()

    # 1. Gradient reaches EVERY admitted master through the full frozen depth.
    for name, parameter in trainable.items():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert bool(torch.isfinite(parameter.grad).all()), f"non-finite gradient on {name}"
        assert bool(parameter.grad.abs().sum() > 0), f"gradient on {name} is exactly zero"

    # 2. No gradient lands on any frozen parameter.
    for name, parameter in frozen.items():
        assert parameter.grad is None, f"frozen parameter {name} received a gradient"
        assert not parameter.requires_grad, f"frozen parameter {name} was unfrozen"

    # 3. Frozen bytes and cache bytes are bit-identical after the backward.
    after = _byte_snapshot(model)
    assert set(after) == set(before)
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor), f"bytes changed during forward/backward: {name}"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_reduced_full_depth_backward_reaches_all_masters_and_touches_no_frozen_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_full_depth_backward_gate(monkeypatch, exact_program=False)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_reduced_full_depth_backward_under_the_exact_forward_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the same gate with the production program selection engaged: the
    full-param flag (structural BI router contract + canonical serving
    grouped top-k + Class-B RoPE + exact indexer selector) and the fused
    serving norm mode.  FlashMLA and the distributed combine remain outside
    this single-GPU projection."""

    _run_full_depth_backward_gate(monkeypatch, exact_program=True)
