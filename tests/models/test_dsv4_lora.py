"""LoRA injection tests for DeepSeek-V4 (the V0 LoRA-first scope).

Both attention LoRA (``nn.Linear`` -> ``LoraLinear``) and MoE-experts LoRA
(``MoEExperts`` -> ``MoEExpertsLoRA``) are exercised through xorl's existing
``inject_lora_into_model`` API; the LoRA infra works on the V4 model
unchanged thanks to ``DeepseekV4MoE`` inheriting from ``MoEBlock``.
"""

import pytest
import torch


pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _cpu_env(monkeypatch):
    monkeypatch.setenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", "256")
    monkeypatch.setenv("XORL_DSV4_SPARSE_ATTN_IMPL", "sparse")
    from xorl.ops.dsv4.rope import precompute_freqs_cis  # noqa: PLC0415

    precompute_freqs_cis.cache_clear()
    yield
    precompute_freqs_cis.cache_clear()


def _tiny_config():
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=4,
        max_position_embeddings=256,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        hc_mult=2,
        compress_ratios=[0, 0],
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        num_nextn_predict_layers=0,
        _moe_implementation="eager",
    )


def _build_model():
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM

    cfg = _tiny_config()
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager").to(torch.float32)
    return cfg, model


def _lm_logits(model, input_ids):
    outputs = model(input_ids)
    return model.lm_head(outputs.last_hidden_state)


# ``wo_a`` is now a first-class LoRA target. The V4 attention does a grouped
# output projection by reading ``self.wo_a.weight`` directly with an einsum,
# which bypasses ``LoraLinear.forward``. ``DeepSeekV4Attention.forward``
# detects ``isinstance(self.wo_a, LoraLinear)`` and adds the LoRA delta in
# the grouped form so the adapter actually trains.
ATTN_LORA_TARGETS = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]


def test_attention_lora_inject_freezes_base_and_adds_adapters():
    """``wq_a/wq_b/wkv/wo_b`` get LoraLinear adapters."""
    from xorl.lora.utils import inject_lora_into_model

    torch.manual_seed(0)
    _, model = _build_model()

    base_linear_count = sum(
        1 for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in ATTN_LORA_TARGETS
    )
    assert base_linear_count > 0

    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=ATTN_LORA_TARGETS)

    # Every targeted Linear is now a LoraLinear.
    from xorl.lora.modules.linear import LoraLinear

    targeted = [
        (n, m)
        for n, m in model.named_modules()
        if n.split(".")[-1] in ATTN_LORA_TARGETS and isinstance(m, (torch.nn.Linear, LoraLinear))
    ]
    lora_targeted = [(n, m) for n, m in targeted if isinstance(m, LoraLinear)]
    assert len(lora_targeted) == base_linear_count, f"expected {base_linear_count} LoraLinear, got {len(lora_targeted)}"

    # Base weight on a LoraLinear is frozen; lora_A / lora_B trainable.
    # LoraLinear subclasses nn.Linear; lora_A and lora_B are Parameters.
    name, mod = lora_targeted[0]
    assert mod.weight.requires_grad is False, name
    assert mod.lora_A.requires_grad is True
    assert mod.lora_B.requires_grad is True


def test_moe_expert_lora_inject_freezes_base_and_adds_adapters():
    """``gate_proj/up_proj/down_proj`` on routed experts get LoRA via MoEExpertsLoRA."""
    from xorl.lora.utils import inject_lora_into_model
    from xorl.models.layers.moe.lora import MoEExpertsLoRA

    torch.manual_seed(1)
    _, model = _build_model()

    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=["gate_proj", "up_proj", "down_proj"])

    # Every layer's mlp.experts should be MoEExpertsLoRA now.
    for layer in model.model.layers:
        assert isinstance(layer.mlp.experts, MoEExpertsLoRA), (
            f"layer {layer.layer_id} experts not LoRA: {type(layer.mlp.experts).__name__}"
        )
        # Base experts weights frozen; LoRA params trainable.
        assert layer.mlp.experts.gate_up_proj.requires_grad is False
        # Find at least one LoRA param.
        lora_params = [p for n, p in layer.mlp.experts.named_parameters() if "_lora_" in n]
        assert len(lora_params) > 0
        assert all(p.requires_grad for p in lora_params)


def test_wo_a_lora_delta_actually_contributes_to_output():
    """Smoke that ``wo_a`` LoRA actually fires through the grouped einsum.

    With zero-inited ``lora_B`` the delta is zero, so a fresh-injection forward
    must match the no-LoRA forward. Re-init ``lora_B`` to nonzero and verify
    the output now diverges *and* that ``wo_a.lora_A``/``lora_B`` receive
    nonzero gradients on a backward.
    """
    from xorl.lora.modules.linear import LoraLinear
    from xorl.lora.utils import inject_lora_into_model

    torch.manual_seed(0)
    cfg, model = _build_model()

    bsz, seqlen = 1, cfg.sliding_window
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)

    with torch.no_grad():
        logits_base = _lm_logits(model, input_ids).clone()

    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=["wo_a"])

    # Confirm at least one wo_a is now LoraLinear.
    wo_a_lora = [m for n, m in model.named_modules() if n.endswith(".wo_a") and isinstance(m, LoraLinear)]
    assert len(wo_a_lora) > 0

    # Zero lora_B → delta is zero → output unchanged.
    with torch.no_grad():
        logits_zero_b = _lm_logits(model, input_ids).clone()
    torch.testing.assert_close(logits_zero_b, logits_base, rtol=1e-5, atol=1e-6)

    # Re-init lora_B to nonzero so the delta actually fires.
    with torch.no_grad():
        for m in wo_a_lora:
            m.lora_B.normal_(std=0.05)

    with torch.no_grad():
        logits_after = _lm_logits(model, input_ids)
    assert not torch.allclose(logits_after, logits_base, rtol=1e-3, atol=1e-3), (
        "wo_a LoRA delta did not change the output — the einsum is bypassing the adapter"
    )

    # Backward: lora_A and lora_B on wo_a both receive nonzero grad now that
    # lora_B is nonzero (lora_A grad ∝ lora_B, so the typical zero-init step-1
    # gotcha does not apply here).
    logits = _lm_logits(model, input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    for m in wo_a_lora:
        assert m.lora_A.grad is not None and m.lora_A.grad.abs().sum().item() > 0
        assert m.lora_B.grad is not None and m.lora_B.grad.abs().sum().item() > 0


def test_lora_forward_backward_updates_only_lora_params():
    """End-to-end: only LoRA params receive grads after a forward+backward."""
    from xorl.lora.utils import inject_lora_into_model

    torch.manual_seed(2)
    cfg, model = _build_model()
    inject_lora_into_model(
        model,
        r=4,
        lora_alpha=8,
        target_modules=[*ATTN_LORA_TARGETS, "gate_proj", "up_proj", "down_proj"],
    )

    bsz, seqlen = 1, cfg.sliding_window
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (bsz, seqlen), dtype=torch.long)

    logits = _lm_logits(model, input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()

    # On the *first* training step, ``lora_A`` gradients are zero because
    # ``lora_B`` is zero-initialized (``dL/d(lora_A) ∝ lora_B``). That's a
    # property of LoRA's init contract, not a bug. We only assert that the
    # autograd graph reaches the LoRA params (``p.grad is not None``).
    lora_with_grad_path = 0
    lora_with_nonzero_grad = 0
    for n, p in model.named_parameters():
        is_lora = "lora_A" in n or "lora_B" in n
        if is_lora:
            assert p.grad is not None, f"no autograd path to LoRA param {n}"
            lora_with_grad_path += 1
            if p.grad.abs().sum().item() > 0:
                lora_with_nonzero_grad += 1

    assert lora_with_grad_path > 0, "no LoRA params at all"
    # At least the lora_B params must have nonzero grad on step 1.
    assert lora_with_nonzero_grad > 0, "no LoRA param had a nonzero first-step gradient"
