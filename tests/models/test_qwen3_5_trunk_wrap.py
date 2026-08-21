"""Exact trunk wrap selection on the Qwen3.5-MoE hybrid.

``wrap_trunk_linears_batch_invariant`` selects modules
by leaf name. On the hybrid this must wrap the full-attention q/k/v/o
projections, the shared expert (gate_up/down AND its sigmoid gate) and the
MoE-dense-layer MLP projections — and, because xorl's GatedDeltaNet uses the
same q/k/v/o_proj leaf names, the linear-attention in/out projections match the
pattern too (unlike serving's fused in_proj_qkvz/in_proj_ba naming). The GDN
a/b/g projections and short convolutions stay outside the name set (silently
skipped: the GDN kernel chain has no serving-side contract), as do the MoE
router gate (contracted separately by the exact model program) and lm_head/embed.
"""

import pytest
import torch

from xorl.lora.modules.linear import LoraLinear
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from xorl.models.transformers.qwen3_5_shared import _apply_qwen35_gdn_exact
from xorl.ops.sglang.batch_invariant_ops import (
    is_trunk_linear_contract_enabled,
    set_trunk_linear_contract,
    wrap_trunk_linears_batch_invariant,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _hybrid_config(**overrides) -> Qwen3_5MoeConfig:
    """Two layers: layer 0 linear-attention + dense MLP, layer 1 full-attention
    + sparse MoE (shared expert included)."""
    kwargs = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        _attn_implementation="eager",
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return Qwen3_5MoeConfig(**kwargs)


def _build(dtype=torch.bfloat16) -> Qwen3_5MoeForCausalLM:
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(_hybrid_config()).to(dtype)


def test_exact_qwen_hook_enables_merged_lora_before_trunk_wrap():
    model = torch.nn.Module()
    model.config = type("Config", (), {"model_type": "xorl_qwen3_5", "_rmsnorm_mode": "sglang_fused"})()
    model.q_proj = LoraLinear(16, 16, r=2, lora_alpha=4, dtype=torch.bfloat16)
    # The hook rejects a model whose zero-centered RMSNorm never resolved, so the
    # stub carries one v2 norm. It is not an nn.Linear, so it stays out of the
    # trunk wrap count below.
    model.norm = Qwen3_5RMSNorm(16, exact_contract=True, rmsnorm_family="v2")

    try:
        assert model.q_proj.exact_merged_forward is False
        wrapped = _apply_qwen35_gdn_exact(model)

        assert model.q_proj.exact_merged_forward is True
        assert wrapped == {"q_proj": 1}
        assert model.q_proj._xorl_bi_trunk_wrapped is True
    finally:
        set_trunk_linear_contract(False)


def test_qwen3_5_hybrid_trunk_wrap_selection():
    model = _build()
    try:
        wrapped = wrap_trunk_linears_batch_invariant(model)
        assert is_trunk_linear_contract_enabled(), "wrapping the trunk must arm the contract lane"

        # Leaf-name counts: q/k/v/o match BOTH the full-attn layer and the
        # GatedDeltaNet layer (2 each); gate_up/down match the dense-layer MLP
        # and the sparse layer's shared expert (2 each); the shared-expert
        # sigmoid gate matches once.
        assert wrapped == {
            "q_proj": 2,
            "k_proj": 2,
            "v_proj": 2,
            "o_proj": 2,
            "gate_up_proj": 2,
            "down_proj": 2,
            "shared_expert_gate": 1,
        }

        linear_attn = model.model.layers[0].linear_attn
        full_attn = model.model.layers[1].self_attn
        sparse_mlp = model.model.layers[1].mlp
        dense_mlp = model.model.layers[0].mlp

        def _is_wrapped(module):
            return getattr(module, "_xorl_bi_trunk_wrapped", False)

        # Full-attention projections wrap.
        assert all(_is_wrapped(m) for m in (full_attn.q_proj, full_attn.k_proj, full_attn.v_proj, full_attn.o_proj))
        # Linear-attention q/k/v/o_proj ALSO wrap (same leaf names) — audit fact,
        # not a contract guarantee: the GDN kernel chain is uncontracted.
        assert all(
            _is_wrapped(m) for m in (linear_attn.q_proj, linear_attn.k_proj, linear_attn.v_proj, linear_attn.o_proj)
        )
        # GDN a/b/g projections and short convolutions are silently skipped.
        assert not _is_wrapped(linear_attn.a_proj)
        assert not _is_wrapped(linear_attn.b_proj)
        assert not _is_wrapped(linear_attn.g_proj)
        assert not _is_wrapped(linear_attn.q_conv1d)
        # Shared expert (MLP + sigmoid gate) and dense-layer MLP wrap.
        assert _is_wrapped(sparse_mlp.shared_expert.gate_up_proj)
        assert _is_wrapped(sparse_mlp.shared_expert.down_proj)
        assert _is_wrapped(sparse_mlp.shared_expert_gate)
        assert _is_wrapped(dense_mlp.gate_up_proj)
        assert _is_wrapped(dense_mlp.down_proj)
        # Router gate is intentionally outside the trunk lane; the exact router
        # owns it. Routed experts hold grouped parameters (no nn.Linear leaves).
        assert not _is_wrapped(sparse_mlp.gate)
        assert not any(_is_wrapped(m) for m in sparse_mlp.experts.modules())
        # lm_head / embeddings never match the name set.
        assert not _is_wrapped(model.lm_head)
        assert not _is_wrapped(model.model.embed_tokens)
    finally:
        set_trunk_linear_contract(False)


@requires_cuda
@pytest.mark.gpu
def test_qwen3_5_full_attn_forward_runs_under_trunk_wrap():
    """Wrapped full-attention + shared-expert + dense projections must run the
    bf16 contract GEMM end-to-end (the runtime guard raises on any non-bf16
    operand)."""
    torch.manual_seed(1)
    config = _hybrid_config(layer_types=["full_attention", "full_attention"], _moe_implementation="eager")
    model = Qwen3_5MoeForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    try:
        wrapped = wrap_trunk_linears_batch_invariant(model)
        assert wrapped["shared_expert_gate"] == 1
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.last_hidden_state.dtype == torch.bfloat16
        assert torch.isfinite(out.last_hidden_state.float()).all()
    finally:
        set_trunk_linear_contract(False)
