"""Grouped MoE expert weights must actually be initialised.

``MoEExperts`` holds ``gate_up_proj``/``down_proj`` as raw ``nn.Parameter``s
allocated with ``torch.empty``. Nothing in an ``_init_weights`` chain reaches a
bare parameter unless the owning model names ``MoEExperts`` explicitly, so a
model that forgets the branch ships experts made of whatever the allocator
returned.

That failure mode is unusually good at looking healthy. Freshly mapped pages are
normally zeroed, so the model merely runs with dead experts; recycled pages
holding an earlier model's weights even produce a plausible ``std``. It surfaces
as a hard failure only when the recycled bytes happen to decode to NaN/Inf, at
which point the forward returns all-NaN.

Both assertions below are needed. The ``std`` check catches zeroed and denormal
garbage. The reproducibility check catches recycled garbage that happens to look
correctly distributed: uninitialised memory is not covered by the seed, so two
seeded constructions only agree if the parameters are really being initialised.
"""

import importlib

import pytest
import torch

from xorl.models.layers.moe import MoEExperts


pytestmark = pytest.mark.cpu


_COMMON = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=32,
    moe_intermediate_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=32,
    pad_token_id=0,
    _attn_implementation="eager",
)

# (config class, causal-LM class, architecture-specific config kwargs)
_MODELS = {
    "qwen3_5_moe": (
        "xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe:Qwen3_5MoeConfig",
        "xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe:Qwen3_5MoeForCausalLM",
        dict(
            head_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            layer_types=["full_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        ),
    ),
    "glm4_moe": (
        "xorl.models.transformers.glm4_moe.configuration_glm4_moe:Glm4MoeConfig",
        "xorl.models.transformers.glm4_moe.modeling_glm4_moe:Glm4MoeForCausalLM",
        dict(n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=0, n_shared_experts=1),
    ),
    "deepseek_v3": (
        "xorl.models.transformers.deepseek_v3.configuration_deepseek_v3:DeepseekV3Config",
        "xorl.models.transformers.deepseek_v3.modeling_deepseek_v3:DeepseekV3ForCausalLM",
        dict(n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=0, n_shared_experts=1),
    ),
    "minimax_m3": (
        "xorl.models.transformers.minimax_m3.configuration_minimax_m3:MiniMaxM3Config",
        "xorl.models.transformers.minimax_m3.modeling_minimax_m3:MiniMaxM3SparseForCausalLM",
        dict(num_local_experts=4, num_experts_per_tok=2),
    ),
    "gpt_oss": (
        "xorl.models.transformers.gpt_oss.configuration_gpt_oss:GptOssConfig",
        "xorl.models.transformers.gpt_oss.modeling_gpt_oss:GptOssForCausalLM",
        dict(num_local_experts=4, num_experts_per_tok=2, head_dim=32),
    ),
}


def _load(spec: str):
    module_path, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_path), attr)


def _build(arch: str, seed: int = 0):
    cfg_spec, model_spec, extra = _MODELS[arch]
    kwargs = dict(_COMMON)
    kwargs.update(extra)
    torch.manual_seed(seed)
    return _load(model_spec)(_load(cfg_spec)(**kwargs))


def _expert_params(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, MoEExperts):
            out[f"{name}.gate_up_proj"] = module.gate_up_proj
            out[f"{name}.down_proj"] = module.down_proj
    return out


@pytest.mark.parametrize("arch", sorted(_MODELS))
def test_grouped_expert_weights_are_initialised(arch):
    model = _build(arch)
    # GptOssConfig carries no initializer_range; its _init_weights falls back to
    # 0.02, so mirror that rather than assuming the attribute exists.
    std = getattr(model.config, "initializer_range", 0.02)
    params = _expert_params(model)
    assert params, f"{arch} built no MoEExperts; the fixture no longer covers the MoE path"

    for name, param in params.items():
        values = param.detach().float()
        assert torch.isfinite(values).all(), f"{arch}: {name} holds non-finite values"
        # Uninitialised memory lands orders of magnitude away from the target,
        # at exactly 0 for zeroed pages and ~1e-43 for denormal garbage, so a
        # wide band still separates "initialised" from "never written".
        assert 0.3 * std < values.std().item() < 3.0 * std, (
            f"{arch}: {name} std={values.std().item():.4g} is not consistent with "
            f"initializer_range={std}; the parameter looks uninitialised"
        )


@pytest.mark.parametrize("arch", sorted(_MODELS))
def test_grouped_expert_weights_are_reproducible_under_a_seed(arch):
    first = _expert_params(_build(arch, seed=0))
    second = _expert_params(_build(arch, seed=0))
    assert first.keys() == second.keys()

    for name in first:
        # Uninitialised memory is not covered by the seed, so this is what
        # catches recycled pages whose distribution happens to look plausible.
        assert torch.equal(first[name].detach(), second[name].detach()), (
            f"{arch}: {name} differs between two seeded constructions; "
            "the parameter is not being initialised from the RNG"
        )
