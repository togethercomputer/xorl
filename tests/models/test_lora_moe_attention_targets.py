"""Regression test for inject_lora_into_model_with_moe attention partition."""

import pytest
import torch.nn as nn

from xorl.lora import LoraLinear, inject_lora_into_model_with_moe


pytestmark = [pytest.mark.cpu]


class _StubConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.num_experts = 0


class _MLALikeBlock(nn.Module):
    def __init__(self, hidden_size: int = 32, q_lora_rank: int = 16, kv_lora_rank: int = 16):
        super().__init__()
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(q_lora_rank, hidden_size, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _GlmLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _StubConfig("xorl_glm5")
        self.self_attn = _MLALikeBlock()


def test_default_targets_cover_all_mla_projections_for_glm5():
    model = _GlmLikeModel()

    inject_lora_into_model_with_moe(model, r=4, lora_alpha=8, target_modules=None)

    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        replaced = getattr(model.self_attn, proj)
        assert isinstance(replaced, LoraLinear), (
            f"{proj} was not LoRA-replaced; the attention/expert split is dropping MLA projections again."
        )


def test_explicit_mla_targets_are_not_filtered_out():
    model = _GlmLikeModel()

    inject_lora_into_model_with_moe(
        model,
        r=4,
        lora_alpha=8,
        target_modules=["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"],
    )

    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
        replaced = getattr(model.self_attn, proj)
        assert isinstance(replaced, LoraLinear), (
            f"{proj} was filtered out of attention_modules even though the caller passed it explicitly."
        )
    assert not isinstance(model.self_attn.o_proj, LoraLinear)
