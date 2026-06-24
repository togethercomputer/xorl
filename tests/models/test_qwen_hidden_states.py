import pytest
import torch

from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


pytestmark = [pytest.mark.cpu]


def _qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        hidden_act="gelu",
        pad_token_id=0,
    )


def _qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        _moe_implementation="eager",
        hidden_act="gelu_pytorch_tanh",
        pad_token_id=0,
    )


def _qwen3_5_moe_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        _moe_implementation="eager",
        layer_types=["full_attention", "full_attention"],
        hidden_act="gelu_pytorch_tanh",
        pad_token_id=0,
    )


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (Qwen3ForCausalLM, _qwen3_config()),
        (Qwen3MoeForCausalLM, _qwen3_moe_config()),
        (Qwen3_5MoeForCausalLM, _qwen3_5_moe_config()),
    ],
)
def test_qwen_causal_lm_preserves_output_hidden_states(model_cls, config):
    model = model_cls(config).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        default_outputs = model(input_ids=input_ids)
        diagnostic_outputs = model(input_ids=input_ids, output_hidden_states=True)

    assert default_outputs.hidden_states is None
    assert diagnostic_outputs.hidden_states is not None
    assert len(diagnostic_outputs.hidden_states) == config.num_hidden_layers + 1
    assert all(
        hidden.shape == diagnostic_outputs.last_hidden_state.shape for hidden in diagnostic_outputs.hidden_states
    )
    torch.testing.assert_close(
        diagnostic_outputs.hidden_states[-1],
        diagnostic_outputs.last_hidden_state,
        equal_nan=True,
    )
