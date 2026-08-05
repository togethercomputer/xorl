"""The model-wide `.to(torch_dtype)` must not quantize the rope frequency table.

``XorlPreTrainedModel._from_config`` calls ``model.to(torch_dtype)``, which casts every
floating-point buffer — including the non-persistent ``inv_freq``. A bf16 frequency table
perturbs cos/sin from position 1 onward, so ``RotaryEmbedding`` must keep an fp32
CPU-computed table and read that in ``forward``.
"""

import pytest
import torch

from xorl.models.auto import build_foundation_model
from xorl.models.layers.rope import ROPE_INIT_FUNCTIONS, RotaryEmbedding
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config


pytestmark = pytest.mark.cpu

HEAD_DIM = 128
ROPE_THETA = 500000.0
MAX_POS = 4096

# One scaling block per registry entry; llama3 is delphi-9.7B's production block.
ROPE_SCALINGS = {
    "default": None,
    "linear": {"rope_type": "linear", "factor": 4.0},
    "dynamic": {"rope_type": "dynamic", "factor": 4.0},
    "yarn": {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 1024,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "longrope": {
        "rope_type": "longrope",
        "long_factor": [1.0] * (HEAD_DIM // 2),
        "short_factor": [1.0] * (HEAD_DIM // 2),
        "original_max_position_embeddings": 1024,
    },
    "llama3": {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
}


def _config(rope_type: str) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_POS,
        rope_theta=ROPE_THETA,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    rope_scaling = ROPE_SCALINGS[rope_type]
    config.rope_scaling = rope_scaling
    if rope_scaling is not None:
        config.rope_parameters = {**rope_scaling, "rope_theta": ROPE_THETA}
    return config


def _build(rope_type: str, dtype: str, rope_native: bool = False):
    return build_foundation_model(
        config_path=_config(rope_type),
        weights_path=None,
        torch_dtype=dtype,
        attn_implementation="eager",
        rope_native=rope_native,
        init_device="cpu",
    )


def test_registry_is_fully_covered():
    assert set(ROPE_SCALINGS) == set(ROPE_INIT_FUNCTIONS)


@pytest.mark.parametrize("rope_type", sorted(ROPE_INIT_FUNCTIONS))
def test_inv_freq_survives_model_wide_bf16_cast(rope_type: str):
    rotary = _build(rope_type, "bfloat16").model.rotary_emb
    table = rotary._resolve_inv_freq(torch.device("cpu"))

    assert table.dtype == torch.float32, f"{rope_type}: rope reads a {table.dtype} frequency table"

    reference, _ = ROPE_INIT_FUNCTIONS[rope_type](_config(rope_type), "cpu")
    assert torch.equal(table, reference.float()), f"{rope_type}: frequency table is not the fp32 CPU reference"


@pytest.mark.parametrize("rope_type", sorted(ROPE_INIT_FUNCTIONS))
def test_bf16_built_cos_sin_matches_fp32_built(rope_type: str):
    """The bf16-built model's rope table must produce the fp32-built model's cos/sin, bitwise."""
    position_ids = torch.arange(MAX_POS)[None, :]
    x_bf16 = torch.zeros(1, MAX_POS, HEAD_DIM, dtype=torch.bfloat16)
    x_fp32 = torch.zeros(1, MAX_POS, HEAD_DIM, dtype=torch.float32)

    with torch.no_grad():
        cos_bf16, sin_bf16 = _build(rope_type, "bfloat16").model.rotary_emb(x_bf16, position_ids)
        cos_fp32, sin_fp32 = _build(rope_type, "float32").model.rotary_emb(x_fp32, position_ids)

    assert torch.equal(cos_bf16.float(), cos_fp32.to(torch.bfloat16).float()), f"{rope_type}: cos differs"
    assert torch.equal(sin_bf16.float(), sin_fp32.to(torch.bfloat16).float()), f"{rope_type}: sin differs"


@pytest.mark.parametrize("rope_type", sorted(ROPE_INIT_FUNCTIONS))
def test_contract_lane_bits_unchanged(rope_type: str):
    """rope_native (the zero-K3 contract lane) built fp32 reads exactly what it read before."""
    position_ids = torch.arange(MAX_POS)[None, :]
    x = torch.zeros(1, MAX_POS, HEAD_DIM, dtype=torch.float32)

    with torch.no_grad():
        contract = _build(rope_type, "float32", rope_native=True).model.rotary_emb(x, position_ids)
        stock = _build(rope_type, "float32", rope_native=False).model.rotary_emb(x, position_ids)

    assert torch.equal(contract[0], stock[0]), f"{rope_type}: contract-lane cos moved"
    assert torch.equal(contract[1], stock[1]), f"{rope_type}: contract-lane sin moved"


def test_native_default_cache_is_lazy_and_follows_execution_device():
    rotary = _build("default", "float32", rope_native=True).model.rotary_emb
    assert rotary._sglang_default_cache is None

    positions = torch.tensor([[0, 663, 960, 1268, 1629]], dtype=torch.long)
    x = torch.zeros(1, positions.shape[1], HEAD_DIM, dtype=torch.float32)
    cos, sin = rotary(x, positions)

    assert rotary._sglang_default_cache is not None
    assert rotary._sglang_default_cache.device == x.device
    assert rotary._sglang_default_cache.dtype == torch.float32
    cached_cos, cached_sin = rotary._sglang_default_cache.index_select(0, positions.flatten()).chunk(2, dim=-1)
    assert torch.equal(cos[..., : HEAD_DIM // 2].reshape_as(cached_cos), cached_cos)
    assert torch.equal(sin[..., : HEAD_DIM // 2].reshape_as(cached_sin), cached_sin)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_architectures_build_default_rope_tables_on_their_serving_devices():
    device = torch.device("cuda")

    glm_config = _config("default")
    glm_config._rope_native = True
    glm_config._rope_class_b = True
    glm_config._glm52_exact_contract = True
    glm_config._qwen35_exact_contract = False
    glm_rotary = RotaryEmbedding(glm_config)

    qwen_config = _config("default")
    qwen_config._rope_native = True
    qwen_config._rope_class_b = False
    qwen_config._glm52_exact_contract = False
    qwen_config._qwen35_exact_contract = True
    qwen_rotary = RotaryEmbedding(qwen_config)

    base, dim = glm_rotary._default_rope_base_and_dim()
    inv_freq_cpu = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    cpu_positions = torch.arange(MAX_POS, dtype=torch.float32)
    cpu_freqs = torch.einsum("i,j->ij", cpu_positions, inv_freq_cpu)
    cpu_table = torch.cat((cpu_freqs.cos(), cpu_freqs.sin()), dim=-1).to(device)

    inv_freq_cuda = inv_freq_cpu.to(device)
    cuda_positions = torch.arange(MAX_POS, dtype=torch.float32, device=device)
    cuda_freqs = torch.einsum("i,j->ij", cuda_positions, inv_freq_cuda)
    cuda_table = torch.cat((cuda_freqs.cos(), cuda_freqs.sin()), dim=-1)

    assert torch.equal(glm_rotary._build_sglang_default_cache(MAX_POS, device), cuda_table)
    assert torch.equal(qwen_rotary._build_sglang_default_cache(MAX_POS, device), cpu_table)
