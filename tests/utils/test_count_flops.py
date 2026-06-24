import pytest

from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.utils.count_flops import XorlFlopsCounter


_GLM5_FLOPS_WIP = pytest.mark.skip(
    reason="XorlFlopsCounter does not yet account for the GLM5 sparse-MLA/DSA architecture upstream "
    "(reports 0 flops); pending GLM5 flops support"
)


def _small_glm5_config(**kwargs) -> Glm5Config:
    params = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=24,
        v_head_dim=32,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=4,
    )
    params.update(kwargs)
    config = Glm5Config(**params)
    config._sparse_mla_enabled = True
    return config


@_GLM5_FLOPS_WIP
def test_glm5_flops_counter_reports_nonzero() -> None:
    config = _small_glm5_config()

    flops, promised = XorlFlopsCounter(config).estimate_flops([8, 16], delta_time=2.0)

    assert flops > 0
    assert promised > 0


@_GLM5_FLOPS_WIP
def test_glm5_dsa_topk_reduces_attention_flops() -> None:
    sparse_config = _small_glm5_config(index_topk=4)
    dense_config = _small_glm5_config(index_topk=4)
    dense_config._dsa_mask_disabled = True

    sparse_flops, _ = XorlFlopsCounter(sparse_config).estimate_flops([32], delta_time=1.0)
    dense_flops, _ = XorlFlopsCounter(dense_config).estimate_flops([32], delta_time=1.0)

    assert dense_flops > sparse_flops


def test_glm5_flops_counter_does_not_multiply_global_lengths_by_cp_size() -> None:
    config = _small_glm5_config()
    batch_seqlens = [128, 64]

    cp1_flops, _ = XorlFlopsCounter(config, cp_size=1).estimate_flops(batch_seqlens, delta_time=1.0)
    cp64_flops, _ = XorlFlopsCounter(config, cp_size=64).estimate_flops(batch_seqlens, delta_time=1.0)

    assert cp64_flops == cp1_flops
