from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xorl.distributed.canonical_moe import LogicalRowOwnership
from xorl.models.auto import resolve_cross_entropy_mode
from xorl.models.transformers.deepseek_v4 import DeepseekV4Config
from xorl.models.transformers.deepseek_v4.exact_contract import (
    DSV4_FLASH_COMPRESS_RATIOS,
    DSV4_FLASH_LOGICAL_FACTOR_COUNT,
    DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT,
    DSV4_FLASH_REQUIRED_TARGET_MODULES,
    DSV4_FLASH_ROUTED_BANK_COUNT,
    DSV4_FLASH_TARGET_ENTITY_COUNT,
    bind_dsv4_flash_adapter_inventory,
    build_dsv4_flash_adapter_inventory,
    validate_dsv4_flash_adapter_program,
    validate_dsv4_flash_official_geometry,
)
from xorl.models.transformers.deepseek_v4.exact_lm_head import (
    DSV4_LM_HEAD_LOCAL_VOCAB_SIZE,
    DSV4_LM_HEAD_TP_SIZE,
    DSV4_LM_HEAD_VOCAB_SIZE,
    Dsv4ExactTP8LmHeadLoraLinear,
    bind_dsv4_exact_lm_head,
    dsv4_lm_head_shard,
)
from xorl.ops.families.dsv4.exact_attention import (
    _causal_window_indices,
    _hybrid_indices_for_positions,
    _hybrid_prefill_indices,
    _window_indices_for_positions,
)


pytestmark = pytest.mark.cpu


def _official_config() -> DeepseekV4Config:
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        num_hash_layers=3,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        compress_rope_theta=160000,
        compress_ratios=list(DSV4_FLASH_COMPRESS_RATIOS),
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        hidden_act="silu",
        swiglu_limit=10.0,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        expert_dtype="fp4",
        num_nextn_predict_layers=1,
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
        rope_scaling={
            "type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )
    return config


def test_official_geometry_is_fail_closed() -> None:
    config = _official_config()
    validate_dsv4_flash_official_geometry(config)

    config.quantization_config = SimpleNamespace(**config.quantization_config)
    config.rope_scaling = SimpleNamespace(**config.rope_scaling)
    validate_dsv4_flash_official_geometry(config)

    config.compress_ratios[2] = 128
    with pytest.raises(ValueError, match="C0/C4/C128 schedule"):
        validate_dsv4_flash_official_geometry(config)


@pytest.mark.parametrize("dp_size,cp_size", [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_dsv4_owner_plane_accepts_every_dp_cp_factorization(dp_size: int, cp_size: int) -> None:
    ordinals = []
    for dp_rank in range(dp_size):
        for cp_rank in range(cp_size):
            ownership = LogicalRowOwnership(
                dp_size,
                cp_size,
                dp_rank,
                cp_rank,
                contributor_count=8,
            )
            ordinals.append(ownership.source_ordinal)
    assert ordinals == list(range(8))


@pytest.mark.parametrize(
    ("model_type", "architectures"),
    [
        ("deepseek_v4", ["OtherForCausalLM"]),
        ("other", ["DeepseekV4ForCausalLM"]),
    ],
)
def test_official_geometry_requires_both_model_identifiers(model_type: str, architectures: list[str]) -> None:
    config = _official_config()
    config.model_type = model_type
    config.architectures = architectures
    with pytest.raises(ValueError, match="model_type='deepseek_v4'"):
        validate_dsv4_flash_official_geometry(config)


def test_adapter_program_rejects_partial_targets_and_non_rank_one() -> None:
    validate_dsv4_flash_adapter_program(
        adapter_rank=1,
        adapter_alpha=1,
        target_modules=DSV4_FLASH_REQUIRED_TARGET_MODULES,
    )
    with pytest.raises(ValueError, match="rank-1/alpha-1"):
        validate_dsv4_flash_adapter_program(adapter_rank=2, adapter_alpha=1)
    with pytest.raises(ValueError, match="target_modules mismatch"):
        validate_dsv4_flash_adapter_program(
            adapter_rank=1,
            adapter_alpha=1,
            target_modules=DSV4_FLASH_REQUIRED_TARGET_MODULES - {"lm_head"},
        )


def test_exact_loss_mode_rejects_non_tp_aware_bi_fused_path() -> None:
    config = _official_config()
    config._dsv4_flash_exact_mode = True

    assert resolve_cross_entropy_mode(config, None) == "compiled"
    assert resolve_cross_entropy_mode(config, "compiled") == "compiled"
    with pytest.raises(ValueError, match="requires ce_mode='compiled'"):
        resolve_cross_entropy_mode(config, "bi_fused")


def test_inventory_derives_exact_345_non_routed_43_banks_and_948_factors() -> None:
    inventory = build_dsv4_flash_adapter_inventory(_official_config())

    assert len(inventory.targets) == DSV4_FLASH_TARGET_ENTITY_COUNT == 388
    assert len(inventory.factors) == DSV4_FLASH_LOGICAL_FACTOR_COUNT == 948
    assert (
        sum(target.kind == "native_mxfp4_routed_bank" for target in inventory.targets) == DSV4_FLASH_ROUTED_BANK_COUNT
    )
    assert DSV4_FLASH_ROUTED_BANK_COUNT == 43
    assert len(inventory.targets) - DSV4_FLASH_ROUTED_BANK_COUNT == DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT
    assert DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT == 345
    assert {factor.dtype for factor in inventory.factors} == {torch.float32}
    assert inventory.role_counts == {
        "attention.wkv": 43,
        "attention.wo_a": 43,
        "attention.wo_b": 43,
        "attention.wq_a": 43,
        "attention.wq_b": 43,
        "output.lm_head": 1,
        "routed_expert.bank": 43,
        "shared_expert.down_proj": 43,
        "shared_expert.gate_proj": 43,
        "shared_expert.up_proj": 43,
    }


def test_inventory_shapes_cover_fused_attention_experts_and_lm_head() -> None:
    inventory = build_dsv4_flash_adapter_inventory(_official_config())
    by_name = {factor.name: factor for factor in inventory.factors}

    prefix = "model.layers.0"
    assert by_name[f"{prefix}.self_attn.wq_a.lora_A"].shape == (1, 4096)
    assert by_name[f"{prefix}.self_attn.wq_a.lora_B"].shape == (1024, 1)
    assert by_name[f"{prefix}.self_attn.wkv.lora_B"].shape == (512, 1)
    assert by_name[f"{prefix}.self_attn.wq_b.lora_B"].shape == (32768, 1)
    assert by_name[f"{prefix}.self_attn.wo_a.lora_A"].shape == (1, 4096)
    assert by_name[f"{prefix}.self_attn.wo_a.lora_B"].shape == (8192, 1)
    assert by_name[f"{prefix}.mlp.experts.gate_proj_lora_A"].shape == (256, 4096, 1)
    assert by_name[f"{prefix}.mlp.experts.down_proj_lora_B"].shape == (256, 1, 4096)
    assert by_name["lm_head.lora_A"].shape == (1, 4096)
    assert by_name["lm_head.lora_B"].shape == (129280, 1)


def test_live_inventory_requires_every_fp32_trainable_factor() -> None:
    config = _official_config()
    inventory = build_dsv4_flash_adapter_inventory(config)

    class _LiveAdapter:
        def __init__(self):
            self.config = config
            self._parameters = {
                factor.name: torch.nn.Parameter(torch.empty(factor.shape, dtype=torch.float32, device="meta"))
                for factor in inventory.factors
            }

        def named_parameters(self):
            return iter(self._parameters.items())

    live = _LiveAdapter()
    assert bind_dsv4_flash_adapter_inventory(live) == inventory
    assert live._dsv4_adapter_inventory == inventory

    live._parameters.pop(next(iter(live._parameters)))
    with pytest.raises(RuntimeError, match="complete 948-factor"):
        bind_dsv4_flash_adapter_inventory(live)


def test_exact_lm_head_uses_eight_contiguous_physical_vocab_shards() -> None:
    assert DSV4_LM_HEAD_TP_SIZE == 8
    assert DSV4_LM_HEAD_LOCAL_VOCAB_SIZE == 16160
    shards = [dsv4_lm_head_shard(rank) for rank in range(DSV4_LM_HEAD_TP_SIZE)]
    assert [shard.vocab_start for shard in shards] == [rank * 16160 for rank in range(8)]
    assert [shard.vocab_end for shard in shards] == [(rank + 1) * 16160 for rank in range(8)]
    assert shards[-1].vocab_end == DSV4_LM_HEAD_VOCAB_SIZE
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        dsv4_lm_head_shard(8)


def test_exact_lm_head_rejects_ordinary_full_weight_value_paths() -> None:
    head = Dsv4ExactTP8LmHeadLoraLinear(
        4096,
        DSV4_LM_HEAD_VOCAB_SIZE,
        r=1,
        lora_alpha=1,
        device="meta",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="selected-logprob"):
        head(torch.empty(1, 4096, device="meta", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="selected-logprob"):
        head.get_delta_weight()


def test_exact_lm_head_binds_to_a_pp2_stage_local_tp8_group(monkeypatch) -> None:
    from xorl.distributed import parallel_state as parallel_state_impl  # noqa: PLC0415
    from xorl.lora.modules.linear import LoraLinear  # noqa: PLC0415

    group = object()
    device_mesh = SimpleNamespace(
        mesh=torch.arange(16).reshape(2, 8),
        mesh_dim_names=("pp", "dp_shard"),
        get_coordinate=lambda: [1, 0],
    )
    state = SimpleNamespace(
        dp_size=8,
        cp_size=1,
        dp_rank=0,
        cp_rank=0,
        cp_enabled=False,
        tp_size=1,
        ep_size=8,
        ep_group=group,
        lm_head_tp_size=8,
        lm_head_tp_group=group,
        device_mesh=device_mesh,
    )
    monkeypatch.setattr(parallel_state_impl, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda actual: 8)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda actual: tuple(range(8, 16)))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda actual=None: 0 if actual is group else 8)
    model = SimpleNamespace(
        lm_head=LoraLinear(
            4096,
            DSV4_LM_HEAD_VOCAB_SIZE,
            r=1,
            lora_alpha=1,
            device=torch.device("meta"),
            dtype=torch.bfloat16,
        )
    )

    bind_dsv4_exact_lm_head(model)

    assert isinstance(model.lm_head, Dsv4ExactTP8LmHeadLoraLinear)
    assert model.lm_head._dsv4_exact_selected_logprob.physical_ranks == tuple(range(8, 16))
    assert model.lm_head._dsv4_exact_selected_logprob.source_ordinal == 0


def test_exact_lm_head_rejects_a_group_outside_the_owner_plane(monkeypatch) -> None:
    from xorl.distributed import parallel_state as parallel_state_impl  # noqa: PLC0415
    from xorl.lora.modules.linear import LoraLinear  # noqa: PLC0415

    group = object()
    ep_group = object()
    state = SimpleNamespace(
        dp_size=8,
        cp_size=1,
        dp_rank=0,
        cp_rank=0,
        cp_enabled=False,
        tp_size=1,
        ep_size=8,
        ep_group=ep_group,
        lm_head_tp_size=8,
        lm_head_tp_group=group,
        device_mesh=SimpleNamespace(
            mesh=torch.arange(16).reshape(2, 8),
            mesh_dim_names=("pp", "dp_shard"),
            get_coordinate=lambda: [1, 0],
        ),
    )
    monkeypatch.setattr(parallel_state_impl, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda actual: 8)
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda actual: tuple(range(4, 12)) if actual is group else tuple(range(8, 16)),
    )
    model = SimpleNamespace(
        lm_head=LoraLinear(
            4096,
            DSV4_LM_HEAD_VOCAB_SIZE,
            r=1,
            lora_alpha=1,
            device=torch.device("meta"),
            dtype=torch.bfloat16,
        )
    )

    with pytest.raises(RuntimeError, match="owner plane"):
        bind_dsv4_exact_lm_head(model)


def test_exact_c4_short_prefill_uses_compact_prefix_then_swa_order() -> None:
    indices, lengths, capacity = _hybrid_prefill_indices(10, 4, torch.device("cpu"))

    assert capacity == 2
    assert lengths.tolist() == [1, 2, 3, 5, 6, 7, 8, 10, 11, 12]
    assert indices[3, :5].tolist() == [0, 2, 3, 4, 5]
    assert indices[7, :10].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert indices[9, :12].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert torch.all(indices[:, 12:] == -1)


def test_exact_c128_short_prefill_reserves_ignored_cache_slot() -> None:
    indices, lengths, capacity = _hybrid_prefill_indices(10, 128, torch.device("cpu"))

    assert capacity == 1
    assert lengths.tolist() == list(range(1, 11))
    assert indices[9, :10].tolist() == list(range(1, 11))
    assert torch.all(indices[:, 10:] == -1)


@pytest.mark.parametrize("dp_size,cp_size", [(1, 8), (2, 4)])
@pytest.mark.parametrize("ratio", [4, 128])
def test_exact_cp_local_queries_select_the_same_global_attention_rows(
    dp_size: int,
    cp_size: int,
    ratio: int,
) -> None:
    """DP factorization and PP placement do not change the CP attention row contract."""

    sequence_length = 128
    local_length = sequence_length // cp_size
    full_window = _causal_window_indices(1, sequence_length, torch.device("cpu"))[0]
    full_hybrid, full_lengths, compressed_capacity = _hybrid_prefill_indices(
        sequence_length,
        ratio,
        torch.device("cpu"),
    )

    for _dp_rank in range(dp_size):
        for cp_rank in range(cp_size):
            start = cp_rank * local_length
            stop = start + local_length
            query_positions = torch.arange(start, stop, dtype=torch.int64)

            local_window = _window_indices_for_positions(query_positions)
            local_hybrid, local_lengths = _hybrid_indices_for_positions(
                query_positions,
                ratio,
                compressed_capacity,
            )

            assert torch.equal(local_window, full_window[start:stop])
            assert torch.equal(local_hybrid, full_hybrid[start:stop])
            assert torch.equal(local_lengths, full_lengths[start:stop])
