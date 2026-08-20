from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import xorl.models.transformers.glm5.modeling_glm5 as modeling_glm5
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.modeling_glm5 import Glm5Attention


pytestmark = pytest.mark.cpu


def _tiny_config() -> Glm5Config:
    config = Glm5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        v_head_dim=5,
        max_position_embeddings=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=2,
        num_nextn_predict_layers=0,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    config._activation_native = True
    return config


class _ExactKvBSpy(Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA):
    """Keep the real integration type without allocating official weights."""

    def __init__(self, *, kv_lora_rank: int, v_head_dim: int) -> None:
        nn.Module.__init__(self)
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.calls: list[tuple[str, torch.Tensor, object | None]] = []

    def forward(
        self,
        input: torch.Tensor,
        *,
        branch: str,
        batch_info: object | None = None,
    ) -> torch.Tensor:
        self.calls.append((branch, input.detach().clone(), batch_info))
        if branch == "q":
            width = self.kv_lora_rank
            offset = 10.0
        elif branch == "v":
            width = self.v_head_dim
            offset = 20.0
        else:  # pragma: no cover - the integration must never request another branch
            raise AssertionError(f"unexpected exact kv_b branch {branch!r}")
        return input[..., :1].expand(*input.shape[:-1], width) + offset


class _IndexerSpy(nn.Module):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk
        self.project_calls: list[dict[str, object]] = []

    def forward(self, hidden_states: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return torch.zeros((*hidden_states.shape[:2], self.topk), dtype=torch.long)

    def project(
        self,
        hidden_states: torch.Tensor,
        *_args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.project_calls.append(kwargs)
        shape = (*hidden_states.shape[:2], 1)
        return hidden_states.new_zeros(shape), hidden_states.new_zeros(shape), hidden_states.new_zeros(shape)

    def select_topk(self, index_q: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return torch.zeros((*index_q.shape[:2], self.topk), dtype=torch.long)


def _attention_with_simple_projections() -> Glm5Attention:
    attention = Glm5Attention(_tiny_config(), layer_idx=0)
    attention.q_a_layernorm = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.indexer = _IndexerSpy(attention.config.index_topk)
    return attention


def _assert_sparse_exact_kv_b_routes_q_and_both_v_sites_without_weight_materialization(
    monkeypatch: pytest.MonkeyPatch,
    cp_enabled: bool,
) -> None:
    torch.manual_seed(0)
    attention = _attention_with_simple_projections()
    exact_kv_b = _ExactKvBSpy(
        kv_lora_rank=attention.kv_lora_rank,
        v_head_dim=attention.v_head_dim,
    )
    attention.kv_b_proj = exact_kv_b
    split_kv_b_weight = MagicMock(side_effect=AssertionError("exact kv_b must not materialize B @ A"))
    monkeypatch.setattr(attention, "_split_kv_b_weight", split_kv_b_weight)
    monkeypatch.setattr(
        modeling_glm5,
        "glm5_apply_rotary_pos_emb",
        lambda q, k, *_args, **_kwargs: (q, k),
    )

    group = object()
    parallel_state = SimpleNamespace(
        cp_enabled=cp_enabled,
        ringattn_enabled=False,
        ulysses_group=group,
    )
    monkeypatch.setattr(modeling_glm5, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(modeling_glm5.dist, "get_rank", lambda _group: 1)
    monkeypatch.setattr(
        attention,
        "_gather_ulysses_sequence_no_grad",
        lambda tensor, _group: torch.cat((tensor, tensor), dim=1),
    )
    monkeypatch.setattr(
        modeling_glm5,
        "gather_outputs",
        lambda tensor, gather_dim, **_kwargs: torch.cat((tensor, tensor), dim=gather_dim),
    )

    batch_size, seq_len = 2, 3
    attn_latent = torch.arange(
        batch_size * seq_len * attention.num_heads * attention.kv_lora_rank,
        dtype=torch.float32,
    ).reshape(batch_size, seq_len, attention.num_heads, attention.kv_lora_rank)
    sparse_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]] = []

    def fake_sparse_mla(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        sparse_calls.append((q, kv, indices, kwargs))
        return attn_latent

    monkeypatch.setattr(modeling_glm5, "sparse_mla_dispatch", fake_sparse_mla)

    hidden_states = torch.randn(batch_size, seq_len, attention.config.hidden_size)
    position_embeddings = (
        torch.zeros(batch_size, seq_len, attention.qk_rope_head_dim),
        torch.zeros(batch_size, seq_len, attention.qk_rope_head_dim),
    )
    sampler_prefill_lengths = torch.tensor([4096, 4096], dtype=torch.int64)
    output, weights = attention.forward_sparse(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        sampler_prefill_lengths=sampler_prefill_lengths,
    )

    assert weights is None
    assert [branch for branch, _input, _batch_info in exact_kv_b.calls] == ["q", "v"]
    q_call, v_call = exact_kv_b.calls
    assert q_call[1].shape == (
        batch_size,
        seq_len,
        attention.num_heads,
        attention.qk_nope_head_dim,
    )
    assert q_call[2] is None
    torch.testing.assert_close(v_call[1], attn_latent)
    assert v_call[2] is None
    expected_value = (
        attn_latent[..., :1].expand(
            batch_size,
            seq_len,
            attention.num_heads,
            attention.v_head_dim,
        )
        + 20.0
    )
    torch.testing.assert_close(output, expected_value.reshape(batch_size, seq_len, -1))
    split_kv_b_weight.assert_not_called()
    assert len(sparse_calls) == 1
    if cp_enabled:
        assert sparse_calls[0][1].shape[1] == seq_len * 2
        assert sparse_calls[0][3]["query_offset"] == seq_len
        assert len(attention.indexer.project_calls) == 1
        assert attention.indexer.project_calls[0]["query_offset"] == seq_len
        assert attention.indexer.project_calls[0]["sampler_prefill_lengths"] is sampler_prefill_lengths
    else:
        assert sparse_calls[0][1].shape[1] == seq_len
        assert "query_offset" not in sparse_calls[0][3]


def test_sparse_exact_kv_b_routes_q_and_both_v_sites_without_weight_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for cp_enabled in (False, True):
        with monkeypatch.context() as case_patch:
            _assert_sparse_exact_kv_b_routes_q_and_both_v_sites_without_weight_materialization(
                case_patch,
                cp_enabled,
            )
