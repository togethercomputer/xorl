from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

import xorl.models.transformers.glm5.exact_lm_head_qlora as lm_head_impl
from xorl.models.transformers.glm5.exact_lm_head_qlora import (
    GLM52_EXACT_TP16_LM_HEAD_CONTRACT_VERSION,
    GLM52_LM_HEAD_HIDDEN_SIZE,
    GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
    GLM52_LM_HEAD_PADDED_VOCAB_SIZE,
    GLM52_LM_HEAD_TP_SIZE,
    GLM52_LM_HEAD_VOCAB_SIZE,
    Glm52ExactTP16LmHeadSelectedLogprob,
    _Glm52ExactTP16LmHeadFunction,
    _local_qlora_surrogate_vjp,
    _rank_order_vocab_from_stacked,
    glm52_lm_head_shard,
)
from xorl.ops.bi_families_v2 import exact_temperature_scale_fp32_logits
from xorl.ops.exact_sampling_transforms import (
    selected_logprob_reference_grad as _selected_logprob_reference_grad,
)
from xorl.ops.exact_sampling_transforms import (
    selected_logprob_reference_grad_filtered as _selected_logprob_reference_grad_filtered,
)
from xorl.ops.exact_sampling_transforms import (
    selected_logprob_reference_grad_partitioned as _selected_logprob_reference_grad_partitioned,
)


def _component(tp_rank: int = 0, tp_group=None) -> Glm52ExactTP16LmHeadSelectedLogprob:
    shard = glm52_lm_head_shard(tp_rank)
    return Glm52ExactTP16LmHeadSelectedLogprob(
        tp_rank=tp_rank,
        vocab_start=shard.vocab_start,
        vocab_end=shard.vocab_end,
        padded_vocab_start=shard.padded_vocab_start,
        padded_vocab_end=shard.padded_vocab_end,
        tp_group=tp_group,
    )


def _meta_operands(rows: int = 2):
    hidden = torch.empty((rows, GLM52_LM_HEAD_HIDDEN_SIZE), dtype=torch.bfloat16, device="meta", requires_grad=True)
    weight = torch.empty(
        (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, GLM52_LM_HEAD_HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="meta",
    )
    lora_A = torch.empty((1, GLM52_LM_HEAD_HIDDEN_SIZE), dtype=torch.float32, device="meta", requires_grad=True)
    lora_B = torch.empty(
        (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, 1),
        dtype=torch.float32,
        device="meta",
        requires_grad=True,
    )
    token_ids = torch.empty((rows,), dtype=torch.int64, device="meta")
    return hidden, weight, lora_A, lora_B, token_ids


def _assert_official_tp16_shards_match_sglang_padding_and_rank_order() -> None:
    shards = [glm52_lm_head_shard(rank) for rank in range(GLM52_LM_HEAD_TP_SIZE)]

    assert GLM52_LM_HEAD_PADDED_VOCAB_SIZE == GLM52_LM_HEAD_VOCAB_SIZE
    assert GLM52_LM_HEAD_VOCAB_SIZE // GLM52_LM_HEAD_TP_SIZE == GLM52_LM_HEAD_LOCAL_VOCAB_SIZE
    assert [shard.vocab_start for shard in shards] == [rank * 9_680 for rank in range(16)]
    assert [shard.vocab_end for shard in shards] == [(rank + 1) * 9_680 for rank in range(16)]
    assert all(shard.padding_rows == 0 for shard in shards)
    assert shards[0].vocab_start == 0
    assert shards[-1].vocab_end == 154_880

    with pytest.raises(TypeError, match="must be an integer"):
        glm52_lm_head_shard(True)
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        glm52_lm_head_shard(16)


def _assert_component_fails_closed_on_shard_ranges() -> None:
    component = _component(7)
    shard = glm52_lm_head_shard(7)

    assert component.contract_version == GLM52_EXACT_TP16_LM_HEAD_CONTRACT_VERSION
    assert component.shard == shard
    assert not tuple(component.parameters())
    assert component.state_dict() == {}
    assert "vocab=[67760,77440)" in repr(component)

    with pytest.raises(ValueError, match="shard range/order mismatch"):
        Glm52ExactTP16LmHeadSelectedLogprob(
            tp_rank=7,
            vocab_start=shard.vocab_start + 1,
            vocab_end=shard.vocab_end,
            padded_vocab_start=shard.padded_vocab_start,
            padded_vocab_end=shard.padded_vocab_end,
        )
    with pytest.raises(TypeError, match="vocab_start must be an integer"):
        Glm52ExactTP16LmHeadSelectedLogprob(
            tp_rank=7,
            vocab_start=float(shard.vocab_start),
            vocab_end=shard.vocab_end,
            padded_vocab_start=shard.padded_vocab_start,
            padded_vocab_end=shard.padded_vocab_end,
        )


def _assert_operand_contract_is_official_local_bf16_rank_one_and_stride_exact() -> None:
    component = _component()
    operands = _meta_operands()
    component._validate_operands(*operands, require_cuda=False)

    hidden, weight, lora_A, lora_B, token_ids = operands
    with pytest.raises(TypeError, match="factor masters must be FP32"):
        component._validate_operands(
            hidden,
            weight,
            lora_A.to(torch.bfloat16),
            lora_B,
            token_ids,
            require_cuda=False,
        )
    with pytest.raises(ValueError, match="local_lora_B shape"):
        component._validate_operands(
            hidden,
            weight,
            lora_A,
            torch.empty((GLM52_LM_HEAD_LOCAL_VOCAB_SIZE - 1, 1), device="meta", requires_grad=True),
            token_ids,
            require_cuda=False,
        )
    with pytest.raises(ValueError, match="sampler-contiguous stride"):
        component._validate_operands(
            hidden,
            torch.empty_strided(
                (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, GLM52_LM_HEAD_HIDDEN_SIZE),
                (1, GLM52_LM_HEAD_LOCAL_VOCAB_SIZE),
                dtype=torch.bfloat16,
                device="meta",
            ),
            lora_A,
            lora_B,
            token_ids,
            require_cuda=False,
        )
    with pytest.raises(RuntimeError, match="token IDs must be in"):
        component._validate_operands(
            hidden,
            weight,
            lora_A,
            lora_B,
            torch.tensor([0, GLM52_LM_HEAD_VOCAB_SIZE], dtype=torch.int64),
            require_cuda=False,
        )
    with pytest.raises(RuntimeError, match="base weight must remain frozen"):
        component._validate_operands(
            hidden,
            weight.requires_grad_(True),
            lora_A,
            lora_B,
            token_ids,
            require_cuda=False,
        )


def _assert_cpu_rejection_happens_before_sglang_import_or_group_use() -> None:
    before = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    component = _component()
    hidden, weight, lora_A, lora_B, token_ids = _meta_operands()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        component(hidden, weight, lora_A, lora_B, token_ids)

    after = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    assert after == before


def _assert_rank_order_vocab_reshape_is_byte_exact_and_has_identity_token_mapping() -> None:
    rows = 2
    local = GLM52_LM_HEAD_LOCAL_VOCAB_SIZE
    row_values = torch.arange(rows * local, dtype=torch.float32).reshape(rows, local)
    shards = torch.stack([row_values + rank * 1_000_000 for rank in range(GLM52_LM_HEAD_TP_SIZE)])

    gathered = _rank_order_vocab_from_stacked(
        shards,
        expected_world_size=GLM52_LM_HEAD_TP_SIZE,
        expected_local_vocab_size=local,
    )
    expected = torch.cat([shards[rank] for rank in range(GLM52_LM_HEAD_TP_SIZE)], dim=-1)

    assert torch.equal(gathered.view(torch.uint8), expected.view(torch.uint8))
    for rank in (0, 1, 7, 15):
        start = rank * local
        assert torch.equal(gathered[:, start : start + local], shards[rank])

    with pytest.raises(ValueError, match="collective rank order"):
        _rank_order_vocab_from_stacked(
            shards.transpose(1, 2).contiguous().transpose(1, 2),
            expected_world_size=GLM52_LM_HEAD_TP_SIZE,
            expected_local_vocab_size=local,
        )


def _assert_local_fp32_surrogate_vjp_matches_standalone_qlora_reference() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8).sub_(7).div_(19).to(torch.bfloat16)
    weight = torch.arange(56, dtype=torch.float32).reshape(7, 8).sub_(23).div_(37).to(torch.bfloat16)
    effective_A = torch.arange(8, dtype=torch.float32).sub_(3).div_(11).reshape(1, 8).to(torch.bfloat16)
    effective_B = torch.arange(7, dtype=torch.float32).sub_(2).div_(13).reshape(7, 1).to(torch.bfloat16)
    grad_logits = torch.arange(21, dtype=torch.float32).reshape(3, 7).sub_(8).div_(17)

    grad_hidden, grad_A, grad_B = _local_qlora_surrogate_vjp(
        hidden,
        weight,
        effective_A,
        effective_B,
        grad_logits,
        needs_input_grad=(True, True, True),
    )

    base_hidden = hidden.detach().clone().requires_grad_(True)
    F.linear(base_hidden, weight).backward(grad_logits.to(torch.bfloat16))
    lora_hidden = hidden.float().detach().requires_grad_(True)
    reference_A = effective_A.float().detach().requires_grad_(True)
    reference_B = effective_B.float().detach().requires_grad_(True)
    F.linear(F.linear(lora_hidden, reference_A), reference_B).backward(grad_logits)

    assert grad_hidden.dtype is torch.float32
    assert grad_A.dtype is torch.float32
    assert grad_B.dtype is torch.float32
    assert torch.equal(grad_hidden, base_hidden.grad.float() + lora_hidden.grad)
    assert torch.equal(grad_A, reference_A.grad)
    assert torch.equal(grad_B, reference_B.grad)


def test_exact_lm_head_cpu_contract(monkeypatch) -> None:
    _assert_official_tp16_shards_match_sglang_padding_and_rank_order()
    _assert_component_fails_closed_on_shard_ranges()
    _assert_tp_group_validation_rejects_size_order_rank_and_backend(monkeypatch)
    monkeypatch.undo()
    _assert_operand_contract_is_official_local_bf16_rank_one_and_stride_exact()
    _assert_cpu_rejection_happens_before_sglang_import_or_group_use()
    _assert_rank_order_vocab_reshape_is_byte_exact_and_has_identity_token_mapping()
    _assert_local_fp32_surrogate_vjp_matches_standalone_qlora_reference()
    _assert_custom_boundary_is_grad_enabled_and_saves_effective_factor_bytes()


def _assert_custom_boundary_is_grad_enabled_and_saves_effective_factor_bytes() -> None:
    captures = {}

    class FakeComponent:
        @staticmethod
        def _exact_forward_value(hidden, weight, effective_A, effective_B, token_ids, temperature):
            captures["A"] = effective_A.clone()
            captures["B"] = effective_B.clone()
            captures["temperature"] = temperature.clone()
            return hidden.float().sum(dim=-1) + weight.float().sum() * 0.0 + token_ids.float() * 0.0

        @staticmethod
        def _surrogate_vjp(
            hidden,
            weight,
            effective_A,
            effective_B,
            token_ids,
            grad_logprob,
            temperature,
            *,
            needs_input_grad,
        ):
            del weight, effective_A, effective_B, token_ids, needs_input_grad
            captures["backward_temperature"] = temperature.clone()
            return grad_logprob.unsqueeze(-1).expand_as(hidden).float(), torch.ones(1, 3), torch.ones(5, 1)

    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16).requires_grad_(True)
    weight = torch.zeros(5, 3, dtype=torch.bfloat16)
    lora_A = torch.tensor([[0.1001, -0.2002, 0.3003]], requires_grad=True)
    lora_B = torch.tensor([[0.1101], [-0.2202], [0.3303], [-0.4404], [0.5505]], requires_grad=True)
    token_ids = torch.tensor([0, 4], dtype=torch.int64)
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)

    logprob = _Glm52ExactTP16LmHeadFunction.apply(
        hidden,
        weight,
        lora_A,
        lora_B,
        token_ids,
        temperature,
        (None, None, None),
        FakeComponent(),
    )
    assert logprob.requires_grad
    assert torch.equal(captures["A"], lora_A.detach().to(torch.bfloat16))
    assert torch.equal(captures["B"], lora_B.detach().to(torch.bfloat16))
    assert torch.equal(captures["temperature"], temperature)

    logprob.sum().backward()
    assert torch.equal(hidden.grad, torch.ones_like(hidden))
    assert torch.equal(lora_A.grad, torch.ones_like(lora_A))
    assert torch.equal(lora_B.grad, torch.ones_like(lora_B))
    assert torch.equal(captures["backward_temperature"], temperature)


def test_temperature_reference_gradient_scales_each_row_before_softmax() -> None:
    logits = torch.tensor([[1.25, -0.5, 0.75], [-1.0, 2.0, 0.25]], dtype=torch.float32)
    token_ids = torch.tensor([2, 1], dtype=torch.int64)
    grad_logprob = torch.tensor([0.5, -0.75], dtype=torch.float32)
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)

    actual = _selected_logprob_reference_grad(logits, token_ids, grad_logprob, temperature)

    reference_logits = logits.detach().requires_grad_(True)
    selected = (
        F.log_softmax(
            reference_logits * (1.0 / temperature).unsqueeze(1),
            dim=-1,
        )
        .gather(1, token_ids.unsqueeze(1))
        .squeeze(1)
    )
    (expected,) = torch.autograd.grad(selected, reference_logits, grad_outputs=grad_logprob)
    assert torch.equal(actual, expected)


def test_identity_reference_gradient_is_unchanged_beside_filtered_row() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0], [2.0, 0.0, -1.0]], dtype=torch.float32)
    token_ids = torch.tensor([0, 0], dtype=torch.int64)
    grad_logprob = torch.tensor([0.5, -0.75], dtype=torch.float32)
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)
    support = torch.tensor([[True, True, True], [True, False, False]])
    identity_rows = torch.tensor([True, False])

    actual = _selected_logprob_reference_grad_partitioned(
        logits,
        token_ids,
        grad_logprob,
        temperature,
        support,
        identity_rows,
    )
    native = _selected_logprob_reference_grad(logits[:1], token_ids[:1], grad_logprob[:1], temperature[:1])
    filtered = _selected_logprob_reference_grad_filtered(
        logits[1:], token_ids[1:], grad_logprob[1:], temperature[1:], support[1:]
    )

    assert torch.equal(actual[0], native[0])
    assert torch.equal(actual[1], filtered[0])


def _assert_tp_group_validation_rejects_size_order_rank_and_backend(monkeypatch) -> None:
    group = object()
    component = _component(3, tp_group=group)
    state = {
        "world": 16,
        "group_rank": 3,
        "global_rank": 3,
        "ranks": list(range(16)),
        "backend": "nccl",
    }
    monkeypatch.setattr(lm_head_impl.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lm_head_impl.dist, "get_world_size", lambda _group: state["world"])
    monkeypatch.setattr(
        lm_head_impl.dist,
        "get_rank",
        lambda requested_group=None: state["global_rank"] if requested_group is None else state["group_rank"],
    )
    monkeypatch.setattr(lm_head_impl.dist, "get_process_group_ranks", lambda _group: state["ranks"])
    monkeypatch.setattr(lm_head_impl.dist, "get_backend", lambda _group: state["backend"])

    assert component._validate_tp_group() is group

    state["world"] = 8
    with pytest.raises(RuntimeError, match="requires TP16"):
        component._validate_tp_group()
    state["world"] = 16
    state["ranks"] = [1, 0, *range(2, 16)]
    with pytest.raises(RuntimeError, match="gather order"):
        component._validate_tp_group()
    state["ranks"] = list(range(16))
    state["group_rank"] = 4
    with pytest.raises(RuntimeError, match="shard/group rank mismatch"):
        component._validate_tp_group()
    state["group_rank"] = 3
    state["backend"] = "gloo"
    with pytest.raises(RuntimeError, match="must use NCCL"):
        component._validate_tp_group()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_official_local_shard_literal_v2_bytes_tail_and_surrogate_gradients() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified GLM-5.2 exact LM-head component requires Hopper")

    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
    from sglang.srt.batch_invariant_ops import (
        head_v2_full_logits_with_lse,
        head_v2_selected_logprob_from_logits,
    )
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    for rank in range(GLM52_LM_HEAD_TP_SIZE):
        expected = glm52_lm_head_shard(rank)
        actual = VocabParallelEmbedding._get_indices(
            GLM52_LM_HEAD_PADDED_VOCAB_SIZE,
            GLM52_LM_HEAD_PADDED_VOCAB_SIZE,
            GLM52_LM_HEAD_VOCAB_SIZE,
            GLM52_LM_HEAD_VOCAB_SIZE,
            rank,
            GLM52_LM_HEAD_TP_SIZE,
        )
        assert (
            actual.org_vocab_start_index,
            actual.org_vocab_end_index,
            actual.padded_org_vocab_start_index,
            actual.padded_org_vocab_end_index,
            actual.num_org_vocab_padding,
        ) == (
            expected.vocab_start,
            expected.vocab_end,
            expected.padded_vocab_start,
            expected.padded_vocab_end,
            0,
        )

    device = torch.device("cuda")
    component = _component(0).to(device)
    rows = 2
    torch.manual_seed(20260807)
    hidden = torch.empty((rows, GLM52_LM_HEAD_HIDDEN_SIZE), dtype=torch.bfloat16, device=device).uniform_(-0.125, 0.125)
    local_weight = torch.empty(
        (GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, GLM52_LM_HEAD_HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.0625, 0.0625)
    lora_A = (
        torch.arange(GLM52_LM_HEAD_HIDDEN_SIZE, dtype=torch.float32, device=device)
        .sub_(3_071)
        .div_(16_384)
        .reshape(1, GLM52_LM_HEAD_HIDDEN_SIZE)
        .requires_grad_(True)
    )
    lora_B = (
        torch.arange(GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, dtype=torch.float32, device=device)
        .sub_(4_839)
        .div_(32_768)
        .reshape(GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, 1)
        .requires_grad_(True)
    )
    hidden_bytes = hidden.view(torch.uint8).clone()
    weight_bytes = local_weight.view(torch.uint8).clone()
    A_bytes = lora_A.view(torch.uint8).clone()
    B_bytes = lora_B.view(torch.uint8).clone()
    effective_A = lora_A.detach().to(torch.bfloat16).contiguous()
    effective_B = lora_B.detach().to(torch.bfloat16).contiguous()

    batch_info = lm_head_impl._single_adapter_lm_head_batch_info(device.index, rows)
    direct_base, _direct_lse = head_v2_full_logits_with_lse(hidden, local_weight)
    direct_a = sgemm_lora_a_fwd(hidden, effective_A.unsqueeze(0), batch_info)
    direct_delta = sgemm_lora_b_fwd(direct_a, effective_B.unsqueeze(0), batch_info)
    expected_local = sgemm_lora_b_fwd(
        direct_a,
        effective_B.unsqueeze(0),
        batch_info,
        base_output=direct_base.clone(),
    )
    actual_local = component._exact_local_logits(hidden, local_weight, effective_A, effective_B)
    warm_local = component._exact_local_logits(hidden, local_weight, effective_A, effective_B)

    assert direct_base.dtype is torch.float32
    assert direct_a.dtype is torch.bfloat16
    assert direct_delta.dtype is torch.bfloat16
    assert torch.equal(expected_local.view(torch.uint8), (direct_base + direct_delta.float()).view(torch.uint8))
    assert torch.equal(actual_local.view(torch.uint8), expected_local.view(torch.uint8))
    assert torch.equal(warm_local.view(torch.uint8), actual_local.view(torch.uint8))
    assert torch.equal(hidden.view(torch.uint8), hidden_bytes)
    assert torch.equal(local_weight.view(torch.uint8), weight_bytes)
    assert torch.equal(lora_A.view(torch.uint8), A_bytes)
    assert torch.equal(lora_B.view(torch.uint8), B_bytes)
    assert torch.equal(effective_A.view(torch.uint8), lora_A.detach().to(torch.bfloat16).view(torch.uint8))
    assert torch.equal(effective_B.view(torch.uint8), lora_B.detach().to(torch.bfloat16).view(torch.uint8))

    stacked = torch.stack(
        [actual_local + torch.tensor(rank / 128, dtype=torch.float32, device=device) for rank in range(16)]
    )
    gathered = _rank_order_vocab_from_stacked(
        stacked,
        expected_world_size=16,
        expected_local_vocab_size=GLM52_LM_HEAD_LOCAL_VOCAB_SIZE,
    )
    expected_gathered = torch.cat([stacked[rank] for rank in range(16)], dim=-1)
    assert torch.equal(gathered.view(torch.uint8), expected_gathered.view(torch.uint8))

    token_ids = torch.tensor([0, GLM52_LM_HEAD_VOCAB_SIZE - 1], dtype=torch.int64, device=device)
    for temperature in (
        None,
        torch.ones(2, dtype=torch.float32, device=device),
        torch.tensor([0.7, 1.3], dtype=torch.float32, device=device),
    ):
        actual_logprob = component._selected_logprob_from_gathered(
            gathered,
            token_ids,
            temperature,
        )
        score_logits = gathered if temperature is None else exact_temperature_scale_fp32_logits(gathered, temperature)
        expected_logprob, _, _ = head_v2_selected_logprob_from_logits(
            score_logits,
            token_ids,
            temperature=None,
        )
        assert torch.equal(actual_logprob.view(torch.uint8), expected_logprob.view(torch.uint8))

    grad_local_logits = (
        torch.arange(rows * GLM52_LM_HEAD_LOCAL_VOCAB_SIZE, dtype=torch.float32, device=device)
        .remainder_(127)
        .sub_(63)
        .div_(64)
        .reshape(rows, GLM52_LM_HEAD_LOCAL_VOCAB_SIZE)
    )
    grad_hidden, grad_A, grad_B = _local_qlora_surrogate_vjp(
        hidden,
        local_weight,
        effective_A,
        effective_B,
        grad_local_logits,
        needs_input_grad=(True, True, True),
    )
    base_hidden = hidden.detach().clone().requires_grad_(True)
    F.linear(base_hidden, local_weight).backward(grad_local_logits.to(torch.bfloat16))
    lora_hidden = hidden.float().detach().requires_grad_(True)
    reference_A = effective_A.float().detach().requires_grad_(True)
    reference_B = effective_B.float().detach().requires_grad_(True)
    F.linear(F.linear(lora_hidden, reference_A), reference_B).backward(grad_local_logits)

    assert grad_hidden.dtype is grad_A.dtype is grad_B.dtype is torch.float32
    assert torch.equal(grad_hidden, base_hidden.grad.float() + lora_hidden.grad)
    assert torch.equal(grad_A, reference_A.grad)
    assert torch.equal(grad_B, reference_B.grad)
