import pytest
import torch
import torch.nn.functional as F

import xorl.distributed.moe.alltoall as alltoall_module
from xorl.distributed.moe.utils import permute, permuted_weights, sort_chunks_by_idxs, unpermute


pytestmark = [pytest.mark.cpu]


def test_permuted_weights_follow_expert_sorted_token_order():
    tokens = torch.arange(12, dtype=torch.float32).view(4, 3)
    selected_experts = torch.tensor(
        [
            [2, 0],
            [1, 2],
            [0, 1],
            [2, 1],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )

    expert_mask = F.one_hot(selected_experts, num_classes=3).permute(2, 1, 0)
    routing_map = expert_mask.sum(dim=1)
    _, permutation_mapping = permute(tokens, routing_map)

    expected = torch.tensor([0.1, 0.7, 0.6, 0.3, 0.2, 0.9, 0.4, 0.8], dtype=torch.float32)

    torch.testing.assert_close(permuted_weights(routing_weights, selected_experts, 3), expected)
    torch.testing.assert_close(permutation_mapping, torch.tensor([0, 2, 1, 2, 3, 0, 1, 3]))


def test_unpermute_only_scatter_adds_preweighted_outputs():
    expert_outputs = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    permutation_mapping = torch.tensor([0, 1, 0], dtype=torch.long)

    output = unpermute(
        expert_outputs,
        hidden_states_shape=torch.Size([2, 2]),
        permutation_mapping=permutation_mapping,
    )

    expected = torch.tensor(
        [
            [4.0, 4.0],
            [2.0, 2.0],
        ]
    )
    torch.testing.assert_close(output, expected)


def test_alltoall_pre_dispatch_routes_scores_with_received_token_order(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

    num_experts = 4
    selection = torch.tensor([0, 2, 3, 5, 7], dtype=torch.long)
    input_splits = [4, 4]
    output_splits = [3, 2]
    num_tokens_per_expert = torch.tensor([[1, 2], [1, 1]], dtype=torch.int64)
    sum_tokens = torch.tensor([2, 3], dtype=torch.int64)

    def fake_preprocess(*, expert_mask, num_experts, ep_group):
        return input_splits, output_splits, num_tokens_per_expert, sum_tokens

    def fake_all_to_all(group, input, output_split_sizes, input_split_sizes):
        return input.index_select(0, selection.to(input.device))

    monkeypatch.setattr(alltoall_module, "preprocess", fake_preprocess)
    monkeypatch.setattr(alltoall_module, "all_to_all", fake_all_to_all)

    hidden_states = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    selected_experts = torch.tensor(
        [
            [2, 0],
            [1, 2],
            [0, 1],
            [2, 1],
        ],
        dtype=torch.long,
    )
    routing_weights = torch.tensor(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    routing_map = expert_mask.sum(dim=1)
    local_tokens, _ = permute(hidden_states, routing_map)
    local_scores = permuted_weights(routing_weights, selected_experts, num_experts)
    expected_tokens = sort_chunks_by_idxs(
        local_tokens.index_select(0, selection), num_tokens_per_expert.ravel(), [0, 2, 1, 3]
    )
    expected_scores = sort_chunks_by_idxs(
        local_scores.index_select(0, selection), num_tokens_per_expert.ravel(), [0, 2, 1, 3]
    )

    permuted_tokens, cumsum, ctx = alltoall_module.alltoall_pre_dispatch(
        hidden_states=hidden_states,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        num_experts=num_experts,
        ep_group=FakeGroup(),
    )

    torch.testing.assert_close(permuted_tokens, expected_tokens)
    torch.testing.assert_close(ctx.expert_scores, expected_scores)
    torch.testing.assert_close(cumsum, torch.tensor([2, 5], dtype=cumsum.dtype))

    ctx.expert_scores.sum().backward()
    expected_grad = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    torch.testing.assert_close(routing_weights.grad, expected_grad)
