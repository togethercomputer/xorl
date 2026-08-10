"""Actual collective-order checks for the GLM-5.2 exact LM-head helpers."""

import os

import pytest
import torch
import torch.distributed as dist

from xorl.models.transformers.glm5.exact_lm_head_qlora import (
    _all_reduce_sum_fp32,
    _Glm52ExactDistributedTP16LmHeadFunction,
    _rank_order_vocab_all_gather,
    _require_equal_nonzero_row_count,
)


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _run_collective_case() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        group = dist.group.WORLD
        local = torch.tensor(
            [
                [rank * 100.0 + 0.0, rank * 100.0 + 1.0, rank * 100.0 + 2.0],
                [rank * 100.0 + 10.0, rank * 100.0 + 11.0, rank * 100.0 + 12.0],
            ],
            dtype=torch.float32,
        )

        gathered = _rank_order_vocab_all_gather(
            local,
            group,
            expected_world_size=2,
            expected_local_vocab_size=3,
        )
        expected = torch.tensor(
            [[0.0, 1.0, 2.0, 100.0, 101.0, 102.0], [10.0, 11.0, 12.0, 110.0, 111.0, 112.0]],
            dtype=torch.float32,
        )
        assert torch.equal(gathered.view(torch.uint8), expected.view(torch.uint8))

        reduced = _all_reduce_sum_fp32(torch.tensor([rank + 0.25, rank + 1.25]), group)
        assert torch.equal(reduced, torch.tensor([1.5, 3.5]))

        class _FakeDistributedComponent:
            tp_group = group

            @staticmethod
            def _validate_tp_group():
                return group

            @staticmethod
            def _exact_forward_value(hidden, weight, effective_A, effective_B, token_ids):
                del weight, effective_A, effective_B
                return hidden.float().sum(dim=-1) + token_ids.float()

            @staticmethod
            def _surrogate_vjp(
                hidden,
                weight,
                effective_A,
                effective_B,
                token_ids,
                grad_logprob,
                *,
                needs_input_grad,
            ):
                del weight, token_ids, needs_input_grad
                grad_sum = grad_logprob.sum()
                return (
                    grad_logprob.unsqueeze(-1).expand_as(hidden).float(),
                    torch.ones_like(effective_A, dtype=torch.float32) * grad_sum,
                    torch.ones_like(effective_B, dtype=torch.float32) * grad_sum * (rank + 1),
                )

        local_hidden = torch.tensor([[rank * 10.0 + 1.0, rank * 10.0 + 2.0]], dtype=torch.bfloat16).requires_grad_(True)
        local_weight = torch.zeros((1, 2), dtype=torch.bfloat16)
        lora_A = torch.tensor([[0.25, -0.5]], dtype=torch.float32, requires_grad=True)
        local_lora_B = torch.tensor([[0.75]], dtype=torch.float32, requires_grad=True)
        local_token_ids = torch.tensor([rank + 10], dtype=torch.int64)
        local_logprob = _Glm52ExactDistributedTP16LmHeadFunction.apply(
            local_hidden,
            local_weight,
            lora_A,
            local_lora_B,
            local_token_ids,
            _FakeDistributedComponent(),
        )
        assert torch.equal(local_logprob, torch.tensor([13.0 if rank == 0 else 34.0]))

        (local_logprob * (rank + 1)).sum().backward()
        assert torch.equal(local_hidden.grad, torch.full_like(local_hidden, rank + 1))
        assert torch.equal(lora_A.grad, torch.full_like(lora_A, 3.0))
        assert torch.equal(local_lora_B.grad, torch.full_like(local_lora_B, 3.0 * (rank + 1)))

        with pytest.raises(ValueError, match="equal source-row counts"):
            _require_equal_nonzero_row_count(torch.zeros((rank + 1, 2)), group)
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":
    from tests.distributed.distributed_utils import run_distributed_script

    def test_rank_order_gather_and_fp32_surrogate_sum_cpu() -> None:
        result = run_distributed_script(os.path.abspath(__file__), num_gpus=2, timeout=120)
        result.assert_success()


if __name__ == "__main__":
    _run_collective_case()
