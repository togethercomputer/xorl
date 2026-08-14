"""Two-rank CPU/Gloo regression for the physical-PP RL objective path."""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import xorl.server.runner.model_runner as model_runner_module
import xorl.trainers.training_utils as training_utils_module
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


_HIDDEN_SIZE = 8
_VOCAB_SIZE = 16
_SEQ_LEN = 3


class _FirstStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=_HIDDEN_SIZE, vocab_size=_VOCAB_SIZE)
        self.embed = nn.Embedding(_VOCAB_SIZE, _HIDDEN_SIZE)
        self.proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(self.embed(input_ids)))


class _TerminalStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=_HIDDEN_SIZE, vocab_size=_VOCAB_SIZE)
        self.proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)
        # The stage emits terminal hidden states. The stable objective
        # dispatcher owns this head and applies the requested RL program.
        self.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden_states))


def _micro_batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "target_tokens": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "logprobs": torch.tensor([[-0.25, -0.5, -0.75]], dtype=torch.float32),
            "advantages": torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32),
        },
        {
            "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
            "labels": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "target_tokens": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "logprobs": torch.tensor([[-0.4, -0.6, -0.8]], dtype=torch.float32),
            "advantages": torch.tensor([[1.25, 0.75, 0.25]], dtype=torch.float32),
        },
    ]


def _build_runner(rank: int) -> ModelRunner:
    torch.manual_seed(1234 + rank)
    model_part = (_FirstStage() if rank == 0 else _TerminalStage()).to(torch.bfloat16)
    runner = object.__new__(ModelRunner)
    runner.rank = rank
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = True
    runner.pp_enabled = True
    runner.pp_num_stages = 2
    runner.has_first_stage = rank == 0
    runner.has_last_stage = rank == 1
    runner.train_config = {"pipeline_parallel_schedule": "1F1B"}
    runner.model = model_part
    runner.model_parts = [model_part]
    runner.pp_stages = [SimpleNamespace(stage_index=rank)]
    runner._pp_schedule_cache = {}
    return runner


def _run_physical_pp_rl_worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        init_parallel_state(
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            tp_size=1,
            ep_size=1,
            pp_size=2,
            ringattn_size=1,
            ulysses_size=1,
            dp_mode="none",
            device_type="cpu",
            cp_fsdp_mode="none",
        )
        ps = get_parallel_state()
        # The host may expose GPUs, but this regression deliberately exercises
        # the Gloo/CPU pipeline transport.
        model_runner_module.get_device_type = lambda: "cpu"
        training_utils_module.get_device_type = lambda: "cpu"
        runner = _build_runner(rank)
        micro_batches = _micro_batches()

        raw_loss, records = runner._forward_backward_pp(
            micro_batches,
            global_valid_tokens=torch.tensor(6),
            loss_fn="importance_sampling",
            loss_fn_params={"compute_kl_stats": True},
            model_id="policy-a",
        )

        parameters = tuple(runner.model_parts[0].parameters())
        local_summary = {
            "rank": rank,
            "raw_loss": raw_loss,
            "record_ids": [record["microbatch_id"] for record in records],
            "record_losses": [float(record["loss"].item()) for record in records],
            "logprob_shapes": [tuple(record["per_token_outputs"]["logprobs"].shape) for record in records],
            "all_grads": all(
                parameter.grad is not None and parameter.grad.isfinite().all() for parameter in parameters
            ),
            "any_nonzero_grad": any(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)) for parameter in parameters
            ),
            "head_grad": rank == 0
            or (
                runner.model_parts[0].lm_head.weight.grad is not None
                and bool(torch.count_nonzero(runner.model_parts[0].lm_head.weight.grad))
            ),
            "dispatcher_active": runner._make_pp_train_loss_fn().active,
        }
        gathered = [None, None]
        dist.all_gather_object(gathered, local_summary, group=ps.pp_group)

        assert all(summary["record_ids"] == [0, 1] for summary in gathered)
        assert all(summary["logprob_shapes"] == [(1, _SEQ_LEN), (1, _SEQ_LEN)] for summary in gathered)
        assert all(summary["all_grads"] and summary["any_nonzero_grad"] for summary in gathered)
        assert gathered[1]["head_grad"]
        assert all(not summary["dispatcher_active"] for summary in gathered)
        assert gathered[0]["record_losses"] == gathered[1]["record_losses"]
        assert gathered[0]["raw_loss"] == pytest.approx(gathered[1]["raw_loss"])
        # Positive advantages make this importance-sampling objective signed
        # negative. The PP SUM must preserve it on the nonterminal rank too.
        assert gathered[0]["raw_loss"] < 0.0
        assert gathered[0]["raw_loss"] == pytest.approx(sum(gathered[0]["record_losses"]))
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_physical_pp_runs_rl_backward_and_broadcasts_records(unused_tcp_port):
    mp.start_processes(
        _run_physical_pp_rl_worker,
        args=(unused_tcp_port,),
        nprocs=2,
        start_method="spawn",
    )
