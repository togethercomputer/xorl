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
    micro_batches = []
    next_token = 1
    for microbatch_id, physical_batch_size in enumerate((1, 2, 3)):
        input_ids = torch.arange(
            next_token,
            next_token + physical_batch_size * _SEQ_LEN,
            dtype=torch.long,
        ).reshape(physical_batch_size, _SEQ_LEN)
        input_ids.remainder_(_VOCAB_SIZE - 1).add_(1)
        labels = input_ids.remainder(_VOCAB_SIZE - 1).add(1)
        row_marker = torch.arange(physical_batch_size, dtype=torch.float32).add_(10 * microbatch_id + 1)
        position_ids = row_marker.to(torch.long)[:, None].expand(-1, _SEQ_LEN).contiguous()
        micro_batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "target_tokens": labels.clone(),
                "logprobs": -0.05 * input_ids.float(),
                "advantages": 0.25 + 0.05 * input_ids.float(),
                # Distinctive real PP row metadata for boundary/order checks.
                "position_ids": position_ids,
            }
        )
        next_token += physical_batch_size * _SEQ_LEN
    return micro_batches


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
        model_runner_module.synchronize = lambda: None
        training_utils_module.get_device_type = lambda: "cpu"
        runner = _build_runner(rank)
        micro_batches = _micro_batches()

        terminal_calls = []
        original_compute = runner._compute_pp_terminal_objective

        def observed_compute(terminal_hidden, objective):
            terminal_calls.append(
                {
                    "microbatch_id": objective.microbatch_id,
                    "loss_fn": objective.loss_fn,
                    "input_shape": tuple(objective.micro_batch["input_ids"].shape),
                    "position_ids": objective.micro_batch["position_ids"].tolist(),
                }
            )
            return original_compute(terminal_hidden, objective)

        runner._compute_pp_terminal_objective = observed_compute

        raw_loss, records = runner._forward_backward_pp(
            micro_batches,
            global_valid_tokens=torch.tensor(18),
            loss_fn="importance_sampling",
            loss_fn_params={"compute_kl_stats": True},
            model_id="policy-a",
        )
        backward_terminal_calls = tuple(terminal_calls)
        terminal_calls.clear()

        forward_only_result = runner._pp_forward_only_loop(
            micro_batches,
            loss_fn="causallm_loss",
            loss_fn_params={"return_per_token": True},
        )
        forward_only_terminal_calls = tuple(terminal_calls)

        parameters = tuple(runner.model_parts[0].parameters())
        local_summary = {
            "rank": rank,
            "raw_loss": raw_loss,
            "record_ids": [record["microbatch_id"] for record in records],
            "record_losses": [float(record["loss"].item()) for record in records],
            "logprob_shapes": [tuple(record["per_token_outputs"]["logprobs"].shape) for record in records],
            "backward_terminal_calls": backward_terminal_calls,
            "forward_only_terminal_calls": forward_only_terminal_calls,
            "forward_only_shapes": [
                tuple(tensor.shape) for tensor in forward_only_result["_pp_raw_per_token_logprobs"]
            ],
            "forward_only_position_ids": forward_only_result["packed_position_ids"],
            "schedule_keys": sorted(runner._pp_schedule_cache),
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

        expected_shapes = [(1, _SEQ_LEN), (2, _SEQ_LEN), (3, _SEQ_LEN)]
        assert all(summary["record_ids"] == [0, 1, 2] for summary in gathered)
        assert all(summary["logprob_shapes"] == expected_shapes for summary in gathered)
        assert all(summary["forward_only_shapes"] == expected_shapes for summary in gathered)
        assert all(summary["all_grads"] and summary["any_nonzero_grad"] for summary in gathered)
        assert gathered[1]["head_grad"]
        assert all(not summary["dispatcher_active"] for summary in gathered)
        assert gathered[0]["record_losses"] == gathered[1]["record_losses"]
        assert gathered[0]["raw_loss"] == pytest.approx(gathered[1]["raw_loss"])
        assert gathered[0]["backward_terminal_calls"] == ()
        assert gathered[0]["forward_only_terminal_calls"] == ()
        assert [call["microbatch_id"] for call in gathered[1]["backward_terminal_calls"]] == [0, 1, 2]
        assert [call["loss_fn"] for call in gathered[1]["backward_terminal_calls"]] == ["importance_sampling"] * 3
        assert [call["microbatch_id"] for call in gathered[1]["forward_only_terminal_calls"]] == [0, 1, 2]
        assert [call["loss_fn"] for call in gathered[1]["forward_only_terminal_calls"]] == ["causallm_loss"] * 3
        for call_group in (
            gathered[1]["backward_terminal_calls"],
            gathered[1]["forward_only_terminal_calls"],
        ):
            assert [call["input_shape"] for call in call_group] == expected_shapes
            assert [call["position_ids"] for call in call_group] == [
                micro_batch["position_ids"].tolist() for micro_batch in micro_batches
            ]
        assert all(
            summary["forward_only_position_ids"]
            == [micro_batch["position_ids"].tolist() for micro_batch in micro_batches]
            for summary in gathered
        )
        expected_schedule_keys = sorted(
            (1, physical_batch_size, _SEQ_LEN, has_loss, "interleaved1f1b")
            for physical_batch_size in (1, 2, 3)
            for has_loss in (False, True)
        )
        assert all(summary["schedule_keys"] == expected_schedule_keys for summary in gathered)
        # Positive advantages make this importance-sampling objective signed
        # negative. The PP SUM must preserve it on the nonterminal rank too.
        assert gathered[0]["raw_loss"] < 0.0
        assert gathered[0]["raw_loss"] == pytest.approx(sum(gathered[0]["record_losses"]))
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_physical_pp_preserves_ragged_boundaries_in_backward_and_forward_only(unused_tcp_port):
    mp.start_processes(
        _run_physical_pp_rl_worker,
        args=(unused_tcp_port,),
        nprocs=2,
        start_method="spawn",
    )
