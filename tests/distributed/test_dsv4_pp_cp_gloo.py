"""CPU/Gloo regression for exact DSV4 context parallelism across a real PP wire."""

from __future__ import annotations

import copy
import os
import types
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import xorl.trainers.training_utils as training_utils_module
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.pipeline_parallel import _pp_forward, _recursive_prune, build_pipeline_schedule, build_pp_stage
from xorl.models.transformers.deepseek_v4 import DeepseekV4Config
from xorl.models.transformers.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from xorl.trainers.training_utils import _set_pp_batch_metadata, align_dsv4_pp_storage_rows


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]

_STORAGE_ROWS = 8
_LIVE_ROWS = 6
_GLOBAL_STORAGE_ROWS = 16
_HIDDEN_SIZE = 32
_N_MICROBATCHES = 2


class _ScaleLayer(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.scale = nn.Parameter(torch.tensor(1.0 + 0.125 * layer_id))

    def forward(self, hidden, **_kwargs):
        return hidden * self.scale


class _MeanHyperConnection:
    @staticmethod
    def block_expand(hidden):
        return hidden.unsqueeze(2).expand(-1, -1, 2, -1).clone()

    @staticmethod
    def block_head(hidden, *_args):
        return hidden.mean(dim=2)


def _config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=4,
        max_position_embeddings=256,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        hc_mult=2,
        compress_ratios=[0, 0],
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        _moe_implementation="eager",
    )


def _model() -> DeepseekV4ForCausalLM:
    config = _config()
    model = DeepseekV4ForCausalLM(config, moe_implementation="eager")
    # Select the production row-layout boundary dynamically, while replacing
    # CUDA-only decoder arithmetic with a differentiable CPU stand-in. The
    # physical PP/CP metadata, compaction, P2P, repack, and VJP remain real.
    config._dsv4_flash_exact_mode = True
    model.model.layers = nn.ModuleList([_ScaleLayer(0), _ScaleLayer(1)])
    model.model.hc_util = _MeanHyperConnection()
    model.model.norm = nn.Identity()
    return model.to(torch.bfloat16).train()


def _micro_batches(cp_rank: int) -> list[dict]:
    storage_indices = torch.tensor([0, 2, 3, 5, 6, 7])
    logical_start = cp_rank * _LIVE_ROWS
    logical = torch.full((1, _STORAGE_ROWS), -1, dtype=torch.int64)
    logical[:, storage_indices] = torch.arange(logical_start, logical_start + _LIVE_ROWS)
    request_ids = torch.full_like(logical, -1)
    request_ids[:, storage_indices] = 0
    request_positions = torch.zeros_like(logical)
    request_positions[:, storage_indices] = torch.arange(logical_start, logical_start + _LIVE_ROWS)
    live_mask = logical >= 0

    batches = []
    for microbatch_id in range(_N_MICROBATCHES):
        input_ids = torch.zeros((1, _STORAGE_ROWS), dtype=torch.int64)
        live_ids = torch.arange(logical_start, logical_start + _LIVE_ROWS) + 1 + 17 * microbatch_id
        input_ids[:, storage_indices] = live_ids.remainder(63).add(1)
        batches.append(
            {
                "input_ids": input_ids,
                "position_ids": torch.arange(_GLOBAL_STORAGE_ROWS).view(1, -1),
                "cu_seq_lens_q": torch.tensor([0, _GLOBAL_STORAGE_ROWS], dtype=torch.int32),
                "cu_seq_lens_k": torch.tensor([0, _GLOBAL_STORAGE_ROWS], dtype=torch.int32),
                "max_length_q": _GLOBAL_STORAGE_ROWS,
                "max_length_k": _GLOBAL_STORAGE_ROWS,
                "_cp_logical_row_indices": logical.clone(),
                "_cp_request_ids": request_ids.clone(),
                "_cp_request_positions": request_positions.clone(),
                "_cp_live_mask": live_mask.clone(),
                "_r3_sample_lengths": [2 * _LIVE_ROWS],
                "num_samples": 1,
            }
        )
    return batches


def _row_kwargs(batch: dict) -> dict:
    return {
        key: batch[key]
        for key in (
            "_cp_logical_row_indices",
            "_cp_request_ids",
            "_cp_request_positions",
            "_cp_live_mask",
            "_r3_sample_lengths",
            "num_samples",
        )
    }


def _run_worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["XORL_DSV4_ROPE_MAX_SEQ_LEN"] = "256"
    dist.init_process_group("gloo", rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        init_parallel_state(
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            tp_size=1,
            ep_size=1,
            pp_size=2,
            ringattn_size=1,
            ulysses_size=2,
            dp_mode="none",
            device_type="cpu",
            cp_fsdp_mode="none",
        )
        ps = get_parallel_state()
        training_utils_module.get_device_type = lambda: "cpu"
        torch.manual_seed(101)
        template = _model()
        baseline = copy.deepcopy(template)
        batches = _micro_batches(ps.cp_rank)

        # The schedule-visible storage negotiation is a real WORLD collective.
        assert align_dsv4_pp_storage_rows(batches, cp_size=ps.cp_size) == _STORAGE_ROWS
        assert all(int(batch["_cp_live_mask"].sum()) == _LIVE_ROWS for batch in batches)

        baseline_outputs = []
        baseline_loss = torch.zeros((), dtype=torch.float32)
        for batch in batches:
            hidden = baseline(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                **_row_kwargs(batch),
            ).last_hidden_state
            baseline_outputs.append(hidden.detach())
            baseline_loss = baseline_loss + hidden.float().square().sum()
        baseline_loss.backward()

        stage_index = ps.pp_rank
        modules = (
            {"model.embed_tokens", "model.layers.0"}
            if stage_index == 0
            else {"model.layers.1", "model.norm", "lm_head"}
        )
        part = copy.deepcopy(template)
        _recursive_prune(part, "", modules)
        part._configure_pp_stage(stage_idx=stage_index, num_stages=2)
        part._pp_is_first = stage_index == 0
        part._pp_is_last = stage_index == 1
        part._pp_stage_idx = stage_index
        part._pp_lm_head_in_loss = True
        part._pp_exact_boundary_contract = True
        part._pp_pipeline_boundary_state = {
            "rank": 4,
            "dtype": torch.bfloat16,
            "shape_suffix": (2, _HIDDEN_SIZE),
            "state": "completed_hyperconnection_residual",
        }
        part._pp_original_forward = part.forward
        part.forward = types.MethodType(_pp_forward, part)

        if stage_index == 0:
            input_args = (torch.empty(1, _STORAGE_ROWS, dtype=torch.int64, device="meta"),)
        else:
            input_args = (torch.empty(1, _STORAGE_ROWS, 2, _HIDDEN_SIZE, dtype=torch.bfloat16, device="meta"),)
        output_shape = (1, _STORAGE_ROWS, 2, _HIDDEN_SIZE) if stage_index == 0 else (1, _STORAGE_ROWS, _HIDDEN_SIZE)
        stage = build_pp_stage(
            part,
            stage_index=stage_index,
            num_stages=2,
            device=torch.device("cpu"),
            pp_group=ps.pp_group,
            input_args=input_args,
            output_args=(torch.empty(*output_shape, dtype=torch.bfloat16, device="meta"),),
        )

        def loss_fn(output, target):
            return (output.float() - target.float()).square().sum()

        schedule = build_pipeline_schedule(
            stages=[stage],
            n_microbatches=_N_MICROBATCHES,
            loss_fn=loss_fn,
            schedule_name="1F1B",
        )
        _set_pp_batch_metadata([part], batches)
        dist.barrier()
        if stage_index == 0:
            schedule_output = schedule.step(
                torch.cat([batch["input_ids"] for batch in batches], dim=0),
                return_outputs=True,
            )
            losses = None
        else:
            losses = []
            schedule_output = schedule.step(
                target=torch.zeros(
                    _N_MICROBATCHES,
                    _STORAGE_ROWS,
                    _HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                ),
                losses=losses,
                return_outputs=True,
            )

        if stage_index == 0:
            gradient_names = ("model.embed_tokens.weight", "model.layers.0.scale")
        else:
            gradient_names = ("model.layers.1.scale",)
        part_params = dict(part.named_parameters())
        baseline_params = dict(baseline.named_parameters())
        gradients_match = True
        for name in gradient_names:
            gradients_match = gradients_match and part_params[name].grad is not None
            if part_params[name].grad is not None:
                gradients_match = gradients_match and torch.allclose(
                    part_params[name].grad.float(),
                    baseline_params[name].grad.float(),
                    rtol=1e-3,
                    atol=1e-3,
                )

        output_matches = True
        if stage_index == 1:
            expected = torch.cat(baseline_outputs, dim=0)
            output_matches = torch.equal(schedule_output, expected)
            output_matches = output_matches and len(losses) == _N_MICROBATCHES
        summary = {
            "rank": rank,
            "stage": stage_index,
            "cp_rank": ps.cp_rank,
            "output_matches": output_matches,
            "gradients_match": gradients_match,
        }
        summaries = [None] * 4
        dist.all_gather_object(summaries, summary)
        assert all(item["output_matches"] for item in summaries)
        assert all(item["gradients_match"] for item in summaries)
        assert {(item["stage"], item["cp_rank"]) for item in summaries} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
    finally:
        dist.destroy_process_group()


def test_real_pp2_cp2_storage_wire_forward_backward(unused_tcp_port):
    mp.start_processes(
        _run_worker,
        args=(unused_tcp_port,),
        nprocs=4,
        start_method="spawn",
    )
