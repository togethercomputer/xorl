"""Two-rank FSDP2 replay for the Qwen3.5/3.6 LoRA projection topology."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.lora.modules.base import LoraModule
from xorl.lora.utils import freeze_base_parameters, inject_lora_into_model
from xorl.models.transformers.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from xorl.models.transformers.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from xorl.server.weight_sync.handler import WeightSyncHandler
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]
_RANK = 16
_TARGETS = ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _target_manifest() -> dict:
    expected_modules = []
    for projection in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"):
        expected_modules.append({"pattern": f"model.layers.*.linear_attn.{projection}", "count": 1, "rank": _RANK})
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        expected_modules.append({"pattern": f"model.layers.*.self_attn.{projection}", "count": 1, "rank": _RANK})
    for projection in ("gate_proj", "up_proj", "down_proj"):
        expected_modules.append(
            {"pattern": f"model.layers.*.mlp.shared_expert.{projection}", "count": 2, "rank": _RANK}
        )
    return {
        "schema_version": 1,
        "target_modules": _TARGETS,
        "expected_modules": expected_modules,
        "allow_unlisted": False,
    }


def _config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        decoder_sparse_step=1,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        _attn_implementation="eager",
        _moe_implementation="eager",
        pad_token_id=0,
    )


def _build(device: torch.device) -> Qwen3_5MoeForCausalLM:
    torch.manual_seed(1234)
    model = Qwen3_5MoeForCausalLM(_config())
    inject_lora_into_model(model, r=_RANK, lora_alpha=_RANK, target_manifest=_target_manifest())
    freeze_base_parameters(model)
    for module in model.modules():
        if isinstance(module, LoraModule):
            module.exact_merged_forward = True
            with torch.no_grad():
                values = torch.linspace(-0.01, 0.01, module.lora_B.numel(), dtype=torch.float32)
                module.lora_B.copy_(values.reshape_as(module.lora_B))
    return model.to(device=device, dtype=torch.bfloat16).train()


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _run_replay() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    try:
        world_size = dist.get_world_size()
        assert world_size == 2
        device = torch.device("cuda", local_rank)
        init_parallel_state(dp_size=world_size, dp_shard_size=world_size, device_type="cuda")
        os.environ["XORL_GDN_BACKEND"] = "fla"
        os.environ["XORL_MOE_SGLANG_FUSED_EXPERTS"] = "0"

        reference = _build(device)
        sharded = _build(device)
        mesh = get_parallel_state().dp_shard_mesh
        fully_shard(sharded, mesh=mesh, reshard_after_forward=False)

        reference_params = dict(reference.named_parameters())
        sharded_params = dict(sharded.named_parameters())
        trainable_names = [name for name, parameter in reference_params.items() if parameter.requires_grad]
        assert trainable_names
        assert all(isinstance(sharded_params[name], DTensor) for name in trainable_names)

        reference_optim = torch.optim.AdamW((reference_params[name] for name in trainable_names), lr=1e-3)
        sharded_optim = torch.optim.AdamW((sharded_params[name] for name in trainable_names), lr=1e-3)
        decisions = [torch.tensor([[3, 5, 7, 11, 13, 17 + index]], device=device) for index in range(4)]
        decision_hashes = []

        for step in range(2):
            reference_optim.zero_grad(set_to_none=True)
            sharded_optim.zero_grad(set_to_none=True)
            reference_loss = torch.zeros((), device=device)
            sharded_loss = torch.zeros((), device=device)
            for input_ids in decisions:
                reference_output = reference(input_ids=input_ids, use_cache=False).last_hidden_state
                sharded_output = sharded(input_ids=input_ids, use_cache=False).last_hidden_state
                assert torch.equal(sharded_output.view(torch.uint8), reference_output.view(torch.uint8))
                if step == 1:
                    decision_hashes.append(_sha256(sharded_output))
                reference_loss = reference_loss + reference_output.float().square().mean()
                sharded_loss = sharded_loss + sharded_output.float().square().mean()
            reference_loss.backward()
            sharded_loss.backward()

            for name in trainable_names:
                sharded_grad = sharded_params[name].grad
                reference_grad = reference_params[name].grad
                assert isinstance(sharded_grad, DTensor), name
                assert reference_grad is not None, name
                full_grad = sharded_grad.full_tensor()
                assert torch.equal(full_grad.view(torch.uint8), reference_grad.view(torch.uint8)), name
                assert torch.isfinite(full_grad).all(), name
                assert torch.count_nonzero(full_grad), name

            reference_optim.step()
            sharded_optim.step()
            for name in trainable_names:
                full_parameter = sharded_params[name].full_tensor()
                assert torch.equal(full_parameter.view(torch.uint8), reference_params[name].view(torch.uint8)), name

        sharded.unshard()
        layer = sharded.model.layers[0]

        class _FakeDTensor:
            pass

        synchronized = dict(
            WeightSyncHandler._extract_params_for_sync(
                layer,
                "model.layers.0",
                _FakeDTensor,
                skip_moe_prefixes={"mlp.experts"},
            )
        )
        gdn = layer.linear_attn
        shared = layer.mlp.shared_expert
        assert torch.equal(synchronized["model.layers.0.linear_attn.q_proj.weight"], gdn.q_proj._merged_weight())
        gate, up = shared._gate_up_weights_for_forward()
        assert torch.equal(
            synchronized["model.layers.0.mlp.shared_expert.gate_up_proj.weight"],
            torch.cat((gate, up), dim=0),
        )
        assert torch.equal(
            synchronized["model.layers.0.mlp.shared_expert.down_proj.weight"],
            shared.down_proj._merged_weight(),
        )

        report = {
            "schema_version": 1,
            "event": "qwen35_lora_projection_fsdp2_four_decision_replay",
            "world_size": world_size,
            "decisions": len(decisions),
            "optimizer_steps": 2,
            "lora_rank": _RANK,
            "trainable_factor_count": len(trainable_names),
            "decision_output_sha256": decision_hashes,
            "raw_output_bytes_equal": True,
            "gradient_bytes_equal": True,
            "post_update_factor_bytes_equal": True,
            "folded_sync_bytes_equal": True,
            "passed": True,
        }
        artifact = os.environ.get("QWEN35_LORA_PROJECTION_REPLAY_ARTIFACT")
        if dist.get_rank() == 0:
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            if artifact:
                output = Path(artifact)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_qwen35_lora_projection_two_rank_fsdp2_four_decision_replay() -> None:
        result = run_distributed_script(__file__, num_gpus=2, timeout=300)
        result.assert_success("Qwen3.5 LoRA projection topology must preserve raw bytes and gradients through FSDP2")


if __name__ == "__main__":
    _run_replay()
