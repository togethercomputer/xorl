"""HF-parity and round-trip tests for the NemotronH checkpoint handler.

The transformers 5.5.3 ``NemotronHForCausalLM`` is the oracle. Its torch-only
SSD fallback mis-propagates state across chunks (see
tests/ops/test_ssm_mamba2.py), so sequence lengths here stay <= chunk_size for
strict comparison. ``time_step_limit`` is set to ``(time_step_min, inf)`` so
the dt clamp matches the HF torch path (which only floors at time_step_min).
"""

import re

import pytest
import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig as HFNemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM as HFNemotronHForCausalLM

from xorl.distributed.parallel_plan import ParallelPlan
from xorl.models.transformers.nemotron_h.checkpoint_handler import (
    NemotronHCheckpointHandler,
    checkpoint_has_per_expert_nemotron_weights,
)
from xorl.models.transformers.nemotron_h.configuration_nemotron_h import NemotronHConfig
from xorl.models.transformers.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from xorl.models.transformers.nemotron_h.parallelize import get_ep_plan


pytestmark = [pytest.mark.cpu]

SEQ_LEN = 16
NUM_EXPERTS = 8

_STACKED_EXPERT_PATTERN = re.compile(r"^(backbone\.layers\.\d+\.mixer\.experts)\.(up|down)_proj$")


def _tiny_hf_config() -> HFNemotronHConfig:
    return HFNemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        layers_block_type=["mamba", "attention", "moe", "mamba", "moe"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=32,  # >= seq len: avoids the HF torch-fallback inter-chunk bug
        intermediate_size=48,
        moe_intermediate_size=24,
        moe_shared_expert_intermediate_size=40,
        moe_latent_size=16,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        time_step_limit=(0.001, float("inf")),
        use_cache=False,
    )


def _build_hf_model() -> HFNemotronHForCausalLM:
    torch.manual_seed(0)
    hf_config = _tiny_hf_config()
    hf_config._attn_implementation = "eager"
    hf_model = HFNemotronHForCausalLM(hf_config).float().eval()
    with torch.no_grad():
        # Randomize the correction bias so the choice-vs-weight score split is exercised.
        for block in hf_model.model.layers:
            if block.block_type == "moe":
                block.mixer.gate.e_score_correction_bias.normal_(std=0.5)
    return hf_model


def _published_checkpoint_layout(hf_model: HFNemotronHForCausalLM) -> dict[str, torch.Tensor]:
    """Re-serialize the HF state dict into the published checkpoint layout:
    ``backbone.*`` prefix and per-expert ``experts.{e}.{up,down}_proj.weight`` keys."""
    checkpoint = {}
    for key, tensor in hf_model.state_dict().items():
        if key.startswith("model."):
            key = "backbone." + key.removeprefix("model.")
        stacked_match = _STACKED_EXPERT_PATTERN.match(key)
        if stacked_match is not None:
            prefix, proj = stacked_match.groups()
            for expert_idx in range(tensor.shape[0]):
                checkpoint[f"{prefix}.{expert_idx}.{proj}_proj.weight"] = tensor[expert_idx].contiguous()
        else:
            checkpoint[key] = tensor
    return checkpoint


def _build_xorl_model(hf_config: HFNemotronHConfig) -> NemotronHForCausalLM:
    config = NemotronHConfig.from_hf_config(hf_config)
    config._attn_implementation = "eager"
    config._moe_implementation = "eager"
    return NemotronHForCausalLM(config).float().eval()


def _run_handler(handler: NemotronHCheckpointHandler, checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    loaded = {}
    for key, tensor in checkpoint.items():
        for name, out_tensor in handler.on_load_weight(key, tensor):
            loaded[name] = out_tensor
    for name, out_tensor in handler.on_load_complete():
        loaded[name] = out_tensor
    return loaded


def test_nemotron_h_hf_parity_and_strict_load_accounting():
    hf_model = _build_hf_model()
    checkpoint = _published_checkpoint_layout(hf_model)
    checkpoint["mtp.layers.0.mixer.q_proj.weight"] = torch.zeros(2, 2)  # must be ignored

    model = _build_xorl_model(hf_model.config)
    handler = model.get_checkpoint_handler(checkpoint_keys=set(checkpoint))
    assert checkpoint_has_per_expert_nemotron_weights(set(checkpoint))

    loaded = _run_handler(handler, checkpoint)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    assert missing == []
    assert unexpected == []

    expert_param = model.model.layers[2].mixer.experts
    latent, intermediate = model.config.moe_latent_size, model.config.moe_intermediate_size
    assert expert_param.gate_up_proj.shape == (NUM_EXPERTS, latent, intermediate)
    assert expert_param.down_proj.shape == (NUM_EXPERTS, intermediate, latent)

    torch.manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))
    with torch.no_grad():
        hf_hidden = hf_model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        outputs = model(input_ids=input_ids)
        logits = model.lm_head(outputs.last_hidden_state)

    torch.testing.assert_close(outputs.last_hidden_state.float(), hf_hidden.float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits.float(), hf_logits.float(), atol=1e-4, rtol=1e-4)


def test_nemotron_h_save_round_trips_to_published_layout():
    hf_model = _build_hf_model()
    checkpoint = _published_checkpoint_layout(hf_model)

    model = _build_xorl_model(hf_model.config)
    handler = model.get_checkpoint_handler(checkpoint_keys=set(checkpoint))
    model.load_state_dict(_run_handler(handler, checkpoint), strict=True)

    saved = {}
    for name, tensor in model.state_dict().items():
        for key, out_tensor in handler.on_save_weight(name, tensor):
            saved[key] = out_tensor

    assert set(saved) == set(checkpoint)
    for key in checkpoint:
        torch.testing.assert_close(saved[key], checkpoint[key], atol=0.0, rtol=0.0)


def test_nemotron_h_handler_accepts_hf_stacked_expert_layout():
    """The transformers 5.x in-memory format stores experts as stacked 3D [E, out, in]."""
    hf_model = _build_hf_model()

    model = _build_xorl_model(hf_model.config)
    handler = model.get_checkpoint_handler(checkpoint_keys=set(hf_model.state_dict()))
    loaded = _run_handler(handler, dict(hf_model.state_dict()))
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    assert missing == []
    assert unexpected == []

    stacked_up = hf_model.state_dict()["model.layers.2.mixer.experts.up_proj"]
    assert torch.equal(
        model.model.layers[2].mixer.experts.gate_up_proj.data,
        stacked_up.transpose(1, 2).contiguous(),
    )


def test_nemotron_h_ep_skip_key_fn():
    handler = NemotronHCheckpointHandler(num_experts=NUM_EXPERTS, ep_rank=0, ep_size=2)
    skip_fn = handler.get_skip_key_fn()
    assert skip_fn is not None

    assert not skip_fn("backbone.layers.2.mixer.experts.3.up_proj.weight")
    assert skip_fn("backbone.layers.2.mixer.experts.4.up_proj.weight")
    assert skip_fn("backbone.layers.4.mixer.experts.7.down_proj.weight")
    assert skip_fn("mtp.layers.0.mixer.q_proj.weight")
    assert not skip_fn("backbone.layers.0.mixer.in_proj.weight")
    assert not skip_fn("lm_head.weight")

    # Single-EP handlers do no filtered loading.
    assert NemotronHCheckpointHandler(num_experts=NUM_EXPERTS).get_skip_key_fn() is None


def test_nemotron_h_ep_aware_loading_slices_and_counts_skips():
    hf_model = _build_hf_model()
    checkpoint = _published_checkpoint_layout(hf_model)

    handler = NemotronHCheckpointHandler(num_experts=NUM_EXPERTS, ep_rank=1, ep_size=2)
    skip_fn = handler.get_skip_key_fn()

    loaded = {}
    for key, tensor in checkpoint.items():
        if skip_fn(key):
            results = handler.on_skip_weight(key)
        else:
            results = handler.on_load_weight(key, tensor)
        for name, out_tensor in results:
            loaded[name] = out_tensor
    handler.on_load_complete()

    local = NUM_EXPERTS // 2
    gate_up = loaded["model.layers.2.mixer.experts.gate_up_proj"]
    assert gate_up.shape[0] == local
    stacked_up = hf_model.state_dict()["model.layers.2.mixer.experts.up_proj"]
    assert torch.equal(gate_up, stacked_up[local:].transpose(1, 2).contiguous())


def test_nemotron_h_ep_plan_targets_expert_params():
    plan = get_ep_plan()
    assert isinstance(plan, ParallelPlan)
    assert plan._is_expert_parameter("model.layers.2.mixer.experts.gate_up_proj")
    assert plan._is_expert_parameter("model.layers.104.mixer.experts.down_proj")
    assert not plan._is_expert_parameter("model.layers.2.mixer.fc1_latent_proj.weight")
    assert not plan._is_expert_parameter("model.layers.0.mixer.in_proj.weight")
    assert plan.fsdp_no_shard_module == {"model.layers.*.mixer.experts"}
