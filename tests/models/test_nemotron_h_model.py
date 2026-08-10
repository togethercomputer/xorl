import pytest
import torch

from xorl.models.module_utils import compute_loss
from xorl.models.transformers.nemotron_h.configuration_nemotron_h import NemotronHConfig
from xorl.models.transformers.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM


pytestmark = [pytest.mark.cpu]

SEQ_LEN = 16


def _tiny_config(**overrides) -> NemotronHConfig:
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        layers_block_type=overrides.pop("layers_block_type", ["mamba", "attention", "moe", "mamba", "moe"]),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=overrides.pop("chunk_size", 32),  # default >= seq len: single-chunk SSD
        intermediate_size=48,
        moe_intermediate_size=24,
        moe_shared_expert_intermediate_size=40,
        moe_latent_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        time_step_limit=(0.001, float("inf")),
        use_cache=False,
        **overrides,
    )
    config._attn_implementation = "eager"
    config._moe_implementation = "eager"
    return config


def _build_model() -> NemotronHForCausalLM:
    torch.manual_seed(0)
    return NemotronHForCausalLM(_tiny_config())


def test_nemotron_h_forward_shape_and_router_logits():
    model = _build_model()
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))

    outputs = model(input_ids=input_ids)
    assert outputs.last_hidden_state.shape == (2, SEQ_LEN, model.config.hidden_size)
    assert outputs.router_logits is None  # output_router_logits defaults to False

    outputs = model(input_ids=input_ids, output_router_logits=True)
    num_moe_layers = model.config.layers_block_type.count("moe")
    assert len(outputs.router_logits) == num_moe_layers
    assert outputs.router_logits[0].shape == (2 * SEQ_LEN, model.config.n_routed_experts)


def test_nemotron_h_backward_reaches_all_mixer_types():
    model = _build_model()
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))

    outputs = model(input_ids=input_ids)
    outputs.last_hidden_state.float().pow(2).mean().backward()

    layers = model.model.layers
    grads = {
        "mamba in_proj": layers[0].mixer.in_proj.weight.grad,
        "mamba A_log": layers[0].mixer.A_log.grad,
        "attention q_proj": layers[1].mixer.q_proj.weight.grad,
        "attention o_proj": layers[1].mixer.o_proj.weight.grad,
        "expert gate_up_proj": layers[2].mixer.experts.gate_up_proj.grad,
        "expert down_proj": layers[2].mixer.experts.down_proj.grad,
        "fc1_latent_proj": layers[2].mixer.fc1_latent_proj.weight.grad,
        "fc2_latent_proj": layers[2].mixer.fc2_latent_proj.weight.grad,
        "shared up_proj": layers[2].mixer.shared_experts.up_proj.weight.grad,
        "router gate": layers[2].mixer.gate.weight.grad,
        "embeddings": model.model.embeddings.weight.grad,
    }
    for name, grad in grads.items():
        if name == "router gate":
            # train_router defaults to False: routing weights are detached and there
            # is no aux loss, so the router gets no gradient.
            assert grad is None, name
            continue
        assert grad is not None, f"missing grad for {name}"
        assert torch.isfinite(grad).all(), f"non-finite grad for {name}"
        assert grad.abs().sum() > 0, f"zero grad for {name}"


def test_nemotron_h_loss_with_labels():
    model = _build_model()
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))
    labels = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))
    labels[:, :3] = -100

    outputs = model(input_ids=input_ids)
    result = compute_loss(
        model.lm_head,
        outputs.last_hidden_state,
        loss_fn_name=None,
        loss_fn_inputs={"labels": labels},
        loss_fn_params={"ce_mode": "eager"},
        logits_to_keep=0,
    )
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.model.layers[0].mixer.in_proj.weight.grad is not None


def test_nemotron_h_gradient_checkpointing_full_layer():
    model = _build_model()
    model.train()
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(0, model.config.vocab_size, (2, SEQ_LEN))

    outputs = model(input_ids=input_ids)
    outputs.last_hidden_state.mean().backward()
    assert model.model.layers[2].mixer.experts.gate_up_proj.grad is not None
    assert model.model.layers[0].mixer.in_proj.weight.grad is not None


def test_nemotron_h_packed_varlen_matches_per_sequence():
    """Packed forward (cu_seq_lens_q) vs running each document separately.

    Attention layers are excluded: the eager backend ignores cu_seq_lens (varlen masking
    is handled by the flash/native backends, GPU-only), so exact CPU parity is only
    defined for mamba + moe blocks. chunk_size=4 puts the first boundary (7) inside a
    chunk and the second (12) on a chunk edge.
    """
    torch.manual_seed(0)
    model = NemotronHForCausalLM(_tiny_config(layers_block_type=["mamba", "moe", "mamba"], chunk_size=4))
    model.eval()
    lengths = [7, 5, 4]
    input_ids = torch.randint(0, model.config.vocab_size, (1, sum(lengths)))
    cu_seqlens = torch.tensor([0, 7, 12, 16], dtype=torch.int32)

    packed = model(input_ids=input_ids, cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens)
    start = 0
    pieces = []
    for length in lengths:
        pieces.append(model(input_ids=input_ids[:, start : start + length]).last_hidden_state)
        start += length
    separate = torch.cat(pieces, dim=1)
    torch.testing.assert_close(packed.last_hidden_state, separate, atol=1e-5, rtol=1e-4)


def test_nemotron_h_packed_varlen_smoke_all_block_types():
    """Packed kwargs flow through mamba + attention + moe blocks (forward + backward)."""
    model = _build_model()
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN))
    cu_seqlens = torch.tensor([0, 7, SEQ_LEN], dtype=torch.int32)

    outputs = model(input_ids=input_ids, cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens)
    assert outputs.last_hidden_state.shape == (1, SEQ_LEN, model.config.hidden_size)
    assert torch.isfinite(outputs.last_hidden_state).all()
    outputs.last_hidden_state.pow(2).mean().backward()
    assert model.model.layers[0].mixer.in_proj.weight.grad is not None
    assert torch.isfinite(model.model.layers[0].mixer.in_proj.weight.grad).all()
