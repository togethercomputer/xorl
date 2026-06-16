"""End-to-end OLMo-2 TP=2 forward + backward on a 2-rank gloo+CPU mesh.

Builds a tiny OLMo-2 model, applies the TP plan from
``xorl.models.transformers.olmo2.parallelize``, runs a full forward, calls
``lm_head`` explicitly, then backprops a scalar loss and checks that every
TP-wrapped parameter received a gradient. Exercises the OLMo-2-specific
machinery the plan introduces: full-axis ``q_norm``/``k_norm`` wrapped
with ``LocalAxisRMSNormShard`` (Shard(0) weight + plain hidden-sharded
input from colwise q/k_proj), driven by ``Olmo2QKRMSNorm.forward`` running
the fused op on local tensors. The rest of the plan is stock
colwise/rowwise (no SP, no PrepareModuleInput).

Can be run two ways:
    1. pytest tests/distributed/test_olmo2_tp_e2e.py -v   (launches torchrun internally)
    2. torchrun --nproc_per_node=2 tests/distributed/test_olmo2_tp_e2e.py  (direct)
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from xorl.distributed.parallel_state import init_parallel_state
from xorl.distributed.torch_parallelize import _build_tp_plan
from xorl.models.transformers.olmo2.configuration_olmo2 import Olmo2Config
from xorl.models.transformers.olmo2.modeling_olmo2 import Olmo2ForCausalLM


def _make_config():
    cfg = Olmo2Config(
        architectures=["Olmo2ForCausalLM"],
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rope_theta=500000.0,
        initializer_range=0.5,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    cfg._attn_implementation = "eager"
    cfg._activation_native = True
    return cfg


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # OLMo-2's Olmo2Model.forward calls get_parallel_state() to pick a CP
    # strategy; initialize it as TP-only so NoopStrategy fires (no all-to-all
    # for ulysses, no ring-attention plumbing), keeping the test focused on
    # the TP plan we're validating.
    init_parallel_state(
        dp_size=1,
        tp_size=world_size,
        dp_mode="none",
        device_type="cpu",
    )
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    torch.manual_seed(0)
    model = Olmo2ForCausalLM(_make_config())

    # Apply unfuse + TP plan via xorl's standard pipeline so we exercise the
    # same code path the production launcher uses.
    model.unfuse_for_tp()
    plan = _build_tp_plan(model)
    parallelize_module(model, mesh, plan)

    # Forward.
    seq_len = 8
    input_ids = torch.randint(0, 64, (1, seq_len), generator=torch.Generator().manual_seed(rank))
    out = model(input_ids=input_ids)
    last_hidden = out.last_hidden_state

    # last_hidden_state comes out of the (un-wrapped) final norm with full
    # hidden replicated across ranks — the o_proj/down_proj rowwise default
    # all-reduces to Replicate, post-norms are NOT in the TP plan, and the
    # residual stream stays Replicate end-to-end. This shape matches what
    # vocab_parallel_cross_entropy expects (full [B, S, H] per rank).
    assert tuple(last_hidden.shape) == (1, seq_len, 16), (
        f"unexpected last_hidden_state shape: got {tuple(last_hidden.shape)}, expected (1, {seq_len}, 16)"
    )
    assert torch.isfinite(last_hidden).all(), "non-finite values in last_hidden_state"

    # Run lm_head explicitly (Olmo2ForCausalLM.forward returns the pre-lm-head
    # last_hidden_state; the loss head normally wraps it). lm_head is
    # ColwiseParallel — local input + vocab-sharded weight produce a local
    # vocab-parallel logits tensor.
    logits = model.lm_head(last_hidden)
    expected_logits = (1, seq_len, 64 // world_size)
    assert tuple(logits.shape) == expected_logits, (
        f"unexpected logits shape: got {tuple(logits.shape)}, expected {expected_logits}"
    )

    # Backward pass: take a scalar loss off the logits and check that gradients
    # reach every TP-wrapped parameter without raising. This catches the
    # redistribute/grad-flow paths a pure forward would miss
    # (LocalAxisRMSNormShard partial reduction, ColwiseParallel/RowwiseParallel
    # input/output redistribute backward, lm_head colwise).
    loss = logits.float().pow(2).mean()
    loss.backward()
    no_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"parameters missing gradients: {no_grad[:5]}"

    if rank == 0:
        print("OLMo-2 TP=2 fwd+bwd succeeded; last_hidden_state shape:", tuple(last_hidden.shape))

    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest

    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    @pytest.mark.cpu
    @pytest.mark.distributed
    def test_olmo2_tp2_fwd_bwd_e2e_cpu():
        """OLMo-2 fwd+bwd survives the full torchtitan-style TP plan on a 2-rank gloo mesh."""
        result = run_distributed_script(SCRIPT_PATH, num_gpus=2, timeout=180)
        result.assert_success()


if __name__ == "__main__":
    main()
