"""W10 (C3): concurrent-batching / composition invariance of the dense forward.

The K3 recipe relies on every logprob being independent of what else shares its
batch. Under batch-invariant mode a row's op output — and a sequence's model
logits — must be bit-identical whether computed alone or co-batched with others.
These tests assert that at the op level (the foundation) and at the dense-model
forward level (batch dimension does not change per-sequence results).

The deeper serving-integration layer (packed block-diagonal composition,
continuous batching, scheduler variance) is exercised by the cross-engine K3
harness; here we lock the invariance that makes it possible.
"""

import pytest
import torch

from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from xorl.ops.sglang.batch_invariant_ops import set_batch_invariant_mode


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

D = 2048


def _assert_ops_are_batch_composition_invariant():
    """Row 0's output is bit-identical alone (M=1) vs in a batch (M=N), under BI."""
    torch.manual_seed(0)
    dev = "cuda"
    n = 64
    x = torch.randn(n, D, device=dev, dtype=torch.bfloat16)
    w = torch.randn(D, D, device=dev, dtype=torch.bfloat16)
    wn = torch.randn(D, device=dev, dtype=torch.bfloat16)

    with set_batch_invariant_mode(True):
        assert torch.equal((x @ w)[:1], x[:1] @ w), "mm not batch-invariant"
        assert torch.equal(
            torch.rms_norm(x, (D,), wn, eps=1e-6)[:1],
            torch.rms_norm(x[:1], (D,), wn, eps=1e-6),
        ), "rms_norm not batch-invariant"
        assert torch.equal(
            torch.log_softmax(x.float(), dim=-1)[:1],
            torch.log_softmax(x[:1].float(), dim=-1),
        ), "log_softmax not batch-invariant"
        assert torch.equal(x.float().mean(-1)[:1], x[:1].float().mean(-1)), "mean not batch-invariant"


@requires_cuda
@pytest.mark.gpu
def test_dense_batch_composition_invariance_policy():
    """A sequence's per-position logits are identical whether it is forwarded
    alone or as the first row of a padded batch, under batch-invariant mode."""
    _assert_ops_are_batch_composition_invariant()

    torch.manual_seed(1)
    dev = "cuda"
    cfg = Qwen3Config(
        hidden_size=D,
        intermediate_size=4096,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=3,
        vocab_size=2048,
        pad_token_id=0,
        _attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg).to(device=dev, dtype=torch.bfloat16).eval()

    la, lb = 24, 40
    seq_a = torch.randint(0, cfg.vocab_size, (1, la), device=dev)
    seq_b = torch.randint(0, cfg.vocab_size, (1, lb), device=dev)

    with set_batch_invariant_mode(True), torch.no_grad():
        # A alone (hidden states fully determine the logprobs; lm_head is a
        # per-row batch-invariant matmul under BI mode).
        out_solo = model(seq_a, position_ids=torch.arange(la, device=dev).unsqueeze(0))
        hidden_solo = out_solo.last_hidden_state[0]  # [la, hidden]

        # A right-padded into a batch with B (attention_mask hides padding).
        pad = lb - la
        a_padded = torch.cat([seq_a, torch.zeros(1, pad, dtype=seq_a.dtype, device=dev)], dim=1)
        batch_ids = torch.cat([a_padded, seq_b], dim=0)  # [2, lb]
        attn = torch.ones(2, lb, dtype=torch.long, device=dev)
        attn[0, la:] = 0  # mask A's padding
        pos = torch.arange(lb, device=dev).unsqueeze(0).expand(2, lb)
        out_batch = model(batch_ids, attention_mask=attn, position_ids=pos)
        hidden_a_in_batch = out_batch.last_hidden_state[0, :la]  # A's real positions

    assert torch.equal(hidden_solo, hidden_a_in_batch), (
        "dense forward is not batch-composition invariant: A's hidden states changed when co-batched"
    )
