import pytest
import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.ops.gated_delta_rule import chunk_gated_delta_rule


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_chunk_gated_delta_rule_backward_real_qwen35_shape():
    """Regression test for the Hopper illegal-memory-access autotune candidate."""
    torch.manual_seed(0)
    device = "cuda"
    batch, seq_len, num_heads, head_dim = 1, 4096, 4, 128

    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    beta = torch.rand(batch, seq_len, num_heads, device=device, dtype=torch.float32).sigmoid().requires_grad_()
    g = F.logsigmoid(torch.randn(batch, seq_len, num_heads, device=device, dtype=torch.float32)).requires_grad_()
    h0 = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=torch.float32, requires_grad=True)

    o, ht = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    loss = o.float().square().mean() + ht.float().square().mean()
    loss.backward()

    for grad in (q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
