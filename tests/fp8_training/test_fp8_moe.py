import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xorl.fp8_training import (
    FP8Linear,
    fp8_block_loop_group_gemm_same_nk,
    fp8_group_gemm_same_mn,
    fp8_scalar_quack_group_gemm_same_nk,
    fp8_triton_grouped_group_gemm_same_mn,
    fp8_triton_grouped_group_gemm_same_nk,
    inject_fp8_training_into_model,
    summarize_fp8_training_model,
)
from xorl.models.layers.moe.experts import MoEExperts


class TinyMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = MoEExperts(num_experts=2, hidden_dim=16, intermediate_size=32, moe_implementation="triton")


class TinyDenseMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(128, 128)
        self.experts = MoEExperts(num_experts=2, hidden_dim=128, intermediate_size=128, moe_implementation="triton")
        self.output_proj = nn.Linear(128, 32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_proj(hidden_states)
        hidden_states = self.experts(hidden_states, routing_weights, selected_experts)
        return self.output_proj(hidden_states)


def test_inject_fp8_training_enables_moe_experts_and_preserves_parameters():
    model = TinyMoEModel()
    original_gate_up = model.experts.gate_up_proj
    original_down = model.experts.down_proj

    changed = inject_fp8_training_into_model(model)

    assert changed == 1
    assert model.experts.gate_up_proj is original_gate_up
    assert model.experts.down_proj is original_down
    assert model.experts.moe_implementation == "quack"
    assert model.experts.fp8_training_enabled is True
    assert model.experts.fp8_training_grouped_backend == "triton_grouped"
    assert model.experts.last_forward_used_fp8 is False


def test_inject_fp8_training_enables_moe_experts_with_per_expert_biases():
    model = TinyMoEModel()
    model.experts.hidden_act = "clamped_swiglu"
    model.experts.gate_up_bias = nn.Parameter(torch.zeros(2, 64))
    model.experts.down_bias = nn.Parameter(torch.zeros(2, 16))
    original_gate_up_bias = model.experts.gate_up_bias
    original_down_bias = model.experts.down_bias

    changed = inject_fp8_training_into_model(model)

    assert changed == 1
    assert model.experts.gate_up_bias is original_gate_up_bias
    assert model.experts.down_bias is original_down_bias
    assert model.experts.moe_implementation == "quack"
    assert model.experts.fp8_training_enabled is True


def test_summarize_fp8_training_model_reports_unused_moe_modules():
    model = TinyMoEModel()
    inject_fp8_training_into_model(model)

    summary = summarize_fp8_training_model(model)

    assert summary["moe_modules"] == 1
    assert summary["moe_fp8_enabled_modules"] == 1
    assert summary["moe_modules_used_fp8"] == 0
    assert summary["unused_moe_module_names"] == ["experts"]


def test_quack_moe_forward_tp_threads_fp8_compute(monkeypatch):
    from xorl.ops.moe import quack as quack_ops  # noqa: PLC0415

    expected = torch.randn(3, 4)
    apply = MagicMock(return_value=expected)
    monkeypatch.setattr(quack_ops.QuackTPMoeExpertsFunction, "apply", apply)

    tp_group = object()
    parallel_state = SimpleNamespace(tp_enabled=True, tp_mesh=SimpleNamespace(get_group=lambda: tp_group))
    with patch("xorl.ops.moe.quack.get_parallel_state", return_value=parallel_state):
        output = quack_ops.quack_moe_forward(
            module=None,
            num_experts=2,
            routing_weights=torch.ones(3, 1),
            selected_experts=torch.zeros(3, 1, dtype=torch.long),
            hidden_states=torch.randn(3, 4),
            gate_proj=torch.randn(2, 4, 8),
            up_proj=torch.randn(2, 4, 8),
            down_proj=torch.randn(2, 8, 4),
            hidden_act="silu",
            fp8_compute=True,
            fp8_grouped_backend="triton_grouped",
            fp8_block_size=64,
        )

    torch.testing.assert_close(output, expected)
    args = apply.call_args.args
    assert args[7] is tp_group
    assert args[9] is False  # activation_native
    assert args[10] is True
    assert args[11] == "triton_grouped"
    assert args[12] == 64


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quack_moe_tp_fp8_single_rank_train_step_updates_master_weights(monkeypatch):
    from xorl.ops.moe import quack as quack_ops  # noqa: PLC0415

    class _NoOpWork:
        def wait(self):
            return None

    def fake_all_reduce(tensor, group=None, async_op=False):
        del tensor, group
        return _NoOpWork() if async_op else None

    monkeypatch.setattr(quack_ops.dist, "all_reduce", fake_all_reduce)

    torch.manual_seed(0)
    num_experts = 2
    hidden_dim = 128
    intermediate_size = 128
    gate_proj = nn.Parameter(
        (torch.randn(num_experts, hidden_dim, intermediate_size, device="cuda", dtype=torch.bfloat16) * 0.02)
    )
    up_proj = nn.Parameter(
        (torch.randn(num_experts, hidden_dim, intermediate_size, device="cuda", dtype=torch.bfloat16) * 0.02)
    )
    down_proj = nn.Parameter(
        (torch.randn(num_experts, intermediate_size, hidden_dim, device="cuda", dtype=torch.bfloat16) * 0.02)
    )
    hidden_states = torch.randn(8, hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    routing_weights = torch.ones(8, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    selected_experts = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], device="cuda", dtype=torch.long)

    before_gate = gate_proj.detach().clone()
    parallel_state = SimpleNamespace(
        tp_enabled=True,
        tp_mesh=SimpleNamespace(get_group=lambda: object()),
    )
    with patch("xorl.ops.moe.quack.get_parallel_state", return_value=parallel_state):
        output = quack_ops.quack_moe_forward(
            module=None,
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            hidden_act="silu",
            fp8_compute=True,
            fp8_grouped_backend="triton_grouped",
        )

    loss = output.float().pow(2).mean()
    loss.backward()
    opt = torch.optim.SGD([gate_proj, up_proj, down_proj], lr=1e-1)
    opt.step()

    assert torch.isfinite(output.float()).all()
    assert hidden_states.grad is not None
    assert routing_weights.grad is not None
    assert gate_proj.grad is not None
    assert up_proj.grad is not None
    assert down_proj.grad is not None
    assert torch.isfinite(gate_proj.grad.float()).all()
    assert torch.isfinite(up_proj.grad.float()).all()
    assert torch.isfinite(down_proj.grad.float()).all()
    assert not torch.equal(gate_proj.detach(), before_gate)


def _grouped_same_nk_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum: torch.Tensor,
    *,
    transpose_b: bool = False,
) -> torch.Tensor:
    starts = torch.cat([torch.zeros(1, device=cumsum.device, dtype=cumsum.dtype), cumsum[:-1]])
    chunks = []
    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum.tolist())):
        rhs = b[expert_idx].float().T if transpose_b else b[expert_idx].float()
        chunks.append(a[start:end].float() @ rhs)
    return torch.cat(chunks, dim=0)


def _grouped_same_mn_reference(a: torch.Tensor, b: torch.Tensor, cumsum: torch.Tensor) -> torch.Tensor:
    starts = torch.cat([torch.zeros(1, device=cumsum.device, dtype=cumsum.dtype), cumsum[:-1]])
    chunks = []
    for start, end in zip(starts.tolist(), cumsum.tolist()):
        chunks.append(a[start:end].float().T @ b[start:end].float())
    return torch.stack(chunks, dim=0)


def _cu_seqlens(cumsum: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1, device=cumsum.device, dtype=torch.int32), cumsum.to(torch.int32)])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_block_grouped_same_nk_helper_matches_bf16_reference():
    torch.manual_seed(0)
    cumsum = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    a = (torch.randn(8, 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    got = fp8_block_loop_group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum,
        max_M=4,
    )
    expected = _grouped_same_nk_reference(a, b, cumsum)

    assert torch.isfinite(got.float()).all()
    assert torch.allclose(got.float(), expected, rtol=0.25, atol=0.5)


@pytest.mark.parametrize(
    ("lengths", "k", "n"),
    [
        ([4, 4], 128, 64),
        ([0, 3, 7], 192, 160),
        ([1, 129, 131], 257, 96),
        ([2, 0, 260], 128, 257),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_triton_grouped_same_nk_helper_matches_bf16_reference(lengths, k, n):
    torch.manual_seed(sum(lengths) + k + n)
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    total_tokens = int(cumsum[-1].item())
    a = (torch.randn(total_tokens, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(len(lengths), k, n, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    got = fp8_triton_grouped_group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum,
        max_M=max(lengths),
    )
    expected = _grouped_same_nk_reference(a, b, cumsum)

    assert torch.isfinite(got.float()).all()
    assert torch.allclose(got.float(), expected, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_triton_grouped_helpers_honor_non_default_block_size():
    torch.manual_seed(0)
    lengths = [3, 5]
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    total_tokens = int(cumsum[-1].item())
    a = (torch.randn(total_tokens, 160, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(len(lengths), 160, 96, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    got_nk = fp8_triton_grouped_group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum,
        max_M=max(lengths),
        block_size=64,
    )
    expected_nk = _grouped_same_nk_reference(a, b, cumsum)

    got_mn = torch.empty(len(lengths), 160, 96, device="cuda", dtype=torch.bfloat16)
    fp8_triton_grouped_group_gemm_same_mn(
        a=a,
        b=got_nk.contiguous(),
        c=got_mn,
        cumsum_K=cumsum,
        max_K=max(lengths),
        transpose_a=True,
        block_size=64,
    )
    expected_mn = _grouped_same_mn_reference(a, got_nk, cumsum)

    assert torch.isfinite(got_nk.float()).all()
    assert torch.allclose(got_nk.float(), expected_nk, rtol=0.25, atol=0.5)
    assert torch.isfinite(got_mn.float()).all()
    assert torch.allclose(got_mn.float(), expected_mn, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_triton_grouped_same_nk_uses_precomputed_cu_seqlens_for_dgrad_shape():
    torch.manual_seed(0)
    lengths = [2, 0, 5]
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    bogus_cumsum = torch.zeros_like(cumsum)
    a = (torch.randn(int(cumsum[-1].item()), 192, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(len(lengths), 96, 192, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    got = fp8_triton_grouped_group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=bogus_cumsum,
        max_M=max(lengths),
        transpose_b=True,
        cu_seqlens_m=_cu_seqlens(cumsum),
    )
    expected = _grouped_same_nk_reference(a, b, cumsum, transpose_b=True)

    assert torch.isfinite(got.float()).all()
    assert torch.allclose(got.float(), expected, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_block_wgrad_helper_matches_bf16_reference():
    torch.manual_seed(0)
    cumsum = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    a = (torch.randn(8, 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b_mn = (torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    got_mn = torch.empty(2, 128, 64, device="cuda", dtype=torch.bfloat16)
    fp8_group_gemm_same_mn(
        a=a,
        b=b_mn,
        c=got_mn,
        cumsum_K=cumsum,
        max_K=4,
        transpose_a=True,
        backend="block_loop",
    )
    expected_mn = _grouped_same_mn_reference(a, b_mn, cumsum)

    assert torch.isfinite(got_mn.float()).all()
    assert torch.allclose(got_mn.float(), expected_mn, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_scalar_quack_same_mn_dispatch_uses_block_loop(monkeypatch):
    from xorl.fp8_training import grouped as fp8_grouped  # noqa: PLC0415

    torch.manual_seed(0)
    lengths = [3, 5]
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    a = (torch.randn(int(cumsum[-1].item()), 96, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b_mn = (torch.randn(int(cumsum[-1].item()), 160, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    got_mn = torch.empty(len(lengths), 96, 160, device="cuda", dtype=torch.bfloat16)
    original_block_loop = fp8_grouped.fp8_block_loop_group_gemm_same_mn
    calls = {"block_loop": 0}

    def spy_block_loop(**kwargs):
        calls["block_loop"] += 1
        return original_block_loop(**kwargs)

    monkeypatch.setattr(fp8_grouped, "fp8_block_loop_group_gemm_same_mn", spy_block_loop)

    fp8_group_gemm_same_mn(
        a=a,
        b=b_mn,
        c=got_mn,
        cumsum_K=cumsum,
        max_K=max(lengths),
        transpose_a=True,
        backend="scalar_quack",
    )
    expected_mn = _grouped_same_mn_reference(a, b_mn, cumsum)

    assert calls["block_loop"] == 1
    assert torch.isfinite(got_mn.float()).all()
    assert torch.allclose(got_mn.float(), expected_mn, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_triton_grouped_wgrad_helper_uses_precomputed_cu_seqlens():
    torch.manual_seed(0)
    lengths = [3, 0, 9]
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    bogus_cumsum = torch.zeros_like(cumsum)
    a = (torch.randn(int(cumsum[-1].item()), 96, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b_mn = (torch.randn(int(cumsum[-1].item()), 160, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    got_mn = torch.empty(len(lengths), 96, 160, device="cuda", dtype=torch.bfloat16)
    fp8_triton_grouped_group_gemm_same_mn(
        a=a,
        b=b_mn,
        c=got_mn,
        cumsum_K=bogus_cumsum,
        max_K=max(lengths),
        transpose_a=True,
        cu_seqlens_k=_cu_seqlens(cumsum),
    )
    expected_mn = _grouped_same_mn_reference(a, b_mn, cumsum)

    assert torch.isfinite(got_mn.float()).all()
    assert torch.allclose(got_mn.float(), expected_mn, rtol=0.25, atol=0.5)


@pytest.mark.parametrize(
    ("lengths", "m", "n"),
    [
        ([4, 4], 128, 64),
        ([0, 0, 0], 64, 128),
        ([0, 3, 7], 64, 32),
        ([1, 129, 131], 96, 160),
        ([0, 0, 2, 260], 256, 257),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_triton_grouped_wgrad_helper_matches_bf16_reference(lengths, m, n):
    torch.manual_seed(sum(lengths) + m + n)
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    total_tokens = int(cumsum[-1].item())
    a = (torch.randn(total_tokens, m, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b_mn = (torch.randn(total_tokens, n, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    got_mn = torch.empty(len(lengths), m, n, device="cuda", dtype=torch.bfloat16)
    fp8_triton_grouped_group_gemm_same_mn(
        a=a,
        b=b_mn,
        c=got_mn,
        cumsum_K=cumsum,
        max_K=max(lengths),
        transpose_a=True,
    )
    expected_mn = _grouped_same_mn_reference(a, b_mn, cumsum)

    assert torch.isfinite(got_mn.float()).all()
    assert torch.allclose(got_mn.float(), expected_mn, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    os.environ.get("XORL_TEST_DEEP_GEMM_FP8") != "1",
    reason="DeepGEMM grouped FP8 binding is opt-in until stable under repeated training calls",
)
def test_fp8_deep_gemm_grouped_helper_subprocess():
    # The installed DeepGEMM Any-argument binding rejects tensor storage after
    # pytest is imported. Validate that backend in a clean child interpreter.
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    script = r"""
import torch

from xorl.fp8_training import fp8_group_gemm_same_nk

def grouped_same_nk_reference(a, b, cumsum):
    starts = torch.cat([torch.zeros(1, device=cumsum.device, dtype=cumsum.dtype), cumsum[:-1]])
    chunks = []
    for expert_idx, (start, end) in enumerate(zip(starts.tolist(), cumsum.tolist())):
        chunks.append(a[start:end].float() @ b[expert_idx].float())
    return torch.cat(chunks, dim=0)


torch.manual_seed(0)
cumsum = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
a = (torch.randn(8, 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
b = (torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
got = fp8_group_gemm_same_nk(a=a, b=b, cumsum_M=cumsum, max_M=4, backend="deep_gemm")
expected = grouped_same_nk_reference(a, b, cumsum)
assert got.dtype == torch.bfloat16
assert torch.isfinite(got.float()).all()
assert torch.allclose(got.float(), expected, rtol=0.25, atol=0.5)
print("deep_gemm_helper_subprocess_ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_scalar_quack_fallback_matches_bf16_reference():
    torch.manual_seed(0)
    cumsum = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    a = (torch.randn(8, 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    got = fp8_scalar_quack_group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum,
        max_M=4,
    )
    expected = _grouped_same_nk_reference(a, b, cumsum)

    assert torch.isfinite(got.float()).all()
    assert torch.allclose(got.float(), expected, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_scalar_quack_uses_cu_seqlens_for_per_expert_scales():
    torch.manual_seed(0)
    lengths = [2, 6]
    cumsum = torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0)
    bogus_cumsum = torch.zeros_like(cumsum)
    a = (torch.randn(int(cumsum[-1].item()), 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(len(lengths), 64, 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    a[: lengths[0]].mul_(0.01)
    b[0].mul_(0.01)

    got = fp8_scalar_quack_group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=bogus_cumsum,
        max_M=max(lengths),
        transpose_b=True,
        cu_seqlens_m=_cu_seqlens(cumsum),
    )
    expected = _grouped_same_nk_reference(a, b, cumsum, transpose_b=True)

    assert torch.isfinite(got.float()).all()
    assert torch.allclose(got.float(), expected, rtol=0.25, atol=0.5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8_grouped_backend", ["triton_grouped", "scalar_quack"])
def test_moe_experts_quack_fp8_train_step_updates_master_weights(fp8_grouped_backend):
    torch.manual_seed(0)
    experts = MoEExperts(num_experts=2, hidden_dim=128, intermediate_size=128, moe_implementation="quack")
    experts = experts.to(device="cuda", dtype=torch.bfloat16)
    experts.fp8_training_enabled = True
    experts.fp8_training_grouped_backend = fp8_grouped_backend
    experts.fp8_training_block_size = 64
    with torch.no_grad():
        experts.gate_up_proj.normal_(mean=0.0, std=0.02)
        experts.down_proj.normal_(mean=0.0, std=0.02)

    opt = torch.optim.SGD(experts.parameters(), lr=1e-1)
    hidden_states = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.ones(8, 1, device="cuda", dtype=torch.bfloat16)
    selected_experts = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], device="cuda", dtype=torch.long)
    before = experts.gate_up_proj.detach().clone()

    output = experts(hidden_states, routing_weights, selected_experts)
    loss = output.float().pow(2).mean()
    loss.backward()
    opt.step()

    assert torch.isfinite(output.float()).all()
    assert experts.last_forward_used_fp8 is True
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None
    assert torch.isfinite(experts.gate_up_proj.grad.float()).all()
    assert torch.isfinite(experts.down_proj.grad.float()).all()
    assert not torch.equal(experts.gate_up_proj.detach(), before)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_experts_quack_fp8_train_step_with_biases_and_clamped_swiglu():
    torch.manual_seed(0)
    experts = MoEExperts(num_experts=2, hidden_dim=128, intermediate_size=128, moe_implementation="quack")
    experts.hidden_act = "clamped_swiglu"
    experts.gate_up_bias = nn.Parameter(torch.zeros(2, 256))
    experts.down_bias = nn.Parameter(torch.zeros(2, 128))
    experts = experts.to(device="cuda", dtype=torch.bfloat16)
    experts.fp8_training_enabled = True
    experts.fp8_training_grouped_backend = "triton_grouped"
    with torch.no_grad():
        experts.gate_up_proj.normal_(mean=0.0, std=0.02)
        experts.down_proj.normal_(mean=0.0, std=0.02)
        experts.gate_up_bias.normal_(mean=0.0, std=0.01)
        experts.down_bias.normal_(mean=0.0, std=0.01)

    opt = torch.optim.SGD(experts.parameters(), lr=1e-1)
    hidden_states = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.ones(8, 1, device="cuda", dtype=torch.bfloat16)
    selected_experts = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], device="cuda", dtype=torch.long)
    before_gate_up_bias = experts.gate_up_bias.detach().clone()
    before_down_bias = experts.down_bias.detach().clone()

    output = experts(hidden_states, routing_weights, selected_experts)
    loss = output.float().pow(2).mean()
    loss.backward()
    opt.step()

    assert torch.isfinite(output.float()).all()
    assert experts.last_forward_used_fp8 is True
    assert experts.gate_up_bias.grad is not None
    assert experts.down_bias.grad is not None
    assert torch.isfinite(experts.gate_up_bias.grad.float()).all()
    assert torch.isfinite(experts.down_bias.grad.float()).all()
    assert not torch.equal(experts.gate_up_bias.detach(), before_gate_up_bias)
    assert not torch.equal(experts.down_bias.detach(), before_down_bias)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8_grouped_backend", ["triton_grouped", "scalar_quack"])
def test_injected_dense_and_moe_fp8_model_train_step_updates_master_weights(fp8_grouped_backend):
    torch.manual_seed(0)
    model = TinyDenseMoEModel()
    changed = inject_fp8_training_into_model(
        model,
        exclude_modules=[],
        moe_grouped_backend=fp8_grouped_backend,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(mean=0.0, std=0.02)
            else:
                param.zero_()

    assert changed == 3
    assert isinstance(model.input_proj, FP8Linear)
    assert isinstance(model.output_proj, FP8Linear)
    assert model.experts.moe_implementation == "quack"
    assert model.experts.fp8_training_enabled is True
    assert model.experts.fp8_training_grouped_backend == fp8_grouped_backend

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    hidden_states = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.ones(8, 1, device="cuda", dtype=torch.bfloat16)
    selected_experts = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], device="cuda", dtype=torch.long)
    target = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
    before_input = model.input_proj.weight.detach().clone()
    before_expert = model.experts.gate_up_proj.detach().clone()
    before_output = model.output_proj.weight.detach().clone()

    output = model(hidden_states, routing_weights, selected_experts)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    opt.step()

    assert model.input_proj.last_forward_used_fp8 is True
    assert model.experts.last_forward_used_fp8 is True
    assert model.output_proj.last_forward_used_fp8 is True
    assert model.input_proj.weight.grad is not None
    assert model.output_proj.weight.grad is not None
    assert model.experts.gate_up_proj.grad is not None
    assert model.experts.down_proj.grad is not None
    assert torch.isfinite(model.input_proj.weight.grad.float()).all()
    assert torch.isfinite(model.output_proj.weight.grad.float()).all()
    assert torch.isfinite(model.experts.gate_up_proj.grad.float()).all()
    assert torch.isfinite(model.experts.down_proj.grad.float()).all()
    assert not torch.equal(model.input_proj.weight.detach(), before_input)
    assert not torch.equal(model.experts.gate_up_proj.detach(), before_expert)
    assert not torch.equal(model.output_proj.weight.detach(), before_output)
