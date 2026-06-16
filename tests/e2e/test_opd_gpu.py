"""CUDA smoke coverage for OPD runner teacher grouping."""

import pytest
import torch

from tests._helpers.opd import make_teacher_files
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.server,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_opd_runner_grouped_teachers_cuda(tmp_path):
    torch.manual_seed(321)
    device = torch.device("cuda")
    vocab_size = 23
    hidden_size = 8
    seq_len = 6
    cache_size = 12

    teacher_heads = {
        "0": torch.randn(vocab_size, hidden_size) / hidden_size**0.5,
        "1": torch.randn(vocab_size, hidden_size) / hidden_size**0.5,
    }
    teacher_caches = {
        "0": torch.randn(cache_size, hidden_size) / hidden_size**0.5,
        "1": torch.randn(cache_size, hidden_size) / hidden_size**0.5,
    }
    teacher_files = make_teacher_files(tmp_path, teacher_heads, teacher_caches)

    runner = object.__new__(ModelRunner)
    runner.train_config = {}
    runner.lm_head_fp32 = True
    runner.pp_enabled = False
    runner._opd_head_manager = None
    runner._opd_head_config = None
    runner._opd_hidden_cache = None
    runner._opd_hidden_config = None

    hidden_states = (torch.randn(1, seq_len, hidden_size, device=device) / hidden_size**0.5).requires_grad_(True)
    student_weight = (torch.randn(vocab_size, hidden_size, device=device) / hidden_size**0.5).requires_grad_(True)
    micro_batch = {
        "labels": torch.tensor([[2, 3, 4, 5, 6, 7]], device=device),
        "teacher_ids": torch.tensor([[0, 0, 0, 1, 1, 1]], device=device),
        "teacher_cache_indices": torch.tensor([[0, 1, 2, 3, 4, 5]], device=device),
        "teacher_weights": torch.tensor([[1.0, 0.5, 1.5, 1.0, 0.25, 2.0]], device=device),
    }

    params = {
        "teacher_heads": teacher_files.heads,
        "teacher_hidden_caches": teacher_files.hidden_caches,
        "num_chunks": 2,
    }
    optimizer = torch.optim.Adam([hidden_states, student_weight], lr=0.1)
    loss_history: list[float] = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        result = runner._compute_opd_micro_batch_loss(
            hidden_states=hidden_states,
            student_weight=student_weight,
            micro_batch=micro_batch,
            params=params,
        )
        assert result.loss.isfinite()
        assert result.metrics["valid_tokens"] == seq_len
        assert result.metrics["opd_num_teachers"] == 2
        loss_history.append(float(result.loss.detach().cpu()))
        result.loss.backward()
        assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
        assert student_weight.grad is not None and student_weight.grad.isfinite().all()
        optimizer.step()

    assert loss_history[-1] < loss_history[0], f"OPD loss did not decrease: {loss_history}"
