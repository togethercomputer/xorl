from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import load_file

from tests._helpers.opd import make_teacher_files
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _make_opd_runner() -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.world_size = 1
    runner.train_config = {}
    runner.lm_head_fp32 = True
    runner.pp_enabled = False
    runner.model_fwd_context = nullcontext()
    runner._opd_head_manager = None
    runner._opd_head_config = None
    runner._opd_hidden_cache = None
    runner._opd_hidden_config = None
    return runner


class _FakeTeacherOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _InputIdHiddenModel:
    def __call__(self, input_ids, **_kwargs):
        return _FakeTeacherOutput(input_ids.float().unsqueeze(-1))


class _RecordingLmHead(torch.nn.Linear):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__(hidden_size, vocab_size, bias=False)
        self.calls = 0
        self.last_input_shape = None

    def forward(self, input):
        self.calls += 1
        self.last_input_shape = tuple(input.shape)
        return super().forward(input)


class _NoopRoutingHandler:
    def setup(self, *_args, **_kwargs):
        return False


def test_forward_uses_no_grad_not_inference_mode():
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=10))
    runner.pp_enabled = False
    runner._adapter_manager = None
    runner.global_forward_backward_step = 0
    runner._routing_handler = _NoopRoutingHandler()
    runner._check_not_sleeping = lambda *_args, **_kwargs: None
    runner._validate_single_tenant = lambda *_args, **_kwargs: None

    seen = {}

    def fake_forward_loop(*_args, **_kwargs):
        seen["grad_enabled"] = torch.is_grad_enabled()
        seen["inference_mode"] = torch.is_inference_mode_enabled()
        return {"total_loss": 0.0, "global_valid_tokens": 1}

    runner._forward_loop = fake_forward_loop

    result = runner.forward([{"input_ids": torch.tensor([[1]])}], loss_fn="teacher_hidden_cache")

    assert result["step"] == 0
    assert seen == {"grad_enabled": False, "inference_mode": False}


@patch("xorl.server.runner.model_runner.get_parallel_state")
def test_opd_metrics_keep_opd_namespace(mock_parallel_state):
    mock_parallel_state.return_value = Mock(dp_enabled=False, loss_parallel_enabled=False)
    accumulated = {}

    ModelRunner._accumulate_loss_metrics(
        accumulated,
        {
            "valid_tokens": 4,
            "opd_kl": 0.5,
            "opd_weighted_kl": 0.6,
            "opd_num_teachers": 2,
            "opd_profile_kl_compute_ms": 10.0,
        },
        "opd_loss",
    )
    ModelRunner._accumulate_loss_metrics(
        accumulated,
        {
            "valid_tokens": 2,
            "opd_kl": 0.2,
            "opd_weighted_kl": 0.3,
            "opd_num_teachers": 1,
            "opd_profile_kl_compute_ms": 20.0,
        },
        "opd_loss",
    )

    result = {}
    ModelRunner._finalize_loss_metrics(accumulated, result)

    assert result["opd_kl"] == pytest.approx((0.5 * 4 + 0.2 * 2) / 6)
    assert result["opd_weighted_kl"] == pytest.approx((0.6 * 4 + 0.3 * 2) / 6)
    assert result["opd_num_teachers:max"] == 2
    assert result["opd_profile_kl_compute_ms"] == pytest.approx(30.0)
    assert not any(key.startswith("is_opd") for key in result)


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def test_opd_metrics_reduce_over_loss_group(mock_parallel_state, _mock_get_device_type):
    loss_group = object()
    mock_parallel_state.return_value = Mock(loss_parallel_enabled=True, loss_group=loss_group)
    accumulated = {}
    ModelRunner._accumulate_loss_metrics(
        accumulated,
        {"valid_tokens": 3, "opd_kl": 0.5, "opd_num_teachers": 2},
        "opd_loss",
    )

    groups = []

    def fake_all_reduce(_tensor, op=None, group=None):
        groups.append(group)

    with (
        patch("xorl.server.runner.model_runner.dist.is_available", return_value=True),
        patch("xorl.server.runner.model_runner.dist.is_initialized", return_value=True),
        patch("xorl.server.runner.model_runner.dist.all_reduce", side_effect=fake_all_reduce),
    ):
        result = {}
        ModelRunner._finalize_loss_metrics(accumulated, result, "opd_loss")

    assert groups
    assert all(group is loss_group for group in groups)


@patch("xorl.server.runner.model_runner.synchronize", lambda: None)
@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def test_forward_loop_accumulates_opd_metrics_without_metric_ops(mock_parallel_state, _mock_get_device_type):
    mock_parallel_state.return_value = Mock(
        cp_enabled=False,
        loss_parallel_enabled=False,
        dp_enabled=False,
    )
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.pp_enabled = False
    runner.model_fwd_context = nullcontext()
    runner._use_distsignsgd = False
    runner._count_global_valid_tokens = lambda _micro_batches: torch.tensor(2)
    runner._collect_per_token_outputs = Mock()

    def fake_compute_micro_batch_loss(_micro_batch, _loss_fn, _params):
        loss = torch.tensor(4.0)
        metrics = {
            "valid_tokens": 2,
            "opd_kl": 0.5,
            "opd_weighted_kl": 0.75,
        }
        return loss, {}, metrics, None, SimpleNamespace()

    runner._compute_micro_batch_loss = fake_compute_micro_batch_loss

    result = runner._forward_loop(
        [{"labels": torch.tensor([[1, 2]])}],
        "opd_loss",
        {},
        compute_backward=False,
    )

    assert result["total_loss"] == pytest.approx(2.0)
    assert result["global_valid_tokens"] == 2
    assert result["opd_kl"] == pytest.approx(0.5)
    assert result["opd_weighted_kl"] == pytest.approx(0.75)


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
def test_opd_runner_masks_cache_indices_per_teacher(_mock_get_device_type, tmp_path):
    torch.manual_seed(7)
    vocab_size = 13
    hidden_size = 4
    seq_len = 4

    teacher_heads = {
        "0": torch.randn(vocab_size, hidden_size) / hidden_size**0.5,
        "1": torch.randn(vocab_size, hidden_size) / hidden_size**0.5,
    }
    teacher_caches = {
        "0": torch.randn(2, hidden_size) / hidden_size**0.5,
        "1": torch.randn(12, hidden_size) / hidden_size**0.5,
    }
    teacher_files = make_teacher_files(tmp_path, teacher_heads, teacher_caches)

    runner = _make_opd_runner()
    hidden_states = (torch.randn(1, seq_len, hidden_size) / hidden_size**0.5).requires_grad_(True)
    student_weight = (torch.randn(vocab_size, hidden_size) / hidden_size**0.5).requires_grad_(True)
    micro_batch = {
        "labels": torch.tensor([[2, 3, 4, 5]]),
        "teacher_ids": torch.tensor([[0, 0, 1, 1]]),
        "teacher_cache_indices": torch.tensor([[0, 1, 10, 11]]),
        "teacher_weights": torch.ones(1, seq_len),
    }
    params = {
        "teacher_heads": teacher_files.heads,
        "teacher_hidden_caches": teacher_files.hidden_caches,
        "num_chunks": 2,
        "opd_kl_backend": "streaming",
        "opd_vocab_chunk_size": 5,
        "opd_profile_timings": True,
    }

    result = runner._compute_opd_micro_batch_loss(
        hidden_states=hidden_states,
        student_weight=student_weight,
        micro_batch=micro_batch,
        params=params,
    )

    assert result.loss.isfinite()
    assert result.metrics["valid_tokens"] == seq_len
    assert result.metrics["opd_num_teachers"] == 2
    assert result.metrics["opd_profile_hidden_fetch_ms"] >= 0.0
    assert result.metrics["opd_profile_head_prepare_ms"] >= 0.0
    assert result.metrics["opd_profile_kl_compute_ms"] >= 0.0
    assert result.metrics["opd_profile_total_ms"] >= result.metrics["opd_profile_kl_compute_ms"]
    result.loss.backward()
    assert hidden_states.grad is not None
    assert student_weight.grad is not None


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
def test_opd_runner_runs_lm_head_anchor_for_fsdp(_mock_get_device_type, tmp_path):
    torch.manual_seed(11)
    seq_len = 3
    hidden_size = 4
    teacher_hidden_size = 5
    vocab_size = 9
    teacher_heads = {"0": torch.randn(vocab_size, teacher_hidden_size) / teacher_hidden_size**0.5}
    teacher_caches = {"0": torch.randn(seq_len, teacher_hidden_size) / teacher_hidden_size**0.5}
    teacher_files = make_teacher_files(tmp_path, teacher_heads, teacher_caches)

    runner = _make_opd_runner()
    hidden_states = (torch.randn(1, seq_len, hidden_size) / hidden_size**0.5).requires_grad_(True)
    lm_head = _RecordingLmHead(hidden_size, vocab_size)
    micro_batch = {
        "labels": torch.tensor([[1, 2, 3]]),
        "teacher_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "teacher_cache_indices": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
    }
    params = {
        "teacher_heads": teacher_files.heads,
        "teacher_hidden_caches": teacher_files.hidden_caches,
        "opd_kl_backend": "streaming",
        "opd_vocab_chunk_size": 4,
    }

    result = runner._compute_opd_micro_batch_loss(
        hidden_states=hidden_states,
        student_weight=lm_head.weight,
        micro_batch=micro_batch,
        params=params,
        student_lm_head=lm_head,
    )

    assert lm_head.calls == 1
    assert lm_head.last_input_shape == (1, hidden_size)
    result.loss.backward()
    assert hidden_states.grad is not None and hidden_states.grad.isfinite().all()
    assert lm_head.weight.grad is not None and lm_head.weight.grad.isfinite().all()


def test_teacher_hidden_cache_splits_packed_batch_and_drops_padding():
    runner = _make_opd_runner()
    hidden_states = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)
    micro_batch = {
        "num_samples": 2,
        # Two real samples with lengths 3 and 2, then a padding segment that
        # also starts at position 0.
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]]),
    }

    chunks = runner._teacher_hidden_chunks_from_batch(hidden_states, micro_batch)

    assert len(chunks) == 2
    assert torch.equal(chunks[0], hidden_states[0, 0:3])
    assert torch.equal(chunks[1], hidden_states[0, 3:5])


def test_teacher_hidden_cache_contributor_skips_duplicate_cp_and_ep_ranks():
    runner = _make_opd_runner()
    runner.rank = 3
    runner.world_size = 8

    assert (
        runner._teacher_hidden_cache_contributor_key(SimpleNamespace(cp_enabled=True, cp_rank=1, ep_enabled=False))
        is None
    )
    assert (
        runner._teacher_hidden_cache_contributor_key(
            SimpleNamespace(cp_enabled=True, cp_rank=0, ep_enabled=False, dp_rank=2)
        )
        == 2
    )
    assert (
        runner._teacher_hidden_cache_contributor_key(SimpleNamespace(cp_enabled=False, ep_enabled=True, ep_rank=1))
        is None
    )

    class FakeEpMesh:
        @staticmethod
        def get_local_rank(dim):
            assert dim == "ep_fsdp"
            return 3

    assert (
        runner._teacher_hidden_cache_contributor_key(
            SimpleNamespace(cp_enabled=False, ep_enabled=True, ep_rank=0, ep_fsdp_device_mesh=FakeEpMesh())
        )
        == 3
    )


def test_teacher_hidden_cache_merge_preserves_logical_slice_order():
    chunks, indices = ModelRunner._merge_teacher_hidden_cache_payloads(
        [
            {"rank": 4, "slice_key": 1, "chunks": [torch.ones(2, 2)]},
            None,
            {"rank": 0, "slice_key": 0, "chunks": [torch.zeros(1, 2)]},
        ]
    )

    assert indices == [[0], [1, 2]]
    assert torch.equal(torch.cat(chunks, dim=0), torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]))


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.gather_outputs")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def test_teacher_hidden_cache_gathers_with_unified_sp_group(mock_parallel_state, mock_gather, _mock_device, tmp_path):
    runner = _make_opd_runner()
    runner.rank = 0
    runner.world_size = 1
    runner.model_fwd_context = nullcontext()

    class FakeModel:
        def __call__(self, **_kwargs):
            return SimpleNamespace(last_hidden_state=torch.arange(1 * 2 * 2, dtype=torch.float32).reshape(1, 2, 2))

    runner.model = FakeModel()
    mock_parallel_state.return_value = SimpleNamespace(
        cp_enabled=True,
        cp_size=4,
        cp_rank=0,
        sp_group="full-sp-group",
        ep_enabled=False,
        dp_rank=0,
    )
    gathered = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)
    mock_gather.return_value = gathered

    result = runner._forward_teacher_hidden_cache(
        [
            {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "_original_position_ids": torch.arange(8, dtype=torch.long).view(1, 8),
            }
        ],
        {"teacher_hidden_cache_path": str(tmp_path / "teacher.safetensors")},
    )

    mock_gather.assert_called_once()
    assert mock_gather.call_args.kwargs["group"] == "full-sp-group"
    assert mock_gather.call_args.kwargs["unpad_dim_size"] == 8
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [list(range(8))]


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def test_teacher_hidden_cache_trims_with_gathered_sp_labels(mock_parallel_state, _mock_get_device_type, tmp_path):
    runner = _make_opd_runner()
    runner.model = _InputIdHiddenModel()
    mock_parallel_state.return_value = Mock(
        cp_enabled=True,
        cp_rank=0,
        cp_size=2,
        ulysses_group=object(),
        ep_enabled=False,
        dp_rank=0,
    )

    full_hidden = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    full_labels = torch.tensor([[-100, -100, -100, -100, 9, -100]])

    def fake_gather_outputs(tensor, **_kwargs):
        return full_hidden if torch.is_floating_point(tensor) else full_labels

    cache_path = tmp_path / "teacher_hidden.safetensors"
    with patch("xorl.server.runner.model_runner.gather_outputs", side_effect=fake_gather_outputs):
        result = runner._forward_teacher_hidden_cache(
            [
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "labels": torch.tensor([[-100, -100, -100]]),
                    "_original_position_ids": torch.arange(6, dtype=torch.long).unsqueeze(0),
                }
            ],
            {
                "teacher_hidden_cache_path": str(cache_path),
                "teacher_hidden_cache_dtype": "float32",
            },
        )

    saved = load_file(str(cache_path))["hidden_states"]
    assert torch.equal(saved, full_hidden[0, :5])
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [list(range(5))]


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def test_teacher_hidden_cache_writer_gathers_all_batch_ranks(
    mock_parallel_state,
    _mock_get_device_type,
    tmp_path,
):
    runner = _make_opd_runner()
    runner.world_size = 2
    runner.model = _InputIdHiddenModel()
    mock_parallel_state.return_value = Mock(cp_enabled=False, ep_enabled=False, dp_rank=0)

    remote_chunk = torch.tensor([[10.0], [11.0], [12.0]])

    def fake_gather_object(payload, object_gather_list, dst):
        assert dst == 0
        object_gather_list[:] = [payload, {"rank": 1, "slice_key": 1, "chunks": [remote_chunk]}]

    cache_path = tmp_path / "teacher_hidden.safetensors"
    with (
        patch("xorl.server.runner.model_runner.dist.is_available", return_value=True),
        patch("xorl.server.runner.model_runner.dist.is_initialized", return_value=True),
        patch("xorl.server.runner.model_runner.dist.get_world_size", return_value=2),
        patch("xorl.server.runner.model_runner.dist.gather_object", side_effect=fake_gather_object),
        patch("xorl.server.runner.model_runner.dist.broadcast_object_list"),
    ):
        result = runner._forward_teacher_hidden_cache(
            [
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "labels": torch.tensor([[1, 2, -100]]),
                }
            ],
            {
                "teacher_hidden_cache_path": str(cache_path),
                "teacher_hidden_cache_dtype": "float32",
            },
        )

    saved = load_file(str(cache_path))["hidden_states"]
    assert torch.equal(saved, torch.tensor([[1.0], [2.0], [10.0], [11.0], [12.0]]))
    assert result["teacher_hidden_cache"]["num_tokens"] == 5
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [[0, 1], [2, 3, 4]]
