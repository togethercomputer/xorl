from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import load_file

from tests._helpers.opd import make_teacher_files
from xorl.data.constants import IGNORE_INDEX
from xorl.ops.loss.opd_loss import OPDLossMetrics
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


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
def test_opd_metric_seeding_aligns_empty_rank_keys(_mock_get_device_type):
    """A rank with only 0-valid-token micro-batches returns
    ``OPDLossMetrics(valid_tokens=0).to_dict()``, which carries none of the
    per-micro-batch ``opd_profile_*_ms`` (sum_max) keys, so its
    _finalize_loss_metrics reduce groups differ in size from a populated rank's
    and the cross-rank all_reduce deadlocks. Seeding the canonical key set first
    must make every rank's reduce groups identical. Regression for the empty-rank
    collective-desync hang. (OPDLossMetrics always emits opd_num_teachers, so the
    desync vector here is the omitted profile group, not the max key.)
    """
    populated = {}
    ModelRunner._accumulate_loss_metrics(
        populated,
        {
            "valid_tokens": 4,
            "opd_kl": 0.5,
            "opd_weighted_kl": 0.6,
            "opd_teacher_weight_mean": 1.0,
            "opd_num_teachers": 2,
            "opd_profile_kl_compute_ms": 10.0,
        },
        "opd_loss",
    )

    # Exactly what _compute_opd_micro_batch_loss returns on a 0-valid rank.
    empty = {}
    ModelRunner._accumulate_loss_metrics(empty, OPDLossMetrics(valid_tokens=0).to_dict(), "opd_loss")

    # Bug precondition: the empty rank carries none of the sum_max profile keys
    # the populated rank has, so the two ranks would issue different-sized
    # collectives in the sum_max group.
    assert "opd_profile_kl_compute_ms" in populated
    assert "opd_profile_kl_compute_ms" not in empty
    assert set(empty) != set(populated)

    # include_profile_metrics is uniform across ranks (it is read from params).
    for acc in (empty, populated):
        ModelRunner._ensure_opd_loss_metric_accumulators(acc, include_profile_metrics=True)

    def reduce_groups(acc):
        groups: dict[str, set[str]] = {}
        for key, entry in acc.items():
            groups.setdefault(entry["op"], set()).add(key)
        return groups

    # Every reduce group now carries the same keys on both ranks -> no size
    # mismatch in _finalize_loss_metrics.
    assert reduce_groups(empty) == reduce_groups(populated)
    assert "opd_num_teachers:max" in empty
    assert "opd_profile_kl_compute_ms" in empty


def test_teacher_hidden_cache_split_skips_padding_segments():
    hidden = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    labels = torch.tensor([[10, 11, -100, 12, 13, -100]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0]])

    rows, cache_indices = ModelRunner._split_hidden_cache_rows(hidden, labels, position_ids)

    assert cache_indices == [[0, 1], [2, 3]]
    assert torch.equal(torch.cat(rows, dim=0), hidden[0, [0, 1, 3, 4]])


def test_teacher_hidden_cache_split_filters_unpacked_masked_targets():
    hidden = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    labels = torch.tensor([[-100, -100, 12, -100, 14, -100]])

    rows, cache_indices = ModelRunner._split_hidden_cache_rows(hidden, labels)

    assert cache_indices == [[0, 1]]
    assert torch.equal(rows[0], hidden[0, [2, 4]])


def test_oprd_last_k_weights_respects_packed_position_resets():
    labels = torch.tensor([[10, 11, 12, 20, IGNORE_INDEX, 21, 22]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    base_weights = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])

    packed_weights = ModelRunner._opd_oprd_last_k_weights(
        labels,
        base_weights=base_weights,
        last_k=2,
        position_ids=position_ids,
    )
    row_tail_weights = ModelRunner._opd_oprd_last_k_weights(
        labels,
        base_weights=base_weights,
        last_k=2,
    )

    torch.testing.assert_close(packed_weights, torch.tensor([[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 7.0]]))
    torch.testing.assert_close(row_tail_weights, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0]]))


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


def test_teacher_hidden_cache_contributor_skips_duplicate_cp_ranks_and_keys_ep_ranks_by_slice():
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
    # Distinct-slice dispatch (default): every EP rank contributes its own slice,
    # keyed by the stage-local rank to mirror batch_slice_rank_and_size.
    assert (
        runner._teacher_hidden_cache_contributor_key(
            SimpleNamespace(cp_enabled=False, ep_enabled=True, ep_rank=1, pp_size=1)
        )
        == 3
    )


def test_teacher_hidden_cache_contributor_legacy_flag_skips_duplicate_ep_ranks(monkeypatch):
    monkeypatch.setenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "1")
    runner = _make_opd_runner()
    runner.rank = 3
    runner.world_size = 8

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
            SimpleNamespace(
                cp_enabled=False,
                ep_enabled=True,
                ep_rank=0,
                ep_size=2,
                dp_shard_in_ep_size=4,
                ep_fsdp_device_mesh=FakeEpMesh(),
            )
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
    # This branch filters cache rows to valid-label positions only (one row per
    # labeled token), rather than trimming to the last-valid prefix.
    assert torch.equal(saved, full_hidden[0, 4:5])
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [[0]]


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
