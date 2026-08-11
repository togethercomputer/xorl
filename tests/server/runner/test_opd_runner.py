import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests._helpers.opd import make_teacher_files
from xorl.data.constants import IGNORE_INDEX
from xorl.distillation import MooncakeHiddenStore, TeacherActivationCache
from xorl.ops.loss.opd_loss import OPDLossMetrics
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _make_opd_runner() -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.local_rank = 0
    runner.world_size = 1
    runner.train_config = {}
    runner.lm_head_fp32 = True
    runner.pp_enabled = False
    runner.model_fwd_context = nullcontext()
    runner._opd_head_manager = None
    runner._opd_head_config = None
    runner._opd_hidden_cache = None
    runner._opd_hidden_config = None
    runner._opd_layer_cache = None
    runner._opd_layer_config = None
    runner._teacher_hidden_cache_store = None
    runner._teacher_hidden_cache_store_config = None
    runner._opd_lm_head_debug_written = set()
    runner._opd_vocab_parallel_loss_debug_written = set()
    runner._opd_packed_sample_debug_written = set()
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
def _assert_opd_metric_aggregation_policy(mock_parallel_state):
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
            "_opd_debug_local_token_kl": torch.tensor([1.0, 2.0]),
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
    ModelRunner._finalize_loss_metrics(accumulated, result, "opd_loss")

    assert result["opd_kl"] == pytest.approx((0.5 * 4 + 0.2 * 2) / 6)
    assert result["opd_weighted_kl"] == pytest.approx((0.6 * 4 + 0.3 * 2) / 6)
    assert result["opd_num_teachers:max"] == 2
    assert result["opd_profile_kl_compute_ms"] == pytest.approx(30.0)
    assert not any(key.startswith("_opd_debug_") for key in result)
    assert not any(key.startswith("is_opd") for key in result)

    _assert_loss_metric_extrema_ignore_zero_valid_microbatches()
    _assert_opd_metrics_reduce_over_loss_group()
    _assert_opd_metric_seeding_aligns_empty_rank_keys()


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_loss_metric_extrema_ignore_zero_valid_microbatches(mock_parallel_state, _mock_get_device_type):
    mock_parallel_state.return_value = Mock(dp_enabled=False, loss_parallel_enabled=False)
    accumulated = {}
    metric_ops = {
        "kl_k3_debug_max": "max",
        "kl_k3_debug_logratio_min": "min",
    }

    ModelRunner._accumulate_loss_metrics(
        accumulated,
        {
            "valid_tokens": 0,
            "kl_k3_debug_max": 1.0,
            "kl_k3_debug_logratio_min": -1.0,
        },
        "importance_sampling",
        metric_ops,
    )
    ModelRunner._accumulate_loss_metrics(
        accumulated,
        {
            "valid_tokens": 4,
            "kl_k3_debug_max": 0.25,
            "kl_k3_debug_logratio_min": -0.1,
        },
        "importance_sampling",
        metric_ops,
    )

    result = {}
    ModelRunner._finalize_loss_metrics(accumulated, result, "importance_sampling")

    assert result["is_kl_k3_debug_max"] == pytest.approx(0.25)
    assert result["is_kl_k3_debug_logratio_min"] == pytest.approx(-0.1)


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_opd_metrics_reduce_over_loss_group(mock_parallel_state, _mock_get_device_type):
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
def _assert_opd_metric_seeding_aligns_empty_rank_keys(_mock_get_device_type):
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


def _assert_opd_packed_cache_and_weight_shaping_policy():
    hidden = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    labels = torch.tensor([[10, 11, -100, 12, 13, -100]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0]])

    rows, cache_indices = ModelRunner._split_hidden_cache_rows(hidden, labels, position_ids)

    assert cache_indices == [[0, 1], [2, 3]]
    assert torch.equal(torch.cat(rows, dim=0), hidden[0, [0, 1, 3, 4]])

    _assert_oprd_last_k_weights_respects_packed_position_resets()
    _assert_teacher_hidden_cache_splits_packed_batch_and_drops_padding()


def _assert_opd_layer_cache_fetcher_streams_layer_slices():
    class FakeLayerCache:
        def __init__(self) -> None:
            self.requested_indices = None
            self.requested_slices = []

        def shape(self, teacher_id):
            return (5, 16, 3)

        def get_layer_slice(self, teacher_id, indices, layer_start, layer_end, *, device, dtype, cache_device=False):
            self.requested_indices = indices.detach().cpu().clone()
            self.requested_slices.append((layer_start, layer_end))
            rows = int(indices.numel())
            layers = int(layer_end - layer_start)
            base = torch.arange(rows * layers * 3, dtype=dtype, device=device).reshape(rows, layers, 3)
            return base + 100 * layer_start

    runner = object.__new__(ModelRunner)
    layer_cache = FakeLayerCache()
    cache_indices = torch.tensor([[10, 11, 12], [13, 14, 15]])
    teacher_mask = torch.tensor([[False, True, False], [True, False, True]])

    fetcher, num_layers = runner._get_opd_teacher_layer_fetcher(
        {"teacher_cache_indices": cache_indices},
        teacher_id=0,
        layer_cache=layer_cache,
        dtype=torch.float32,
        teacher_mask=teacher_mask,
        valid_mask=teacher_mask,
        cache_device=True,
    )

    first = fetcher(0, 2)
    second = fetcher(2, 5)

    assert num_layers == 5
    torch.testing.assert_close(layer_cache.requested_indices, torch.tensor([11, 13, 15]))
    assert layer_cache.requested_slices == [(0, 2), (2, 5)]
    assert first.shape == (3, 2, 3)
    assert second.shape == (3, 3, 3)


def _assert_oprd_last_k_weights_respects_packed_position_resets():
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


def test_opd_runner_policy(tmp_path, monkeypatch):
    loss_root = tmp_path / "loss"
    loss_root.mkdir()
    _assert_opd_loss_execution_policy(tmp_path=loss_root)
    with monkeypatch.context() as cache_patch:
        _assert_teacher_hidden_cache_distributed_assembly_policy(monkeypatch=cache_patch)
    debug_root = tmp_path / "debug"
    debug_root.mkdir()
    _assert_opd_debug_artifact_policy(debug_root)


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
def _assert_opd_loss_execution_policy(_mock_get_device_type, tmp_path):
    _assert_opd_packed_cache_and_weight_shaping_policy()
    _assert_opd_layer_cache_fetcher_streams_layer_slices()
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
    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    hidden_caches = {tid: store.put_hidden(f"opd/test/teacher/{tid}/hidden", t) for tid, t in teacher_caches.items()}

    runner = _make_opd_runner()
    runner._opd_hidden_cache = TeacherActivationCache(hidden_caches, mooncake_store=store, enable_async=False)
    runner._opd_hidden_config = repr(hidden_caches)
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
        "teacher_hidden_caches": hidden_caches,
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

    _assert_opd_runner_runs_lm_head_anchor_for_fsdp(tmp_path=tmp_path / "anchor")
    _assert_opd_metric_aggregation_policy()


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
def _assert_opd_runner_runs_lm_head_anchor_for_fsdp(_mock_get_device_type, tmp_path):
    tmp_path.mkdir(parents=True)
    torch.manual_seed(11)
    seq_len = 3
    hidden_size = 4
    teacher_hidden_size = 5
    vocab_size = 9
    teacher_heads = {"0": torch.randn(vocab_size, teacher_hidden_size) / teacher_hidden_size**0.5}
    teacher_caches = {"0": torch.randn(seq_len, teacher_hidden_size) / teacher_hidden_size**0.5}
    teacher_files = make_teacher_files(tmp_path, teacher_heads, teacher_caches)
    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    hidden_caches = {tid: store.put_hidden(f"opd/test/teacher/{tid}/hidden", t) for tid, t in teacher_caches.items()}

    runner = _make_opd_runner()
    runner._opd_hidden_cache = TeacherActivationCache(hidden_caches, mooncake_store=store, enable_async=False)
    runner._opd_hidden_config = repr(hidden_caches)
    hidden_states = (torch.randn(1, seq_len, hidden_size) / hidden_size**0.5).requires_grad_(True)
    lm_head = _RecordingLmHead(hidden_size, vocab_size)
    micro_batch = {
        "labels": torch.tensor([[1, 2, 3]]),
        "teacher_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "teacher_cache_indices": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
    }
    params = {
        "teacher_heads": teacher_files.heads,
        "teacher_hidden_caches": hidden_caches,
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


def _assert_teacher_hidden_cache_splits_packed_batch_and_drops_padding():
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


def _assert_teacher_hidden_cache_contributor_policy():
    runner = _make_opd_runner()
    runner.rank = 3
    runner.world_size = 8

    assert (
        runner._teacher_hidden_cache_contributor_key(SimpleNamespace(cp_enabled=True, cp_rank=1, ep_enabled=False))
        is None
    )
    runner.rank = 4
    assert (
        runner._teacher_hidden_cache_contributor_key(
            SimpleNamespace(cp_enabled=True, cp_rank=0, cp_size=2, ep_enabled=False, pp_size=1)
        )
        == 2
    )
    runner.rank = 3
    # Distinct-slice dispatch (default): every EP rank contributes its own slice,
    # keyed by the stage-local rank to mirror batch_slice_rank_and_size.
    assert (
        runner._teacher_hidden_cache_contributor_key(
            SimpleNamespace(cp_enabled=False, ep_enabled=True, ep_rank=1, pp_size=1)
        )
        == 3
    )


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.gather_outputs")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_teacher_hidden_cache_distributed_assembly_policy(
    mock_parallel_state,
    mock_gather,
    _mock_device,
    monkeypatch,
):
    _assert_teacher_hidden_cache_contributor_policy()

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

    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    runner._get_teacher_hidden_cache_store = lambda params: store
    result = runner._forward_teacher_hidden_cache(
        [
            {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "_original_position_ids": torch.arange(8, dtype=torch.long).view(1, 8),
            }
        ],
        {},
    )

    mock_gather.assert_called_once()
    assert mock_gather.call_args.kwargs["group"] == "full-sp-group"
    assert mock_gather.call_args.kwargs["unpad_dim_size"] == 8
    assert result["teacher_hidden_cache"]["backend"] == "mooncake"
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [list(range(8))]

    _assert_teacher_hidden_cache_trims_with_gathered_sp_labels()
    _assert_teacher_hidden_cache_writer_gathers_all_batch_ranks()
    _assert_teacher_hidden_cache_mooncake_integration_policy()


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_teacher_hidden_cache_trims_with_gathered_sp_labels(mock_parallel_state, _mock_get_device_type):
    runner = _make_opd_runner()
    runner.model = _InputIdHiddenModel()
    mock_parallel_state.return_value = Mock(
        cp_enabled=True,
        cp_rank=0,
        cp_size=2,
        tp_size=1,
        pp_size=1,
        ulysses_group=object(),
        ep_enabled=False,
        dp_rank=0,
        dp_size=1,
    )

    full_hidden = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    full_labels = torch.tensor([[-100, -100, -100, -100, 9, -100]])

    def fake_gather_outputs(tensor, **_kwargs):
        return full_hidden if torch.is_floating_point(tensor) else full_labels

    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    runner._get_teacher_hidden_cache_store = lambda params: store
    with patch("xorl.server.runner.model_runner.gather_outputs", side_effect=fake_gather_outputs):
        result = runner._forward_teacher_hidden_cache(
            [
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "labels": torch.tensor([[-100, -100, -100]]),
                    "_original_position_ids": torch.arange(6, dtype=torch.long).unsqueeze(0),
                }
            ],
            {"teacher_hidden_cache_dtype": "float32"},
        )

    saved = store.get_hidden_from_metadata(result["teacher_hidden_cache"])
    # This branch filters cache rows to valid-label positions only (one row per
    # labeled token), rather than trimming to the last-valid prefix.
    assert torch.equal(saved, full_hidden[0, 4:5])
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [[0]]


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_teacher_hidden_cache_writer_gathers_all_batch_ranks(
    mock_parallel_state,
    _mock_get_device_type,
):
    runner = _make_opd_runner()
    runner.world_size = 2
    runner.model = _InputIdHiddenModel()
    mock_parallel_state.return_value = Mock(
        cp_enabled=False, ep_enabled=False, dp_rank=0, dp_size=2, tp_size=1, pp_size=1
    )

    remote_chunk = torch.tensor([[10.0], [11.0], [12.0]])

    def fake_gather_object(payload, object_gather_list, dst):
        assert dst == 0
        object_gather_list[:] = [payload, {"rank": 1, "slice_key": 1, "chunks": [remote_chunk]}]

    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    runner._get_teacher_hidden_cache_store = lambda params: store
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
            {"teacher_hidden_cache_dtype": "float32"},
        )

    saved = store.get_hidden_from_metadata(result["teacher_hidden_cache"])
    assert torch.equal(saved, torch.tensor([[1.0], [2.0], [10.0], [11.0], [12.0]]))
    assert result["teacher_hidden_cache"]["num_tokens"] == 5
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [[0, 1], [2, 3, 4]]


class _FakeMooncakeClient:
    """In-memory stand-in for the Mooncake distributed store."""

    def __init__(self):
        self.objects = {}

    def put(self, key, value):
        self.objects[key] = bytes(value)
        return 0

    def get(self, key):
        return self.objects.get(key, b"")

    def is_exist(self, key):
        return 1 if key in self.objects else 0

    def remove(self, key):
        self.objects.pop(key, None)
        return 0


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_teacher_hidden_cache_mooncake_integration_policy(mock_parallel_state, _mock_device):
    runner = _make_opd_runner()
    runner.model = _InputIdHiddenModel()
    mock_parallel_state.return_value = Mock(
        cp_enabled=False, ep_enabled=False, dp_rank=0, dp_size=1, tp_size=1, pp_size=1
    )

    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    # The forward path fetches the store via _get_teacher_hidden_cache_store;
    # shadow it with the injected fake so no live Mooncake master is needed.
    runner._get_teacher_hidden_cache_store = lambda params: store

    result = runner._forward_teacher_hidden_cache(
        [
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[1, 2, -100]]),
            }
        ],
        {"teacher_hidden_cache_dtype": "float32"},
    )

    meta = result["teacher_hidden_cache"]
    # No safetensors path is required and none is produced.
    assert "path" not in meta
    assert meta["backend"] == "mooncake"
    assert meta["key"]
    assert meta["tensor_key"] == "hidden_states"
    assert meta["tensor_shapes"] == {"hidden_states": [2, 1]}
    assert meta["tensor_dtypes"] == {"hidden_states": "float32"}
    assert meta["num_tokens"] == 2
    assert meta["cache_indices_by_sample"] == [[0, 1]]
    assert result["teacher_prefill_tokens"] == 2
    assert "teacher_hidden_cache_write_s" in result

    # The stored bytes round-trip to the real valid-label hidden rows.
    fetched = store.get_hidden_from_metadata(meta)
    assert torch.equal(fetched, torch.tensor([[1.0], [2.0]]))

    _assert_teacher_hidden_cache_mooncake_roundtrips_through_activation_cache()


@patch("xorl.server.runner.model_runner.get_device_type", return_value="cpu")
@patch("xorl.server.runner.model_runner.get_parallel_state")
def _assert_teacher_hidden_cache_mooncake_roundtrips_through_activation_cache(mock_parallel_state, _mock_device):
    """Producer metadata -> TeacherActivationCache.get returns the right rows."""
    runner = _make_opd_runner()
    runner.model = _InputIdHiddenModel()
    mock_parallel_state.return_value = Mock(
        cp_enabled=False, ep_enabled=False, dp_rank=0, dp_size=1, tp_size=1, pp_size=1
    )

    store = MooncakeHiddenStore(client=_FakeMooncakeClient(), get_retry_max_wait_s=0.0)
    runner._get_teacher_hidden_cache_store = lambda params: store

    result = runner._forward_teacher_hidden_cache(
        [{"input_ids": torch.tensor([[4, 5, 6, 7]]), "labels": torch.tensor([[4, 5, 6, 7]])}],
        {"teacher_hidden_cache_dtype": "float32"},
    )
    meta = result["teacher_hidden_cache"]

    # Consumer indexes the cache by teacher_cache_indices via the Mooncake backend.
    tac = TeacherActivationCache({"0": meta}, mooncake_store=store, enable_async=False)
    try:
        out = tac.get("0", torch.tensor([[0, 2, 3]]), device="cpu", dtype=torch.float32)
        assert torch.equal(out[0], torch.tensor([[4.0], [6.0], [7.0]]))
    finally:
        tac.close()


def _assert_opd_debug_artifact_policy(tmp_path):
    runner = _make_opd_runner()
    debug_path = tmp_path / "vp_loss.jsonl"

    runner._maybe_write_opd_vocab_parallel_loss_debug(
        {"opd_debug_vocab_parallel_loss_path": str(debug_path)},
        teacher_id=7,
        metrics={
            "opd_kl": 0.25,
            "opd_weighted_kl": 0.5,
            "opd_vocab_parallel_group_tokens": 11,
            "opd_vocab_parallel_kl_sum": 2.75,
            "opd_vocab_parallel_weighted_kl_sum": 5.5,
            "opd_oprd_loss": 0.125,
            "opd_hidden_match_loss": 0.125,
            "opd_teacher_weight_mean": 1.0,
        },
        loss=torch.tensor(1.25, requires_grad=True),
        local_valid_tokens=3,
        vocab_start=128,
        vocab_end=256,
    )

    rows = [json.loads(line) for line in (tmp_path / "vp_loss.rank00000.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["teacher_id"] == 7
    assert row["vocab_start"] == 128
    assert row["vocab_end"] == 256
    assert row["local_valid_tokens"] == 3
    assert row["group_valid_tokens"] == 11
    assert row["opd_vocab_parallel_kl_sum"] == pytest.approx(2.75)
    assert row["opd_vocab_parallel_weighted_kl_sum"] == pytest.approx(5.5)
    assert row["model_runner_local_kl_contribution"] == pytest.approx(0.75)
    assert row["model_runner_local_weighted_kl_contribution"] == pytest.approx(1.5)
    assert row["loss_detached"] == pytest.approx(1.25)
    assert row["loss_requires_grad"] is True

    _assert_packed_sample_debug_writer_records_teacher_segments(tmp_path / "packed")


def _assert_packed_sample_debug_writer_records_teacher_segments(tmp_path):
    tmp_path.mkdir(parents=True)
    runner = _make_opd_runner()
    debug_path = tmp_path / "packed_samples.jsonl"
    micro_batch = {
        "batch_id": 13,
        "num_samples": 3,
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[IGNORE_INDEX, 5, 6, IGNORE_INDEX, 7, IGNORE_INDEX, IGNORE_INDEX, 8]]),
        "target_tokens": torch.tensor([[99, 5, 6, 98, 7, 97, 96, 8]], dtype=torch.long),
        "teacher_ids": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.long),
        "teacher_cache_indices": torch.tensor([[10, 11, 12, 20, 21, 30, 31, 32]], dtype=torch.long),
        "teacher_weights": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]),
        "input_ids": torch.tensor([[100, 101, 102, 200, 201, 300, 301, 302]], dtype=torch.long),
        "opd_region_ids": torch.tensor([[0, 1, 1, 2, 2, 3, 3, 3]], dtype=torch.long),
        "opd_sample_ok": torch.tensor([[1, 1, 1, 0, 0, 1, 1, 1]], dtype=torch.long),
        "packed_row_source_batch_ids": [10, 20, 30],
        "packed_row_source_request_ids": ["req-a", "req-b", "req-c"],
        "packed_row_source_num_samples": [1, 1, 1],
        "packed_row_source_token_spans": [[0, 3], [3, 5], [5, 8]],
        "packed_row_source_group_size": 3,
    }
    component_tensor = torch.tensor(
        [
            [
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
                [20.0, 21.0],
                [22.0, 23.0],
                [30.0, 31.0],
                [32.0, 33.0],
                [34.0, 35.0],
            ]
        ]
    )
    student_component_debug = [
        {"layer": 2, "name": "mlp", "order": 10, "tensor": component_tensor + 100.0},
        {"layer": 2, "name": "layer_input", "order": 0, "tensor": component_tensor},
    ]

    for teacher_id in (0, 1):
        student_hidden_debug = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) if teacher_id == 0 else torch.tensor([[7.0, 8.0]])
        )
        runner._maybe_write_opd_packed_sample_debug(
            {"opd_debug_packed_sample_path": str(debug_path)},
            teacher_id=teacher_id,
            micro_batch=micro_batch,
            metrics={
                "opd_kl": 0.25,
                "opd_weighted_kl": 0.5,
                "opd_vocab_parallel_group_tokens": 4,
                "_opd_debug_local_token_kl": torch.tensor([0.1, 0.2, 0.3]) if teacher_id == 0 else torch.tensor([0.4]),
                "_opd_debug_local_weighted_token_kl": torch.tensor([1.0, 2.0, 3.0])
                if teacher_id == 0
                else torch.tensor([4.0]),
                "_opd_debug_local_token_weight": torch.tensor([2.0, 3.0, 5.0])
                if teacher_id == 0
                else torch.tensor([8.0]),
            },
            loss=torch.tensor(1.5),
            local_valid_tokens=4,
            vocab_start=128,
            vocab_end=256,
            backend="vocab_parallel",
            student_hidden_debug=student_hidden_debug,
            student_component_debug=student_component_debug,
        )

    rows = [json.loads(line) for line in (tmp_path / "packed_samples.rank00000.jsonl").read_text().splitlines()]
    assert [(row["teacher_id"], row["segment_index"]) for row in rows] == [(0, 0), (0, 1), (1, 2)]

    first = rows[0]
    assert first["segment_start"] == 0
    assert first["segment_end"] == 3
    assert first["segment_teacher_valid_tokens"] == 2
    assert first["teacher_cache_valid"] == {
        "count": 2,
        "first": 11,
        "last": 12,
        "max": 12,
        "min": 11,
        "sum": 23,
    }
    assert first["labels_teacher_valid"]["sum"] == 11
    assert first["input_ids_segment"]["sum"] == 303
    assert first["position_ids_segment_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([0, 1, 2], dtype=torch.long),
        dtype=torch.long,
    )
    assert first["teacher_cache_valid_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([11, 12], dtype=torch.long),
        dtype=torch.long,
    )
    assert first["input_ids_segment_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([100, 101, 102], dtype=torch.long),
        dtype=torch.long,
    )
    assert first["labels_segment_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([IGNORE_INDEX, 5, 6], dtype=torch.long),
        dtype=torch.long,
    )
    assert first["labels_teacher_valid_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([5, 6], dtype=torch.long),
        dtype=torch.long,
    )
    assert first["target_tokens_segment_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([99, 5, 6], dtype=torch.long),
        dtype=torch.long,
    )
    assert first["teacher_weights_valid_sha256"] == ModelRunner._opd_debug_tensor_sha256(
        torch.tensor([2.0, 3.0], dtype=torch.float32),
        dtype=torch.float32,
    )
    assert first["teacher_weight_valid_sum"] == pytest.approx(5.0)
    assert first["segment_debug_token_offset_start"] == 0
    assert first["segment_debug_token_offset_end"] == 2
    assert first["segment_debug_token_count"] == 2
    assert first["segment_debug_missing_tokens"] == 0
    assert first["packed_row_source_batch_ids"] == [10, 20, 30]
    assert first["packed_row_source_request_ids"] == ["req-a", "req-b", "req-c"]
    assert first["packed_row_source_num_samples"] == [1, 1, 1]
    assert first["packed_row_source_token_spans"] == [[0, 3], [3, 5], [5, 8]]
    assert first["packed_row_source_group_size"] == 3
    assert first["segment_source_overlaps"] == [
        {
            "source_batch_id": 10,
            "source_index": 0,
            "source_num_samples": 1,
            "source_request_id": "req-a",
            "source_token_end": 3,
            "source_token_start": 0,
            "overlap_end": 3,
            "overlap_start": 0,
            "overlap_tokens": 3,
        }
    ]
    assert first["segment_kl_sum_local"] == pytest.approx(0.3)
    assert first["segment_kl_mean_local"] == pytest.approx(0.15)
    assert first["segment_weighted_kl_sum_local"] == pytest.approx(3.0)
    assert first["segment_token_weight_sum_local"] == pytest.approx(5.0)
    assert first["segment_student_hidden"]["shape"] == [2, 2]
    assert first["segment_student_hidden"]["count"] == 4
    assert first["segment_student_hidden"]["sample_sum"] == pytest.approx(10.0)
    assert first["segment_student_hidden"]["sample_sq_sum"] == pytest.approx(30.0)
    assert first["segment_student_hidden"]["sample_first_values"] == [1.0, 2.0, 3.0, 4.0]
    assert first["segment_student_hidden"]["sample_sha256"]
    assert [component["name"] for component in first["segment_student_components"]] == ["layer_input", "mlp"]
    assert first["segment_student_components"][0]["summary"]["shape"] == [2, 2]
    assert first["segment_student_components"][0]["summary"]["sample_sum"] == pytest.approx(54.0)
    assert first["segment_student_components"][0]["summary"]["sample_first_values"] == [12.0, 13.0, 14.0, 15.0]
    assert first["segment_student_components"][1]["summary"]["sample_sum"] == pytest.approx(454.0)
    assert first["opd_region_id_counts"] == {"1": 2}
    assert first["opd_sample_ok_counts"] == {"1": 3}

    second = rows[1]
    assert second["segment_debug_token_offset_start"] == 2
    assert second["segment_debug_token_offset_end"] == 3
    assert second["segment_source_overlaps"][0]["source_batch_id"] == 20
    assert second["segment_source_overlaps"][0]["overlap_tokens"] == 2
    assert second["segment_kl_sum_local"] == pytest.approx(0.3)
    assert second["segment_student_hidden"]["shape"] == [1, 2]
    assert second["segment_student_hidden"]["sample_sum"] == pytest.approx(11.0)

    third = rows[2]
    assert third["teacher_id"] == 1
    assert third["segment_start"] == 5
    assert third["segment_teacher_valid_tokens"] == 1
    assert third["segment_source_overlaps"][0]["source_batch_id"] == 30
    assert third["segment_source_overlaps"][0]["overlap_tokens"] == 3
    assert third["teacher_cache_valid"]["first"] == 32
    assert third["teacher_weight_valid_mean"] == pytest.approx(8.0)
    assert third["segment_debug_token_offset_start"] == 0
    assert third["segment_debug_token_offset_end"] == 1
    assert third["segment_kl_sum_local"] == pytest.approx(0.4)
    assert third["segment_weighted_kl_sum_local"] == pytest.approx(4.0)
    assert third["segment_token_weight_sum_local"] == pytest.approx(8.0)
    assert third["segment_student_hidden"]["shape"] == [1, 2]
    assert third["segment_student_hidden"]["sample_sum"] == pytest.approx(15.0)
