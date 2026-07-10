"""Tests for PP bubble profiling.

Pure-math helpers (interval merge, analytic bubble, P2P estimate, patching) run on CPU;
the CUDA-event machinery has a single-GPU, single-stage schedule test marked gpu.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from xorl.distributed.pp_profiling import (
    PPBubbleProfiler,
    analytic_bubble_fraction,
    estimate_p2p_bytes_per_step,
    merge_busy_intervals,
)


class TestMergeBusyIntervals:
    @pytest.mark.cpu
    def test_empty_and_degenerate(self):
        assert merge_busy_intervals([]) == 0.0
        assert merge_busy_intervals([(1.0, 1.0)]) == 0.0  # zero-length dropped
        assert merge_busy_intervals([(2.0, 1.0)]) == 0.0  # inverted dropped

    @pytest.mark.cpu
    def test_single_and_disjoint(self):
        assert merge_busy_intervals([(0.0, 1.0)]) == pytest.approx(1.0)
        assert merge_busy_intervals([(0.0, 1.0), (2.0, 3.5)]) == pytest.approx(2.5)

    @pytest.mark.cpu
    def test_overlapping_counted_once(self):
        # [0,2] and [1,3] overlap on [1,2] -> union length 3
        assert merge_busy_intervals([(0.0, 2.0), (1.0, 3.0)]) == pytest.approx(3.0)
        # Fully contained interval adds nothing
        assert merge_busy_intervals([(0.0, 4.0), (1.0, 2.0)]) == pytest.approx(4.0)
        # Touching endpoints merge without a gap
        assert merge_busy_intervals([(0.0, 1.0), (1.0, 2.0)]) == pytest.approx(2.0)

    @pytest.mark.cpu
    def test_unsorted_input(self):
        assert merge_busy_intervals([(5.0, 6.0), (0.0, 1.0), (0.5, 2.0)]) == pytest.approx(3.0)


class TestAnalyticBubbleFraction:
    @pytest.mark.cpu
    def test_1f1b_gpipe_textbook_values(self):
        # (p-1)/(m+p-1): 27% at p=4,m=8; 30% at p=8,m=16 (GOALS.md bubble arithmetic)
        assert analytic_bubble_fraction("1F1B", 4, 1, 8) == pytest.approx(3 / 11)
        assert analytic_bubble_fraction("GPipe", 4, 1, 8) == pytest.approx(3 / 11)
        assert analytic_bubble_fraction("1F1B", 8, 1, 16) == pytest.approx(7 / 23)
        assert analytic_bubble_fraction("1f1b", 2, 1, 8) == pytest.approx(1 / 9)

    @pytest.mark.cpu
    def test_interleaved_divides_bubble(self):
        # (p-1)/(v*m+p-1)
        assert analytic_bubble_fraction("Interleaved1F1B", 4, 2, 8) == pytest.approx(3 / 19)
        assert analytic_bubble_fraction("Interleaved1F1B", 2, 2, 16) == pytest.approx(1 / 33)
        # v=1 degenerates to the 1F1B value
        assert analytic_bubble_fraction("Interleaved1F1B", 4, 1, 8) == pytest.approx(3 / 11)

    @pytest.mark.cpu
    def test_zero_bubble_schedules(self):
        assert analytic_bubble_fraction("InterleavedZeroBubble", 4, 2, 8) == 0.0
        assert analytic_bubble_fraction("ZBVZeroBubble", 2, 2, 8) == 0.0
        assert analytic_bubble_fraction("DualPipeV", 4, 2, 16) == 0.0

    @pytest.mark.cpu
    def test_pp1_has_no_bubble(self):
        assert analytic_bubble_fraction("1F1B", 1, 1, 4) == 0.0

    @pytest.mark.cpu
    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            analytic_bubble_fraction("NotASchedule", 2, 1, 8)
        with pytest.raises(ValueError):
            analytic_bubble_fraction("1F1B", 2, 2, 8)  # single-stage schedule with v=2
        with pytest.raises(ValueError):
            analytic_bubble_fraction("1F1B", 0, 1, 8)


class _FakeRecvInfo:
    def __init__(self, buffer):
        self.buffer = buffer


class _FakePlaceholder:
    pass


class _FakeIOStage:
    """Minimal stand-in exposing the PipelineStage attributes the P2P estimator reads."""

    def __init__(
        self,
        stage_index,
        num_stages,
        group_rank,
        stage_to_rank,
        in_shape=None,
        out_shape=None,
        has_backward=True,
        dtype=torch.bfloat16,
    ):
        self.stage_index = stage_index
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        self.group_rank = group_rank
        self.stage_index_to_group_rank = stage_to_rank
        self.has_backward = has_backward
        if self.is_first:
            self.args_recv_info = {0: (_FakePlaceholder(),)}
        else:
            self.args_recv_info = {0: (_FakeRecvInfo(torch.zeros(in_shape, dtype=dtype)),)}
        self._outputs_meta = (torch.zeros(out_shape, dtype=dtype),) if out_shape else ()
        self.act_send_info = {0: [stage_index + 1]} if not self.is_last else {0: []}

    def get_outputs_meta(self):
        return self._outputs_meta


class TestEstimateP2PBytes:
    SHAPE = (2, 128, 64)  # bf16 -> 32768 bytes
    NBYTES = 2 * 128 * 64 * 2

    @pytest.mark.cpu
    def test_middle_stage_counts_all_four_flows(self):
        stage_to_rank = {0: 0, 1: 1, 2: 2, 3: 3}
        stage = _FakeIOStage(1, 4, 1, stage_to_rank, in_shape=self.SHAPE, out_shape=self.SHAPE)
        # fwd recv + fwd send + grad recv + grad send, x 8 microbatches
        assert estimate_p2p_bytes_per_step([stage], 8) == 4 * self.NBYTES * 8

    @pytest.mark.cpu
    def test_first_and_last_stage_skip_edge_flows(self):
        stage_to_rank = {0: 0, 1: 1}
        first = _FakeIOStage(0, 2, 0, stage_to_rank, out_shape=self.SHAPE)
        last = _FakeIOStage(1, 2, 1, stage_to_rank, in_shape=self.SHAPE, out_shape=self.SHAPE)
        # first: fwd send + grad recv; last: fwd recv + grad send (no send: act_send_info empty)
        assert estimate_p2p_bytes_per_step([first], 4) == 2 * self.NBYTES * 4
        assert estimate_p2p_bytes_per_step([last], 4) == 2 * self.NBYTES * 4

    @pytest.mark.cpu
    def test_forward_only_skips_grad_flows(self):
        stage_to_rank = {0: 0, 1: 1, 2: 2, 3: 3}
        stage = _FakeIOStage(1, 4, 1, stage_to_rank, in_shape=self.SHAPE, out_shape=self.SHAPE, has_backward=False)
        assert estimate_p2p_bytes_per_step([stage], 8) == 2 * self.NBYTES * 8

    @pytest.mark.cpu
    def test_same_rank_adjacency_excluded(self):
        # ZBV at pp=2: rank 1 owns stages [1, 2]; the 1->2 handoff is rank-local.
        stage_to_rank = {0: 0, 1: 1, 2: 1, 3: 0}
        s1 = _FakeIOStage(1, 4, 1, stage_to_rank, in_shape=self.SHAPE, out_shape=self.SHAPE)
        s2 = _FakeIOStage(2, 4, 1, stage_to_rank, in_shape=self.SHAPE, out_shape=self.SHAPE)
        # s1: fwd recv from 0 + grad send to 0 (send to 2 local); s2 mirrored.
        assert estimate_p2p_bytes_per_step([s1, s2], 2) == 4 * self.NBYTES * 2

    @pytest.mark.cpu
    def test_unpopulated_stage_returns_none(self):
        class _Broken:
            stage_index = 1
            is_first = False
            is_last = False
            group_rank = 1
            stage_index_to_group_rank = {0: 0, 1: 1, 2: 0}
            has_backward = True
            args_recv_info = {}

            def get_outputs_meta(self):
                raise RuntimeError("not configured")

        assert estimate_p2p_bytes_per_step([_Broken()], 4) is None


class _FakeComputeStage:
    def __init__(self, stage_index=0):
        self.stage_index = stage_index
        self.calls = []

    def forward_one_chunk(self, *args, **kwargs):
        self.calls.append("fwd")
        return "out"

    def backward_one_chunk(self, *args, **kwargs):
        self.calls.append("bwd")

    def backward_weight_one_chunk(self, *args, **kwargs):
        self.calls.append("bwd_w")


class _FakeSchedule:
    def __init__(self, stages):
        self._stages = stages
        self._n_microbatches = 4


class TestProfilerPatching:
    @pytest.mark.cpu
    def test_patch_passthrough_and_restore(self):
        stages = [_FakeComputeStage(0), _FakeComputeStage(2)]
        profiler = PPBubbleProfiler(_FakeSchedule(stages))
        for stage in stages:
            for name in ("forward_one_chunk", "backward_one_chunk", "backward_weight_one_chunk"):
                assert name in stage.__dict__  # instance-patched
        # Outside step_scope the wrappers pass straight through (no CUDA needed)
        assert stages[0].forward_one_chunk(0, ()) == "out"
        stages[0].backward_one_chunk(0, loss=None, full_backward=True)
        stages[1].backward_weight_one_chunk(0)
        assert stages[0].calls == ["fwd", "bwd"] and stages[1].calls == ["bwd_w"]
        profiler.close()
        for stage in stages:
            assert "forward_one_chunk" not in stage.__dict__
        assert stages[0].forward_one_chunk(0, ()) == "out"  # class method restored

    @pytest.mark.cpu
    def test_double_patch_rejected(self):
        schedule = _FakeSchedule([_FakeComputeStage(0)])
        profiler = PPBubbleProfiler(schedule)
        with pytest.raises(RuntimeError, match="already instance-patched"):
            PPBubbleProfiler(schedule)
        profiler.close()

    @pytest.mark.cpu
    def test_single_stage_schedule_attr(self):
        class _SingleSchedule:
            def __init__(self, stage):
                self._stage = stage
                self._n_microbatches = 2

        stage = _FakeComputeStage(0)
        profiler = PPBubbleProfiler(_SingleSchedule(stage))
        assert profiler.stages == [stage]
        profiler.close()

    @pytest.mark.cpu
    def test_report_without_steps_raises(self):
        profiler = PPBubbleProfiler(_FakeSchedule([_FakeComputeStage(0)]))
        with pytest.raises(RuntimeError, match="No instrumented steps"):
            profiler.report()
        profiler.close()


@pytest.mark.gpu
class TestProfilerCudaEvents:
    def test_single_stage_gpipe_step(self, tmp_path):
        """Trivial 1-GPU, 1-stage GPipe schedule: events, bubble, memory, p2p=0."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")

        n_microbatches = 4
        device = torch.device("cuda", 0)
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path}/pg_store", rank=0, world_size=1, device_id=device
        )
        try:
            model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16)).to(device)
            stage = PipelineStage(model, 0, 1, device)
            schedule = ScheduleGPipe(
                stage,
                n_microbatches=n_microbatches,
                loss_fn=lambda out, tgt: ((out - tgt) ** 2).sum(),
                scale_grads=False,
            )
            x = torch.randn(8, 16, device=device)
            y = torch.randn(8, 16, device=device)

            schedule.step(x, target=y, losses=[])  # warmup (shape inference)
            profiler = PPBubbleProfiler(schedule)
            with profiler.step_scope():
                schedule.step(x, target=y, losses=[])
            stats = profiler.report()

            assert stats["actions"]["fwd"]["count"] == n_microbatches
            assert stats["actions"]["bwd_full"]["count"] == n_microbatches
            assert stats["actions"]["bwd_input"]["count"] == 0
            assert stats["actions"]["bwd_weight"]["count"] == 0
            assert stats["busy_time_s"] > 0
            assert stats["busy_time_s"] <= stats["step_time_s"] + 1e-4  # actions lie inside the step window
            assert stats["bubble_fraction"] is not None and 0.0 <= stats["bubble_fraction"] < 1.0
            assert stats["peak_memory_bytes"] >= 0
            assert stats["p2p_bytes"] == 0  # single stage: no sends or recvs
            profiler.close()
        finally:
            dist.destroy_process_group()
