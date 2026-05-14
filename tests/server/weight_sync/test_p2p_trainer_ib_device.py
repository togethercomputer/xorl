import pytest

from xorl.server.weight_sync import handler as handler_mod
from xorl.server.weight_sync.handler import WeightSyncHandler, _select_p2p_ib_device


def test_selects_global_rank_mapping_when_map_covers_world(monkeypatch):
    monkeypatch.setenv("P2P_TRAINER_IB_DEVICES_PER_RANK", "mlx5_0;mlx5_1;mlx5_2;mlx5_3")
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert _select_p2p_ib_device(rank=2, world_size=4) == "mlx5_2"


def test_selects_local_rank_mapping_when_map_is_per_node(monkeypatch):
    monkeypatch.setenv("P2P_TRAINER_IB_DEVICES_PER_RANK", "mlx5_0;mlx5_1;mlx5_2;mlx5_3")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert _select_p2p_ib_device(rank=9, world_size=16) == "mlx5_1"


def test_falls_back_to_single_trainer_ib_device(monkeypatch):
    monkeypatch.delenv("P2P_TRAINER_IB_DEVICES_PER_RANK", raising=False)
    monkeypatch.delenv("P2P_TRAINER_GPU_TO_IB_DEVICE_MAP", raising=False)
    monkeypatch.setenv("P2P_TRAINER_IB_DEVICE", "mlx5_6")

    assert _select_p2p_ib_device(rank=3, world_size=8) == "mlx5_6"


def test_selects_physical_gpu_mapping_from_visible_gpu_indices(monkeypatch):
    monkeypatch.delenv("P2P_TRAINER_IB_DEVICES_PER_RANK", raising=False)
    monkeypatch.delenv("P2P_TRAINER_IB_DEVICE", raising=False)
    monkeypatch.setenv("P2P_TRAINER_GPU_TO_IB_DEVICE_MAP", "0=mlx5_2,1=mlx5_3,3=mlx5_5")
    monkeypatch.setenv("P2P_TRAINER_VISIBLE_GPU_INDICES", "3,1,0")

    monkeypatch.setenv("LOCAL_RANK", "0")
    assert _select_p2p_ib_device(rank=0, world_size=8) == "mlx5_5"

    monkeypatch.setenv("LOCAL_RANK", "1")
    assert _select_p2p_ib_device(rank=1, world_size=8) == "mlx5_3"


def test_selects_physical_gpu_mapping_from_numeric_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("P2P_TRAINER_IB_DEVICES_PER_RANK", raising=False)
    monkeypatch.delenv("P2P_TRAINER_IB_DEVICE", raising=False)
    monkeypatch.delenv("P2P_TRAINER_VISIBLE_GPU_INDICES", raising=False)
    monkeypatch.delenv("SELECTED_GPU_INDICES", raising=False)
    monkeypatch.setenv("P2P_TRAINER_GPU_TO_IB_DEVICE_MAP", "0:mlx5_2;6:mlx5_6")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert _select_p2p_ib_device(rank=0, world_size=8) == "mlx5_6"


def test_empty_mapping_entry_uses_autodiscovery(monkeypatch):
    monkeypatch.setenv("P2P_TRAINER_IB_DEVICES_PER_RANK", "mlx5_0;;mlx5_2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("P2P_TRAINER_IB_DEVICE", raising=False)

    assert _select_p2p_ib_device(rank=1, world_size=3) is None


def test_sync_abort_marker_roundtrip(tmp_path):
    class Trainer:
        train_config = {"output_dir": str(tmp_path)}

    handler = WeightSyncHandler(rank=3, world_size=8, trainer=Trainer())
    path = handler._sync_abort_path("weight_sync_group", "iter-1")

    handler._mark_sync_abort(path, RuntimeError("transfer failed"))

    with pytest.raises(RuntimeError, match="rank=3: transfer failed"):
        handler._raise_if_sync_aborted(path)

    handler._clear_sync_abort(path)
    handler._raise_if_sync_aborted(path)


def test_p2p_rank_summary_includes_handler_transfer_totals(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")

    class Backend:
        def stats_summary(self):
            return {"total_bytes": 321, "num_buckets": 4, "main_thread_s": 0.2}

    handler = WeightSyncHandler(rank=6, world_size=8, trainer=None)

    summary = handler._build_p2p_rank_summary(
        Backend(),
        is_sender=True,
        transfer_wall_s=0.5,
        total_bytes=123,
        num_parameters=7,
        num_buckets=3,
        ib_device="mlx5_2",
        phase_s={"direct_ep_s": 0.4},
    )

    assert summary["rank"] == 6
    assert summary["local_rank"] == 2
    assert summary["total_bytes"] == 123
    assert summary["num_parameters"] == 7
    assert summary["num_buckets"] == 3
    assert summary["has_transfers"] is True


def test_aggregate_p2p_transfer_totals_sums_all_rank_counters():
    total_bytes, num_parameters, num_buckets = WeightSyncHandler._aggregate_p2p_transfer_totals(
        [
            {"rank": 0, "total_bytes": 100, "num_parameters": 4, "num_buckets": 1},
            {"rank": 1, "total_bytes": 300, "num_parameters": 6, "num_buckets": 2},
            {"rank": 2, "total_bytes": 0, "num_parameters": 0, "num_buckets": 0},
        ],
        total_bytes=100,
        num_parameters=4,
        num_buckets=1,
    )

    assert total_bytes == 400
    assert num_parameters == 10
    assert num_buckets == 3


def test_p2p_transfer_status_gather_reports_peer_failure(monkeypatch):
    handler = WeightSyncHandler(rank=1, world_size=2, trainer=None)

    monkeypatch.setattr(handler_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(handler_mod.dist, "is_initialized", lambda: True)

    def fake_all_gather_object(gathered, local_status):
        gathered[0] = {"rank": 0, "ok": False, "error": "RuntimeError: transfer failed"}
        gathered[1] = local_status

    monkeypatch.setattr(handler_mod.dist, "all_gather_object", fake_all_gather_object)

    statuses = handler._gather_p2p_transfer_statuses(None)

    assert statuses == [
        {"rank": 0, "ok": False, "error": "RuntimeError: transfer failed"},
        {"rank": 1, "ok": True},
    ]
