from types import SimpleNamespace

import torch

from xorl.server.runner.utils import routing_replay_handler as rrh


def _routing(start: int, length: int) -> list[list[list[int]]]:
    return [[[start + i, start + i + 1000]] for i in range(length)]


def _handler() -> rrh.RoutingReplayHandler:
    return rrh.RoutingReplayHandler(torch.nn.Module())


def test_sp_routing_uses_actual_position_ids_length_without_128_padding(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_size=4, cp_rank=1))
    micro_batches = [
        {
            "input_ids": torch.zeros(1, 93, dtype=torch.long),
            "position_ids": torch.arange(372, dtype=torch.long).view(1, 372),
            "num_samples": 1,
        }
    ]

    per_mb = _handler()._build_per_mb_routing(micro_batches, [_routing(0, 372)], num_layers_in_data=1, topk=2)

    assert len(per_mb) == 1
    assert per_mb[0].shape == (93, 1, 2)
    assert per_mb[0][0, 0, 0].item() == 93
    assert per_mb[0][-1, 0, 0].item() == 185


def test_sp_routing_pads_to_actual_position_ids_length(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_size=4, cp_rank=3))
    micro_batches = [
        {
            "input_ids": torch.zeros(1, 96, dtype=torch.long),
            "position_ids": torch.arange(384, dtype=torch.long).view(1, 384),
            "num_samples": 1,
        }
    ]

    per_mb = _handler()._build_per_mb_routing(micro_batches, [_routing(0, 372)], num_layers_in_data=1, topk=2)

    assert per_mb[0].shape == (96, 1, 2)
    assert per_mb[0][0, 0, 0].item() == 288
    assert per_mb[0][83, 0, 0].item() == 371
    assert per_mb[0][84, 0].tolist() == [0, 1]


def test_sp_routing_slices_unpacked_rows_independently(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_size=4, cp_rank=1))
    micro_batches = [
        {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "position_ids": torch.arange(24, dtype=torch.long).view(2, 12),
            "num_samples": 2,
        }
    ]

    per_mb = _handler()._build_per_mb_routing(
        micro_batches,
        [_routing(0, 12), _routing(100, 12)],
        num_layers_in_data=1,
        topk=2,
    )

    assert per_mb[0].shape == (6, 1, 2)
    assert per_mb[0][:, 0, 0].tolist() == [3, 4, 5, 103, 104, 105]


def test_ringattn_routing_uses_zigzag_layout_before_sp_slice(monkeypatch):
    expected_by_rank = {
        0: [0, 1],
        1: [6, 7],
        2: [2, 3],
        3: [4, 5],
    }
    for cp_rank, expected in expected_by_rank.items():
        monkeypatch.setattr(
            rrh,
            "get_parallel_state",
            lambda cp_rank=cp_rank: SimpleNamespace(
                cp_enabled=True,
                cp_size=4,
                cp_rank=cp_rank,
                ringattn_size=2,
            ),
        )
        micro_batches = [
            {
                "input_ids": torch.zeros(1, 2, dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5]], dtype=torch.long),
                "_original_position_ids": torch.arange(8, dtype=torch.long).view(1, 8),
                "num_samples": 1,
            }
        ]

        per_mb = _handler()._build_per_mb_routing(micro_batches, [_routing(0, 8)], num_layers_in_data=1, topk=2)

        assert per_mb[0].shape == (2, 1, 2)
        assert per_mb[0][:, 0, 0].tolist() == expected


def test_ringattn_routing_zigzag_respects_packed_document_boundaries(monkeypatch):
    monkeypatch.setattr(
        rrh,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, cp_size=2, cp_rank=0, ringattn_size=2),
    )
    micro_batches = [
        {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "position_ids": torch.tensor([[0, 3, 0, 3, 1, 2, 1, 2]], dtype=torch.long),
            "_original_position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long),
            "num_samples": 2,
        }
    ]

    per_mb = _handler()._build_per_mb_routing(
        micro_batches,
        [_routing(0, 4), _routing(4, 4)],
        num_layers_in_data=1,
        topk=2,
    )

    assert per_mb[0].shape == (4, 1, 2)
    assert per_mb[0][:, 0, 0].tolist() == [0, 3, 4, 7]


def test_routing_truncates_excess_to_micro_batch_size(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False))
    micro_batches = [{"input_ids": torch.zeros(1, 3, dtype=torch.long), "num_samples": 1}]

    per_mb = _handler()._build_per_mb_routing(micro_batches, [_routing(0, 4)], num_layers_in_data=1, topk=2)

    assert per_mb[0].shape == (3, 1, 2)
    assert per_mb[0][:, 0, 0].tolist() == [0, 1, 2]
