import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.moe.routing_replay import RoutingReplay
from xorl.server.runner.utils import routing_replay_handler as rrh


def _routing(start: int, length: int) -> list[list[list[int]]]:
    return [[[start + i, start + i + 1000]] for i in range(length)]


def _handler() -> rrh.RoutingReplayHandler:
    return rrh.RoutingReplayHandler(torch.nn.Module())


def test_handler_discovers_replayable_moe_in_later_virtual_pipeline_part():
    first_part = torch.nn.Module()
    first_part.config = SimpleNamespace(num_experts_per_tok=2)
    later_part = torch.nn.Module()
    later_part.layer = torch.nn.Module()
    later_part.layer.mlp = MoEBlock(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        intermediate_size=16,
        moe_implementation="eager",
    )
    later_part.layer.mlp._routing_replay = RoutingReplay()

    handler = rrh.RoutingReplayHandler([first_part, later_part])

    assert handler.model is first_part
    assert handler.models == (first_part, later_part)
    assert handler.get_moe_blocks() == [later_part.layer.mlp]


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


def test_routing_weight_builder_preserves_float_values_and_padding(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False))
    micro_batches = [{"input_ids": torch.zeros(1, 3, dtype=torch.long), "num_samples": 1}]
    weights = [[[[0.25, 0.75]], [[0.10, 0.90]]]]

    per_mb = _handler()._build_per_mb_routing(
        micro_batches,
        weights,
        num_layers_in_data=1,
        topk=2,
        tensor_dtype=torch.float32,
    )

    assert per_mb[0].dtype == torch.float32
    assert per_mb[0].shape == (3, 1, 2)
    torch.testing.assert_close(
        per_mb[0][:, 0, :],
        torch.tensor([[0.25, 0.75], [0.10, 0.90], [0.50, 0.50]], dtype=torch.float32),
    )


def test_aligned_sp_routing_maps_source_rows_to_local_physical_rows(monkeypatch):
    routed_experts = [_routing(10, 3), _routing(20, 2)]
    routed_weights = [
        [[[0.10, 0.90]], [[0.20, 0.80]], [[0.30, 0.70]]],
        [[[0.40, 0.60]], [[0.60, 0.40]]],
    ]
    expected_by_rank = {
        0: ([10, 11, 12], [[0.10, 0.90], [0.20, 0.80], [0.30, 0.70]]),
        1: ([20, 21], [[0.40, 0.60], [0.60, 0.40]]),
    }

    for cp_rank, (expected_experts, expected_weights) in expected_by_rank.items():
        monkeypatch.setattr(
            rrh,
            "get_parallel_state",
            lambda cp_rank=cp_rank: SimpleNamespace(
                cp_enabled=True,
                cp_size=2,
                cp_rank=cp_rank,
                ringattn_size=1,
            ),
        )
        if cp_rank == 0:
            logical_rows = [0, 1, 2] + [-1] * 61
            request_ids = [0, 0, 0] + [-1] * 61
            request_positions = [0, 1, 2] + [0] * 61
        else:
            logical_rows = [3, 4] + [-1] * 62
            request_ids = [1, 1] + [-1] * 62
            request_positions = [0, 1] + [0] * 62
        live_mask = [logical_row >= 0 for logical_row in logical_rows]
        micro_batches = [
            {
                "input_ids": torch.zeros(1, 64, dtype=torch.long),
                "position_ids": torch.zeros(1, 128, dtype=torch.long),
                "num_samples": 2,
                "_cp_logical_row_indices": torch.tensor([logical_rows]),
                "_cp_live_mask": torch.tensor([live_mask]),
                "_cp_request_ids": torch.tensor([request_ids]),
                "_cp_request_positions": torch.tensor([request_positions]),
            }
        ]

        per_mb_experts = _handler()._build_per_mb_routing(
            micro_batches,
            routed_experts,
            num_layers_in_data=1,
            topk=2,
        )
        per_mb_weights = _handler()._build_per_mb_routing(
            micro_batches,
            routed_weights,
            num_layers_in_data=1,
            topk=2,
            tensor_dtype=torch.float32,
        )

        live_rows = len(expected_experts)
        assert per_mb_experts[0].shape == (64, 1, 2)
        assert per_mb_experts[0][:live_rows, 0, 0].tolist() == expected_experts
        assert per_mb_experts[0][live_rows:, 0].tolist() == [[0, 1]] * (64 - live_rows)
        torch.testing.assert_close(
            per_mb_weights[0][:live_rows, 0],
            torch.tensor(expected_weights, dtype=torch.float32),
        )
        torch.testing.assert_close(
            per_mb_weights[0][live_rows:, 0],
            torch.tensor([[0.5, 0.5]] * (64 - live_rows), dtype=torch.float32),
        )


def test_aligned_sp_routing_rejects_request_identity_mismatch(monkeypatch):
    monkeypatch.setattr(
        rrh,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, cp_size=2, cp_rank=0, ringattn_size=1),
    )
    micro_batches = [
        {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "position_ids": torch.zeros(1, 4, dtype=torch.long),
            "num_samples": 2,
            "_cp_logical_row_indices": torch.tensor([[0, -1]]),
            "_cp_live_mask": torch.tensor([[True, False]]),
            "_cp_request_ids": torch.tensor([[1, -1]]),
            "_cp_request_positions": torch.tensor([[0, 0]]),
        }
    ]

    with pytest.raises(ValueError, match="request .* maps to"):
        _handler()._build_per_mb_routing(
            micro_batches,
            [_routing(10, 1), _routing(20, 1)],
            num_layers_in_data=1,
            topk=2,
        )


# --- SGLang routed_experts decode contract (Lever 2 shared-selection K3) ---
# These lock the exact wire format the K3 harness now relies on: SGLang exports
# return_routed_experts as base64 int32 with shape (tokens, layers, top_k), and
# RoutingReplayHandler must decode it without corruption before record(). A
# silent decode mismatch here would make a shared-selection K3 replay garbage
# expert selections, so this is validated on CPU before spending GPU capacity.


def test_decode_routed_experts_sglang_dict_base64_int32_format():
    handler = _handler()
    arr = np.arange(3 * 2 * 8, dtype=np.int32).reshape(3, 2, 8)
    item = {"data": base64.b64encode(arr.tobytes()).decode("ascii"), "shape": [3, 2, 8]}

    decoded = handler.decode_routed_experts_item(item, num_moe_layers=2)

    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, arr)


def test_decode_routed_experts_sglang_dict_honors_rows_view():
    handler = _handler()
    arr = np.arange(4 * 2 * 2, dtype=np.int32).reshape(4, 2, 2)
    item = {"data": base64.b64encode(arr.tobytes()).decode("ascii"), "shape": [4, 2, 2], "rows": 3}

    decoded = handler.decode_routed_experts_item(item, num_moe_layers=2)

    np.testing.assert_array_equal(decoded, arr[:3])


def test_decode_routed_experts_base64_string_infers_shape_from_model_topk():
    model = SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=8))
    handler = rrh.RoutingReplayHandler(model)
    assert handler._model_topk == 8
    arr = np.arange(4 * 2 * 8, dtype=np.int32).reshape(4, 2, 8)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")

    decoded = handler.decode_routed_experts_item(b64, num_moe_layers=2)

    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, arr)


def test_decode_routed_experts_nested_list_passthrough():
    handler = _handler()
    nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    assert handler.decode_routed_experts_item(nested, num_moe_layers=2) == nested


def test_decode_routed_experts_accepts_decoded_numpy_array():
    handler = _handler()
    arr = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)

    decoded = handler.decode_routed_experts_item(arr, num_moe_layers=2)

    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, arr)


def test_decode_routed_expert_logits_accepts_decoded_tensor():
    handler = _handler()
    tensor = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)

    decoded = handler.decode_routed_expert_logits_item(tensor, num_moe_layers=2)

    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, tensor.numpy())


def test_decode_routed_experts_topk_extracted_from_nested_text_config():
    # Qwen3.6 nests num_experts_per_tok under text_config; the handler must read
    # it there so shape inference does not mispick top-k from row 0.
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_experts_per_tok=8)))

    assert rrh.RoutingReplayHandler(model)._model_topk == 8


def test_routing_weight_builder_accepts_decoded_numpy_arrays(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_size=2, cp_rank=1))
    micro_batches = [
        {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "position_ids": torch.arange(6, dtype=torch.long).view(1, 6),
            "num_samples": 1,
        }
    ]
    weights = [np.arange(6 * 1 * 2, dtype=np.float32).reshape(6, 1, 2)]

    per_mb = _handler()._build_per_mb_routing(
        micro_batches,
        weights,
        num_layers_in_data=1,
        topk=2,
        tensor_dtype=torch.float32,
    )

    assert per_mb[0].dtype == torch.float32
    assert per_mb[0].shape == (3, 1, 2)
    torch.testing.assert_close(per_mb[0][:, 0, :], torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]))


def test_replay_staging_is_one_layer_major_backing_buffer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source = torch.arange(4 * 3 * 2, dtype=torch.int64).reshape(4, 3, 2)

    staged = _handler()._stage_layer_major_replay_tensor(source)

    assert staged.shape == (3, 4, 2)
    assert staged.is_contiguous()
    assert torch.equal(staged[1], source[:, 1, :])
    assert staged[0].untyped_storage().data_ptr() == staged[2].untyped_storage().data_ptr()


def test_fill_routing_replay_populates_device_ready_layer_views(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    handler = _handler()
    blocks = [SimpleNamespace(_routing_replay=rrh.RoutingReplay()) for _ in range(2)]
    monkeypatch.setattr(handler, "get_moe_blocks", lambda: blocks)
    ids = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)
    weights = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2) / 10

    try:
        assert handler.fill_routing_replay(
            [{"input_ids": torch.zeros(1, 3, dtype=torch.long), "num_samples": 1}],
            [ids],
            [weights],
        )
        for layer, block in enumerate(blocks):
            replay = block._routing_replay
            assert len(replay.top_indices_list) == len(replay.top_weights_list) == 1
            torch.testing.assert_close(replay.top_indices_list[0], torch.from_numpy(ids[:, layer]).long())
            torch.testing.assert_close(replay.top_weights_list[0], torch.from_numpy(weights[:, layer]))
        assert (
            blocks[0]._routing_replay.top_indices_list[0].untyped_storage().data_ptr()
            == blocks[1]._routing_replay.top_indices_list[0].untyped_storage().data_ptr()
        )
        assert handler.last_setup_metrics["r3_replay_setup_s"] >= 0.0
        assert handler.last_setup_metrics["r3_replay_staged_bytes"] == float(ids.size * 8 + weights.nbytes)
    finally:
        rrh.RoutingReplay.clear_all()
