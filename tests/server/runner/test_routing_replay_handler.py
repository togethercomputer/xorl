import base64
from types import SimpleNamespace

import numpy as np
import torch

from xorl.server.runner.utils import routing_replay_handler as rrh


def _routing(start: int, length: int) -> list[list[list[int]]]:
    return [[[start + i, start + i + 1000]] for i in range(length)]


def _handler() -> rrh.RoutingReplayHandler:
    return rrh.RoutingReplayHandler(torch.nn.Module())


def _assert_sp_routing_uses_actual_position_length_with_and_without_padding(monkeypatch):
    unpadded_batches = [
        {
            "input_ids": torch.zeros(1, 93, dtype=torch.long),
            "position_ids": torch.arange(372, dtype=torch.long).view(1, 372),
            "num_samples": 1,
        }
    ]
    padded_batches = [
        {
            "input_ids": torch.zeros(1, 96, dtype=torch.long),
            "position_ids": torch.arange(384, dtype=torch.long).view(1, 384),
            "num_samples": 1,
        }
    ]

    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_size=4, cp_rank=1))
    unpadded = _handler()._build_per_mb_routing(unpadded_batches, [_routing(0, 372)], 1, 2)
    assert len(unpadded) == 1
    assert unpadded[0].shape == (93, 1, 2)
    assert unpadded[0][[0, -1], 0, 0].tolist() == [93, 185]

    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, cp_size=4, cp_rank=3))
    padded = _handler()._build_per_mb_routing(padded_batches, [_routing(0, 372)], 1, 2)
    assert padded[0].shape == (96, 1, 2)
    assert padded[0][[0, 83], 0, 0].tolist() == [288, 371]
    assert padded[0][84, 0].tolist() == [0, 1]


def _assert_sp_routing_slices_unpacked_rows_independently(monkeypatch):
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


def _assert_ringattn_routing_uses_zigzag_layout_before_sp_slice(monkeypatch):
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


def _assert_ringattn_routing_zigzag_respects_packed_document_boundaries(monkeypatch):
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


def _assert_routing_truncates_excess_to_micro_batch_size(monkeypatch):
    monkeypatch.setattr(rrh, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False))
    micro_batches = [{"input_ids": torch.zeros(1, 3, dtype=torch.long), "num_samples": 1}]

    per_mb = _handler()._build_per_mb_routing(micro_batches, [_routing(0, 4)], num_layers_in_data=1, topk=2)

    assert per_mb[0].shape == (3, 1, 2)
    assert per_mb[0][:, 0, 0].tolist() == [0, 1, 2]


def _assert_routing_weight_builder_preserves_float_values_and_padding(monkeypatch):
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


# --- SGLang routed_experts decode contract (Lever 2 shared-selection K3) ---
# These lock the exact wire format the K3 harness now relies on: SGLang exports
# return_routed_experts as base64 int32 with shape (tokens, layers, top_k), and
# RoutingReplayHandler must decode it without corruption before record(). A
# silent decode mismatch here would make a shared-selection K3 replay garbage
# expert selections, so this is validated on CPU before spending GPU capacity.


def _assert_decode_routed_experts_sglang_dict_base64_int32_format():
    handler = _handler()
    arr = np.arange(3 * 2 * 8, dtype=np.int32).reshape(3, 2, 8)
    item = {"data": base64.b64encode(arr.tobytes()).decode("ascii"), "shape": [3, 2, 8]}

    decoded = handler.decode_routed_experts_item(item, num_moe_layers=2)

    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, arr)


def _assert_decode_routed_experts_base64_string_infers_shape_from_model_topk():
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_experts_per_tok=8)))
    handler = rrh.RoutingReplayHandler(model)
    assert handler._model_topk == 8
    arr = np.arange(4 * 2 * 8, dtype=np.int32).reshape(4, 2, 8)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")

    decoded = handler.decode_routed_experts_item(b64, num_moe_layers=2)

    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, arr)


def _assert_decode_routed_replay_accepts_materialized_python_numpy_and_tensor_values():
    handler = _handler()
    nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert handler.decode_routed_experts_item(nested, num_moe_layers=2) == nested

    arr = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)
    decoded = handler.decode_routed_experts_item(arr, num_moe_layers=2)
    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, arr)

    tensor = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
    decoded = handler.decode_routed_expert_logits_item(tensor, num_moe_layers=2)
    assert isinstance(decoded, np.ndarray)
    np.testing.assert_array_equal(decoded, tensor.numpy())


def _assert_routing_weight_builder_accepts_decoded_numpy_arrays(monkeypatch):
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


def test_routing_replay_context_parallel_layout_contract(monkeypatch):
    with monkeypatch.context() as sequence_patch:
        _assert_sp_routing_uses_actual_position_length_with_and_without_padding(sequence_patch)
        _assert_sp_routing_slices_unpacked_rows_independently(sequence_patch)
        _assert_routing_truncates_excess_to_micro_batch_size(sequence_patch)

    with monkeypatch.context() as ring_patch:
        _assert_ringattn_routing_uses_zigzag_layout_before_sp_slice(ring_patch)
        _assert_ringattn_routing_zigzag_respects_packed_document_boundaries(ring_patch)


def test_routing_replay_wire_decode_and_weight_tensor_contract(monkeypatch):
    _assert_decode_routed_experts_sglang_dict_base64_int32_format()
    _assert_decode_routed_experts_base64_string_infers_shape_from_model_topk()
    _assert_decode_routed_replay_accepts_materialized_python_numpy_and_tensor_values()
    _assert_routing_weight_builder_preserves_float_values_and_padding(monkeypatch)
    _assert_routing_weight_builder_accepts_decoded_numpy_arrays(monkeypatch)
