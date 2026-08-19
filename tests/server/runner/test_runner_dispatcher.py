import json
from types import SimpleNamespace

import pytest
import torch

import xorl.server.runner.runner_dispatcher as runner_dispatcher_module
from xorl.server.orchestrator.packing import SequentialPacker
from xorl.server.runner.runner_dispatcher import RunnerDispatcher
from xorl.server.runner.utils.batch_utils import batch_packed_rows
from xorl.server.side_payloads import MooncakeSidePayloadStore, put_r3_mooncake_payload_refs


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeEPMesh:
    def __init__(self, ep_fsdp_rank: int) -> None:
        self.ep_fsdp_rank = ep_fsdp_rank

    def get_local_rank(self, name: str) -> int:
        assert name == "ep_fsdp"
        return self.ep_fsdp_rank


def _dispatcher(rank: int, world_size: int) -> RunnerDispatcher:
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = rank
    dispatcher.world_size = world_size
    return dispatcher


def test_dispatcher_enables_cp_alignment_from_resolved_exact_gdn_contract(monkeypatch):
    captured = {}

    class SequenceCollator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    parallel_state = SimpleNamespace(cp_enabled=True, cp_size=2, cp_rank=0)
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(runner_dispatcher_module, "TextSequenceShardCollator", SequenceCollator)
    monkeypatch.setattr(runner_dispatcher_module, "WeightSyncHandler", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner_dispatcher_module, "AdapterCoordinator", lambda *_args, **_kwargs: object())

    trainer = SimpleNamespace(
        model_config_obj=SimpleNamespace(
            _qwen35_exact_contract=True,
            layer_types=["full_attention", "linear_attention"],
        )
    )
    dispatcher = RunnerDispatcher(trainer=trainer, rank=0, world_size=2)

    assert dispatcher._sequence_shard_collator is not None
    assert captured == {"gdn_exact_cp_align": True}


class FakeMooncakeClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.get_calls: list[str] = []

    def put(self, key: str, value: bytes) -> int:
        self.objects[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        self.get_calls.append(key)
        return self.objects.get(key, b"")

    def is_exist(self, key: str) -> int:
        return 1 if key in self.objects else 0

    def remove(self, key: str, force: bool = False) -> int:
        del force
        self.objects.pop(key, None)
        return 0


def _mooncake_store() -> tuple[MooncakeSidePayloadStore, FakeMooncakeClient]:
    client = FakeMooncakeClient()
    return MooncakeSidePayloadStore(client=client, get_retry_max_wait_s=0.0), client


def _batch(batch_id: int, *, num_samples: int = 1) -> dict:
    return {
        "request_id": "req-test",
        "batch_id": batch_id,
        "input_ids": [[batch_id, batch_id + 1]],
        "labels": [[batch_id + 2, batch_id + 3]],
        "position_ids": [[0, 1]],
        "num_samples": num_samples,
    }


def _span_item(path, values: torch.Tensor) -> dict:
    return {
        "schema": "xorl.r3.spans.v1",
        "rows": values.shape[0],
        "shape": list(values.shape),
        "dtype": "int32",
        "spans": [
            {
                "path": str(path),
                "offset": 0,
                "source_row": 0,
                "rows": values.shape[0],
                "row_nbytes": values.shape[1] * values.shape[2] * 4,
                "source_shape": list(values.shape),
                "dtype": "int32",
            }
        ],
    }


def test_sglang_file_loader_mmaps_only_requested_datum_slice(tmp_path, monkeypatch):
    values = torch.arange(12, dtype=torch.int32).reshape(3, 2, 2)
    selected = tmp_path / "selected.bin"
    selected.write_bytes(values.numpy().tobytes())
    missing = _span_item(tmp_path / "must-not-be-read.bin", values)
    ref = {
        "__xorl_routing_payload_ref__": True,
        "transport": "sglang_files",
        "version": 1,
        "format": "spans",
        "kind": "routed_experts",
        "count": 3,
        "items": [missing, _span_item(selected, values), missing],
    }
    monkeypatch.setenv("XORL_R3_SHARED_ROOTS", str(tmp_path))

    loaded = _dispatcher(0, 1)._load_routing_payload_slice(ref, 1, 1)

    assert len(loaded) == 1
    assert torch.equal(loaded[0], values)


def test_sglang_file_loader_reassembles_prefix_spans_rank_locally(tmp_path, monkeypatch):
    first = torch.arange(16, dtype=torch.int32).reshape(4, 2, 2)
    second = torch.arange(100, 112, dtype=torch.int32).reshape(3, 2, 2)
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(first.numpy().tobytes())
    second_path.write_bytes(second.numpy().tobytes())
    item = _span_item(first_path, first)
    item["rows"] = 5
    item["shape"][0] = 5
    item["spans"][0]["rows"] = 2
    item["spans"].append(_span_item(second_path, second)["spans"][0] | {"rows": 3})
    ref = {
        "__xorl_routing_payload_ref__": True,
        "transport": "sglang_files",
        "version": 1,
        "format": "spans",
        "kind": "routed_experts",
        "count": 1,
        "items": [item],
    }
    monkeypatch.setenv("XORL_R3_SHARED_ROOTS", str(tmp_path))

    loaded = _dispatcher(0, 1)._load_routing_payload_slice(ref, 0, 1)

    assert torch.equal(loaded[0], torch.cat((first[:2], second), dim=0))


def _parallel_state(**overrides):
    return SimpleNamespace(
        cp_size=overrides.get("cp_size", 1),
        pp_enabled=overrides.get("pp_enabled", False),
        pp_size=overrides.get("pp_size", 1),
        ep_enabled=overrides.get("ep_enabled", False),
        ep_size=overrides.get("ep_size", 1),
        dp_shard_in_ep_size=overrides.get("dp_shard_in_ep_size", 1),
        ep_fsdp_device_mesh=overrides.get("ep_fsdp_device_mesh"),
        ringattn_enabled=overrides.get("ringattn_enabled", False),
    )


def test_select_batches_keeps_existing_dp_distribution_without_ep(monkeypatch):
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    batches = [_batch(10), _batch(20), _batch(30), _batch(40)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=2, world_size=4)._select_and_prepare_batches(
        batches,
        routed_experts=["r0", "r1", "r2", "r3"],
        routed_expert_logits=["l0", "l1", "l2", "l3"],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[30, 31]]))
    assert routed_experts == ["r2"]
    assert routed_logits == ["l2"]


def test_select_batches_gives_each_ep_rank_a_distinct_slice(monkeypatch):
    state = _parallel_state(
        ep_enabled=True,
        ep_size=8,
        dp_shard_in_ep_size=1,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=0),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    batches = [_batch(10 * (i + 1)) for i in range(8)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        batches,
        routed_experts=[f"r{i}" for i in range(8)],
        routed_expert_logits=[f"l{i}" for i in range(8)],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[60, 61]]))
    assert routed_experts == ["r5"]
    assert routed_logits == ["l5"]


def test_select_batches_pads_ep_ranks_beyond_real_batches_with_dummies(monkeypatch):
    state = _parallel_state(
        ep_enabled=True,
        ep_size=8,
        dp_shard_in_ep_size=1,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=0),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        [_batch(10, num_samples=2), _batch(20)],
        routed_experts=["r0", "r1", "r2"],
        routed_expert_logits=["l0", "l1", "l2"],
    )

    assert len(my_batches) == 1
    assert my_batches[0]["num_samples"] == 0
    assert torch.all(my_batches[0]["labels"] == -100)
    assert routed_experts == []
    assert routed_logits == []


def test_select_batches_shares_one_slice_across_cp_ranks_under_ep(monkeypatch):
    state = _parallel_state(
        cp_size=2,
        ep_enabled=True,
        ep_size=4,
        dp_shard_in_ep_size=2,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=1),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    batches = [_batch(10 * (i + 1)) for i in range(4)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        batches,
        routed_experts=[f"r{i}" for i in range(4)],
        routed_expert_logits=[f"l{i}" for i in range(4)],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[30, 31]]))
    assert routed_experts == ["r2"]
    assert routed_logits == ["l2"]


def test_select_batches_legacy_flag_broadcasts_one_slice_to_all_ep_ranks(monkeypatch):
    monkeypatch.setenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "1")
    state = _parallel_state(
        ep_enabled=True,
        ep_size=8,
        dp_shard_in_ep_size=1,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=0),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        [_batch(10, num_samples=2)],
        routed_experts=["r0", "r1"],
        routed_expert_logits=["l0", "l1"],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[10, 11]]))
    assert my_batches[0]["num_samples"] == 2
    assert routed_experts == ["r0", "r1"]
    assert routed_logits == ["l0", "l1"]


def test_select_batches_legacy_flag_uses_ep_fsdp_rank_for_ep_group_slices(monkeypatch):
    monkeypatch.setenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "1")
    state = _parallel_state(
        ep_enabled=True,
        ep_size=4,
        dp_shard_in_ep_size=2,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=1),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    my_batches, routed_experts, routed_logits = _dispatcher(rank=6, world_size=8)._select_and_prepare_batches(
        [_batch(10), _batch(20)],
        routed_experts=["r0", "r1"],
        routed_expert_logits=["l0", "l1"],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[20, 21]]))
    assert routed_experts == ["r1"]
    assert routed_logits == ["l1"]


def test_microbatch_diagnostic_dump_includes_r3_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("XORL_MICROBATCH_DIAGNOSTIC_TENSORS", "1")
    dispatcher = _dispatcher(rank=3, world_size=8)
    state = _parallel_state(ep_enabled=True, ep_size=8)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[-100, 4, 5]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "logprobs": torch.tensor([[0.0, -0.2, -0.3]]),
        "num_samples": 1,
    }
    routed_experts = [{"data": "experts", "shape": [3, 1, 2]}]
    routed_logits = [{"data": "weights", "shape": [3, 1, 2]}]

    dispatcher._maybe_dump_microbatch_diagnostic(
        [batch],
        loss_fn="importance_sampling",
        loss_fn_params={"diagnostic_microbatch_dump_dir": str(tmp_path), "diagnostic_microbatch_request_id": "req/abc"},
        parallel_state=state,
        with_backward=True,
        model_id="default",
        routed_experts=routed_experts,
        routed_expert_logits=routed_logits,
    )

    summary_path = tmp_path / "microbatch_req_abc_rank00003.json"
    tensor_path = tmp_path / "microbatch_req_abc_rank00003.pt"
    assert summary_path.exists()
    assert tensor_path.exists()
    assert '"routed_experts_count": 1' in summary_path.read_text(encoding="utf-8")

    payload = torch.load(tensor_path, weights_only=False)
    assert payload["routed_experts"] == routed_experts
    assert payload["routed_expert_logits"] == routed_logits
    assert torch.equal(payload["micro_batches"][0]["input_ids"], batch["input_ids"])


def test_select_batches_loads_only_routing_ref_slice(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    root = tmp_path / "payload"
    experts_dir = root / "routed_experts"
    logits_dir = root / "routed_expert_logits"
    experts_dir.mkdir(parents=True)
    logits_dir.mkdir(parents=True)
    expert_metadata = []
    logit_metadata = []
    for idx in range(4):
        experts = torch.tensor([[[idx, idx + 1]]], dtype=torch.int32)
        logits = torch.tensor([[[float(idx), float(idx + 1)]]], dtype=torch.float32)
        expert_data = experts.numpy().tobytes()
        logit_data = logits.numpy().tobytes()
        (experts_dir / f"{idx:06d}.bin").write_bytes(expert_data)
        (logits_dir / f"{idx:06d}.bin").write_bytes(logit_data)
        expert_metadata.append({"shape": [1, 1, 2], "dtype": "int32", "nbytes": len(expert_data)})
        logit_metadata.append({"shape": [1, 1, 2], "dtype": "float32", "nbytes": len(logit_data)})
    manifest = {
        "format": "xorl-r3-raw",
        "version": 2,
        "routed_experts": {"count": 4, "items": expert_metadata},
        "routed_expert_logits": {"count": 4, "items": logit_metadata},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    expert_ref = {
        runner_dispatcher_module.ROUTING_PAYLOAD_REF_KEY: True,
        "transport": "filesystem",
        "version": 2,
        "manifest": str(manifest_path),
        "kind": "routed_experts",
        "count": 4,
    }
    logits_ref = {
        runner_dispatcher_module.ROUTING_PAYLOAD_REF_KEY: True,
        "transport": "filesystem",
        "version": 2,
        "manifest": str(manifest_path),
        "kind": "routed_expert_logits",
        "count": 4,
    }

    batches = [_batch(10), _batch(20), _batch(30), _batch(40)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=2, world_size=4)._select_and_prepare_batches(
        batches,
        routed_experts=expert_ref,
        routed_expert_logits=logits_ref,
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[30, 31]]))
    assert len(routed_experts) == len(routed_logits) == 1
    assert torch.equal(routed_experts[0], torch.tensor([[[2, 3]]], dtype=torch.int32))
    assert torch.equal(routed_logits[0], torch.tensor([[[2.0, 3.0]]], dtype=torch.float32))


def test_select_batches_rejects_legacy_pickle_routing_ref(tmp_path):
    manifest_path = tmp_path / "manifest.pkl"
    manifest_path.write_bytes(b"untrusted pickle bytes")
    ref = {
        runner_dispatcher_module.ROUTING_PAYLOAD_REF_KEY: True,
        "manifest": str(manifest_path),
        "kind": "routed_experts",
        "count": 1,
    }

    with pytest.raises(ValueError, match="Legacy pickle routing payload references are disabled"):
        _dispatcher(rank=0, world_size=1)._load_routing_payload_slice(ref, 0, 1)


def test_select_batches_rejects_symlinked_routing_manifest(tmp_path):
    real_manifest = tmp_path / "real.json"
    real_manifest.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.symlink_to(real_manifest)
    ref = {
        runner_dispatcher_module.ROUTING_PAYLOAD_REF_KEY: True,
        "transport": "filesystem",
        "version": 2,
        "manifest": str(manifest_path),
        "kind": "routed_experts",
        "count": 1,
    }

    with pytest.raises(ValueError, match="Invalid R3 filesystem manifest path"):
        _dispatcher(rank=0, world_size=1)._load_routing_payload_slice(ref, 0, 1)


def test_select_batches_loads_only_mooncake_routing_ref_slice(monkeypatch):
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())
    store, client = _mooncake_store()
    expert_ref, logits_ref, _ = put_r3_mooncake_payload_refs(
        request_id="req",
        routed_experts=[[[[idx, idx + 1]]] for idx in range(4)],
        routed_expert_logits=[[[[float(idx), float(idx + 1)]]] for idx in range(4)],
        store=store,
        chunk_ranges=[(0, 1), (1, 1), (2, 1), (3, 1)],
    )
    assert expert_ref is not None and logits_ref is not None

    dispatcher = _dispatcher(rank=2, world_size=4)
    dispatcher._routing_payload_store = store
    batches = [_batch(10), _batch(20), _batch(30), _batch(40)]
    my_batches, routed_experts, routed_logits = dispatcher._select_and_prepare_batches(
        batches,
        routed_experts=expert_ref,
        routed_expert_logits=logits_ref,
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[30, 31]]))
    assert [item.tolist() for item in routed_experts] == [[[[2, 3]]]]
    assert [item.tolist() for item in routed_logits] == [[[[2.0, 3.0]]]]
    assert client.get_calls == [
        expert_ref["items"]["routed_experts"][2]["key"],
        logits_ref["items"]["routed_expert_logits"][2]["key"],
    ]


def test_select_batches_world_size_one_loads_mooncake_routing_refs(monkeypatch):
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())
    store, client = _mooncake_store()
    expert_ref, logits_ref, _ = put_r3_mooncake_payload_refs(
        request_id="req",
        routed_experts=[[[[0, 1]]], [[[2, 3]]]],
        routed_expert_logits=[[[[0.0, 1.0]]], [[[2.0, 3.0]]]],
        store=store,
    )
    assert expert_ref is not None and logits_ref is not None

    dispatcher = _dispatcher(rank=0, world_size=1)
    dispatcher._routing_payload_store = store
    my_batches, routed_experts, routed_logits = dispatcher._select_and_prepare_batches(
        [_batch(10), _batch(20)],
        routed_experts=expert_ref,
        routed_expert_logits=logits_ref,
    )

    assert len(my_batches) == 2
    assert [item.tolist() for item in routed_experts] == [[[[0, 1]]], [[[2, 3]]]]
    assert [item.tolist() for item in routed_logits] == [[[[0.0, 1.0]]], [[[2.0, 3.0]]]]
    assert client.get_calls == [
        expert_ref["items"]["routed_experts"][0]["key"],
        logits_ref["items"]["routed_expert_logits"][0]["key"],
    ]


# ============================================================================
# End-to-end acceptance: balanced_dp packer + dispatcher => zero dummy batches
# ============================================================================


def test_balanced_dp_packing_yields_zero_dispatcher_dummies(monkeypatch):
    """The redesign's primary acceptance criterion (spec section 5.1):

    Packing with strategy='balanced_dp' at the dispatcher's dp_size produces
    N == k*dp_size rows, so EVERY rank gets the same number of REAL batches and
    no rank runs a dummy (num_samples == 0) filler.
    """
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    dp_size = 8
    world_size = 8  # non-EP, cp=pp=1 -> dispatcher dp_size == world_size
    data = [{"input_ids": list(range(200 + 37 * i)), "target_tokens": list(range(200 + 37 * i))} for i in range(40)]
    packer = SequentialPacker(
        enable_packing=True, log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=dp_size
    )
    raw_batches = packer.pack(data, max_seq_len=8192, request_id="acc")
    assert len(raw_batches) % dp_size == 0

    round_counts = set()
    total_real = 0
    for rank in range(world_size):
        my_batches, _, _ = _dispatcher(rank=rank, world_size=world_size)._select_and_prepare_batches(raw_batches)
        round_counts.add(len(my_batches))
        # No dummy fillers: every batch this rank runs is real.
        assert all(b["num_samples"] > 0 for b in my_batches)
        total_real += sum(b["num_samples"] for b in my_batches)

    # Lockstep: identical round count across ranks (collective invariant).
    assert len(round_counts) == 1
    # Every datum trained exactly once, nothing dropped or duplicated.
    assert total_real == len(data)


def test_sequential_packing_still_pads_dummies_when_rows_below_dp(monkeypatch):
    """Contrast case: legacy sequential under-fills -> dispatcher still pads."""
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    dp_size = 8
    # Few large samples -> sequential makes fewer rows than dp_size -> dummies.
    data = [{"input_ids": list(range(7000)), "target_tokens": list(range(7000))} for _ in range(4)]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1, strategy="sequential")
    raw_batches = packer.pack(data, max_seq_len=8192, request_id="seq")
    assert len(raw_batches) < dp_size

    saw_dummy = False
    for rank in range(dp_size):
        my_batches, _, _ = _dispatcher(rank=rank, world_size=dp_size)._select_and_prepare_batches(raw_batches)
        if any(b["num_samples"] == 0 for b in my_batches):
            saw_dummy = True
    assert saw_dummy


def test_merge_per_token_outputs_restores_logical_slice_order_and_skips_empty_slices():
    payloads = [
        {
            "rank": 2,
            "slice_rank": 2,
            "packed_logprobs": [[2.0]],
            "packed_position_ids": [[0]],
            "per_sample_k3": [0.2],
        },
        {"rank": 3, "slice_rank": 3, "packed_logprobs": [], "packed_position_ids": []},
        {
            "rank": 0,
            "slice_rank": 0,
            "packed_logprobs": [[0.0]],
            "packed_position_ids": [[0]],
            "per_sample_k3": [0.0],
        },
        {
            "rank": 1,
            "slice_rank": 1,
            "packed_logprobs": [[1.0]],
            "packed_position_ids": [[0]],
            "per_sample_k3": [0.1],
        },
    ]

    merged = RunnerDispatcher._merge_per_token_output_payloads(payloads)

    assert merged == {
        "packed_logprobs": [[0.0], [1.0], [2.0]],
        "packed_position_ids": [[0], [0], [0]],
        "per_sample_k3": [0.0, 0.1, 0.2],
    }


def test_merge_per_token_outputs_deduplicates_coherent_cp_replicas():
    replica = {
        "slice_rank": 0,
        "packed_logprobs": [[-1.25, -2.5]],
        "packed_position_ids": [[0, 1]],
    }
    merged = RunnerDispatcher._merge_per_token_output_payloads([{"rank": 0, **replica}, {"rank": 1, **replica}])

    assert merged["packed_logprobs"] == [[-1.25, -2.5]]
    assert merged["packed_position_ids"] == [[0, 1]]


def test_merge_per_token_outputs_rejects_replica_disagreement():
    with pytest.raises(RuntimeError, match="disagree.*slice 0.*ranks \\[0, 1\\]"):
        RunnerDispatcher._merge_per_token_output_payloads(
            [
                {
                    "rank": 0,
                    "slice_rank": 0,
                    "packed_logprobs": [[-1.0]],
                    "packed_position_ids": [[0]],
                },
                {
                    "rank": 1,
                    "slice_rank": 0,
                    "packed_logprobs": [[-1.5]],
                    "packed_position_ids": [[0]],
                },
            ]
        )


def test_completion_rendezvous_trims_payload_then_rank0_merges_afterward(monkeypatch):
    dispatcher = _dispatcher(rank=0, world_size=2)
    dispatcher.cpu_group = object()
    monkeypatch.setattr(dispatcher, "_batch_parallel_rank_and_size", lambda *_args: (0, 2))

    def fake_gather_object(payload, gathered, *, dst, group):
        assert dst == 0
        assert group is dispatcher.cpu_group
        assert payload == {
            "rank": 0,
            "slice_rank": 0,
            "packed_logprobs": [[-1.0]],
            "packed_position_ids": [[0]],
            "per_sample_k3": [0.0, 0.1],
        }
        gathered[:] = [
            payload,
            {
                "rank": 1,
                "slice_rank": 1,
                "packed_logprobs": [[-2.0]],
                "packed_position_ids": [[0]],
                "per_sample_k3": [0.2],
            },
        ]

    monkeypatch.setattr(runner_dispatcher_module.dist, "gather_object", fake_gather_object)
    result = {
        "packed_logprobs": [[-1.0], [999.0]],
        "packed_position_ids": [[0], [999]],
        "per_sample_k3": [0.0, 0.1, 999.0],
    }

    gathered = dispatcher._completion_rendezvous(
        result,
        [_batch(10, num_samples=2), _batch(20, num_samples=0)],
        _parallel_state(),
        is_rank0=True,
    )

    assert result["packed_logprobs"] == [[-1.0], [999.0]]
    dispatcher._merge_completion_payloads(result, gathered)

    assert result == {
        "packed_logprobs": [[-1.0], [-2.0]],
        "packed_position_ids": [[0], [0]],
        "per_sample_k3": [0.0, 0.1, 0.2],
    }


def test_shard_and_slice_batches_slices_routing_weights_with_ids():
    dispatcher = _dispatcher(rank=0, world_size=1)
    dispatcher._validate_batch_shapes = lambda batch, batch_idx=0: True

    batches = [
        {
            "input_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[1, 2]]),
            "_r3_datum_offset": 1,
            "_r3_datum_count": 2,
        }
    ]

    sharded, routed_experts, routed_logits = dispatcher._shard_and_slice_batches(
        batches,
        routed_experts=["r0", "r1", "r2", "r3"],
        routed_expert_logits=["l0", "l1", "l2", "l3"],
        cp_enabled=False,
        parallel_state=_parallel_state(),
    )

    assert sharded == batches
    assert routed_experts == ["r1", "r2"]
    assert routed_logits == ["l1", "l2"]
    assert "_r3_datum_offset" not in sharded[0]
    assert "_r3_datum_count" not in sharded[0]


def test_select_batches_rank_local_rowbatch_preserves_original_dp_slice(monkeypatch):
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    batches = [_batch(10 * (i + 1)) for i in range(6)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=0, world_size=4)._select_and_prepare_batches(
        batches,
        loss_fn_params={"opd_packed_row_batch_size": 2},
    )

    assert routed_experts is None
    assert routed_logits is None
    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[10, 11, 20, 21]]))
    assert torch.equal(my_batches[0]["position_ids"], torch.tensor([[0, 1, 0, 1]]))
    assert torch.equal(my_batches[0]["cu_seq_lens_q"], torch.tensor([0, 2, 4], dtype=torch.int32))
    assert my_batches[0]["num_samples"] == 2
    assert my_batches[0]["packed_row_source_batch_ids"] == [10, 20]
    assert my_batches[0]["packed_row_source_request_ids"] == ["req-test", "req-test"]
    assert my_batches[0]["packed_row_source_num_samples"] == [1, 1]
    assert my_batches[0]["packed_row_source_token_spans"] == [[0, 2], [2, 4]]
    assert my_batches[0]["packed_row_source_group_size"] == 2


def test_batch_packed_rows_records_source_provenance_for_unmerged_rows():
    first = _batch(10)
    second = _batch(20)
    first["teacher_id"] = 0
    second["teacher_id"] = 1

    grouped = batch_packed_rows([first, second], row_batch_size=2)

    assert len(grouped) == 2
    assert grouped[0]["batch_id"] == 0
    assert grouped[0]["packed_row_source_batch_ids"] == [10]
    assert grouped[0]["packed_row_source_request_ids"] == ["req-test"]
    assert grouped[0]["packed_row_source_num_samples"] == [1]
    assert grouped[0]["packed_row_source_token_spans"] == [[0, 2]]
    assert grouped[0]["packed_row_source_group_size"] == 1
    assert grouped[1]["batch_id"] == 1
    assert grouped[1]["packed_row_source_batch_ids"] == [20]


def test_shard_and_slice_batches_keeps_r3_logits_aligned_with_experts():
    dispatcher = _dispatcher(rank=0, world_size=1)
    batch = {
        "input_ids": torch.tensor([[10, 11]]),
        "labels": torch.tensor([[12, 13]]),
        "position_ids": torch.tensor([[0, 1]]),
        "num_samples": 2,
        "_r3_datum_offset": 1,
        "_r3_datum_count": 2,
    }

    my_batches, routed_experts, routed_logits = dispatcher._shard_and_slice_batches(
        [batch],
        routed_experts=["r0", "r1", "r2", "r3"],
        routed_expert_logits=["l0", "l1", "l2", "l3"],
        cp_enabled=False,
        parallel_state=_parallel_state(),
    )

    assert len(my_batches) == 1
    assert "_r3_datum_offset" not in my_batches[0]
    assert "_r3_datum_count" not in my_batches[0]
    assert routed_experts == ["r1", "r2"]
    assert routed_logits == ["l1", "l2"]
