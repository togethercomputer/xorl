"""Gates for the GLM-5.2 full-param checksummed byte-payload protocol.

Everything here is CPU-runnable: the protocol moves bytes, not arithmetic.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION,
    GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION,
    GLM52_FULLPARAM_DENSE_MLP_CONTRACT_VERSION,
    Glm52ExactFullParamRouterWeight,
    Glm52ExactTP1BlockFP8FullParamLinear,
)
from xorl.server.weight_sync.glm52_fullparam_payload import (
    GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION,
    Glm52ExpectedPayloadField,
    Glm52ExpectedPayloadInventory,
    Glm52ExpectedPayloadItem,
    Glm52FullParamPayloadError,
    activate_glm52_prepared_checkpoint,
    apply_glm52_fullparam_payload,
    build_glm52_expected_payload_inventory,
    load_glm52_fullparam_payload,
    publish_glm52_fullparam_payload,
    save_glm52_fullparam_payload,
    unpack_glm52_payload_field,
    verify_glm52_fullparam_payload,
)


def _seeded_linear(in_features: int = 128, out_features: int = 256) -> Glm52ExactTP1BlockFP8FullParamLinear:
    """A CPU component with deterministic cache bytes and a fresh identity."""

    module = Glm52ExactTP1BlockFP8FullParamLinear(in_features, out_features, device=torch.device("cpu"))
    with torch.no_grad():
        module.weight_master.zero_()
        raw_bytes = (torch.arange(module.quantized_weight_f32.numel() * 4, dtype=torch.int64) % 251).to(torch.uint8)
        module.quantized_weight_f32.copy_(raw_bytes.view(torch.float32).reshape(module.quantized_weight_f32.shape))
        scale_values = torch.arange(module.weight_scale_inv.numel(), dtype=torch.float32)
        module.weight_scale_inv.copy_(scale_values.remainder(31).add(1).div(32).reshape(module.weight_scale_inv.shape))
    module._record_master_identity()
    return module


def _seeded_router(num_experts: int = 8, hidden_size: int = 64) -> Glm52ExactFullParamRouterWeight:
    module = Glm52ExactFullParamRouterWeight(num_experts, hidden_size, device=torch.device("cpu"))
    checkpoint = (
        torch.arange(num_experts * hidden_size, dtype=torch.float32)
        .reshape(num_experts, hidden_size)
        .sub_(97)
        .div_(53)
        .to(torch.bfloat16)
    )
    module.load_from_bf16(checkpoint)
    return module


def _payload(weight_version: str = "step-7", weight_step: int = 7):
    return publish_glm52_fullparam_payload(
        [
            ("model.layers.0.mlp.gate_proj", _seeded_linear()),
            ("model.layers.1.mlp.gate", _seeded_router()),
        ],
        weight_version=weight_version,
        weight_step=weight_step,
    )


def _expected_inventory(payload) -> Glm52ExpectedPayloadInventory:
    """Build a test fixture; production inventories come from receiver admission."""

    return Glm52ExpectedPayloadInventory(
        items=tuple(
            Glm52ExpectedPayloadItem(
                target=item.target,
                kind=item.kind,
                contract_version=item.contract_version,
                fields=tuple(
                    Glm52ExpectedPayloadField(
                        name=field.name,
                        dtype=field.dtype,
                        shape=field.shape,
                    )
                    for field in item.fields
                ),
            )
            for item in payload.items
        )
    )


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


def test_publish_snapshots_the_consumed_caches_immutably(monkeypatch) -> None:
    from xorl.server.weight_sync import glm52_fullparam_payload as payload_module

    published_logs: list[str] = []
    monkeypatch.setattr(
        payload_module.logger,
        "info",
        lambda message, *args: published_logs.append(message % args),
    )
    linear = _seeded_linear()
    router = _seeded_router()
    payload = publish_glm52_fullparam_payload(
        [("dense", linear), ("router", router)],
        weight_version="step-3",
    )

    assert payload.protocol_version == GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION
    assert [item.kind for item in payload.items] == ["block_fp8_linear", "bf16_router"]

    dense_item = payload.items[0]
    assert [field.name for field in dense_item.fields] == ["weight", "weight_scale_inv"]
    # Publication owns its bytes: a snapshot, never an alias of the caches, so
    # a later refresh cannot rewrite checksummed bytes.
    assert dense_item.fields[0].data.data_ptr() != linear.quantized_weight_f32.data_ptr()
    assert dense_item.fields[1].data.data_ptr() != linear.weight_scale_inv.data_ptr()
    router_item = payload.items[1]
    assert router_item.fields[0].data.data_ptr() != router._effective_weight.data_ptr()

    # The snapshot reproduces the exact bytes the forward consumed.
    assert torch.equal(
        unpack_glm52_payload_field(dense_item.fields[0]).view(torch.uint8),
        linear._cached_fp8_weight().view(torch.uint8),
    )
    assert torch.equal(unpack_glm52_payload_field(dense_item.fields[1]), linear.weight_scale_inv)
    assert torch.equal(unpack_glm52_payload_field(router_item.fields[0]), router._effective_weight)

    assert len(published_logs) == 1
    assert "payload published" in published_logs[0]

    verify_glm52_fullparam_payload(payload)


def test_published_payload_survives_a_later_cache_refresh(tmp_path) -> None:
    """A later cache refresh cannot change an already published snapshot."""

    linear = _seeded_linear()
    original_bytes = linear.quantized_weight_f32.detach().clone().view(torch.uint8).flatten()
    payload = publish_glm52_fullparam_payload([("dense", linear)], weight_version="step-1", weight_step=1)

    # Simulate the next optimizer step's refresh: overwrite the cache bytes
    # in place (a CPU stand-in for refresh_quantized_cache's copy_).
    with torch.no_grad():
        linear.weight_master.add_(1.0)
        flipped = linear.quantized_weight_f32.view(torch.uint8)
        flipped.copy_(255 - flipped)
    linear._record_master_identity()

    field = payload.items[0].fields[0]
    assert torch.equal(field.data, original_bytes), "published bytes were mutated by the cache refresh"
    assert not torch.equal(
        field.data,
        linear.quantized_weight_f32.detach().view(torch.uint8).flatten(),
    ), "the cache mutation itself did not take (fixture bug)"
    verify_glm52_fullparam_payload(payload)

    # The payload round-trips its ORIGINAL bytes through disk after the refresh.
    directory = str(tmp_path / "post-refresh")
    save_glm52_fullparam_payload(payload, directory)
    reloaded = load_glm52_fullparam_payload(directory)
    assert torch.equal(reloaded.items[0].fields[0].data, original_bytes)

    # A fresh publication carries the refreshed bytes under a new step.
    successor = publish_glm52_fullparam_payload([("dense", linear)], weight_version="step-2", weight_step=2)
    assert not torch.equal(successor.items[0].fields[0].data, original_bytes)
    assert successor.items[0].fields[0].checksum != field.checksum


def test_publish_fails_closed_on_stale_unknown_duplicate_and_empty() -> None:
    linear = _seeded_linear()
    with torch.no_grad():
        linear.weight_master.add_(1.0)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        publish_glm52_fullparam_payload([("dense", linear)], weight_version="v")

    with pytest.raises(Glm52FullParamPayloadError, match="does not declare an admitted"):
        publish_glm52_fullparam_payload(
            [("plain", torch.nn.Linear(4, 4))],
            weight_version="v",
        )

    module = _seeded_linear()
    with pytest.raises(Glm52FullParamPayloadError, match="Duplicate payload target"):
        publish_glm52_fullparam_payload([("a", module), ("a", module)], weight_version="v")

    with pytest.raises(Glm52FullParamPayloadError, match="empty"):
        publish_glm52_fullparam_payload([], weight_version="v")

    with pytest.raises(Glm52FullParamPayloadError, match="weight_version"):
        publish_glm52_fullparam_payload([("a", module)], weight_version="")
    with pytest.raises(Glm52FullParamPayloadError, match="weight_version"):
        publish_glm52_fullparam_payload([("a", module)], weight_version=7)  # type: ignore[arg-type]
    with pytest.raises(Glm52FullParamPayloadError, match="weight_step"):
        publish_glm52_fullparam_payload([("a", module)], weight_version="v", weight_step=True)


# ---------------------------------------------------------------------------
# Round-trip and tamper detection
# ---------------------------------------------------------------------------


def test_save_load_verify_roundtrip_preserves_every_byte(tmp_path) -> None:
    payload = _payload()
    directory = str(tmp_path / "sync")
    save_glm52_fullparam_payload(payload, directory)
    loaded = load_glm52_fullparam_payload(directory)

    assert loaded.protocol_version == payload.protocol_version
    assert loaded.weight_version == payload.weight_version
    assert loaded.manifest_checksum == payload.manifest_checksum
    for original_item, loaded_item in zip(payload.items, loaded.items, strict=True):
        assert (original_item.target, original_item.kind) == (loaded_item.target, loaded_item.kind)
        for original_field, loaded_field in zip(original_item.fields, loaded_item.fields, strict=True):
            assert original_field.checksum == loaded_field.checksum
            assert torch.equal(original_field.data, loaded_field.data)

    with pytest.raises(Glm52FullParamPayloadError, match="refusing to overwrite"):
        save_glm52_fullparam_payload(payload, directory)

    # Verification cannot be disabled at the trust boundary.
    with pytest.raises(TypeError, match="verify"):
        load_glm52_fullparam_payload(directory, verify=False)  # type: ignore[call-arg]


def test_payload_serializer_failure_leaves_no_visible_or_staged_tree(tmp_path, monkeypatch) -> None:
    from xorl.server.weight_sync import glm52_fullparam_payload as payload_module

    payload = _payload()
    output = tmp_path / "atomic-output"
    original = payload_module._create_regular_at
    calls = 0

    def fail_after_one_file(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected exclusive-write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(payload_module, "_create_regular_at", fail_after_one_file)
    with pytest.raises(OSError, match="exclusive-write failure"):
        save_glm52_fullparam_payload(payload, str(output))
    assert not output.exists()
    assert not [entry for entry in os.listdir(tmp_path) if ".atomic-output.staging-" in entry]


def test_atomic_rename_is_the_nonthrowing_payload_commit_point(tmp_path, monkeypatch) -> None:
    from xorl.server.weight_sync import glm52_fullparam_payload as payload_module

    payload = _payload()
    output = tmp_path / "committed"
    real_fsync = payload_module.os.fsync

    def fail_only_after_visibility(descriptor: int) -> None:
        if output.exists():
            raise OSError("injected post-rename parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(payload_module.os, "fsync", fail_only_after_visibility)
    save_glm52_fullparam_payload(payload, str(output))
    assert output.is_dir()
    loaded = load_glm52_fullparam_payload(str(output))
    assert loaded.manifest_checksum == payload.manifest_checksum


@pytest.mark.parametrize("malicious_kind", ["traversal", "symlink", "fifo"])
def test_payload_loader_rejects_noncanonical_or_special_byte_files(tmp_path, malicious_kind) -> None:
    payload = _payload()
    directory = tmp_path / malicious_kind
    save_glm52_fullparam_payload(payload, str(directory))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    victim_name = manifest["items"][0]["fields"][0]["file"]
    victim = directory / victim_name

    if malicious_kind == "traversal":
        manifest["items"][0]["fields"][0]["file"] = "../outside.bin"
        manifest_path.write_text(json.dumps(manifest))
        match = "plain filename"
    else:
        victim.unlink()
        outside = tmp_path / f"outside-{malicious_kind}.bin"
        if malicious_kind == "symlink":
            outside.write_bytes(b"x" * payload.items[0].fields[0].data.numel())
            victim.symlink_to(outside)
            match = "securely open|nonsymlink"
        else:
            os.mkfifo(victim)
            match = "regular nonsymlink"

    with pytest.raises(Glm52FullParamPayloadError, match=match):
        load_glm52_fullparam_payload(str(directory))


def test_payload_loader_rejects_symlink_directory(tmp_path) -> None:
    payload = _payload()
    real = tmp_path / "real"
    save_glm52_fullparam_payload(payload, str(real))
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(Glm52FullParamPayloadError, match="symlink components"):
        load_glm52_fullparam_payload(str(alias))


def test_single_flipped_byte_is_detected_before_any_application(tmp_path) -> None:
    payload = _payload()
    directory = str(tmp_path / "sync")
    save_glm52_fullparam_payload(payload, directory)

    bin_files = sorted(name for name in os.listdir(directory) if name.endswith(".bin"))
    victim = os.path.join(directory, bin_files[0])
    with open(victim, "r+b") as handle:
        handle.seek(11)
        original = handle.read(1)
        handle.seek(11)
        handle.write(bytes([original[0] ^ 0x01]))

    with pytest.raises(Glm52FullParamPayloadError, match="Checksum mismatch"):
        load_glm52_fullparam_payload(directory)

    # Restore and confirm the payload verifies again (the flip was the only issue).
    with open(victim, "r+b") as handle:
        handle.seek(11)
        handle.write(original)
    load_glm52_fullparam_payload(directory)


def test_manifest_tampering_and_partial_manifests_fail_closed(tmp_path) -> None:
    payload = _payload()
    directory = str(tmp_path / "sync")
    save_glm52_fullparam_payload(payload, directory)
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    def _write(mutated) -> None:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(mutated, handle)

    # Dtype rewrite: the pinned kind schema rejects it before anything else.
    mutated = json.loads(json.dumps(manifest))
    mutated["items"][0]["fields"][1]["dtype"] = "bfloat16"
    _write(mutated)
    with pytest.raises(Glm52FullParamPayloadError, match="field schema"):
        load_glm52_fullparam_payload(directory)

    # Shape shrink that changes the byte count: rejected on byte accounting.
    mutated = json.loads(json.dumps(manifest))
    mutated["items"][0]["fields"][0]["shape"][0] //= 2
    _write(mutated)
    with pytest.raises(Glm52FullParamPayloadError, match="carries .* bytes"):
        load_glm52_fullparam_payload(directory)

    # Shape rewrite that keeps the byte count: the header-bound checksum trips.
    mutated = json.loads(json.dumps(manifest))
    original_shape = mutated["items"][0]["fields"][0]["shape"]
    mutated["items"][0]["fields"][0]["shape"] = [original_shape[1], original_shape[0]]
    _write(mutated)
    with pytest.raises(Glm52FullParamPayloadError, match="Checksum mismatch"):
        load_glm52_fullparam_payload(directory)

    # Weight-version relabel: field headers bind the version, so replaying
    # old bytes under a new version label is detected.
    mutated = json.loads(json.dumps(manifest))
    mutated["weight_version"] = "step-8"
    _write(mutated)
    with pytest.raises(Glm52FullParamPayloadError, match="Checksum mismatch"):
        load_glm52_fullparam_payload(directory)

    # Manifest checksum relabel.
    mutated = json.loads(json.dumps(manifest))
    mutated["manifest_checksum"] = "0" * 64
    _write(mutated)
    with pytest.raises(Glm52FullParamPayloadError, match="manifest checksum mismatch"):
        load_glm52_fullparam_payload(directory)

    # Missing byte file.
    _write(manifest)
    bin_files = sorted(name for name in os.listdir(directory) if name.endswith(".bin"))
    os.remove(os.path.join(directory, bin_files[-1]))
    with pytest.raises(Glm52FullParamPayloadError, match="byte file missing"):
        load_glm52_fullparam_payload(directory)


def test_verify_rejects_protocol_and_schema_violations() -> None:
    payload = _payload()

    import dataclasses

    with pytest.raises(Glm52FullParamPayloadError, match="Unsupported payload protocol"):
        verify_glm52_fullparam_payload(dataclasses.replace(payload, protocol_version="v0"))

    item = payload.items[0]
    reordered = dataclasses.replace(item, fields=tuple(reversed(item.fields)))
    with pytest.raises(Glm52FullParamPayloadError, match="field schema"):
        verify_glm52_fullparam_payload(dataclasses.replace(payload, items=(reordered,)))

    unknown = dataclasses.replace(item, kind="int4_linear")
    with pytest.raises(Glm52FullParamPayloadError, match="Unknown payload kind"):
        verify_glm52_fullparam_payload(dataclasses.replace(payload, items=(unknown,)))

    duplicated = dataclasses.replace(payload, items=(item, item))
    with pytest.raises(Glm52FullParamPayloadError, match="Duplicate payload target"):
        verify_glm52_fullparam_payload(duplicated)

    with pytest.raises(Glm52FullParamPayloadError, match="weight_step"):
        verify_glm52_fullparam_payload(dataclasses.replace(payload, weight_step=True))
    with pytest.raises(Glm52FullParamPayloadError, match="weight_version"):
        verify_glm52_fullparam_payload(dataclasses.replace(payload, weight_version=3))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Two-phase application
# ---------------------------------------------------------------------------


class _RecordingBlockFP8Receiver:
    """Serving-shaped stub: applicable AND snapshottable (transactional contract)."""

    def __init__(self, out_features: int = 256, in_features: int = 128, fail_after: int | None = None) -> None:
        self.applied: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._weight = torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        self.weight_scale_inv = torch.zeros(2, 1, dtype=torch.float32)
        self._fail_after = fail_after

    def fp8_weight(self) -> torch.Tensor:
        return self._weight

    def load_prequantized(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        assert weight.dtype is torch.float8_e4m3fn
        assert scale.dtype is torch.float32
        if self._fail_after is not None and len(self.applied) >= self._fail_after:
            # Adversarial loader: MUTATES first, then throws ONCE — the failing
            # receiver itself must be rolled back. One-shot so the rollback
            # invocation succeeds; the permanent-failure case (rollback also
            # failing -> UNKNOWN-state error) is asserted separately.
            self._fail_after = None
            self._weight = weight.clone()
            self.weight_scale_inv = scale.clone()
            raise RuntimeError("injected commit failure after mutation")
        self._weight = weight.clone()
        self.weight_scale_inv = scale.clone()
        self.applied.append((weight, scale))


def _receiver_owned_basic_inventory(
    dense_receiver: _RecordingBlockFP8Receiver,
    router_receiver: torch.Tensor,
) -> Glm52ExpectedPayloadInventory:
    """Admission constants plus receiver geometry, never incoming payload fields."""

    return build_glm52_expected_payload_inventory(
        (
            (
                "model.layers.0.mlp.gate_proj",
                "block_fp8_linear",
                GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION,
                dense_receiver,
            ),
            (
                "model.layers.1.mlp.gate",
                "bf16_router",
                GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION,
                router_receiver,
            ),
        )
    )


def _g():
    from xorl.server.weight_sync.glm52_fullparam_payload import Glm52WeightVersionGuard

    return Glm52WeightVersionGuard()


def test_version_guard_rejects_noninteger_restart_state() -> None:
    from xorl.server.weight_sync.glm52_fullparam_payload import Glm52WeightVersionGuard

    with pytest.raises(Glm52FullParamPayloadError, match="last_step"):
        Glm52WeightVersionGuard(True)


def test_apply_verifies_and_stages_everything_before_mutating_any_receiver() -> None:
    payload = _payload()
    dense_receiver = _RecordingBlockFP8Receiver()
    router_receiver = torch.zeros(8, 64, dtype=torch.bfloat16)

    # Phase-1 failure on the SECOND item: the first receiver must stay untouched.
    def failing_resolver(target: str, kind: str):
        if kind == "block_fp8_linear":
            return dense_receiver
        return None

    with pytest.raises(Glm52FullParamPayloadError, match="No receiver resolved"):
        apply_glm52_fullparam_payload(
            payload,
            failing_resolver,
            expected_inventory=_expected_inventory(payload),
            version_guard=_g(),
        )
    assert dense_receiver.applied == []

    # Router receiver dtype/shape admission also fails before any application.
    def wrong_router_resolver(target: str, kind: str):
        if kind == "block_fp8_linear":
            return dense_receiver
        return torch.zeros(8, 64, dtype=torch.float32)

    with pytest.raises(Glm52FullParamPayloadError, match="must be BF16"):
        apply_glm52_fullparam_payload(
            payload,
            wrong_router_resolver,
            expected_inventory=_expected_inventory(payload),
            version_guard=_g(),
        )
    assert dense_receiver.applied == []

    # Success path applies every item.
    def resolver(target: str, kind: str):
        return dense_receiver if kind == "block_fp8_linear" else router_receiver

    from xorl.server.weight_sync.glm52_fullparam_payload import Glm52WeightVersionGuard as _Guard

    with pytest.raises(Glm52FullParamPayloadError, match="persistent Glm52WeightVersionGuard"):
        apply_glm52_fullparam_payload(
            payload,
            resolver,
            expected_inventory=_expected_inventory(payload),
            version_guard=None,  # type: ignore[arg-type]
        )
    apply_glm52_fullparam_payload(
        payload,
        resolver,
        expected_inventory=_receiver_owned_basic_inventory(dense_receiver, router_receiver),
        version_guard=_Guard(),
    )
    assert len(dense_receiver.applied) == 1
    weight, scale = dense_receiver.applied[0]
    assert tuple(weight.shape) == (256, 128)
    assert tuple(scale.shape) == (2, 1)
    expected_router = unpack_glm52_payload_field(payload.items[1].fields[0])
    assert torch.equal(router_receiver, expected_router)

    # Transactional commit: a failure mid-apply rolls back earlier receivers.
    from xorl.server.weight_sync.glm52_fullparam_payload import Glm52WeightVersionGuard

    failing_dense = _RecordingBlockFP8Receiver(fail_after=0)
    rollback_router = torch.full((8, 64), 7.0, dtype=torch.bfloat16)
    with pytest.raises(Glm52FullParamPayloadError, match="rolled back"):
        apply_glm52_fullparam_payload(
            payload,
            lambda target, kind: failing_dense if kind == "block_fp8_linear" else rollback_router,
            expected_inventory=_expected_inventory(payload),
            version_guard=Glm52WeightVersionGuard(),
        )
    assert torch.equal(rollback_router, torch.full((8, 64), 7.0, dtype=torch.bfloat16))
    # The failing receiver mutated before throwing; byte-verify it was rolled
    # back to its pre-apply bytes too.
    assert torch.count_nonzero(failing_dense.fp8_weight().view(torch.uint8)) == 0
    assert torch.count_nonzero(failing_dense.weight_scale_inv) == 0

    # Permanent loader failure: rollback also fails -> distinct UNKNOWN-state
    # error naming the receiver, never a false "rolled back" claim.
    class _PermanentlyFailing(_RecordingBlockFP8Receiver):
        def load_prequantized(self, weight, scale):
            self._weight = weight.clone()
            raise RuntimeError("permanent loader failure")

    with pytest.raises(Glm52FullParamPayloadError, match="UNKNOWN state.*gate_proj"):
        apply_glm52_fullparam_payload(
            payload,
            lambda target, kind: _PermanentlyFailing() if kind == "block_fp8_linear" else rollback_router,
            expected_inventory=_expected_inventory(payload),
            version_guard=Glm52WeightVersionGuard(),
        )

    # Monotonic version guard: regressions RAISE and apply nothing.
    guard = Glm52WeightVersionGuard()
    fresh_dense = _RecordingBlockFP8Receiver()
    fresh_router = torch.zeros(8, 64, dtype=torch.bfloat16)
    apply_glm52_fullparam_payload(
        payload,
        lambda target, kind: fresh_dense if kind == "block_fp8_linear" else fresh_router,
        expected_inventory=_expected_inventory(payload),
        version_guard=guard,
    )
    assert guard.last_step == payload.weight_step
    with pytest.raises(Glm52FullParamPayloadError, match="regression"):
        apply_glm52_fullparam_payload(
            payload,
            lambda target, kind: fresh_dense if kind == "block_fp8_linear" else fresh_router,
            expected_inventory=_expected_inventory(payload),
            version_guard=guard,
        )
    assert len(fresh_dense.applied) == 1


def test_apply_rejects_receiver_without_serving_contract() -> None:
    payload = _payload()

    with pytest.raises(Glm52FullParamPayloadError, match="load_prequantized"):
        apply_glm52_fullparam_payload(
            payload,
            lambda target, kind: object(),
            expected_inventory=_expected_inventory(payload),
            version_guard=_g(),
        )


def test_in_memory_updates_are_serialized_by_the_guard_owned_transaction() -> None:
    first_payload = _payload(weight_version="step-7", weight_step=7)
    second_payload = _payload(weight_version="step-8", weight_step=8)
    first_loader_entered = threading.Event()
    release_first_loader = threading.Event()
    second_resolver_entered = threading.Event()

    class BlockingFirstReceiver(_RecordingBlockFP8Receiver):
        def load_prequantized(self, weight, scale):
            if not first_loader_entered.is_set():
                first_loader_entered.set()
                assert release_first_loader.wait(timeout=5)
            super().load_prequantized(weight, scale)

    dense_receiver = BlockingFirstReceiver()
    router_receiver = torch.zeros(8, 64, dtype=torch.bfloat16)
    inventory = _receiver_owned_basic_inventory(dense_receiver, router_receiver)
    guard = _g()

    def first_resolver(target: str, kind: str):
        return dense_receiver if kind == "block_fp8_linear" else router_receiver

    def second_resolver(target: str, kind: str):
        second_resolver_entered.set()
        return dense_receiver if kind == "block_fp8_linear" else router_receiver

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            apply_glm52_fullparam_payload,
            first_payload,
            first_resolver,
            expected_inventory=inventory,
            version_guard=guard,
        )
        assert first_loader_entered.wait(timeout=5)
        second = executor.submit(
            apply_glm52_fullparam_payload,
            second_payload,
            second_resolver,
            expected_inventory=inventory,
            version_guard=guard,
        )
        assert not second_resolver_entered.wait(timeout=0.2), "second update escaped the guard transaction"
        release_first_loader.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert second_resolver_entered.is_set()
    assert len(dense_receiver.applied) == 2
    assert guard.last_step == 8


@pytest.mark.parametrize("receiver_kind", ["memory", "hf_checkpoint"])
def test_receivers_require_the_exact_independent_semantic_inventory(receiver_kind) -> None:
    """A checksum-valid subset or semantic drift must fail before any apply."""

    from dataclasses import replace

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    payload = _payload()
    inventory = _expected_inventory(payload)
    partial_payload = publish_glm52_fullparam_payload(
        [(payload.items[0].target, _seeded_linear())],
        weight_version=payload.weight_version,
        weight_step=payload.weight_step,
    )
    inventory_without_router = replace(inventory, items=inventory.items[:1])
    wrong_contract_item = replace(
        inventory.items[0],
        contract_version=f"{inventory.items[0].contract_version}-unadmitted",
    )
    wrong_contract = replace(
        inventory,
        items=(wrong_contract_item,) + inventory.items[1:],
    )
    first_field = inventory.items[0].fields[0]
    wrong_shape_field = replace(
        first_field,
        shape=(first_field.shape[0] + 1,) + first_field.shape[1:],
    )
    wrong_shape_item = replace(
        inventory.items[0],
        fields=(wrong_shape_field,) + inventory.items[0].fields[1:],
    )
    wrong_shape = replace(
        inventory,
        items=(wrong_shape_item,) + inventory.items[1:],
    )

    def assert_rejected(candidate, expected, message: str) -> None:
        guard = _g()
        if receiver_kind == "memory":

            def resolver_must_not_run(target: str, kind: str):
                pytest.fail(f"resolver ran for invalid inventory: {target} {kind}")

            with pytest.raises(Glm52FullParamPayloadError, match=message):
                apply_glm52_fullparam_payload(
                    candidate,
                    resolver_must_not_run,
                    expected_inventory=expected,
                    version_guard=guard,
                )
        else:
            with pytest.raises(Glm52FullParamPayloadError, match=message):
                apply_glm52_fullparam_payload_to_hf_checkpoint(
                    candidate,
                    "must-not-be-read",
                    "must-not-be-written",
                    glm52_fullparam_hf_name_mapping(candidate),
                    expected_inventory=expected,
                )
        assert guard.last_step == -1

    assert_rejected(partial_payload, inventory, "inventory target mismatch.*missing")
    assert_rejected(payload, inventory_without_router, "inventory target mismatch.*unexpected")
    assert_rejected(payload, wrong_contract, "contract_version mismatch")
    assert_rejected(payload, wrong_shape, "field inventory mismatch")


def test_hf_checkpoint_receiver_rewrites_exactly_the_mapped_tensors(tmp_path) -> None:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
    )

    payload = _payload()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    dense_weight = unpack_glm52_payload_field(payload.items[0].fields[0])
    dense_scale = unpack_glm52_payload_field(payload.items[0].fields[1])
    router_weight = unpack_glm52_payload_field(payload.items[1].fields[0])
    untouched = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    save_safetensors(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.zeros_like(dense_weight),
            "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.zeros_like(dense_scale),
            "model.layers.1.mlp.gate.weight": torch.zeros_like(router_weight),
            "model.untouched.weight": untouched,
        },
        str(base / "model.safetensors"),
    )
    mapping = {
        "model.layers.0.mlp.gate_proj": {
            "weight": "model.layers.0.mlp.gate_proj.weight",
            "weight_scale_inv": "model.layers.0.mlp.gate_proj.weight_scale_inv",
        },
        "model.layers.1.mlp.gate": {"weight": "model.layers.1.mlp.gate.weight"},
    }

    out = tmp_path / "updated"
    hf_guard = _g()
    receiver_inventory = _receiver_owned_basic_inventory(
        _RecordingBlockFP8Receiver(),
        torch.zeros_like(router_weight),
    )
    prepared = apply_glm52_fullparam_payload_to_hf_checkpoint(
        payload,
        str(base),
        str(out),
        mapping,
        expected_inventory=receiver_inventory,
    )
    assert hf_guard.last_step == -1, "materialization must not activate the serving version"
    failing_guard = _g()

    def fail_activation(_directory: str) -> None:
        raise RuntimeError("injected engine reload failure")

    with pytest.raises(Glm52FullParamPayloadError, match="serving state may be UNKNOWN"):
        activate_glm52_prepared_checkpoint(
            prepared,
            fail_activation,
            version_guard=failing_guard,
        )
    assert failing_guard.last_step == -1

    activated: list[str] = []
    activate_glm52_prepared_checkpoint(
        prepared,
        activated.append,
        version_guard=hf_guard,
    )
    assert activated == [str(out)]
    assert hf_guard.last_step == payload.weight_step
    updated = load_safetensors(str(out / "model.safetensors"))
    assert torch.equal(
        updated["model.layers.0.mlp.gate_proj.weight"].view(torch.uint8),
        dense_weight.view(torch.uint8),
    )
    assert torch.equal(updated["model.layers.0.mlp.gate_proj.weight_scale_inv"], dense_scale)
    assert torch.equal(updated["model.layers.1.mlp.gate.weight"], router_weight)
    assert torch.equal(updated["model.untouched.weight"], untouched)
    assert (out / "config.json").read_text() == (base / "config.json").read_text()

    with pytest.raises(Glm52FullParamPayloadError, match="in place"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(base),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )
    with pytest.raises(Glm52FullParamPayloadError, match="canonical payload-to-checkpoint mapping"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(tmp_path / "x"),
            {"model.layers.0.mlp.gate_proj": mapping["model.layers.0.mlp.gate_proj"]},
            expected_inventory=_expected_inventory(payload),
        )
    bad_dtype_dir = tmp_path / "bad"
    bad_dtype_dir.mkdir()
    save_safetensors(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.zeros_like(dense_weight).to(torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.zeros_like(dense_scale),
            "model.layers.1.mlp.gate.weight": torch.zeros_like(router_weight),
        },
        str(bad_dtype_dir / "model.safetensors"),
    )
    with pytest.raises(Glm52FullParamPayloadError, match="dtype/shape drift"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(bad_dtype_dir),
            str(tmp_path / "y"),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )
    # The atomic contract refuses a pre-existing target directory outright.
    with pytest.raises(Glm52FullParamPayloadError, match="already exists"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(out),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )


def test_hf_checkpoint_receiver_folds_full_param_router_wrapper(tmp_path) -> None:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    router = _seeded_router()
    payload = publish_glm52_fullparam_payload(
        [("model.layers.3.mlp.gate.full_param", router)],
        weight_version="step-1",
        weight_step=1,
    )
    router_weight = unpack_glm52_payload_field(payload.items[0].fields[0])
    base = tmp_path / "base-router-wrapper"
    base.mkdir()
    save_safetensors(
        {"model.layers.3.mlp.gate.weight": torch.zeros_like(router_weight)},
        str(base / "model.safetensors"),
    )

    mapping = glm52_fullparam_hf_name_mapping(payload)
    assert mapping == {
        "model.layers.3.mlp.gate.full_param": {
            "weight": "model.layers.3.mlp.gate.weight",
        }
    }
    out = tmp_path / "updated-router-wrapper"
    apply_glm52_fullparam_payload_to_hf_checkpoint(
        payload,
        str(base),
        str(out),
        mapping,
        expected_inventory=_expected_inventory(payload),
    )
    updated = load_safetensors(str(out / "model.safetensors"))
    assert torch.equal(updated["model.layers.3.mlp.gate.weight"], router_weight)


def test_hf_checkpoint_receiver_rejects_same_shape_cross_layer_remap(tmp_path) -> None:
    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    targets = (
        "model.layers.3.mlp.gate.full_param",
        "model.layers.4.mlp.gate.full_param",
    )
    payload = publish_glm52_fullparam_payload(
        [(target, _seeded_router()) for target in targets],
        weight_version="step-1",
        weight_step=1,
    )
    mapping = glm52_fullparam_hf_name_mapping(payload)
    redirected = {target: dict(fields) for target, fields in mapping.items()}
    redirected[targets[0]]["weight"], redirected[targets[1]]["weight"] = (
        redirected[targets[1]]["weight"],
        redirected[targets[0]]["weight"],
    )

    base = tmp_path / "base-cross-layer-remap"
    base.mkdir()
    router_weight = unpack_glm52_payload_field(payload.items[0].fields[0])
    save_safetensors(
        {mapping[target]["weight"]: torch.zeros_like(router_weight) for target in targets},
        str(base / "model.safetensors"),
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(Glm52FullParamPayloadError, match="caller-directed tensor remap"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(output),
            redirected,
            expected_inventory=_expected_inventory(payload),
        )
    assert not output.exists()


def test_hf_mapping_covers_the_complete_glm52_checkpoint_update_scope() -> None:
    """Regress the architecture's exact 334-item / 1,629-field update shape."""

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        glm52_fullparam_hf_name_mapping,
    )

    dense = _seeded_dense_mlp(hidden=64, intermediate=128)
    bank = _seeded_expert_bank(local_experts=1, hidden=128, intermediate=128)
    bank.assign_global_expert_range(0, 1)
    expert = bank.checkpoint_publications()[0][1]
    router = _seeded_router(num_experts=8, hidden_size=8)
    publications = [(f"model.layers.{layer}.mlp", dense) for layer in range(3)]
    publications.extend((f"model.layers.3.mlp.experts.{expert_id}", expert) for expert_id in range(256))
    publications.extend((f"model.layers.{layer}.mlp.gate.full_param", router) for layer in range(3, 78))

    payload = publish_glm52_fullparam_payload(
        publications,
        weight_version="step-1",
        weight_step=1,
    )
    mapping = glm52_fullparam_hf_name_mapping(payload)
    mapped_names = [hf_name for item_mapping in mapping.values() for hf_name in item_mapping.values()]
    projection_suffixes = {
        f"{projection}_proj.{field}"
        for projection in ("gate", "up", "down")
        for field in ("weight", "weight_scale_inv")
    }
    expected_names = {f"model.layers.{layer}.mlp.{suffix}" for layer in range(3) for suffix in projection_suffixes}
    expected_names.update(
        f"model.layers.3.mlp.experts.{expert_id}.{suffix}" for expert_id in range(256) for suffix in projection_suffixes
    )
    expected_names.update(f"model.layers.{layer}.mlp.gate.weight" for layer in range(3, 78))
    synthetic_weight_map = dict.fromkeys(expected_names, "model.safetensors")

    assert len(payload.items) == 334
    assert len(mapped_names) == 1_629
    assert len(set(mapped_names)) == len(mapped_names)
    assert set(mapped_names) == set(synthetic_weight_map)
    assert "model.layers.10.mlp.gate.weight" in mapped_names
    assert not any(name.endswith(".gate.full_param.weight") for name in mapped_names)


def test_hf_canonical_mapping_rejects_global_destination_collisions() -> None:
    from xorl.server.weight_sync.glm52_fullparam_payload import glm52_fullparam_hf_name_mapping

    payload = publish_glm52_fullparam_payload(
        [
            ("model.layers.3.mlp.gate", _seeded_router()),
            ("model.layers.3.mlp.gate.full_param", _seeded_router()),
        ],
        weight_version="step-1",
        weight_step=1,
    )
    with pytest.raises(Glm52FullParamPayloadError, match="destination collision"):
        glm52_fullparam_hf_name_mapping(payload)


@pytest.mark.parametrize("entry_kind", ["symlink", "fifo"])
def test_hf_checkpoint_rejects_special_base_entries(tmp_path, entry_kind) -> None:
    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    payload = _payload()
    mapping = glm52_fullparam_hf_name_mapping(payload)
    base = tmp_path / f"base-{entry_kind}"
    base.mkdir()
    save_safetensors(
        {
            mapping[payload.items[0].target]["weight"]: torch.zeros_like(
                unpack_glm52_payload_field(payload.items[0].fields[0])
            ),
            mapping[payload.items[0].target]["weight_scale_inv"]: torch.zeros_like(
                unpack_glm52_payload_field(payload.items[0].fields[1])
            ),
            mapping[payload.items[1].target]["weight"]: torch.zeros_like(
                unpack_glm52_payload_field(payload.items[1].fields[0])
            ),
        },
        str(base / "model.safetensors"),
    )
    special = base / "config.json"
    if entry_kind == "symlink":
        outside = tmp_path / "outside-config.json"
        outside.write_text("{}")
        special.symlink_to(outside)
    else:
        os.mkfifo(special)

    with pytest.raises(Glm52FullParamPayloadError, match="regular nonsymlink"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(tmp_path / f"out-{entry_kind}"),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )


def test_hf_index_rejects_traversal_shard_names(tmp_path) -> None:
    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    payload = _payload()
    mapping = glm52_fullparam_hf_name_mapping(payload)
    tensors = {
        hf_name: torch.zeros_like(unpack_glm52_payload_field(field))
        for item in payload.items
        for field in item.fields
        for hf_name in (mapping[item.target][field.name],)
    }
    base = tmp_path / "indexed-base"
    base.mkdir()
    save_safetensors(tensors, str(base / "model-00001-of-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "../model-00001-of-00001.safetensors")})
    )

    with pytest.raises(Glm52FullParamPayloadError, match="plain filename"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(tmp_path / "indexed-out"),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )


def test_hf_output_parent_symlink_is_rejected(tmp_path) -> None:
    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    payload = _payload()
    mapping = glm52_fullparam_hf_name_mapping(payload)
    base = tmp_path / "base"
    base.mkdir()
    save_safetensors(
        {
            hf_name: torch.zeros_like(unpack_glm52_payload_field(field))
            for item in payload.items
            for field in item.fields
            for hf_name in (mapping[item.target][field.name],)
        },
        str(base / "model.safetensors"),
    )
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(Glm52FullParamPayloadError, match="symlink components"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(alias_parent / "output"),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )


def test_hf_checkpoint_verifies_every_staged_replacement_was_emitted(tmp_path, monkeypatch) -> None:
    from safetensors.torch import save_file as real_save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
        glm52_fullparam_hf_name_mapping,
    )

    payload = _payload()
    mapping = glm52_fullparam_hf_name_mapping(payload)
    tensors = {
        hf_name: torch.zeros_like(unpack_glm52_payload_field(field))
        for item in payload.items
        for field in item.fields
        for hf_name in (mapping[item.target][field.name],)
    }
    base = tmp_path / "base"
    base.mkdir()
    real_save_safetensors(tensors, str(base / "model.safetensors"))
    omitted = mapping[payload.items[0].target][payload.items[0].fields[0].name]

    def dropping_save(state, filename, *args, **kwargs):
        state = dict(state)
        state.pop(omitted)
        return real_save_safetensors(state, filename, *args, **kwargs)

    monkeypatch.setattr("safetensors.torch.save_file", dropping_save)
    output = tmp_path / "output"
    with pytest.raises(Glm52FullParamPayloadError, match="omitted staged replacement"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(output),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )
    assert not output.exists()


def test_hf_checkpoint_receiver_is_atomic_on_shard_write_failure(tmp_path, monkeypatch) -> None:
    """An injected write failure leaves no partial checkpoint or guard commit."""

    from safetensors.torch import save_file as save_safetensors

    from xorl.server.weight_sync.glm52_fullparam_payload import (
        apply_glm52_fullparam_payload_to_hf_checkpoint,
    )

    payload = _payload()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    dense_weight = unpack_glm52_payload_field(payload.items[0].fields[0])
    dense_scale = unpack_glm52_payload_field(payload.items[0].fields[1])
    router_weight = unpack_glm52_payload_field(payload.items[1].fields[0])
    save_safetensors(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.zeros_like(dense_weight),
            "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.zeros_like(dense_scale),
            "model.layers.1.mlp.gate.weight": torch.zeros_like(router_weight),
        },
        str(base / "model.safetensors"),
    )
    mapping = {
        "model.layers.0.mlp.gate_proj": {
            "weight": "model.layers.0.mlp.gate_proj.weight",
            "weight_scale_inv": "model.layers.0.mlp.gate_proj.weight_scale_inv",
        },
        "model.layers.1.mlp.gate": {"weight": "model.layers.1.mlp.gate.weight"},
    }
    out = tmp_path / "updated"

    def failing_save(*args, **kwargs):
        raise OSError("injected shard-write failure")

    monkeypatch.setattr("safetensors.torch.save_file", failing_save)
    guard = _g()
    with pytest.raises(OSError, match="injected shard-write failure"):
        apply_glm52_fullparam_payload_to_hf_checkpoint(
            payload,
            str(base),
            str(out),
            mapping,
            expected_inventory=_expected_inventory(payload),
        )
    assert not out.exists(), "final directory must be absent after a failed materialization"
    leftovers = [entry for entry in os.listdir(tmp_path) if "staging" in entry]
    assert leftovers == [], f"staging directories were not cleaned up: {leftovers}"
    assert guard.last_step == -1, "version guard must not commit a failed materialization"

    # After the failure the receiver still works end to end (fresh process
    # state not required), proving the base was untouched.
    monkeypatch.undo()
    apply_glm52_fullparam_payload_to_hf_checkpoint(
        payload,
        str(base),
        str(out),
        mapping,
        expected_inventory=_expected_inventory(payload),
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# Checkpoint-form kinds: publish what the loader consumes
# ---------------------------------------------------------------------------


def _seeded_dense_mlp(hidden: int = 64, intermediate: int = 128):
    """A CPU dense composite with deterministic split-derived cache bytes."""

    from xorl.models.transformers.glm5.exact_fullparam_fp8 import Glm52FullParamDenseMLP

    module = Glm52FullParamDenseMLP(hidden, intermediate, device=torch.device("cpu"))
    for index, linear in enumerate((module.gate_up_proj, module.down_proj)):
        with torch.no_grad():
            linear.weight_master.zero_()
            raw = (
                torch.arange(linear.quantized_weight_f32.numel() * 4, dtype=torch.int64)
                .mul(index * 7 + 3)
                .remainder(251)
                .to(torch.uint8)
            )
            linear.quantized_weight_f32.copy_(raw.view(torch.float32).reshape(linear.quantized_weight_f32.shape))
            scale = torch.arange(linear.weight_scale_inv.numel(), dtype=torch.float32)
            linear.weight_scale_inv.copy_(scale.remainder(17).add(1).div(8).reshape(linear.weight_scale_inv.shape))
        linear._record_master_identity()
    return module


def _seeded_expert_bank(local_experts: int = 4, hidden: int = 128, intermediate: int = 128):
    """A CPU full-param bank with deterministic cache bytes and fresh identity."""

    from xorl.models.transformers.glm5.exact_fullparam_experts import (
        Glm52FullParamBlockFP8RoutedExperts,
    )

    bank = Glm52FullParamBlockFP8RoutedExperts(local_experts, hidden, intermediate, device=torch.device("cpu"))
    with torch.no_grad():
        bank.gate_up_weight_master.zero_()
        bank.down_weight_master.zero_()
        for name in ("gate_up_packed_weight_f32", "down_packed_weight_f32"):
            cache = getattr(bank, name)
            raw = (
                torch.arange(cache.numel() * 4, dtype=torch.int64)
                .mul(11 if "gate" in name else 13)
                .remainder(249)
                .to(torch.uint8)
            )
            cache.copy_(raw.view(torch.float32).reshape(cache.shape))
        for name in ("gate_up_weight_scale_inv", "down_weight_scale_inv"):
            scale = getattr(bank, name)
            values = torch.arange(scale.numel(), dtype=torch.float32)
            scale.copy_(values.remainder(23).add(1).div(16).reshape(scale.shape))
    bank._record_master_identity()
    return bank


def test_dense_mlp_checkpoint_item_round_trips_through_the_loader_fuse() -> None:
    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8DenseMLP

    module = _seeded_dense_mlp()
    payload = publish_glm52_fullparam_payload([("model.layers.0.mlp", module)], weight_version="v1", weight_step=1)
    item = payload.items[0]
    assert item.kind == "block_fp8_dense_mlp"
    assert [field.name for field in item.fields] == [
        "gate",
        "gate_scale_inv",
        "up",
        "up_scale_inv",
        "down",
        "down_scale_inv",
    ]

    # Published split fields byte-match the trainer's split views of the fused cache.
    for field, view in zip(item.fields, module.publishable_checkpoint_projections(), strict=True):
        assert torch.equal(
            unpack_glm52_payload_field(field).view(torch.uint8),
            view.contiguous().view(torch.uint8),
        )

    # The loader's real consumption (gate rows first, then up rows, scales in
    # the same order) reproduces the trainer's fused cache byte-for-byte.
    receiver = Glm52NativeBlockFP8DenseMLP(module.hidden_size, module.intermediate_size, device=torch.device("cpu"))
    apply_glm52_fullparam_payload(
        payload,
        lambda target, kind: receiver,
        expected_inventory=build_glm52_expected_payload_inventory(
            (
                (
                    "model.layers.0.mlp",
                    "block_fp8_dense_mlp",
                    GLM52_FULLPARAM_DENSE_MLP_CONTRACT_VERSION,
                    receiver,
                ),
            )
        ),
        version_guard=_g(),
    )
    assert torch.equal(
        receiver.gate_up_proj.fp8_weight().view(torch.uint8),
        module.gate_up_proj._cached_fp8_weight().view(torch.uint8),
    )
    assert torch.equal(receiver.gate_up_proj.weight_scale_inv.detach(), module.gate_up_proj.weight_scale_inv)
    assert torch.equal(
        receiver.down_proj.fp8_weight().view(torch.uint8),
        module.down_proj._cached_fp8_weight().view(torch.uint8),
    )
    assert torch.equal(receiver.down_proj.weight_scale_inv.detach(), module.down_proj.weight_scale_inv)

    # HF mapping helper covers the checkpoint-form fields mechanically.
    from xorl.server.weight_sync.glm52_fullparam_payload import glm52_fullparam_hf_name_mapping

    mapping = glm52_fullparam_hf_name_mapping(payload)
    assert mapping == {
        "model.layers.0.mlp": {
            "gate": "model.layers.0.mlp.gate_proj.weight",
            "gate_scale_inv": "model.layers.0.mlp.gate_proj.weight_scale_inv",
            "up": "model.layers.0.mlp.up_proj.weight",
            "up_scale_inv": "model.layers.0.mlp.up_proj.weight_scale_inv",
            "down": "model.layers.0.mlp.down_proj.weight",
            "down_scale_inv": "model.layers.0.mlp.down_proj.weight_scale_inv",
        }
    }


def test_expert_checkpoint_items_round_trip_through_the_production_pair_buffer() -> None:
    from xorl.models.transformers.glm5.exact_fullparam_experts import (
        GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION,
    )
    from xorl.models.transformers.glm5.native_fp8 import (
        Glm52NativeBlockFP8Experts,
        Glm52NativeExpertSlotReceiver,
        NativeBlockFP8ExpertPairBuffer,
    )

    bank = _seeded_expert_bank()
    with pytest.raises(RuntimeError, match="no assigned global expert range"):
        bank.checkpoint_publications()
    bank.assign_global_expert_range(4, 8)  # EP rank 1 of 2, experts 4..7
    with pytest.raises(RuntimeError, match="reassignment"):
        bank.assign_global_expert_range(0, 8)
    publications = bank.checkpoint_publications()
    assert [global_id for global_id, _ in publications] == [4, 5, 6, 7]

    payload = publish_glm52_fullparam_payload(
        [(f"model.layers.3.mlp.experts.{global_id}", unit) for global_id, unit in publications],
        weight_version="v1",
        weight_step=1,
    )
    assert [item.kind for item in payload.items] == ["block_fp8_expert"] * 4

    # Route the published checkpoint tensors through the PRODUCTION pair
    # buffer (the loader's real consumption): the assembled packed state must
    # equal the trainer bank's cache byte-for-byte.
    class _Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
            self.model.layers[3].mlp = torch.nn.Module()
            self.model.layers[3].mlp.experts = Glm52NativeBlockFP8Experts(8, 128, 128, device=torch.device("cpu"))

    container = _Container()
    # Stored bank must be EP-local for the pair buffer; shrink packed params.
    frozen = container.model.layers[3].mlp.experts
    for name in (
        "gate_up_packed_weight_f32",
        "gate_up_weight_scale_inv",
        "down_packed_weight_f32",
        "down_weight_scale_inv",
    ):
        parameter = getattr(frozen, name)
        setattr(
            frozen,
            name,
            torch.nn.Parameter(parameter.data[:4].clone(), requires_grad=False),
        )
    buffer = NativeBlockFP8ExpertPairBuffer(container, ep_rank=1, ep_size=2, num_experts=8)
    emitted: dict[str, torch.Tensor] = {}
    field_to_suffix = {
        "gate": "gate_proj.weight",
        "gate_scale_inv": "gate_proj.weight_scale_inv",
        "up": "up_proj.weight",
        "up_scale_inv": "up_proj.weight_scale_inv",
        "down": "down_proj.weight",
        "down_scale_inv": "down_proj.weight_scale_inv",
    }
    for item in payload.items:
        for field in item.fields:
            key = f"{item.target}.{field_to_suffix[field.name]}"
            result = buffer.try_consume(key, unpack_glm52_payload_field(field))
            assert result is not None, f"pair buffer rejected {key}"
            for target, value in result or []:
                emitted[target] = value
    buffer.validate_complete()
    assert torch.equal(
        emitted["model.layers.3.mlp.experts.gate_up_packed_weight_f32"].view(torch.uint8),
        bank.gate_up_packed_weight_f32.detach().view(torch.uint8),
    )
    assert torch.equal(
        emitted["model.layers.3.mlp.experts.gate_up_weight_scale_inv"],
        bank.gate_up_weight_scale_inv.detach(),
    )
    assert torch.equal(
        emitted["model.layers.3.mlp.experts.down_packed_weight_f32"].view(torch.uint8),
        bank.down_packed_weight_f32.detach().view(torch.uint8),
    )
    assert torch.equal(
        emitted["model.layers.3.mlp.experts.down_weight_scale_inv"],
        bank.down_weight_scale_inv.detach(),
    )

    # The in-memory slot receivers implement the same consumption per slot.
    serving = Glm52NativeBlockFP8Experts(4, 128, 128, device=torch.device("cpu"))
    with torch.no_grad():
        for name in (
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        ):
            getattr(serving, name).zero_()
    slot_receivers = {
        f"model.layers.3.mlp.experts.{global_id}": Glm52NativeExpertSlotReceiver(serving, global_id - 4)
        for global_id in range(4, 8)
    }
    apply_glm52_fullparam_payload(
        payload,
        lambda target, kind: slot_receivers[target],
        expected_inventory=build_glm52_expected_payload_inventory(
            tuple(
                (
                    target,
                    "block_fp8_expert",
                    GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION,
                    receiver,
                )
                for target, receiver in slot_receivers.items()
            )
        ),
        version_guard=_g(),
    )
    assert torch.equal(
        serving.gate_up_packed_weight_f32.detach().view(torch.uint8),
        bank.gate_up_packed_weight_f32.detach().view(torch.uint8),
    )
    assert torch.equal(serving.gate_up_weight_scale_inv.detach(), bank.gate_up_weight_scale_inv.detach())
    assert torch.equal(
        serving.down_packed_weight_f32.detach().view(torch.uint8),
        bank.down_packed_weight_f32.detach().view(torch.uint8),
    )
    assert torch.equal(serving.down_weight_scale_inv.detach(), bank.down_weight_scale_inv.detach())

    # Staleness is delegated to the owning bank at materialization time.
    with torch.no_grad():
        bank.gate_up_weight_master.add_(1.0)
    with pytest.raises(RuntimeError, match="stale"):
        publish_glm52_fullparam_payload(
            [(f"model.layers.3.mlp.experts.{gid}", unit) for gid, unit in publications],
            weight_version="v2",
            weight_step=2,
        )

    # The fused bank kind has no checkpoint form: the disk-route mapping refuses it.
    from xorl.server.weight_sync.glm52_fullparam_payload import glm52_fullparam_hf_name_mapping

    bank2 = _seeded_expert_bank()
    bank_payload = publish_glm52_fullparam_payload([("bank", bank2)], weight_version="v1", weight_step=1)
    with pytest.raises(Glm52FullParamPayloadError, match="no HF checkpoint form"):
        glm52_fullparam_hf_name_mapping(bank_payload)


def test_expert_global_range_validation_fails_closed() -> None:
    bank = _seeded_expert_bank()
    with pytest.raises(ValueError, match="positive multiple"):
        bank.assign_global_expert_range(0, 6)
    with pytest.raises(ValueError, match="not a valid EP-local block"):
        bank.assign_global_expert_range(2, 8)
    with pytest.raises(ValueError, match="not a valid EP-local block"):
        bank.assign_global_expert_range(8, 8)
    bank.assign_global_expert_range(0, 8)
    bank.assign_global_expert_range(0, 8)  # identical reassignment is idempotent
    assert bank.global_expert_ids == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Per-rank partial merge for step publication
# ---------------------------------------------------------------------------


def _expert_partial(expert_start: int, *, version: str = "step-9", step: int = 9):
    """One EP rank's expert-only partial payload (4 local experts)."""

    bank = _seeded_expert_bank()
    bank.assign_global_expert_range(expert_start, 8)
    return publish_glm52_fullparam_payload(
        [(f"model.layers.3.mlp.experts.{global_id}", unit) for global_id, unit in bank.checkpoint_publications()],
        weight_version=version,
        weight_step=step,
    )


def test_merge_combines_disjoint_rank_partials_and_reverifies() -> None:
    from xorl.server.weight_sync.glm52_fullparam_payload import merge_glm52_fullparam_payloads

    rank0 = publish_glm52_fullparam_payload(
        [("model.layers.1.mlp.gate", _seeded_router())],
        weight_version="step-9",
        weight_step=9,
    )
    merged = merge_glm52_fullparam_payloads([_expert_partial(4), rank0, _expert_partial(0)])

    assert merged.weight_version == "step-9" and merged.weight_step == 9
    # Deterministic order and complete coverage.
    assert [item.target for item in merged.items] == sorted(
        ["model.layers.1.mlp.gate"] + [f"model.layers.3.mlp.experts.{i}" for i in range(8)]
    )
    # Field checksums carried verbatim from the partials remain valid.
    verify_glm52_fullparam_payload(merged)
    # The merged manifest checksum is freshly minted and self-consistent
    # (verify above recomputed it); a byte flip is still caught.
    flipped = merged.items[0].fields[0].data.clone()
    flipped[0] ^= 0xFF
    tampered_field = (
        merged.items[0]
        .fields[0]
        .__class__(
            name=merged.items[0].fields[0].name,
            dtype=merged.items[0].fields[0].dtype,
            shape=merged.items[0].fields[0].shape,
            data=flipped,
            checksum=merged.items[0].fields[0].checksum,
        )
    )
    tampered_item = merged.items[0].__class__(
        target=merged.items[0].target,
        kind=merged.items[0].kind,
        contract_version=merged.items[0].contract_version,
        fields=(tampered_field,) + merged.items[0].fields[1:],
    )
    tampered = merged.__class__(
        protocol_version=merged.protocol_version,
        weight_version=merged.weight_version,
        weight_step=merged.weight_step,
        items=(tampered_item,) + merged.items[1:],
        manifest_checksum=merged.manifest_checksum,
    )
    with pytest.raises(Glm52FullParamPayloadError, match="Checksum mismatch"):
        verify_glm52_fullparam_payload(tampered)


def test_merge_fails_closed_on_mismatch_overlap_and_empty() -> None:
    from xorl.server.weight_sync.glm52_fullparam_payload import merge_glm52_fullparam_payloads

    with pytest.raises(Glm52FullParamPayloadError, match="at least one"):
        merge_glm52_fullparam_payloads([])

    with pytest.raises(Glm52FullParamPayloadError, match="mismatched weight_step"):
        merge_glm52_fullparam_payloads([_expert_partial(0, step=9), _expert_partial(4, step=8, version="step-9")])
    with pytest.raises(Glm52FullParamPayloadError, match="mismatched weight_version"):
        merge_glm52_fullparam_payloads([_expert_partial(0, version="step-9"), _expert_partial(4, version="other")])
    with pytest.raises(Glm52FullParamPayloadError, match="Duplicate payload target"):
        merge_glm52_fullparam_payloads([_expert_partial(0), _expert_partial(0)])
