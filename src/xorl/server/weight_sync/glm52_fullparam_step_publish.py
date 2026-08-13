"""Step-boundary refresh + payload publication for GLM-5.2 full-param training.

After every optimizer step the trainer refreshes each admitted component's
published cache so the next forward consumes post-step bytes. When a publish
root is configured, it also materializes one combined checksummed payload for
the step.

Distribution model:

- ``refresh_glm52_fullparam_caches`` runs on EVERY rank, every step (the
  staleness gates would otherwise refuse the next publication, and the
  trainer's own next forward would consume stale bytes).
- Expert items are rank-owned (globally-indexed targets, disjoint across EP
  ranks by the bank's assigned range) and are published by every owning rank.
- Dense composites and routers are replicated and byte-identical; rank 0
  publishes them.
- Every publishing rank saves its partial payload to
  ``{publish_root}/step_{step:06d}/rank_{rank:02d}`` on the SHARED
  filesystem; after a collective error rendezvous, rank 0 loads every partial back (full
  checksum re-verification), merges them into one manifest, validates
  completeness against the model's own enumeration expectation, saves the
  combined payload to ``.../combined``, and writes a completion marker. A
  second error rendezvous keeps the gang in lockstep and propagates
  rank-0 merge/write failures to every peer instead of stranding them.

Fail closed everywhere: a missing rank contribution, an unexpectedly present
step directory, a duplicate target, or an incomplete expert cover RAISES —
the optimizer step's API response then carries the error instead of a
  successful result, and nothing downstream can consume a partial step.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable

import torch.distributed as dist
from torch import nn

from xorl.server.weight_sync.glm52_fullparam_payload import (
    Glm52FullParamPayloadError,
    load_glm52_fullparam_payload,
    merge_glm52_fullparam_payloads,
    publish_glm52_fullparam_payload,
    save_glm52_fullparam_payload,
)


logger = logging.getLogger(__name__)

_COMBINED_DIRNAME = "combined"
_COMMIT_MARKER_FILENAME = "COMMITTED.json"
_EXPERT_KIND = "block_fp8_expert"


def _rendezvous_phase(
    phase: str,
    error: Exception | None,
    *,
    barrier: Callable[[], None] | None,
) -> None:
    """Make every rank finish a fallible phase before any rank can continue.

    Production uses ``all_gather_object`` over bounded error strings. Every
    local exception is caught before this call, so a nonzero-rank write error
    or rank-0 merge error makes all peers raise at the same rendezvous rather
    than deadlocking at a later raw barrier. ``barrier`` is only a deterministic
    single-process test seam; it is never selected by production callers.
    """

    if barrier is not None:
        barrier()
        if error is not None:
            raise error
        return

    if dist.is_available() and dist.is_initialized():
        local = None if error is None else f"{type(error).__name__}: {str(error)[:512]}"
        gathered: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(gathered) if message]
        if failures:
            raise Glm52FullParamPayloadError(f"GLM-5.2 full-param {phase} failed collectively: " + "; ".join(failures))
        return

    if error is not None:
        raise error


def _expected_global_targets(model: nn.Module) -> set[str]:
    """The complete target set a full (all-rank) publication must cover.

    Non-expert targets come from this rank's own enumeration (they are
    replicated, so every rank agrees).  Expert targets are derived from each
    trainable bank's DECLARED global range: ``{bank_fqn}.{global_id}`` for
    every global expert id — which is exactly what the union of all EP
    owners' enumerations produces when no contribution is missing.
    """

    from xorl.models.transformers.glm5.exact_fullparam_admission import (  # noqa: PLC0415
        iter_glm52_fullparam_publication,
    )
    from xorl.models.transformers.glm5.exact_fullparam_experts import (  # noqa: PLC0415
        Glm52FullParamBlockFP8RoutedExperts,
    )

    expected: set[str] = set()
    for target, module in iter_glm52_fullparam_publication(model):
        if getattr(module, "glm52_fullparam_payload_kind", None) != _EXPERT_KIND:
            expected.add(target)
    for name, module in model.named_modules():
        if isinstance(module, Glm52FullParamBlockFP8RoutedExperts):
            expected.update(f"{name}.{global_id}" for global_id in range(module.num_global_experts))
    return expected


def _write_commit_marker(step_dir: str, result: dict) -> None:
    """Atomically and durably expose the marker after all payload bytes."""

    marker_path = os.path.join(step_dir, _COMMIT_MARKER_FILENAME)
    if os.path.lexists(marker_path):
        raise Glm52FullParamPayloadError(f"Step commit marker {marker_path!r} already exists; refusing to overwrite it")
    directory_descriptor = os.open(
        step_dir,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    staging_path = ""
    marker_visible = False
    try:
        descriptor, staging_path = tempfile.mkstemp(prefix=f".{_COMMIT_MARKER_FILENAME}.", suffix=".tmp", dir=step_dir)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging_path, marker_path)
        marker_visible = True
        try:
            os.fsync(directory_descriptor)
        except BaseException as durability_error:
            # Do not leave a live consumer-visible commit marker after
            # reporting failure. Remove it and persist that removal before
            # propagating the error into the fatal post-mutation path.
            try:
                os.unlink(marker_path)
                marker_visible = False
                os.fsync(directory_descriptor)
            except BaseException as cleanup_error:
                raise Glm52FullParamPayloadError(
                    "Step commit marker became visible but its directory fsync failed, and cleanup could not "
                    "be durably confirmed; publication state is UNKNOWN and must not be consumed"
                ) from cleanup_error
            raise Glm52FullParamPayloadError(
                "Step commit marker directory fsync failed; the visible marker was removed and publication "
                "was not committed"
            ) from durability_error
    except BaseException:
        if not marker_visible:
            try:
                if staging_path:
                    os.unlink(staging_path)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(directory_descriptor)


def glm52_fullparam_step_boundary(
    model: nn.Module,
    *,
    step: int,
    publish_root: str | None,
    rank: int,
    world_size: int,
    version_tag: str = "glm52-fullparam",
    barrier: Callable[[], None] | None = None,
) -> dict:
    """Refresh caches on every rank; publish one combined payload when configured.

    Returns a JSON-serializable publication result. ``step`` must be the post-step
    global step (it becomes the payload's monotonic ``weight_step``).
    """

    from xorl.models.transformers.glm5.exact_fullparam_admission import (  # noqa: PLC0415
        iter_glm52_fullparam_publication,
        refresh_glm52_fullparam_caches,
    )

    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise Glm52FullParamPayloadError(
            f"step-boundary publication requires a positive post-step integer step, got {step!r}"
        )
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise Glm52FullParamPayloadError(f"world_size must be a positive integer, got {world_size!r}")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise Glm52FullParamPayloadError(f"rank must be in [0, {world_size}), got {rank!r}")
    if dist.is_available() and dist.is_initialized():
        actual = (dist.get_rank(), dist.get_world_size())
        if actual != (rank, world_size):
            raise Glm52FullParamPayloadError(
                f"publication rank/world_size {(rank, world_size)} disagrees with distributed state {actual}"
            )
    receipt: dict = {"published": False, "step": step, "rank": rank}
    weight_version = f"{version_tag}-step-{step}"
    step_dir = os.path.join(publish_root, f"step_{step:06d}") if publish_root is not None else ""
    contributed = False
    local_error: Exception | None = None
    try:
        refreshed = refresh_glm52_fullparam_caches(model)
        receipt["refreshed_components"] = refreshed
        if publish_root is not None:
            rank_dir = os.path.join(step_dir, f"rank_{rank:02d}")
            if os.path.exists(rank_dir):
                raise Glm52FullParamPayloadError(
                    f"Publication directory {rank_dir!r} already exists; refusing to overwrite a prior step artifact"
                )

            components = iter_glm52_fullparam_publication(model)
            if rank != 0:
                components = [
                    (target, module)
                    for target, module in components
                    if getattr(module, "glm52_fullparam_payload_kind", None) == _EXPERT_KIND
                ]
            if components:
                partial = publish_glm52_fullparam_payload(components, weight_version=weight_version, weight_step=step)
                save_glm52_fullparam_payload(partial, rank_dir)
                contributed = True
            receipt["contributed_items"] = len(components)
    except Exception as exc:  # every rank must still enter the rendezvous
        local_error = exc

    _rendezvous_phase("local refresh/publication", local_error, barrier=barrier)
    if publish_root is None:
        return receipt

    merge_error: Exception | None = None
    try:
        if rank == 0:
            if not contributed:
                raise Glm52FullParamPayloadError(
                    "rank 0 published no components; the admission guarantees at least dense/router items"
                )
            expected_rank_entries = [f"rank_{owner:02d}" for owner in range(world_size)]
            actual_rank_entries = sorted(
                entry
                for entry in os.listdir(step_dir)
                if entry.startswith("rank_") and os.path.isdir(os.path.join(step_dir, entry))
            )
            if actual_rank_entries != expected_rank_entries:
                raise Glm52FullParamPayloadError(
                    "Step publication requires exactly one contribution directory per rank; "
                    f"expected={expected_rank_entries}, got={actual_rank_entries}"
                )
            rank_dirs = [os.path.join(step_dir, entry) for entry in expected_rank_entries]
            partials = [load_glm52_fullparam_payload(directory) for directory in rank_dirs]
            combined = merge_glm52_fullparam_payloads(partials)

            expected_targets = _expected_global_targets(model)
            actual_targets = {item.target for item in combined.items}
            if actual_targets != expected_targets:
                missing = sorted(expected_targets - actual_targets)
                surplus = sorted(actual_targets - expected_targets)
                raise Glm52FullParamPayloadError(
                    "Combined step payload does not cover the model's declared publication scope; "
                    f"missing={missing[:8]} (of {len(missing)}), unexpected={surplus[:8]} (of {len(surplus)}) "
                    "— a rank contribution is missing or foreign artifacts entered the step directory"
                )

            combined_dir = os.path.join(step_dir, _COMBINED_DIRNAME)
            save_glm52_fullparam_payload(combined, combined_dir)
            receipt.update(
                {
                    "published": True,
                    "weight_version": weight_version,
                    "weight_step": step,
                    "step_dir": step_dir,
                    "combined_dir": combined_dir,
                    "manifest_checksum": combined.manifest_checksum,
                    "items": len(combined.items),
                    "contributing_ranks": len(partials),
                    "world_size": world_size,
                }
            )
            _write_commit_marker(step_dir, receipt)
            logger.info(
                "GLM-5.2 full-param step publication complete: step=%d items=%d ranks=%d manifest=%s dir=%s",
                step,
                len(combined.items),
                len(partials),
                combined.manifest_checksum[:16],
                combined_dir,
            )
        else:
            receipt.update({"published": True, "weight_version": weight_version, "step_dir": step_dir})
    except Exception as exc:  # nonzero ranks wait here and receive rank-0 failure
        merge_error = exc

    _rendezvous_phase("combined publication", merge_error, barrier=barrier)
    return receipt


__all__ = ["glm52_fullparam_step_boundary"]
