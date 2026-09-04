"""Fold-aware sparse-delta weight sync for the ZORL parameter server.

A full push unshards, exports, and re-quantizes every synchronized parameter
even when a fold touches only a subset and changes relatively few quantized
bytes. This module limits that work to fold-touched parameters and transmits
their encoded byte differences.

This module implements the fold-aware fast path:

1. The ZORL fresh_ab fold records exactly which base params it touched
   (``ModelRunner._apply_zorl_fresh_ab_base_update`` — MoE experts
   gate_up_proj/down_proj + LoRA-targeted dense weights). Nothing else
   changes after the first full sync, so the sync param set is restricted
   to the fold-touched params.
2. MoE expert params are either ordinary FSDP2 ``Shard(0)`` DTensors, or
   EP-local tensors whose explicit eFSDP placement is ``Shard(1)`` while
   their expert-dimension ownership is carried separately by ``SpecInfo``.
   The latter are gathered only inside their eFSDP group; one group leader
   emits the group's contiguous expert slab. The resulting slabs are
   quantized, byte-diffed against the previously-shipped state, and
   escape-encoded without a per-expert Python loop.
3. Rank 0 concatenates the per-rank encodings (expert slabs are disjoint,
   ascending index ranges, so streams merge with one boundary-gap fixup),
   writes a single ``delta_packed_v1`` file to shared storage, and POSTs it
   through the existing ``/update_weights_from_sparse_delta`` receiver
   path (fused-MoE names — one entry per (layer, projection) instead of
   one per expert; the sglang receiver gained vectorized fused-MoE
   locators for these names).

Bit-correctness contract: the receiver's served bytes after applying the
delta equal what the full push would have produced. To guarantee this the
fast path reuses the exact quantization kernels/policies of the full path
(``WeightSyncHandler._quantize_fp8_stack`` with the same
``XORL_P2P_FP8_QUANTIZE_DEVICE`` execution-device policy — CPU and GPU
quantization are NOT bit-identical, so the mode must match the full-path
mode used to establish the baseline), and the baseline is only advanced
after every endpoint confirms (absolute-value scatters are idempotent, so
a failed step is re-sent cumulatively).

LoRA-merge note: the full path ships ``base + lora_delta`` merged weights.
Under ZORL fresh_ab the parent adapter's B factor is required to be zero
(the fold applies to the base), so ``lora_delta`` is exactly ``+0.0``
everywhere; the fast path mirrors the merge with a scalar ``+ 0.0`` (which
reproduces the reference's ``-0.0 -> +0.0`` flips) and verifies the
zero-B invariant each sync (falls back to the full path otherwise).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)

try:
    from torch.distributed._tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from torch.distributed.tensor.placement_types import Shard

    _HAS_DTENSOR = True
except ImportError:  # pragma: no cover - CPU-only test environments
    DTensor = None  # type: ignore[assignment]
    compute_local_shape_and_global_offset = None  # type: ignore[assignment]
    Shard = None  # type: ignore[assignment]
    _HAS_DTENSOR = False


class FoldSparseDeltaUnsupported(RuntimeError):
    """Raised when the fold-aware fast path cannot run for this sync.

    The caller (WeightSyncHandler) treats this as "fall back to the full
    sync path" — never as a hard failure.
    """


def fold_sparse_delta_enabled(sparse_delta_config: Optional[Dict[str, Any]]) -> bool:
    if sparse_delta_config and "fold_aware" in sparse_delta_config:
        return bool(sparse_delta_config.get("fold_aware"))
    return os.environ.get("XORL_SPARSE_DELTA_FOLD_AWARE", "0").strip().lower() in {"1", "true", "yes", "on"}


def fold_sparse_delta_fallback_method(sparse_delta_config: Optional[Dict[str, Any]]) -> str:
    if sparse_delta_config and sparse_delta_config.get("fallback_sync_method"):
        return str(sparse_delta_config["fallback_sync_method"])
    return os.environ.get("XORL_SPARSE_DELTA_FOLD_FALLBACK_METHOD", "p2p").strip() or "p2p"


def fold_baseline_device() -> str:
    """Where the previously-shipped quantized bytes are cached.

    ``cpu`` (default): stores the baseline in host RAM and pays a pageable D2H
    copy in GPU quantize mode.
    ``cuda``: keeps the baseline on the GPU and avoids those baseline copies,
    at the cost of persistent device memory plus a transient staged copy.
    """
    return os.environ.get("XORL_SPARSE_DELTA_FOLD_BASELINE_DEVICE", "cpu").strip().lower()


# ============================================================================
# Sync plan
# ============================================================================


@dataclass
class ExpertParamPlan:
    """One fold-touched fused MoE expert param (3D, ``[E, K, N]`` training layout)."""

    train_name: str  # cleaned training param name, e.g. model.layers.3.mlp.experts.gate_up_proj
    param: torch.nn.Parameter
    hf_prefix: str  # e.g. model.layers.3.mlp.experts
    kind: str  # "gate_up" | "down"
    lora_merge_halves: Tuple[bool, bool]  # mirror + 0.0 on (first, second) last-dim half
    lora_b_params: Tuple[torch.nn.Parameter, ...] = ()  # zero-B invariant check
    # EP+eFSDP stores only this rank's EP-owned experts in the live model and
    # represents the remaining eFSDP sharding as DTensor Shard(1).  The EP
    # dimension is therefore implicit in ``param.shape``.  These fields make
    # the receiver-global expert coordinates explicit for sparse transport.
    implicit_ep: bool = False
    expert_offset: int = 0
    expert_total: Optional[int] = None
    transport_owner: bool = True

    @property
    def weight_name(self) -> str:
        return f"{self.hf_prefix}.{'gate_up_proj' if self.kind == 'gate_up' else 'down_proj'}"

    @property
    def scale_name(self) -> str:
        return f"{self.weight_name}_scale_inv"


@dataclass
class DenseParamPlan:
    """One fold-touched dense 2D weight (e.g. shared_expert projections)."""

    train_name: str  # e.g. model.layers.3.mlp.shared_expert.down_proj.weight
    param: torch.nn.Parameter
    lora_merge: bool
    lora_b_params: Tuple[torch.nn.Parameter, ...] = ()

    @property
    def sort_key(self) -> str:
        return self.train_name


@dataclass
class FoldSyncPlan:
    expert_params: List[ExpertParamPlan] = field(default_factory=list)
    dense_params: List[DenseParamPlan] = field(default_factory=list)
    fold_index: int = -1
    model_id: Optional[str] = None

    @property
    def param_names(self) -> List[str]:
        return [p.train_name for p in self.expert_params] + [p.train_name for p in self.dense_params]


def _clean_param_name(name: str) -> str:
    # Mirror of ModelRunner._resolve_zorl_fresh_ab_base_params name cleaning.
    return name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")


def _strip_orig_mod(name: str) -> str:
    if "_orig_mod" not in name:
        return name
    return ".".join(part for part in name.split(".") if part != "_orig_mod")


def build_fold_sync_plan(
    model: torch.nn.Module,
    fold_param_names: Sequence[str],
    *,
    fold_index: int = -1,
    model_id: Optional[str] = None,
    skip_param_fn: Optional[Callable[[str], bool]] = None,
) -> FoldSyncPlan:
    """Map fold-touched training param names onto a sync plan.

    ``fold_param_names`` are the names recorded by the fresh_ab fold
    (cleaned of FSDP/compile wrappers). Raises
    :class:`FoldSparseDeltaUnsupported` for anything the fast path cannot
    represent so the caller can fall back to the full sync.
    """
    from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415
    from xorl.models.layers.moe.lora import MoEExpertsLoRA  # noqa: PLC0415

    named_params: Dict[str, torch.nn.Parameter] = {}
    for raw_name, param in model.named_parameters():
        named_params[_clean_param_name(raw_name)] = param
    named_modules: Dict[str, torch.nn.Module] = {}
    for raw_name, module in model.named_modules():
        named_modules[_clean_param_name(raw_name)] = module

    spec_info_by_name = {
        _clean_param_name(raw_name): spec for raw_name, spec in (getattr(model, "_fqn2spec_info", None) or {}).items()
    }

    plan = FoldSyncPlan(fold_index=fold_index, model_id=model_id)
    for name in sorted(fold_param_names):
        if skip_param_fn is not None and skip_param_fn(name):
            logger.info("[FoldSparseDelta] skipping fold param %s (skip pattern)", name)
            continue
        param = named_params.get(name)
        if param is None:
            raise FoldSparseDeltaUnsupported(f"fold-touched param {name!r} not found on the model")

        parent_name, _, leaf = name.rpartition(".")
        parent = named_modules.get(parent_name)
        if (
            param.data.ndim == 3
            and isinstance(parent, (MoEExperts, MoEExpertsLoRA))
            and leaf in ("gate_up_proj", "down_proj")
        ):
            if not getattr(parent, "gated", True):
                raise FoldSparseDeltaUnsupported(f"non-gated MoE experts unsupported: {name!r}")
            kind = "gate_up" if leaf == "gate_up_proj" else "down"
            merge_halves = (False, False)
            lora_b: Tuple[torch.nn.Parameter, ...] = ()
            if isinstance(parent, MoEExpertsLoRA):
                targets = parent.lora_config.target_modules or []
                if kind == "gate_up":
                    merge_halves = ("gate_proj" in targets, "up_proj" in targets)
                    lora_b = tuple(
                        getattr(parent, f"{proj}_lora_B")
                        for proj, m in zip(("gate_proj", "up_proj"), merge_halves)
                        if m
                    )
                else:
                    merged = "down_proj" in targets
                    merge_halves = (merged, merged)
                    if merged:
                        lora_b = (parent.down_proj_lora_B,)
            implicit_ep, expert_offset, expert_total, transport_owner = _implicit_ep_expert_layout(
                param.data,
                spec_info_by_name.get(name) or getattr(param, "spec_info", None),
            )
            plan.expert_params.append(
                ExpertParamPlan(
                    train_name=name,
                    param=param,
                    hf_prefix=_strip_orig_mod(parent_name),
                    kind=kind,
                    lora_merge_halves=merge_halves,
                    lora_b_params=lora_b,
                    implicit_ep=implicit_ep,
                    expert_offset=expert_offset,
                    expert_total=expert_total,
                    transport_owner=transport_owner,
                )
            )
        elif param.data.ndim == 2:
            lora_merge = False
            lora_b: Tuple[torch.nn.Parameter, ...] = ()
            if leaf == "weight" and parent is not None:
                # LoraLinear import kept light: duck-type on get_delta_weight.
                lora_merge = hasattr(parent, "get_delta_weight") and hasattr(parent, "lora_B")
                if lora_merge:
                    lora_b = (parent.lora_B,)
            plan.dense_params.append(
                DenseParamPlan(train_name=name, param=param, lora_merge=lora_merge, lora_b_params=lora_b)
            )
        else:
            raise FoldSparseDeltaUnsupported(
                f"fold-touched param {name!r} has unsupported layout ndim={param.data.ndim}"
            )
    return plan


# ============================================================================
# Local slab access
# ============================================================================


def _implicit_ep_expert_layout(data: Any, spec_info: Any) -> Tuple[bool, int, Optional[int], bool]:
    """Resolve receiver-global coordinates for EP-local ``Shard(1)`` experts.

    ``ParallelPlan`` first slices expert dim 0 across the EP mesh, then FSDP2
    wraps that EP-local tensor on the orthogonal ``ep_fsdp`` mesh.  The live
    DTensor consequently exposes only ``Shard(1)``; its shape's dim 0 is the
    number of experts owned by one EP rank, not the receiver-global count.
    """
    if not (_HAS_DTENSOR and isinstance(data, DTensor)) or spec_info is None:
        return False, 0, None, True
    ep_placement = getattr(spec_info, "placement", None)
    if not isinstance(ep_placement, Shard) or int(ep_placement.dim) != 0:
        return False, 0, None, True
    placements = tuple(data.placements)
    shard_dims = [int(p.dim) for p in placements if isinstance(p, Shard)]
    if shard_dims not in ([1], []):
        return False, 0, None, True
    mesh = getattr(spec_info, "ep_fsdp_mesh", None)
    if mesh is None:
        raise FoldSparseDeltaUnsupported("implicit EP expert layout is missing its ep_fsdp mesh")
    ep_size = int(mesh["ep"].size())
    ep_rank = int(mesh.get_local_rank("ep"))
    ep_fsdp_rank = int(mesh.get_local_rank("ep_fsdp"))
    local_experts = int(data.shape[0])
    return True, ep_rank * local_experts, ep_size * local_experts, ep_fsdp_rank == 0


def expert_slab(entry: ExpertParamPlan) -> Tuple[torch.Tensor, int, int]:
    """Return the full matrix slab for this plan entry and global E coordinates."""
    if not entry.implicit_ep:
        return local_expert_slab(entry.param)
    data = entry.param.data
    if not (_HAS_DTENSOR and isinstance(data, DTensor)):
        raise FoldSparseDeltaUnsupported("implicit EP expert param is not a DTensor")
    placements = tuple(data.placements)
    shard_dims = [int(p.dim) for p in placements if isinstance(p, Shard)]
    if any(not isinstance(p, Shard) for p in placements if not _is_replicate(p)):
        raise FoldSparseDeltaUnsupported(f"unsupported placements for implicit EP expert param: {placements}")
    if shard_dims not in ([1], []):
        raise FoldSparseDeltaUnsupported(
            f"implicit EP expert param must be Shard(1) or replicated, got placements={placements}"
        )
    full = data.full_tensor()
    if tuple(full.shape) != tuple(data.shape):
        raise FoldSparseDeltaUnsupported(
            f"implicit EP expert gather shape mismatch: full={tuple(full.shape)} expected={tuple(data.shape)}"
        )
    expert_total = entry.expert_total
    if expert_total is None:
        raise FoldSparseDeltaUnsupported("implicit EP expert layout has no receiver-global expert count")
    if entry.expert_offset + int(full.shape[0]) > int(expert_total):
        raise FoldSparseDeltaUnsupported(
            f"implicit EP expert slab {tuple(full.shape)}@{entry.expert_offset} exceeds total {expert_total}"
        )
    return full, int(entry.expert_offset), int(expert_total)


def local_expert_slab(param: torch.nn.Parameter) -> Tuple[torch.Tensor, int, int]:
    """Return (local_slab [E_local, K, N], first_global_expert, E_total)."""
    data = param.data
    if _HAS_DTENSOR and isinstance(data, DTensor):
        placements = tuple(data.placements)
        shard_dims = [p.dim for p in placements if isinstance(p, Shard)]
        if any(not isinstance(p, Shard) for p in placements if not _is_replicate(p)):
            raise FoldSparseDeltaUnsupported(f"unsupported placements for expert param: {placements}")
        if shard_dims not in ([0], []):
            raise FoldSparseDeltaUnsupported(
                f"expert param must be Shard(0) or replicated, got placements={placements}"
            )
        local = data.to_local()
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tuple(data.shape), data.device_mesh, placements
        )
        if tuple(local.shape) != tuple(local_shape):
            raise FoldSparseDeltaUnsupported(
                f"padded/uneven FSDP shard unsupported: local={tuple(local.shape)} expected={tuple(local_shape)}"
            )
        return local, int(global_offset[0]), int(data.shape[0])
    return data, 0, int(data.shape[0])


def _is_replicate(placement: Any) -> bool:
    return placement.__class__.__name__ == "Replicate"


def full_dense_tensor(param: torch.nn.Parameter) -> torch.Tensor:
    """All ranks participate; returns the full (unsharded) tensor."""
    data = param.data
    if _HAS_DTENSOR and isinstance(data, DTensor):
        return data.full_tensor()
    return data


# ============================================================================
# Byte diff + escape encoding
# ============================================================================


def exact_byte_diff(
    current: torch.Tensor,
    baseline: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bitwise diff of two same-shape tensors.

    Returns (flat_indices int64 CPU, changed element VALUES from ``current``
    as a same-width RAW-BIT integer view — fp8 tensors do not pickle across
    ``dist.gather_object`` and bit views also keep ``-0.0``/NaN payloads
    intact; the real dtype travels in the segment metadata and is restored
    with ``.view(dtype)`` at pack time). ``baseline=None`` = all changed.
    """
    flat = current.reshape(-1)
    if baseline is None:
        idx = torch.arange(flat.numel(), dtype=torch.int64, device=flat.device)
        return idx, _raw_bits_view(flat).clone()

    if baseline.dtype != current.dtype or baseline.numel() != current.numel():
        raise FoldSparseDeltaUnsupported(
            f"baseline layout changed: {baseline.dtype}/{baseline.numel()} vs {current.dtype}/{current.numel()}"
        )
    elem = flat.element_size()
    if elem == 1:
        changed = flat.view(torch.uint8) != baseline.reshape(-1).view(torch.uint8)
    elif elem == 4:
        changed = flat.view(torch.int32) != baseline.reshape(-1).view(torch.int32)
    elif elem == 2:
        changed = flat.view(torch.int16) != baseline.reshape(-1).view(torch.int16)
    else:  # pragma: no cover
        cb = flat.view(torch.uint8).reshape(flat.numel(), elem)
        bb = baseline.reshape(-1).view(torch.uint8).reshape(flat.numel(), elem)
        changed = torch.any(cb != bb, dim=1)
    idx = changed.nonzero(as_tuple=False).flatten().to(torch.int64)
    values = _select_by_bytes(flat, idx)
    return idx, values


def _raw_bits_view(flat: torch.Tensor) -> torch.Tensor:
    """Same-width integer bit view (fp8-safe for pickling, sign-of-zero exact)."""
    elem = flat.element_size()
    if elem == 1:
        return flat.view(torch.uint8)
    if elem == 2:
        return flat.view(torch.int16)
    if elem == 4:
        return flat.view(torch.int32)
    return flat  # pragma: no cover


def _select_by_bytes(flat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """index_select on the raw-bit view (fp8/bf16-safe, pickle-safe)."""
    return _raw_bits_view(flat).index_select(0, idx)


_ENCODE_FN: Optional[Callable[..., Any]] = None


def load_delta_encoding_encode(
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: Optional[bool] = None,
) -> Callable[..., Any]:
    """Load (once) ``delta_encoding.encoding.compression.encode``.

    The escape-continuation wire format belongs to the ``delta-encoding``
    library — we call its encoder (C++ native extension when enabled via
    ``XORL_DELTA_ENCODING_USE_NATIVE_EXTENSION``) instead of reimplementing
    it, so sender bytes are canonically what the receiver's decoder expects.
    """
    global _ENCODE_FN
    if _ENCODE_FN is not None:
        return _ENCODE_FN
    from xorl.server.weight_sync.sparse_delta_files import prepare_delta_encoding_runtime  # noqa: PLC0415

    if delta_encoding_path is None:
        delta_encoding_path = os.environ.get("XORL_DELTA_ENCODING_PATH") or None
    if use_native_extension is None:
        use_native_extension = os.environ.get("XORL_DELTA_ENCODING_USE_NATIVE_EXTENSION", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    prepare_delta_encoding_runtime(
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )
    import importlib  # noqa: PLC0415

    compression = importlib.import_module("delta_encoding.encoding.compression")
    _ENCODE_FN = compression.encode
    return _ENCODE_FN


def escape_encode_sorted(indices: torch.Tensor, *, first_gap_from: int = 0) -> torch.Tensor:
    """Escape-continuation encode SORTED int64 indices via ``delta-encoding``.

    ``first_gap_from`` sets the reference point of the first gap so per-rank
    segments can be concatenated by the merger: gaps are shift-invariant, so
    encoding ``indices - first_gap_from`` makes the library's from-0 first
    gap equal ``indices[0] - first_gap_from`` while later gaps are unchanged.
    """
    nnz = indices.numel()
    if nnz == 0:
        return torch.empty(0, dtype=torch.uint8)
    encode_fn = load_delta_encoding_encode()
    shifted = indices.to(torch.int64)
    if first_gap_from:
        shifted = shifted - int(first_gap_from)
    if int(shifted[0]) < 0:
        raise ValueError("escape_encode_sorted requires indices >= first_gap_from")
    encoded = encode_fn(
        shifted,
        torch.empty(0, dtype=torch.uint8),
        (int(shifted[-1]) + 1,),
    )
    return encoded.flat_deltas


@dataclass
class RankEntrySegment:
    """One rank's contribution to one packed tensor entry."""

    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    nnz: int
    first_index: int  # global flat index of the first changed element (-1 if empty)
    last_index: int
    tail_deltas: torch.Tensor  # escape stream for indices[1:] (empty when nnz<=1)
    values: torch.Tensor  # raw-bit values (uint8/int16/int32 views), CPU

    @staticmethod
    def from_indices(
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> "RankEntrySegment":
        """``indices``/``values`` may live on GPU (diff device); the encoded
        segment is always stored on CPU (it crosses ``dist.gather_object``)."""
        nnz = int(indices.numel())
        if nnz == 0:
            return RankEntrySegment(name, shape, dtype, 0, -1, -1, torch.empty(0, dtype=torch.uint8), values.cpu())
        first = int(indices[0])
        last = int(indices[-1])
        tail = escape_encode_sorted(indices[1:], first_gap_from=first) if nnz > 1 else torch.empty(0, dtype=torch.uint8)
        return RankEntrySegment(name, shape, dtype, nnz, first, last, tail.cpu(), values.cpu())


def merge_rank_segments(segments: Sequence[RankEntrySegment]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge per-rank segments (ascending, disjoint index ranges) into one
    (flat_deltas, values) escape stream pair."""
    segs = [s for s in segments if s.nnz > 0]
    if not segs:
        return torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=segments[0].dtype if segments else torch.uint8)
    segs.sort(key=lambda s: s.first_index)
    prev_last = 0
    delta_chunks: List[torch.Tensor] = []
    value_chunks: List[torch.Tensor] = []
    for pos, seg in enumerate(segs):
        if pos > 0 and seg.first_index <= prev_last:
            raise ValueError(f"overlapping rank segments for {seg.name!r}")
        boundary_gap = seg.first_index - prev_last
        boundary = escape_encode_sorted(torch.tensor([seg.first_index], dtype=torch.int64), first_gap_from=prev_last)
        assert boundary.numel() == boundary_gap // 255 + 1
        delta_chunks.append(boundary)
        if seg.tail_deltas.numel():
            delta_chunks.append(seg.tail_deltas)
        value_chunks.append(seg.values)
        prev_last = seg.last_index
    return torch.cat(delta_chunks), torch.cat(value_chunks)


# ============================================================================
# Baseline store
# ============================================================================


class FoldBaselineStore:
    """Per-process cache of the previously-shipped quantized bytes.

    Keyed by (scope, train param name). Values are CPU tensors:
      {"weight": fp8 stack (local layout), "scale": fp32 stack,
       "expert_offset": int, "expert_count": int}
    Dense entries live under ("dense", hf_name) on rank 0 only.

    ``stage`` records the new state without publishing it; ``commit``
    publishes every staged entry (call only after every receiver confirmed),
    ``rollback`` drops the staged state so the next sync re-diffs against
    the still-valid old baseline (cumulative resend).
    """

    _stores: Dict[str, "FoldBaselineStore"] = {}

    def __init__(self) -> None:
        self._committed: Dict[Any, Dict[str, Any]] = {}
        self._staged: Dict[Any, Dict[str, Any]] = {}
        # Delivery-state flags (not part of the staged/committed baseline):
        # last_attempt_failed => the next delta widens the
        # base_weight_version precondition to the committed version PLUS
        # every version a failed attempt may have stamped (endpoints that
        # already applied a failed attempt sit on the NEW version; the
        # cumulative absolute-value resend converges them all — but a
        # receiver on NEITHER version, e.g. a scorer that restarted onto
        # checkpoint weights during the failure window, must still 400 into
        # the force_prime path instead of silently applying a delta against
        # bytes it never held). force_prime => a receiver reported a
        # base-version mismatch; the next sync must re-prime + full-push
        # instead of diffing.
        self.last_attempt_failed: bool = False
        self.force_prime: bool = False
        # Versions failed attempts may have stamped on a subset of receivers
        # since the last commit.
        self.failed_weight_versions: List[str] = []

    @classmethod
    def for_scope(cls, scope: str) -> "FoldBaselineStore":
        store = cls._stores.get(scope)
        if store is None:
            store = cls()
            cls._stores[scope] = store
        return store

    @classmethod
    def reset_all(cls) -> None:
        cls._stores.clear()

    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        return self._committed.get(key)

    def has(self, key: Any) -> bool:
        return key in self._committed

    def stage(self, key: Any, entry: Dict[str, Any]) -> None:
        self._staged[key] = entry

    def commit(self) -> int:
        count = len(self._staged)
        self._committed.update(self._staged)
        self._staged.clear()
        # A commit means every receiver confirmed this state; clear the
        # delivery-failure flags.
        self.last_attempt_failed = False
        self.force_prime = False
        self.failed_weight_versions = []
        return count

    def note_failed_weight_version(self, weight_version: Optional[str]) -> None:
        """Record a version a failed sync may have stamped on some receivers."""
        if weight_version and weight_version not in self.failed_weight_versions:
            self.failed_weight_versions.append(str(weight_version))

    def rollback(self) -> None:
        self._staged.clear()

    def committed_keys(self) -> List[Any]:
        return list(self._committed)

    def staged_get(self, key: Any) -> Optional[Dict[str, Any]]:
        return self._staged.get(key)

    def staged_keys(self) -> List[Any]:
        return list(self._staged)

    def clear(self) -> None:
        """Drop everything (used to free an adopted prime-lite capture)."""
        self._committed.clear()
        self._staged.clear()


# ============================================================================
# Expert slab quantize + diff
# ============================================================================


def hf_expert_stack(local_slab_bf16: torch.Tensor) -> torch.Tensor:
    """Training ``[E_local, K, N]`` (input-major) -> HF ``[E_local, N, K]``.

    For fused gate_up storage (N == 2*I, gate on the first half of the last
    dim) the permuted stack has rows [0:I] == gate and rows [I:2I] == up —
    exactly the receiver's fused w13 layout and bit-identical, block-row by
    block-row, to the full path's per-expert transpose+quantize.
    """
    return local_slab_bf16.permute(0, 2, 1).contiguous()


def mirror_zero_lora_merge(
    slab_bf16: torch.Tensor,
    merge_halves: Tuple[bool, bool],
) -> torch.Tensor:
    """Mirror the full path's ``base + lora_delta`` merge for zero-B LoRA.

    The reference adds a ``+0.0`` delta tensor (bmm with B == 0 yields +0.0
    under FMA/zero-init accumulation), whose only bit-level effect is
    flipping ``-0.0`` base values to ``+0.0``. ``x + 0.0`` (scalar) has the
    identical effect.
    """
    if not any(merge_halves):
        return slab_bf16
    if all(merge_halves):
        return slab_bf16 + 0.0
    out = slab_bf16.clone()
    half = out.shape[-1] // 2
    if merge_halves[0]:
        out[..., :half] = out[..., :half] + 0.0
    if merge_halves[1]:
        out[..., half:] = out[..., half:] + 0.0
    return out


def check_zero_lora_b(plan: FoldSyncPlan) -> float:
    """Max |lora_B| across this rank's local shards (0.0 required for fresh_ab)."""
    max_abs = 0.0
    params: List[torch.nn.Parameter] = []
    for entry in plan.expert_params:
        params.extend(entry.lora_b_params)
    for entry in plan.dense_params:
        params.extend(entry.lora_b_params)
    for p in params:
        data = p.data
        if _HAS_DTENSOR and isinstance(data, DTensor):
            data = data.to_local()
        if data.numel():
            max_abs = max(max_abs, float(data.detach().abs().max()))
    return max_abs


@dataclass
class ExpertSyncResult:
    segments: List[RankEntrySegment]
    staged_keys: List[Any]
    dense_bytes: int
    changed_values: int
    prime_only: bool


def process_expert_param(
    entry: ExpertParamPlan,
    baseline_store: FoldBaselineStore,
    *,
    quantize_stack_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    block_size: Tuple[int, int],
    prime_only: bool = False,
) -> Tuple[List[RankEntrySegment], int, int]:
    """Quantize this rank's expert slab, diff vs baseline, stage new baseline.

    Returns (segments, dense_bytes_processed, changed_values). In
    ``prime_only`` mode no diff/segments are produced — the quantized state
    is staged so a subsequent full-path sync establishes the receiver state
    that matches it.
    """
    local, expert_offset, e_total = expert_slab(entry)
    e_local, k_dim, n_dim = local.shape
    block_row, block_col = block_size
    if entry.kind == "gate_up":
        inter = n_dim // 2
        if inter % block_row != 0:
            raise FoldSparseDeltaUnsupported(
                f"{entry.train_name}: intermediate {inter} not divisible by FP8 block {block_row}"
            )
    if n_dim % 2 != 0 and entry.kind == "gate_up":
        raise FoldSparseDeltaUnsupported(f"{entry.train_name}: odd fused gate_up width {n_dim}")

    merged = mirror_zero_lora_merge(local.to(torch.bfloat16), entry.lora_merge_halves)
    stack = hf_expert_stack(merged)  # [E_local, N, K]
    del merged
    quantized, scale_inv = quantize_stack_fn(stack)
    del stack
    quantized = quantized.contiguous()
    scale_inv = scale_inv.contiguous()
    # The byte diff runs wherever the quantizer left the tensors (GPU in the
    # gpu-quantize mode, where a CPU compare would require a pageable D2H
    # copy). The baseline cache device is configurable so callers can trade
    # persistent device memory for avoiding that copy.
    keep_on_gpu = quantized.device.type == "cuda" and fold_baseline_device() in {"cuda", "gpu"}
    if keep_on_gpu or quantized.device.type == "cpu":
        quantized_store, scale_store = quantized, scale_inv
    else:
        quantized_store, scale_store = quantized.cpu(), scale_inv.cpu()

    dense_bytes = local.numel() * local.element_size()
    key = ("expert", entry.train_name)
    baseline = baseline_store.get(key)
    baseline_store.stage(
        key,
        {
            "weight": quantized_store,
            "scale": scale_store,
            "expert_offset": expert_offset,
            "expert_count": e_local,
        },
    )
    if prime_only:
        return [], dense_bytes, 0

    if baseline is not None and (
        baseline.get("expert_offset") != expert_offset or baseline.get("expert_count") != e_local
    ):
        raise FoldSparseDeltaUnsupported(f"{entry.train_name}: expert shard layout changed since baseline")

    # Every eFSDP rank participates in the Shard(1) full_tensor gather and
    # retains the same baseline, but only rank 0 of each eFSDP group emits
    # this EP slab.  Otherwise gathered rank payloads contain overlapping
    # receiver indices.
    if not entry.transport_owner:
        return [], 0, 0

    per_expert_numel = quantized.shape[1] * quantized.shape[2]
    per_expert_scale_numel = scale_inv.shape[1] * scale_inv.shape[2]
    weight_full_shape = (e_total, quantized.shape[1], quantized.shape[2])
    scale_full_shape = (e_total, scale_inv.shape[1], scale_inv.shape[2])
    for shape in (weight_full_shape, scale_full_shape):
        numel = shape[0] * shape[1] * shape[2]
        if numel > 2**31 - 1:
            raise FoldSparseDeltaUnsupported(f"{entry.weight_name}: numel {numel} exceeds int32 index space")

    base_w = base_s = None
    if baseline is not None:
        base_w = baseline["weight"]
        base_s = baseline["scale"]
        if quantized.device.type != "cpu":
            base_w = base_w.to(quantized.device, non_blocking=True)
            base_s = base_s.to(scale_inv.device, non_blocking=True)
    w_idx, w_vals = exact_byte_diff(quantized, base_w)
    s_idx, s_vals = exact_byte_diff(scale_inv, base_s)
    w_idx += expert_offset * per_expert_numel
    s_idx += expert_offset * per_expert_scale_numel

    segments = [
        RankEntrySegment.from_indices(entry.weight_name, weight_full_shape, quantized.dtype, w_idx, w_vals),
        RankEntrySegment.from_indices(entry.scale_name, scale_full_shape, scale_inv.dtype, s_idx, s_vals),
    ]
    changed = int(w_idx.numel() + s_idx.numel())
    return segments, dense_bytes, changed


def process_dense_params(
    plan: FoldSyncPlan,
    baseline_store: FoldBaselineStore,
    *,
    is_rank0: bool,
    unfuse_fn: Callable[[List[Tuple[str, torch.Tensor]]], List[Tuple[str, torch.Tensor]]],
    quantize_buffer_fn: Callable[[List[Tuple[str, torch.Tensor]]], List[Tuple[str, torch.Tensor]]],
    prime_only: bool = False,
) -> Tuple[List[RankEntrySegment], int, int]:
    """Full-gather + reference-quantize the (small) dense fold params.

    All ranks must call this (``full_tensor`` is collective); only rank 0
    diffs/stages. Reuses the full path's unfuse + quantize functions so the
    emitted names/bytes are bit-identical to a full push.

    Choreography contract: the collective ``full_tensor()`` gathers for
    EVERY dense param are posted first, before any fallible rank-0-local
    work (bf16 merge / unfuse / quantize / diff). A rank-0 failure in the
    local phase therefore cannot leave peer ranks' gathers unmatched — the
    caller catches it and feeds a rank consensus instead.
    """
    if not plan.dense_params:
        return [], 0, 0

    # Phase 1 — collective: every rank gathers every dense param.
    gathered: List[Tuple[DenseParamPlan, torch.Tensor]] = []
    for entry in sorted(plan.dense_params, key=lambda e: e.sort_key):
        full = full_dense_tensor(entry.param)
        if is_rank0:
            gathered.append((entry, full))
        else:
            del full
    if not is_rank0:
        return [], 0, 0

    # Phase 2 — rank-0 local: merge/unfuse/quantize/diff.
    buffer: List[Tuple[str, torch.Tensor]] = []
    dense_bytes = 0
    for entry, full in gathered:
        merged = full.to(torch.bfloat16)
        if entry.lora_merge:
            merged = merged + 0.0
        dense_bytes += merged.numel() * merged.element_size()
        buffer.append((entry.train_name, merged))
    gathered.clear()

    buffer = unfuse_fn(buffer)
    buffer = quantize_buffer_fn(buffer)

    segments: List[RankEntrySegment] = []
    changed = 0
    for hf_name, tensor in buffer:
        tensor = tensor.contiguous().cpu()
        if tensor.numel() > 2**31 - 1:
            raise FoldSparseDeltaUnsupported(f"{hf_name}: numel exceeds int32 index space")
        key = ("dense", hf_name)
        baseline = baseline_store.get(key)
        baseline_store.stage(key, {"tensor": tensor})
        if prime_only:
            continue
        idx, vals = exact_byte_diff(tensor, None if baseline is None else baseline["tensor"])
        changed += int(idx.numel())
        segments.append(RankEntrySegment.from_indices(hf_name, tuple(tensor.shape), tensor.dtype, idx, vals))
    return segments, dense_bytes, changed


def process_captured_target(
    plan: FoldSyncPlan,
    baseline_store: FoldBaselineStore,
    target_store: FoldBaselineStore,
    *,
    is_rank0: bool,
) -> Tuple[List[RankEntrySegment], int, int]:
    """Diff a captured exact receiver target against the shipped baseline.

    A reset cannot merely re-quantize the restored BF16 master: the sampler
    boots from pre-quantized FP8 checkpoint bytes, including a very small
    quantizer/checkpoint byte correction and checkpoint scales preserved by
    the prime-lite overlay. ``target_store`` is that overlay-corrected capture.
    This helper emits the inverse delta from the current served baseline to
    those exact load-time bytes and stages the exact target as the next
    baseline.
    """
    meta = target_store.get(("meta", "prefold"))
    if meta is None:
        raise FoldSparseDeltaUnsupported("reset target has no prime-lite capture")
    captured_names = tuple(meta.get("param_names") or ())
    if captured_names != tuple(sorted(plan.param_names)):
        raise FoldSparseDeltaUnsupported("reset target parameter set does not match the fold plan")

    segments: List[RankEntrySegment] = []
    dense_bytes = 0
    changed = 0
    for entry in plan.expert_params:
        key = ("expert", entry.train_name)
        target = target_store.get(key)
        baseline = baseline_store.get(key)
        if target is None:
            raise FoldSparseDeltaUnsupported(f"reset target missing {entry.train_name}")
        if baseline is None:
            raise FoldSparseDeltaUnsupported(f"reset baseline missing {entry.train_name}")
        for layout_field in ("expert_offset", "expert_count"):
            if int(target[layout_field]) != int(baseline[layout_field]):
                raise FoldSparseDeltaUnsupported(f"reset target {entry.train_name} {layout_field} changed")

        target_w = target["weight"]
        target_s = target["scale"]
        base_w = baseline["weight"].to(target_w.device, non_blocking=True)
        base_s = baseline["scale"].to(target_s.device, non_blocking=True)
        w_idx, w_vals = exact_byte_diff(target_w, base_w)
        s_idx, s_vals = exact_byte_diff(target_s, base_s)
        offset = int(target["expert_offset"])
        full_experts = int(entry.expert_total or entry.param.shape[0])
        baseline_store.stage(key, target)
        if not entry.transport_owner:
            continue
        w_idx += offset * target_w.shape[1] * target_w.shape[2]
        s_idx += offset * target_s.shape[1] * target_s.shape[2]
        segments.extend(
            [
                RankEntrySegment.from_indices(
                    entry.weight_name,
                    (full_experts, target_w.shape[1], target_w.shape[2]),
                    target_w.dtype,
                    w_idx,
                    w_vals,
                ),
                RankEntrySegment.from_indices(
                    entry.scale_name,
                    (full_experts, target_s.shape[1], target_s.shape[2]),
                    target_s.dtype,
                    s_idx,
                    s_vals,
                ),
            ]
        )
        dense_bytes += int(target_w.numel() * target_w.element_size())
        changed += int(w_idx.numel() + s_idx.numel())

    if is_rank0:
        dense_keys = sorted(
            key
            for key in target_store.committed_keys()
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "dense"
        )
        for key in dense_keys:
            target = target_store.get(key)
            baseline = baseline_store.get(key)
            if target is None or baseline is None:
                raise FoldSparseDeltaUnsupported(f"reset dense baseline missing {key[1]}")
            tensor = target["tensor"]
            idx, vals = exact_byte_diff(tensor, baseline["tensor"].to(tensor.device))
            segments.append(RankEntrySegment.from_indices(key[1], tuple(tensor.shape), tensor.dtype, idx, vals))
            dense_bytes += int(tensor.numel() * tensor.element_size())
            changed += int(idx.numel())
            baseline_store.stage(key, target)
        if plan.dense_params:
            marker = ("dense_primed", tuple(sorted(e.train_name for e in plan.dense_params)))
            if not target_store.has(marker):
                raise FoldSparseDeltaUnsupported("reset target has no dense marker")
            baseline_store.stage(marker, {})

    return segments, dense_bytes, changed


# ============================================================================
# Prime-lite: checkpoint-primed baselines (skip the one-time priming push)
# ============================================================================
#
# Receivers may start from a pre-quantized FP8 checkpoint while the trainer
# owns a higher-precision master. A full push can be skipped only when an
# overlay proves that trainer-side quantization reproduces the receiver
# checkpoint and describes any fixed byte or scale differences. The
# prime-lite flow is:
#
#   1. An external builder compares the trainer's quantized checkpoint view
#      with the receiver checkpoint and writes an overlay containing fixed
#      byte/scale corrections and expected digests.
#   2. Before the FIRST fresh_ab fold mutates the base params, the trainer
#      captures quantize(master@step0) (``run_prefold_capture``), verifies
#      it against the overlay digests, and falls back to a full push on any
#      mismatch. The corrected state is committed under ``PRIME_LITE_SCOPE``.
#   3. The first fold sync adopts that capture as its committed baseline
#      (``adopt_prefold_baseline``) instead of priming + full-pushing. The
#      first delta is an ordinary diff against the receivers' served bytes.
#      No receiver-side change is needed: the delta ships with no
#      ``base_weight_version`` precondition (fresh receivers are unstamped)
#      and stamps the version chain itself.
# ============================================================================

PRIME_LITE_SCOPE = "__zorl_prefold__"
PRIME_OVERLAY_VERSION = 1


def prime_lite_overlay_path(sparse_delta_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Overlay artifact path; unset => prime-lite disabled (full-push prime)."""
    if sparse_delta_config and sparse_delta_config.get("prime_lite_overlay"):
        return str(sparse_delta_config["prime_lite_overlay"])
    return os.environ.get("XORL_SPARSE_DELTA_FOLD_PRIME_OVERLAY", "").strip() or None


def normalize_receiver_name(name: str) -> str:
    """Canonical entry key: strip any model wrapper prefix up to ``layers.<i>``.

    ``model.layers.3.mlp.experts.gate_up_proj`` and
    ``model.language_model.layers.3.mlp.experts.gate_up_proj`` (the HF
    checkpoints' naming) both map to ``layers.3.mlp.experts.gate_up_proj``.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return ".".join(parts[i:])
    return name


def _digest_u8(view_u8: torch.Tensor) -> bytes:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(view_u8.contiguous().view(torch.uint8).cpu().numpy().tobytes()).digest()


def _digest_tensor_to_bytes(digest_row: torch.Tensor) -> bytes:
    return bytes(digest_row.to(torch.uint8).tolist())


def overlay_expert_entry(
    expected_weight: torch.Tensor,
    receiver_weight: torch.Tensor,
    receiver_scale: torch.Tensor,
) -> Dict[str, Any]:
    """Build one overlay entry for a fused expert stack.

    ``expected_weight``: quantize(bf16 checkpoint) fused stack ``[E, N, K]``
    (fp8). ``receiver_weight``: the FP8 checkpoint's bytes in the same fused
    layout. ``receiver_scale``: the checkpoint's scale_inv stack
    ``[E, nR, nC]`` (stored dtype, bf16 for the Qwen FP8 checkpoints).
    """
    if expected_weight.shape != receiver_weight.shape:
        raise ValueError(f"shape mismatch: {tuple(expected_weight.shape)} vs {tuple(receiver_weight.shape)}")
    e_total = expected_weight.shape[0]
    exp_u8 = expected_weight.reshape(e_total, -1).view(torch.uint8).cpu()
    rec_u8 = receiver_weight.reshape(e_total, -1).view(torch.uint8).cpu()
    mismatch = (exp_u8 != rec_u8).reshape(-1)
    idx = mismatch.nonzero(as_tuple=False).flatten().to(torch.int64)
    vals = rec_u8.reshape(-1)[idx].clone()
    digests = torch.stack(
        [torch.frombuffer(bytearray(_digest_u8(exp_u8[e])), dtype=torch.uint8) for e in range(e_total)]
    )
    return {
        "shape": tuple(int(s) for s in expected_weight.shape),
        "mismatch_idx": idx,
        "mismatch_val": vals,
        "scale": receiver_scale.contiguous().cpu(),
        "digests": digests,
    }


def overlay_dense_entry(expected_weight: torch.Tensor, receiver_weight: torch.Tensor) -> Dict[str, Any]:
    """Overlay entry for one dense (2D) fold param in receiver HF naming."""
    if expected_weight.shape != receiver_weight.shape:
        raise ValueError(f"shape mismatch: {tuple(expected_weight.shape)} vs {tuple(receiver_weight.shape)}")
    exp_u8 = expected_weight.reshape(-1).view(torch.uint8).cpu()
    rec_u8 = receiver_weight.reshape(-1).view(torch.uint8).cpu()
    mismatch = exp_u8 != rec_u8
    idx = mismatch.nonzero(as_tuple=False).flatten().to(torch.int64)
    return {
        "shape": tuple(int(s) for s in expected_weight.shape),
        "mismatch_idx": idx,
        "mismatch_val": rec_u8[idx].clone(),
        "digest": torch.frombuffer(bytearray(_digest_u8(exp_u8)), dtype=torch.uint8),
    }


def overlay_dense_scale_entry(receiver_scale: torch.Tensor) -> Dict[str, Any]:
    return {"scale": receiver_scale.contiguous().cpu()}


def load_prime_overlay(path: str) -> Dict[str, Any]:
    overlay = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(overlay, dict) or overlay.get("version") != PRIME_OVERLAY_VERSION:
        raise FoldSparseDeltaUnsupported(
            f"unsupported prime-lite overlay at {path!r}: version={overlay.get('version') if isinstance(overlay, dict) else '?'}"
        )
    return overlay


def make_prefold_quantize_fns(
    model: torch.nn.Module,
    quantization: Dict[str, Any],
) -> Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Tuple[int, int], str]:
    """Quantize/unfuse closures matching the fold sync path bit-for-bit.

    Mirrors ``WeightSyncHandler._sync_zorl_fold_sparse_delta``'s closures
    (same static kernels, same ``XORL_P2P_FP8_QUANTIZE_DEVICE`` policy) so
    the captured bytes equal what the sync's fast path will produce.
    Returns (quantize_stack_fn, unfuse_fn, quantize_buffer_fn,
    (block_row, block_col), quantize_mode_tag).
    """
    from xorl.server.weight_sync.handler import WeightSyncHandler  # noqa: PLC0415

    quantization = dict(quantization or {})
    if quantization.get("quant_method") != "fp8":
        raise FoldSparseDeltaUnsupported("prime-lite capture requires fp8 sync quantization")
    block_row, block_col = WeightSyncHandler._fp8_block_size(quantization)
    fp8_dtype, fp8_max = WeightSyncHandler._fp8_dtype_and_max(quantization)
    exec_gpu = (
        WeightSyncHandler._fp8_quantization_execution_device() in {"gpu", "cuda"}
        and torch.cuda.is_available()
        and fp8_dtype == torch.float8_e4m3fn
        and block_row == block_col
    )
    quant_target = None if exec_gpu else "cpu"

    def quantize_stack(stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return WeightSyncHandler._quantize_fp8_stack(
            stack,
            fp8_dtype=fp8_dtype,
            fp8_max=fp8_max,
            block_size_row=block_row,
            block_size_col=block_col,
            target_device=quant_target,
            phase_s=None,
            phase_prefix="prefold_fp8",
        )

    def unfuse(buffer: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return WeightSyncHandler._unfuse_for_inference(buffer, model, clone_slices=False)

    def quantize_buffer(buffer: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return WeightSyncHandler._quantize_buffer_for_fp8(
            buffer,
            quantization_config=quantization,
            target_device="cpu",
            phase_s=None,
            phase_prefix="prefold_dense_fp8",
        )

    tag = f"{WeightSyncHandler._fp8_quantization_execution_device()}|block{block_row}x{block_col}"
    return quantize_stack, unfuse, quantize_buffer, (block_row, block_col), tag


def _apply_overlay_to_staged(
    plan: FoldSyncPlan,
    overlay: Dict[str, Any],
    store: FoldBaselineStore,
    *,
    is_rank0: bool,
) -> None:
    """Turn staged quantize(master@step0) entries into the receivers' bytes.

    Digest-gates every local expert / dense tensor against the overlay's
    expected quantized bytes first (catching master != bf16 checkpoint and
    quantizer drift), then scatters the checkpoint's mismatching weight
    bytes and replaces every scale by the upcast checkpoint scale.
    """
    experts_ov = overlay.get("experts") or {}
    for entry in plan.expert_params:
        staged = store.staged_get(("expert", entry.train_name))
        if staged is None:
            raise FoldSparseDeltaUnsupported(f"prime-lite: no staged baseline for {entry.train_name}")
        ov = experts_ov.get(normalize_receiver_name(entry.weight_name))
        if ov is None:
            raise FoldSparseDeltaUnsupported(f"prime-lite: overlay has no entry for {entry.weight_name}")
        weight = staged["weight"]
        scale = staged["scale"]
        offset = int(staged["expert_offset"])
        count = int(staged["expert_count"])
        full_shape = tuple(int(s) for s in ov["shape"])
        if tuple(weight.shape[1:]) != full_shape[1:] or offset + count > full_shape[0]:
            raise FoldSparseDeltaUnsupported(
                f"prime-lite: {entry.weight_name} layout {tuple(weight.shape)}@{offset} "
                f"does not fit overlay shape {full_shape}"
            )
        per_expert = full_shape[1] * full_shape[2]
        digests = ov["digests"]
        weight_u8_cpu = weight.detach().reshape(count, -1).view(torch.uint8).cpu()
        bad = [e for e in range(count) if _digest_u8(weight_u8_cpu[e]) != _digest_tensor_to_bytes(digests[offset + e])]
        if bad:
            raise FoldSparseDeltaUnsupported(
                f"prime-lite: {entry.weight_name}: quantized-master digest mismatch on "
                f"{len(bad)}/{count} local experts (first: {offset + bad[0]}) — master@step0 is not "
                "the overlay's bf16 checkpoint, or the quantizer does not reproduce the overlay build"
            )
        idx = ov["mismatch_idx"]
        vals = ov["mismatch_val"]
        lo, hi = offset * per_expert, (offset + count) * per_expert
        in_slab = (idx >= lo) & (idx < hi)
        if bool(in_slab.any()):
            local_idx = (idx[in_slab] - lo).to(torch.long).to(weight.device)
            weight.view(torch.uint8).reshape(-1)[local_idx] = vals[in_slab].to(weight.device)
        ov_scale = ov["scale"][offset : offset + count].to(torch.float32)
        if tuple(ov_scale.shape) != tuple(scale.shape):
            raise FoldSparseDeltaUnsupported(
                f"prime-lite: {entry.scale_name} shape {tuple(scale.shape)} != overlay {tuple(ov_scale.shape)}"
            )
        scale.copy_(ov_scale.to(scale.device))

    if not is_rank0:
        return
    dense_ov = overlay.get("dense") or {}
    for key in store.staged_keys():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == "dense"):
            continue
        hf_name = key[1]
        tensor = store.staged_get(key)["tensor"]
        ov = dense_ov.get(normalize_receiver_name(hf_name))
        if ov is None:
            raise FoldSparseDeltaUnsupported(f"prime-lite: overlay has no dense entry for {hf_name}")
        if hf_name.endswith("_scale_inv"):
            ov_scale = ov["scale"].to(torch.float32)
            if tuple(ov_scale.shape) != tuple(tensor.shape):
                raise FoldSparseDeltaUnsupported(
                    f"prime-lite: {hf_name} shape {tuple(tensor.shape)} != overlay {tuple(ov_scale.shape)}"
                )
            tensor.copy_(ov_scale)
            continue
        if tuple(int(s) for s in ov["shape"]) != tuple(tensor.shape):
            raise FoldSparseDeltaUnsupported(
                f"prime-lite: {hf_name} shape {tuple(tensor.shape)} != overlay {tuple(ov['shape'])}"
            )
        if _digest_u8(tensor.reshape(-1)) != _digest_tensor_to_bytes(ov["digest"]):
            raise FoldSparseDeltaUnsupported(
                f"prime-lite: {hf_name}: quantized-master digest mismatch — master@step0 is not the "
                "overlay's bf16 checkpoint, or the quantizer does not reproduce the overlay build"
            )
        idx = ov["mismatch_idx"]
        if int(idx.numel()):
            tensor.view(torch.uint8).reshape(-1)[idx.to(torch.long)] = ov["mismatch_val"]


def run_prefold_capture(
    plan: FoldSyncPlan,
    overlay: Dict[str, Any],
    *,
    store: FoldBaselineStore,
    quantize_stack_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    unfuse_fn: Callable[[List[Tuple[str, torch.Tensor]]], List[Tuple[str, torch.Tensor]]],
    quantize_buffer_fn: Callable[[List[Tuple[str, torch.Tensor]]], List[Tuple[str, torch.Tensor]]],
    is_rank0: bool,
    block_size: Tuple[int, int],
    quantize_mode_tag: str,
) -> Tuple[bool, str]:
    """Quantize the PRISTINE master, verify + correct to receiver bytes, stage.

    Must run BEFORE the first fresh_ab fold mutates the base params. Stages
    into ``store`` WITHOUT committing — the caller runs a cross-rank
    consensus and commits (all ranks) or rolls back (any failure). The dense
    ``full_tensor`` gathers are collective, so this function must be entered
    by every rank; expert/local failures are carried into the return value
    (mirroring the handler's prime-pass choreography).
    """
    error: Optional[str] = None
    try:
        if overlay.get("version") != PRIME_OVERLAY_VERSION:
            raise FoldSparseDeltaUnsupported(f"overlay version {overlay.get('version')} unsupported")
        max_b = check_zero_lora_b(plan)
        if max_b != 0.0:
            raise FoldSparseDeltaUnsupported(
                f"parent lora_B is non-zero (max abs {max_b:.3e}); prime-lite requires B == 0"
            )
        for entry in plan.expert_params:
            process_expert_param(
                entry,
                store,
                quantize_stack_fn=quantize_stack_fn,
                block_size=block_size,
                prime_only=True,
            )
    except Exception as exc:  # noqa: BLE001 - carried into the caller's consensus
        logger.warning("[PrimeLite] expert capture failed", exc_info=True)
        error = f"{type(exc).__name__}: {exc}"
    # Collective phase — always post the dense gathers (see docstring).
    try:
        process_dense_params(
            plan,
            store,
            is_rank0=is_rank0,
            unfuse_fn=unfuse_fn,
            quantize_buffer_fn=quantize_buffer_fn,
            prime_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PrimeLite] dense capture failed", exc_info=True)
        error = error or f"{type(exc).__name__}: {exc}"
    if error is None:
        try:
            _apply_overlay_to_staged(plan, overlay, store, is_rank0=is_rank0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PrimeLite] overlay verification/correction failed", exc_info=True)
            error = f"{type(exc).__name__}: {exc}"
    if error is not None:
        store.rollback()
        return False, error
    if is_rank0 and plan.dense_params:
        store.stage(("dense_primed", tuple(sorted(e.train_name for e in plan.dense_params))), {})
    store.stage(
        ("meta", "prefold"),
        {
            "quantize_mode": quantize_mode_tag,
            "param_names": tuple(sorted(plan.param_names)),
            "overlay_meta": dict(overlay.get("meta") or {}),
        },
    )
    return True, ""


def adopt_prefold_baseline(
    store: FoldBaselineStore,
    plan: FoldSyncPlan,
    *,
    quantize_mode_tag: str,
    fold_index: int,
    is_rank0: bool,
    prefold_store: Optional[FoldBaselineStore] = None,
) -> Tuple[bool, str]:
    """Adopt a committed prime-lite capture as this sync group's baseline.

    The captured state IS the receivers' current served bytes (they booted
    from the FP8 checkpoint and have never been synced), so adoption commits
    immediately — no push is needed to establish it. The first delta then
    ships without a ``base_weight_version`` precondition (meta
    ``weight_version`` None: fresh receivers are unstamped) and stamps the
    version chain itself.
    """
    prefold = prefold_store if prefold_store is not None else FoldBaselineStore.for_scope(PRIME_LITE_SCOPE)
    meta = prefold.get(("meta", "prefold"))
    if meta is None:
        return False, "no prefold capture"
    if meta.get("quantize_mode") != quantize_mode_tag:
        return False, (f"prefold quantize mode {meta.get('quantize_mode')!r} != sync mode {quantize_mode_tag!r}")
    for entry in plan.expert_params:
        if not prefold.has(("expert", entry.train_name)):
            return False, f"prefold capture missing {entry.train_name}"
    if is_rank0 and plan.dense_params:
        dense_marker = ("dense_primed", tuple(sorted(e.train_name for e in plan.dense_params)))
        if not prefold.has(dense_marker):
            return False, "prefold capture missing the dense baselines"
    for key in prefold.committed_keys():
        if isinstance(key, tuple) and key and key[0] in ("expert", "dense", "dense_primed"):
            store.stage(key, prefold.get(key))
    store.stage(
        ("meta", "fold_sync"),
        {
            "quantize_mode": quantize_mode_tag,
            "last_primed_fold_index": int(fold_index),
            "weight_version": None,
        },
    )
    store.commit()
    prefold.clear()
    return True, ""


# ============================================================================
# Packed-file assembly (rank 0)
# ============================================================================


def fold_delta_transport(sparse_delta_config: Optional[Dict[str, Any]]) -> str:
    """How rank 0 ships the packed fold delta to the receivers.

    ``file`` (default): write a ``.packed`` file on the shared FS and POST
    its path (every receiver reads the whole file over NFS).
    ``rdma``: keep the packed bytes in pinned host memory and RDMA-write them
    into per-receiver Mooncake staging buffers (parallel fan-out); the POST
    then carries a staging apply instead of a path. Falls back to ``file``
    within the sync when RDMA is unavailable (no Mooncake / old receivers).
    """
    if sparse_delta_config and sparse_delta_config.get("transport"):
        value = str(sparse_delta_config["transport"]).strip().lower()
    else:
        value = os.environ.get("XORL_SPARSE_DELTA_TRANSPORT", "file").strip().lower()
    if value not in {"file", "rdma"}:
        logger.warning("[FoldSparseDelta] invalid sparse-delta transport %r; using 'file'", value)
        return "file"
    return value


def _merge_rank_segments_to_encoded(
    all_rank_segments: Sequence[Sequence[RankEntrySegment]],
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Merge per-rank segments into ``{name: EncodedDelta}`` (+ total nnz).

    Entry order follows the first-seen order across ranks (deterministic:
    every rank produced the same entry-name sequence). Entries whose merged
    nnz is 0 are still emitted (nnz=0) so the receiver sees the full
    fold-touched set and the payload is never empty.
    """
    from xorl.server.weight_sync.sparse_delta_files import prepare_delta_encoding_runtime  # noqa: PLC0415

    prepare_delta_encoding_runtime(
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )
    import importlib  # noqa: PLC0415

    types_mod = importlib.import_module("delta_encoding.encoding.types")
    EncodedDelta = types_mod.EncodedDelta

    by_name: Dict[str, List[RankEntrySegment]] = {}
    order: List[str] = []
    for rank_segments in all_rank_segments:
        if not rank_segments:
            continue
        for seg in rank_segments:
            if seg.name not in by_name:
                by_name[seg.name] = []
                order.append(seg.name)
            by_name[seg.name].append(seg)

    encoded: Dict[str, Any] = {}
    total_nnz = 0
    for name in order:
        segs = by_name[name]
        shape = segs[0].shape
        dtype = segs[0].dtype
        for seg in segs:
            if seg.shape != shape or seg.dtype != dtype:
                raise ValueError(f"inconsistent shape/dtype across ranks for {name!r}")
        flat_deltas, values = merge_rank_segments(segs)
        values = values.view(dtype)
        encoded[name] = EncodedDelta(flat_deltas=flat_deltas, values=values, shape=tuple(shape))
        total_nnz += int(values.numel())

    if not encoded:
        raise ValueError("no sparse-delta entries to write")
    return encoded, total_nnz


def build_packed_delta_file(
    all_rank_segments: Sequence[Sequence[RankEntrySegment]],
    output_path: str,
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
) -> Dict[str, Any]:
    """Merge per-rank segments and write one ``delta_packed_v1`` file."""
    import importlib  # noqa: PLC0415

    encoded, total_nnz = _merge_rank_segments_to_encoded(
        all_rank_segments,
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )
    packed = importlib.import_module("delta_encoding.encoding.packed")
    written = packed.write_packed_file(encoded, output_path)
    written = str(written)
    return {
        "path": written,
        "tensors": len(encoded),
        "nnz": total_nnz,
        "packed_bytes": os.path.getsize(written),
    }


def build_packed_delta_payload(
    all_rank_segments: Sequence[Sequence[RankEntrySegment]],
    *,
    delta_encoding_path: Optional[str] = None,
    use_native_extension: bool = False,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    """Merge per-rank segments into ONE in-memory ``delta_packed_v1`` payload.

    Byte-identical to what :func:`build_packed_delta_file` writes (both call
    ``pack_delta_buffer``); returned pinned so the RDMA transport can register
    it with Mooncake and fan it out without touching the shared filesystem.
    """
    import importlib  # noqa: PLC0415

    encoded, total_nnz = _merge_rank_segments_to_encoded(
        all_rank_segments,
        delta_encoding_path=delta_encoding_path,
        use_native_extension=use_native_extension,
    )
    packed = importlib.import_module("delta_encoding.encoding.packed")
    buf, _entries = packed.pack_delta_buffer(encoded, pin_memory=pin_memory and torch.cuda.is_available())
    return {
        "payload": buf,
        "tensors": len(encoded),
        "nnz": total_nnz,
        "packed_bytes": int(buf.numel()),
    }


# ============================================================================
# Timing helper
# ============================================================================


class PhaseTimer:
    def __init__(self) -> None:
        self.phases: Dict[str, float] = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.phases[name] = self.phases.get(name, 0.0) + (now - self._t)
        self._t = now
