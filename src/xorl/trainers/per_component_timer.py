"""Per-component CUDA-event timing for decoder-style models.

This is intended for short throughput diagnostics. It attaches hooks to common
decoder-layer submodules and reports aggregate forward, backward, and
recompute timing buckets across all layers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn

from xorl.utils.device import get_device_type


_LAYER_SUBMODULE_PHASES: tuple[tuple[str, str, str], ...] = (
    ("input_layernorm", "fwd_norm/input", "bwd_norm/input"),
    ("self_attn", "fwd_attn/total", "bwd_attn/total"),
    ("self_attn.indexer", "fwd_attn/indexer", "bwd_attn/indexer"),
    ("post_attention_layernorm", "fwd_norm/post_attn", "bwd_norm/post_attn"),
    ("mlp", "fwd_mlp_or_moe/total", "bwd_mlp_or_moe/total"),
    ("mlp.gate", "fwd_moe/gate", "bwd_moe/gate"),
    ("mlp.experts", "fwd_moe/experts", "bwd_moe/experts"),
    ("mlp.shared_experts", "fwd_moe/shared", "bwd_moe/shared"),
)


def _resolve_submodule(root: nn.Module, dotted_path: str) -> nn.Module | None:
    obj: Any = root
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
        if not isinstance(obj, nn.Module):
            return None
    return obj


def _is_decoder_layer(module: nn.Module) -> bool:
    return type(module).__name__.endswith("DecoderLayer")


class PerComponentTimer:
    """Aggregates per-component CUDA-event timings across decoder layers."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and get_device_type() == "cuda"
        self._hook_handles: list[Any] = []
        self._attached = False
        self._mode: str = "idle"
        self._fwd_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._bwd_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._recompute_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def attach(self, model: nn.Module) -> int:
        """Register hooks on decoder layers. Returns the number of layers found."""
        if not self.enabled or self._attached:
            return 0
        n_layers = 0
        for module in model.modules():
            if not _is_decoder_layer(module):
                continue
            n_layers += 1
            for dotted_path, fwd_phase, bwd_phase in _LAYER_SUBMODULE_PHASES:
                submodule = _resolve_submodule(module, dotted_path)
                if submodule is not None:
                    self._attach_one(submodule, fwd_phase, bwd_phase)
        self._attached = True
        return n_layers

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._attached = False

    def set_mode(self, mode: str) -> None:
        assert mode in ("fwd", "bwd", "idle"), mode
        self._mode = mode

    def start_step(self) -> None:
        self._fwd_pairs.clear()
        self._bwd_pairs.clear()
        self._recompute_pairs.clear()
        self._mode = "idle"

    def end_step(self) -> dict[str, float]:
        """Synchronize CUDA once and return aggregate seconds by phase name."""
        if not self.enabled:
            return {}
        self._mode = "idle"
        torch.cuda.synchronize()
        result: dict[str, float] = {}

        def _accumulate(pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]) -> None:
            for phase, events in pairs.items():
                total_ms = 0.0
                for start, end in events:
                    total_ms += start.elapsed_time(end)
                result[phase] = total_ms / 1000.0

        _accumulate(self._fwd_pairs)
        _accumulate(self._bwd_pairs)
        recompute_pairs = {
            phase.replace("fwd_", "recompute_", 1): events for phase, events in self._recompute_pairs.items()
        }
        _accumulate(recompute_pairs)
        return result

    def _attach_one(self, module: nn.Module, fwd_phase: str, bwd_phase: str) -> None:
        fwd_state: dict[int, torch.cuda.Event] = {}

        def fwd_pre(mod: nn.Module, _inputs: Any) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            fwd_state[id(mod)] = event

        def fwd_post(mod: nn.Module, _inputs: Any, _output: Any) -> None:
            start = fwd_state.pop(id(mod), None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            bucket = self._recompute_pairs if self._mode == "bwd" else self._fwd_pairs
            bucket[fwd_phase].append((start, end))

        self._hook_handles.append(module.register_forward_pre_hook(fwd_pre))
        self._hook_handles.append(module.register_forward_hook(fwd_post))

        bwd_state: dict[int, torch.cuda.Event] = {}

        def bwd_pre(mod: nn.Module, _grad_output: Any) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            bwd_state[id(mod)] = event

        def bwd_post(mod: nn.Module, _grad_input: Any, _grad_output: Any) -> None:
            start = bwd_state.pop(id(mod), None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._bwd_pairs[bwd_phase].append((start, end))

        self._hook_handles.append(module.register_full_backward_pre_hook(bwd_pre))
        self._hook_handles.append(module.register_full_backward_hook(bwd_post))
