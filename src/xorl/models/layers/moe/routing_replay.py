"""
Megatron/SLIME-style MoE routing replay for deterministic checkpoint recomputation.

MoE training with gradient checkpointing (+ pipeline parallelism) requires
deterministic routing on recomputation.  Flash attention non-determinism causes
different hidden_states on recompute -> different top-k routing -> shape
mismatches in EP all-to-all dispatch.

Design (matches Megatron RouterReplay):
    - One ``RoutingReplay`` instance per MoE layer, registered in a class-level list.
    - Dual read pointers: ``forward_index`` for R3/repeated forward,
      ``backward_index`` for checkpoint recompute.
    - Global stage flag controls behaviour: ``None | "record" | "replay_forward"
      | "replay_backward"``.

Stage switching lifecycle::

    Non-PP:                              PP:
    set("replay_backward")               set("replay_backward")
    for mb in micro_batches:             _pp_forward temporarily sets "record"
      set("record")                        model layers run with "record"
      model.forward()  # records         _pp_forward restores "replay_backward"
      set("replay_backward")             loss.backward()
      loss.backward()  # pop_backward      checkpoint recompute -> "replay_backward"
      reset_all_backward()                 -> pop_backward
    set(None)                            set(None)
    clear_all()                          clear_all()
"""

from typing import ClassVar, List, Optional

import torch


class RoutingReplay:
    """Per-MoE-layer routing replay with dual-index for PP + checkpoint."""

    _instances: ClassVar[List["RoutingReplay"]] = []

    def __init__(self):
        self.forward_index: int = 0
        self.backward_index: int = 0
        self.top_indices_list: List[torch.Tensor] = []  # CPU pinned
        RoutingReplay._instances.append(self)

    @torch.compiler.disable
    def record(self, selected_experts: torch.Tensor):
        """Append routing decision (CPU pinned copy).

        Disabled for torch.compile — pin_memory and list-append side
        effects are not supported by Inductor/Dynamo.
        """
        buf = torch.empty_like(selected_experts, device="cpu", pin_memory=True)
        buf.copy_(selected_experts)
        self.top_indices_list.append(buf)

    @torch.compiler.disable
    def pop_forward(self) -> torch.Tensor:
        """Read routing for forward replay, advance forward_index."""
        idx = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return idx.to(torch.cuda.current_device(), non_blocking=True)

    @torch.compiler.disable
    def pop_backward(self) -> torch.Tensor:
        """Read routing for checkpoint recompute, advance backward_index."""
        idx = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return idx.to(torch.cuda.current_device(), non_blocking=True)

    def reset_forward(self):
        self.forward_index = 0

    def reset_backward(self):
        self.backward_index = 0

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list.clear()

    @classmethod
    def clear_all(cls):
        for inst in cls._instances:
            inst.clear()

    @classmethod
    def reset_all_forward(cls):
        for inst in cls._instances:
            inst.reset_forward()

    @classmethod
    def reset_all_backward(cls):
        for inst in cls._instances:
            inst.reset_backward()


# ---------------------------------------------------------------------------
# Global stage
# ---------------------------------------------------------------------------
_replay_stage: Optional[str] = None  # None | "record" | "replay_forward" | "replay_backward"


def get_replay_stage() -> Optional[str]:
    return _replay_stage


def set_replay_stage(stage: Optional[str]) -> None:
    global _replay_stage
    _replay_stage = stage


# ---------------------------------------------------------------------------
# R3 (Rollout Routing Replay) mode
# ---------------------------------------------------------------------------
# When R3 is active, routing decisions are pre-populated from inference data
# instead of recorded during forward.  _pp_forward uses this to switch to
# "replay_forward" instead of "record" so MoE blocks pop pre-populated
# routing rather than computing fresh routing from the gate.
_r3_pre_populated: bool = False


def set_r3_mode(enabled: bool) -> None:
    """Enable/disable R3 pre-populated routing mode."""
    global _r3_pre_populated
    _r3_pre_populated = enabled


def is_r3_mode() -> bool:
    """Check if R3 pre-populated routing mode is active."""
    return _r3_pre_populated
