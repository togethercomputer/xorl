from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.optimizer import Optimizer

from ..utils import logging


logger = logging.get_logger(__name__)


class MultiOptimizer(Optimizer, Stateful):
    """
    A container that handles multiple optimizers (for ep and non-ep parameters when ep+fsdp2 is enabled)

    Mapping of name -> torch.optim.Optimizer with convenience methods.
    Compatible with torch.distributed.checkpoint optimizer APIs that accept a Mapping.

    This class is needed for EP+FSDP2 case because EP and non-EP param have different FSDP sharding dimension (dim-0 vs. dim-1).
    """

    def __init__(
        self,
        root_model: nn.Module,
        optimizers: dict,
        key_names: list[str],
    ):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Return all param_groups from all internal optimizers."""
        all_groups = []
        for opt in self.optimizers_dict.values():
            all_groups.extend(opt.param_groups)
        return all_groups

    @property
    def state(self) -> Dict[torch.nn.Parameter, Any]:
        """Return merged state dict from all internal optimizers."""
        merged_state: Dict[torch.nn.Parameter, Any] = {}
        for opt in self.optimizers_dict.values():
            merged_state.update(opt.state)
        return merged_state

    def step(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            sd = get_optimizer_state_dict(self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True))
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(
                    f"Key clash detected while merging state dict for optimizer '{name}': {', '.join(sorted(overlap))}"
                )
            else:
                logger.info_rank0(
                    f"MultiOptimizer merged '{name}' state dict ({len(sd)} keys, total {len(merged) + len(sd)})"
                )
            merged.update(sd)

        return merged

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            set_optimizer_state_dict(
                self.model,
                opt,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def register_step_pre_hook(self, hook):
        return [opt.register_step_pre_hook(hook) for opt in self.optimizers_dict.values()]

    def __len__(self) -> int:
        return len(self.optimizers_dict)
