"""Local model output dataclasses.

Imports ``ModelOutput`` from transformers as the sole base class (provides
OrderedDict behavior required by ``PreTrainedModel`` internals such as
``generate()`` and pipeline), then defines all concrete output types locally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MoeModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MoeCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    # Per-layer residual-stream hidden states (all-layer OPRD). Populated only
    # when forward(output_hidden_states=True); None otherwise.
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


__all__ = [
    "ModelOutput",
    "BaseModelOutput",
    "CausalLMOutput",
    "MoeCausalLMOutput",
    "MoeModelOutput",
]
