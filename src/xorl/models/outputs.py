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
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MoeModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPastAndLastHiddenState(ModelOutput):
    """
    Extended CausalLMOutputWithPast that also includes last_hidden_state.

    This avoids storing all layer hidden states when only the last layer is needed,
    significantly reducing memory usage for large models.

    Args:
        loss: Language modeling loss (optional)
        logits: Prediction scores of the language modeling head
        past_key_values: Pre-computed key/value pairs for efficient generation
        attentions: Attention weights of all layers (only if output_attentions=True)
        last_hidden_state: Hidden state of the last layer (always available)
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class MoeCausalLMOutputWithPastAndLastHiddenState(ModelOutput):
    """
    Extended MoeCausalLMOutputWithPast that also includes last_hidden_state.

    This avoids storing all layer hidden states when only the last layer is needed,
    significantly reducing memory usage for large models.

    Args:
        loss: Language modeling loss (optional)
        logits: Prediction scores of the language modeling head
        aux_loss: Auxiliary load balancing loss for MoE
        router_logits: Router logits for all layers
        past_key_values: Pre-computed key/value pairs for efficient generation
        attentions: Attention weights of all layers (only if output_attentions=True)
        last_hidden_state: Hidden state of the last layer (always available)
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


__all__ = [
    "ModelOutput",
    "BaseModelOutputWithPast",
    "CausalLMOutputWithPast",
    "CausalLMOutputWithPastAndLastHiddenState",
    "MoeCausalLMOutputWithPast",
    "MoeCausalLMOutputWithPastAndLastHiddenState",
    "MoeModelOutputWithPast",
]
