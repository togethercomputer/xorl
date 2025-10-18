from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class LossOutput:
    """Standardized return type for all loss functions."""

    loss: torch.Tensor
    per_token_logprobs: Optional[torch.Tensor] = None
    per_token_loss: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, Any]] = None
