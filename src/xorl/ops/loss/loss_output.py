from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class LossOutput:
    """Standardized return type for all loss functions.

    ``metric_ops`` tags ``metrics`` keys whose cross-mb / cross-rank composition
    isn't the default mean (``"min"``/``"max"``). The sidecar (rather than a
    tagged-value type in ``metrics``) keeps the metrics dict directly
    JSON-serializable for untagged consumers.
    """

    loss: torch.Tensor
    per_token_logprobs: Optional[torch.Tensor] = None
    per_token_loss: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, Any]] = None
    metric_ops: Optional[Dict[str, str]] = None
