"""OLMo-2-specific tensor-parallel ``ParallelStyle``.

OLMo-2 declares ``q_norm``/``k_norm`` over the full ``num_heads * head_dim``
axis (rather than per-head + reshape-first like every other model in the
repo). Under colwise ``q_proj``/``k_proj``, the q/k tensors arrive with a
sharded hidden axis, so a full-hidden RMSNorm weight can't be applied
directly. ``LocalAxisRMSNormShard`` shards the 1-D RMSNorm weight along
dim 0 so each rank's slice matches its local q/k slice; the
DTensor-aware ``Olmo2QKRMSNorm.forward`` (in ``modeling_olmo2.py``) then
runs the fused op on locals and computes a local-axis RMS — matching
HuggingFace's ``Olmo2RMSNorm`` reference behavior.

This style does not compose with the other models in this repo: per-head
QK norm reshape-first models stay on stock ``ColwiseParallel`` (their
norm weight is ``[head_dim]`` and the post-reshape activation has full
``head_dim`` per rank). The custom style is therefore scoped to
``olmo2/`` and not exported as generic TP infrastructure.
"""

from torch import nn
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


class LocalAxisRMSNormShard(ParallelStyle):
    """Shard a 1-D RMSNorm weight along dim 0 with no input/output redistribute.

    See module docstring for the OLMo-2 use case. The companion forward
    is ``Olmo2QKRMSNorm.forward`` which detects the Shard(0) DTensor weight
    and runs ``F.rms_norm`` on local tensors.
    """

    def _apply(self, module: nn.Module, device_mesh) -> nn.Module:
        for name, param in module.named_parameters(recurse=False):
            sharded = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, sharded)
        return module
