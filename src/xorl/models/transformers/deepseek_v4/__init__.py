"""DeepSeek V4 model.

**Import side effect:** loading this module calls
:func:`_register_with_transformers` at the bottom of this file, which mutates
the global ``transformers.AutoConfig`` and ``AutoModelForCausalLM`` registries
to map ``model_type="deepseek_v4"`` (the upstream HF Flash convention) to
this package's ``DeepseekV4Config`` / ``DeepseekV4ForCausalLM``. The call is
idempotent — ``exist_ok=True`` makes re-imports a no-op and deliberately
overrides the first-party ``deepseek_v4`` mapping that transformers ships
since 5.4: XoRL's training stack depends on the vendored classes, not the
upstream ones. The behavior can be opted out via
``XORL_DSV4_AUTOREGISTER=0`` for downstream consumers that import xorl
solely to introspect classes and want the upstream registry untouched.
"""

import os

from .checkpoint_handler import (
    DeepseekV4CheckpointHandler,
    LoadSummary,
    load_hf_state_dict_into_model,
    stream_load_hf_directory_into_model,
)
from .configuration_deepseek_v4 import DeepseekV4Config
from .modeling_deepseek_v4 import (
    DeepSeekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4MoE,
    DeepseekV4PreTrainedModel,
    cast_dsv4_model_dtype,
)


# Wire config_class on the PreTrainedModel base for downstream loaders.
DeepseekV4PreTrainedModel.config_class = DeepseekV4Config


def _register_with_transformers() -> None:
    """Register DSv4 with ``transformers.AutoConfig`` / ``AutoModelForCausalLM``.

    The on-disk HF Flash ``config.json`` declares ``model_type =
    "deepseek_v4"``. Registering our vendored ``DeepseekV4Config`` +
    ``DeepseekV4ForCausalLM`` against that ``model_type`` makes
    ``AutoConfig.from_pretrained(snapshot)`` and
    ``AutoModelForCausalLM.from_pretrained(snapshot)`` resolve to the xorl
    classes end-to-end. transformers ships its own ``deepseek_v4`` mapping
    since 5.4, so the registration must override it (``exist_ok=True``):
    silently deferring to the upstream class would hand the training stack a
    config/model pair it was never validated against.

    ``DeepseekV4Config`` itself uses ``model_type = "xorl_deepseek_v4"``
    (an internal namespace to distinguish from any future upstream
    transformers class); we register the *upstream* name explicitly so
    AutoConfig dispatches correctly when reading the HF disk format.
    """
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    upstream_model_type = "deepseek_v4"

    # ``exist_ok=True`` both keeps re-imports idempotent and overrides the
    # first-party ``deepseek_v4`` mapping transformers ships since 5.4. A
    # ``try/except ValueError`` here would silently lose that override and
    # resolve snapshots to the upstream classes instead.
    AutoConfig.register(upstream_model_type, DeepseekV4Config, exist_ok=True)
    AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)


if os.environ.get("XORL_DSV4_AUTOREGISTER", "1") != "0":
    _register_with_transformers()


__all__ = [
    "DeepseekV4Config",
    "DeepSeekV4Attention",
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4MLP",
    "DeepseekV4Model",
    "DeepseekV4MoE",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4CheckpointHandler",
    "LoadSummary",
    "cast_dsv4_model_dtype",
    "load_hf_state_dict_into_model",
    "stream_load_hf_directory_into_model",
]
