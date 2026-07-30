"""DeepSeek V4 model.

**Import side effect:** loading this module calls
:func:`_register_with_transformers` at the bottom of this file, which mutates
the global ``transformers.AutoConfig`` and ``AutoModelForCausalLM`` registries
to map ``model_type="deepseek_v4"`` (the upstream HF Flash convention) to
this package's ``DeepseekV4Config`` / ``DeepseekV4ForCausalLM``. The call is
idempotent — re-imports trap the ``ValueError`` from a duplicate
``AutoConfig.register``. The behavior can be opted out via
``XORL_DSV4_AUTOREGISTER=0`` for downstream consumers that import xorl
solely to introspect classes and want the upstream registry untouched (or
that want to defer to a future first-party ``transformers.DeepseekV4Config``).
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
    "deepseek_v4"``, which transformers does not yet ship a class for.
    Registering our vendored ``DeepseekV4Config`` + ``DeepseekV4ForCausalLM``
    against that ``model_type`` makes ``AutoConfig.from_pretrained(snapshot)``
    and ``AutoModelForCausalLM.from_pretrained(snapshot)`` work end-to-end.

    ``DeepseekV4Config`` itself uses ``model_type = "xorl_deepseek_v4"``
    (an internal namespace to distinguish from any future upstream
    transformers class); we register the *upstream* name explicitly so
    AutoConfig dispatches correctly when reading the HF disk format.
    """
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    upstream_model_type = "deepseek_v4"

    # ``AutoConfig.register`` raises ``ValueError`` if the model_type is
    # already registered — be idempotent so re-imports don't crash.
    try:
        AutoConfig.register(upstream_model_type, DeepseekV4Config)
    except ValueError:
        pass

    try:
        AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM)
    except ValueError:
        pass


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
