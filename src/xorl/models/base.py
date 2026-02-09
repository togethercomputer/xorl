"""Minimal base class for xorl models, replacing transformers.PreTrainedModel."""

from functools import partial

import torch
from torch import nn

from ..utils import logging


logger = logging.get_logger(__name__)


class XorlPreTrainedModel(nn.Module):
    """Base class for all xorl models.

    Provides the minimal infrastructure previously supplied by
    ``transformers.PreTrainedModel``:

    * ``post_init()`` — applies ``_init_weights`` to every sub-module.
    * ``_from_config()`` — class-method used by the loader to create a model
      from a config dict (handles ``torch_dtype`` and ``attn_implementation``).
    * ``gradient_checkpointing_enable / disable`` — propagates the flag and
      the checkpoint function to every module that declares
      ``gradient_checkpointing``.
    """

    config_class = None
    base_model_prefix = "model"
    _no_split_modules = []
    _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def post_init(self):
        """Initialize weights and apply weight tying."""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Override in subclass to initialize weights."""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def _from_config(cls, config, **kwargs):
        """Create a model instance from *config*.

        Handles the ``torch_dtype`` and ``attn_implementation`` keyword
        arguments that the loader passes through.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)
        attn_implementation = kwargs.pop("attn_implementation", None)

        if attn_implementation:
            config._attn_implementation = attn_implementation

        model = cls(config)

        if torch_dtype is not None:
            model.to(torch_dtype)

        return model

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        gc_func = partial(
            torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs
        )

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = gc_func

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
                module._gradient_checkpointing_func = None

    # ------------------------------------------------------------------
    # Checkpoint handler (per-model load/save transforms)
    # ------------------------------------------------------------------

    def get_checkpoint_handler(self, **kwargs):
        """Return a CheckpointHandler for this model, or None for default behavior.

        Override in model subclasses that need custom load/save transforms.

        Args:
            **kwargs: Context from the loading infrastructure:
                checkpoint_keys (Set[str]): keys in the checkpoint (for format detection)
                ep_rank (int): expert parallel rank
                ep_size (int): expert parallel group size
                is_broadcast (bool): True if using rank0_load_and_broadcast path
        """
        return None

    # ------------------------------------------------------------------
    # Embedding helpers (used by loader for weight tying)
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        pass
