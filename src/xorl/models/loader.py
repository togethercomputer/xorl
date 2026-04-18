from typing import Callable

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
)

from ..utils import logging
from .module_utils import all_ranks_load_weights, init_empty_weights
from .registry import ModelRegistry


logger = logging.get_logger(__name__)


class ModelLoader:
    """Unified model loader for both HuggingFace and custom xorl models.

    Takes a model factory callable (e.g., ``AutoModelForCausalLM.from_config``
    or ``model_cls._from_config``) and handles meta-init, device placement,
    and weight loading.
    """

    def __init__(self, model_factory: Callable[..., nn.Module], description: str = ""):
        self.model_factory = model_factory
        self.description = description

    def load_model(
        self,
        init_kwargs: dict,
        weights_path: str | None = None,
        empty_init: bool = False,
        init_device: str = "cuda",
    ) -> nn.Module:
        logger.info_rank0(
            f"Loading model ({self.description})\n"
            f"init_device: {init_device}, empty_init: {empty_init}, weights_path: {weights_path}"
        )

        if weights_path is None:
            if init_device == "meta":
                with init_empty_weights():
                    model = self.model_factory(**init_kwargs)
            else:
                with torch.device(init_device):
                    model = self.model_factory(**init_kwargs)
        else:
            with init_empty_weights():
                model = self.model_factory(**init_kwargs)
            if not empty_init:
                all_ranks_load_weights(model, weights_path, init_device)

        return model


def _get_model_arch_from_config(model_config):
    arch_name = model_config.architectures
    if isinstance(arch_name, list):
        arch_name = arch_name[0]
    return arch_name


def get_loader(model_config) -> ModelLoader:
    model_arch = _get_model_arch_from_config(model_config)

    if model_arch in ModelRegistry.supported_models:
        model_cls = ModelRegistry.get_model_cls_from_model_arch(model_arch)
        return ModelLoader(model_cls._from_config, description=f"xorl/{model_arch}")

    if "ForCausalLM" in model_arch and type(model_config) in AutoModelForCausalLM._model_mapping.keys():
        return ModelLoader(AutoModelForCausalLM.from_config, description=f"huggingface/{model_arch}")

    return ModelLoader(AutoModel.from_config, description=f"huggingface/{model_arch}")
