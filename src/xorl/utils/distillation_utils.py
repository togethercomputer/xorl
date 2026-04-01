"""Utility functions for distillation training."""

import json
import logging

from huggingface_hub import hf_hub_download
from safetensors import safe_open


logger = logging.getLogger(__name__)


def load_lm_head_from_hf_model(model_id: str, token: str = None):
    """
    Load only the lm_head weights from a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
        token: HuggingFace token if model is private

    Returns:
        lm_head weight tensor
    """
    # 1. Download the index file
    index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", token=token)

    # 2. Parse the index to find which shard contains lm_head
    with open(index_path, "r") as f:
        index = json.load(f)

    # 3. Find the shard file for lm_head.weight
    weight_map = index.get("weight_map", {})
    lm_head_file = weight_map.get("lm_head.weight")

    if not lm_head_file:
        raise ValueError(f"lm_head.weight not found in {model_id}")

    logger.info(f"lm_head.weight is in: {lm_head_file}")

    # 4. Download only that specific shard
    shard_path = hf_hub_download(repo_id=model_id, filename=lm_head_file, token=token)

    # 5. Load only the lm_head.weight from the shard
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        lm_head_weight = f.get_tensor("lm_head.weight")

    return lm_head_weight
