"""Helpers for writing simulator configs that are runnable by XORL."""

from __future__ import annotations

import copy
from typing import Any


SIMULATOR_ONLY_TOP_LEVEL_SECTIONS = ("simulator", "_simulator")


def runtime_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with simulator-only metadata removed."""

    rendered = copy.deepcopy(config)
    for section_name in SIMULATOR_ONLY_TOP_LEVEL_SECTIONS:
        rendered.pop(section_name, None)
    return rendered
