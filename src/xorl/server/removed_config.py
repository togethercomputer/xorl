"""Explicit tombstones for configuration fields removed from the server API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_AUTHORITATIVE_OWNERSHIP_MIGRATION = (
    "adapter-gradient ownership is authoritative-only; remove this field instead of selecting a legacy, "
    "observe, or shadow mode"
)
_ZORL_REMOVAL_MIGRATION = "ZORL was removed; remove this field and migrate training to forward_backward plus optim_step"


# One inventory is shared by YAML loading, CLI overrides, and public request
# validation. Keys are deliberately limited to retired names: unrelated unknown
# fields retain the existing rolling-upgrade/combined-config behavior.
REMOVED_CONFIGURATION_FIELDS: dict[str, str] = {
    "adapter_gradient_ownership_mode": _AUTHORITATIVE_OWNERSHIP_MIGRATION,
    "adapter_gradient_ownership_shadow_canary": _AUTHORITATIVE_OWNERSHIP_MIGRATION,
    "enable_zorl": _ZORL_REMOVAL_MIGRATION,
    "zorl_b_sigma": _ZORL_REMOVAL_MIGRATION,
    "zorl_num_perturbation_pairs": _ZORL_REMOVAL_MIGRATION,
    "zorl_a_refresh_interval": _ZORL_REMOVAL_MIGRATION,
    "zorl_antithetic_sampling": _ZORL_REMOVAL_MIGRATION,
    "zorl_a_init": _ZORL_REMOVAL_MIGRATION,
    "zorl_seed": _ZORL_REMOVAL_MIGRATION,
    "zorl": _ZORL_REMOVAL_MIGRATION,
    "zorl_config": _ZORL_REMOVAL_MIGRATION,
}


def reject_removed_configuration_fields(value: Any, *, context: str) -> Any:
    """Reject known retired keys at any mapping depth and return ``value`` unchanged."""

    found: list[tuple[str, str]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if not isinstance(node, Mapping):
            return
        for raw_key, child in node.items():
            key = str(raw_key)
            field_path = (*path, key)
            migration = REMOVED_CONFIGURATION_FIELDS.get(key)
            if migration is not None:
                found.append((".".join(field_path), migration))
                # A zorl/zorl_config parent tombstones the complete retired
                # object; do not emit redundant errors for its child aliases.
                continue
            visit(child, field_path)

    visit(value, ())
    if found:
        details = "; ".join(f"{field_path}: {migration}" for field_path, migration in found)
        raise ValueError(f"{context} contains removed configuration field(s): {details}")
    return value
