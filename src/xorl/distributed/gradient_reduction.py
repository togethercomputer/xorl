"""Validated contracts for replicated-gradient reduction."""

from enum import Enum


class GradientReductionDomain(str, Enum):
    """How a replicated parameter's EP gradient is reduced."""

    NONE = "none"
    EP_SUM = "ep_sum"
    ALREADY_REDUCED = "already_reduced"


def validate_gradient_reduction_domain(value: str | GradientReductionDomain) -> GradientReductionDomain:
    """Normalize a reduction-domain value and reject unknown contracts."""

    if isinstance(value, GradientReductionDomain):
        return value
    try:
        return GradientReductionDomain(value)
    except ValueError as exc:
        raise ValueError(
            f"Unknown gradient reduction domain {value!r}; expected one of "
            f"{', '.join(domain.value for domain in GradientReductionDomain)}"
        ) from exc
