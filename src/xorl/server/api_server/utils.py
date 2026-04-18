"""Pure utility functions and constants for the API server."""

import math
import re
from typing import Any, Dict, Optional

from fastapi import HTTPException, status


# Error message for model_id not registered
MODEL_ID_NOT_REGISTERED_ERROR = (
    "model_id '{model_id}' has not been registered. "
    "You must call /api/v1/create_model or /api/v1/create_session first to register your model_id before "
    "calling other /api/v1/ endpoints. "
    "Try initializing the TrainingClient by calling create_lora_training_client on the ServiceClient."
)


def sanitize_float_for_json(value: Any) -> Any:
    """
    Sanitize float values for JSON serialization.

    Converts NaN and Inf to None (which is valid JSON null).
    Leaves other values unchanged.

    Args:
        value: Value to sanitize

    Returns:
        Sanitized value (None if NaN/Inf, otherwise original value)
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def sanitize_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively sanitize a dictionary for JSON serialization.

    Converts all NaN and Inf float values to None.

    Args:
        data: Dictionary to sanitize

    Returns:
        Sanitized dictionary
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = sanitize_dict_for_json(value)
        elif isinstance(value, list):
            result[key] = [sanitize_float_for_json(v) for v in value]
        else:
            result[key] = sanitize_float_for_json(value)
    return result


# Regex pattern for valid model_id: alphanumeric, underscore, hyphen only
# Must start with alphanumeric, can contain alphanumeric, underscore, hyphen
# Length: 1-128 characters
MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


def validate_model_id(model_id: Optional[str]) -> str:
    """
    Validate that model_id contains only allowed characters.

    If model_id is None or empty, returns "default".

    model_id must:
    - Start with an alphanumeric character (a-z, A-Z, 0-9)
    - Contain only alphanumeric characters, underscores, or hyphens
    - Be 1-128 characters long

    Args:
        model_id: The model identifier to validate

    Returns:
        The validated model_id, or "default" if model_id was None or empty

    Raises:
        HTTPException: If model_id contains invalid characters
    """
    # Default to "default" if model_id is None or empty
    if not model_id:
        return "default"

    if not MODEL_ID_PATTERN.match(model_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_id '{model_id}'. model_id must start with an alphanumeric character "
            f"and contain only alphanumeric characters, underscores (_), or hyphens (-). "
            f"Length must be 1-128 characters.",
        )

    return model_id
