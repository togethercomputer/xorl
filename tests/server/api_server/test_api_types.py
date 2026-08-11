"""
Tests for API request/response types (Pydantic models).

Tests validation, serialization, and edge cases for the updated API structure.
"""

import pytest
from pydantic import ValidationError

from xorl.server.api_server.api_types import (
    CreateModelRequest,
    CreateSessionRequest,
    Datum,
    DatumInput,
    ForwardBackwardRequest,
    TensorData,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_session_api_configuration_compatibility_policy():
    cases = (
        (
            CreateModelRequest,
            {
                "base_model": "Qwen/Qwen3-8B",
                "zorl_config": {"enabled": True},
            },
            "zorl_config",
            "ZORL was removed",
        ),
        (
            CreateModelRequest,
            {
                "base_model": "Qwen/Qwen3-8B",
                "lora_config": {"rank": 8, "adapter_gradient_ownership_mode": "legacy"},
            },
            "lora_config.adapter_gradient_ownership_mode",
            "authoritative-only",
        ),
        (
            CreateSessionRequest,
            {
                "lora_config": {"rank": 8, "zorl_seed": 123},
            },
            "lora_config.zorl_seed",
            "ZORL was removed",
        ),
    )
    for request_type, payload, field_path, migration in cases:
        with pytest.raises(ValidationError) as exc_info:
            request_type.model_validate(payload)

        message = str(exc_info.value)
        assert field_path in message
        assert migration in message

    request = CreateModelRequest.model_validate(
        {
            "base_model": "Qwen/Qwen3-8B",
            "rolling_client_metadata": {"version": 2},
            "lora_config": {"rank": 8, "future_lora_hint": True},
        }
    )

    assert request.lora_config is not None
    assert request.lora_config.model_extra == {"future_lora_hint": True}

    forward = ForwardBackwardRequest(
        session_id="legacy-session",
        forward_backward_input=DatumInput(data=[Datum(model_input={"input_ids": [1]}, loss_fn_inputs={"labels": [1]})]),
    )
    assert forward.model_id == "legacy-session"


def test_tensor_data_preserves_ranked_training_inputs():
    datum = Datum(
        model_input={"input_ids": TensorData(data=[1, 2, 3], dtype="int64", shape=[3])},
        loss_fn_inputs={
            "teacher_hidden_states": TensorData(
                data=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                dtype="float32",
                shape=[3, 2],
            ),
            "routed_experts": TensorData(data=list(range(12)), dtype="int64", shape=[2, 3, 2]),
        },
    )

    plain = datum.to_plain_dict()

    assert plain["model_input"]["input_ids"] == [1, 2, 3]
    assert plain["loss_fn_inputs"]["teacher_hidden_states"] == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
    ]
    assert plain["loss_fn_inputs"]["routed_experts"] == [
        [[0, 1], [2, 3], [4, 5]],
        [[6, 7], [8, 9], [10, 11]],
    ]
