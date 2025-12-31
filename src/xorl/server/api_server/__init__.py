"""
API Server package for xorl training system.

Provides unified FastAPI server with integrated EngineClient for
communicating with the training engine backend.
"""

from xorl.server.api_server.engine_client import EngineClient
from xorl.server.api_server.api_server import APIServer
from xorl.server.api_server.api_types import (
    TensorData,
    Datum,
    DatumInput,
    ForwardRequest,
    ForwardResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    LossFnOutput,
    AdamParams,
    OptimStepRequest,
    OptimStepResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    LoadWeightsRequest,
    LoadWeightsResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    SaveLoRAOnlyRequest,
    SaveLoRAOnlyResponse,
    HealthCheckResponse,
    ErrorResponse,
)

__all__ = [
    "EngineClient",
    "APIServer",
    # Request/Response models
    "TensorData",
    "Datum",
    "DatumInput",
    "ForwardRequest",
    "ForwardResponse",
    "ForwardBackwardRequest",
    "ForwardBackwardResponse",
    "LossFnOutput",
    "AdamParams",
    "OptimStepRequest",
    "OptimStepResponse",
    "SaveWeightsRequest",
    "SaveWeightsResponse",
    "LoadWeightsRequest",
    "LoadWeightsResponse",
    "SaveWeightsForSamplerRequest",
    "SaveWeightsForSamplerResponse",
    "SaveLoRAOnlyRequest",
    "SaveLoRAOnlyResponse",
    "HealthCheckResponse",
    "ErrorResponse",
]
