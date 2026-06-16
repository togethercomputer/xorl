"""
API Server package for xorl training system.

Provides unified FastAPI server with integrated OrchestratorClient for
communicating with the training engine backend.
"""

from xorl.server.api_server.api_types import (
    AdamParams,
    Datum,
    DatumInput,
    ErrorResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    ForwardRequest,
    ForwardResponse,
    HealthCheckResponse,
    LoadWeightsRequest,
    LoadWeightsResponse,
    LoRAConfigRequest,
    LossFnOutput,
    OptimizerConfigRequest,
    OptimStepRequest,
    OptimStepResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    TensorData,
)
from xorl.server.api_server.orchestrator_client import OrchestratorClient
from xorl.server.api_server.server import APIServer


__all__ = [
    "OrchestratorClient",
    "APIServer",
    # Request/Response models
    "AdamParams",
    "TensorData",
    "Datum",
    "DatumInput",
    "ForwardRequest",
    "ForwardResponse",
    "ForwardBackwardRequest",
    "ForwardBackwardResponse",
    "LossFnOutput",
    "LoRAConfigRequest",
    "OptimizerConfigRequest",
    "OptimStepRequest",
    "OptimStepResponse",
    "SaveWeightsRequest",
    "SaveWeightsResponse",
    "LoadWeightsRequest",
    "LoadWeightsResponse",
    "SaveWeightsForSamplerRequest",
    "SaveWeightsForSamplerResponse",
    "HealthCheckResponse",
    "ErrorResponse",
]
