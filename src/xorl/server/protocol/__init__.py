"""
Server Protocol — cross-cutting message definitions.

Re-exports every public symbol from the three protocol modules so that
callers can do either:

    from xorl.server.protocol import OrchestratorRequest, RunnerDispatchCommand
    from xorl.server.protocol.operations import ModelPassData, OptimStepData
    from xorl.server.protocol.api_orchestrator import OrchestratorRequest
    from xorl.server.protocol.orchestrator_runner import RunnerResponse
"""

# Typed operation payloads
from xorl.server.protocol.api_orchestrator import (  # noqa: F401
    OrchestratorOutputs,
    OrchestratorRequest,
    OutputType,
    RequestType,
)
from xorl.server.protocol.operations import (  # noqa: F401
    AbortData,
    AbortGradientEpochData,
    AdapterStateData,
    EmptyData,
    KillSessionData,
    LoadStateData,
    ModelPassData,
    OperationPayload,
    OptimStepData,
    RegisterAdapterData,
    RegisterSessionData,
    SaveFullWeightsData,
    SaveLoraOnlyData,
    SaveStateData,
    SyncWeightsData,
    payload_from_dict,
    payload_to_dict,
)
from xorl.server.protocol.orchestrator_runner import (  # noqa: F401
    BaseMessage,
    MessageType,
    RunnerAck,
    RunnerDispatchCommand,
    RunnerReady,
    RunnerResponse,
    deserialize_message,
    serialize_message,
)
