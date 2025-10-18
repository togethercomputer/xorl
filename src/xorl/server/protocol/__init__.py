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
from xorl.server.protocol.operations import (  # noqa: F401
    AbortData,
    AdapterStateData,
    EmptyData,
    KillSessionData,
    LoadStateData,
    ModelPassData,
    OperationPayload,
    OptimStepData,
    RegisterAdapterData,
    SaveFullWeightsData,
    SaveLoraOnlyData,
    SaveStateData,
    SyncWeightsData,
    payload_from_dict,
    payload_to_dict,
)

from xorl.server.protocol.api_orchestrator import (  # noqa: F401
    OrchestratorOutputs,
    OrchestratorRequest,
    OutputType,
    RequestType,
    create_error_output,
    create_forward_backward_output,
    create_health_check_output,
    create_load_adapter_state_output,
    create_load_state_output,
    create_optim_step_output,
    create_save_adapter_state_output,
    create_save_lora_only_output,
    create_save_state_output,
    create_sleep_output,
    create_sync_weights_output,
    create_wake_up_output,
    get_operation_from_request,
    is_streaming_output,
    validate_output,
    validate_request,
)
from xorl.server.protocol.orchestrator_runner import (  # noqa: F401
    BaseMessage,
    RunnerDispatchCommand,
    MessageType,
    RunnerAck,
    RunnerReady,
    RunnerResponse,
    create_ack_for_request,
    create_response_for_request,
    deserialize_message,
    serialize_message,
)
