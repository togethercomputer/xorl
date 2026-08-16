"""Production wiring for GLM-5.2 full-param block-FP8 training.

The admission must be activated by PRODUCTION model construction — the
ServerArguments -> config dict -> ModelRunner -> build_training_model chain —
with no bespoke entrypoint, and every invalid combination must fail closed
BEFORE any model is built.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

import xorl.server.runner.model_runner as model_runner_module
import xorl.server.runner.runner_dispatcher as runner_dispatcher_module
from xorl.server.orchestrator.orchestrator import _validate_operation_contract
from xorl.server.protocol.operations import OptimStepData
from xorl.server.runner.model_runner import FullParamOptimizerMutationFailure, ModelRunner
from xorl.server.runner.runner_dispatcher import RunnerDispatcher
from xorl.server.server_arguments import ServerArguments


pytestmark = [pytest.mark.cpu, pytest.mark.server]

_MODEL = "zai-org/GLM-5.2-FP8"


def _fullparam_args(**overrides) -> ServerArguments:
    values = dict(
        model_path=_MODEL,
        glm52_fullparam_fp8_training=True,
        glm52_fullparam_trainable_expert_layers=[3],
        freeze_router=False,
        moe_implementation="triton",
        ep_dispatch="alltoall",
        init_device="cuda",
        data_parallel_mode="fsdp2",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=16,
        ringattn_parallel_size=1,
        ulysses_parallel_size=16,
        data_parallel_replicate_size=1,
        data_parallel_shard_size=1,
        cp_fsdp_mode="all",
    )
    values.update(overrides)
    return ServerArguments(**values)


def test_server_arguments_accept_the_fullparam_row_and_carry_it_to_the_runner_config() -> None:
    arguments = _fullparam_args()
    config = arguments.to_config_dict()
    assert config["train"]["glm52_fullparam_fp8_training"] is True
    assert config["train"]["glm52_fullparam_trainable_expert_layers"] == [3]
    assert config["train"]["init_device"] == "cuda"
    assert config["train"]["expert_parallel_size"] == 16
    assert config["train"]["ulysses_parallel_size"] == 16
    assert config["train"]["enable_mixed_precision"] is True
    assert config["train"]["skip_param_upcast"] is True
    assert config["train"]["max_grad_norm"] == pytest.approx(1.0)
    assert _fullparam_args(max_grad_norm=0).to_config_dict()["train"]["max_grad_norm"] == 0.0


def test_legacy_sampler_save_alias_is_rejected_before_generic_dcp_dispatch() -> None:
    train_config = _fullparam_args().to_config_dict()["train"]
    with pytest.raises(RuntimeError, match="cannot use save_weights_for_sampler"):
        _validate_operation_contract("save_weights_for_sampler", train_config)
    _validate_operation_contract("save_state", train_config)
    _validate_operation_contract("save_weights_for_sampler", {})


def test_dispatcher_tail_failure_poison_is_classified_as_fullparam(monkeypatch) -> None:
    class Trainer:
        train_config = {"glm52_fullparam_fp8_training": True}
        _glm52_fullparam_poisoned = False

        def optim_step(self, **_kwargs):
            return {"sparse_delta_capture": {"rank": 0}}

    class Coordinator:
        @staticmethod
        def auto_load_if_evicted(_model_id):
            return False, None

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.world_size = 1
    dispatcher.trainer = Trainer()
    dispatcher._adapter_coordinator = Coordinator()
    monkeypatch.setattr(runner_dispatcher_module, "sparse_delta_capture_enabled", lambda _config: True)

    def fail_manifest(_captures):
        raise OSError("injected handler-tail failure")

    monkeypatch.setattr(runner_dispatcher_module, "write_sparse_source_delta_global_manifest", fail_manifest)
    payload = OptimStepData(model_id="default", sparse_delta_capture={"enabled": True})
    with pytest.raises(FullParamOptimizerMutationFailure, match="handler tail failed after mutation"):
        asyncio.run(dispatcher._handle_optim_step({"payload": payload}))
    assert dispatcher.trainer._glm52_fullparam_poisoned is True


@pytest.mark.parametrize("invalid", [None, True, "not-a-number", float("nan"), float("inf")])
def test_server_arguments_reject_invalid_gradient_clip_thresholds(invalid) -> None:
    with pytest.raises(ValueError, match="max_grad_norm must be a finite number"):
        _fullparam_args(max_grad_norm=invalid)


def test_gradient_clip_reaches_a_real_server_optimizer_step(monkeypatch) -> None:
    """Exercise the configured clipping threshold through the real server step.

    Keep the geometry to one two-element CPU parameter, but retain
    ``ModelRunner`` clipping, optimizer mutation, and cache-refresh wiring.
    Only distributed and platform plumbing is replaced.
    """

    train_config = _fullparam_args(max_grad_norm=1.0).to_config_dict()["train"]
    assert train_config["max_grad_norm"] == pytest.approx(1.0)

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.world_size = 1
    runner.is_sleeping = False
    runner._adapter_manager = None
    runner._use_distsignsgd = False
    runner._accumulated_valid_tokens = {}
    runner._accumulated_active_microbatches = {}
    runner._accumulated_active_voter_total = {}
    runner.train_config = train_config
    runner.lora_config = {
        "enable_lora": False,
        "enable_qlora": False,
        "merge_lora_interval": 0,
        "reset_optimizer_on_merge": False,
    }
    runner.model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        runner.model.weight.zero_()
    runner.model.weight.grad = torch.tensor([[3.0, 4.0]])
    runner.optimizer = torch.optim.SGD(runner.model.parameters(), lr=0.1)
    runner.pp_enabled = False
    runner.global_step = 0

    parallel_state = type("ParallelState", (), {"fsdp_group": None, "pp_group": None})()
    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(model_runner_module, "all_reduce", lambda value, group=None: value)
    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)
    monkeypatch.setattr(model_runner_module.torch.cuda, "empty_cache", lambda: None)

    import xorl.models.transformers.glm5.exact_fullparam_admission as admission

    refresh_calls = []
    monkeypatch.setattr(admission, "refresh_glm52_fullparam_caches", lambda model: refresh_calls.append(model) or 0)

    runner.train_config = dict(train_config)
    del runner.train_config["max_grad_norm"]
    weight_before_rejection = runner.model.weight.detach().clone()
    gradient_before_rejection = runner.model.weight.grad.detach().clone()
    with pytest.raises(ValueError, match="max_grad_norm must be configured"):
        ModelRunner.optim_step(runner, model_id="default")
    assert runner.global_step == 0
    assert torch.equal(runner.model.weight, weight_before_rejection)
    assert torch.equal(runner.model.weight.grad, gradient_before_rejection)

    runner.train_config = train_config
    result = ModelRunner.optim_step(runner, model_id="default")

    assert result["step"] == 1
    assert result["grad_norm"] == pytest.approx(5.0)
    assert torch.isfinite(torch.tensor(result["grad_norm"]))
    assert refresh_calls == [runner.model]
    # [3, 4] / ||[3, 4]|| is [0.6, 0.8], then SGD(lr=0.1).
    assert torch.allclose(runner.model.weight, torch.tensor([[-0.06, -0.08]]), atol=1e-6, rtol=0.0)

    def fresh_runner() -> ModelRunner:
        candidate = object.__new__(ModelRunner)
        candidate.__dict__ = runner.__dict__.copy()
        candidate._accumulated_valid_tokens = {}
        candidate._accumulated_active_microbatches = {}
        candidate._accumulated_active_voter_total = {}
        candidate.model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            candidate.model.weight.zero_()
        candidate.model.weight.grad = torch.tensor([[1.0, 0.0]])
        candidate.optimizer = torch.optim.SGD(candidate.model.parameters(), lr=0.1)
        candidate.global_step = 0
        return candidate

    # A failure in the common command tail is just as fatal as a refresh
    # failure: weights and the step counter have already advanced.
    tail_runner = fresh_runner()

    def fail_tail():
        raise OSError("injected tail failure")

    tail_runner._maybe_merge_lora = fail_tail
    with pytest.raises(FullParamOptimizerMutationFailure, match="failed after optimizer mutation"):
        ModelRunner.optim_step(tail_runner, model_id="default")
    assert tail_runner.global_step == 1
    assert tail_runner._glm52_fullparam_poisoned is True
    assert not torch.equal(tail_runner.model.weight, torch.zeros_like(tail_runner.model.weight))

    # An optimizer implementation may mutate before raising. Treat the call
    # itself as the mutation boundary rather than waiting for it to return.
    step_runner = fresh_runner()
    real_step = step_runner.optimizer.step

    def mutate_then_fail():
        real_step()
        raise OSError("injected optimizer failure")

    step_runner.optimizer.step = mutate_then_fail
    with pytest.raises(FullParamOptimizerMutationFailure, match="failed after optimizer mutation"):
        ModelRunner.optim_step(step_runner, model_id="default")
    assert step_runner.global_step == 0
    assert step_runner._glm52_fullparam_poisoned is True
    assert not torch.equal(step_runner.model.weight, torch.zeros_like(step_runner.model.weight))

    runner.model.weight.grad = torch.tensor([[1.0, 0.0]])

    def fail_refresh(_model):
        raise OSError("injected publication failure")

    monkeypatch.setattr(admission, "refresh_glm52_fullparam_caches", fail_refresh)
    with pytest.raises(FullParamOptimizerMutationFailure, match="failed after optimizer mutation"):
        ModelRunner.optim_step(runner, model_id="default")
    assert runner.global_step == 2
    assert runner._glm52_fullparam_poisoned is True

    failed_step_weight = runner.model.weight.detach().clone()
    with pytest.raises(FullParamOptimizerMutationFailure, match="restart from the last committed checkpoint"):
        ModelRunner.optim_step(runner, model_id="default")
    assert runner.global_step == 2
    assert torch.equal(runner.model.weight, failed_step_weight)


def test_server_arguments_fail_closed_on_invalid_fullparam_rows() -> None:
    with pytest.raises(ValueError, match="exclusive full-weight mode"):
        _fullparam_args(enable_lora=True)
    with pytest.raises(ValueError, match="exclusive full-weight mode"):
        _fullparam_args(enable_fp8_training=True)
    with pytest.raises(ValueError, match="exclusive full-weight mode"):
        _fullparam_args(enable_qarl=True)
    with pytest.raises(ValueError, match="non-empty"):
        _fullparam_args(glm52_fullparam_trainable_expert_layers=None)
    with pytest.raises(ValueError, match="non-empty"):
        _fullparam_args(glm52_fullparam_trainable_expert_layers=[])
    with pytest.raises(ValueError, match="unique non-negative"):
        _fullparam_args(glm52_fullparam_trainable_expert_layers=[3, 3])
    with pytest.raises(ValueError, match="unique non-negative"):
        _fullparam_args(glm52_fullparam_trainable_expert_layers=[-1])
    with pytest.raises(ValueError, match="freeze_router=True"):
        _fullparam_args(freeze_router=True)
    with pytest.raises(ValueError, match="moe_implementation"):
        _fullparam_args(moe_implementation=None)
    with pytest.raises(ValueError, match="ep_dispatch"):
        _fullparam_args(ep_dispatch="deepep")
    with pytest.raises(ValueError, match="enable_mixed_precision=True"):
        _fullparam_args(enable_mixed_precision=False)
    # A dangling scope without the mode is a config bug, not a silent no-op.
    with pytest.raises(ValueError, match="only meaningful"):
        ServerArguments(model_path=_MODEL, glm52_fullparam_trainable_expert_layers=[3])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("init_device", "meta"),
        ("data_parallel_mode", "none"),
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("expert_parallel_size", 1),
        ("ringattn_parallel_size", 2),
        ("ulysses_parallel_size", 1),
        ("data_parallel_replicate_size", 2),
        ("data_parallel_shard_size", 16),
        ("cp_fsdp_mode", "none"),
        ("lm_head_tensor_parallel_size", 16),
        ("fsdp_sharded_lm_head_loss", True),
        ("enable_full_shard", False),
        ("reshard_after_forward", False),
    ],
)
def test_server_arguments_carries_fullparam_topology_without_a_family_allowlist(field, value) -> None:
    arguments = _fullparam_args(**{field: value})
    assert getattr(arguments, field) == value


def test_build_training_model_refuses_semantic_fullparam_contradictions_before_any_build() -> None:
    from xorl.trainers.model_builder import build_training_model

    common = dict(config_path="nonexistent/config", weights_path="nonexistent/weights")

    with pytest.raises(ValueError, match="exclusive full-weight mode"):
        build_training_model(glm52_fullparam_fp8_training=True, enable_lora=True, **common)
    with pytest.raises(ValueError, match="exclusive full-weight mode"):
        build_training_model(glm52_fullparam_fp8_training=True, enable_qlora=True, enable_lora=True, **common)
    with pytest.raises(ValueError, match="exclusive full-weight mode"):
        build_training_model(glm52_fullparam_fp8_training=True, enable_fp8_training=True, **common)
    with pytest.raises(ValueError, match="non-empty glm52_fullparam_trainable_expert_layers"):
        build_training_model(glm52_fullparam_fp8_training=True, **common)
    with pytest.raises(ValueError, match="non-empty glm52_fullparam_trainable_expert_layers"):
        build_training_model(
            glm52_fullparam_fp8_training=True,
            glm52_fullparam_trainable_expert_layers=[],
            **common,
        )
    with pytest.raises(ValueError, match="unique non-negative"):
        build_training_model(
            glm52_fullparam_fp8_training=True,
            glm52_fullparam_trainable_expert_layers=[3, 3],
            **common,
        )
    with pytest.raises(ValueError, match="only meaningful"):
        build_training_model(glm52_fullparam_trainable_expert_layers=[3], **common)

    mode_common = dict(
        glm52_fullparam_fp8_training=True,
        glm52_fullparam_trainable_expert_layers=[3],
        freeze_router=False,
        merge_qkv=True,
        moe_implementation="triton",
        ep_dispatch="alltoall",
        **common,
    )
    for override in (
        {"freeze_router": True},
        {"merge_qkv": False},
        {"moe_implementation": None},
        {"ep_dispatch": "deepep"},
    ):
        with pytest.raises(ValueError, match="rejects unsupported configuration"):
            build_training_model(**{**mode_common, **override})


def test_publish_dir_wiring_carries_to_train_config_and_fails_closed(tmp_path) -> None:
    publish_dir = str(tmp_path / "publish")
    arguments = _fullparam_args(glm52_fullparam_publish_dir=publish_dir)
    config = arguments.to_config_dict()
    assert config["train"]["glm52_fullparam_publish_dir"] == publish_dir

    assert _fullparam_args().to_config_dict()["train"]["glm52_fullparam_publish_dir"] is None

    with pytest.raises(ValueError, match="only meaningful"):
        ServerArguments(model_path=_MODEL, glm52_fullparam_publish_dir="publish")
    with pytest.raises(ValueError, match="non-empty path"):
        _fullparam_args(glm52_fullparam_publish_dir="   ")


def test_validate_glm5_training_mode_fullparam_branch() -> None:
    from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
    from xorl.models.transformers.glm5.support import validate_glm5_training_mode

    config = Glm5Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        mlp_layer_types=["dense"],
        pad_token_id=0,
    )
    valid = dict(
        enable_qlora=False,
        freeze_router=False,
        merge_qkv=True,
        moe_implementation="triton",
        ep_dispatch="alltoall",
        glm52_fullparam_fp8_training=True,
    )
    validate_glm5_training_mode(config, **valid)

    with pytest.raises(ValueError, match="cannot be combined with QLoRA"):
        validate_glm5_training_mode(config, **{**valid, "enable_qlora": True})
    with pytest.raises(ValueError, match="freeze_router=True"):
        validate_glm5_training_mode(config, **{**valid, "freeze_router": True})
    with pytest.raises(ValueError, match="ep_dispatch"):
        validate_glm5_training_mode(config, **{**valid, "ep_dispatch": "deepep"})
    with pytest.raises(ValueError, match="moe_implementation"):
        validate_glm5_training_mode(config, **{**valid, "moe_implementation": "eager"})
    with pytest.raises(ValueError, match="merge_qkv"):
        validate_glm5_training_mode(config, **{**valid, "merge_qkv": False})
    # The mode gate does not fire for non-GLM5 configs.
    validate_glm5_training_mode(object(), **valid)


def test_glm52_contract_flag_resolution_for_the_fullparam_lane() -> None:
    """Full-param is a GLM exact family without claiming the frozen scoring
    contract.  Downstream consumers use the neutral family stamp rather than
    a full-param-specific dispatch key."""

    from types import SimpleNamespace

    from xorl.models.exact_contract import (
        EXACT_CONTRACT_FAMILY_GLM52,
        glm52_exact_forward_enabled,
        glm52_fullparam_training_enabled,
        resolve_exact_contract_family,
    )

    config = SimpleNamespace(
        _glm52_block_fp8_qlora=False,
        _glm52_fullparam_training=True,
        _glm52_exact_contract=False,
    )
    assert glm52_fullparam_training_enabled(config)
    assert glm52_exact_forward_enabled(config)
    assert resolve_exact_contract_family(config) == EXACT_CONTRACT_FAMILY_GLM52

    # The production model resolver refuses the mode before building any
    # non-GLM model; full-param cannot manufacture a GLM family stamp.
    from transformers import PretrainedConfig

    from xorl.models.auto import build_foundation_model

    with pytest.raises(ValueError, match="official canonical GLM-5.2"):
        build_foundation_model(
            PretrainedConfig(),
            glm52_fullparam_fp8_training=True,
        )

    # Full-param admission accepts the neutral classifier's input shape: lane
    # flag on, scoring-only flag off.  Claiming both would misdescribe a
    # trainable trunk as frozen.
    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        _validate_official_fullparam_config,
    )

    official = SimpleNamespace(
        vocab_size=154880,
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=78,
        num_attention_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        v_head_dim=256,
        first_k_dense_replace=3,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        attention_bias=False,
        tie_word_embeddings=False,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
        _ep_dispatch="alltoall",
        _glm52_fullparam_training=True,
        _glm52_exact_contract=False,
    )
    _validate_official_fullparam_config(official)  # must not raise


def test_fullparam_composites_are_exempt_from_the_decoder_bf16_cast() -> None:
    """The FSDP2 wrap folds decoder-layer submodules into a BF16 param-cast
    policy unless their classes are in get_ignore_modules_in_mixed_precision.
    The full-param dense composites and routers own FP32 masters + byte
    caches, and a BF16 cast would corrupt both. Expert banks take the
    dedicated expert-FSDP branch (fsdp_requires_full_precision) instead."""

    from types import SimpleNamespace

    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        Glm52FullParamTopkRouter,
    )
    from xorl.models.transformers.glm5.exact_fullparam_experts import (
        Glm52FullParamBlockFP8RoutedExperts,
    )
    from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
        Glm52ExactTP1BlockFP8FullParamLinear,
        Glm52FullParamDenseMLP,
    )
    from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM

    ignored = Glm5ForCausalLM.get_ignore_modules_in_mixed_precision(
        SimpleNamespace(config=SimpleNamespace(quantization_config={"quant_method": "fp8"}))
    )
    assert Glm52FullParamDenseMLP in ignored
    assert Glm52FullParamTopkRouter in ignored
    assert Glm52ExactTP1BlockFP8FullParamLinear in ignored
    # Banks must NOT be double-wrapped: they use the expert branch's
    # no-cast handling via fsdp_requires_full_precision.
    assert Glm52FullParamBlockFP8RoutedExperts not in ignored
    assert Glm52FullParamBlockFP8RoutedExperts.fsdp_requires_full_precision is True

    # No quantization config: the exemption list stays absent (BF16 lanes).
    assert (
        Glm5ForCausalLM.get_ignore_modules_in_mixed_precision(
            SimpleNamespace(config=SimpleNamespace(quantization_config=None))
        )
        is None
    )
