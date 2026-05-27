import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _load_server_arguments_fn():
    module_path = Path(__file__).resolve().parents[2] / "src" / "xorl" / "server" / "launcher.py"
    spec = importlib.util.spec_from_file_location("xorl_test_launcher", module_path)
    assert spec is not None and spec.loader is not None

    fake_api_server_pkg = types.ModuleType("xorl.server.api_server")
    fake_api_server_pkg.__path__ = []
    fake_api_server_mod = types.ModuleType("xorl.server.api_server.server")
    fake_api_server_mod.APIServer = object
    fake_api_server_pkg.server = fake_api_server_mod

    fake_orchestrator_pkg = types.ModuleType("xorl.server.orchestrator")
    fake_orchestrator_pkg.__path__ = []
    fake_orchestrator_mod = types.ModuleType("xorl.server.orchestrator.orchestrator")
    fake_orchestrator_mod.Orchestrator = object
    fake_orchestrator_pkg.orchestrator = fake_orchestrator_mod

    fake_utils_pkg = types.ModuleType("xorl.server.utils")
    fake_utils_pkg.__path__ = []
    fake_network_mod = types.ModuleType("xorl.server.utils.network")
    fake_network_mod.read_address_file = lambda *args, **kwargs: None
    fake_utils_pkg.network = fake_network_mod
    fake_session_spec_mod = types.ModuleType("xorl.server.session_spec")
    fake_session_spec_mod.build_default_session_spec = lambda *args, **kwargs: None

    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "xorl.server.api_server": fake_api_server_pkg,
            "xorl.server.api_server.server": fake_api_server_mod,
            "xorl.server.orchestrator": fake_orchestrator_pkg,
            "xorl.server.orchestrator.orchestrator": fake_orchestrator_mod,
            "xorl.server.session_spec": fake_session_spec_mod,
            "xorl.server.utils": fake_utils_pkg,
            "xorl.server.utils.network": fake_network_mod,
        },
    ):
        spec.loader.exec_module(module)

    return module.load_server_arguments


load_server_arguments = _load_server_arguments_fn()


def test_load_server_arguments_threads_signsgd_through_nested_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "optimizer": "signsgd",
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.optimizer == "signsgd"
    assert args.load_weights_mode == "grouped"
    assert args.to_config_dict()["train"]["optimizer"] == "signsgd"
    assert args.to_config_dict()["train"]["load_weights_mode"] == "grouped"


def test_load_server_arguments_threads_distsignsgd_through_nested_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "optimizer": "distsignsgd",
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.optimizer == "distsignsgd"
    assert args.to_config_dict()["train"]["optimizer"] == "distsignsgd"


def test_load_server_arguments_threads_adapter_state_load_mode_into_lora_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "lora": {
                    "enable_lora": True,
                    "adapter_state_load_mode": "rank0_broadcast",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.adapter_state_load_mode == "rank0_broadcast"
    assert args.to_config_dict()["lora"]["adapter_state_load_mode"] == "rank0_broadcast"


def test_load_server_arguments_threads_activation_gpu_limit_into_train_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "enable_activation_offload": True,
                    "activation_gpu_limit": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))

    assert args.activation_gpu_limit == pytest.approx(0.25)
    assert args.to_config_dict()["train"]["activation_gpu_limit"] == pytest.approx(0.25)


def test_load_server_arguments_rejects_broadcast_load_weights_mode(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "load_weights_mode": "broadcast",
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported load_weights_mode"):
        load_server_arguments(str(config_path))


def test_load_server_arguments_rejects_merge_lora_interval_for_server_multi_adapter(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "lora": {
                    "enable_lora": True,
                    "merge_lora_interval": 16,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="merge_lora_interval is not supported"):
        load_server_arguments(str(config_path))


def test_load_server_arguments_rejects_pipeline_parallel_multi_adapter_lora(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "pipeline_parallel_size": 2,
                },
                "lora": {
                    "enable_lora": True,
                    "adapter_state_load_mode": "rank0_broadcast",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pipeline_parallel_size > 1 is not supported"):
        load_server_arguments(str(config_path))


def test_load_server_arguments_threads_muon_gram_newton_schulz_through_nested_config(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "train": {
                    "optimizer": "muon",
                    "optimizer_dtype": "bf16",
                    "muon_lr": 0.03,
                    "muon_ns_algorithm": "gram_newton_schulz",
                    "muon_ns_use_quack_kernels": False,
                    "muon_gram_ns_num_restarts": 2,
                    "muon_gram_ns_restart_iterations": [2],
                    "muon_grouped_gram_ns_fp32_byte_limit": 23,
                    "muon_grad_dtype": "fp32",
                    "muon_update_dtype": "bf16",
                    "muon_force_momentum_path": True,
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    optimizer_kwargs = args.to_config_dict()["train"]["optimizer_kwargs"]
    assert optimizer_kwargs["muon_lr"] == pytest.approx(0.03)
    assert optimizer_kwargs["muon_ns_algorithm"] == "gram_newton_schulz"
    assert optimizer_kwargs["muon_ns_use_quack_kernels"] is False
    assert optimizer_kwargs["muon_gram_ns_num_restarts"] == 2
    assert optimizer_kwargs["muon_gram_ns_restart_iterations"] == [2]
    assert optimizer_kwargs["muon_momentum_dtype"] == torch.bfloat16
    assert optimizer_kwargs["muon_grad_dtype"] == torch.float32
    assert optimizer_kwargs["muon_update_dtype"] == torch.bfloat16
    assert optimizer_kwargs["muon_force_momentum_path"] is True

    train_config = args.to_config_dict()["train"]
    assert args.optimizer == "muon"
    assert args.muon_ns_algorithm == "gram_newton_schulz"
    assert train_config["muon_ns_algorithm"] == "gram_newton_schulz"
    assert train_config["muon_ns_use_quack_kernels"] is False
    assert train_config["muon_gram_ns_num_restarts"] == 2
    assert train_config["muon_gram_ns_restart_iterations"] == [2]
    assert train_config["muon_grouped_gram_ns_fp32_byte_limit"] == 23
    assert args.muon_grad_dtype == "fp32"
    assert args.muon_update_dtype == "bf16"
    assert args.muon_force_momentum_path is True
    assert train_config["muon_grad_dtype"] == "fp32"
    assert train_config["muon_update_dtype"] == "bf16"
    assert train_config["muon_force_momentum_path"] is True


def test_load_server_arguments_preserves_runner_compatibility_fields(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                    "record_routing_weights": False,
                },
                "train": {
                    "enable_full_determinism": True,
                    "optimizer": "muon",
                    "cautious_weight_decay": True,
                    "muon_distributed_mode": "full_gradient",
                    "moe_grad_reduce_mode": "bf16_a2a_fp32_sum",
                    "output_dir": str(tmp_path / "outputs"),
                },
                "lora": {
                    "lora_export_format": "sglang_shared_outer",
                },
            }
        ),
        encoding="utf-8",
    )

    args = load_server_arguments(str(config_path))
    config = args.to_config_dict()

    assert config["model"]["record_routing_weights"] is False
    assert config["train"]["enable_full_determinism"] is True
    assert config["train"]["cautious_weight_decay"] is True
    assert config["train"]["muon_distributed_mode"] == "full_gradient"
    assert config["train"]["moe_grad_reduce_mode"] == "bf16_a2a_fp32_sum"
    assert config["train"]["optimizer_kwargs"]["muon_distributed_mode"] == "full_gradient"
    assert config["lora"]["lora_export_format"] == "sglang_shared_outer"
