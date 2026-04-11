import pytest
import yaml

from xorl.server.launcher import load_server_arguments


pytestmark = [pytest.mark.cpu, pytest.mark.server]


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
    assert args.to_config_dict()["train"]["optimizer"] == "signsgd"


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
                    "muon_ns_algorithm": "gram_newton_schulz",
                    "muon_ns_use_quack_kernels": False,
                    "muon_gram_ns_num_restarts": 2,
                    "muon_gram_ns_restart_iterations": [2],
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
    train_config = args.to_config_dict()["train"]

    assert args.optimizer == "muon"
    assert args.muon_ns_algorithm == "gram_newton_schulz"
    assert train_config["muon_ns_algorithm"] == "gram_newton_schulz"
    assert train_config["muon_ns_use_quack_kernels"] is False
    assert train_config["muon_gram_ns_num_restarts"] == 2
    assert train_config["muon_gram_ns_restart_iterations"] == [2]
    assert args.muon_grad_dtype == "fp32"
    assert args.muon_update_dtype == "bf16"
    assert args.muon_force_momentum_path is True
    assert train_config["muon_grad_dtype"] == "fp32"
    assert train_config["muon_update_dtype"] == "bf16"
    assert train_config["muon_force_momentum_path"] is True
