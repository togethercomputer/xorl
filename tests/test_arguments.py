import sys

import pytest
import torch
import yaml

from xorl.arguments import Arguments, parse_args


pytestmark = [pytest.mark.cpu]


def test_parse_args_accepts_signsgd_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "optimizer": "signsgd",
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.optimizer == "signsgd"
    assert args.train.optimizer_kwargs == {}


def test_parse_args_wires_muon_kwargs_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "Qwen/Qwen3-8B",
                },
                "data": {
                    "datasets": [{"path": "dummy", "type": "tokenized"}],
                },
                "train": {
                    "init_device": "meta",
                    "output_dir": str(tmp_path / "outputs"),
                    "optimizer": "muon",
                    "optimizer_dtype": "bf16",
                    "muon_ns_algorithm": "gram_newton_schulz",
                    "muon_ns_use_quack_kernels": False,
                    "muon_gram_ns_num_restarts": 2,
                    "muon_gram_ns_restart_iterations": [2],
                    "muon_grad_dtype": "fp32",
                    "muon_update_dtype": "fp32",
                    "muon_force_momentum_path": True,
                    "use_wandb": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])

    args = parse_args(Arguments)

    assert args.train.optimizer == "muon"
    assert args.train.optimizer_kwargs["muon_ns_algorithm"] == "gram_newton_schulz"
    assert args.train.optimizer_kwargs["muon_ns_use_quack_kernels"] is False
    assert args.train.optimizer_kwargs["muon_gram_ns_num_restarts"] == 2
    assert args.train.optimizer_kwargs["muon_gram_ns_restart_iterations"] == [2]
    assert args.train.optimizer_kwargs["muon_momentum_dtype"] is torch.bfloat16
    assert args.train.optimizer_kwargs["muon_grad_dtype"] is torch.float32
    assert args.train.optimizer_kwargs["muon_update_dtype"] is torch.float32
    assert args.train.optimizer_kwargs["muon_force_momentum_path"] is True
