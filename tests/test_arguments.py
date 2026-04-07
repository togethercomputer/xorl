import sys

import pytest
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
