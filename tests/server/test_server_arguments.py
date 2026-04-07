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
