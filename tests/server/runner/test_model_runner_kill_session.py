"""Tests for LoRA kill-session checkpoint safety in ModelRunner."""

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "model_runner.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_model_runner_kill_session", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ModelRunner = _MODULE.ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeAdapterManager:
    def __init__(self, has_adapter: bool):
        self._has_adapter = has_adapter
        self.removed = []

    def has_adapter(self, model_id: str) -> bool:
        return self._has_adapter

    def remove_adapter(self, model_id: str) -> None:
        self.removed.append(model_id)


def _build_runner(tmp_path: Path, *, has_adapter: bool):
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.lora_config = {"enable_lora": True}
    runner.train_config = {"output_dir": str(tmp_path)}
    runner._adapter_manager = _FakeAdapterManager(has_adapter=has_adapter)
    runner._lora_session_specs = {
        "policy-a": {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
        }
    }
    runner._accumulated_valid_tokens = {"policy-a": 17}
    return runner


def test_kill_session_rejects_missing_checkpoint_for_nonresident_lora_session(tmp_path):
    runner = _build_runner(tmp_path, has_adapter=False)

    with pytest.raises(FileNotFoundError, match="no evicted checkpoint exists"):
        runner.kill_session("policy-a", save_checkpoint=True)

    assert "policy-a" in runner._lora_session_specs
    assert runner._accumulated_valid_tokens["policy-a"] == 17
    assert runner._adapter_manager.removed == []


def test_kill_session_reuses_existing_evicted_checkpoint_for_nonresident_lora_session(tmp_path):
    runner = _build_runner(tmp_path, has_adapter=False)
    evicted_path = tmp_path / "adapters" / "evicted" / "policy-a"
    evicted_path.mkdir(parents=True)
    (evicted_path / "metadata.json").write_text('{"saved": true}', encoding="utf-8")

    result = runner.kill_session("policy-a", save_checkpoint=True)
    promoted_path = tmp_path / "weights" / "policy-a" / "session_policy-a_final"

    assert result == {
        "success": True,
        "message": "LoRA session 'policy-a' killed successfully.",
        "checkpoint_path": str(promoted_path),
    }
    assert (promoted_path / "metadata.json").read_text(encoding="utf-8") == '{"saved": true}'
    assert "policy-a" not in runner._lora_session_specs
    assert "policy-a" not in runner._accumulated_valid_tokens
    assert runner._adapter_manager.removed == []
