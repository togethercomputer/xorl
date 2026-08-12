import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script",
    ["capture_dsv4_exact_trace", "replay_dsv4_exact_trace"],
)
def test_dsv4_qualification_endpoints_are_loopback_only(script):
    module = _load_script(script)
    assert module._loopback_base_url("http://127.0.0.1:30000/") == ("http://127.0.0.1:30000")
    assert module._loopback_base_url("https://[::1]:8443") == "https://[::1]:8443"

    for unsafe in (
        "http://example.com:30000",
        "http://localhost:30000",
        "http://127.0.0.1:30000/api",
        "http://user:password@127.0.0.1:30000",
        "file:///tmp/socket",
    ):
        with pytest.raises(ValueError):
            module._loopback_base_url(unsafe)


def test_selected_expert_capture_uses_bounded_non_pickle_numpy(tmp_path):
    module = _load_script("qualify_dsv4_marlin_lora")
    capture_path = tmp_path / "selected.npy"
    np.save(
        capture_path,
        np.array([[0, 1, 2, 3, 4, 5], [250, 251, 252, 253, 254, 255]], dtype=np.int32),
    )

    loaded = module._load_selected_experts(
        capture_path,
        tokens=1,
        top_k=6,
        global_num_experts=256,
    )
    assert loaded.dtype == torch.int64
    assert loaded.tolist() == [[0, 1, 2, 3, 4, 5]]

    np.save(capture_path, np.array([[0, 1, 2, 3, 4, 256]], dtype=np.int64))
    with pytest.raises(ValueError, match="global expert range"):
        module._load_selected_experts(
            capture_path,
            tokens=1,
            top_k=6,
            global_num_experts=256,
        )

    np.save(capture_path, np.zeros((1, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="integer NumPy array"):
        module._load_selected_experts(
            capture_path,
            tokens=1,
            top_k=6,
            global_num_experts=256,
        )
