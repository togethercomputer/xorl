from types import SimpleNamespace

import pytest
import torch

from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _RoutingHandler:
    def __init__(self, *, fail_setup: bool = False):
        self.fail_setup = fail_setup
        self.cleanup_calls = 0

    def setup(self, *_args, **_kwargs):
        if self.fail_setup:
            raise RuntimeError("injected R3 setup failure")
        return True

    def cleanup(self):
        self.cleanup_calls += 1


def test_forward_backward_cleans_r3_state_after_failure():
    runner = object.__new__(ModelRunner)
    runner._adapter_manager = None
    runner._routing_handler = _RoutingHandler()
    runner._forward_backward_impl = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected R3 forward failure")
    )

    with pytest.raises(RuntimeError, match="injected R3 forward failure"):
        runner.forward_backward([], routed_experts=[[[[0]]]])

    assert runner._routing_handler.cleanup_calls == 1


def test_forward_cleans_partial_r3_state_when_setup_fails():
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=8, _dsv4_flash_exact_mode=False))
    runner._adapter_manager = None
    runner._routing_handler = _RoutingHandler(fail_setup=True)
    runner.pp_enabled = False
    runner._check_not_sleeping = lambda *_args, **_kwargs: None
    runner._validate_single_tenant = lambda *_args, **_kwargs: None

    with pytest.raises(RuntimeError, match="injected R3 setup failure"):
        runner.forward(
            [{"input_ids": torch.tensor([[1]])}],
            routed_experts=[[[[0]]]],
        )

    assert runner._routing_handler.cleanup_calls == 1
